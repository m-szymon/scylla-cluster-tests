# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright (c) 2026 ScyllaDB

"""Substring (infix) search performance test.

ScyllaDB's `pattern_index` answers `WHERE column LIKE '%keyword%' LIMIT n` from an n-gram index
on the vector-store node instead of scanning the table. This is the benchmark for it: load display
names, build the index, and measure how long the build takes, how large the index is, and what the
containment queries cost.

The flow -- plan, datasets, cumulative shard steps, index build timing, query sets -- lives in
search_perf_test.py and is shared with the full-text benchmark. This module is the substring half
of it: the rune script to run, the vocabulary to report in, and the names to report under. The one
thing it adds to the shared flow is the index size, read from vector-store's own gauges after each
build, since how much memory an n-gram index costs per row is the question the design left open.

Driven by a YAML plan (the `search_test_config` param) naming the datasets, the shards and the
query configurations to test. See docs/substring-search-test.md.

Results go to the Argus tables named below. Run from a developer machine, where JOB_NAME is unset,
TestConfig.init_argus_client resolves to the replay-only client instead, and every row is written as
JSONL into the run's logdir rather than posted -- which is how this test is normally run.
"""

import base64
import os
import re
import threading
import time

import requests
from argus.client.generic_result import ColumnMetadata, ResultType, StaticGenericResultTable, Status

from sdcm.argus_results import submit_results_to_argus
from search_perf_test import (
    DEFAULT_MAX_INDEX_WAIT,
    DEFAULT_SCHEMA_TIMEOUT,
    LatteScriptParams,
    SearchPerformanceTest,
    SearchWorkload,
    _checked_name,
    _query_duration,
    _timeout_minutes,
)
from sdcm.utils.vector_store_index import index_build_columns

SUBSTRING_BASE_DIR = "data_dir/latte/substring_search"


def _checked_direction(value) -> str:
    """The plan's 'order' key: 'desc' (the default) or 'asc'."""
    if value not in ("desc", "asc"):
        raise ValueError(f"'order' must be 'desc' or 'asc', got {value!r}")
    return value


def _checked_match(value) -> str:
    """The plan's 'match' key: where the keyword has to occur, 'contains' (the default), 'prefix'
    or 'suffix'."""
    if value not in ("contains", "prefix", "suffix"):
        raise ValueError(f"'match' must be 'contains', 'prefix' or 'suffix', got {value!r}")
    return value


def _checked_window(value) -> float:
    """Validate a plan's 'window': where in the order a windowed query starts, as a fraction.

    0 is 'no window' (the first page). Anything at or above 1 would bound the query below every
    value in the corpus and measure an empty answer, so it is a plan bug rather than an extreme.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid query window {value!r}: expected a number in [0, 1)")
    if not 0.0 <= float(value) < 1.0:
        raise ValueError(f"Invalid query window {value!r}: expected a fraction in [0, 1)")
    return float(value)


# The column of the index build table counting what was indexed. Named once: it goes both into the
# table definition and into every row submitted, and Argus keys the table's history by it.
SUBSTRING_BUILD_COUNT_COLUMN = "name_count"

# vector-store's gauges for a substring index, and how a metric line of the Prometheus text format
# looks. Labels are matched by name rather than by position, since their order is the exporter's
# business, and the index name is matched case-folded because that is how vector-store knows it.
INDEX_SIZE_METRIC = "pattern_index_size_bytes"
SEGMENT_COUNT_METRIC = "pattern_segment_count"
# One sample per segment, labelled 'segment' with the segment's ordinal. The bounds are the sort
# column's values (vector-store undoes the sort-key bias on export), or absent for an unordered index.
SEGMENT_DOCS_METRIC = "pattern_segment_docs"
SEGMENT_SORT_MIN_METRIC = "pattern_segment_sort_min"
SEGMENT_SORT_MAX_METRIC = "pattern_segment_sort_max"
# Running totals of what the index's searches did, growing since the index was created. The delta
# over a query phase, divided by the searches served in it, says what one query cost the index --
# which is what tells a slow walk from a wide-segment layout when the latency alone cannot.
SEARCHES_METRIC = "pattern_search_total"
# P3b progress: an index with the rewrite on is not settled until every range has been moved.
REWRITE_RANGES_TOTAL_METRIC = "pattern_rewrite_ranges_total"
REWRITE_RANGES_DONE_METRIC = "pattern_rewrite_ranges_done"
REWRITE_DOCS_METRIC = "pattern_rewrite_docs_total"
L0_DOCS_METRIC = "pattern_l0_docs"
# The merges an index's own ticks ask for (Tantivy's merges after a commit are not counted):
# running totals of started and ended, and how many run now. An index is not settled while one
# runs; 'started' far ahead of 'ended' is a merge queued again while it still ran (run a0e51aa9).
MERGES_STARTED_METRIC = "pattern_merges_started"
MERGES_ENDED_METRIC = "pattern_merges_ended"
MERGES_RUNNING_METRIC = "pattern_merges_running"
# Column of the search latency table -> (metric, unit scale applied to the per-query value).
WALK_PER_QUERY_METRICS = {
    "walk_us_per_query": ("pattern_search_walk_seconds_total", 1e6),
    # Part of the walk time: turning the page into primary ids, which is the store reads unless
    # the index keeps the id as a column.
    "page_resolve_us_per_query": ("pattern_search_page_resolve_seconds_total", 1e6),
    # Also part of the walk time: reading every segment's bounds before the first posting, and
    # how many columns had to be opened for it (misses of vector-store's per-segment cache).
    "prepare_us_per_query": ("pattern_search_prepare_seconds_total", 1e6),
    "column_opens_per_query": ("pattern_search_column_opens_total", 1),
    "segments_considered_per_query": ("pattern_search_segments_considered_total", 1),
    "segments_opened_per_query": ("pattern_search_segments_opened_total", 1),
    "postings_per_query": ("pattern_search_postings_scanned_total", 1),
    "heap_entrants_per_query": ("pattern_search_heap_entrants_total", 1),
    "store_reads_per_query": ("pattern_search_store_reads_total", 1),
}
SEARCHES_COLUMN = "index_searches"
WALK_COLUMNS = [ColumnMetadata(name=SEARCHES_COLUMN, unit="", type=ResultType.INTEGER)] + [
    ColumnMetadata(
        name=column,
        unit="us" if column.endswith("_us_per_query") else "",
        type=ResultType.FLOAT,
        higher_is_better=False,
    )
    for column in WALK_PER_QUERY_METRICS
]
# Where a query phase's time went, per node: ScyllaDB's reactors busy (out of all its shards), and
# the loader's and the index node's CPUs busy (out of all their CPUs), averaged over the phase. Tells
# a phase that ScyllaDB's row reads limit from one the client or the index node limits.
REACTOR_BUSY_METRIC = "scylla_reactor_cpu_busy_ms"
DB_CPU_COLUMN = "db_reactor_busy_pct"
# The busiest shard: a mean over shards hides one that is the limit.
DB_MAX_SHARD_COLUMN = "db_reactor_max_shard_pct"
# The DB node's CPUs as the kernel sees them: all busy time, and the parts of it that are the
# hypervisor's (steal) and the network's (softirq), which no reactor counts as its own.
DB_HOST_CPU_COLUMN = "db_cpu_busy_pct"
DB_STEAL_COLUMN = "db_steal_pct"
DB_SOFTIRQ_COLUMN = "db_softirq_pct"
LOADER_CPU_COLUMN = "loader_cpu_busy_pct"
VS_CPU_COLUMN = "vs_cpu_busy_pct"
CPU_COLUMNS = [
    ColumnMetadata(name=column, unit="%", type=ResultType.FLOAT, higher_is_better=False)
    for column in (
        DB_CPU_COLUMN,
        DB_MAX_SHARD_COLUMN,
        DB_HOST_CPU_COLUMN,
        DB_STEAL_COLUMN,
        DB_SOFTIRQ_COLUMN,
        LOADER_CPU_COLUMN,
        VS_CPU_COLUMN,
    )
]
# A query phase with a writer ('write_rate' in the plan) adds rows while it runs; these say what
# that cost the index it queried. 'ingested_rows' is how many changes the index took in through CDC
# during the phase, 'index_lag_*' how long one took from its write in ScyllaDB to the commit that
# made it searchable -- vector-store's 'indexing_lag_seconds' histogram of the fine CDC reader (the
# wide one indexes every change again, minutes later), so the p99 is the bound of its bucket, and a
# p99 past the last bucket (300 s) reads as INDEX_LAG_OVERFLOW_S.
# 'write_op_rate' is what the writer achieved, 'segments' and 'l0_rows' the layout the phase
# ended with (rows in wide segments when the last rewrite was planned), and
# 'rewrite_ranges_done' the rewrite work done during the phase. 'merges_started' and 'merges_ended'
# are the merges the index's ticks asked for and saw end during the phase (MERGES_*_METRIC).
INDEXING_LAG_METRIC = "indexing_lag_seconds"
INDEXING_LAG_READER = "fine"
INDEX_LAG_OVERFLOW_S = 600.0
INGESTED_ROWS_COLUMN = "ingested_rows"
WRITE_OP_RATE_COLUMN = "write_op_rate"
INDEX_LAG_MEAN_COLUMN = "index_lag_mean_s"
INDEX_LAG_P99_COLUMN = "index_lag_p99_s"
SEGMENTS_COLUMN = "segments"
L0_ROWS_COLUMN = "l0_rows"
REWRITE_RANGES_COLUMN = "rewrite_ranges_done"
MERGES_STARTED_COLUMN = "merges_started"
MERGES_ENDED_COLUMN = "merges_ended"
INGEST_COLUMNS = [
    ColumnMetadata(name=INGESTED_ROWS_COLUMN, unit="", type=ResultType.INTEGER),
    ColumnMetadata(name=WRITE_OP_RATE_COLUMN, unit="op/s", type=ResultType.INTEGER),
    ColumnMetadata(name=INDEX_LAG_MEAN_COLUMN, unit="s", type=ResultType.FLOAT, higher_is_better=False),
    ColumnMetadata(name=INDEX_LAG_P99_COLUMN, unit="s", type=ResultType.FLOAT, higher_is_better=False),
    ColumnMetadata(name=SEGMENTS_COLUMN, unit="", type=ResultType.INTEGER, higher_is_better=False),
    ColumnMetadata(name=L0_ROWS_COLUMN, unit="", type=ResultType.INTEGER, higher_is_better=False),
    ColumnMetadata(name=REWRITE_RANGES_COLUMN, unit="", type=ResultType.INTEGER),
    ColumnMetadata(name=MERGES_STARTED_COLUMN, unit="", type=ResultType.INTEGER),
    ColumnMetadata(name=MERGES_ENDED_COLUMN, unit="", type=ResultType.INTEGER),
]
# The writer: 'write_rate' rows a second from one loader, for the phase's duration plus
# WRITER_TAIL_SECS -- it starts with the query phase and has the same start-up to go through, so
# the tail keeps it writing until the queries are done. It ends by itself rather than being killed:
# a killed latte container is reported as a stress failure. Its rows take their names from the
# first WRITER_NAMES_ROWS names of the dataset's first shard.
WRITE_VALUES = ("sequential", "random")
WRITER_TAIL_SECS = 30
WRITER_CONCURRENCY = 64
WRITER_NAMES_FILE = "writer_names.tsv"
WRITER_NAMES_ROWS = 100_000
# An index variant's 'order': what the index orders by, as substring.rn's named shapes -- absent,
# the test case's 'order_by' column; 'table', no order_by (table order); 'unordered'; 'tie', the
# 'tie_column' descending.
VARIANT_ORDERS = ("table", "unordered", "tie")
# How many of the index node's runtime threads were running, on average over a window of the
# phase, sampled from /proc: tells a node whose workers are all busy from one waiting on something.
VS_WORKERS_COLUMN = "vs_workers_running"
VS_WORKER_THREAD = "tokio-rt-worker"
VS_THREADS_COLUMNS = [ColumnMetadata(name=VS_WORKERS_COLUMN, unit="", type=ResultType.FLOAT, higher_is_better=False)]
# Run on the index node as 'sh vs_threads.sh <process name> <delay secs> <window secs> <out>': every
# 10 ms of the window, the state of each of the process's threads, written to <out> once done as
# '<name> <state> <average count>' lines, the names' trailing digits dropped (R running or
# runnable, S sleeping, D in I/O). The file '<out>.stop' ends it early, at the end of the phase:
# over what it sampled until then, or with no samples if the window had not begun -- a short phase
# does not wait out the delay. Plain sh with 'read' builtins and no pidof or pgrep, which a
# container need not have.
VS_THREAD_SAMPLER = r"""pid=
for d in /proc/[0-9]*; do
    read -r comm < "$d/comm" 2>/dev/null && [ "$comm" = "$1" ] && pid=${d#/proc/} && break
done
i=0
while [ "$i" -lt "$2" ] && [ ! -e "$4.stop" ]; do
    sleep 1
    i=$((i + 1))
done
end=$(( $(date +%s) + $3 ))
n=0
tmp=$(mktemp)
while [ -n "$pid" ] && [ "$(date +%s)" -lt "$end" ] && [ ! -e "$4.stop" ]; do
    for f in /proc/"$pid"/task/*/stat; do
        read -r line < "$f" || continue
        rest=${line#*\(}
        name=${rest%\)*}
        after=${line##*\) }
        printf '%s|%s\n' "$name" "${after%% *}"
    done >> "$tmp"
    n=$((n + 1))
    sleep 0.01
done
if [ -z "$pid" ]; then
    echo "no process named $1" > "$4.tmp"
elif [ "$n" -eq 0 ]; then
    echo "no samples: the phase ended before the window began" > "$4.tmp"
else
    sed 's/[0-9-]*|/|/; s/ /_/g' "$tmp" | sort | uniq -c \
        | awk -v n="$n" '{split($2, a, "|"); printf "%s %s %.2f\n", a[1], a[2], $1 / n}' > "$4.tmp"
fi
rm -f "$tmp"
mv "$4.tmp" "$4"
"""
# A query entry with 'profile: true' samples the DB node's CPUs during the phase: a system-wide
# 'perf record' (as scylla-perf-collector does; 'perf report --comm scylla' narrows it), from
# PROFILE_DELAY_SECS into the phase (past the client's start-up: latte's queries began 33-66 s
# after the phase did in run 0216688f, whose 10 s delay profiled idle nodes) for PROFILE_SECS, PROFILE_HZ
# samples a second per CPU of the cpu-clock timer -- an EC2 VM exposes no hardware cycle counter --
# with DWARF call stacks, since a release build keeps no frame pointers. The recording is gzipped
# and collected to '<logdir>/perf/'; the node's binary is stripped, so it is read against the
# unstripped build with the same build id.
PROFILE_DELAY_SECS = 70
PROFILE_SECS = 20
PROFILE_HZ = 99
PROFILE_STACK_BYTES = 8192
# A fixed delay was not enough: run a0e51aa9's first profile, 70 s into its phase, recorded an idle
# DB node (96% in the idle task) while the CPU window started a little later read it 83% busy --
# latte's start-up varies from phase to phase. So PROFILE_SCRIPT waits PROFILE_MIN_DELAY_SECS, then
# until the node's CPUs are PROFILE_BUSY_PCT busy over a second (for at most PROFILE_MAX_WAIT_SECS
# in all), then PROFILE_SETTLE_SECS, and only then records; its log says how long it waited.
PROFILE_MIN_DELAY_SECS = 20
PROFILE_MAX_WAIT_SECS = 120
PROFILE_BUSY_PCT = 25
PROFILE_SETTLE_SECS = 10
# Run as 'sh profile.sh <out> <hz> <stack bytes> <secs>' with the waits above baked in.
PROFILE_SCRIPT = rf"""busy() {{
    a=$(awk '/^cpu /{{print $2+$3+$4+$7+$8, $2+$3+$4+$5+$6+$7+$8+$9}}' /proc/stat)
    sleep 1
    b=$(awk '/^cpu /{{print $2+$3+$4+$7+$8, $2+$3+$4+$5+$6+$7+$8+$9}}' /proc/stat)
    echo "$a $b" | awk '{{d = $4 - $2; print (d > 0 ? int(100 * ($3 - $1) / d) : 0)}}'
}}
sleep {PROFILE_MIN_DELAY_SECS}
waited={PROFILE_MIN_DELAY_SECS}
while [ "$waited" -lt {PROFILE_MAX_WAIT_SECS} ] && [ "$(busy)" -lt {PROFILE_BUSY_PCT} ]; do
    waited=$((waited + 1))
done
echo "busy after $waited s; recording from $((waited + {PROFILE_SETTLE_SECS})) s"
sleep {PROFILE_SETTLE_SECS}
perf record -a -e cpu-clock -F "$2" --call-graph "dwarf,$3" -o "$1" -- sleep "$4"
"""
# Run on each node as 'sh cpu_window.sh <delay secs> <window secs> <out>': the node's CPU reading
# (the clock, /proc/stat's 'cpu' line, and ScyllaDB's reactor busy times where there is a ScyllaDB)
# into '<out>.a' PROFILE_DELAY_SECS into the phase and into '<out>.b' at the end of the window or
# when '<out>.stop' appears, whichever is first; '<out>.done' once it is over. So the CPU columns
# cover the same window as the profiles and the thread states -- the queries' -- and not the
# client's start-up, which diluted every CPU figure of run 0216688f by ~1.75x. A phase that ends
# before the window has no '.a', and its columns fall back to the whole phase.
CPU_WINDOW_SCRIPT = r"""snap() {
    { date +%s.%N; head -1 /proc/stat; curl -s --max-time 5 http://localhost:9180/metrics 2>/dev/null \
        | grep '^scylla_reactor_cpu_busy_ms'; } > "$1.tmp"
    mv "$1.tmp" "$1"
}
i=0
while [ "$i" -lt "$1" ] && [ ! -e "$3.stop" ]; do
    sleep 1
    i=$((i + 1))
done
if [ ! -e "$3.stop" ]; then
    snap "$3.a"
    i=0
    while [ "$i" -lt "$2" ] && [ ! -e "$3.stop" ]; do
        sleep 1
        i=$((i + 1))
    done
    snap "$3.b"
fi
touch "$3.done"
"""
# How many segments the layout log lists in full; the summary columns cover the rest.
SEGMENT_LAYOUT_LOG_ROWS = 64
_METRIC_LINE_RE_TEMPLATE = r"^{metric}\{{(?P<labels>[^}}]*)\}}\s+(?P<value>[0-9.eE+-]+)\s*$"
_LABEL_RE = re.compile(r'(?P<name>\w+)="(?P<value>[^"]*)"')

# The '-P' name that makes substring.rn create the index in its 'schema' phase. Not part of
# LatteScriptParams, which describes only what the shared flow drives.
WITH_INDEX_PARAM = "with_index"
# The '-P' names carrying one extra WITH OPTIONS entry, name and value apart: latte splices a
# value into the script as a literal, so a quoted fragment would not parse.
EXTRA_OPTION_PARAMS = (
    ("extra_option_name", "extra_option_value"),
    ("extra_option2_name", "extra_option2_value"),
    ("extra_option3_name", "extra_option3_value"),
)

# A dataset may ask for several indexes over the one load, differing only in their options:
#
#   index_variants:
#     - label: store_id
#     - label: fast_id
#       options: {segment_rows: '0'}
#
# All of them are created before the load and ingest the same CDC stream, so they get equivalent
# layouts, which two separate runs would not (merge timing differs per run). ScyllaDB picks one of
# them for every query and the choice is not by name, so each round of the query sets first finds
# out which index is serving (a short probe, then a look at which one's search total moved), labels
# its rows with that variant, and drops the index afterwards so the next round reaches another.
# With 'index_during_load' the variants ingest the same CDC stream; without it they are created
# after the load, one after another, and each is built by vector-store's full scan -- the backfill
# case, whose wide segments are what the rewrite option exists for.
INDEX_VARIANTS_KEY = "index_variants"
SERVED_INDEX_PROBE_DURATION = "5s"

# Polling for 'index_during_load'. The count is asked for on this interval while the index catches
# up with the load; the settle loop then watches the segment count until it stops moving, because
# Tantivy keeps merging after the writes stop and query latency depends on how many segments a
# search has to visit.
INDEX_COUNT_POLL_INTERVAL_SECS = 15.0
SETTLE_POLL_INTERVAL_SECS = 10.0
SETTLE_STABLE_POLLS = 3
# Before each variant round, the indexes still standing must have stopped taking in changes: when
# one index is dropped, vector-store may restart the CDC readers of the others (its session is
# reset), and they read the last ten minutes of the log again -- in a local run, 212k changes,
# minutes old, in the first phase of the next round. Quiet is CDC_QUIET_POLLS polls in a row on
# SETTLE_POLL_INTERVAL_SECS with fewer than CDC_QUIET_CHANGES new changes each.
CDC_QUIET_POLLS = 2
CDC_QUIET_CHANGES = 100
DEFAULT_SETTLE_TIMEOUT_SECS = 600
# A missed scrape or status poll is one lost sample (the runner reaches the index node over the
# internet), but an index node that answers nothing for this long is gone: run fd5928db's ran out
# of memory at 09:24:55, and the run polled it for 47 more minutes before a query probe failed.
VS_UNRESPONSIVE_ABORT_SECS = 180


# The index node's memory once a minute, into its journal, which reaches the runner's copy of the
# node's log (hosts/<node>/messages.log) as it is written: a pattern index lives in RAM, and when
# the node runs out, its log is all that is left of it (run fd5928db's stopped mid-line).
VS_MEMORY_LOG_SCRIPT = r"""while true; do
  m=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "total_mb=%d available_mb=%d", t/1024, a/1024}' /proc/meminfo)
  r=$(ps -C vector-store -o rss= | awk '{s+=$1} END{printf "%d", s/1024}')
  logger -t vs-memory "$m vector_store_rss_mb=$r"
  sleep 60
done
"""


class VectorStoreUnresponsive(RuntimeError):
    """The index node has answered no request for VS_UNRESPONSIVE_ABORT_SECS."""


# The diagnostics -- CPU readings and windows, the thread sampler, profiles, the memory log -- reach
# the nodes over SSH, and an SSH call has no bound of its own while it connects: run a0e51aa9's index
# node, out of memory, still accepted connections, and the test sat in one for six minutes until it
# was killed. So each such call is given its own timeout plus DIAGNOSTIC_GRACE_SECS (or
# DIAGNOSTIC_DEFAULT_SECS when it has none) on a thread of its own, and abandoned past that; and the
# index node's are not tried at all while it answers no HTTP request (VectorStoreUnresponsive's count).
DIAGNOSTIC_GRACE_SECS = 30
DIAGNOSTIC_DEFAULT_SECS = 120


class DiagnosticSkipped(RuntimeError):
    """A diagnostic SSH call given up on: past its bound, or to a node known not to answer."""


class _BoundedRemoter:
    """A node's remoter whose calls give up after their bound; see DIAGNOSTIC_GRACE_SECS."""

    def __init__(self, remoter, node_name: str):
        self._remoter = remoter
        self._node_name = node_name

    def __getattr__(self, name):
        call = getattr(self._remoter, name)

        def bounded(*args, **kwargs):
            bound = (kwargs.get("timeout") or DIAGNOSTIC_DEFAULT_SECS) + DIAGNOSTIC_GRACE_SECS
            outcome = {}

            def target():
                try:
                    outcome["result"] = call(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - handed to the caller below
                    outcome["error"] = exc

            thread = threading.Thread(target=target, name=f"diagnostic-{self._node_name}", daemon=True)
            thread.start()
            thread.join(bound)
            if thread.is_alive():
                raise DiagnosticSkipped(f"{name} on {self._node_name} still running after {bound:.0f} s; left behind")
            if "error" in outcome:
                raise outcome["error"]
            return outcome["result"]

        return bounded


class SubstringIndexBuildResult(StaticGenericResultTable):
    class Meta:
        name = "Substring Index Build Time"
        description = "Substring search index build time and throughput"
        Columns = index_build_columns(SUBSTRING_BUILD_COUNT_COLUMN, "names")


class SubstringIndexSizeResult(StaticGenericResultTable):
    """How much the index costs, per build.

    'bytes_per_name' is the number the sizing question is really about: an n-gram index stores every
    substring of min_gram..max_gram characters of every value, so its cost per row depends on the
    length of the names and on max_gram, not on the row count. Reported next to the absolute size so
    that two corpus sizes in one run can be compared directly.
    """

    class Meta:
        name = "Substring Index Size"
        description = "Substring search index size on the vector-store node, per index build"
        Columns = [
            ColumnMetadata(name="index_size_bytes", unit="bytes", type=ResultType.INTEGER, higher_is_better=False),
            ColumnMetadata(name="bytes_per_name", unit="bytes", type=ResultType.FLOAT, higher_is_better=False),
            ColumnMetadata(name=SUBSTRING_BUILD_COUNT_COLUMN, unit="names", type=ResultType.INTEGER),
            ColumnMetadata(name="segment_count", unit="", type=ResultType.INTEGER),
            # How wide the segments are, as a share of the whole sort-column range. An ordered search
            # skips a segment only when its span cannot hold a page entry, so a mean near 100% means
            # nothing is skipped and every ordered query walks every match; the point of these two
            # is to say so before the query rows are read.
            ColumnMetadata(name="segment_span_mean_pct", unit="%", type=ResultType.FLOAT, higher_is_better=False),
            ColumnMetadata(name="segment_span_max_pct", unit="%", type=ResultType.FLOAT, higher_is_better=False),
            # How long after the load ended the index held every row, when indexing during the
            # load. The ingestion cost of an index option -- a merge policy that rewrites its tail
            # on every commit, say -- shows up here and nowhere else.
            ColumnMetadata(name="catch_up_secs", unit="s", type=ResultType.FLOAT, higher_is_better=False),
            # The rewrite of wide segments (P3b): rows it found in wide segments when it planned,
            # and the ranges it moved them in. Zero for an index without the option, or one that
            # never needed it.
            ColumnMetadata(name="l0_rows", unit="rows", type=ResultType.INTEGER, higher_is_better=False),
            ColumnMetadata(name="rewrite_ranges", unit="", type=ResultType.INTEGER),
        ]


SUBSTRING_WORKLOAD = SearchWorkload(
    name="substring_search",
    base_dir=SUBSTRING_BASE_DIR,
    script=f"{SUBSTRING_BASE_DIR}/substring.rn",
    hdr_tag="fn--search",
    item_noun="names",
    index_prefix="sub_idx",
    default_keyspace="substring_bench",
    remote_root="/tmp/substring",
    latency_legend="Substring search (LIKE '%keyword%') query latency.",
    build_result_table=SubstringIndexBuildResult,
    build_count_column=SUBSTRING_BUILD_COUNT_COLUMN,
    # The names substring.rn uses. It mirrors the vocabulary of fts.rn where the flow is shared and
    # renames the rest to what this workload indexes -- names rather than documents.
    params=LatteScriptParams(
        dataset_dir="substring_data_dir",
        records_file="names_file",
        record_count="name_count",
        queries_file="queries_file",
        qrels_file="qrels_file",
        search_limit="search_limit",
        compute_accuracy="compute_accuracy",
        index_name="index_name",
        max_index_wait="max_index_wait_secs",
        min_probes="min_successful_probes",
        schema_cleanup="schema_cleanup",
        drop_index="drop_index",
    ),
    step_records_file_key="names_file",
    default_records_file="names.tsv",
    default_shard_suffix="names_{:03d}.tsv",
)


def parse_index_gauge(
    metrics_text: str, metric: str, keyspace: str, index_name: str, with_labels: dict | None = None
) -> float | None:
    """Read one vector-store gauge for one index out of the Prometheus text exposition.

    Returns None when the index has no sample for the metric, which is what an older vector-store
    without these gauges, or an index vector-store has not accounted for yet, looks like.
    """
    wanted_keyspace = keyspace.lower()
    wanted_index = index_name.lower()
    line_re = re.compile(_METRIC_LINE_RE_TEMPLATE.format(metric=re.escape(metric)), re.MULTILINE)
    for match in line_re.finditer(metrics_text):
        labels = {m.group("name"): m.group("value") for m in _LABEL_RE.finditer(match.group("labels"))}
        if (
            labels.get("keyspace", "").lower() == wanted_keyspace
            and labels.get("index_name", "").lower() == wanted_index
            and _has_labels(labels, with_labels)
        ):
            return float(match.group("value"))
    return None


def _has_labels(labels: dict, wanted: dict | None) -> bool:
    """Whether a sample carries each wanted label value -- or no such label at all, as a series of
    an older vector-store that did not split it."""
    return all(labels.get(name, value) == value for name, value in (wanted or {}).items())


def checked_index_variants(dataset: dict) -> list[dict]:
    """The dataset's 'index_variants' as [{label, options}], validated, or [] when it has none."""
    variants = dataset.get(INDEX_VARIANTS_KEY) or []
    if not variants:
        return []
    checked = []
    for variant in variants:
        if not isinstance(variant, dict) or "label" not in variant:
            raise ValueError(f"Dataset {dataset.get('name')!r}: every index variant needs a 'label'")
        label = _checked_name(str(variant["label"]), "index variant label")
        options = variant.get("options") or {}
        if not isinstance(options, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in options.items()
        ):
            raise ValueError(f"Index variant {label!r}: 'options' must map option names to string values")
        index_option_params(options)  # rejects what the script cannot carry
        order = variant.get("order", "")
        if order and order not in VARIANT_ORDERS:
            raise ValueError(f"Index variant {label!r}: 'order' must be one of {VARIANT_ORDERS}, not {order!r}")
        if order and "order_by" in options:
            raise ValueError(f"Index variant {label!r}: 'order' and an 'order_by' option say the same thing twice")
        if any(label == seen["label"] for seen in checked):
            raise ValueError(f"Index variant label {label!r} is used twice")
        checked.append({"label": label, "options": options, "order": order})
    return checked


def index_option_params(options: dict) -> str:
    """The '-P' flags carrying a variant's extra index options to substring.rn, or "" for none.

    As many options per variant as the script has slots for; the values and names are plain
    words, since they travel through a shell command line and into a rune literal.
    """
    if not options:
        return ""
    if len(options) > len(EXTRA_OPTION_PARAMS):
        raise ValueError(
            f"An index variant can set up to {len(EXTRA_OPTION_PARAMS)} extra options, not {sorted(options)}"
        )
    flags = ""
    for (name_param, value_param), (name, value) in zip(EXTRA_OPTION_PARAMS, sorted(options.items())):
        for text in (name, value):
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", text):
                raise ValueError(
                    f"Index option {name!r}={value!r}: only letters, digits, '_', '.' and '-' can be passed"
                )
        flags += f'-P {name_param}=\\"{name}\\" -P {value_param}=\\"{value}\\" '
    return flags


def parse_index_gauge_series(
    metrics_text: str, metric: str, keyspace: str, index_name: str, label: str, with_labels: dict | None = None
) -> dict[str, float]:
    """Every sample of one gauge for one index, keyed by the value of *label*.

    For the per-segment gauges, where one metric has a sample per segment ordinal. Empty when the
    index has no samples, for the same reasons 'parse_index_gauge' returns None.
    """
    wanted_keyspace = keyspace.lower()
    wanted_index = index_name.lower()
    line_re = re.compile(_METRIC_LINE_RE_TEMPLATE.format(metric=re.escape(metric)), re.MULTILINE)
    series = {}
    for match in line_re.finditer(metrics_text):
        labels = {m.group("name"): m.group("value") for m in _LABEL_RE.finditer(match.group("labels"))}
        if (
            labels.get("keyspace", "").lower() == wanted_keyspace
            and labels.get("index_name", "").lower() == wanted_index
            and label in labels
            and _has_labels(labels, with_labels)
        ):
            series[labels[label]] = float(match.group("value"))
    return series


def parse_segment_layout(
    metrics_text: str, keyspace: str, index_name: str
) -> list[tuple[int, float, float | None, float | None]]:
    """The index's segments as (ordinal, live rows, sort min, sort max), by ordinal.

    The bounds are None for an unordered index, which exports none.
    """
    docs = parse_index_gauge_series(metrics_text, SEGMENT_DOCS_METRIC, keyspace, index_name, "segment")
    lows = parse_index_gauge_series(metrics_text, SEGMENT_SORT_MIN_METRIC, keyspace, index_name, "segment")
    highs = parse_index_gauge_series(metrics_text, SEGMENT_SORT_MAX_METRIC, keyspace, index_name, "segment")
    return [(int(ordinal), docs[ordinal], lows.get(ordinal), highs.get(ordinal)) for ordinal in sorted(docs, key=int)]


def segment_spans_pct(segments: list[tuple[int, float, float | None, float | None]]) -> list[float]:
    """Each segment's span of the sort column as a percentage of the range every segment covers together.

    A one-value range, or an unordered index, makes every span 0: there is nothing to prune by and
    nothing the pruning could gain, which is the same thing from the query's point of view.
    """
    bounded = [(low, high) for _, _, low, high in segments if low is not None and high is not None]
    if not bounded:
        return [0.0 for _ in segments]
    overall_low = min(low for low, _ in bounded)
    overall_high = max(high for _, high in bounded)
    overall = overall_high - overall_low
    if overall <= 0:
        return [0.0 for _ in segments]
    return [
        (round(100.0 * (high - low) / overall, 2) if low is not None and high is not None else 0.0)
        for _, _, low, high in segments
    ]


def walk_per_query(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """The per-query cost of the searches served between two scrapes of the walk totals.

    *before* and *after* map metric names (SEARCHES_METRIC and the ones WALK_PER_QUERY_METRICS
    names) to the scraped values. Returns the WALK_COLUMNS cells, or {} when no search was served
    in between -- a phase whose queries never reached the index has no per-query cost to report,
    and dividing by zero would say it was free.
    """
    searches = after.get(SEARCHES_METRIC, 0.0) - before.get(SEARCHES_METRIC, 0.0)
    if searches <= 0:
        return {}
    cells = {SEARCHES_COLUMN: int(searches)}
    for column, (metric, scale) in WALK_PER_QUERY_METRICS.items():
        if metric in after and metric in before:
            cells[column] = round(scale * (after[metric] - before[metric]) / searches, 3)
    return cells


def parse_proc_stat(text: str) -> dict[str, float] | None:
    """Jiffies of all CPUs from the 'cpu' line of /proc/stat: total, busy, steal and softirq, or None.

    Busy is everything but idle and iowait; guest time is already inside user and nice.
    """
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "cpu":
            values = [float(v) for v in fields[1:9]] + [0.0] * (8 - len(fields[1:9]))
            user, nice, system, idle, iowait, irq, softirq, steal = values
            total = sum(values)
            return {"total": total, "busy": total - idle - iowait, "steal": steal, "softirq": softirq}
    return None


def parse_reactor_busy(metrics_text: str) -> dict[str, float] | None:
    """ScyllaDB's reactor busy time in ms by shard, or None."""
    shards = {}
    for line in metrics_text.splitlines():
        if line.startswith((REACTOR_BUSY_METRIC + "{", REACTOR_BUSY_METRIC + " ")):
            shard = re.search(r'shard="(\d+)"', line)
            shards[shard.group(1) if shard else str(len(shards))] = float(line.rsplit(None, 1)[1])
    return shards or None


def _stat_share(before: dict | None, after: dict | None, field: str) -> float | None:
    if not before or not after or after["total"] <= before["total"]:
        return None
    return round(100.0 * (after[field] - before[field]) / (after["total"] - before["total"]), 1)


def cpu_busy_pct(before: dict, after: dict) -> dict[str, float]:
    """The CPU_COLUMNS cells for a phase, from two '_scrape_cpu' readings; a missing side, no cell."""
    cells = {}
    elapsed_ms = 1000.0 * (after.get("time", 0.0) - before.get("time", 0.0))
    db_before, db_after = before.get("db"), after.get("db")
    if db_before and db_after and elapsed_ms > 0 and db_after.keys() == db_before.keys():
        shares = [100.0 * (db_after[shard] - db_before[shard]) / elapsed_ms for shard in db_after]
        cells[DB_CPU_COLUMN] = round(sum(shares) / len(shares), 1)
        cells[DB_MAX_SHARD_COLUMN] = round(max(shares), 1)
    for column, key, field in (
        (DB_HOST_CPU_COLUMN, "db_host", "busy"),
        (DB_STEAL_COLUMN, "db_host", "steal"),
        (DB_SOFTIRQ_COLUMN, "db_host", "softirq"),
        (LOADER_CPU_COLUMN, "loader", "busy"),
        (VS_CPU_COLUMN, "vs", "busy"),
    ):
        share = _stat_share(before.get(key), after.get(key), field)
        if share is not None:
            cells[column] = share
    return cells


def parse_thread_states(text: str) -> dict[str, float]:
    """'<thread name> <state> <average threads>' lines of the index node's sampler, as a mapping."""
    states = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 3:
            states[f"{fields[0]} {fields[1]}"] = float(fields[2])
    return states


def parse_cpu_snapshot(text: str) -> dict:
    """One CPU_WINDOW_SCRIPT reading as {time, stat, reactor}, each part only when it is there."""
    lines = text.splitlines()
    snapshot = {}
    if lines:
        try:
            snapshot["time"] = float(lines[0])
        except ValueError:
            return {}
    if (stat := parse_proc_stat(text)) is not None:
        snapshot["stat"] = stat
    if (reactor := parse_reactor_busy(text)) is not None:
        snapshot["reactor"] = reactor
    return snapshot


def parse_indexing_lag(metrics_text: str, keyspace: str, index_name: str) -> dict:
    """The index's 'indexing_lag_seconds' histogram as {buckets: {le: count}, sum, count}, or {}.

    The fine CDC reader's: the wide one indexes every change again, minutes later, which would
    count each change twice and put its re-read's delay into the lag.
    """
    fine = {"reader": INDEXING_LAG_READER}
    buckets = parse_index_gauge_series(metrics_text, f"{INDEXING_LAG_METRIC}_bucket", keyspace, index_name, "le", fine)
    total = parse_index_gauge(metrics_text, f"{INDEXING_LAG_METRIC}_sum", keyspace, index_name, fine)
    count = parse_index_gauge(metrics_text, f"{INDEXING_LAG_METRIC}_count", keyspace, index_name, fine)
    if count is None or total is None:
        return {}
    return {"buckets": buckets, "sum": total, "count": count}


def cdc_changes_total(metrics_text: str, keyspace: str, index_name: str) -> float | None:
    """How many changes the index has taken in through CDC, by every reader, or None."""
    counts = parse_index_gauge_series(metrics_text, f"{INDEXING_LAG_METRIC}_count", keyspace, index_name, "reader")
    if counts:
        return sum(counts.values())
    return parse_index_gauge(metrics_text, f"{INDEXING_LAG_METRIC}_count", keyspace, index_name)


def indexing_lag_cells(before: dict, after: dict) -> dict:
    """The ingest cells for the changes indexed between two 'parse_indexing_lag' readings.

    An index that took in nothing reports 0 rows and no lag: there was nothing to be late with.
    A side missing (an index created in between, a failed scrape) leaves every cell empty.
    """
    if not before or not after:
        return {}
    count = after["count"] - before["count"]
    if count <= 0:
        return {INGESTED_ROWS_COLUMN: 0}
    cells = {
        INGESTED_ROWS_COLUMN: int(count),
        INDEX_LAG_MEAN_COLUMN: round((after["sum"] - before["sum"]) / count, 3),
    }
    finite = sorted((float(le), value) for le, value in after["buckets"].items() if le != "+Inf")
    p99 = INDEX_LAG_OVERFLOW_S
    for le, value in finite:
        if value - before["buckets"].get(_bucket_label(le, before["buckets"]), 0.0) >= 0.99 * count:
            p99 = le
            break
    cells[INDEX_LAG_P99_COLUMN] = p99
    return cells


def _bucket_label(le: float, buckets: dict) -> str:
    """The key of *buckets* whose bound is *le*: Prometheus writes '1' or '1.0' as it pleases."""
    for label in buckets:
        if label != "+Inf" and float(label) == le:
            return label
    return ""


def duration_seconds(duration: str) -> int:
    """A plan duration ('60s', '5m', '1h') in seconds."""
    match = re.fullmatch(r"(\d+)([hms])", duration)
    if not match:
        raise ValueError(f"Not a duration: {duration!r}")
    return int(match.group(1)) * {"h": 3600, "m": 60, "s": 1}[match.group(2)]


def checked_writer(query: dict) -> tuple[int, str] | None:
    """A query entry's writer as (rows a second, 'sequential' or 'random'), or None for no writer."""
    if "write_rate" not in query:
        if "write_values" in query:
            raise ValueError(f"Query set {query.get('set')!r}: 'write_values' without 'write_rate'")
        return None
    rate = query["write_rate"]
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise ValueError(f"Query set {query.get('set')!r}: 'write_rate' must be a positive number of rows a second")
    values = query.get("write_values", "sequential")
    if values not in WRITE_VALUES:
        raise ValueError(
            f"Query set {query.get('set')!r}: 'write_values' must be one of {WRITE_VALUES}, not {values!r}"
        )
    return rate, values


def queries_for_variant(queries: list, label: str) -> list:
    """The query entries a variant's round runs: the ones naming it in 'variants', or naming none."""
    return [query for query in queries if "variants" not in query or label in query["variants"]]


def variant_params(variant: dict) -> str:
    """The '-P' flags that create an index variant: its options, and its order when it names one."""
    params = index_option_params(variant["options"])
    if variant.get("order"):
        params += rf"-P index_order=\"{variant['order']}\" "
    return params


class SubstringSearchTest(SearchPerformanceTest):
    """Substring search (LIKE '%keyword%') performance test.

    Runs multi-dataset, multi-step substring benchmarks from a YAML plan, in the repo or on S3 (see
    'resolve_test_config_path'): per-shard loading, index building, query execution and reporting
    all come from 'SearchPerformanceTest'.
    """

    WORKLOAD = SUBSTRING_WORKLOAD

    def query_shape_label(self, query: dict) -> str:
        """Name the shape an ordered plan asks a query set in.

        A set run plain, ordered and windowed produces three rows whose latencies are not comparable
        configurations of one measurement but answers to three different questions, so the shape
        goes in the row label rather than in a column.
        """
        ordered = bool(query.get("ordered", False))
        window = _checked_window(query.get("window", 0.0))
        direction = _checked_direction(query.get("order", "desc"))
        match = _checked_match(query.get("match", "contains"))
        if window > 0.0:
            shape = f" ordered from {window:g}" if ordered else f" window from {window:g}"
        else:
            shape = " ordered" if ordered else ""
        if query.get("order_column"):
            shape += f" by {query['order_column']}"
        if direction == "asc" and (ordered or window > 0.0):
            shape += " asc"
        if match != "contains":
            shape = f" {match}{shape}"
        if writer := checked_writer(query):
            shape += f" +{writer[0]}/s {writer[1]} writes"
        # Which of the dataset's index variants answered; "" outside a variant round. Looked up
        # rather than read, since plan validation calls this on a stand-in for the test.
        variant = getattr(self, "_variant_label", "")
        return f"{shape} [{variant}]" if variant else shape

    def _probe_shape(self, query: dict) -> str:
        """The ORDER BY of `query` and nothing else: what decides which index serves it."""
        if not (query.get("ordered") or _checked_window(query.get("window", 0.0)) > 0.0):
            return ""
        direction = _checked_direction(query.get("order", "desc"))
        params = "-P search_ordered=true "
        if query.get("order_column"):
            params += r"-P search_order_column=\"tie\" "
        if direction != "desc":
            params += rf"-P search_direction=\"{direction}\" "
        return params

    def extra_search_params(self, query: dict, record_count: int) -> str:
        """Translate the plan's 'ordered' and 'window' keys into substring.rn's '-P' flags.

        'window: 0.5' asks for the page that begins halfway down the order, which is how a later
        page is measured: latte drives no CQL paging, but a page resuming at a cursor is exactly a
        query bounded by it, and keeping that flat is what the cursor is for.
        """
        ordered = bool(query.get("ordered", False))
        window = _checked_window(query.get("window", 0.0))
        direction = _checked_direction(query.get("order", "desc"))
        match = _checked_match(query.get("match", "contains"))
        checked_writer(query)
        params = rf"-P search_match=\"{match}\" " if match != "contains" else ""
        if query.get("check_order"):
            # A plain query's order is defined only by an index in table order, so the plan says
            # where to check it; an ordered one is always checked.
            if not query.get("qrels"):
                raise ValueError(f"Query set {query.get('set')!r}: 'check_order' needs 'qrels'")
            params += "-P check_order=true "
        order_column = query.get("order_column")
        if order_column is not None:
            if not self._tie_column or order_column != self._tie_column:
                raise ValueError(
                    f"Query set {query.get('set')!r} orders by {order_column!r}, but the table's tie column "
                    f"('tie_column' in 'latte_schema_parameters') is {self._tie_column!r}"
                )
            if not ordered or window > 0.0:
                raise ValueError(
                    f"Query set {query.get('set')!r}: 'order_column' goes with 'ordered: true' and no window"
                )
            params += r"-P search_order_column=\"tie\" "
        if not ordered and window <= 0.0:
            return params
        if not self._order_by:
            # Nothing downstream would fail: the index has no sort column, so substring.rn refuses
            # the shape -- but it refuses inside the loader, after the corpus is loaded and indexed.
            raise ValueError(
                f"Query set {query.get('set')!r} asks for an ordered or windowed search, but "
                f"'latte_schema_parameters' sets no 'order_by', so the index has nothing to order by"
            )
        # A windowed search is ordered too: a range on the ordering column comes with its ORDER BY,
        # since an index with order_by serves only queries ordered by its columns.
        params += "-P search_ordered=true "
        if direction != "desc":
            params += rf"-P search_direction=\"{direction}\" "
        if window > 0.0:
            params += f"-P search_window_from={window} -P sort_value_count={record_count} "
        return params

    @property
    def _order_by(self) -> str:
        """The column the index orders by, from the test case, or "" when it has none."""
        return (self.params.get("latte_schema_parameters") or {}).get("order_by") or ""

    @property
    def _tie_column(self) -> str:
        """The table's low-cardinality column, from the test case, or "" when it has none."""
        return (self.params.get("latte_schema_parameters") or {}).get("tie_column") or ""

    def validate_dataset(self, dataset: dict) -> None:
        """Check what the shared validation cannot: each query's 'variants' against the dataset's."""
        labels = [variant["label"] for variant in checked_index_variants(dataset)]
        for step in dataset.get("steps", []):
            for query in step.get("queries", []):
                named = query.get("variants")
                if named is None:
                    continue
                if not labels:
                    raise ValueError(f"Query set {query.get('set')!r} names variants, but the dataset has none")
                if not isinstance(named, list) or not named or not set(named) <= set(labels):
                    raise ValueError(
                        f"Query set {query.get('set')!r}: 'variants' must list some of {labels}, not {named!r}"
                    )

    # Set per dataset from the plan's 'index_during_load'; see _run_dataset.
    _index_during_load = False
    _dataset_name = ""
    _step_idx = -1
    _ingesting_index_name = ""
    _max_index_wait = DEFAULT_MAX_INDEX_WAIT
    # Set per dataset from the plan's 'max_settle_secs': how long an index may go on merging, or
    # rewriting, after it has caught up before the queries start anyway.
    _max_settle = DEFAULT_SETTLE_TIMEOUT_SECS
    # The dataset's 'index_variants', the ones of the current step still standing (index name ->
    # label), and the label of the one serving the round in progress.
    _variants: list = []
    _live_variants: dict = {}
    _variant_label = ""

    def _run_dataset(self, dataset):
        """Note whether this dataset indexes while it loads, then run it as usual.

        'index_during_load: true' in the plan creates the index before the rows are written, so the
        index node ingests through CDC while latte is still loading, and the separate full-scan
        build is skipped altogether. It is for runs that care about query throughput and latency
        rather than about how long a build takes -- with it, no build time is measured at all.
        """
        self._index_during_load = bool(dataset.get("index_during_load"))
        self._dataset_name = _checked_name(dataset["name"], "dataset name")
        self._step_idx = -1
        self._ingesting_index_name = ""
        self._max_index_wait = dataset.get("max_index_wait_secs", DEFAULT_MAX_INDEX_WAIT)
        self._max_settle = dataset.get("max_settle_secs", DEFAULT_SETTLE_TIMEOUT_SECS)
        self._variants = checked_index_variants(dataset)
        self._live_variants = {}
        self._variant_label = ""
        self._write_next = 0  # the next sequential sort value a writer gives; the table starts empty
        if self._index_during_load:
            self.log.info("Dataset '%s': indexing during load; no build time will be measured", self._dataset_name)
        return super()._run_dataset(dataset)

    def _load_shard(self, local_ds_dir, remote_ds_dir, shard_file, shard_count, max_load_wait):
        """Load a shard, keeping the first one's first names for the writers of the query phases.

        Taken here because a shard fetched from S3 is deleted once it is loaded.
        """
        writer_names = os.path.join(local_ds_dir, WRITER_NAMES_FILE)
        if not os.path.isfile(writer_names):
            with open(os.path.join(local_ds_dir, shard_file), encoding="utf-8") as source:
                lines = [line for _, line in zip(range(WRITER_NAMES_ROWS), source)]
            with open(writer_names, "w", encoding="utf-8") as names:
                names.writelines(lines)
        return super()._load_shard(local_ds_dir, remote_ds_dir, shard_file, shard_count, max_load_wait)

    def _load_step_shards(self, step, bucket, prefix, local_ds_dir, remote_ds_dir, max_load_wait):
        """Create the index before the step's rows are loaded, when indexing during load."""
        self._step_idx += 1
        if self._index_during_load:
            # The name the flow will build this step under. Recomputed rather than passed, because
            # the load happens before the flow names the index -- _build_index checks the two agree.
            self._ingesting_index_name = f"{self.WORKLOAD.index_prefix}_{self._dataset_name}_{self._step_idx}"
            if self._variants:
                # One index per variant, named after it; the flow's own name stays the handle the
                # step is built and dropped under (see _drop_index).
                self._live_variants = {}
                for variant in self._variants:
                    name = f"{self._ingesting_index_name}_{variant['label']}"
                    self._create_index_for_ingestion(name, variant_params(variant))
                    self._live_variants[name] = variant["label"]
            else:
                self._create_index_for_ingestion(self._ingesting_index_name)
        loaded = super()._load_step_shards(step, bucket, prefix, local_ds_dir, remote_ds_dir, max_load_wait)
        # The clock the catch-up time of an index ingesting during the load is measured from.
        self._load_finished_at = time.monotonic()
        return loaded

    def _create_index_for_ingestion(self, index_name, option_params="", when="before the load, to index while loading"):
        """Create the index by itself, so that vector-store fills it: from the rows as they are
        written when created before the load, by its own full scan when created after it.

        Only the index existing in the schema matters, not which statement created it: the index
        node picks it up and ingests the base table's CDC log either way.
        """
        self.log.info("Creating index '%s' %s", index_name, when)
        self._run_latte(
            f"latte schema {self.WORKLOAD.script} "
            f"-P {WITH_INDEX_PARAM}=true "
            f'-P {self.WORKLOAD.params.index_name}=\\"{index_name}\\" '
            f"{option_params}",
            duration=_timeout_minutes(DEFAULT_SCHEMA_TIMEOUT),
        )

    def _build_index(self, record_count, max_index_wait, index_name, keyspace) -> float | None:
        """Build the index, or wait for the one that has been ingesting all along.

        Either way the size is read afterwards: vector-store refreshes an index's gauges when
        '/metrics' is scraped, so one request once the index is complete is both the cheapest and
        the most accurate moment to ask.
        """
        if self._index_during_load:
            if index_name != self._ingesting_index_name:
                raise ValueError(
                    f"Indexed into {self._ingesting_index_name!r} during the load but the flow "
                    f"expects {index_name!r}; the index naming in search_perf_test.py has changed"
                )
            names = list(self._live_variants) or [index_name]
            catch_up = self._wait_for_indexes_count(keyspace, names, record_count, max_index_wait)
            for name in names:
                self._wait_for_index_settled(keyspace, name, max_index_wait)
                self._report_index_size(keyspace, name, record_count, catch_up_secs=catch_up.get(name))
            build_seconds = None  # nothing was built here, so there is no build time to report
        elif self._variants:
            # The backfill case: every variant is created now, on the loaded table, and built by
            # vector-store's full scan. 'catch_up_secs' is then the build time, from the moment
            # the last CREATE INDEX was issued.
            self._ingesting_index_name = index_name
            self._live_variants = {}
            for variant in self._variants:
                name = f"{index_name}_{variant['label']}"
                self._create_index_for_ingestion(
                    name, variant_params(variant), when="on the loaded table, to build by full scan"
                )
                self._live_variants[name] = variant["label"]
            self._load_finished_at = time.monotonic()
            names = list(self._live_variants)
            catch_up = self._wait_for_indexes_count(keyspace, names, record_count, max_index_wait)
            for name in names:
                self._wait_for_index_settled(keyspace, name, max_index_wait)
                self._report_index_size(keyspace, name, record_count, catch_up_secs=catch_up.get(name))
            build_seconds = None  # per variant, in the size row, rather than one build time
        else:
            build_seconds = super()._build_index(record_count, max_index_wait, index_name=index_name, keyspace=keyspace)
            self._report_index_size(keyspace, index_name, record_count)
        # The index the query phases that follow will be priced against (see 'search_phase_end').
        # With variants it is not known until a round finds out which one serves.
        self._queried_index = None if self._live_variants else (keyspace, index_name)
        return build_seconds

    def _run_step_queries(self, step, dataset_name, defaults, record_count, step_number, local_ds_dir, remote_ds_dir):
        """Run the step's query sets once per index variant, or once as usual without variants.

        Without 'variants' on any query, each round runs every query against whichever index
        ScyllaDB picks, as before. With them, the rounds follow the plan's variant order: each runs
        the queries naming its variant (or none), and its first query has to reach that variant --
        an index serves only queries ordered its way, so a plan whose queries reach another index
        than it names is a plan or a selection bug, reported rather than measured under the wrong
        label.
        """
        # What a query phase's writer needs: where the step's files are, and what it has to write.
        self._step_context = {
            "defaults": defaults,
            "record_count": record_count,
            "local_ds_dir": local_ds_dir,
            "remote_ds_dir": remote_ds_dir,
        }
        self._write_next = max(getattr(self, "_write_next", 0), record_count)
        if not self._live_variants:
            return super()._run_step_queries(
                step, dataset_name, defaults, record_count, step_number, local_ds_dir, remote_ds_dir
            )
        keyspace = self._keyspace()
        filtered = any("variants" in query for query in step.get("queries", []))
        while self._live_variants:
            expected = None
            round_step = step
            if filtered:
                live_labels = set(self._live_variants.values())
                pending = [
                    (variant["label"], queries_for_variant(step["queries"], variant["label"]))
                    for variant in self._variants
                    if variant["label"] in live_labels
                ]
                pending = [(label, queries) for label, queries in pending if queries]
                if not pending:
                    break  # the variants left have nothing to answer; the flow drops them
                expected, queries = pending[0]
                round_step = dict(step, queries=queries)
            self._wait_for_cdc_quiet(keyspace, list(self._live_variants))
            served = self._detect_served_index(keyspace, round_step, defaults, local_ds_dir, remote_ds_dir)
            self._variant_label = self._live_variants[served]
            if expected is not None and self._variant_label != expected:
                raise RuntimeError(
                    f"The {round_step['queries'][0]['set']!r} probe reached variant {self._variant_label!r}, but "
                    f"the plan runs those queries against {expected!r}: the query's shape picks another index"
                )
            self._queried_index = (keyspace, served)
            self.log.info(
                "Query round against index '%s' (variant '%s'); %d variant(s) to go after it",
                served,
                self._variant_label,
                len(self._live_variants) - 1,
            )
            try:
                super()._run_step_queries(
                    round_step, dataset_name, defaults, record_count, step_number, local_ds_dir, remote_ds_dir
                )
            finally:
                self._variant_label = ""
            if len(self._live_variants) == 1:
                break  # the last one is dropped by the flow, under the step's own name
            self._drop_index(served, keyspace, self._max_index_wait)
        return None

    def _keyspace(self) -> str:
        return (self.params.get("latte_schema_parameters") or {}).get("keyspace") or self.WORKLOAD.default_keyspace

    def _detect_served_index(self, keyspace, step, defaults, local_ds_dir, remote_ds_dir) -> str:
        """Which of the live variant indexes ScyllaDB routes the step's queries to.

        A few seconds of the first query set, in its own shape, bracketed by two scrapes of every
        candidate's search total: the one that moved is the one serving. The shape matters: a query
        with ORDER BY goes to an index ordered that way, one without to an index in table order, so
        a plain probe could find another index than the step's queries use, or none. The choice is
        ScyllaDB's and stable for as long as the set of indexes does not change, which is why a
        round ends by dropping the index it measured.
        """
        params = self.WORKLOAD.params
        qset = _checked_name(step["queries"][0]["set"], "query set name")
        queries_file = f"queries_{qset}.tsv"
        before = self._scrape_search_totals(keyspace, list(self._live_variants))
        self._run_latte(
            f"latte run -f search {self.WORKLOAD.script} --duration {SERVED_INDEX_PROBE_DURATION} "
            rf"-P {params.dataset_dir}=\"{remote_ds_dir}/\" "
            rf"-P {params.queries_file}=\"{queries_file}\" "
            f"-P {params.compute_accuracy}=false "
            f"-P {params.search_limit}={defaults.get('limit', 20)} "
            f"{self._probe_shape(step['queries'][0])}"
            f"--concurrency=4 --retry-number 1 ",
            files_to_stage=[(os.path.join(local_ds_dir, queries_file), os.path.join(remote_ds_dir, queries_file))],
        )
        after = self._scrape_search_totals(keyspace, list(self._live_variants))
        moved = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in self._live_variants}
        served = max(moved, key=moved.get)
        if moved[served] <= 0:
            raise RuntimeError(
                f"The probe reached none of the candidate indexes {sorted(self._live_variants)}: {moved}"
            )
        return served

    def _scrape_search_totals(self, keyspace, index_names) -> dict[str, float]:
        """'pattern_search_total' per index, for the ones that export it."""
        metrics_text = self._vs_metrics()
        totals = {}
        for name in index_names:
            value = parse_index_gauge(metrics_text, SEARCHES_METRIC, keyspace, name)
            if value is not None:
                totals[name] = value
        return totals

    def _drop_index(self, index_name, keyspace, max_index_wait):
        """Drop one index -- or, asked for the step's own name while variants stand, all of them."""
        if index_name in self._live_variants:
            super()._drop_index(index_name, keyspace, max_index_wait)
            del self._live_variants[index_name]
            return
        if index_name == self._ingesting_index_name and self._live_variants:
            for name in list(self._live_variants):
                self._drop_index(name, keyspace, max_index_wait)
            return
        super()._drop_index(index_name, keyspace, max_index_wait)

    def search_extra_columns(self):
        return super().search_extra_columns() + WALK_COLUMNS + CPU_COLUMNS + VS_THREADS_COLUMNS + INGEST_COLUMNS

    def search_phase_begin(self, query: dict):
        # 'all_loaders: true' runs the phase from every loader at once: a second client, to tell a
        # limit of the client from one of the cluster.
        self._fan_out = bool(query.get("all_loaders", False))
        ingest = self._scrape_ingest()
        writer = self._start_writer(query)
        profiles = self._start_profiles(query) if query.get("profile") else []
        sampler = self._start_thread_sampler()
        cpu_window = self._start_cpu_window()
        return {
            "walk": self._scrape_walk_totals(),
            "cpu": self._scrape_cpu(),
            "profiles": profiles,
            "sampler": sampler,
            "cpu_window": cpu_window,
            "writer": writer,
            "ingest": ingest,
        }

    def search_phase_end(self, phase) -> dict:
        """What one query of the phase cost the index, how busy each node was, and what the writes did.

        The walk totals count every search the index served, so they also see whatever else queried
        it meanwhile -- nothing does in this flow. The CPU covers the window the profiles and the
        thread states do, PROFILE_DELAY_SECS in, when the phase lasts that long, and the whole latte
        run otherwise. The writer is waited for after the windows are closed, so its tail is not in
        them. A scrape that failed on either side leaves the cells empty rather than reporting a
        delta against nothing.
        """
        self._fan_out = False
        walk_after = self._scrape_walk_totals()
        cpu_after = self._scrape_cpu()
        window = self._collect_cpu_window(phase["cpu_window"])
        if window:
            cpu = cpu_busy_pct(*window)
            span = window[1]["time"] - window[0]["time"]
            what = f"over {span:.0f} s from {PROFILE_DELAY_SECS} s into the phase"
        else:
            cpu = cpu_busy_pct(phase["cpu"], cpu_after)
            what = "over the whole phase"
        if cpu:
            self.log.info("CPU busy %s: %s", what, ", ".join(f"{column}={value}" for column, value in cpu.items()))
        cpu |= self._collect_thread_sampler(phase["sampler"])
        for node, name in phase["profiles"]:
            self._collect_profile(node, name)
        cells = dict(cpu)
        if phase["writer"] is not None:
            cells |= self._finish_writer(phase["writer"])
        ingest = self._ingest_cells(phase["ingest"], self._scrape_ingest())
        if ingest:
            self.log.info(
                "Ingest over the phase: %s", ", ".join(f"{column}={value}" for column, value in ingest.items())
            )
        cells |= ingest
        walk_before = phase["walk"]
        if walk_before is None or walk_after is None:
            return cells
        walk = walk_per_query(walk_before, walk_after)
        if walk:
            self.log.info(
                "Index work per query: %s",
                ", ".join(f"{column}={value}" for column, value in walk.items()),
            )
        else:
            self.log.warning("The index served no searches during the phase; no per-query cost to report")
        return walk | cells

    def _start_writer(self, query: dict):
        """Start the phase's writer, if the plan gives it one; its stress thread, or None."""
        writer = checked_writer(query)
        if writer is None:
            return None
        rate, values = writer
        context = self._step_context
        seconds = duration_seconds(_query_duration(query, context["defaults"])) + WRITER_TAIL_SECS
        self._writer_count = getattr(self, "_writer_count", 0) + 1
        prefix = f"w{self._writer_count:03d}_"
        base = self._write_next
        # Every sort value it may give, so the next writer's are all newer: rows it did not get to
        # leave a gap, which an order does not mind.
        self._write_next += rate * seconds
        params = self.WORKLOAD.params
        local_names = os.path.join(context["local_ds_dir"], WRITER_NAMES_FILE)
        remote_names = os.path.join(context["remote_ds_dir"], WRITER_NAMES_FILE)
        self.log.info(
            "Writer '%s': %d rows a second, %s sort values from %d, for %d s", prefix, rate, values, base, seconds
        )
        return self.run_latte_thread(
            stress_cmd=(
                f"latte run -f write {self.WORKLOAD.script} "
                f"--duration {seconds}s --rate={rate} --concurrency={WRITER_CONCURRENCY} "
                rf"-P {params.dataset_dir}=\"{context['remote_ds_dir']}/\" "
                rf"-P {params.records_file}=\"{WRITER_NAMES_FILE}\" "
                rf"-P write_prefix=\"{prefix}\" -P write_base={base} "
                rf"-P write_values=\"{values}\" -P write_sort_range={max(context['record_count'], 1)} "
            ),
            duration=_timeout_minutes(seconds + DEFAULT_SCHEMA_TIMEOUT),
            extra_files_to_stage=[(local_names, remote_names)],
            round_robin=True,
        )

    def _finish_writer(self, thread) -> dict:
        """Wait for the writer to end by itself; the rate it achieved, as a cell."""
        try:
            results, errors = thread.parse_results()
        except Exception as exc:  # noqa: BLE001 - the writer is load, not the measurement
            self.log.warning("The phase's writer did not finish cleanly: %s", exc)
            return {}
        if errors:
            self.log.warning("The phase's writer reported errors: %s", errors)
        rates = [float(result.get("op rate") or 0) for result in results if isinstance(result, dict)]
        if not rates:
            return {}
        self.log.info("The phase's writer wrote %.0f rows a second", sum(rates))
        return {WRITE_OP_RATE_COLUMN: int(sum(rates))}

    def _scrape_ingest(self) -> dict | None:
        """The queried index's lag histogram, segment count, L0 rows and rewrite progress, or None."""
        queried = getattr(self, "_queried_index", None)
        if queried is None:
            return None
        keyspace, index_name = queried
        metrics_text = self._try_vs_metrics(index_name)
        if metrics_text is None:
            return None
        return {
            "lag": parse_indexing_lag(metrics_text, keyspace, index_name),
            "segments": parse_index_gauge(metrics_text, SEGMENT_COUNT_METRIC, keyspace, index_name),
            "l0": parse_index_gauge(metrics_text, L0_DOCS_METRIC, keyspace, index_name),
            "ranges_done": parse_index_gauge(metrics_text, REWRITE_RANGES_DONE_METRIC, keyspace, index_name),
            "merges_started": parse_index_gauge(metrics_text, MERGES_STARTED_METRIC, keyspace, index_name),
            "merges_ended": parse_index_gauge(metrics_text, MERGES_ENDED_METRIC, keyspace, index_name),
        }

    @staticmethod
    def _ingest_cells(before: dict | None, after: dict | None) -> dict:
        """The INGEST_COLUMNS cells from two '_scrape_ingest' readings."""
        if before is None or after is None:
            return {}
        cells = indexing_lag_cells(before["lag"], after["lag"])
        if after["segments"] is not None:
            cells[SEGMENTS_COLUMN] = int(after["segments"])
        if after["l0"] is not None:
            cells[L0_ROWS_COLUMN] = int(after["l0"])
        for key, column in (
            ("ranges_done", REWRITE_RANGES_COLUMN),
            ("merges_started", MERGES_STARTED_COLUMN),
            ("merges_ended", MERGES_ENDED_COLUMN),
        ):
            # .get: a reading from before vector-store exported the merges has no such key.
            if before.get(key) is not None and after.get(key) is not None:
                cells[column] = int(max(after[key] - before[key], 0))
        return cells

    def _diag(self, node) -> _BoundedRemoter:
        """The node's remoter for a diagnostic: bounded, and refused for a silent index node."""
        if getattr(self, "_vs_silent_since", None) is not None:
            try:
                is_vs = node is self._vector_store_node()
            except Exception:  # noqa: BLE001 - no index node, so this is not it
                is_vs = False
            if is_vs:
                raise DiagnosticSkipped(f"{node.name} answers no HTTP request; not reaching it over SSH either")
        return _BoundedRemoter(node.remoter, node.name)

    def _cpu_window_nodes(self) -> list:
        """(key, node) for each CPU the window reads, as '_scrape_cpu' keys them."""
        return [
            ("db", self.db_cluster.nodes[0]),
            ("loader", self.loaders.nodes[0]),
            ("vs", self._vector_store_node()),
        ]

    def _start_cpu_window(self) -> list:
        """Start CPU_WINDOW_SCRIPT on each node; [(key, node, out)] for the ones it started on."""
        self._cpu_window_count = getattr(self, "_cpu_window_count", 0) + 1
        out = f"/tmp/cpu-window-{self._cpu_window_count:02d}"
        installed = self.__dict__.setdefault("_cpu_window_installed", set())
        script = base64.b64encode(CPU_WINDOW_SCRIPT.encode()).decode()
        started = []
        for key, node in self._cpu_window_nodes():
            try:
                if node.name not in installed:
                    self._diag(node).run(f"echo {script} | base64 -d > /tmp/cpu_window.sh", verbose=False, timeout=30)
                    installed.add(node.name)
                self._diag(node).run(
                    f"rm -f {out}.a {out}.b {out}.stop {out}.done; "
                    f"setsid sh /tmp/cpu_window.sh {PROFILE_DELAY_SECS} {PROFILE_SECS} {out} "
                    f"> /dev/null 2>&1 < /dev/null &",
                    verbose=False,
                    timeout=30,
                )
                started.append((key, node, out))
            except Exception as exc:  # noqa: BLE001 - a diagnostic, not the measurement
                self.log.warning("Could not start the CPU window on %s: %s", node.name, exc)
        return started

    def _collect_cpu_window(self, started: list) -> tuple[dict, dict] | None:
        """Close the windows; ('_scrape_cpu'-shaped reading at the start, at the end), or None.

        None when the DB node's window never began -- the phase was shorter than the delay -- or
        could not be read; the caller then reports the whole phase.
        """
        readings = ({}, {})
        for key, node, out in started:
            try:
                text = (
                    self._diag(node)
                    .run(
                        f"touch {out}.stop; timeout 30 sh -c 'while [ ! -e {out}.done ]; do sleep 1; done'; "
                        f"cat {out}.a 2>/dev/null; echo '--- end of window ---'; cat {out}.b 2>/dev/null",
                        ignore_status=True,
                        verbose=False,
                        timeout=60,
                    )
                    .stdout
                )
            except Exception as exc:  # noqa: BLE001 - a diagnostic, not the measurement
                self.log.warning("Could not read the CPU window on %s: %s", node.name, exc)
                continue
            first, _, second = text.partition("--- end of window ---")
            for reading, part in zip(readings, (first, second)):
                snapshot = parse_cpu_snapshot(part.strip())
                if key == "db":
                    if "time" in snapshot:
                        reading["time"] = snapshot["time"]
                    if "reactor" in snapshot:
                        reading["db"] = snapshot["reactor"]
                    if "stat" in snapshot:
                        reading["db_host"] = snapshot["stat"]
                elif "stat" in snapshot:
                    reading[key] = snapshot["stat"]
        if "time" not in readings[0] or "time" not in readings[1] or readings[1]["time"] <= readings[0]["time"]:
            return None
        return readings

    def _run_latte(self, stress_cmd, files_to_stage=None, **kwargs):
        """One loader per command, as the base runs it, except in an 'all_loaders' phase."""
        if not getattr(self, "_fan_out", False):
            return super()._run_latte(stress_cmd, files_to_stage, **kwargs)
        thread = self.run_latte_thread(
            stress_cmd=stress_cmd, extra_files_to_stage=files_to_stage or [], round_robin=False, **kwargs
        )
        self.verify_stress_thread(thread)
        return thread

    def _start_thread_sampler(self) -> str | None:
        """Sample the index node's thread states in the background; the output file, or None."""
        node = self._vector_store_node()
        self._sampler_count = getattr(self, "_sampler_count", 0) + 1
        out = f"/tmp/vs-threads-{self._sampler_count:02d}.txt"
        try:
            if not getattr(self, "_sampler_installed", False):
                script = base64.b64encode(VS_THREAD_SAMPLER.encode()).decode()
                self._diag(node).run(f"echo {script} | base64 -d > /tmp/vs_threads.sh", verbose=False, timeout=30)
                self._sampler_installed = True
            self._diag(node).run(
                f"rm -f {out} {out}.stop; setsid sh /tmp/vs_threads.sh vector-store {PROFILE_DELAY_SECS} {PROFILE_SECS} {out} "
                f"> /dev/null 2>&1 < /dev/null &",
                verbose=False,
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001 - a diagnostic, not the measurement
            self.log.warning("Could not start the index node's thread sampler: %s", exc)
            return None
        return out

    def _collect_thread_sampler(self, out: str | None) -> dict:
        """The VS_THREADS_COLUMNS cell from the sampler's output, stopping it at the end of the phase."""
        if out is None:
            return {}
        node = self._vector_store_node()
        try:
            self._diag(node).run(
                f"touch {out}.stop; timeout 30 sh -c 'while [ ! -e {out} ]; do sleep 1; done'",
                ignore_status=True,
                verbose=False,
                timeout=60,
            )
            text = self._diag(node).run(f"cat {out}", ignore_status=True, verbose=False, timeout=30).stdout
        except Exception as exc:  # noqa: BLE001 - a diagnostic, not the measurement
            self.log.warning("Could not read the index node's thread states: %s", exc)
            return {}
        states = parse_thread_states(text)
        if not states:
            # No samples is what a phase shorter than PROFILE_DELAY_SECS gets, not a failure.
            log = self.log.info if text.startswith("no samples") else self.log.warning
            log("The index node's thread sampler said: %s", text.strip()[:300])
            return {}
        self.log.info(
            "Index node threads (average count by name and state): %s",
            ", ".join(f"{key}={value}" for key, value in sorted(states.items())),
        )
        return {VS_WORKERS_COLUMN: states.get(f"{VS_WORKER_THREAD} R", 0.0)}

    def _perf_ready(self, node) -> bool:
        """Whether 'perf' runs on the node, installing it the first time it is asked for.

        Asked once per node and run: a node where it cannot be had (a container, a kernel with no
        tools package) is reported once and its profiles skipped, not the run failed.
        """
        available = self.__dict__.setdefault("_perf_available", {})
        if node.name not in available:
            try:
                if not self._diag(node).sudo("perf --version", ignore_status=True, verbose=False, timeout=30).ok:
                    self._install_perf(node)
                available[node.name] = self._diag(node).sudo("perf --version", ignore_status=True, timeout=30).ok
            except Exception as exc:  # noqa: BLE001 - a profile is an extra, not the measurement
                self.log.warning("Could not install perf on %s: %s", node.name, exc)
                available[node.name] = False
            if not available[node.name]:
                self.log.warning("perf does not run on %s; its profiles are skipped", node.name)
        return available[node.name]

    def _install_perf(self, node) -> None:
        """Install the running kernel's perf on an apt node, in bounded time.

        Not 'install_package', which retries a failed install for minutes: this runs between two
        query phases, and a package that is not there will not appear. A node without apt (the
        containerized backends) is left as it is.
        """
        if not self._diag(node).run("command -v apt-get", ignore_status=True, verbose=False, timeout=30).ok:
            return
        kernel = self._diag(node).run("uname -r", verbose=False, timeout=30).stdout.strip()
        self._diag(node).sudo(
            "apt-get -o Acquire::http::Timeout=60 -o Acquire::Retries=3 update", ignore_status=True, timeout=300
        )
        self._diag(node).sudo(
            "DEBIAN_FRONTEND=noninteractive apt-get install "
            "-o DPkg::Lock::Timeout=120 -o Acquire::http::Timeout=60 -o Acquire::Retries=3 "
            '-o Dpkg::Options::="--force-confold" -o Dpkg::Options::="--force-confdef" '
            f"-y linux-tools-{kernel} linux-tools-common",
            ignore_status=True,
            timeout=600,
            retry=2,
        )
        # An HWE kernel's tools next to an older 'linux-tools-common', whose 'perf' wrapper then
        # looks under the wrong version ("perf not found for kernel"): the installed binary itself.
        # Searched for rather than guessed: run a0e51aa9's arm64 index node (6.14.0-1011-aws) had
        # the tools package installed and its binary at neither path the guess named.
        if not self._diag(node).sudo("perf --version", ignore_status=True, verbose=False, timeout=30).ok:
            self._diag(node).sudo(
                f'sh -c \'p=/usr/lib/linux-tools/{kernel}/perf; [ -x "$p" ] || '
                'p=$(find /usr/lib -maxdepth 3 -type f -name perf -path "*tools*" -perm -u+x 2>/dev/null | sort -r | head -1); '
                '[ -n "$p" ] && ln -sf "$p" /usr/local/bin/perf\'',
                ignore_status=True,
                timeout=30,
            )
        if not self._diag(node).sudo("perf --version", ignore_status=True, verbose=False, timeout=30).ok:
            found = self._diag(node).run(
                "ls -d /usr/lib/linux*tools* 2>&1", ignore_status=True, verbose=False, timeout=30
            )
            self.log.warning(
                "No perf binary for kernel %s on %s; tools directories: %s", kernel, node.name, found.stdout.strip()
            )

    def _start_profiles(self, query: dict) -> list:
        """Start sampling the DB node and the index node; the (node, recording) pairs started."""
        self._profile_count = getattr(self, "_profile_count", 0) + 1
        base = f"perf-{self._profile_count:02d}-{_checked_name(query['set'], 'query set name')}"
        started = []
        for node, side in ((self.db_cluster.nodes[0], "db"), (self._vector_store_node(), "vs")):
            name = f"{base}-{side}"
            if self._perf_ready(node) and self._start_profile(node, name):
                started.append((node, name))
        if started:
            self.log.info("Profiling for %s as %s", query, ", ".join(name for _, name in started))
        return started

    def _start_profile(self, node, name: str) -> bool:
        """Start PROFILE_SCRIPT on the node in the background: a system-wide recording once the
        node is busy."""
        script = base64.b64encode(PROFILE_SCRIPT.encode()).decode()
        try:
            self._diag(node).run(f"echo {script} | base64 -d > /tmp/profile.sh", verbose=False, timeout=30)
            self._diag(node).sudo(
                f"setsid sh /tmp/profile.sh /tmp/{name}.data {PROFILE_HZ} {PROFILE_STACK_BYTES} {PROFILE_SECS} "
                f"> /tmp/{name}.log 2>&1 < /dev/null &",
                verbose=False,
                timeout=30,
            )
        except Exception:  # noqa: BLE001 - a profile is an extra, not the measurement
            return False
        return True

    def _collect_profile(self, node, name: str) -> None:
        """Wait for the recording to end, then gzip it and bring it to '<logdir>/perf/'."""
        dest = os.path.join(self.logdir, "perf")
        # '[p]erf-...' so that the pattern does not match the waiting shell's own command line.
        pattern = f"/tmp/[{name[0]}]{name[1:]}.data"
        try:
            self._diag(node).run(
                f"timeout {PROFILE_MAX_WAIT_SECS + PROFILE_SETTLE_SECS + PROFILE_SECS + 60} sh -c "
                f"'while pgrep -f \"{pattern}\" > /dev/null; do sleep 2; done'",
                ignore_status=True,
                verbose=False,
                timeout=PROFILE_MAX_WAIT_SECS + PROFILE_SETTLE_SECS + PROFILE_SECS + 90,
            )
            log = self._diag(node).sudo(f"tail -3 /tmp/{name}.log", ignore_status=True, verbose=False).stdout.strip()
            self.log.info("Profile %s: %s", name, log.replace("\n", " | "))
            # One root shell for both: 'sudo a && b' would run b as the login user, on a root-owned file.
            self._diag(node).sudo(f"sh -c 'gzip -1 -f /tmp/{name}.data && chmod 644 /tmp/{name}.data.gz'", timeout=300)
            os.makedirs(dest, exist_ok=True)
            self._diag(node).receive_files(src=f"/tmp/{name}.data.gz", dst=dest, timeout=600)
            self._diag(node).sudo(f"rm -f /tmp/{name}.data.gz", ignore_status=True, verbose=False)
        except Exception as exc:  # noqa: BLE001 - a profile is an extra, not the measurement
            self.log.warning("Could not collect profile %s: %s", name, exc)

    def _scrape_cpu(self) -> dict:
        """ScyllaDB's reactor busy times and the DB, loader and index nodes' /proc/stat, now.

        Each part is left out when its read fails, which must not end the run.
        """
        reading = {"time": time.monotonic()}
        reads = (
            ("db", lambda: self.db_cluster.nodes[0], "curl -s http://localhost:9180/metrics", parse_reactor_busy),
            ("db_host", lambda: self.db_cluster.nodes[0], "head -1 /proc/stat", parse_proc_stat),
            ("loader", lambda: self.loaders.nodes[0], "head -1 /proc/stat", parse_proc_stat),
            ("vs", self._vector_store_node, "head -1 /proc/stat", parse_proc_stat),
        )
        for key, node, cmd, parse in reads:
            try:
                result = self._diag(node()).run(cmd, ignore_status=True, verbose=False, timeout=30)
                value = parse(result.stdout) if result.ok else None
            except Exception as exc:  # noqa: BLE001 - a scrape failure must not end the run
                self.log.warning("Could not read the %s CPU: %s", key, exc)
                value = None
            if value is not None:
                reading[key] = value
        return reading

    def _scrape_walk_totals(self) -> dict[str, float] | None:
        """The index's search totals now, or None if there is no index or the scrape failed."""
        queried = getattr(self, "_queried_index", None)
        if queried is None:
            return None
        keyspace, index_name = queried
        metrics_text = self._try_vs_metrics(index_name)
        if metrics_text is None:
            return None
        totals = {}
        for metric in [SEARCHES_METRIC] + [metric for metric, _ in WALK_PER_QUERY_METRICS.values()]:
            value = parse_index_gauge(metrics_text, metric, keyspace, index_name)
            if value is not None:
                totals[metric] = value
        return totals

    def _log_segment_layout(self, index_name, segments, spans_pct):
        """One line per segment: what an ordered search sees when it decides what to skip."""
        self.log.info(
            "Index '%s' layout: %d segments, mean span %.1f%% of the sort range, widest %.1f%%",
            index_name,
            len(segments),
            sum(spans_pct) / len(spans_pct) if spans_pct else 0.0,
            max(spans_pct, default=0.0),
        )
        for (ordinal, docs, low, high), span in list(zip(segments, spans_pct))[:SEGMENT_LAYOUT_LOG_ROWS]:
            if low is None or high is None:
                self.log.info("  segment %3d: %9d rows", ordinal, int(docs))
            else:
                self.log.info(
                    "  segment %3d: %9d rows, sort %s..%s (%.1f%%)",
                    ordinal,
                    int(docs),
                    f"{low:.0f}",
                    f"{high:.0f}",
                    span,
                )
        if len(segments) > SEGMENT_LAYOUT_LOG_ROWS:
            self.log.info("  ... and %d more", len(segments) - SEGMENT_LAYOUT_LOG_ROWS)

    # What _index_status answers when vector-store did not: the poll is skipped, not counted as 0.
    UNANSWERED = object()

    def _index_status(self, client, keyspace, index_name):
        """The index's status, None while vector-store has not discovered it, or UNANSWERED.

        A status request that times out or cannot connect is one missed sample, not a failed run:
        the runner reaches the index node over the internet, and a run of several hours sees the
        odd dropped connection (run eb0023ac lost its second half to one 30 s read timeout). The
        caller's deadline still bounds the wait.
        """
        try:
            status = client.get_index_status_or_none(keyspace.lower(), index_name.lower())
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            self._note_vs_answer(answered=False)
            self.log.warning("No status for index '%s' this time, polling on: %s", index_name, exc)
            return self.UNANSWERED
        self._note_vs_answer(answered=True)
        return status

    def _note_vs_answer(self, answered: bool):
        """Track how long the index node has answered nothing; past VS_UNRESPONSIVE_ABORT_SECS of
        silence, raise VectorStoreUnresponsive rather than poll a dead node until a deadline."""
        now = time.monotonic()
        if answered:
            self._vs_silent_since = None
            return
        since = getattr(self, "_vs_silent_since", None)
        if since is None:
            self._vs_silent_since = now
        elif now - since >= VS_UNRESPONSIVE_ABORT_SECS:
            raise VectorStoreUnresponsive(
                f"The vector-store node has answered no request for {now - since:.0f} s; giving up on the run"
            )

    def _vs_metrics(self) -> str:
        """The index node's /metrics text. A failed scrape raises, and counts towards giving up."""
        try:
            text = self._vector_store_api_client().request("GET", "/metrics").text
        except requests.exceptions.ConnectionError, requests.exceptions.Timeout:
            self._note_vs_answer(answered=False)
            raise
        self._note_vs_answer(answered=True)
        return text

    def _try_vs_metrics(self, index_name: str | None = None) -> str | None:
        """'_vs_metrics', or None after a warning: a missed scrape must not end the run, unless the
        node has gone silent for good (VectorStoreUnresponsive)."""
        try:
            return self._vs_metrics()
        except VectorStoreUnresponsive:
            raise
        except Exception as exc:  # noqa: BLE001 - a scrape failure must not end the run
            target = f" for '{index_name}'" if index_name else ""
            self.log.warning("Could not scrape vector-store metrics%s: %s", target, exc)
            return None

    def _wait_for_indexes_count(self, keyspace, index_names, record_count, timeout) -> dict[str, float]:
        """Wait until every named index holds every loaded row; per index, how long after the load
        it got there.

        Polled together rather than one after another, so that an index that caught up while
        another was still ingesting is credited with the time it actually took.
        """
        client = self._vector_store_api_client()
        load_end = getattr(self, "_load_finished_at", None) or time.monotonic()
        deadline = time.monotonic() + timeout
        pending = {name: -1 for name in index_names}
        caught_up = {}
        while pending:
            for name in list(pending):
                status = self._index_status(client, keyspace, name)
                if status is self.UNANSWERED:
                    continue
                count = (status or {}).get("count", 0)
                if count >= record_count:
                    caught_up[name] = round(max(time.monotonic() - load_end, 0.0), 1)
                    self.log.info(
                        "Index '%s' has caught up: %d of %d %s, %.1f s after the load",
                        name,
                        count,
                        record_count,
                        self.WORKLOAD.item_noun,
                        caught_up[name],
                    )
                    del pending[name]
                elif count != pending[name]:
                    self.log.info(
                        "Index '%s' ingesting: %d of %d %s", name, count, record_count, self.WORKLOAD.item_noun
                    )
                    pending[name] = count
            if not pending:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Indexes {sorted(pending)} had not caught up with {record_count} "
                    f"{self.WORKLOAD.item_noun} after {timeout}s of ingestion: {pending}"
                )
            time.sleep(INDEX_COUNT_POLL_INTERVAL_SECS)
        return caught_up

    def _wait_for_index_count(self, keyspace, index_name, record_count, timeout):
        """Wait until the index holds every row that was loaded.

        This is the completion test, and it is deliberately not "is the last row I wrote there?".
        The index node consumes the base table's CDC log, which is read per stream, one per token
        range, and the streams advance independently -- so the last row written can be indexed
        while another stream is still far behind. Asking for the count asks about the whole index.
        """
        client = self._vector_store_api_client()
        # Folded, because that is how vector-store knows an index (see 'index_key').
        ks, idx = keyspace.lower(), index_name.lower()
        deadline = time.monotonic() + timeout
        last_seen = -1
        while True:
            status = self._index_status(client, ks, idx)
            count = last_seen if status is self.UNANSWERED else (status or {}).get("count", 0)
            if count >= record_count:
                self.log.info(
                    "Index '%s' has caught up: %d of %d %s", index_name, count, record_count, self.WORKLOAD.item_noun
                )
                return
            if count != last_seen:
                self.log.info(
                    "Index '%s' ingesting: %d of %d %s", index_name, count, record_count, self.WORKLOAD.item_noun
                )
                last_seen = count
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Index {index_name!r} held {count} of {record_count} "
                    f"{self.WORKLOAD.item_noun} after {timeout}s of ingestion"
                )
            time.sleep(INDEX_COUNT_POLL_INTERVAL_SECS)

    def _wait_for_cdc_quiet(self, keyspace, index_names):
        """Wait until the indexes stop taking in changes through CDC; see CDC_QUIET_POLLS."""
        started = time.monotonic()
        previous, quiet = None, 0
        while time.monotonic() - started < self._max_settle:
            metrics_text = self._try_vs_metrics()
            if metrics_text is None:
                time.sleep(SETTLE_POLL_INTERVAL_SECS)
                continue
            totals = {name: cdc_changes_total(metrics_text, keyspace, name) or 0.0 for name in index_names}
            if previous is not None:
                moved = sum(totals[name] - previous.get(name, 0.0) for name in totals)
                quiet = quiet + 1 if moved < CDC_QUIET_CHANGES else 0
                if quiet >= CDC_QUIET_POLLS:
                    self.log.info("CDC quiet on %s after %.0f s", ", ".join(index_names), time.monotonic() - started)
                    return
                if moved >= CDC_QUIET_CHANGES:
                    self.log.info("Indexes still taking in changes: %d in %.0f s", moved, SETTLE_POLL_INTERVAL_SECS)
            previous = totals
            time.sleep(SETTLE_POLL_INTERVAL_SECS)
        self.log.warning(
            "Indexes %s still taking in changes after %d s; the round starts anyway", index_names, self._max_settle
        )

    def _wait_for_index_settled(self, keyspace, index_name, timeout):
        """Wait until the segment count stops moving, or give up and say so.

        An index ingested incrementally is many small segments that Tantivy goes on merging after
        the writes stop, and a search visits every one of them, so querying the moment the last row
        lands measures a transient rather than the index. Not reaching a stable count is reported
        rather than failed: the queries are still worth running, they just start against a moving
        target, and the segment count in the size row says what they ran against.
        """
        started = time.monotonic()
        deadline = started + min(timeout, self._max_settle)
        previous, stable = None, 0
        while True:
            segments = self._read_index_gauge(keyspace, index_name, SEGMENT_COUNT_METRIC)
            if segments is None:
                self.log.warning(
                    "No '%s' sample for '%s'; not waiting for it to settle", SEGMENT_COUNT_METRIC, index_name
                )
                return
            # A rewrite in progress moves a range per tick and changes the layout every time; the
            # index is settled only once it has finished (and the merges after it have stopped).
            ranges_total = self._read_index_gauge(keyspace, index_name, REWRITE_RANGES_TOTAL_METRIC) or 0
            ranges_done = self._read_index_gauge(keyspace, index_name, REWRITE_RANGES_DONE_METRIC) or 0
            rewriting = ranges_done < ranges_total
            # A segment count can hold still while merges run, and run a0e51aa9's did, for ten
            # minutes, with a backlog of thousands; a vector-store without the gauge reads as none.
            merging = self._read_index_gauge(keyspace, index_name, MERGES_RUNNING_METRIC) or 0
            if rewriting:
                self.log.info(
                    "Index '%s' rewriting wide segments: %d of %d ranges moved, %d segments, %d merges running",
                    index_name,
                    int(ranges_done),
                    int(ranges_total),
                    int(segments),
                    int(merging),
                )
            elif merging:
                self.log.info(
                    "Index '%s' merging: %d merges running, %d segments", index_name, int(merging), int(segments)
                )
            stable = stable + 1 if segments == previous and not rewriting and not merging else 0
            previous = segments
            if stable >= SETTLE_STABLE_POLLS:
                self.log.info(
                    "Index '%s' settled at %d segments, %.0f s after it caught up",
                    index_name,
                    int(segments),
                    time.monotonic() - started,
                )
                return
            if time.monotonic() >= deadline:
                self.log.warning(
                    "Index '%s' still merging at %d segments %.0f s after it caught up; querying anyway",
                    index_name,
                    int(segments),
                    time.monotonic() - started,
                )
                return
            time.sleep(SETTLE_POLL_INTERVAL_SECS)

    def _read_index_gauge(self, keyspace, index_name, metric) -> float | None:
        """One vector-store gauge for one index, or None if it cannot be read."""
        metrics_text = self._try_vs_metrics(index_name)
        if metrics_text is None:
            return None
        return parse_index_gauge(metrics_text, metric, keyspace, index_name)

    def _report_index_size(self, keyspace, index_name, record_count, catch_up_secs=None):
        """Read the index size gauges off the vector-store node and submit them as one row.

        A missing gauge is reported as a missing measurement rather than a failure, exactly as a
        missing build time is: the index answers queries either way, and the query phase that
        follows is the more important half of the run.
        """
        metrics_text = self._try_vs_metrics(index_name)
        if metrics_text is None:
            return

        size_bytes = parse_index_gauge(metrics_text, INDEX_SIZE_METRIC, keyspace, index_name)
        if size_bytes is None:
            self.log.warning("No '%s' sample for index '%s'; skipping index size", INDEX_SIZE_METRIC, index_name)
            return
        segments = parse_index_gauge(metrics_text, SEGMENT_COUNT_METRIC, keyspace, index_name)
        bytes_per_name = round(size_bytes / record_count, 2) if record_count else 0.0
        layout = parse_segment_layout(metrics_text, keyspace, index_name)
        spans_pct = segment_spans_pct(layout)
        if layout:
            self._log_segment_layout(index_name, layout, spans_pct)

        self.log.info(
            "Index '%s': %d bytes for %d names (%.2f bytes/name), %s segments",
            index_name,
            int(size_bytes),
            record_count,
            bytes_per_name,
            int(segments) if segments is not None else "unknown",
        )

        result_table = SubstringIndexSizeResult()
        row_key = index_name
        result_table.add_result(column="index_size_bytes", row=row_key, value=int(size_bytes), status=Status.UNSET)
        result_table.add_result(column="bytes_per_name", row=row_key, value=bytes_per_name, status=Status.UNSET)
        result_table.add_result(
            column=SUBSTRING_BUILD_COUNT_COLUMN, row=row_key, value=record_count, status=Status.UNSET
        )
        result_table.add_result(
            column="segment_count", row=row_key, value=int(segments) if segments is not None else 0, status=Status.UNSET
        )
        if spans_pct:
            result_table.add_result(
                column="segment_span_mean_pct",
                row=row_key,
                value=round(sum(spans_pct) / len(spans_pct), 2),
                status=Status.UNSET,
            )
            result_table.add_result(
                column="segment_span_max_pct", row=row_key, value=max(spans_pct), status=Status.UNSET
            )
        if catch_up_secs is not None:
            result_table.add_result(column="catch_up_secs", row=row_key, value=catch_up_secs, status=Status.UNSET)
        l0_rows = parse_index_gauge(metrics_text, L0_DOCS_METRIC, keyspace, index_name)
        rewrite_ranges = parse_index_gauge(metrics_text, REWRITE_RANGES_TOTAL_METRIC, keyspace, index_name)
        if l0_rows is not None:
            result_table.add_result(column="l0_rows", row=row_key, value=int(l0_rows), status=Status.UNSET)
        if rewrite_ranges is not None:
            result_table.add_result(
                column="rewrite_ranges", row=row_key, value=int(rewrite_ranges), status=Status.UNSET
            )
            if rewrite_ranges:
                self.log.info(
                    "Index '%s': %d rows in wide segments were rewritten in %d ranges",
                    index_name,
                    int(l0_rows or 0),
                    int(rewrite_ranges),
                )
        submit_results_to_argus(self.test_config.argus_client(), result_table)

    def _start_vs_memory_log(self):
        """Start VS_MEMORY_LOG_SCRIPT on the index node; a diagnostic, so a failure only warns."""
        script = base64.b64encode(VS_MEMORY_LOG_SCRIPT.encode()).decode()
        try:
            node = self._vector_store_node()
            self._diag(node).run(f"echo {script} | base64 -d > /tmp/vs_memory.sh", verbose=False, timeout=30)
            self._diag(node).run(
                "setsid sh /tmp/vs_memory.sh > /dev/null 2>&1 < /dev/null &", verbose=False, timeout=30
            )
        except Exception as exc:  # noqa: BLE001 - a diagnostic, not the measurement
            self.log.warning("Could not start the memory log on the vector-store node: %s", exc)

    def test_substring_search(self):
        self._start_vs_memory_log()
        self.run_search_benchmark()
