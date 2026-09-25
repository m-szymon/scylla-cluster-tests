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

ScyllaDB's `substring_index` answers `WHERE column LIKE '%keyword%' LIMIT n` from an n-gram index
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

import os
import re
import time

from argus.client.generic_result import ColumnMetadata, ResultType, StaticGenericResultTable, Status

from sdcm.argus_results import submit_results_to_argus
from search_perf_test import (
    DEFAULT_MAX_INDEX_WAIT,
    DEFAULT_SCHEMA_TIMEOUT,
    LatteScriptParams,
    SearchPerformanceTest,
    SearchWorkload,
    _checked_name,
    _timeout_minutes,
)
from sdcm.utils.vector_store_index import index_build_columns

SUBSTRING_BASE_DIR = "data_dir/latte/substring_search"


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
INDEX_SIZE_METRIC = "substring_index_size_bytes"
SEGMENT_COUNT_METRIC = "substring_segment_count"
# One sample per segment, labelled 'segment' with the segment's ordinal. The bounds are the sort
# column's values (vector-store undoes the sort-key bias on export), or absent for an unordered index.
SEGMENT_DOCS_METRIC = "substring_segment_docs"
SEGMENT_SORT_MIN_METRIC = "substring_segment_sort_min"
SEGMENT_SORT_MAX_METRIC = "substring_segment_sort_max"
# Running totals of what the index's searches did, growing since the index was created. The delta
# over a query phase, divided by the searches served in it, says what one query cost the index --
# which is what tells a slow walk from a wide-segment layout when the latency alone cannot.
SEARCHES_METRIC = "substring_search_total"
# Column of the search latency table -> (metric, unit scale applied to the per-query value).
WALK_PER_QUERY_METRICS = {
    "walk_us_per_query": ("substring_search_walk_seconds_total", 1e6),
    # Part of the walk time: turning the page into primary ids, which is the store reads unless
    # the index keeps the id as a column.
    "page_resolve_us_per_query": ("substring_search_page_resolve_seconds_total", 1e6),
    "segments_considered_per_query": ("substring_search_segments_considered_total", 1),
    "segments_opened_per_query": ("substring_search_segments_opened_total", 1),
    "postings_per_query": ("substring_search_postings_scanned_total", 1),
    "heap_entrants_per_query": ("substring_search_heap_entrants_total", 1),
    "store_reads_per_query": ("substring_search_store_reads_total", 1),
}
SEARCHES_COLUMN = "index_searches"
WALK_COLUMNS = [ColumnMetadata(name=SEARCHES_COLUMN, unit="", type=ResultType.INTEGER)] + [
    ColumnMetadata(name=column, unit="us" if column.endswith("_us_per_query") else "", type=ResultType.FLOAT, higher_is_better=False)
    for column in WALK_PER_QUERY_METRICS
]
# How many segments the layout log lists in full; the summary columns cover the rest.
SEGMENT_LAYOUT_LOG_ROWS = 64
_METRIC_LINE_RE_TEMPLATE = r"^{metric}\{{(?P<labels>[^}}]*)\}}\s+(?P<value>[0-9.eE+-]+)\s*$"
_LABEL_RE = re.compile(r'(?P<name>\w+)="(?P<value>[^"]*)"')

# The '-P' name that makes substring.rn create the index in its 'schema' phase. Not part of
# LatteScriptParams, which describes only what the shared flow drives.
WITH_INDEX_PARAM = "with_index"
# The '-P' names carrying one extra WITH OPTIONS entry, name and value apart: latte splices a
# value into the script as a literal, so a quoted fragment would not parse.
EXTRA_OPTION_NAME_PARAM = "extra_option_name"
EXTRA_OPTION_VALUE_PARAM = "extra_option_value"

# A dataset may ask for several indexes over the one load, differing only in their options:
#
#   index_variants:
#     - label: store_id
#     - label: fast_id
#       options: {poc_option_1: 'true'}
#
# All of them are created before the load and ingest the same CDC stream, so they get equivalent
# layouts, which two separate runs would not (merge timing differs per run). ScyllaDB picks one of
# them for every query and the choice is not by name, so each round of the query sets first finds
# out which index is serving (a short probe, then a look at which one's search total moved), labels
# its rows with that variant, and drops the index afterwards so the next round reaches another.
# Needs 'index_during_load': a variant built by a full scan after the load would have a different
# layout from one fed by CDC, and the comparison would be about that instead.
INDEX_VARIANTS_KEY = "index_variants"
SERVED_INDEX_PROBE_DURATION = "5s"

# Polling for 'index_during_load'. The count is asked for on this interval while the index catches
# up with the load; the settle loop then watches the segment count until it stops moving, because
# Tantivy keeps merging after the writes stop and query latency depends on how many segments a
# search has to visit.
INDEX_COUNT_POLL_INTERVAL_SECS = 15.0
SETTLE_POLL_INTERVAL_SECS = 10.0
SETTLE_STABLE_POLLS = 3
DEFAULT_SETTLE_TIMEOUT_SECS = 600


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


def parse_index_gauge(metrics_text: str, metric: str, keyspace: str, index_name: str) -> float | None:
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
        ):
            return float(match.group("value"))
    return None


def checked_index_variants(dataset: dict) -> list[dict]:
    """The dataset's 'index_variants' as [{label, options}], validated, or [] when it has none."""
    variants = dataset.get(INDEX_VARIANTS_KEY) or []
    if not variants:
        return []
    if not dataset.get("index_during_load"):
        raise ValueError(f"Dataset {dataset.get('name')!r}: '{INDEX_VARIANTS_KEY}' needs 'index_during_load: true'")
    checked = []
    for variant in variants:
        if not isinstance(variant, dict) or "label" not in variant:
            raise ValueError(f"Dataset {dataset.get('name')!r}: every index variant needs a 'label'")
        label = _checked_name(str(variant["label"]), "index variant label")
        options = variant.get("options") or {}
        if not isinstance(options, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in options.items()):
            raise ValueError(f"Index variant {label!r}: 'options' must map option names to string values")
        index_option_params(options)  # rejects what the script cannot carry
        if any(label == seen["label"] for seen in checked):
            raise ValueError(f"Index variant label {label!r} is used twice")
        checked.append({"label": label, "options": options})
    return checked


def index_option_params(options: dict) -> str:
    """The '-P' flags carrying a variant's extra index option to substring.rn, or "" for none.

    One option per variant for now, which is what the script takes; the value and name are plain
    words, since they travel through a shell command line and into a rune literal.
    """
    if not options:
        return ""
    if len(options) > 1:
        raise ValueError(f"An index variant can set one extra option, not {sorted(options)}")
    ((name, value),) = options.items()
    for text in (name, value):
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", text):
            raise ValueError(f"Index option {name!r}={value!r}: only letters, digits, '_', '.' and '-' can be passed")
    return f'-P {EXTRA_OPTION_NAME_PARAM}=\\"{name}\\" -P {EXTRA_OPTION_VALUE_PARAM}=\\"{value}\\" '


def parse_index_gauge_series(metrics_text: str, metric: str, keyspace: str, index_name: str, label: str) -> dict[str, float]:
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
        ):
            series[labels[label]] = float(match.group("value"))
    return series


def parse_segment_layout(metrics_text: str, keyspace: str, index_name: str) -> list[tuple[int, float, float | None, float | None]]:
    """The index's segments as (ordinal, live rows, sort min, sort max), by ordinal.

    The bounds are None for an unordered index, which exports none.
    """
    docs = parse_index_gauge_series(metrics_text, SEGMENT_DOCS_METRIC, keyspace, index_name, "segment")
    lows = parse_index_gauge_series(metrics_text, SEGMENT_SORT_MIN_METRIC, keyspace, index_name, "segment")
    highs = parse_index_gauge_series(metrics_text, SEGMENT_SORT_MAX_METRIC, keyspace, index_name, "segment")
    return [
        (int(ordinal), docs[ordinal], lows.get(ordinal), highs.get(ordinal))
        for ordinal in sorted(docs, key=int)
    ]


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
        if window > 0.0:
            shape = f" ordered from {window:g}" if ordered else f" window from {window:g}"
        else:
            shape = " ordered" if ordered else ""
        # Which of the dataset's index variants answered; "" outside a variant round. Looked up
        # rather than read, since plan validation calls this on a stand-in for the test.
        variant = getattr(self, "_variant_label", "")
        return f"{shape} [{variant}]" if variant else shape

    def extra_search_params(self, query: dict, record_count: int) -> str:
        """Translate the plan's 'ordered' and 'window' keys into substring.rn's '-P' flags.

        'window: 0.5' asks for the page that begins halfway down the order, which is how a later
        page is measured: latte drives no CQL paging, but a page resuming at a cursor is exactly a
        query bounded by it, and keeping that flat is what the cursor is for.
        """
        ordered = bool(query.get("ordered", False))
        window = _checked_window(query.get("window", 0.0))
        if not ordered and window <= 0.0:
            return ""
        if not self._order_by:
            # Nothing downstream would fail: the index has no sort column, so substring.rn refuses
            # the shape -- but it refuses inside the loader, after the corpus is loaded and indexed.
            raise ValueError(
                f"Query set {query.get('set')!r} asks for an ordered or windowed search, but "
                f"'latte_schema_parameters' sets no 'order_by', so the index has nothing to order by"
            )
        params = f"-P search_ordered={'true' if ordered else 'false'} "
        if window > 0.0:
            params += f"-P search_window_from={window} -P sort_value_count={record_count} "
        return params

    @property
    def _order_by(self) -> str:
        """The column the index orders by, from the test case, or "" when it has none."""
        return (self.params.get("latte_schema_parameters") or {}).get("order_by") or ""

    # Set per dataset from the plan's 'index_during_load'; see _run_dataset.
    _index_during_load = False
    _dataset_name = ""
    _step_idx = -1
    _ingesting_index_name = ""
    _max_index_wait = DEFAULT_MAX_INDEX_WAIT
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
        self._variants = checked_index_variants(dataset)
        self._live_variants = {}
        self._variant_label = ""
        if self._index_during_load:
            self.log.info("Dataset '%s': indexing during load; no build time will be measured", self._dataset_name)
        return super()._run_dataset(dataset)

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
                    self._create_index_for_ingestion(name, index_option_params(variant["options"]))
                    self._live_variants[name] = variant["label"]
            else:
                self._create_index_for_ingestion(self._ingesting_index_name)
        return super()._load_step_shards(step, bucket, prefix, local_ds_dir, remote_ds_dir, max_load_wait)

    def _create_index_for_ingestion(self, index_name, option_params=""):
        """Create the index up front, so that rows are indexed as they are written.

        Only the index existing in the schema matters, not which statement created it: the index
        node picks it up and ingests the base table's CDC log either way.
        """
        self.log.info("Creating index '%s' before the load, to index while loading", index_name)
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
            for name in self._live_variants or [index_name]:
                self._wait_for_index_count(keyspace, name, record_count, max_index_wait)
                self._wait_for_index_settled(keyspace, name, max_index_wait)
                self._report_index_size(keyspace, name, record_count)
            build_seconds = None  # nothing was built here, so there is no build time to report
        else:
            build_seconds = super()._build_index(record_count, max_index_wait, index_name=index_name, keyspace=keyspace)
            self._report_index_size(keyspace, index_name, record_count)
        # The index the query phases that follow will be priced against (see 'search_phase_end').
        # With variants it is not known until a round finds out which one serves.
        self._queried_index = None if self._live_variants else (keyspace, index_name)
        return build_seconds

    def _run_step_queries(self, step, dataset_name, defaults, record_count, step_number, local_ds_dir, remote_ds_dir):
        """Run the step's query sets once per index variant, or once as usual without variants."""
        if not self._live_variants:
            return super()._run_step_queries(
                step, dataset_name, defaults, record_count, step_number, local_ds_dir, remote_ds_dir
            )
        keyspace = self._keyspace()
        while self._live_variants:
            served = self._detect_served_index(keyspace, step, defaults, local_ds_dir, remote_ds_dir)
            self._variant_label = self._live_variants[served]
            self._queried_index = (keyspace, served)
            self.log.info(
                "Query round against index '%s' (variant '%s'); %d variant(s) to go after it",
                served,
                self._variant_label,
                len(self._live_variants) - 1,
            )
            try:
                super()._run_step_queries(
                    step, dataset_name, defaults, record_count, step_number, local_ds_dir, remote_ds_dir
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

        A few seconds of the first query set, plain, bracketed by two scrapes of every candidate's
        search total: the one that moved is the one serving. The choice is ScyllaDB's and stable
        for as long as the set of indexes does not change, which is why a round ends by dropping
        the index it measured.
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
            f"--concurrency=4 --retry-number 1 ",
            files_to_stage=[(os.path.join(local_ds_dir, queries_file), os.path.join(remote_ds_dir, queries_file))],
        )
        after = self._scrape_search_totals(keyspace, list(self._live_variants))
        moved = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in self._live_variants}
        served = max(moved, key=moved.get)
        if moved[served] <= 0:
            raise RuntimeError(f"The probe reached none of the candidate indexes {sorted(self._live_variants)}: {moved}")
        return served

    def _scrape_search_totals(self, keyspace, index_names) -> dict[str, float]:
        """'substring_search_total' per index, for the ones that export it."""
        metrics_text = self._vector_store_api_client().request("GET", "/metrics").text
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
        return super().search_extra_columns() + WALK_COLUMNS

    def search_phase_begin(self):
        return self._scrape_walk_totals()

    def search_phase_end(self, phase) -> dict:
        """What one query of the phase cost the index, from the walk totals before and after it.

        The totals count every search the index served, so they also see whatever else queried it
        meanwhile -- nothing does in this flow. A scrape that failed on either side leaves the cells
        empty rather than reporting a delta against nothing.
        """
        after = self._scrape_walk_totals()
        if phase is None or after is None:
            return {}
        cells = walk_per_query(phase, after)
        if cells:
            self.log.info(
                "Index work per query: %s",
                ", ".join(f"{column}={value}" for column, value in cells.items()),
            )
        else:
            self.log.warning("The index served no searches during the phase; no per-query cost to report")
        return cells

    def _scrape_walk_totals(self) -> dict[str, float] | None:
        """The index's search totals now, or None if there is no index or the scrape failed."""
        queried = getattr(self, "_queried_index", None)
        if queried is None:
            return None
        keyspace, index_name = queried
        try:
            metrics_text = self._vector_store_api_client().request("GET", "/metrics").text
        except Exception as exc:  # noqa: BLE001 - a scrape failure must not end the run
            self.log.warning("Could not scrape vector-store metrics for '%s': %s", index_name, exc)
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
                self.log.info("  segment %3d: %9d rows, sort %s..%s (%.1f%%)", ordinal, int(docs), f"{low:.0f}", f"{high:.0f}", span)
        if len(segments) > SEGMENT_LAYOUT_LOG_ROWS:
            self.log.info("  ... and %d more", len(segments) - SEGMENT_LAYOUT_LOG_ROWS)

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
            status = client.get_index_status_or_none(ks, idx)
            count = (status or {}).get("count", 0)
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

    def _wait_for_index_settled(self, keyspace, index_name, timeout):
        """Wait until the segment count stops moving, or give up and say so.

        An index ingested incrementally is many small segments that Tantivy goes on merging after
        the writes stop, and a search visits every one of them, so querying the moment the last row
        lands measures a transient rather than the index. Not reaching a stable count is reported
        rather than failed: the queries are still worth running, they just start against a moving
        target, and the segment count in the size row says what they ran against.
        """
        deadline = time.monotonic() + min(timeout, DEFAULT_SETTLE_TIMEOUT_SECS)
        previous, stable = None, 0
        while True:
            segments = self._read_index_gauge(keyspace, index_name, SEGMENT_COUNT_METRIC)
            if segments is None:
                self.log.warning(
                    "No '%s' sample for '%s'; not waiting for it to settle", SEGMENT_COUNT_METRIC, index_name
                )
                return
            stable = stable + 1 if segments == previous else 0
            previous = segments
            if stable >= SETTLE_STABLE_POLLS:
                self.log.info("Index '%s' settled at %d segments", index_name, int(segments))
                return
            if time.monotonic() >= deadline:
                self.log.warning("Index '%s' still merging at %d segments; querying anyway", index_name, int(segments))
                return
            time.sleep(SETTLE_POLL_INTERVAL_SECS)

    def _read_index_gauge(self, keyspace, index_name, metric) -> float | None:
        """One vector-store gauge for one index, or None if it cannot be read."""
        try:
            metrics_text = self._vector_store_api_client().request("GET", "/metrics").text
        except Exception as exc:  # noqa: BLE001 - a scrape failure must not end the run
            self.log.warning("Could not scrape vector-store metrics for '%s': %s", index_name, exc)
            return None
        return parse_index_gauge(metrics_text, metric, keyspace, index_name)

    def _report_index_size(self, keyspace, index_name, record_count):
        """Read the index size gauges off the vector-store node and submit them as one row.

        A missing gauge is reported as a missing measurement rather than a failure, exactly as a
        missing build time is: the index answers queries either way, and the query phase that
        follows is the more important half of the run.
        """
        try:
            metrics_text = self._vector_store_api_client().request("GET", "/metrics").text
        except Exception as exc:  # noqa: BLE001 - a scrape failure must not end the run
            self.log.warning("Could not scrape vector-store metrics for '%s': %s", index_name, exc)
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
                column="segment_span_mean_pct", row=row_key, value=round(sum(spans_pct) / len(spans_pct), 2), status=Status.UNSET
            )
            result_table.add_result(column="segment_span_max_pct", row=row_key, value=max(spans_pct), status=Status.UNSET)
        submit_results_to_argus(self.test_config.argus_client(), result_table)

    def test_substring_search(self):
        self.run_search_benchmark()
