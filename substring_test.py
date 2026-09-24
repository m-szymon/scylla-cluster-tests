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

import re
import time

from argus.client.generic_result import ColumnMetadata, ResultType, StaticGenericResultTable, Status

from sdcm.argus_results import submit_results_to_argus
from search_perf_test import (
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
_METRIC_LINE_RE_TEMPLATE = r"^{metric}\{{(?P<labels>[^}}]*)\}}\s+(?P<value>[0-9.eE+-]+)\s*$"
_LABEL_RE = re.compile(r'(?P<name>\w+)="(?P<value>[^"]*)"')

# The '-P' name that makes substring.rn create the index in its 'schema' phase. Not part of
# LatteScriptParams, which describes only what the shared flow drives.
WITH_INDEX_PARAM = "with_index"

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
            return f" ordered from {window:g}" if ordered else f" window from {window:g}"
        return " ordered" if ordered else ""

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
            self._create_index_for_ingestion(self._ingesting_index_name)
        return super()._load_step_shards(step, bucket, prefix, local_ds_dir, remote_ds_dir, max_load_wait)

    def _create_index_for_ingestion(self, index_name):
        """Create the index up front, so that rows are indexed as they are written.

        Only the index existing in the schema matters, not which statement created it: the index
        node picks it up and ingests the base table's CDC log either way.
        """
        self.log.info("Creating index '%s' before the load, to index while loading", index_name)
        self._run_latte(
            f"latte schema {self.WORKLOAD.script} "
            f"-P {WITH_INDEX_PARAM}=true "
            f'-P {self.WORKLOAD.params.index_name}=\\"{index_name}\\" ',
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
            self._wait_for_index_count(keyspace, index_name, record_count, max_index_wait)
            self._wait_for_index_settled(keyspace, index_name, max_index_wait)
            build_seconds = None  # nothing was built here, so there is no build time to report
        else:
            build_seconds = super()._build_index(record_count, max_index_wait, index_name=index_name, keyspace=keyspace)
        self._report_index_size(keyspace, index_name, record_count)
        return build_seconds

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
        submit_results_to_argus(self.test_config.argus_client(), result_table)

    def test_substring_search(self):
        self.run_search_benchmark()
