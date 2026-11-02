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

"""Unit tests for the substring half of the search performance flow.

The flow itself is covered by test_search_perf_test.py and the descriptor checks mirror
test_fts_test.py: what is left to check is that the descriptor still matches substring.rn, and that
the index-size gauge parser reads what vector-store actually exports. The parser is the one piece of
logic this module owns rather than describes, and it fails quietly -- a label that no longer matches
reads as "no measurement" rather than as an error.
"""

import dataclasses
import logging
import os
import re
import subprocess
import time

import pytest
import requests
import yaml

# NOTE: the test class is reached through the module rather than imported by name -- pytest's
#       unittest collector picks up any 'unittest.TestCase' subclass in a test module's namespace,
#       and 'SubstringSearchTest' is one through ClusterTester.
import substring_test
from substring_test import (
    CPU_COLUMNS,
    DB_CPU_COLUMN,
    DB_HOST_CPU_COLUMN,
    DB_MAX_SHARD_COLUMN,
    DB_SOFTIRQ_COLUMN,
    DB_STEAL_COLUMN,
    INDEX_SIZE_METRIC,
    LOADER_CPU_COLUMN,
    VS_CPU_COLUMN,
    SEARCHES_COLUMN,
    SEARCHES_METRIC,
    SEGMENT_COUNT_METRIC,
    SUBSTRING_BUILD_COUNT_COLUMN,
    SUBSTRING_WORKLOAD,
    WALK_COLUMNS,
    WALK_PER_QUERY_METRICS,
    WITH_INDEX_PARAM,
    SubstringIndexBuildResult,
    checked_index_variants,
    cpu_busy_pct,
    index_option_params,
    parse_index_gauge,
    parse_proc_stat,
    parse_reactor_busy,
    parse_segment_layout,
    parse_thread_states,
    segment_spans_pct,
    walk_per_query,
)
from sdcm import sct_abs_path

RUNE_PARAM_RE = re.compile(r"""latte::param!\(\s*["'](?P<name>[^"']+)["']""")
RUNE_FUNCTION_RE = re.compile(r"^pub async fn (?P<name>\w+)", re.MULTILINE)

# An excerpt of what vector-store's /metrics answers with, one index per keyspace/index_name pair.
# The index name is folded, as Scylla folds unquoted identifiers, while the test creates the index
# as 'sub_idx_names_10M_0' -- the point of the case-folding in the parser.
METRICS_SAMPLE = """\
# HELP pattern_index_size_bytes Total size of a substring search index (bytes)
# TYPE pattern_index_size_bytes gauge
pattern_index_size_bytes{keyspace="substring_bench",index_name="sub_idx_names_10m_0"} 3221225472
pattern_index_size_bytes{keyspace="substring_bench",index_name="sub_idx_names_10m_1"} 32212254720
# HELP pattern_segment_count Number of segments in a substring search index
# TYPE pattern_segment_count gauge
pattern_segment_count{keyspace="substring_bench",index_name="sub_idx_names_10m_0"} 8
# HELP fts_index_size_bytes Total size of a full-text search index (bytes)
# TYPE fts_index_size_bytes gauge
fts_index_size_bytes{keyspace="substring_bench",index_name="sub_idx_names_10m_0"} 999
"""


@pytest.fixture(scope="module")
def script_source():
    with open(sct_abs_path(SUBSTRING_WORKLOAD.script), encoding="utf-8") as script:
        return script.read()


def test_every_mapped_parameter_is_a_parameter_of_the_script(script_source):
    """The descriptor's whole point is mapping onto substring.rn's own names, so check it does."""
    declared = set(RUNE_PARAM_RE.findall(script_source))
    mapped = {getattr(SUBSTRING_WORKLOAD.params, field.name) for field in dataclasses.fields(SUBSTRING_WORKLOAD.params)}
    assert mapped <= declared, f"not parameters of {SUBSTRING_WORKLOAD.script}: {sorted(mapped - declared)}"


def test_the_index_option_parameters_exist(script_source):
    """The test case passes these through 'latte_schema_parameters'; a rename there makes the run
    silently build an index with the server defaults instead of the ones it reports."""
    declared = set(RUNE_PARAM_RE.findall(script_source))
    assert {"min_gram", "max_gram", "case_sensitive"} <= declared


def test_the_with_index_parameter_exists(script_source):
    """'index_during_load' creates the index through this parameter before the rows are written. If
    it is renamed, the schema call quietly creates no index, the load writes into an unindexed
    table, and the run then waits for a count that never arrives."""
    assert WITH_INDEX_PARAM in set(RUNE_PARAM_RE.findall(script_source))


def test_the_schema_phase_can_create_the_index(script_source):
    """The same call has to actually act on it: 'schema' creates the index when it is set."""
    assert "if WITH_INDEX {" in script_source


def test_the_hdr_tag_names_a_function_of_the_script(script_source):
    """The tag is 'fn--<function>'; latte emits nothing under it if that function is not there,
    and the latency table would come out empty."""
    function = SUBSTRING_WORKLOAD.hdr_tag.removeprefix("fn--")
    assert function in set(RUNE_FUNCTION_RE.findall(script_source))


def test_the_phases_the_flow_invokes_exist(script_source):
    """'load', 'build_index' and 'search' are the contract between the flow and any rune script."""
    functions = set(RUNE_FUNCTION_RE.findall(script_source))
    assert {"load", "build_index", "search"} <= functions


def test_the_query_is_a_containment_like(script_source):
    """The whole feature under test is that a '%keyword%' LIKE is answered from the index. A script
    that lost the LIKE, or grew an ALLOW FILTERING, would still run and still report latency -- of a
    table scan."""
    assert "LIKE :pattern" in script_source
    assert "ALLOW FILTERING" not in script_source


def test_the_step_record_file_key_shares_the_scripts_vocabulary():
    """A plan says 'names_file:' because substring.rn calls it that; the two must not drift apart."""
    assert SUBSTRING_WORKLOAD.step_records_file_key == SUBSTRING_WORKLOAD.params.records_file


def test_index_build_table_counts_names():
    """The count column is named once and reused, since Argus keys the table's history by it."""
    columns = SubstringIndexBuildResult.Meta.Columns
    assert SUBSTRING_BUILD_COUNT_COLUMN in {column.name for column in columns}
    assert SUBSTRING_WORKLOAD.build_count_column == SUBSTRING_BUILD_COUNT_COLUMN


def test_result_names_are_the_ones_the_history_is_under():
    """Renaming any of these silently starts a new, empty history."""
    assert SubstringIndexBuildResult.Meta.name == "Substring Index Build Time"
    assert SUBSTRING_WORKLOAD.name == "substring_search"  # cycle names: 'substring_search_p99_100ms'
    assert SUBSTRING_WORKLOAD.item_noun == "names"  # row labels: 'ds | 1,000,000 names | char2'
    assert SUBSTRING_WORKLOAD.index_prefix == "sub_idx"


def test_the_test_case_entry_point_is_the_one_the_docs_call():
    """test-cases/substring-search/*.yaml and docs/substring-search-test.md name this sub_test."""
    assert callable(substring_test.SubstringSearchTest.test_substring_search)
    assert substring_test.SubstringSearchTest.WORKLOAD is SUBSTRING_WORKLOAD


def test_gauge_is_read_for_the_named_index():
    assert parse_index_gauge(METRICS_SAMPLE, INDEX_SIZE_METRIC, "substring_bench", "sub_idx_names_10M_0") == 3221225472
    assert parse_index_gauge(METRICS_SAMPLE, SEGMENT_COUNT_METRIC, "substring_bench", "sub_idx_names_10M_0") == 8


def test_gauge_does_not_leak_between_indexes_or_metrics():
    """Three ways to read the wrong number: the next step's index, another keyspace, and the
    full-text gauge, whose name ends in the same words and whose labels are identical."""
    assert parse_index_gauge(METRICS_SAMPLE, INDEX_SIZE_METRIC, "substring_bench", "sub_idx_names_10M_1") == 32212254720
    assert parse_index_gauge(METRICS_SAMPLE, INDEX_SIZE_METRIC, "other_ks", "sub_idx_names_10M_0") is None
    assert parse_index_gauge(METRICS_SAMPLE, "index_size_bytes", "substring_bench", "sub_idx_names_10M_0") is None


def test_a_missing_gauge_is_no_measurement_rather_than_an_error():
    """An index vector-store has not accounted for yet, and an older build without these gauges,
    both look like this -- and neither should end a run that is otherwise fine."""
    assert parse_index_gauge(METRICS_SAMPLE, INDEX_SIZE_METRIC, "substring_bench", "sub_idx_absent") is None
    assert parse_index_gauge("", INDEX_SIZE_METRIC, "substring_bench", "sub_idx_names_10M_0") is None


# --- what the index did -------------------------------------------------------------------------

LAYOUT_SAMPLE = """\
pattern_segment_docs{index_name="sub_idx_names_10m_0",keyspace="substring_bench",segment="0"} 4000000
pattern_segment_docs{index_name="sub_idx_names_10m_0",keyspace="substring_bench",segment="1"} 5000000
pattern_segment_docs{index_name="sub_idx_names_10m_0",keyspace="substring_bench",segment="2"} 1000000
pattern_segment_docs{index_name="sub_idx_names_10m_1",keyspace="substring_bench",segment="0"} 7
pattern_segment_sort_min{index_name="sub_idx_names_10m_0",keyspace="substring_bench",segment="0"} 0
pattern_segment_sort_max{index_name="sub_idx_names_10m_0",keyspace="substring_bench",segment="0"} 4000000
pattern_segment_sort_min{index_name="sub_idx_names_10m_0",keyspace="substring_bench",segment="1"} 3000000
pattern_segment_sort_max{index_name="sub_idx_names_10m_0",keyspace="substring_bench",segment="1"} 10000000
pattern_segment_sort_min{index_name="sub_idx_names_10m_0",keyspace="substring_bench",segment="2"} 9000000
pattern_segment_sort_max{index_name="sub_idx_names_10m_0",keyspace="substring_bench",segment="2"} 10000000
"""


def test_segment_layout_is_read_by_ordinal_for_the_named_index():
    layout = parse_segment_layout(LAYOUT_SAMPLE, "substring_bench", "sub_idx_names_10M_0")
    assert layout == [
        (0, 4000000, 0, 4000000),
        (1, 5000000, 3000000, 10000000),
        (2, 1000000, 9000000, 10000000),
    ]


def test_an_unordered_index_has_segments_without_bounds():
    assert parse_segment_layout(LAYOUT_SAMPLE, "substring_bench", "sub_idx_names_10M_1") == [(0, 7, None, None)]
    assert parse_segment_layout("", "substring_bench", "sub_idx_names_10M_0") == []


def test_segment_spans_are_a_share_of_the_whole_range():
    layout = parse_segment_layout(LAYOUT_SAMPLE, "substring_bench", "sub_idx_names_10M_0")
    assert segment_spans_pct(layout) == [40.0, 70.0, 10.0]


def test_a_layout_with_nothing_to_prune_by_spans_nothing():
    """Unordered, or every value the same: either way the pruning has nothing to work with."""
    assert segment_spans_pct([(0, 7, None, None), (1, 3, None, None)]) == [0.0, 0.0]
    assert segment_spans_pct([(0, 7, 5, 5), (1, 3, 5, 5)]) == [0.0, 0.0]
    assert segment_spans_pct([]) == []


def _totals(searches, **per_metric):
    totals = {SEARCHES_METRIC: searches}
    for column, (metric, _) in WALK_PER_QUERY_METRICS.items():
        if column in per_metric:
            totals[metric] = per_metric[column]
    return totals


def test_the_walk_is_priced_per_query_over_the_phase():
    before = _totals(100, walk_us_per_query=1.0, segments_opened_per_query=10, postings_per_query=1000)
    after = _totals(300, walk_us_per_query=1.1, segments_opened_per_query=410, postings_per_query=201000)
    cells = walk_per_query(before, after)
    assert cells[SEARCHES_COLUMN] == 200
    assert cells["walk_us_per_query"] == pytest.approx(500.0)
    assert cells["segments_opened_per_query"] == 2.0
    assert cells["postings_per_query"] == 1000.0
    # A total the scrape did not have is a cell left empty, not a zero.
    assert "store_reads_per_query" not in cells


def test_a_phase_that_reached_no_search_is_not_priced():
    assert walk_per_query(_totals(100), _totals(100)) == {}
    assert walk_per_query({}, {}) == {}


def test_every_priced_quantity_is_a_column():
    names = {column.name for column in WALK_COLUMNS}
    assert names == {SEARCHES_COLUMN, *WALK_PER_QUERY_METRICS}


REACTOR_SAMPLE = """\
# HELP scylla_reactor_cpu_busy_ms Total cpu busy time in milliseconds
# TYPE scylla_reactor_cpu_busy_ms counter
scylla_reactor_cpu_busy_ms{shard="0"} 1500
scylla_reactor_cpu_busy_ms{shard="1"} 2500
scylla_reactor_cpu_steal_time_ms{shard="0"} 7
"""


def test_reactor_busy_is_read_by_shard():
    assert parse_reactor_busy(REACTOR_SAMPLE) == {"0": 1500.0, "1": 2500.0}
    assert parse_reactor_busy("") is None


def test_proc_stat_splits_busy_steal_and_softirq():
    text = "cpu  100 10 50 800 40 5 20 30 0 0\ncpu0 1 2 3 4 5 6 7 8 0 0\n"
    assert parse_proc_stat(text) == {"total": 1055.0, "busy": 215.0, "steal": 30.0, "softirq": 20.0}
    assert parse_proc_stat("intr 1 2 3") is None


def _stat(total, busy, steal=0.0, softirq=0.0):
    return {"total": total, "busy": busy, "steal": steal, "softirq": softirq}


def test_cpu_is_priced_over_the_phase():
    before = {
        "time": 10.0,
        "db": {"0": 4000.0, "1": 2000.0},
        "db_host": _stat(1000.0, 100.0, 10.0, 5.0),
        "loader": _stat(1000.0, 160.0),
        "vs": _stat(100.0, 0.0),
    }
    after = {
        "time": 20.0,
        "db": {"0": 14000.0, "1": 7000.0},
        "db_host": _stat(2000.0, 700.0, 110.0, 55.0),
        "loader": _stat(2000.0, 460.0),
    }
    cells = cpu_busy_pct(before, after)
    # Shard 0 busy all 10 s, shard 1 half of them; the host 600 of 1000 jiffies, 100 stolen, 50 in
    # softirq; the loader 300 of 1000; no vs reading after.
    assert cells == {
        DB_CPU_COLUMN: 75.0,
        DB_MAX_SHARD_COLUMN: 100.0,
        DB_HOST_CPU_COLUMN: 60.0,
        DB_STEAL_COLUMN: 10.0,
        DB_SOFTIRQ_COLUMN: 5.0,
        LOADER_CPU_COLUMN: 30.0,
    }
    assert cpu_busy_pct({}, after) == {}
    # A changed set of shards (a restart in between) is not a phase's CPU.
    assert DB_CPU_COLUMN not in cpu_busy_pct(before, after | {"db": {"0": 1.0}})


def test_thread_states_are_read_from_the_sampler():
    text = "tokio-rt-worker R 3.61\ntokio-rt-worker S 1.39\nsome noise\n"
    assert parse_thread_states(text) == {"tokio-rt-worker R": 3.61, "tokio-rt-worker S": 1.39}


def _run_sampler(tmp_path, delay: int, window: int, stop_after: float) -> str:
    """The sampler's output for this test's own process, the stop file touched after 'stop_after'."""
    script = tmp_path / "vs_threads.sh"
    script.write_text(substring_test.VS_THREAD_SAMPLER)
    out = tmp_path / "out"
    with open(f"/proc/{os.getpid()}/comm") as comm:
        name = comm.read().strip()
    sampler = subprocess.Popen(["sh", str(script), name, str(delay), str(window), str(out)])
    time.sleep(stop_after)
    (tmp_path / "out.stop").touch()
    sampler.wait(timeout=10)
    return out.read_text()


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="the sampler reads /proc")
def test_the_thread_sampler_does_not_wait_out_its_delay_at_the_end_of_a_phase(tmp_path):
    # A phase shorter than the delay: the end of the phase stops the sampler with nothing sampled.
    started = time.monotonic()
    text = _run_sampler(tmp_path, delay=600, window=20, stop_after=0.5)
    assert time.monotonic() - started < 5
    assert text.startswith("no samples")
    assert parse_thread_states(text) == {}


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="the sampler reads /proc")
def test_the_thread_sampler_reports_what_it_sampled_when_stopped_in_the_window(tmp_path):
    started = time.monotonic()
    text = _run_sampler(tmp_path, delay=0, window=600, stop_after=1.0)
    assert time.monotonic() - started < 5
    assert parse_thread_states(text)


def test_every_cpu_quantity_is_a_column():
    assert {column.name for column in CPU_COLUMNS} == {
        DB_CPU_COLUMN,
        DB_MAX_SHARD_COLUMN,
        DB_HOST_CPU_COLUMN,
        DB_STEAL_COLUMN,
        DB_SOFTIRQ_COLUMN,
        LOADER_CPU_COLUMN,
        VS_CPU_COLUMN,
    }


# --- index variants ------------------------------------------------------------------------------


def test_variants_are_validated_and_normalised():
    dataset = {
        "name": "d",
        "index_during_load": True,
        "index_variants": [{"label": "uncapped"}, {"label": "capped", "options": {"segment_rows": "100000"}}],
    }
    assert checked_index_variants(dataset) == [
        {"label": "uncapped", "options": {}, "order": ""},
        {"label": "capped", "options": {"segment_rows": "100000"}, "order": ""},
    ]
    assert checked_index_variants({"name": "d"}) == []
    # Without 'index_during_load' the variants are built after the load, by a full scan each.
    assert checked_index_variants({"name": "d", "index_variants": [{"label": "wide"}]}) == [
        {"label": "wide", "options": {}, "order": ""}
    ]


@pytest.mark.parametrize(
    "dataset,error",
    [
        ({"name": "d", "index_during_load": True, "index_variants": [{"options": {}}]}, "needs a 'label'"),
        ({"name": "d", "index_during_load": True, "index_variants": [{"label": "a"}, {"label": "a"}]}, "used twice"),
        (
            {"name": "d", "index_during_load": True, "index_variants": [{"label": "a", "options": {"k": 1}}]},
            "string values",
        ),
        (
            {"name": "d", "index_during_load": True, "index_variants": [{"label": "a", "options": {"k": "it's"}}]},
            "only letters",
        ),
        (
            {
                "name": "d",
                "index_during_load": True,
                "index_variants": [{"label": "a", "options": {"k": "1", "j": "2", "i": "3", "h": "4"}}],
            },
            "up to 3 extra options",
        ),
    ],
)
def test_bad_variants_are_plan_errors(dataset, error):
    with pytest.raises(ValueError, match=error):
        checked_index_variants(dataset)


def test_the_option_params_are_the_ones_the_script_takes(script_source):
    assert (
        index_option_params({"rewrite": "true"})
        == '-P extra_option_name=\\"rewrite\\" -P extra_option_value=\\"true\\" '
    )
    assert index_option_params({}) == ""
    assert index_option_params({"verify_candidates": "scylla", "segment_rows": "100000", "rewrite": "false"}) == (
        '-P extra_option_name=\\"rewrite\\" -P extra_option_value=\\"false\\" '
        '-P extra_option2_name=\\"segment_rows\\" -P extra_option2_value=\\"100000\\" '
        '-P extra_option3_name=\\"verify_candidates\\" -P extra_option3_value=\\"scylla\\" '
    )
    for param in (
        "extra_option_name",
        "extra_option_value",
        "extra_option2_name",
        "extra_option2_value",
        "extra_option3_name",
        "extra_option3_value",
    ):
        assert f'param!("{param}"' in script_source


def test_the_variant_label_ends_the_row_label():
    test = object.__new__(substring_test.SubstringSearchTest)
    test._variant_label = "capped"
    assert test.query_shape_label({"set": "char2", "ordered": True}) == " ordered [capped]"
    assert test.query_shape_label({"set": "char2"}) == " [capped]"
    test._variant_label = ""
    assert test.query_shape_label({"set": "char2"}) == ""


# --- ordering ---------------------------------------------------------------------------------


def test_the_ordering_parameters_exist(script_source):
    """The plan's 'ordered' and 'window' keys become these. A rename makes latte ignore the flag
    and run the plain query, which still reports a latency -- of the wrong question."""
    declared = set(RUNE_PARAM_RE.findall(script_source))
    assert {
        "order_by",
        "search_ordered",
        "search_window_from",
        "sort_value_count",
        "search_direction",
        "search_match",
    } <= declared


def test_the_script_can_emit_an_order_by_clause(script_source):
    """Declaring the parameters is not enough; the query has to be built from them."""
    assert "ORDER BY ${column} DESC" in script_source
    assert "ORDER BY ${column} ASC" in script_source
    assert "'order_by': '${ORDER_BY}', " in script_source


def test_an_ordered_shape_without_an_ordered_index_is_refused(script_source):
    """The one failure here that yields a plausible number for the wrong thing: without a sort
    column the ordered clauses are dropped and the plain query is measured under the ordered row."""
    assert "search_ordered/search_window_from need an index created with order_by" in script_source


@pytest.mark.parametrize(
    "query, expected",
    [
        ({"set": "char2"}, ""),
        ({"set": "char2", "ordered": True}, " ordered"),
        ({"set": "char2", "ordered": True, "window": 0.5}, " ordered from 0.5"),
        ({"set": "char2", "window": 0.5}, " window from 0.5"),
        ({"set": "char2", "ordered": True, "order": "asc"}, " ordered asc"),
        ({"set": "char2", "ordered": True, "window": 0.5, "order": "asc"}, " ordered from 0.5 asc"),
        # A direction without an order to apply it to changes nothing, so it is not in the label.
        ({"set": "char2", "order": "asc"}, ""),
        ({"set": "char2", "match": "prefix"}, " prefix"),
        ({"set": "char2", "match": "suffix", "ordered": True}, " suffix ordered"),
    ],
)
def test_the_shape_is_part_of_the_row_label(query, expected):
    """Argus keys a cell by (row, column), so a set asked plain and ordered in one step must not
    share a label -- the two would push conflicting latencies into a single row. The shape belongs
    in the label rather than a column because it changes what the number means."""
    assert substring_test.SubstringSearchTest.query_shape_label(None, query) == expected


@pytest.mark.parametrize("window", [-0.1, 1.0, 1.5, "half", True])
def test_a_window_outside_the_order_is_a_plan_error(window):
    """A fraction at or above 1 bounds the query below every value in the corpus and would measure
    an empty answer at a plausible-looking latency."""
    with pytest.raises(ValueError):
        substring_test._checked_window(window)


def test_the_plans_only_ask_for_ordering_where_the_index_can_order():
    """A plan entry asking for 'ordered' against a test case with no 'order_by' would fail inside
    the loader, after the corpus is loaded and indexed. Checked here against the tracked pairs."""
    for case_name, plan_name in (
        ("substring-search-test-docker.yaml", "local_config.yaml"),
        ("substring-search-test.yaml", "aws_config.yaml"),
    ):
        with open(sct_abs_path(f"test-cases/substring-search/{case_name}"), encoding="utf-8") as f:
            case = yaml.safe_load(f)
        with open(sct_abs_path(f"{substring_test.SUBSTRING_BASE_DIR}/{plan_name}"), encoding="utf-8") as f:
            plan = yaml.safe_load(f)
        order_by = (case.get("latte_schema_parameters") or {}).get("order_by") or ""
        asks_for_ordering = any(
            query.get("ordered") or query.get("window")
            for dataset in plan["datasets"]
            for step in dataset.get("steps", [])
            for query in step.get("queries", [])
        )
        assert not asks_for_ordering or order_by, f"{plan_name} asks for ordering, {case_name} sets no order_by"


@pytest.mark.parametrize(
    "query, expected",
    [
        ({"set": "char2"}, ""),
        ({"set": "char2", "match": "prefix"}, '-P search_match=\\"prefix\\" '),
        ({"set": "char2", "ordered": True}, "-P search_ordered=true "),
        ({"set": "char2", "ordered": True, "order": "asc"}, '-P search_ordered=true -P search_direction=\\"asc\\" '),
        (
            {"set": "char2", "match": "suffix", "ordered": True, "window": 0.5},
            '-P search_match=\\"suffix\\" -P search_ordered=true -P search_window_from=0.5 -P sort_value_count=3000 ',
        ),
        # A window comes with its ORDER BY: an ordered index serves no other query.
        ({"set": "char2", "window": 0.5}, "-P search_ordered=true -P search_window_from=0.5 -P sort_value_count=3000 "),
    ],
)
def test_the_direction_and_the_match_become_script_parameters(query, expected):
    """'order: asc' and 'match: prefix|suffix' reach latte as parameters; the defaults are not
    spelled out, so a script that predates them runs unchanged."""
    test = object.__new__(substring_test.SubstringSearchTest)
    test.params = {"latte_schema_parameters": {"order_by": "register_time"}}
    assert test.extra_search_params(query, 3000) == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ({"set": "char2"}, ""),
        ({"set": "char2", "match": "prefix"}, ""),
        ({"set": "char2", "ordered": True}, "-P search_ordered=true "),
        ({"set": "char2", "window": 0.5, "order": "asc"}, '-P search_ordered=true -P search_direction=\\"asc\\" '),
    ],
)
def test_the_served_index_probe_asks_in_the_queries_shape(query, expected):
    """The probe that finds the serving index orders as the step's queries do and nothing more:
    the ORDER BY is what picks among the pattern indexes on a column."""
    assert substring_test.SubstringSearchTest._probe_shape(None, query) == expected


@pytest.mark.parametrize("query", [{"set": "char2", "order": "up"}, {"set": "char2", "match": "infix"}])
def test_an_unknown_direction_or_match_is_refused(query):
    with pytest.raises(ValueError):
        substring_test.SubstringSearchTest.query_shape_label(None, query)


class _FlakyStatusClient:
    """Answers each status request from a script: an exception to raise or a count to report."""

    def __init__(self, answers):
        self.answers = list(answers)

    def get_index_status_or_none(self, keyspace, index):
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return {"count": answer}


def _test_waiting_on(client, monkeypatch):
    test = object.__new__(substring_test.SubstringSearchTest)
    test.log = logging.getLogger("test_substring_test")
    test.WORKLOAD = SUBSTRING_WORKLOAD
    test._vector_store_api_client = lambda: client
    monkeypatch.setattr(substring_test.time, "sleep", lambda _secs: None)
    return test


@pytest.mark.parametrize(
    "failure", [requests.exceptions.ReadTimeout("read timed out"), requests.exceptions.ConnectionError("reset")]
)
def test_a_status_poll_that_gets_no_answer_is_skipped_not_fatal(failure, monkeypatch):
    """One dropped status request is a missed sample: the wait polls on and sees the index catch up."""
    client = _FlakyStatusClient([5, failure, 10])
    test = _test_waiting_on(client, monkeypatch)
    caught_up = test._wait_for_indexes_count("ks", ["idx"], 10, timeout=60)
    assert list(caught_up) == ["idx"]
    assert client.answers == []


def test_the_single_index_wait_also_polls_through_a_dropped_request(monkeypatch):
    client = _FlakyStatusClient([requests.exceptions.ReadTimeout("read timed out"), 10])
    test = _test_waiting_on(client, monkeypatch)
    test._wait_for_index_count("ks", "idx", 10, timeout=60)
    assert client.answers == []


def test_an_index_that_never_answers_still_times_out(monkeypatch):
    """Skipping unanswered polls does not stretch the wait past its deadline."""
    client = _FlakyStatusClient([requests.exceptions.ReadTimeout("read timed out")] * 1000)
    test = _test_waiting_on(client, monkeypatch)
    clock = iter(range(0, 10000, 10))
    monkeypatch.setattr(substring_test.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        test._wait_for_indexes_count("ks", ["idx"], 10, timeout=60)


def _sleeps_move_the_clock(monkeypatch):
    """A clock that only the waits' sleeps advance, so that a poll count is a span of time."""
    now = [0.0]
    monkeypatch.setattr(substring_test.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(substring_test.time, "sleep", lambda secs: now.__setitem__(0, now[0] + secs))


def test_an_index_node_silent_for_minutes_ends_the_wait_long_before_its_deadline(monkeypatch):
    """Run fd5928db polled a node that had run out of memory for 47 minutes; now it gives up."""
    client = _FlakyStatusClient([requests.exceptions.ReadTimeout("read timed out")] * 1000)
    test = _test_waiting_on(client, monkeypatch)
    _sleeps_move_the_clock(monkeypatch)
    with pytest.raises(substring_test.VectorStoreUnresponsive):
        test._wait_for_indexes_count("ks", ["idx"], 10, timeout=7200)
    polls_to_give_up = 1000 - len(client.answers)
    assert (
        polls_to_give_up
        <= substring_test.VS_UNRESPONSIVE_ABORT_SECS / substring_test.INDEX_COUNT_POLL_INTERVAL_SECS + 2
    )


def test_an_answer_restarts_the_count_of_silence(monkeypatch):
    """Silences shorter than VS_UNRESPONSIVE_ABORT_SECS, one after another, are missed samples."""
    timeout = requests.exceptions.ReadTimeout("read timed out")
    gaps = int(substring_test.VS_UNRESPONSIVE_ABORT_SECS / substring_test.INDEX_COUNT_POLL_INTERVAL_SECS) - 2
    client = _FlakyStatusClient([timeout] * gaps + [5] + [timeout] * gaps + [10])
    test = _test_waiting_on(client, monkeypatch)
    _sleeps_move_the_clock(monkeypatch)
    test._wait_for_indexes_count("ks", ["idx"], 10, timeout=7200)
    assert client.answers == []


class _MetricsClient:
    """A vector-store client whose /metrics answers come from a list: text, or an exception to raise."""

    def __init__(self, answers):
        self.answers = list(answers)

    def request(self, method, path):
        assert (method, path) == ("GET", "/metrics")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return dataclasses.make_dataclass("Response", ["text"])(answer)


def test_a_missed_scrape_is_no_reading_but_a_silent_node_ends_the_run(monkeypatch):
    client = _MetricsClient([requests.exceptions.ReadTimeout("read timed out")] * 100)
    test = _test_waiting_on(client, monkeypatch)
    clock = iter(range(0, 100000, 30))
    monkeypatch.setattr(substring_test.time, "monotonic", lambda: next(clock))
    assert test._try_vs_metrics("idx") is None
    with pytest.raises(substring_test.VectorStoreUnresponsive):
        for _ in range(100):
            test._try_vs_metrics("idx")


def test_the_cdc_wait_gives_up_on_a_silent_node_rather_than_waiting_out_the_settle_timeout(monkeypatch):
    client = _MetricsClient([requests.exceptions.ConnectionError("refused")] * 1000)
    test = _test_waiting_on(client, monkeypatch)
    test._max_settle = 1800
    clock = iter(range(0, 100000, 10))
    monkeypatch.setattr(substring_test.time, "monotonic", lambda: next(clock))
    with pytest.raises(substring_test.VectorStoreUnresponsive):
        test._wait_for_cdc_quiet("ks", ["idx"])


class _SlowRemoter:
    def __init__(self, secs):
        self.secs = secs

    def run(self, cmd, **_):
        time.sleep(self.secs)
        return cmd

    def sudo(self, cmd, **_):
        raise ValueError(cmd)


def test_a_diagnostic_ssh_call_that_hangs_is_abandoned_at_its_bound(monkeypatch):
    """Run a0e51aa9 sat six minutes in one call to an index node out of memory."""
    monkeypatch.setattr(substring_test, "DIAGNOSTIC_GRACE_SECS", 0)
    remoter = substring_test._BoundedRemoter(_SlowRemoter(5), "vs-1")
    started = time.monotonic()
    with pytest.raises(substring_test.DiagnosticSkipped):
        remoter.run("head -1 /proc/stat", timeout=0.2)
    assert time.monotonic() - started < 2


def test_a_bounded_diagnostic_call_returns_or_raises_as_the_call_does():
    remoter = substring_test._BoundedRemoter(_SlowRemoter(0), "db-1")
    assert remoter.run("uptime", timeout=5) == "uptime"
    with pytest.raises(ValueError):
        remoter.sudo("false", timeout=5)


def test_a_silent_index_node_is_not_reached_over_ssh_either():
    test = object.__new__(substring_test.SubstringSearchTest)
    vs = dataclasses.make_dataclass("Node", [("remoter", object), ("name", str)])(_SlowRemoter(0), "vs-1")
    db = dataclasses.make_dataclass("Node", [("remoter", object), ("name", str)])(_SlowRemoter(0), "db-1")
    test._vector_store_node = lambda: vs
    assert test._diag(vs).run("uptime") == "uptime"
    test._vs_silent_since = 0.0
    with pytest.raises(substring_test.DiagnosticSkipped):
        test._diag(vs)
    assert test._diag(db).run("uptime") == "uptime"


def test_the_memory_log_script_reads_meminfo_and_the_process(tmp_path):
    """The script's one iteration, with logger and sleep swapped for an echo and an exit."""
    script = substring_test.VS_MEMORY_LOG_SCRIPT.replace("logger -t vs-memory", "echo").replace("sleep 60", "break")
    path = tmp_path / "vs_memory.sh"
    path.write_text(script)
    out = subprocess.run(["sh", str(path)], capture_output=True, text=True, check=True).stdout
    assert re.fullmatch(r"total_mb=\d+ available_mb=\d+ vector_store_rss_mb=\d+\n", out), out


class _FakeRemoter:
    def __init__(self, has_apt, perf_runs=True):
        self.has_apt = has_apt
        self.perf_runs = perf_runs
        self.sudo_cmds = []

    def run(self, cmd, **_):
        if cmd == "command -v apt-get":
            return dataclasses.make_dataclass("R", [("ok", bool)])(self.has_apt)
        return dataclasses.make_dataclass("R", [("stdout", str)])("6.8.0-1017-aws\n")

    def sudo(self, cmd, **kwargs):
        if cmd == "perf --version":
            return dataclasses.make_dataclass("R", [("ok", bool)])(self.perf_runs)
        self.sudo_cmds.append((cmd, kwargs))
        return dataclasses.make_dataclass("R", [("ok", bool)])(True)


def _installer():
    test = object.__new__(substring_test.SubstringSearchTest)
    test.log = logging.getLogger("test_substring_test")
    return test


def test_perf_is_installed_with_bounded_apt_calls():
    node = dataclasses.make_dataclass("Node", [("remoter", object), ("name", str)])(_FakeRemoter(has_apt=True), "n")
    _installer()._install_perf(node)
    update, install = node.remoter.sudo_cmds
    assert "Acquire::http::Timeout=60" in update[0] and update[1]["timeout"]
    assert install[0].startswith("DEBIAN_FRONTEND=noninteractive apt-get install")
    assert "DPkg::Lock::Timeout=120" in install[0]
    assert install[0].endswith("-y linux-tools-6.8.0-1017-aws linux-tools-common")
    assert install[1]["timeout"] and install[1]["ignore_status"]


def test_perf_binary_is_linked_when_the_wrapper_misses_the_kernel():
    node = dataclasses.make_dataclass("Node", [("remoter", object), ("name", str)])(
        _FakeRemoter(has_apt=True, perf_runs=False), "n"
    )
    _installer()._install_perf(node)
    _, _, link = node.remoter.sudo_cmds
    assert "/usr/lib/linux-tools/6.8.0-1017-aws/perf" in link[0]
    # Searched for, not guessed (run a0e51aa9's arm64 node had it at neither guessed path).
    assert "find /usr/lib" in link[0] and "-name perf" in link[0]
    assert "ln -sf" in link[0] and "/usr/local/bin/perf" in link[0]


def test_perf_is_not_installed_without_apt():
    node = dataclasses.make_dataclass("Node", [("remoter", object), ("name", str)])(_FakeRemoter(has_apt=False), "n")
    _installer()._install_perf(node)
    assert node.remoter.sudo_cmds == []


def test_the_profile_waits_on_a_busy_reading_of_the_node():
    """PROFILE_SCRIPT's 'busy' function, run once: a whole percentage of the CPUs busy over a second."""
    busy = substring_test.PROFILE_SCRIPT.split("\n}\n", 1)[0] + "\n}\nbusy\n"
    out = subprocess.run(["sh", "-c", busy], capture_output=True, text=True, check=True).stdout
    assert re.fullmatch(r"\d+\n", out) and 0 <= int(out) <= 100, out


def test_the_profile_records_only_after_its_waits():
    script = substring_test.PROFILE_SCRIPT
    assert script.index(f"sleep {substring_test.PROFILE_MIN_DELAY_SECS}") < script.index("perf record")
    assert f"-lt {substring_test.PROFILE_BUSY_PCT} ]" in script
    assert script.index(f"sleep {substring_test.PROFILE_SETTLE_SECS}") < script.index("perf record")


# --- writers, lag and the CPU window ---------------------------------------------------------------

LAG_METRICS = """\
indexing_lag_seconds_bucket{index_name="ix",keyspace="ks",le="0.05",reader="wide"} 7
indexing_lag_seconds_bucket{index_name="ix",keyspace="ks",le="0.05",reader="fine"} %d
indexing_lag_seconds_bucket{index_name="ix",keyspace="ks",le="0.5",reader="fine"} %d
indexing_lag_seconds_bucket{index_name="ix",keyspace="ks",le="5",reader="fine"} %d
indexing_lag_seconds_bucket{index_name="ix",keyspace="ks",le="300",reader="fine"} %d
indexing_lag_seconds_bucket{index_name="ix",keyspace="ks",le="+Inf",reader="fine"} %d
indexing_lag_seconds_sum{index_name="ix",keyspace="ks",reader="wide"} 9999.0
indexing_lag_seconds_sum{index_name="ix",keyspace="ks",reader="fine"} %f
indexing_lag_seconds_count{index_name="ix",keyspace="ks",reader="wide"} 999
indexing_lag_seconds_count{index_name="ix",keyspace="ks",reader="fine"} %d
indexing_lag_seconds_count{index_name="other",keyspace="ks",reader="fine"} 999
"""


def _lag(counts, total):
    return substring_test.parse_indexing_lag(LAG_METRICS % (*counts, total, counts[-1]), "ks", "ix")


def test_the_lag_of_a_phase_is_read_from_the_histogram_delta():
    before = _lag((10, 10, 10, 10, 10), 0.1)
    # 200 more changes: 100 under 50 ms, 98 more under 0.5 s, 2 more under 5 s.
    after = _lag((110, 208, 210, 210, 210), 30.1)
    assert substring_test.indexing_lag_cells(before, after) == {
        substring_test.INGESTED_ROWS_COLUMN: 200,
        substring_test.INDEX_LAG_MEAN_COLUMN: 0.15,
        substring_test.INDEX_LAG_P99_COLUMN: 0.5,
    }


def test_a_lag_past_the_last_bucket_reads_as_the_overflow_value():
    before = _lag((0, 0, 0, 0, 0), 0.0)
    after = _lag((0, 0, 0, 50, 100), 50000.0)
    cells = substring_test.indexing_lag_cells(before, after)
    assert cells[substring_test.INDEX_LAG_P99_COLUMN] == substring_test.INDEX_LAG_OVERFLOW_S


def test_the_lag_is_the_fine_readers_and_a_series_without_readers_still_reads():
    """The wide reader indexes every change again minutes later; its series is not the lag."""
    reading = _lag((1, 2, 3, 4, 5), 0.5)
    assert reading["count"] == 5 and reading["sum"] == 0.5 and reading["buckets"]["0.05"] == 1
    legacy = LAG_METRICS.replace(',reader="fine"', "") % (1, 2, 3, 4, 5, 0.5, 5)
    legacy = "\n".join(line for line in legacy.splitlines() if 'reader="wide"' not in line)
    assert substring_test.parse_indexing_lag(legacy, "ks", "ix")["count"] == 5


def test_a_phase_without_writes_has_no_lag_and_a_missing_scrape_no_cells():
    reading = _lag((10, 10, 10, 10, 10), 0.1)
    assert substring_test.indexing_lag_cells(reading, reading) == {substring_test.INGESTED_ROWS_COLUMN: 0}
    assert substring_test.indexing_lag_cells({}, reading) == {}
    assert substring_test.parse_indexing_lag("", "ks", "ix") == {}


def test_every_ingest_quantity_is_a_column():
    names = {column.name for column in substring_test.INGEST_COLUMNS}
    test = substring_test.SubstringSearchTest
    before = {
        "lag": _lag((0,) * 5, 0.0),
        "segments": 40.0,
        "l0": 0.0,
        "ranges_done": 3.0,
        "merges_started": 10.0,
        "merges_ended": 9.0,
    }
    after = {
        "lag": _lag((5, 5, 5, 5, 5), 0.1),
        "segments": 41.0,
        "l0": 1000.0,
        "ranges_done": 7.0,
        "merges_started": 25.0,
        "merges_ended": 21.0,
    }
    cells = test._ingest_cells(before, after)
    assert set(cells) | {substring_test.WRITE_OP_RATE_COLUMN} == names
    assert cells[substring_test.REWRITE_RANGES_COLUMN] == 4
    assert cells[substring_test.MERGES_STARTED_COLUMN] == 15
    assert cells[substring_test.MERGES_ENDED_COLUMN] == 12


def test_a_vector_store_without_merge_gauges_leaves_their_cells_empty():
    before = {"lag": _lag((0,) * 5, 0.0), "segments": 40.0, "l0": 0.0, "ranges_done": 3.0}
    cells = substring_test.SubstringSearchTest._ingest_cells(before, dict(before))
    assert substring_test.MERGES_STARTED_COLUMN not in cells


def _settling(monkeypatch, readings):
    """A test whose gauges read off 'readings', one dict per poll; sleeps advance a clock."""
    test = object.__new__(substring_test.SubstringSearchTest)
    test.log = logging.getLogger("test_substring_test")
    test._max_settle = 3600
    polls = iter(readings)
    current = {}

    def read(_keyspace, _index, metric):
        if metric == substring_test.SEGMENT_COUNT_METRIC:
            current.clear()
            current.update(next(polls))
        return current.get(metric)

    test._read_index_gauge = read
    now = [0.0]
    monkeypatch.setattr(substring_test.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(substring_test.time, "sleep", lambda secs: now.__setitem__(0, now[0] + secs))
    return test, now


def test_an_index_with_merges_running_is_not_settled_however_still_its_segment_count(monkeypatch):
    """Run a0e51aa9's index held 712 segments for three polls with thousands of merges queued."""
    segments, running = substring_test.SEGMENT_COUNT_METRIC, substring_test.MERGES_RUNNING_METRIC
    readings = [{segments: 712, running: 36}] * 6 + [{segments: 700, running: 0}] * 4
    test, now = _settling(monkeypatch, readings)
    test._wait_for_index_settled("ks", "idx", timeout=3600)
    # Settled on the polls after the merges stopped, not on the first three still ones.
    assert now[0] >= 8 * substring_test.SETTLE_POLL_INTERVAL_SECS


def test_a_cpu_snapshot_reads_the_clock_the_stat_line_and_the_reactors():
    text = (
        "1793357687.25\n"
        "cpu  100 0 50 800 50 0 10 5 0 0\n"
        'scylla_reactor_cpu_busy_ms{shard="0"} 1000\n'
        'scylla_reactor_cpu_busy_ms{shard="1"} 2000\n'
    )
    snapshot = substring_test.parse_cpu_snapshot(text)
    assert snapshot["time"] == 1793357687.25
    assert snapshot["stat"]["busy"] == 165.0
    assert snapshot["reactor"] == {"0": 1000.0, "1": 2000.0}
    assert substring_test.parse_cpu_snapshot("cpu  1 2 3 4\n") == {}
    assert substring_test.parse_cpu_snapshot("") == {}


def _run_cpu_window(tmp_path, delay: int, window: int, stop_after: float) -> tuple[str, str]:
    script = tmp_path / "cpu_window.sh"
    script.write_text(substring_test.CPU_WINDOW_SCRIPT)
    out = tmp_path / "w"
    window_proc = subprocess.Popen(["sh", str(script), str(delay), str(window), str(out)])
    time.sleep(stop_after)
    (tmp_path / "w.stop").touch()
    window_proc.wait(timeout=10)
    assert (tmp_path / "w.done").exists()
    return tuple((tmp_path / f"w.{part}").read_text() if (tmp_path / f"w.{part}").exists() else "" for part in "ab")


@pytest.mark.skipif(not os.path.isfile("/proc/stat"), reason="the window reads /proc/stat")
def test_the_cpu_window_reads_the_window_and_skips_a_phase_that_ends_first(tmp_path):
    first, second = _run_cpu_window(tmp_path, delay=0, window=600, stop_after=1.5)
    start, end = substring_test.parse_cpu_snapshot(first), substring_test.parse_cpu_snapshot(second)
    assert end["time"] > start["time"]
    assert end["stat"]["total"] > start["stat"]["total"]
    # A phase shorter than the delay: stopped before the window began, it read nothing, at once.
    short = tmp_path / "short"
    short.mkdir()
    started = time.monotonic()
    assert _run_cpu_window(short, delay=600, window=20, stop_after=0.5) == ("", "")
    assert time.monotonic() - started < 5


@pytest.mark.parametrize(
    "query, expected",
    [
        ({"set": "s"}, None),
        ({"set": "s", "write_rate": 2000}, (2000, "sequential")),
        ({"set": "s", "write_rate": 5000, "write_values": "random"}, (5000, "random")),
    ],
)
def test_a_writer_is_read_from_the_query_entry(query, expected):
    assert substring_test.checked_writer(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        {"set": "s", "write_values": "random"},
        {"set": "s", "write_rate": 0},
        {"set": "s", "write_rate": "2k"},
        {"set": "s", "write_rate": True},
        {"set": "s", "write_rate": 10, "write_values": "shuffled"},
    ],
)
def test_a_bad_writer_is_a_plan_error(query):
    with pytest.raises(ValueError):
        substring_test.checked_writer(query)


def test_a_variant_round_runs_the_queries_that_name_it_or_none():
    queries = [{"set": "a"}, {"set": "b", "variants": ["table"]}, {"set": "c", "variants": ["tier", "table"]}]
    assert [q["set"] for q in substring_test.queries_for_variant(queries, "table")] == ["a", "b", "c"]
    assert [q["set"] for q in substring_test.queries_for_variant(queries, "tier")] == ["a", "c"]
    assert [q["set"] for q in substring_test.queries_for_variant(queries, "time")] == ["a"]


def test_a_variant_can_say_what_its_index_orders_by():
    dataset = {"name": "d", "index_variants": [{"label": "t", "order": "table"}, {"label": "time"}]}
    variants = checked_index_variants(dataset)
    assert substring_test.variant_params(variants[0]) == '-P index_order=\\"table\\" '
    assert substring_test.variant_params(variants[1]) == ""
    with pytest.raises(ValueError):
        checked_index_variants({"name": "d", "index_variants": [{"label": "t", "order": "token"}]})
    with pytest.raises(ValueError):
        checked_index_variants(
            {"name": "d", "index_variants": [{"label": "t", "order": "table", "options": {"order_by": "x"}}]}
        )


def test_the_variants_a_query_names_must_be_the_datasets():
    test = object.__new__(substring_test.SubstringSearchTest)
    dataset = {
        "name": "d",
        "index_variants": [{"label": "table", "order": "table"}, {"label": "tier", "order": "tie"}],
        "steps": [{"queries": [{"set": "a", "variants": ["table"]}]}],
    }
    test.validate_dataset(dataset)
    dataset["steps"][0]["queries"][0]["variants"] = ["time"]
    with pytest.raises(ValueError):
        test.validate_dataset(dataset)
    with pytest.raises(ValueError):
        test.validate_dataset({"name": "d", "steps": [{"queries": [{"set": "a", "variants": ["table"]}]}]})


@pytest.mark.parametrize(
    "query, expected",
    [
        ({"set": "char2", "ordered": True, "order_column": "tier"}, " ordered by tier"),
        ({"set": "char2", "write_rate": 2000}, " +2000/s sequential writes"),
        (
            {"set": "char2", "ordered": True, "write_rate": 5000, "write_values": "random"},
            " ordered +5000/s random writes",
        ),
    ],
)
def test_the_order_column_and_the_writer_are_part_of_the_row_label(query, expected):
    assert substring_test.SubstringSearchTest.query_shape_label(None, query) == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        (
            {"set": "char2", "ordered": True, "order_column": "tier"},
            '-P search_order_column=\\"tie\\" -P search_ordered=true ',
        ),
        ({"set": "char2", "qrels": True, "check_order": True}, "-P check_order=true "),
        ({"set": "char2", "write_rate": 2000}, ""),
    ],
)
def test_the_order_column_and_the_order_check_become_script_parameters(query, expected):
    test = object.__new__(substring_test.SubstringSearchTest)
    test.params = {"latte_schema_parameters": {"order_by": "register_time", "tie_column": "tier"}}
    assert test.extra_search_params(query, 3000) == expected


@pytest.mark.parametrize(
    "query",
    [
        {"set": "char2", "ordered": True, "order_column": "rating"},
        {"set": "char2", "order_column": "tier"},
        {"set": "char2", "ordered": True, "window": 0.5, "order_column": "tier"},
        {"set": "char2", "check_order": True},
    ],
)
def test_an_order_column_or_order_check_the_table_cannot_honour_is_a_plan_error(query):
    test = object.__new__(substring_test.SubstringSearchTest)
    test.params = {"latte_schema_parameters": {"order_by": "register_time", "tie_column": "tier"}}
    with pytest.raises(ValueError):
        test.extra_search_params(query, 3000)


def test_the_probe_orders_by_the_tie_column_when_the_queries_do():
    query = {"set": "char2", "ordered": True, "order_column": "tier"}
    assert substring_test.SubstringSearchTest._probe_shape(None, query) == (
        '-P search_ordered=true -P search_order_column=\\"tie\\" '
    )


def test_a_tie_value_is_what_the_script_derives():
    """derive_order_qrels.py's ground truth for the tie order assumes the script's tie_of()."""
    with open(sct_abs_path(f"{substring_test.SUBSTRING_BASE_DIR}/substring.rn"), encoding="utf-8") as f:
        source = f.read()
    assert "((sort % 1000003) * 7919 + 13) % 1000003 % TIE_VALUES" in source


def test_the_changes_an_index_took_in_count_every_reader():
    text = LAG_METRICS % (1, 2, 3, 4, 5, 0.5, 5)
    assert substring_test.cdc_changes_total(text, "ks", "ix") == 5 + 999
    assert substring_test.cdc_changes_total(text, "ks", "nothing") is None
