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
import re

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
        {"label": "uncapped", "options": {}},
        {"label": "capped", "options": {"segment_rows": "100000"}},
    ]
    assert checked_index_variants({"name": "d"}) == []
    # Without 'index_during_load' the variants are built after the load, by a full scan each.
    assert checked_index_variants({"name": "d", "index_variants": [{"label": "wide"}]}) == [
        {"label": "wide", "options": {}}
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
    assert "ORDER BY ${ORDER_BY} DESC" in script_source
    assert "ORDER BY ${ORDER_BY} ASC" in script_source
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
    ],
)
def test_the_direction_and_the_match_become_script_parameters(query, expected):
    """'order: asc' and 'match: prefix|suffix' reach latte as parameters; the defaults are not
    spelled out, so a script that predates them runs unchanged."""
    test = object.__new__(substring_test.SubstringSearchTest)
    test.params = {"latte_schema_parameters": {"order_by": "register_time"}}
    assert test.extra_search_params(query, 3000) == expected


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


class _FakeRemoter:
    def __init__(self, has_apt):
        self.has_apt = has_apt
        self.sudo_cmds = []

    def run(self, cmd, **_):
        if cmd == "command -v apt-get":
            return dataclasses.make_dataclass("R", [("ok", bool)])(self.has_apt)
        return dataclasses.make_dataclass("R", [("stdout", str)])("6.8.0-1017-aws\n")

    def sudo(self, cmd, **kwargs):
        self.sudo_cmds.append((cmd, kwargs))


def test_perf_is_installed_with_bounded_apt_calls():
    node = dataclasses.make_dataclass("Node", [("remoter", object)])(_FakeRemoter(has_apt=True))
    substring_test.SubstringSearchTest._install_perf(node)
    update, install = node.remoter.sudo_cmds
    assert "Acquire::http::Timeout=60" in update[0] and update[1]["timeout"]
    assert install[0].startswith("DEBIAN_FRONTEND=noninteractive apt-get install")
    assert "DPkg::Lock::Timeout=120" in install[0]
    assert install[0].endswith("-y linux-tools-6.8.0-1017-aws linux-tools-common")
    assert install[1]["timeout"] and install[1]["ignore_status"]


def test_perf_is_not_installed_without_apt():
    node = dataclasses.make_dataclass("Node", [("remoter", object)])(_FakeRemoter(has_apt=False))
    substring_test.SubstringSearchTest._install_perf(node)
    assert node.remoter.sudo_cmds == []
