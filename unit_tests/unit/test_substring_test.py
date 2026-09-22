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
import re

import pytest

# NOTE: the test class is reached through the module rather than imported by name -- pytest's
#       unittest collector picks up any 'unittest.TestCase' subclass in a test module's namespace,
#       and 'SubstringSearchTest' is one through ClusterTester.
import substring_test
from substring_test import (
    INDEX_SIZE_METRIC,
    SEGMENT_COUNT_METRIC,
    SUBSTRING_BUILD_COUNT_COLUMN,
    SUBSTRING_WORKLOAD,
    WITH_INDEX_PARAM,
    SubstringIndexBuildResult,
    parse_index_gauge,
)
from sdcm import sct_abs_path

RUNE_PARAM_RE = re.compile(r"""latte::param!\(\s*["'](?P<name>[^"']+)["']""")
RUNE_FUNCTION_RE = re.compile(r"^pub async fn (?P<name>\w+)", re.MULTILINE)

# An excerpt of what vector-store's /metrics answers with, one index per keyspace/index_name pair.
# The index name is folded, as Scylla folds unquoted identifiers, while the test creates the index
# as 'sub_idx_names_10M_0' -- the point of the case-folding in the parser.
METRICS_SAMPLE = """\
# HELP substring_index_size_bytes Total size of a substring search index (bytes)
# TYPE substring_index_size_bytes gauge
substring_index_size_bytes{keyspace="substring_bench",index_name="sub_idx_names_10m_0"} 3221225472
substring_index_size_bytes{keyspace="substring_bench",index_name="sub_idx_names_10m_1"} 32212254720
# HELP substring_segment_count Number of segments in a substring search index
# TYPE substring_segment_count gauge
substring_segment_count{keyspace="substring_bench",index_name="sub_idx_names_10m_0"} 8
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
    mapped = {
        getattr(SUBSTRING_WORKLOAD.params, field.name) for field in dataclasses.fields(SUBSTRING_WORKLOAD.params)
    }
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
