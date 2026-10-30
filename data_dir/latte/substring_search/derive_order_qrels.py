#!/usr/bin/env python3
"""Add ground truth for the orders other than the sort column's to a derived query matrix.

``derive_query_matrix.py`` writes, for every query set, the expected page of ``ORDER BY <sort>
DESC LIMIT n``. A pattern index answers in three more orders this script writes pages for, each as
a copy of the set's keywords under a new set name:

``<set>_table``
    a plain ``LIKE`` (no ORDER BY) against an index with no ``order_by``, which answers in table
    order: by token of the partition key -- Murmur3 of the ``user_id`` bytes -- then by the key's
    bytes. Checked with ``check_order: true``.
``<set>_tier``
    ``ORDER BY tier DESC`` against an index ordered ``'tier DESC'``: ``tier`` is the table's
    low-cardinality column (``tie_column`` in ``substring.rn``), five values derived from the sort
    value, so most rows tie and the ties come in table order.
``<set>_asc``
    ``ORDER BY <sort> ASC``: the sort column's index the other way round.

The ``tier`` of a row is ``substring.rn``'s ``tie_of()``; keep the two in step. Matching is
case-sensitive, as the AWS and local runs index. Usage::

    derive_order_qrels.py --dataset names_1M_matrix
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

from cassandra.metadata import Murmur3Token

# The sets of derive_query_matrix.py that are not windowed: a window is a range on the sort column,
# which neither the table nor the tier order has.
BASE_SETS = [
    "char1",
    "char2",
    "char4",
    "char8",
    "char16",
    "char32",
    "miss",
    "char1_rare",
    "char2_rare",
    "char4_rare",
    "char8_common",
    "char12_common",
]
TIE_VALUES = 5


def tie_of(sort: int) -> int:
    """substring.rn's tie_of(): the row's tier, from its sort value."""
    return ((sort % 1000003) * 7919 + 13) % 1000003 % TIE_VALUES


def table_key(uid: str) -> tuple[int, bytes]:
    """Where a row is in table order: its partition key's token, then the key's bytes."""
    key = uid.encode("utf-8")
    return Murmur3Token.hash_fn(key), key


def read_rows(dataset_dir: str) -> list[tuple[str, str, int]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "shards", "*.tsv"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                uid, name, sort = line.rstrip("\n").split("\t")
                rows.append((uid, name, int(sort)))
    return rows


def read_queries(path: str) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        return [tuple(line.rstrip("\n").split("\t")[:2]) for line in f if line.strip()]


ORDERS = {
    "table": lambda row: table_key(row[0]),
    "tier": lambda row: (-tie_of(row[2]),) + table_key(row[0]),
    "asc": lambda row: row[2],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset directory next to this script")
    ap.add_argument("--limit", type=int, default=20, help="the LIMIT the ground truth is for")
    args = ap.parse_args()
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset)
    rows = read_rows(dataset_dir)
    print(f"{len(rows)} rows")
    counts = []
    for base in BASE_SETS:
        queries_path = os.path.join(dataset_dir, f"queries_{base}.tsv")
        queries = read_queries(queries_path)
        matches = {qid: [row for row in rows if keyword in row[1]] for qid, keyword in queries}
        for order, key in ORDERS.items():
            name = f"{base}_{order}"
            shutil.copyfile(queries_path, os.path.join(dataset_dir, f"queries_{name}.tsv"))
            with open(os.path.join(dataset_dir, f"qrels_{name}.tsv"), "w", encoding="utf-8") as out:
                for qid, _ in queries:
                    for uid, _, _ in sorted(matches[qid], key=key)[: args.limit]:
                        out.write(f"{qid}\t{uid}\t1\n")
            per_query = [min(len(matches[qid]), args.limit) for qid, _ in queries]
            counts.append((name, sum(per_query) / len(per_query)))
        print(f"{base}: matches per keyword {[len(matches[qid]) for qid, _ in queries]}")
    with open(os.path.join(dataset_dir, "expected_counts_orders.tsv"), "w", encoding="utf-8") as out:
        out.write("set\tmean_rows\n")
        for name, mean in counts:
            out.write(f"{name}\t{mean:.1f}\n")


if __name__ == "__main__":
    main()
