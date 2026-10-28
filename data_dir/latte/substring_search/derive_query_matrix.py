#!/usr/bin/env python3
"""Derive a corpus whose query sets cross keyword length with how many names a keyword matches.

The generated corpora (``generate_local_dataset.py``) tie the two together: their short keywords
match hundreds of thousands of names and fill every page, their long keywords are needles that match
one name. A comparison across lengths then mostly compares 20 rows read per query with 1. This
script takes an existing ordered corpus and writes a copy with the missing corners filled in:

``char1_rare``, ``char2_rare``
    one and two characters that occur in 1 to 19 names. The corpus has none -- every one of its
    characters is in tens of thousands of names -- so they are made: characters absent from the
    corpus, each appended (or prepended) to that many randomly chosen names. Appending keeps every
    substring a name already had, so every other keyword matches exactly what it matched before.
``char4_rare``
    four characters that occur in 1 to 19 names, chosen from the corpus by measured count.
``char8_common``, ``char12_common``
    eight and twelve characters that occur in at least 20 names, so that they fill a page.

Every query set gets exact ground truth for an ordered query (``ORDER BY <sort> DESC LIMIT
--limit``): ``qrels_<set>.tsv`` lists the expected rows *in the expected order*, which is fully
determined because the sort value is the row's position in the corpus. For each set there is also a
``<set>_w50`` copy whose ground truth is the page below the window cursor ``substring.rn`` uses for
``window: 0.5`` (``sort < (1 - 0.5) * row count``).

``expected_counts.tsv`` gives, per set, the rows a query returns on average and per keyword, which
is what a performance run without ground truth compares its ``result_count`` with.

Two uses::

    # the 10M corpus for AWS: its own char1..char32 and miss sets are kept, the new sets added
    derive_query_matrix.py --source names_10M_long --dest names_10M_matrix

    # a 1M local corpus for accuracy runs: the first source shard, in 10 shards of 100k,
    # with every set chosen anew from it
    derive_query_matrix.py --source names_10M_long --dest names_1M_matrix --first-shards 1 --split 10

The source must be an ordered corpus written in sort order (``--sort-order sequential``): the ground
truth is found by walking it backwards. Matching is case-sensitive, as is the index the AWS runs
use; a case-insensitive run would need folded ground truth.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
from collections import Counter

SEED = 20261028

# Candidates for the made-up rare characters: CJK characters a name generator for this corpus
# never uses. The ones the corpus does contain are filtered out at run time.
RARE_CHAR_POOL = "鑫璇瑾琪珂琳璐瑶钰铭锦镜霖靖韬颢馨骏鹤麟黛鼎龄翊翎羲聆舒芊芮苒茜荀莘菁萱蔚蕙蕾薇"

# How many names each made-up rare keyword is put into: spread over 1-19 so that a set averages a
# partial page and still has keywords at both ends.
RARE_COUNTS = [1, 2, 3, 4, 6, 8, 10, 13, 16, 19]

# Sets kept from the source when the whole source is copied: the earlier runs measured these, so
# keeping them keeps the new run comparable. A subset chooses them anew from its own names.
KEPT_SETS = ["char1", "char2", "char4", "char8", "char16", "char32", "miss"]

KEYWORDS_PER_SET = 10
MAX_NAME_CHARS = 32


def read_shard(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            uid, name, sort = line.rstrip("\n").split("\t")
            yield uid, name, int(sort)


def read_queries(path):
    with open(path, encoding="utf-8") as f:
        return [tuple(line.rstrip("\n").split("\t")[:2]) for line in f if line.strip()]


def spread(items, k):
    """k items evenly spaced through an already sorted list."""
    if len(items) <= k:
        return list(items)
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def gram_counts(names, n):
    counts = Counter()
    for name in names:
        counts.update({name[i : i + n] for i in range(len(name) - n + 1)})
    return counts


def exact_counts(shards, keywords):
    """How many names contain each keyword, over the whole corpus, by the keyword's own n-grams."""
    by_len = {}
    for kw in keywords:
        by_len.setdefault(len(kw), set()).add(kw)
    counts = Counter()
    for path in shards:
        for _, name, _ in read_shard(path):
            for n, wanted in by_len.items():
                hit = {name[i : i + n] for i in range(len(name) - n + 1)} & wanted
                counts.update(hit)
    return counts


def choose_by_count(rng, sample, shards, n, lo, hi, scale, label):
    """KEYWORDS_PER_SET n-character keywords whose exact count is within [lo, hi], spread by count.

    Candidates come from the sample's counts scaled to the corpus, then are counted exactly.
    """
    counts = gram_counts(sample, n)
    # A rare band is wide open at the bottom of a sample; a common one must already be reached in it,
    # or the pick is mostly near-singletons.
    floor = lo / 2 if lo == 1 else lo
    window = [g for g, c in counts.items() if floor <= c * scale <= hi * 2]
    if not window:
        # Nothing reaches the band: take the most frequent and say so.
        window = [g for g, _ in counts.most_common(200)]
        print(f"  {label}: no {n}-gram near [{lo}, {hi}], trying the most frequent")
    # Up to 40 candidates per sample count: a plain random pick is nearly all of the most numerous
    # count, which for a rare band is 1.
    by_count = {}
    for g in window:
        by_count.setdefault(counts[g], []).append(g)
    del counts
    candidates = []
    for c in sorted(by_count):
        grams = sorted(by_count[c])
        candidates += rng.sample(grams, min(40, len(grams)))
    if len(candidates) > 400:
        candidates = rng.sample(candidates, 400)
    exact = exact_counts(shards, candidates)
    fitting = [(c, g) for g, c in exact.items() if lo <= c <= hi]
    if len(fitting) < KEYWORDS_PER_SET:
        print(f"  {label}: only {len(fitting)} {n}-grams within [{lo}, {hi}]")
        if not fitting:
            fitting = sorted(((c, g) for g, c in exact.items() if c > 0), reverse=lo > 1)[:KEYWORDS_PER_SET]
    # Spread over the distinct counts first, so a set covers its band rather than its mode.
    by_exact = {}
    for c, g in sorted(fitting):
        by_exact.setdefault(c, []).append(g)
    picked = []
    while len(picked) < KEYWORDS_PER_SET and any(by_exact.values()):
        levels = [c for c in sorted(by_exact) if by_exact[c]]
        for c in spread(levels, KEYWORDS_PER_SET - len(picked)):
            picked.append(by_exact[c].pop(0))
    return sorted(picked, key=lambda g: exact[g])


def needles(rng, sample, n, label):
    """Keywords of n characters taken from names in the sample, each found in that one name only."""
    counts = gram_counts(sample, n)
    once = sorted(g for g, c in counts.items() if c == 1 and not g.isascii())
    if len(once) < KEYWORDS_PER_SET:
        once = sorted(g for g, c in counts.items() if c == 1)
    print(f"  {label}: {len(once)} {n}-character needles to choose from")
    return rng.sample(once, KEYWORDS_PER_SET)


def write_queries(dest, name, keywords):
    with open(os.path.join(dest, f"queries_{name}.tsv"), "w", encoding="utf-8") as f:
        for i, kw in enumerate(keywords):
            f.write(f"{name}_{i:03d}\t{kw}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="source dataset directory, next to this script")
    ap.add_argument("--dest", required=True, help="dataset directory to write, next to this script")
    ap.add_argument("--first-shards", type=int, default=0, help="use only the first N source shards")
    ap.add_argument("--split", type=int, default=0, help="write the rows in this many shards")
    ap.add_argument("--limit", type=int, default=20, help="the LIMIT the ground truth is for")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(here, args.source)
    dest = os.path.join(here, args.dest)
    rng = random.Random(args.seed)

    src_shards = sorted(glob.glob(os.path.join(source, "shards", "names_*.tsv")))
    if args.first_shards:
        src_shards = src_shards[: args.first_shards]
    subset = bool(args.first_shards)

    # One pass: row count, sort order, the corpus' characters, and which rows may take a token.
    print(f"reading {len(src_shards)} shards of {source}")
    total = 0
    alphabet = set()
    short_rows = []
    for path in src_shards:
        for uid, name, sort in read_shard(path):
            if sort != total:
                raise SystemExit(f"{path}: row {total} has sort value {sort}; the source must be in sort order")
            alphabet.update(name)
            if len(name) <= MAX_NAME_CHARS - 2:
                short_rows.append(total)
            total += 1
    print(f"  {total} rows, {len(alphabet)} distinct characters")

    rare_chars = [c for c in RARE_CHAR_POOL if c not in alphabet]
    if len(rare_chars) < 3 * KEYWORDS_PER_SET:
        raise SystemExit(f"only {len(rare_chars)} characters of the pool are absent from the corpus")
    rare1 = rare_chars[:KEYWORDS_PER_SET]
    rare2 = [rare_chars[KEYWORDS_PER_SET + 2 * i] + rare_chars[KEYWORDS_PER_SET + 2 * i + 1] for i in range(KEYWORDS_PER_SET)]

    # Which rows carry which token, and whether it goes before or after the name.
    tokens = {}
    chosen = rng.sample(short_rows, 2 * sum(RARE_COUNTS))
    del short_rows
    it = iter(chosen)
    for group in (rare1, rare2):
        for token, count in zip(group, RARE_COUNTS):
            for _ in range(count):
                tokens[next(it)] = (token, rng.random() < 0.5)

    # Write the corpus.
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.join(dest, "shards"))
    split = args.split or len(src_shards)
    per_shard = -(-total // split)
    dest_shards = [os.path.join(dest, "shards", f"names_{i:03d}.tsv") for i in range(split)]
    out = None
    row = 0
    sample = []
    sample_size = min(total, 1_000_000)
    for path in src_shards:
        for uid, name, sort in read_shard(path):
            if row % per_shard == 0:
                if out:
                    out.close()
                out = open(dest_shards[row // per_shard], "w", encoding="utf-8")
            if row in tokens:
                token, before = tokens[row]
                name = token + name if before else name + token
            out.write(f"{uid}\t{name}\t{sort}\n")
            if row < sample_size:
                sample.append(name)
            row += 1
    out.close()
    print(f"wrote {total} rows in {split} shards to {dest}")
    scale = total / sample_size

    # The query sets.
    sets = {}
    if subset:
        common = {k: [kw for _, kw in read_queries(os.path.join(source, f"queries_{k}.tsv"))] for k in ("char1", "char2", "miss")}
        sets.update(common)
        sets["char4"] = choose_by_count(rng, sample, dest_shards, 4, 1000, total, scale, "char4")
        sets["char8"] = needles(rng, sample, 8, "char8")
        sets["char16"] = needles(rng, sample, 16, "char16")
        sets["char32"] = needles(rng, sample, 32, "char32")
    else:
        for k in KEPT_SETS:
            sets[k] = [kw for _, kw in read_queries(os.path.join(source, f"queries_{k}.tsv"))]
    sets["char1_rare"] = rare1
    sets["char2_rare"] = rare2
    sets["char4_rare"] = choose_by_count(rng, sample, dest_shards, 4, 1, args.limit - 1, scale, "char4_rare")
    sets["char8_common"] = choose_by_count(rng, sample, dest_shards, 8, args.limit, total, scale, "char8_common")
    sets["char12_common"] = choose_by_count(rng, sample, dest_shards, 12, args.limit, total, scale, "char12_common")
    del sample

    # Ground truth: walk the corpus from the highest sort value down, keeping each keyword's first
    # `limit` matches above the window cursor and its first `limit` below it. A keyword drops out
    # once both pages are full, so the common ones cost little.
    cursor = int((1.0 - 0.5) * total)
    keywords = sorted({kw for kws in sets.values() for kw in kws})
    top = {kw: [] for kw in keywords}
    below = {kw: [] for kw in keywords}
    matches = Counter()
    active = set(keywords)
    for path in reversed(dest_shards):
        if not active:
            break
        with open(path, encoding="utf-8") as f:
            rows = f.read().splitlines()
        for line in reversed(rows):
            uid, name, sort = line.split("\t")
            sort = int(sort)
            for kw in [kw for kw in active if kw in name]:
                matches[kw] += 1
                if len(top[kw]) < args.limit:
                    top[kw].append(uid)
                if sort < cursor and len(below[kw]) < args.limit:
                    below[kw].append(uid)
                if len(below[kw]) == args.limit:
                    active.discard(kw)
        del rows

    with open(os.path.join(dest, "expected_counts.tsv"), "w", encoding="utf-8") as counts:
        counts.write("set\tmean_rows\tmatches_per_keyword\n")
        for name, kws in sets.items():
            for suffix, pages in (("", top), ("_w50", below)):
                set_name = name + suffix
                write_queries(dest, set_name, kws)
                with open(os.path.join(dest, f"qrels_{set_name}.tsv"), "w", encoding="utf-8") as f:
                    for i, kw in enumerate(kws):
                        for uid in pages[kw]:
                            f.write(f"{set_name}_{i:03d}\t{uid}\t1\n")
                rows = [len(pages[kw]) for kw in kws]
                # A keyword that dropped out early matched at least `limit` names below the cursor.
                found = [str(matches[kw]) if kw in active or matches[kw] < args.limit else f">={matches[kw]}" for kw in kws]
                mean = sum(rows) / len(rows)
                counts.write(f"{set_name}\t{mean:.1f}\t{','.join(found)}\n")
                if not suffix:
                    print(f"  {set_name:14} rows/query {mean:5.1f}  matches {','.join(found)}")
    print(f"window cursor for 0.5: sort < {cursor}")


if __name__ == "__main__":
    main()
