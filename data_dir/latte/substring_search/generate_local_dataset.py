#!/usr/bin/env python3
"""Generate the display-name corpora the substring search benchmark runs against.

Called with no arguments it writes the tiny ``local_tiny`` dataset ``local_config.yaml`` reads,
the same way the full-text generator does::

    python3 data_dir/latte/substring_search/generate_local_dataset.py

Called with ``--names``/``--shards`` it writes a corpus of any size, which is how the AWS plans get
their data -- ``aws_config.yaml`` expects ``names_10M`` unless it is pointed elsewhere::

    python3 data_dir/latte/substring_search/generate_local_dataset.py \\
        --dataset names_10M --names 10000000 --shards 100

The output is deterministic for a given (seed, name count, shard count), so regenerating is always
safe and two machines produce the same corpus.

File formats consumed by ``substring.rn``:

* ``shards/names_NNN.tsv`` -- ``user_id<TAB>nickname``
* ``names.tsv``            -- same, for the non-sharded path
* ``queries_<set>.tsv``    -- ``query_id<TAB>keyword`` (a bare keyword; the script wraps it in %%)
* ``qrels_<set>.tsv``      -- ``query_id<TAB>user_id<TAB>grade``, grade 1 for every row that
                              contains the keyword

The names imitate the workload this index was built for: a live-streaming site whose display names
are mostly 2 to 6 CJK characters, with a minority carrying Latin letters or digits. What matters for
the benchmark is not that they read naturally but that keyword selectivity is controlled, so the
query sets below are chosen by measured frequency rather than by hand:

``char1``   one character, matching a large fraction of the corpus. The hot case: the index has to
            stop at LIMIT rather than collect every match.
``char2``   two characters, the typical search-box query.
``char4``   four characters, longer than the default ``max_gram`` of 3, so the index intersects
            three-character grams and verifies each candidate against the stored name.
``latin``   Latin substrings, which a case-insensitive index has to fold on both sides.
``miss``    keywords no name contains: the cost of an empty answer.

Ground truth (``qrels_*.tsv``) is written only for query sets whose every keyword matches at most
``--qrels-cap`` names, since recall against a limit of 20 is meaningless once the answer is larger
than the limit. In practice that means the ``char4``, ``latin`` and ``miss`` sets of a small corpus;
``--qrels-cap 0`` turns ground truth off entirely, which is what the large corpora use.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter

SEED = 20260921

# Enough common Hanzi to make substrings of length 1-4 behave like real names: a few hundred
# thousand distinct 2-6 character combinations, with a long tail.
SURNAMES = "王李张刘陈杨黄赵周吴徐孙马朱胡林郭何高罗郑梁谢宋唐许韩冯邓曹彭曾"
GIVEN = "明红刚强军伟芳娜秀静丽敏艳杰涛超磊鹏飞龙凤云海山川风雨雪月星辰"
# Words that show up inside names, so that multi-character keywords have something to match. The
# first few are the ones the feature was demonstrated with.
WORDS = ["将军", "元帅", "司令", "南宫", "将领", "国王", "玩家", "粉丝团", "直播间", "小号", "大神", "队长"]
PREFIXES = ["小", "大", "老", "阿", "新", "超级", "最强"]
LATIN = ["gamer", "king", "ng", "pro", "live", "star", "fan", "top", "cool", "new"]

# Roughly the shape described for the real table: mostly short CJK names, a minority Latin or mixed.
P_WORD_NAME = 0.35  # prefix + word (+ suffix)
P_PERSON_NAME = 0.40  # surname + given name(s)
P_LATIN_NAME = 0.15  # Latin word (+ digits)
# the rest: digits only

MAX_NAME_CHARS = 32


def _make_name(rng: random.Random) -> str:
    roll = rng.random()
    if roll < P_WORD_NAME:
        name = rng.choice(WORDS)
        if rng.random() < 0.5:
            name = rng.choice(PREFIXES) + name
        if rng.random() < 0.25:
            name += rng.choice(GIVEN)
    elif roll < P_WORD_NAME + P_PERSON_NAME:
        name = rng.choice(SURNAMES) + "".join(rng.choice(GIVEN) for _ in range(rng.randint(1, 2)))
        if rng.random() < 0.15:
            name = rng.choice(PREFIXES) + name
    elif roll < P_WORD_NAME + P_PERSON_NAME + P_LATIN_NAME:
        name = rng.choice(LATIN)
        if rng.random() < 0.4:
            name = name.capitalize() if rng.random() < 0.5 else name.upper()
        if rng.random() < 0.5:
            name += str(rng.randint(1, 99999))
    else:
        name = str(rng.randint(100, 9999999))
    return name[:MAX_NAME_CHARS]


def _generate_names(count: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [(f"u{i:09d}", _make_name(rng)) for i in range(count)]


def _write_tsv(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write("\t".join(str(field) for field in row) + "\n")


def _substring_counts(names: list[tuple[str, str]], length: int, sample: int) -> Counter:
    """How many of the (sampled) names contain each substring of the given length."""
    counts: Counter = Counter()
    for _, name in names[:sample]:
        seen = {name[i : i + length] for i in range(len(name) - length + 1)}
        counts.update(seen)
    return counts


def _pick_keywords(counts: Counter, sampled: int, low: float, high: float, wanted: int) -> list[str]:
    """Keywords whose share of the sampled names falls in [low, high), most frequent first."""
    picked = [(kw, n) for kw, n in counts.most_common() if low <= n / sampled < high and not kw.isascii()]
    return [kw for kw, _ in picked[:wanted]]


def _latin_keywords(counts: Counter, wanted: int) -> list[str]:
    picked = [kw for kw, _ in counts.most_common() if kw.isascii() and kw.isalpha()]
    return picked[:wanted]


def _exact_matches(names: list[tuple[str, str]], keyword: str, cap: int) -> list[str] | None:
    """Every id whose name contains the keyword, or None once there are more than `cap` of them."""
    folded = keyword.lower()
    hits = []
    for user_id, name in names:
        if folded in name.lower():
            hits.append(user_id)
            if len(hits) > cap:
                return None
    return hits


def generate(dataset: str, names_count: int, shards: int, out_root: str, seed: int, qrels_cap: int) -> None:
    ds_dir = os.path.join(out_root, dataset)
    print(f"generating {names_count} names into {ds_dir} ({shards} shard(s))")
    names = _generate_names(names_count, seed)

    if shards > 1:
        per_shard = (names_count + shards - 1) // shards
        for shard in range(shards):
            chunk = names[shard * per_shard : (shard + 1) * per_shard]
            if not chunk:
                break
            _write_tsv(os.path.join(ds_dir, "shards", f"names_{shard:03d}.tsv"), chunk)
    else:
        _write_tsv(os.path.join(ds_dir, "names.tsv"), names)

    # Frequency is estimated from a sample: the bands below only need to be roughly right, and a
    # full pass over a 10M-name corpus for every length would dominate the runtime.
    sample = min(len(names), 200_000)
    print(f"choosing keywords from a sample of {sample} names")
    counts1 = _substring_counts(names, 1, sample)
    counts2 = _substring_counts(names, 2, sample)
    counts4 = _substring_counts(names, 4, sample)

    query_sets = {
        # A single character carried by a large share of the names.
        "char1": _pick_keywords(counts1, sample, 0.01, 1.0, 10),
        # Two characters, the typical query.
        "char2": _pick_keywords(counts2, sample, 0.001, 0.2, 10),
        # Four characters: past max_gram, so the index has to verify its candidates.
        "char4": _pick_keywords(counts4, sample, 0.0, 0.01, 10),
        "latin": _latin_keywords(_substring_counts(names, 3, sample), 10),
        # Keywords no generated name can contain: none of the pools hold these characters.
        "miss": ["㊙㊗", "ΩΨΔ", "zzqx"],
    }

    for qset, keywords in query_sets.items():
        if not keywords:
            print(f"  query set {qset}: no keyword matched the frequency band, skipped")
            continue
        _write_tsv(
            os.path.join(ds_dir, f"queries_{qset}.tsv"),
            [(f"{qset}_{i:03d}", kw) for i, kw in enumerate(keywords)],
        )
        print(f"  query set {qset}: {len(keywords)} keywords -> {keywords}")

        if qrels_cap <= 0:
            continue
        rows = []
        capped = False
        for i, kw in enumerate(keywords):
            hits = _exact_matches(names, kw, qrels_cap)
            if hits is None:
                capped = True
                break
            rows.extend((f"{qset}_{i:03d}", user_id, 1) for user_id in hits)
        if capped:
            print(f"    no ground truth: a keyword matches more than {qrels_cap} names")
        else:
            _write_tsv(os.path.join(ds_dir, f"qrels_{qset}.tsv"), rows)
            print(f"    ground truth: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="local_tiny", help="dataset directory name (default: local_tiny)")
    parser.add_argument("--names", type=int, default=3000, help="how many names to generate (default: 3000)")
    parser.add_argument("--shards", type=int, default=3, help="how many shard files to split them into (default: 3)")
    parser.add_argument(
        "--out",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="where to write the dataset directory (default: next to this script)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    parser.add_argument(
        "--qrels-cap",
        type=int,
        default=50,
        help="skip ground truth for a query set whose keyword matches more names than this; 0 disables it (default: 50)",
    )
    args = parser.parse_args()
    generate(args.dataset, args.names, args.shards, args.out, args.seed, args.qrels_cap)


if __name__ == "__main__":
    main()
