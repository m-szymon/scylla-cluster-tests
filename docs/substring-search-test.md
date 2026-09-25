# Substring search performance test

`substring_test.SubstringSearchTest.test_substring_search` benchmarks ScyllaDB's `substring_index`:
the index that answers `WHERE column LIKE '%keyword%' LIMIT n` from an n-gram index on the
vector-store node instead of scanning the table.

It exists to answer the questions the feature design left open:

| question | what the run reports |
|---|---|
| Do containment queries meet p99 < 100 ms at 10M names, and at what throughput? | one latency table per `expected_p99_read_ms` in the plan, one row per query configuration |
| How much memory does the index cost per indexed name? | **Substring Index Size**: bytes, bytes per name and segment count |
| Can an ordered query skip segments, or does it walk every match? | the segment span columns of the size table, and the per-query walk columns of every latency row |
| How long does the index take to build, and at what rate? | **Substring Index Build Time**, from the build-oriented plan only |
| What does `ORDER BY` cost, and does a later page cost more than the first? | the ordered rows against the stage-1 run's plain ones, and the windowed rows against the ordered |
| Does that depend on the order the rows arrived in? | the `names_10M_shuffled` dataset, same corpus, adverse arrival order |

The default plan answers the first two and deliberately does not measure the third: it indexes while
it loads, so there is no separate build to time. See *Indexing during the load* below.

The orchestration is shared with the full-text benchmark (`search_perf_test.py`, see
[fts-search-test.md](fts-search-test.md)). This test is a descriptor on top of it -- a rune script,
a vocabulary and a set of result tables -- plus the index-size reading, which is specific to the
question above.

## What the run does

Per dataset in the plan, and per step within it:

1. drop and recreate the keyspace and table, once per dataset;
2. load the step's shards of display names into ScyllaDB, on top of everything the earlier steps
   loaded, through `latte run -f load`;
3. get the index ready, in one of two ways -- see *Indexing during the load* -- and read how large
   it is from vector-store's `substring_index_size_bytes` gauge;
4. run every query set the step names, each as its own latte `search` phase with its own
   concurrency, rate and duration.

## Indexing during the load

A dataset with `index_during_load: true` creates the index **before** the rows are written, so the
index node ingests through the base table's CDC log while latte is still loading, and no separate
full-scan build happens at all. This is what the default plan does, because the questions it
answers are about queries.

Knowing when ingestion is finished is the only hard part, and the run asks the index for its
document count, waiting until that reaches the number of rows loaded. It deliberately does not
probe for the last row written. The CDC log is read per stream, one per token range, and the
streams advance independently, so the last row written can be indexed while another stream is still
far behind. Under a concurrent load there is no single last row either. A last-row probe would pass
early and the benchmark would run against a partial index, which is the worst failure mode
available here, because nothing looks wrong -- the answers are just quietly too small.

The run then waits for the segment count to stop moving, up to ten minutes. An index ingested
incrementally is many small segments that Tantivy goes on merging after the writes stop, and a
search visits every one of them, so querying the moment the last row lands measures a transient
rather than the index. If the count has not stabilised in time the run says so and queries anyway;
the segment count in the size row records what the queries actually ran against.

No build time is reported in this mode, because nothing was built in a way worth timing. Use
`aws_build_config.yaml` for that, which loads first and builds afterwards in three steps, at 1M and
twice at 10M -- the second 10M build being a control, since the first runs on a cluster still
settling from a bulk load.

Query keywords come from the dataset's `queries_<set>.tsv` as bare words; the rune script wraps each
one into `'%keyword%'` and binds it. Nothing in the query path uses `ALLOW FILTERING`, which is the
whole point: a run that accidentally scanned would report plausible latencies for the wrong thing.

## Ordering, and the shape of a query set

An index created with an `order_by` column answers newest-first and takes a range on that column, so
a query set can be asked in three shapes. The shape is a property of the query entry:

```yaml
- set: char2                    # plain:    WHERE nickname LIKE ? LIMIT 20
- set: char2
  ordered: true                 # ordered:  ... ORDER BY register_time DESC LIMIT 20
- set: char2
  ordered: true
  window: 0.5                   # windowed: ... AND register_time < ? ORDER BY ... LIMIT 20
```

The shape goes into the Argus **row label**, not into a column, because it changes what the latency
means: an ordered row and a plain row are answers to two different questions rather than two
configurations of one measurement. Two entries naming one set in one step would otherwise collide
on the label and push conflicting numbers into a single row.

**An index with `order_by` takes the ordered walk for every query**, plain ones included: the walk
is chosen from the index's option, not from whether the query said `ORDER BY`. So on such an index a
plain row and an ordered row of the same set cost the same, and neither is the unordered baseline.
The price of ordering is an ordered row against a plain row from an index created *without*
`order_by` -- in practice the stage-1 run, which served the same names.

`ordered` and `window` need `order_by: 'register_time'` in the test case's
`latte_schema_parameters`, which is what gives the table its sort column and the index its option.
A plan asking for either without it is rejected before the run loads anything.

### Why `window` stands in for paging

The point of the cursor the feature introduced is that page 50 costs what page 1 costs: a later page
resumes the index's walk instead of restarting it and skipping. latte drives no CQL paging, so the
benchmark cannot ask for page 50 directly -- but a page that resumes at a cursor *is* a query
bounded by that cursor, which is exactly what `window: 0.5` sends. A windowed row far slower than
the ordered row next to it is the cursor failing to do its job.

`window` is a fraction of the way down the order, so `0.5` starts halfway. Ground truth cannot
describe a windowed answer -- it covers the whole corpus, and the rows below the cursor are
correctly absent but count as misses -- so do not put `qrels: true` on a windowed entry.

### Arrival order is the variable that matters

The index prunes a segment by the span of sort values it holds, and a segment holds whatever arrived
together. Load the accounts oldest first and each segment is one contiguous slice of the range, so
most segments cannot hold a top-20 row and are never opened. Load them shuffled -- what a backfill
by partition key, or a restore, produces -- and every segment spans nearly the whole range, nothing
prunes, and an ordered query walks every match.

That is the open risk of the ordering design, so the plan measures it rather than arguing it. The
corpus generator takes `--sort-order sequential|shuffled`; both write the same names and the same
set of sort values and differ only in which row carries which, so the comparison is about layout
alone:

```sh
python3 data_dir/latte/substring_search/generate_local_dataset.py \
    --dataset names_10M_shuffled --names 10000000 --shards 100 --qrels-cap 0 --sort-order shuffled
```

`aws_config.yaml` runs the ordered rows twice, once per corpus. If the two agree, pruning is not
what carries the ordered query and the segment work P3 plans is less urgent than the design note
assumes. If the shuffled rows are far worse, the gap is the size of the problem.

Whether pruning happened is reported rather than inferred from the latency. After the index
settles, the size row carries `segment_span_mean_pct` and `segment_span_max_pct`: each segment's
span of the sort column as a share of the whole range, from vector-store's per-segment gauges
(`substring_segment_docs`, `substring_segment_sort_min/max`), and the test log lists every segment.
A mean near 100% means no segment can ever be skipped. Then every latency row carries what one
query cost the index, from the delta of the walk totals (`substring_search_*_total`) over the phase:
`walk_us_per_query`, `prepare_us_per_query` (the part of the walk spent reading segment bounds before
the first posting) and `column_opens_per_query` (columns opened for it), `page_resolve_us_per_query`
(the part spent turning the page into primary ids), `segments_considered_per_query`, `segments_opened_per_query`,
`postings_per_query`, `heap_entrants_per_query` and `store_reads_per_query`. Postings scanned is
the number to read first: an ordered query that scans tens of thousands of postings for a page of
20 is walking segments it could not skip, and that is a layout problem, not a query one.

To compare two index configurations on the same layout, a dataset can declare `index_variants`
(each a `label` and optionally one extra option, e.g. `options: {poc_option_1: 'true'}`): every variant is created
before the load and ingests the same CDC stream, then the step's query sets run once per variant,
with `[label]` at the end of each row label. ScyllaDB decides which index answers a query and not
by name, so each round starts with a five-second probe to find out which one is serving, and ends
by dropping it so the next round reaches another. The size row, the layout and the per-query walk
columns are reported per variant. `poc_option_1..4` are placeholders ScyllaDB stores and passes
through; what they mean is vector-store's business and is documented there.

The corpus now carries a third column (`user_id<TAB>nickname<TAB>register_time`), so a corpus
generated before this existed has to be regenerated before an ordered run. The rune script says so
rather than inserting nulls, which would look like a recall bug in the index.

## Running it from a developer machine against AWS

This is the normal way to run it. SCT runs on your machine, the cluster runs on AWS.

The two halves are not symmetric, which is worth knowing before you start: the vector-store node
builds from git, so pushing the branch is all it needs, while the database node installs a package,
so the Scylla branch has to be built into one first.

### 1. Push the vector-store branch

```sh
git -C vector-store push <your-fork> substring-index
```

`test-cases/substring-search/substring-search-test.yaml` names the fork and ref:

```yaml
vector_store_source_repo: 'https://github.com/m-szymon/vector-store.git'
vector_store_source_ref: 'substring-index'
```

Provisioning then builds vector-store from source over the newest vector-store AMI, which is how the
full-text test reaches an unreleased vector-store too. The repo value is handed to `git remote add`
on the node, so it is a full URL rather than `owner/name`, and the fork has to be readable without
credentials. The fetch is shallow, one ref.

### 2. Build a Scylla unified package

Pushing the `scylladb` branch does nothing for the run: the database node installs Scylla from a
package or boots an AMI, and neither is a git branch. You do **not** need to build an AMI, though.
`unified_package` takes the URL of a relocatable package, and SCT provisions a stock Ubuntu 24.04
image and installs it there (`sct_config.py` section 6.0.1, which also forces
`use_preinstalled_scylla=False` and picks the base image for the architecture).

```sh
cd scylladb
./tools/toolchain/dbuild ninja dist-unified-dev
# -> build/dev/dist/tar/scylla-unified-<version>.<arch>.tar.gz, e.g.
#    scylla-unified-2026.4.0~dev-0.20260921.d6716cf8568c.x86_64.tar.gz

pkg=$(ls -t build/dev/dist/tar/scylla-unified-*.tar.gz | head -1)
aws s3 cp "$pkg" s3://<your-bucket>/ --acl public-read
export SCT_UNIFIED_PACKAGE=https://<your-bucket>.s3.amazonaws.com/$(basename "$pkg")
```

The version string embeds the commit the package was built from, so `git log -1 --format=%h` on the
branch and the sha in the filename should agree -- the quickest way to catch a package built before
the last fix.

Any URL the node can fetch works; S3 is just the convenient one when you already have AWS
credentials. The test case pins nothing here, because the URL is yours, so a run without
`SCT_UNIFIED_PACKAGE` (or `SCT_AMI_ID_DB_SCYLLA`, if you do have a Scylla AMI with the feature) fails
at config validation rather than at the `CREATE CUSTOM INDEX` statement half an hour later.

### 3. Generate the corpus

The plan's dataset has no `base_url`, so the names are read from your machine and staged onto the
loader. No S3 bucket is needed.

```sh
python3 data_dir/latte/substring_search/generate_local_dataset.py \
    --dataset names_10M --names 10000000 --shards 100 --qrels-cap 0
```

That writes about 250 MB into `data_dir/latte/substring_search/names_10M/`, deterministically: the
same arguments give the same corpus on any machine. The names imitate the workload the index was
built for -- mostly 2 to 6 CJK characters, a minority Latin or numeric -- and the query sets are
chosen by measured frequency, so `char1` really is the hot case and `char4` really does exercise the
gram intersection. To pull the corpus from S3 instead, upload the directory and add
`base_url: s3://bucket/prefix` to the dataset in the plan.

### A note on SSH

The test case sets `ip_ssh_connections: 'public'`. SCT's default is `private`, which is correct when
it runs on a cloud runner inside the VPC, as it does under Jenkins. Driven from a developer machine
the private address is unreachable, and the run dies about 25 minutes in with
`Waiting for SSH to be up: timeout` on a node that booted perfectly and whose public address accepts
your key. Only the runner-to-node path changes; the nodes still reach each other privately.

The cost is that the corpus is staged to the loader across the internet rather than inside the VPC.
For anything beyond a feasibility run, use an SCT cloud runner and drop the setting.

### 4. Run

```sh
export SCT_UNIFIED_PACKAGE=https://<your-bucket>.s3.amazonaws.com/scylla-unified-....tar.gz
./sct.py run-test substring_test.SubstringSearchTest.test_substring_search --backend aws \
    --config test-cases/substring-search/substring-search-test.yaml
```

A smaller first run: generate fewer names into a differently named dataset, copy `aws_config.yaml`,
trim its shard lists and point `SCT_SEARCH_TEST_CONFIG` at the copy.

### Argus

Turn it off per run:

```sh
export SCT_ENABLE_ARGUS=false
```

Every SCT config field has an `SCT_<NAME>` environment override, and this one is left at its default
in the test cases on purpose, so the test can still report normally if it is ever run from a
pipeline. With it off, every row the test submits is written to `argus_replay_log_*.jsonl` in the
run's logdir and nothing is posted. That file is where the build times and index sizes end up.

Its one side effect: `LatteStressThread` re-runs `latte schema` before every command
(`latte_thread.py:199`) instead of once per script. The rune script's schema is idempotent DDL --
`CREATE KEYSPACE`/`CREATE TABLE IF NOT EXISTS`, no index -- so this costs a few round trips and
changes nothing else. If you want latte to behave exactly as it does under Jenkins, leave Argus
enabled and set `export JOB_NAME=local_run` instead: `init_argus_client` goes replay-only when the
job name is exactly that.

Running with neither is what you want to avoid. `sdcm/utils/ci_tools.py` supplies `local_run` as the
fallback for an unset `JOB_NAME`, but `docker/env/hydra.sh` passes `-e JOB_NAME="${JOB_NAME}"`
unconditionally, so an unset one arrives inside the container as an **empty string**, which is not
the fallback value. The run then posts to the real Argus and fills its log with
`No SCTTestRun found matching ...` and `Failed to submit heartbeat to argus`. Not fatal, but they
retry with backoff and bury everything else.

## Reading the results

| where | what |
|---|---|
| `argus_replay_log_*.jsonl` in the logdir | every result row: build time, indexing throughput, index size, bytes per name, and the latency rows |
| the test log | `Index 'sub_idx_...': N bytes for M names (X bytes/name), S segments` per build, the build time line next to it, the segment layout under `Index '...' layout:` and `Index work per query:` after each query phase |
| latte's `.hdr` files on the loader | the raw latency histograms behind the p99 |
| Grafana screenshots, with `n_monitor_nodes: 1` | server side latency and reactor stalls, which is how you tell a slow index node from a slow database |

A rate row whose measured throughput came in below the rate it asked for did not keep up, and its
latency is a queueing artifact rather than a reading. If the unthrottled row lands well under the
target, suspect the loader before the cluster and raise `n_loaders`.

The number to compare across steps is **bytes per name**: an n-gram index stores every substring of
`min_gram..max_gram` characters of every value, so its cost per row depends on name length and
`max_gram`, not on the row count. If 1M and 10M names disagree on it, something other than the index
grew.

## The plan

`search_test_config` names a YAML plan; two are in the repo.

`data_dir/latte/substring_search/aws_config.yaml` (the default)
: 10M names in one step, indexed during the load, then thirteen query phases of two minutes each.
  `char2`, `char1` and `char4` each run at a ladder of rates up to and past the 10k ops/s the design
  was written against, ending unthrottled to find the ceiling, plus one `latin` and one `miss`
  phase.

`data_dir/latte/substring_search/aws_build_config.yaml`
: the build-timing plan: 1M, then 10M, then a rebuild on the same 10M, with a shorter set of
  queries after each.

`data_dir/latte/substring_search/local_config.yaml`
: 3000 synthetic names for the docker smoke run below.

A step is `shards:` plus `queries:`. An empty `shards` list loads nothing and only rebuilds the
index, which is how the plan samples build time twice on one corpus. A query entry names a set and
may override `limit`, `concurrency`, `rate`, `duration` and `expected_p99_read_ms` from the
dataset's `defaults`. `qrels: true` turns on the correctness check described below.

Index options are not part of the plan; they are `latte_schema_parameters` in the test case, because
SCT appends those to every latte invocation and the index is created inside the `build_index` phase:

```yaml
latte_schema_parameters:
  keyspace: 'substring_bench'
  table: 'users'
  min_gram: 1
  max_gram: 3
  case_sensitive: 'false'
  order_by: 'register_time'
```

`max_gram` is the knob that trades index size for the cost of long keywords: at 3, a keyword of four
or more characters is answered by intersecting three-character grams and verifying each candidate.

## Correctness under load

Containment is exact, so a query set can carry ground truth and the run will check it:

```yaml
- set: char4
  qrels: true
```

The rune script then compares the ids it got back against `qrels_<set>.tsv`. Only the set matters,
since containment has no ranking, so it records two numbers:

`precision`
: the share of returned rows that really do contain the keyword, so **1.0 is the only acceptable
  value**. Anything less means the index returned a row it should not have, which is what a broken
  verification step after the gram intersection looks like. It is recorded for every query,
  including ones whose answer should be empty. Note this is not `metrics::precision_at_k`, which
  divides by the limit and would score a perfect three-row answer at 3/20.

`recall`
: the share of the rows that contain the keyword which came back, so again 1.0 unless something is
  wrong. Recorded only where there is ground truth to recall.

Two things make recall read low for reasons that are not bugs: a true answer larger than the limit,
and ground truth generated over a corpus larger than the step has loaded. The generator writes
ground truth only for sets whose every keyword matches at most `--qrels-cap` names, and a plan
should put `qrels: true` on a step that has loaded the whole dataset -- which is why the large
corpora use `--qrels-cap 0` and the check is a property of the small ones.

## Local smoke run on docker

Validates the orchestration without provisioning anything. The numbers are meaningless.

```sh
# once: build the two images from the branches
cd <scylladb>     && ./tools/toolchain/dbuild ninja dist-dev \
                  && ./dist/docker/redhat/build_docker.sh --mode dev
cd <vector-store> && docker build -t local/vector-store:substring .

# once: generate the tiny corpus
python3 data_dir/latte/substring_search/generate_local_dataset.py

./docker/env/hydra.sh run-test substring_test.SubstringSearchTest.test_substring_search \
  --backend docker --config test-cases/substring-search/substring-search-test-docker.yaml
```

The docker backend takes prebuilt images only, so `vector_store_source_repo`/`_ref` must not be set
there -- `sct_config.py` rejects a source build on that backend.

## Sizing the nodes

Measured locally, not guessed: the index node indexes about 110,000 names a second and the index
costs about 33 bytes per name, falling as the corpus grows (54 bytes at 3,000 names, 36 at 300,000,
33 at 2,000,000). Indexing 10M names is therefore a matter of a minute or two, and the index is a
few hundred megabytes rather than the few gigabytes the design assumed.

Two caveats on that size. The corpus is synthetic and draws on only a few dozen distinct Han
characters, while real display names draw on thousands; vocabulary drives the term dictionary, so a
real index will be larger. And the figure is the index node's own gauge, not resident memory.

What follows for the instance types:

`instance_type_vector_store`
: memory can be modest -- `c8g.xlarge` gives 8 GiB, several times the headroom needed. The vCPU
  count should not be cut, because the index node serves the queries and its cores are what set the
  throughput ceiling this run measures.

`instance_type_db`
: not idle. Every containment query returns up to `limit` primary keys and Scylla then reads those
  rows from the base table, so 10k queries/s at a limit of 20 means 200k row reads/s here. The data
  is small enough to sit in cache, but this is a likelier constraint than index-node memory.

`instance_type_loader`
: one loader has to generate the whole query rate. If the unthrottled phase lands near a suspiciously
  round number, check loader CPU before concluding anything about the cluster.

The time in a run goes into loading 10M rows through one loader and staging 190 MB of shards over
SSH, not into indexing. The two-hour ingestion budget in the plan is deliberately generous.

## What this test does not cover

* **Real CQL paging.** Ordering and a later page are measured (see *Ordering* above), but through a
  range bound rather than by paging a result set, because latte drives no paging. What that does not
  cover is the paging state itself: the cursor's round trip through `paging_state`, and whether the
  pages of one query are disjoint and cover everything. That is checked in the ScyllaDB tree
  instead.
* **Mixed workloads.** Names are loaded, then queried. Nothing updates a name while the queries run,
  so the index's write path is measured only as build throughput.
* **More than one index.** Real use indexes both a nickname and a username column and merges the two
  answers in the application. The plan builds one index at a time.
* **Failure modes.** No nemesis, no node loss, no vector-store restart.
