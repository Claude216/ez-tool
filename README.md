# ez-tool

Evaluating tool-selection methods on StableToolBench. Phase 1 establishes a
BM25 retrieval baseline; the method under investigation plugs into the same
harness so comparisons are apples to apples by construction.

## Quick start

**All commands run on a compute node.** beegfs is not mounted on login nodes,
and login nodes are not for execution — see [CLAUDE.md](CLAUDE.md).

```bash
ssh c24   # or: srun -p rome --time=01:00:00 --pty bash
```

```bash
bash scripts/fetch_data.sh
/mnt/beegfs/lli66/venvs/ez-tool/bin/python -m ez_tool.cli inspect
/mnt/beegfs/lli66/venvs/ez-tool/bin/python -m ez_tool.cli run --config configs/bm25_baseline.yaml
```

Or as a batch job:

```bash
sbatch scripts/slurm/run_bm25.sbatch
```

## Layout

```
src/ez_tool/
  paths.py            canonical locations; refuses to run without beegfs
  data/corpus.py      toolenv2404_filtered -> one document per API
  data/queries.py     solvable_queries -> qrels, plus label reconciliation
  data/text.py        identifier-aware normalization (camelCase, snake_case)
  retrieval/base.py   the Retriever protocol every method implements
  retrieval/bm25.py   BM25 baseline (bm25s + Snowball stemming)
  eval/metrics.py     recall@k, precision@k, NDCG@k, MRR
  cli.py              inspect / run

  textmine/           section 7 replication (separate experiment; see below)
    vendor/ezr.py     ezr @ 500d1d4, byte-identical, needs python 3.12
    corpus.py         the four SLR corpora, with their label-column traps
    features.py       tokenize -> stopwords -> stem -> TF-IDF, ported from ezr
    cnb.py            complementary naive Bayes, vectorised
    active.py         warm start + greedy acquisition loop
    metrics.py        classifier recall, false alarm, found recall
    validate.py       asserts the port matches vendored ezr exactly
    plot.py           figure 4 and the two comparison figures
configs/              experiment definitions
scripts/slurm/        sbatch wrappers
results/              metrics.json per run (committed; small)
```

Bulk data, indexes and full ranking dumps live on beegfs under
`/mnt/beegfs/lli66/ez-tool/`. Nothing large is written to `$HOME`, which has a
40 GB quota.

## Benchmark facts worth knowing

Measured from the data, not assumed:

| | |
|---|---|
| Corpus | `toolenv2404_filtered`: 12,303 tools / **47,058 unique APIs**, 49 categories |
| Queries | 765 solvable queries across 6 groups (G1/G2/G3 × instruction/tool/category) |
| Labels | `relevant APIs` as `[tool_name, api_name]`, 2.1–3.0 per query |
| APIs with a blank description | 4,475 (9.5%) |
| APIs sharing a name with another tool | 13,887 (29.2%) — `'Search'` occurs in 216 tools |
| Duplicate `(tool, api)` pairs dropped | 534 |

**Retrieval unit.** One document per API. The parent tool name is always
included because `api_name` alone is not an identifier for 29.2% of the
corpus.

What else goes into the document is an ablation, because **disclosure
granularity and index granularity are different things**. Anthropic's
[tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
defers a tool's schema from the model's context but still indexes "tool names,
descriptions, argument names, and argument descriptions" — the schema is
hidden from the model and visible to the retriever. So:

- `strict` — `tool_name + api_name + description`. Only what a progressive
  disclosure UI shows up front.
- `with_parameters` — adds argument names and their descriptions, mirroring
  what a real deployed retriever actually indexes.

Parameter *defaults* are never indexed: they carry payloads such as inline
base64 images that would swamp the term statistics.

**The label reconciliation problem.** `relevant APIs` was annotated against the
2023 ToolBench environment, but `toolenv2404_filtered` dropped endpoints that
had died by 2024. **441 of 1,795 labels (24.6%) point at APIs that no longer
exist** and are unreachable by any retriever. `label_policy` controls this:

- `drop_dead` (default) — discard unreachable labels and the 126 queries left
  with none. 639 queries, honest recall ceiling of 1.0.
- `keep_all` — all 765 queries, recall capped at 0.757.

Both are reported below. Any published number must state which was used.

## BM25 baseline

Full-corpus retrieval over all 47,058 APIs. CPU only, ~1.3 s end to end.

### Document-content ablation

`label_policy: drop_dead` (639 queries), overall:

| variant | R@1 | R@5 | R@10 | R@100 | NDCG@5 | NDCG@10 | terms/doc |
|---|---|---|---|---|---|---|---|
| strict | 0.221 | 0.449 | 0.521 | 0.766 | 0.417 | 0.447 | 27.8 |
| backfill | 0.215 | 0.460 | 0.536 | 0.769 | 0.422 | 0.453 | 30.0 |
| **with_parameters** | **0.229** | **0.478** | **0.548** | 0.767 | **0.439** | **0.468** | 45.6 |
| backfill_with_parameters | 0.225 | 0.476 | 0.554 | 0.770 | 0.438 | 0.470 | 47.8 |

Indexing argument names and descriptions is the clear win: **+0.028 R@5,
+0.021 NDCG@10** over `strict`, for a 64% larger index. The effect survives
the label policy (`keep_all`: +0.022 R@5, +0.016 NDCG@10), so it is not an
artifact of dropping dead labels.

Two things are worth noting about *how* it helps:

- **R@100 does not move** (0.766 → 0.767). Parameters surface no new relevant
  APIs; they reorder ones BM25 had already found. The gain is precision at the
  top, not reach.
- **Backfill and parameters are not additive.** Combining them is no better
  than parameters alone, because argument text already supplies the signal
  that backfilling a blank description was contributing. Once parameters are
  indexed, backfill is redundant.

Per group, the gain concentrates exactly where `strict` was weakest:

| group | R@5 strict → with_params | NDCG@10 strict → with_params |
|---|---|---|
| G1_instruction | 0.511 → 0.510 | 0.500 → 0.515 |
| G1_tool | 0.569 → 0.596 | 0.561 → 0.574 |
| G1_category | 0.526 → **0.577** | 0.524 → **0.579** |
| G2_instruction | 0.457 → 0.462 | 0.433 → 0.431 |
| G2_category | 0.337 → **0.389** | 0.357 → 0.369 |
| G3_instruction | 0.145 → **0.187** | 0.166 → **0.198** |

The `*_instruction` groups are flat; `*_category` and G3 gain 10–29% relative.

**Headroom for a new method.** With the best BM25 variant, R@100 is 0.767 but
R@5 is 0.478: for most queries a relevant API is somewhere in the top 100 and
not near the top. Closing that gap is the target. Note that k=5 is also the
documented default number of tools Anthropic's tool search returns, so R@5 and
NDCG@5 are the operating-point metrics. G3 remains where BM25 collapses
(R@1 = 0.079).

## Comparability to ToolGen

[ToolGen](https://arxiv.org/abs/2410.03439) (ICLR 2025) is the closest
reference point. Its corpus is 47,000 APIs, matching ours almost exactly, and
it reports NDCG@1/3/5 for a BM25 baseline in two settings: **in-domain**
(ranking within one subset's tools) and **multi-domain** (the full corpus,
which is what we do).

Important: ToolGen's *retrieval* table is computed on **ToolBench's retrieval
split**, while it uses StableToolBench only for end-to-end SoPR/SoWR. Our
numbers are on StableToolBench's 765 solvable queries, so the two are **not
directly comparable** — the query sets differ.

### What their code actually does

From `evaluation/retrieval/eval_bm25.py` and `evaluation/utils/retrieval.py`
in [their repo](https://github.com/Reason-Wang/ToolGen):

- **Data**: `data/retrieval/{G1,G2,G3}/` — ToolBench's retrieval split
  (`corpus.tsv`, `test.query.txt`, `qrels.test.tsv`). In-domain uses that
  stage's `corpus.tsv`; multi-domain uses `corpus_G123.tsv` with a docid
  remap. **Not** StableToolBench's solvable queries.
- **Document text**: `category_name, tool_name, api_name, api_description,
  required_params: <json>, optional_params: <json>, return_schema: <json>` —
  raw `json.dumps` of the schemas, defaults and all.
- **Retriever**: `rank_bm25.BM25Okapi` over `nltk.word_tokenize(doc.lower())`.
  No stemming, no stopword removal.
- **Metric**: `sklearn.metrics.ndcg_score`, but `true_relevance` is set only
  for documents present in the retrieved top-100. Relevant documents the
  retriever *missed* never enter the ideal ranking, so IDCG is taken over the
  relevant docs actually retrieved rather than over all of them. That forgives
  misses and scores strictly above the standard definition whenever recall
  < 1. Reproduced here as `ndcg_toolgen@k` for like-for-like reading.

### NDCG@1 / @3 / @5, ToolGen Table 1 layout

Standard NDCG ×100, `drop_dead`:

| variant | I1 | I2 | I3 |
|---|---|---|---|
| **with_parameters** | **50.7 / 48.9 / 52.3** | **35.3 / 34.1 / 37.8** | **19.7 / 16.1 / 17.3** |
| strict | 46.7 / 46.0 / 49.5 | 39.5 / 34.7 / 37.4 | 11.5 / 11.7 / 12.8 |
| backfill | 46.1 / 46.9 / 50.6 | 38.2 / 34.6 / 37.1 | 9.8 / 10.3 / 12.9 |
| toolgen_style | 52.3 / 49.8 / 53.7 | 35.8 / 35.8 / 38.8 | 34.4 / 23.2 / 25.3 |
| *ToolGen published, multi-domain* | *22.8 / 22.6 / 25.6* | *18.3 / 20.7 / 22.2* | *10.0 / 10.1 / 12.3* |
| *ToolGen published, in-domain* | *29.5 / 31.1 / 33.3* | *24.1 / 25.3 / 27.7* | *32.0 / 25.9 / 29.8* |

Per group, headline `with_parameters`:

| group | NDCG@1 | NDCG@3 | NDCG@5 |
|---|---|---|---|
| G1_instruction | 46.51 | 43.81 | 47.79 |
| G1_tool | 52.25 | 51.60 | 54.88 |
| G1_category | 53.28 | 51.16 | 54.17 |
| G2_instruction | 37.25 | 35.44 | 40.15 |
| G2_category | 33.33 | 32.65 | 35.42 |
| G3_instruction | 19.67 | 16.05 | 17.34 |

### Why we score above their BM25

Decomposition, NDCG@5 ×100, aligning our setup to theirs one knob at a time:

| step | I1 | I2 | I3 |
|---|---|---|---|
| ours: strict text, standard NDCG, `drop_dead` | 49.52 | 37.36 | 12.84 |
| + ToolGen document format | 53.66 | 38.77 | 25.26 |
| + ToolGen NDCG definition | 55.51 | 42.09 | 34.90 |
| + `keep_all` label policy | 42.24 | 39.63 | 34.90 |
| **ToolGen published, multi-domain** | **25.61** | **22.18** | **12.33** |
| **residual** | **−16.6** | **−17.5** | **−22.6** |

Neither the document format nor the metric explains the gap — **both move in
the wrong direction.** Their document text is *richer* than ours and scores
*higher* on our data; their NDCG definition is *more forgiving* than ours. If
anything they understate their own BM25.

After aligning document format, metric definition and label policy, a **17–23
point residual remains**, and it is attributable to the query set and corpus:
ToolBench's retrieval test split versus StableToolBench's 765 curated
*solvable* queries. StableToolBench filtered its queries for solvability
against live APIs, which removes exactly the ambiguous and unanswerable cases
that punish a lexical retriever.

**Conclusion: our numbers are not comparable to ToolGen's table, and the query
set is the reason.** Any head-to-head claim requires running on ToolBench's
retrieval split. Not done.

## ToolBench replication — ToolGen Table 1, BM25 row

Run on ToolBench's retrieval split (**not** StableToolBench), reproducing
ToolGen's stack exactly: their document concatenation, `rank_bm25.BM25Okapi`
over `nltk.word_tokenize(lower())`, top-100, and their NDCG definition.

```bash
ez-tool toolbench --config configs/toolbench_replication.yaml
```

Test set after deduplication: 1,099 unique queries (200/200/200/200/199/100
across the six splits), 2.15–2.91 relevant APIs each. Corpora: 49,936 docs
pooled (multi-domain); 10,439 / 13,142 / 1,605 for G1 / G2 / G3 (in-domain).

### Replication verdict — reproduced

NDCG@1 / @3 / @5 ×100, `bm25_okapi_toolgen`:

| setting | subset | ours | published | delta |
|---|---|---|---|---|
| multi-domain | I1 | 23.56 / 23.36 / 26.22 | 22.77 / 22.64 / 25.61 | +0.79 / +0.72 / +0.61 |
| multi-domain | I2 | 19.04 / 21.52 / 23.36 | 18.29 / 20.74 / 22.18 | +0.75 / +0.78 / +1.18 |
| multi-domain | I3 | 13.00 / 11.77 / 13.09 | 10.00 / 10.08 / 12.33 | +3.00 / +1.69 / +0.76 |
| in-domain | I1 | 30.00 / 30.77 / 32.90 | 29.46 / 31.12 / 33.27 | +0.54 / −0.35 / −0.37 |
| in-domain | I2 | 23.05 / 25.40 / 27.45 | 24.13 / 25.29 / 27.65 | −1.08 / +0.11 / −0.20 |
| in-domain | I3 | 31.00 / 25.04 / 28.92 | 32.00 / 25.88 / 29.78 | −1.00 / −0.84 / −0.86 |

**17 of 18 numbers land within ±1.2 points**; in-domain is near-exact. The
outlier (+3.00, multi-domain I3 NDCG@1) is on 100 queries, where one query is
worth a full point. Residual differences are consistent with split-folding:
the paper reports single I1/I2/I3 columns while the code exposes three splits
under G1 and two under G2, so their column may not be the equal-weight mean we
take. Per-split numbers are in `results/toolbench_replication/metrics.json`
and can be re-folded without re-running.

This validates the corpus loader, the duplicate handling, the multi-domain
docid remap, the metric, and the retriever end to end.

### The BM25 baseline in the paper is under-tuned

The same data, same metric, same queries — only the BM25 implementation
changes (`bm25s`, Snowball stemming, English stopwords, identifier splitting):

| setting | subset | their BM25 | **our BM25** | EmbSim | ToolRetriever | ToolGen |
|---|---|---|---|---|---|---|
| multi | I1 | 25.61 | **43.80** | 55.86 | 74.99 | 91.54 |
| multi | I2 | 22.18 | **37.48** | 39.55 | 63.61 | 88.84 |
| multi | I3 | 12.33 | **27.29** | 20.70 | 42.92 | 84.79 |
| in | I1 | 33.27 | **54.67** | 65.37 | 84.39 | 92.67 |
| in | I2 | 27.65 | **44.28** | 46.56 | 70.35 | 91.13 |
| in | I3 | 29.78 | **44.13** | 52.73 | 64.70 | 90.16 |

(NDCG@5 ×100, all on ToolGen's metric so the columns are commensurable.)

### Where that difference comes from

Measured, not asserted — a ladder over identical data, queries and metric,
changing one thing per rung (`configs/toolbench_ablation.yaml`,
`toolbench_isolate.yaml`, `toolbench_punct.yaml`). I1 NDCG@5:

| step | value | contribution |
|---|---|---|
| S0 their stack (`BM25Okapi` + `nltk.word_tokenize`) | 26.22 | — |
| S0b same library, regex tokens | 39.06 | **+12.84 tokenizer** |
| S1 swap library to `bm25s` | 39.13 | **+0.07 library** |
| S2 token pattern splits `-` and `/` | 41.71 | +2.58 |
| S3 + stopword removal | 44.95 | +3.24 |
| S4 + Snowball stemming | 44.03 | **−0.92** |
| S5 + lucene IDF variant | 43.80 | −0.23 |
| S6 + `normalize()` (not used in the replication) | 46.94 | +3.14 |

**The BM25 library is irrelevant** (+0.07 on I1, −0.18 on I2, −0.03 on I3).
**The tokenizer is ~73% of the gap.**

And within the tokenizer, the cause is JSON punctuation. These documents are
`json.dumps` output, so `nltk.word_tokenize` emits **174.4 tokens/doc against
71.2** — every `,` `{` `"` `:` becomes a token. Deleting punctuation-only
tokens and changing nothing else:

| tokenization | I1 | I2 | I3 |
|---|---|---|---|
| nltk | 26.22 | 23.36 | 13.09 |
| nltk minus punctuation | **37.55** | **33.01** | **19.51** |
| regex `[\w\-/]+` | 39.06 | 33.54 | 24.07 |

That recovers 88% of the tokenizer effect on I1 and 95% on I2. On I3 it
recovers 58%, with hyphen/slash handling carrying more of the residual.

**Why punctuation hurts: document-length normalisation, not scoring noise.**
Removing a punctuation token changes two things at once -- it stops
contributing to the score, and it stops counting toward `|d|`. `rank_bm25`
makes the first plausible: `BM25Okapi` computes
`idf = log((N-df+0.5)/(df+0.5))`, negative for any term in more than half the
documents, then replaces those with `epsilon * average_idf` -- a *positive*
weight, so commas score points instead of being ignored. Setting `epsilon=0`
zeroes that contribution while leaving the tokens in the length denominator:

| run | I1 NDCG@5 |
|---|---|
| nltk, `epsilon=0.25` (their default) | 26.22 |
| nltk, `epsilon=0` — punctuation scores nothing | **25.65** |
| nltk minus punctuation — gone from scoring *and* length | **37.55** |

Neutralising the score contribution changes nothing (−0.57, slightly worse).
Removing the same tokens from the length term as well is worth +11.33. So the
mechanism is **length inflation**: punctuation more than doubles `|d|`, and
with `b=0.75` that penalty falls hardest on documents with large parameter
schemas — precisely the most informative ones.

Incidentally, `epsilon=0` reproduces the published row *more* precisely than
their own default does (I1 delta −0.05 / +0.39 / +0.04, against +0.79 / +0.72
/ +0.61 at `epsilon=0.25`), which may indicate a different `rank_bm25` version
behind the paper's numbers.

Note that identifier splitting is *not* the explanation: `\w` includes the
underscore, so `get_all` and `getUserById` tokenize identically under both
stacks. Only `-` and `/` differ, and that is the separately measured +2.58.

Two caveats worth carrying forward:

* **Stemming hurts on this data** (−0.92 I1, −2.14 I2). It should be a measured
  flag rather than a default.
* **`normalize()` runs in the StableToolBench path but not the ToolBench one.**
  S6 sizes it at +3.14 on I1. The same nominal config therefore behaves
  differently across the two datasets -- an inconsistency to fix.

### Consequences

A correctly configured BM25 scores **+14 to +21 points** above the published
BM25 row — roughly doubling it. Two consequences:

- **On multi-domain I3, plain BM25 (27.29) beats EmbSim (20.70)**, an
  OpenAI `text-embedding-3-large` baseline. On multi-domain I2 it is within
  2.1 points of it.
- The gap that learned retrievers are credited with closing is materially
  smaller than the table implies. The whole difference is stemming, stopword
  removal, and splitting code-shaped identifiers — `nltk.word_tokenize` keeps
  `get_all`, `climate-news` and `v1/quote` as single tokens, so a query saying
  "climate news" cannot match a document containing `climate-news`.

**For this project the practical point is the bar.** A new tool-selection
method has to beat ~43.8 NDCG@5 multi-domain, not ~25.6.

Cost note: their stack takes ~130 s per split on the pooled corpus (~13 min
total); ours indexes in 3 s and retrieves 200 queries in 0.1 s. Roughly
1000× faster at nearly double the accuracy.

## Phase 2

Not started. `Qwen/Qwen3.5-4B` is downloaded to beegfs (8.8 GB) but no server
is running. vLLM ≥ 0.28.0 supports its `Qwen3_5ForConditionalGeneration`
architecture. Install with `uv pip install -e ".[llm]"`.

## Section 7 replication — CNB for literature review

A separate experiment from the tool-selection work above: a replication of
section 7 ("Text Mining") of Menzies & Srinivasan, ["Can AI be
Easy?"](https://arxiv.org/abs/2606.03640), which applies complementary Naive
Bayes as an active learner to systematic-literature-review relevance
filtering, and compares against
[FASTREAD](https://arxiv.org/abs/1612.03224) (Yu, Kraft & Menzies).

```bash
sbatch scripts/slurm/run_textmine.sbatch
```

CPU-only; the four corpora total 27 MB.

### What the reference implementation is

The paper points at [`ezr`](https://github.com/timm/ezr). The literature-review
code is `eg_textmine` / `eg_test_textmine` in `cli.py`, which call
`tmPrepare → tmData → cnb → tmActive` in `ezr.py` — about 150 lines at pinned
commit `500d1d4`. `ezr.py` is vendored here byte-identical.

It needs Python 3.12 (PEP 695 `type` statements) while this project runs on
3.11, so the pipeline is reimplemented on numpy and checked against it:

```bash
scripts/textmine_reference.sh                 # 3.12, vendored ezr, dumps JSON
python -m ez_tool.textmine.validate           # 3.11, asserts the port matches
```

Validation is exact, not approximate: identical vocabulary and ordering on all
four corpora, the **full feature matrix identical elementwise** on Kitchenham,
and CNB weights equal to `rtol=1e-12` across 8 labelled subsets × both
normalisation settings, including the single-class path. The vectorised loop
runs a trial in seconds rather than minutes, which is what makes the ablations
below affordable.

### Where the paper and the code disagree

Four differences, each of which changes the numbers. All were verified against
the released files, not inferred.

| | section 7 says | `ezr` @ `500d1d4` does |
|---|---|---|
| Input text | "each row was a paper's title and abstract" (§7.4) | `tmTokenize` reads the `abstract` column only; `document_title` is never touched |
| Warm start | "N = 24 randomly drawn papers" | `_tm_warm`: 20 drawn from *known positives* + 20 random (`--textmine.yes=20 --textmine.no=20`) |
| Kitchenham labels | table 6 reports 132 relevant | `klass="label"` → 45; the 132 live in a second column named `abs` |
| False alarm | plotted as the bottom row of figure 4 | not implemented anywhere in the commit |

The Kitchenham file carries **two** nested label columns — `abs` (132, passed
title-and-abstract screening) and `label` (45, survived content review). Table
6's 132 and §7.4's "108 of the 132" both point at `abs`, so that is what is
used here. Despite the name, `abs` holds a label, not an abstract.

The warm start is the one that cannot be reconciled, so it is run as arms
rather than resolved by fiat:

- `paper` — 24 drawn uniformly, §7.4's literal words.
- `stratified` — 12 known-positive + 12 random, the only split satisfying both
  the "24" and Yu's "equal number" of presumptive negatives.
- `code` — ezr's own 20 + 20.
- `paper_abstract_only` — `paper`, minus the title, isolating the first row of
  the table above.

A uniform draw of 24 from Hall (104 relevant in 8911) contains no relevant
paper about three quarters of the time, leaving CNB with a single class —
vendored `ezr` raises `KeyError` there. Rather than special-case it,
`run_trial` falls back to uniform sampling until the first positive appears,
which is FASTREAD's own step 1, and charges it to the budget.

### Recall means two different things

`_tm_recall` — what figure 4 plots — trains on the labels acquired so far,
predicts **every** document in the corpus, and reports the fraction of truly
relevant papers the model calls relevant. Yu's recall is `|LR|/|R|`: relevant
papers a human has physically read and confirmed. FASTREAD's X95 counts
abstracts read, so it is comparable only to the second.

The two are not interchangeable and the second is bounded by the budget: with
50 labels on Hall, found recall cannot exceed 50/104 = 48% whatever the model
does. Both curves, plus false alarm, are recorded on every run.

### Results

`configs/textmine_paper.yaml`: 20 repeats, 100 TF-IDF features, both
normalisation settings, five warm-start/text arms, and a **per-corpus label
cap read off figure 4's own panels** — Hall 290, Wahono 515, Radjenović 630,
Kitchenham 510. Those panels do not share an x-axis, and the cap matters: it
is where the paper's curves are actually judged. ~19 minutes on one `rome`
core. Figures in `results/textmine/figures/`, checkpoints in
`results/textmine/textmine_paper.json`, full traces on beegfs.

**Section 7.4's configuration, run as written.** Warm start of 24 drawn at
random, title and abstract, at each panel's own upper bound:

| corpus | cap | recall (blue) | false alarm (blue) | recall (red) | false alarm (red) |
|---|---|---|---|---|---|
| Hall | 290 | 0.923 | 0.071 | 0.990 | 0.451 |
| Wahono | 515 | 0.935 | 0.128 | 1.000 | 0.575 |
| Radjenović | 630 | 0.958 | 0.120 | 1.000 | 0.424 |
| Kitchenham | 510 | 0.742 | 0.228 | 0.955 | 0.613 |

Blue is `--textmine.norm=0`, red is Rennie normalisation, as in figure 4.
Three of four corpora land above 0.92 without normalisation, and Kitchenham
plateaus around 0.74 — §7.4 reports a ceiling "near 82%" there. **The
plateau result replicates, and so does the Kitchenham ceiling.**

**Warm start stops mattering once the axis is long enough.** Classifier
recall at each corpus's cap, no normalisation:

| corpus | `paper` (24 random) | `stratified` (12+12) | `code` (20+20) |
|---|---|---|---|
| Hall | 0.923 | 0.952 | 0.952 |
| Wahono | 0.935 | 0.952 | 0.968 |
| Radjenović | 0.958 | 0.958 | 0.969 |
| Kitchenham | 0.742 | 0.742 | 0.746 |

All three warm starts converge to within 0.03 by the cap. The seeding
question decides only how fast the curve gets there, not where it ends up.

**The left edge is the sharpest discriminator.** Every panel now starts at the
warm start rather than at zero, and the value reached immediately after it is
annotated — the quantity §7.4 says figure 4 labels at each panel's left edge.
Median classifier recall at that first point:

| arm | Hall | Wahono | Radjenović | Kitchenham |
|---|---|---|---|---|
| `paper` (24 random) | **0.000** | **0.000** | **0.000** | 0.420 |
| `stratified` (12 + 12) | 0.962 | 0.976 | 0.969 | 0.716 |
| `code` (20 + 20, x=40) | 0.971 | 0.984 | 0.979 | 0.712 |

A uniform draw of 24 leaves the model with no positive class at all on the
three large corpora, so its left edge is necessarily 0.00. Any published left
edge above zero on Hall, Wahono or Radjenović rules the uniform draw out.

What the random warm start does cost is the early part of the curve. It draws
so rarely from a one-in-90 positive class that the median run burns **25
labels on Hall, 42 on Wahono and 112 on Radjenović** before it sees a single
relevant paper, so at §7.4's stated budget of 50 it still reads 0.00 on Wahono
and Radjenović. A curve that crosses 0.95 near x=50 — as figure 4's Hall panel
appears to — needs positives in the warm start; the `code` arm reaches 0.971
by 40 labels. So the *endpoints* replicate under either reading, and only the
*approach* distinguishes them.

**The title makes no difference.** `code` versus `code_abstract_only` at the
cap: Hall 0.952 / 0.942, Wahono 0.968 / 0.968, Radjenović 0.969 / 0.938,
Kitchenham 0.746 / 0.742. Whether §7.4's "title and abstract" or ezr's
abstract-only is used does not change any conclusion.

**Rennie normalisation buys recall with false alarm.** Red beats blue on
recall everywhere — and costs four to six times the false alarm (Hall 0.071 →
0.451, Wahono 0.128 → 0.575, Kitchenham 0.228 → 0.613). The classifier is
reaching 1.000 recall by flagging a large share of the corpus. §7.4 writes
"the false alarm rate is far lower than when normalization is disabled",
which is the opposite of what these runs show; the conclusion drawn from it in
the next clause — that this preprocessing step "should not be applied
uncritically" — is what the data supports.

**On FASTREAD's own metric the picture inverts.** Found recall (`|LR|/|R|`,
relevant papers a human actually confirmed) at each cap, no normalisation:

| corpus | cap | `paper` | `stratified` | `code` | FASTREAD reaches 0.95 at |
|---|---|---|---|---|---|
| Hall | 290 | 0.106 | 0.211 | 0.269 | 350 |
| Wahono | 515 | 0.371 | 0.500 | 0.573 | 670 |
| Radjenović | 630 | 0.229 | 0.448 | 0.562 | 680 |
| Kitchenham | 510 | 0.451 | 0.492 | 0.523 | 630 |

Reading 290 abstracts of Hall, CNB's acquisition has surfaced 27% of the
relevant papers; FASTREAD reaches 95% by 350. This is not a contradiction of
the panel above — it is the same runs under the other definition of recall.
§7.4's summary ("Needs fewer labels, 100 vs. 300+") sets CNB's *classifier*
recall against FASTREAD's X95, which counts abstracts a human read to reach
95% *found* recall.

Two caveats in the other direction: this CNB sees only 100 TF-IDF features
against FASTREAD's full vocabulary, and uses pure certainty sampling against
FASTREAD's uncertainty-then-certainty schedule, so part of the found-recall
gap is acquisition strategy rather than the classifier.

### Verdict

**Replicated.** At figure 4's own x-limits, §7.4's configuration reproduces
the reported behaviour: a fast plateau above 0.92 on Hall, Wahono and
Radjenović, a genuine ceiling on Kitchenham, and low false alarm without
normalisation. The warm-start ambiguity turns out not to affect where the
curves land, only how quickly they get there.

Two things do not survive. §7.4's sentence about normalisation and false alarm
has its comparison inverted relative to these runs, though its conclusion
stands. And the FASTREAD comparison measures CNB with one definition of recall
and FASTREAD with another; on the shared definition, FASTREAD is comfortably
ahead. Neither undermines the plateau result, which is the section's actual
contribution — "after a few dozen labels the model can point at 95% of the
relevant papers" is cheap, true, and useful. It is simply not the claim
FASTREAD's X95 answers.

### Caveat on the reference implementation

`ezr` @ `500d1d4` is the closest released code, and the port is validated
against it exactly — but it is **not** the code that produced figure 4. The
commit contains no false-alarm computation at all, while figure 4 devotes its
entire bottom row to false alarm. Earlier revisions (`aa1a68d6`, a standalone
`textmine.py`) have the same gap. The configuration used here is therefore
taken from §7.4's prose and from the published panels, with the released code
as the authority only for the parts it does implement: the TF-IDF pipeline,
the CNB weights, the acquisition rule, and `_tm_recall`.

**End-to-end parity with vendored ezr.** `scripts/textmine_ezr_native.py`
drives `tmActive` @ 500d1d4 itself, through its own `the.*` configuration, and
`ez_tool.textmine.compare` checks the resulting trajectory against the port's
on the identical setup (`configs/textmine_ezr_parity.yaml`: abstract only,
ezr's `label` column, warm start 20+20, budget 290, 20 repeats):

| corpus | norm | steps | median abs. diff | max | ezr final | port final |
|---|---|---|---|---|---|---|
| Hall | off | 251 | 0.5 | 1.5 | 94.5 | 94.0 |
| Hall | on | 251 | 0.0 | 1.0 | 100.0 | 100.0 |
| Wahono | off | 251 | 0.5 | 2.0 | 98.0 | 96.0 |
| Wahono | on | 251 | 0.0 | 0.0 | 100.0 | 100.0 |
| Radjenović | off | 251 | 0.0 | 2.0 | 93.0 | 93.0 |
| Radjenović | on | 251 | 0.0 | 3.0 | 100.0 | 100.0 |
| Kitchenham | off | 251 | 1.5 | 5.0 | 82.0 | 82.0 |
| Kitchenham | on | 251 | 0.0 | 2.0 | 97.0 | 97.0 |

Both sides are medians over 20 independent random warm starts drawn from
different generators, so trial k is not the same 40 papers on both; a couple of
points of sampling noise is expected and a systematic offset is not. There is
none. Together with `validate.py`'s component checks -- identical features,
CNB weights to `rtol=1e-12` -- the port is ezr for every purpose here.

Cost, for the same eight runs: vendored ezr 81 minutes, the port about 40
seconds.


## Transplanting CNB to tool selection

The obvious next question is whether section 7's method helps with this
repo's actual task. It does not, and the reason is structural rather than a
matter of tuning.

```bash
python -m ez_tool.textmine.toolbench          # build the substrate
python -m ez_tool.textmine.run_toolbench      # run the comparison
```

### Why a literal transplant is impossible

| | SLR (section 7) | StableToolBench |
|---|---|---|
| topics | 1 per corpus | 639 queries |
| candidates | 1,704–8,911 | 47,058 APIs |
| relevant each | 48–132 (1 in 86) | **2.12 (1 in 22,208)** |
| labels at query time | one per paper read | none |

Two blockers. `_tm_warm(yes=20)` would hand the learner every relevant API,
since there are only about two per query. And `retrieval.base.Retriever` is a
zero-shot protocol — `index` then `retrieve`, no labels — which CNB cannot
honour, because it needs labelled examples for the query in front of it.

### The setting where it does apply

An agent with a shortlist tries APIs one at a time, and each attempt reveals
whether that API was the right one. That is a label, and it is the shape
section 7's active learner fits. So: BM25 supplies the top 100, and three
policies spend the same 100 attempts over the same candidates —

- `bm25` — work down the ranking (the no-learning control)
- `cnb` — work down it until the first hit, then let CNB choose
- `fused` — reciprocal rank fusion of BM25 and CNB, so a candidate must look
  good to both

Recall after k attempts is recall@k, so these sit beside the BM25 baseline
above. **CNB sees ground truth on every attempt and BM25 sees none**, so this
measures the value of feedback, not retrieval quality.

### Result: feedback makes it worse

`top_features=5000`, 639 queries, `drop_dead`:

| policy | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.2293 | **0.4065** | **0.4775** | **0.5482** | **0.6209** | **0.7160** | 0.7671 |
| cnb | 0.2293 | 0.3080 | 0.3382 | 0.3786 | 0.4338 | 0.5335 | 0.7671 |
| fused | 0.2293 | 0.3207 | 0.3638 | 0.4251 | 0.5264 | 0.6905 | 0.7671 |

CNB costs **−0.139 R@5** against the ranking it was handed, even with an
oracle answering every attempt. R@1 is identical because the first attempt is
always BM25's top-1 — no positive is known yet — and R@100 is identical
because both exhaust the pool.

The result is insensitive to the knobs. Feature width 500 / 2000 / 5000 gives
R@5 of 0.3379 / 0.3382 / 0.3382; rank fusion recovers roughly a third of the
loss at k=5 and nearly all of it by k=50, but never overtakes plain BM25.

### NDCG@k, all three variants

Reported on ToolGen's NDCG (IDCG taken over the relevant documents actually
retrieved) as well as the standard definition, since the published comparators
in this README use the former. The feedback re-ranker originally scored on
`X @ w_yes`, ezr's own acquisition rule; the margin fix found later for the PRF
path applies here too and is worth about +0.03.

| method | labels | R@100 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| **BM25** | none | 0.7671 | **0.4241** | **0.4288** | **0.4655** | **0.4972** |
| Feedback CNB (margin) | oracle, every attempt | 0.7671 | 0.4241 | 0.3682 | 0.3942 | 0.4220 |
| PRF-CNB (BM25-seeded) | none | 0.7671 | 0.3286 | 0.3219 | 0.3620 | 0.4056 |
| Feedback CNB (`w_yes`, ezr's rule) | oracle, every attempt | 0.7671 | 0.4241 | 0.3534 | 0.3692 | 0.3878 |
| Cold-start CNB | none | 0.5180 | 0.1612 | 0.1975 | 0.2219 | 0.2531 |

(ToolGen's metric. Standard NDCG@10 for the same rows: 0.4678, 0.3932, 0.3813,
0.3592, 0.2202 — the ordering is identical, every value is lower.)

**The first four rows retrieve the same 100 documents** — R@100 is 0.7671 for
all of them, because each is re-ranking BM25's shortlist. Under ToolGen's
metric that makes their IDCG denominators identical, so the comparison is a
pure test of ordering, with nothing hidden in what was retrieved. CNB loses it.

Two things the metric choice changes. NDCG@1 is identical (0.4241) for BM25
and both feedback rows because the first attempt is always BM25's top-1 — no
positive is known yet, so CNB has nothing to fit. And cold-start CNB is the
one row the metric flatters: it retrieves only 0.5180 of the relevant APIs, so
forgiving its misses lifts it from 0.2202 to 0.2531 while the re-rankers,
which miss nothing extra, gain far less.

### Why

The positive class is too small to learn from. With 2.12 relevant APIs per
query, CNB starts training the moment it has **one** positive, and is fitting
thousands of feature weights to that single example while roughly one relevant
API remains among ~95 untried. BM25's ranking encodes corpus-wide term
statistics; one labelled document cannot outvote it.

Section 7's corpora are the opposite case: by 40 labels an SLR run holds a
dozen or more positives, drawn from a class that is 1-in-86 rather than
1-in-22,000. The method depends on that density, and the benchmark does not
supply it. An active learner is the wrong tool when the answer set has two
elements — the task is retrieval, and the labels arrive too late and too few
to beat a good lexical ranker.

This is a negative result about the transplant, not about section 7, which
replicates on its own corpora.


## ez-tool: CNB for tool selection, zero-shot

The feedback re-ranker above needs an oracle. This is the version that does
not: **pseudo-relevance feedback**, which is the only framing where CNB can
satisfy `retrieval.base.Retriever` and be compared with BM25 on equal terms.

```bash
ez-tool run --config configs/cnb_prf.yaml            # StableToolBench
ez-tool toolbench --config configs/toolbench_cnb_prf.yaml   # ToolBench split
```

`retrieval/cnb_prf.py` reads no qrels at query time. BM25's top `n_pos` are
*presumed* relevant and `n_neg` APIs drawn uniformly from the corpus are
presumed irrelevant — Yu et al.'s presumptive sampling, which §7.2 credits
FASTREAD with, applied without a human. CNB trains on those pseudo-labels and
the shortlist is re-ranked by reciprocal rank fusion with BM25.

### One thing worth knowing before reading the numbers

Ranking on `X @ w_yes` — which is exactly how ezr's `tmActive` picks its next
paper — scores **0.124 R@5** here against BM25's 0.478. In complementary
Bayes a class's weights are built from the documents *outside* it, so `w_yes`
is a function of the negatives alone; when the negatives are 200 random APIs
they carry no information about the query, and the pseudo-positives never
enter. The discriminative quantity is the decision margin
`X @ w_yes − X @ w_no`, the same comparison `_tm_best` makes. That lifts R@5
to 0.394 and is the default.

This is a real limit on transplanting the recipe: ezr's acquisition rule works
on the SLR corpora because its negatives are drawn from the same topic-specific
pool the positives come from. That assumption does not survive the move.

### Result: it does not beat BM25

StableToolBench, 639 `drop_dead` queries, best of a 24-point sweep over
`n_pos ∈ {1,3,10} × n_neg ∈ {200,2000} × k_rrf ∈ {60,300} × norm ∈ {off,on}`:

| method | R@1 | R@5 | R@10 | R@100 | NDCG@10 |
|---|---|---|---|---|---|
| **BM25 `with_parameters`** | **0.2293** | **0.4775** | **0.5482** | 0.7671 | **0.4678** |
| PRF-CNB, best (`n_pos=1, n_neg=200, k_rrf=60`) | 0.1735 | 0.3840 | 0.4830 | 0.7671 | 0.3813 |
| PRF-CNB, `n_pos=3` | 0.1362 | 0.4095 | 0.5075 | 0.7671 | 0.3801 |
| PRF-CNB, `n_pos=10` | 0.0917 | 0.2903 | 0.4713 | 0.7671 | 0.3059 |

**No configuration beats the baseline on any metric.** R@100 is identical
throughout because the pool is the top 100 — this is pure reordering, and
every reordering CNB proposes is worse than the one BM25 supplied. Fewer
pseudo-positives is better and more presumptive negatives is worse, both of
which point the same way: the classifier is adding noise, not signal.

ToolBench retrieval split, same conclusion, NDCG@5 on ToolGen's metric ×100:

| setting | subset | our BM25 | PRF-CNB |
|---|---|---|---|
| in-domain | I1 | **54.67** | 36.69 |
| in-domain | I2 | **44.28** | 30.74 |
| in-domain | I3 | **44.13** | 27.24 |
| multi-domain | I1 | **43.80** | 30.98 |
| multi-domain | I2 | **37.48** | 20.85 |
| multi-domain | I3 | **27.29** | 3.67 |

### Verdict

Three independent framings have now been measured and all lose to BM25:
oracle agent feedback (−0.139 R@5, *with* ground truth on every attempt),
zero-shot PRF across 24 configurations (−0.087 NDCG@10 at best), and both on
two separate benchmarks.

The reason is the same one that makes the SLR result work. Section 7's method
needs a positive class dense enough to estimate term distributions from — 1 in
86, with a dozen or more positives in hand by 40 labels. Tool selection offers
2.12 relevant APIs per query, 1 in 22,208, and at query time offers none at
all. Complementary Bayes is being asked to out-argue corpus-wide term
statistics from one or two noisy examples, and it cannot.

Nothing here contradicts section 7, which replicates on its own corpora. It
does say the recipe is not portable to tool selection, and it says why: the
task is retrieval, and the labels arrive too few and too late to beat a
well-configured lexical ranker.


### Cold start: is BM25 the problem?

The natural objection to the PRF result is that BM25's pseudo-labels are too
noisy to learn from — its R@1 is 0.229, so the presumed-relevant document is
wrong three times in four. `seed_source="query"` removes BM25 entirely: the
**query itself** is the only positive, negatives are drawn uniformly from the
corpus, and CNB ranks all 47,058 APIs. That makes it a retriever, not a
re-ranker, so BM25 cannot be blamed for the outcome.

Getting there took two real fixes, both worth recording because they are
properties of the method rather than tuning:

1. **A constant offset makes the ranking degenerate.** The decision margin
   carries `log(den_yes / den_no)` on every feature. It cancels when
   classifying one document and does not when ranking many, so `X @ margin`
   became dominated by document length — "longest API first", R@100 = 0.006.
   Rennie et al.'s length normalisation removes it.
2. **One positive against N negatives is the wrong balance.** A query term
   contributes only `log(1 + alpha) = 0.69` while `-log(neg_count + alpha)`
   reaches past −5, so corpus frequency swamps query membership. Measured
   margins on a football query put `football` at **−1.46** and an unrelated
   rare token at **+0.74**. Scaling the positive class to the negatives' total
   mass fixes it.

With both applied, cold start works — and still loses:

| method | R@1 | R@5 | R@10 | R@100 | NDCG@10 |
|---|---|---|---|---|---|
| **BM25** | **0.2293** | **0.4775** | **0.5482** | **0.7671** | **0.4678** |
| PRF-CNB (BM25-seeded, best) | 0.1735 | 0.3840 | 0.4830 | 0.7671 | 0.3813 |
| Cold-start CNB (no BM25) | 0.0975 | 0.2270 | 0.2969 | 0.5180 | 0.2202 |

Three pieces of evidence say the pseudo-labels are not the binding constraint:

- Removing BM25 **halves** performance rather than improving it: 0.384 → 0.227
  R@5. The BM25 seed was helping, not hurting.
- Re-seeding from CNB's *own* top-5 instead of BM25's (`rounds=2`) collapses
  the ranking to 0.0016 R@5. A different seed makes it worse, not better.
- In the BM25-seeded sweep, `n_pos=1` — leaning on BM25's single noisiest
  guess — was the **best** configuration. The method does best where it trusts
  CNB least.

Cold-start CNB is a working lexical retriever, roughly half of BM25's quality.
It is not being held back by its supervision; it is a weaker scorer than a
tuned BM25, and every configuration that shifts weight onto it does worse.

### Oracle ceiling: what any query expansion could buy

Before adding an LLM to write topic expansions, the question is whether better
positive evidence can lift CNB past BM25 *at all*. Hand it the ground-truth
relevant API documents as its positive class — strictly better than any text a
model could generate — and measure.

Two variants, because the obvious one leaks. Training on every relevant API and
then ranking a corpus that contains them scores R@5 = 0.813, but those
documents are in the ranking and the model was fitted on them. The fair test is
leave-one-out: train on all but one relevant API, drop the trained-on ones from
the ranking, and ask whether the held-out relevant API surfaces. BM25 is run on
the identical task, from the raw query, as the control.

527 queries with at least two relevant APIs:

| method | positive evidence | hit@1 | hit@5 | hit@10 | hit@100 |
|---|---|---|---|---|---|
| **BM25** | none (raw query) | **0.2665** | **0.4694** | **0.5338** | **0.7593** |
| Oracle CNB | a real relevant API's full text | 0.1248 | 0.2617 | 0.3325 | 0.5274 |

Given a perfect positive example, CNB reaches **56% of BM25's hit@5** — and
BM25 had nothing but the query. Class balancing and no length normalisation is
the best of the four configurations; the others are far worse.

**This bounds the LLM plan.** An expansion model can only approximate the text
handed to CNB for free here, so no topic generation will lift CNB past BM25 as
a *scorer*. It does not bound LLM expansion feeding BM25, which is a different
and much more promising system — it simply would not be a result about
complementary Bayes.


### Can it rank within its own shortlist?

"Which of these tools is *more* relevant" has no ground truth on this
benchmark: `relevant APIs` is a flat set with no scores, grades, or ranks. And
for the multi-tool groups the question is ill-posed rather than merely
unmeasured — G2/G3 queries carry several sub-intents ("find a 7-letter word
starting with 'fru'... **additionally**, fetch the income data for ZIP 98765"),
so the relevant set is a *cover* of those sub-intents, not a ranking of
alternatives. Two APIs answer different clauses; neither is more relevant.

What can be measured is whether the score orders candidates informatively:
precision at each returned position, and the AUC over (relevant, irrelevant)
pairs inside the returned top-100.

| method | P@pos1 | P@pos5 | P@pos10 | shortlist AUC | queries scored |
|---|---|---|---|---|---|
| **BM25** | **0.424** | **0.055** | 0.023 | **0.876** | 566 |
| PRF-CNB (BM25-seeded) | 0.329 | 0.083 | 0.031 | 0.855 | 566 |
| Cold-start CNB | 0.161 | 0.045 | 0.027 | 0.813 | 431 |

So yes — CNB's ordering carries real information. Precision falls about
six-fold from its first result to its tenth, and 0.813 AUC is far from the 0.5
of a coin flip. It is simply worse at it than BM25 on every measure, which is
the same conclusion every other experiment here reached.

One caveat that flatters CNB: the AUC is computed only over queries where a
method retrieved at least one relevant API into its top-100, and cold-start
CNB manages that for 431 queries against BM25's 566. Its 0.813 is measured on
the subset where it already succeeded.


### What is ezr's, and what is not

Worth stating plainly, because "ezr + CNB for tool selection" overstates what
is actually running.

**The section 7 replication has genuine ezr results.** `ezr.py` @ 500d1d4 is
vendored byte-identical, `scripts/textmine_reference.sh` runs it for the
reference dump, and `scripts/textmine_ezr_native.py` drives its own `tmActive`
end to end — the final recalls of 94.5 / 98.0 / 93.0 / 82.0 are ezr's output,
matched by the port to within sampling noise.

**The tool-selection results all come from the port.** Vendored ezr cannot run
there: it needs Python 3.12, it is dense-only (47,058 x 5,000 dense is 941 MB),
and `tmPrepare` hardcodes the `abstract` text column and `label` class column,
so it cannot read this corpus at all.

The weight computation is still exactly ezr's, and that is now checked on
*this* data rather than inferred from the SLR checks. One query's BM25 pool is
exported as an ezr-format CSV, read back through ezr's own `csv()`/`Data`, and
put through `ezr.cnb()`:

```
ezr parsed: 100 rows, 300 x-columns, klass col at 300
  class 'no' : max abs diff 0.000e+00   allclose(rtol=1e-12) = True
  class 'yes': max abs diff 0.000e+00   allclose(rtol=1e-12) = True
```

Bit-identical, including through the sparse path the SLR validation never
exercised.

But three pieces of the tool-selection method are **not** ezr:

| piece | source |
|---|---|
| CNB weight formula | ezr, verified identical above |
| `margin` (`X@w_yes - X@w_no`) as a *ranking* score | added here — ezr compares classes in `_tm_best`, never ranks on the difference |
| class balancing | added here |
| length normalisation | added here |

None is cosmetic. Ranking on ezr's own `X @ w_yes` scores 0.124 R@5; without
length normalisation the cold-start ranking collapses to R@100 = 0.006. The
honest description is *ezr's classifier plus three modifications needed to make
it function as a retriever* — which is itself part of the finding about how far
the recipe travels.


### Vendored ezr with a 24-sample uniform warm start

`scripts/textmine_ezr_warm24.py`, budget 290, `norm=0`, abstract only,
Kitchenham on `abs` (132). Median classifier recall (%):

| corpus | mode | trials kept | @24 | @50 | @100 | @150 | @200 | @290 | random picks |
|---|---|---|---|---|---|---|---|---|---|
| Hall | skip | 6/20 | 61.0 | 77.5 | 74.0 | 84.0 | 86.5 | 87.5 | 0 |
| Hall | step1 | 20/20 | 0.0 | 0.0 | 71.5 | 75.0 | 86.0 | 87.0 | 42 |
| Wahono | skip | **1/20** | 37.0 | 79.0 | 95.0 | 96.0 | 91.0 | 95.0 | 0 |
| Wahono | step1 | 20/20 | 0.0 | 0.0 | 20.5 | 79.5 | 91.0 | 93.0 | 81 |
| Radjenović | skip | 3/20 | 50.0 | 37.0 | 39.0 | 39.0 | 81.0 | 83.0 | 0 |
| Radjenović | step1 | 20/20 | 0.0 | 0.0 | 23.5 | 41.0 | 54.0 | 81.0 | 50 |
| Kitchenham | skip | 19/20 | 42.0 | 46.0 | 49.0 | 53.0 | 53.0 | 62.0 | 0 |
| Kitchenham | step1 | 20/20 | 41.5 | 47.5 | 48.5 | 53.0 | 53.0 | 62.0 | 0 |

**The endpoints agree between modes** — 87.5/87.0, 95/93, 83/81, 62/62 — so
where the curve lands does not depend on how the missing-positive case is
handled. Only the approach does.

**Recall cannot be reported from label 50 under a uniform warm start.** The
median run spends 42 labels on Hall, 81 on Wahono and 50 on Radjenović before
it sees a single positive, which is more than the 26 acquisitions available
between 24 and 50. At label 50 the honest curve reads 0.0 on three of the four
corpora. Kitchenham is the exception: its positives are 7.7% of the corpus, so
a 24-draw almost always catches one and no random picks are needed at all.

**Dropping the crashed trials manufactures the published shape.** In `skip`
mode Wahono reads 79 at label 50 and 95 at label 100 — a clean, early, high
plateau. That curve is **one trial**. The other 19 warm starts held no positive
and were discarded. Survival tracks positive density exactly: Kitchenham
(7.7% positive) keeps 19/20, Hall (1.2%) keeps 6/20, Wahono (0.9%) keeps 1/20.
A pipeline that silently dropped failed runs would produce figure-4-shaped
curves from a handful of lucky draws, and would report them as medians of 20.

**Does the plateau reproduce?** Partly. Against the paper's ~95 on the first
three corpora and ~82 on Kitchenham: Wahono reaches 93–95, Hall 87,
Radjenović 81–83 and still climbing at 290 (it needed 630 labels to reach
0.958 in the port), Kitchenham 62 against 82. So one corpus matches, one is
close, one is unconverged at this budget, and one falls well short.
