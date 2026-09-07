# ToolBench feasibility diagnostics for a CNB tool retriever

> **Partly superseded by [REPORT_FOLLOWUP.md](REPORT_FOLLOWUP.md).** A second
> round re-measured Diagnostic 2 against size-matched hard negatives and
> **downgraded** this report's cross-cutting "NO-GO on any usability claim":
> the 53.7% figure below reflects pool size and cross-domain topicality more
> than copied wording. The tool-level GO and its 5.1% bound are **confirmed**;
> the API-level NO-GO is **strengthened**. The follow-up also found train/test
> near-duplication affecting 74.5% of the evaluation set. Read that report's
> revised go/no-go table as authoritative.

**Question.** Is a complement naive Bayes retriever in the style of EZR §7 viable
on ToolBench G1, before any model is built?

**Answer.** Yes at the tool and category levels, no at the API level — but the
reason for the API-level "no" is not the one the task anticipated, and one of the
three threats turns out to be fatal to the *claim* rather than to the *method*.

| | verdict |
|---|---|
| **Diagnostic 1** — class count | Support is adequate at all three levels. The API level fails for a different reason found in Step 0: **94.3% of G1 instances carry 2+ ground-truth APIs**, so the label is a set and A3's single-class assumption does not hold there. |
| **Diagnostic 2** — reverse generation | **Confounded.** A parameter-free Jaccard ranker puts ground truth in the top 5 for **53.7%** of instances out of 10,355 candidates (chance: 0.05%). |
| **Diagnostic 3** — query sparsity | **Not a problem.** 0.04% of queries have zero features at EZR's default `top=100`, far below the ~10% threshold. The tokenizer, however, is separately broken on tool-side identifiers. |

Everything below is computed from files on disk. Inferences are labelled
**[inferred]**; everything unlabelled is measured.

Reproduce with `python scripts/run_diagnostics.py` on a compute node; raw
numbers in [`diagnostics.json`](diagnostics.json).

---

## Step 0 — ground truth about the data itself

### Files and release

| item | value |
|---|---|
| G1 instances | `/mnt/beegfs/lli66/ez-tool/data/raw/toolgen_data/data/retrieval/G1/{train,test_G1_instruction,test_G1_tool,test_G1_category}.json` |
| release | HuggingFace `reasonwang/ToolGen-Datasets`, `data.tar.gz`, blob `f80aa7422a39bc69b1d7c963f064fff8d94988f9dd153b12790087749cc2e7e0`, commit `9823c75ee3eabc790c11c1df924a084fe2fb5e53`, files dated 2024-10-09 |
| cross-checked against | `THUNLP-MT/StableToolBench` @ `aa4ed9f4737ad98bd706663f01d63623c3427812`; `Reason-Wang/ToolGen` @ `6839374a255810efe69deea4056eec5c55e25802` |

### The instance-count discrepancy

Neither published figure appears on disk. Counted directly:

| split | train | test | total |
|---|---|---|---|
| **G1** | 88,395 | 600 | **88,995** |
| G2 | 86,670 | 400 | 87,070 |
| G3 | 25,609 | 100 | 25,709 |
| **all** | 200,674 | 1,100 | **201,774** |

StableToolBench's `solvable_queries/test_instruction/` holds **765** queries
across all six groups (G1: 474). That is the evaluation set, not an instance
count.

So: 88,995 G1 instances on disk; 201,774 across G1–G3. **Neither 126,486 nor
12,657 is reproducible from these files**, and the 12,657 figure is smaller than
even one group's train split.

**[inferred]** The three numbers most likely count different things — this
release is ToolBench's *retrieval* split (one row per instruction, with its
sampled candidate API pool), whereas 126,486 is usually quoted for
(instruction, solution-path) pairs and 12,657 for a filtered subset that
survived solution-path generation. I could not verify this: ToolBench's own
`data/instruction/G*_query.json` and answer files are **not** on disk, so the
reconciliation is a blocker, not a finding. Every number in this report is from
the 88,995 G1 instances above.

### Label extraction

The label is the **`relevant APIs`** field of each record: a list of
`[tool_name, api_name]` pairs. Documentation comes from the same record's
`api_list` (the candidate pool the generator was shown); its union over all G1
instances is 10,355 distinct APIs.

**Instances dropped: 0.** Every one of the 88,995 records had a non-empty query,
a query id, and a well-formed `relevant APIs` list, and every ground-truth pair
resolved against the `api_list` union with **exact string matching** — no fuzzy
matching was needed. Case/whitespace folding merged only 2 near-duplicate API
keys (10,357 → 10,355). Query ids are unique; no train/test leakage by id.

Two data-quality notes, not drops: 1,579 instances (1.8%) repeat an earlier
query's exact text under a new id, and at least one "query" is a stringified
Python list (`"['get all the desserts', 'get all the beverages']"`).

### Ground-truth class counts

| level | distinct classes appearing as G1 ground truth |
|---|---|
| category | **49** |
| tool | **3,225** |
| API | **10,056** (of 10,355 in the pool) |

### The finding that reframes the task

`relevant APIs` is a **list**, and in G1 it is usually not a singleton:

| ground-truth APIs per instance | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|
| instances | 5,064 | 65,280 | 14,089 | 3,300 | 1,262 |
| share | **5.7%** | 73.4% | 15.8% | 3.7% | 1.4% |

But those APIs are always drawn from **one tool**:

* distinct tools per instance: **1, for all 88,995 instances (100%)**
* distinct categories per instance: 1 for 88,689 (99.66%); 2 for 306, caused by
  41 tool names that appear under more than one category

So "G1 = single-tool" means single *tool*, not single *API*. **A3's "class = the
tool that should be called" is exactly well-posed at the tool level and
essentially well-posed at the category level. At the API level the label is a
set for 94.3% of instances, and A3 as specified does not apply there** — that is
a scope error in the design, independent of anything CNB does.

---

## Diagnostic 1 — label distribution and effective class count

> **Verdict: support is adequate at every level by the median rule; but at tool
> level 47% of classes sit at exactly 3 instances, and the API level is ruled
> out by the multi-label finding above, not by support.**

Counts are over label *mentions* — a class's support is the number of instances
listing it — which is the sample a per-class CNB estimator actually sees.

| | category | tool | API |
|---|---|---|---|
| classes | 49 | 3,225 | 10,056 |
| **effective classes (exp H)** | **28.6** | **1,859.3** | **6,595.3** |
| min | 18 | 1 | 1 |
| 25th | 448 | 3 | 4 |
| **median** | **1,209** | **12** | **13** |
| 75th | 2,158 | 50 | 27 |
| max | 12,514 | 395 | 218 |
| classes with 1 | 0 | 10 | 205 |
| classes with ≤3 | 0 | 1,528 (47.4%) | 2,261 (22.5%) |
| classes with ≤5 | 0 | 1,536 (47.6%) | 2,816 (28.0%) |
| classes with ≤10 | 0 | 1,593 | 4,315 |
| top-10 share of all labels | 58.6% | 1.8% | 0.7% |

Figures: [category](figures/d1_support_category.png) ·
[tool](figures/d1_support_tool.png) · [API](figures/d1_support_api.png)

**The 49-vs-effective-count check comes out the other way.** The category level's
effective class count is 28.6, not 8 — the distribution is skewed (the top 10 of
49 categories hold 58.6% of labels, `Data` alone 12,514) but not collapsed. With
~29 effective classes at a median of 1,209 instances each, the category level
sits **inside** Rennie's original regime (~20 classes, ~1,000 documents each),
not outside it. The "K=49 is safe" argument survives.

**The tool level is bimodal, and the median hides it.** The histogram has a spike
of **1,511 tools with exactly 3 instances** — 47% of the label space — then a gap,
then a broad mode around 30–90. **[inferred]** the spike is a generation
artifact: ToolBench appears to have sampled a fixed 3 queries for most tools.

The mitigating number: those 1,528 starved classes hold only **5.1% of all
labels**. So the failure is bounded — CNB's weight vectors will be
indistinguishable for about half the classes, but those classes account for
roughly one query in twenty. Class-weighted and instance-weighted views of this
data disagree, and both belong in any downstream error analysis.

**Decision rule applied.** Median instances-per-class is 1,209 / 12 / 13 — all
above ~5. By the stated rule, **no level is excluded on support grounds**. The
API level is nonetheless excluded, by Step 0's multi-label finding.

---

## Diagnostic 2 — lexical leakage from reverse generation

> **Verdict: confounded. A zero-parameter word-overlap ranker puts ground truth
> in the top 5 for 53.7% of instances out of 10,355 candidates — 1,100× chance.
> No usability claim can be made on this data.**

Both sides go through the same tokenizer (EZR's pipeline with identifier
splitting on). API documentation = `api_name` + `api_description` + every
required and optional parameter's **name and description**. `tool_name` and
`category_name` are excluded — they are label components at the coarser levels,
and including them would let a label leak into its own feature vector.
Parameter **defaults are never indexed** (base64 payloads). For instances with
several ground-truth APIs, the best-matching one is used.

### Jaccard overlap

| | mean | p5 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|---|
| ground-truth API | **0.163** | 0.042 | 0.103 | 0.154 | 0.211 | 0.318 |
| random non-relevant API (10 draws) | **0.013** | 0.000 | 0.006 | 0.011 | 0.018 | 0.032 |

**Ratio of means: 12.4×.** Per-instance ratio: median 13.1, quartiles 7.2–25.7
(undefined for 4,576 instances where the random overlap was exactly 0).
The two distributions barely touch —
[figures/d2_overlap_distribution.png](figures/d2_overlap_distribution.png).

### The parameter-free overlap ranker

Ranking all 10,355 G1 APIs. Integer overlap scores tie heavily, so both the
optimistic bound (ties resolved in ground truth's favour) and the expectation
under random tie-breaking are reported.

| ranker | median rank | top-1 | top-5 | top-10 |
|---|---|---|---|---|
| raw `\|Q ∩ D\|`, random tie-break | 62 | 6.4% | 18.4% | 25.0% |
| raw `\|Q ∩ D\|`, optimistic | 30 | 13.3% | 26.3% | 34.5% |
| **Jaccard (length-normalised)** | **4** | **30.5%** | **53.7%** | **61.6%** |
| chance | 5,178 | 0.01% | 0.05% | 0.10% |

[figures/d2_rank_cdf.png](figures/d2_rank_cdf.png)

**Length normalisation is the entire gap between the two rankers.** Raw `|Q ∩ D|`
rewards long documents, so verbose APIs crowd out the right one; dividing by the
union — the single correction every real lexical retriever makes — moves the
median rank from 62 to 4. The honest reading of "a parameter-free retriever" is
the normalised one.

Leakage is uniform across categories, so it cannot be excluded by subsetting.
Per-category ratio of means ranges 11.3–15.4 for the eight largest categories
(Data 11.5, Finance 15.4, Sports 13.2, Entertainment 13.5, Social 11.8, Tools
13.1, News_Media 13.7, Business 11.3).

**Consequence, stated explicitly.** A retriever with no parameters, no training,
and no tuning solves this task better than half the time. Any lexical method —
CNB and the BM25 baseline alike — will therefore look strong, and the number it
reports will be substantially a measurement of ChatGPT's paraphrasing habits
rather than of semantic tool matching. This **does not invalidate a CNB-vs-BM25
comparison**: the bias acts on both, and their difference is still interpretable.
It **does invalidate any claim about real-world usability**, and an external,
non-LLM-generated dataset is required before such a claim can be made.

---

## Diagnostic 3 — feature sparsity under EZR's vocabulary

> **Verdict: the bag-of-words representation is adequate — 0.04% of queries have
> zero features at EZR's default `top=100`, against a ~10% threshold. The
> tokenizer is nonetheless broken on the tool side and the stemmer is
> destructive; both need fixing for reasons other than sparsity.**

EZR's pipeline is re-implemented in `src/ez_tool/diagnostics/textpipe.py` rather
than imported, per scope. **It was validated against a dump produced by the
byte-identical vendored `ezr.py` and reproduces its top-100 vocabulary exactly —
term for term, including tie order — on all four EZR corpora (Hall, Radjenovic,
Wahono, Kitchenham).** The vocabulary is built from the queries, since in A3 the
queries are the rows.

### Query sparsity

Queries hold a median of **22** terms after tokenize → stop words → stem
(mean 22.6, min 3, max 143) —
[figures/d3_query_length.png](figures/d3_query_length.png). Those are token
counts *with* repeats; the "non-zero features" column below counts *distinct*
terms, which is why it tops out at 19 rather than 22.

| top-N | vocab | mean non-zero | median non-zero | **zero features** | ≤2 features |
|---|---|---|---|---|---|
| 100 | 100 | 9.78 | 10 | **0.04%** | 0.65% |
| 500 | 500 | 15.09 | 15 | 0.01% | 0.05% |
| 1,000 | 1,000 | 16.91 | 17 | 0.01% | 0.03% |
| 5,000 | 5,000 | 19.13 | 19 | 0.00% | 0.00% |
| all | 11,123 | 19.32 | 19 | 0.00% | 0.00% |

[figures/d3_nonzero_features.png](figures/d3_nonzero_features.png)

The threat does not materialise: even at `top=100`, a hundred terms cover ten of
a typical query's twenty-two. **[inferred]** the reason is the same phenomenon
Diagnostic 2 measured — ChatGPT-generated instructions are formulaic, so a small
vocabulary covers them well. Low sparsity here is a *symptom of the confound*,
not evidence of a healthy representation.

### Tokenizer damage on the tool side

`\b[a-zA-Z]+\b` treats `_` as a word character, so a snake_case identifier
contains no word boundary and tokenizes to **nothing at all** — not to
fragments, to the empty list. `re.findall(r'\b[a-zA-Z]+\b', 'get_weather_forecast')`
returns `[]`. `sendMessage` survives as one unsplit token.

(Tool names are counted as 3,249 distinct raw strings here, against the 3,225
case-folded classes used elsewhere — the tokenizer sees the raw form.)

| | API names (10,355) | tool names (3,249) |
|---|---|---|
| emptied by EZR's tokenizer | **4.04%** | **5.29%** |
| damaged (output differs from split version) | **19.33%** | **23.39%** |
| emptied after adding `_`/camelCase splitting | 0.61% | 0.46% |

The damage is 4–5% rather than near-total because ToolBench API names are mostly
human-readable ("Get Product Details"), not identifiers. That is a property of
this corpus, not of the tokenizer, and will not hold for a real tool registry.

### Top-30 terms at N=100

**Queries** — dominated by *instruction-template* boilerplate, none of which
EZR's abstract-tuned stop-word list removes:

> lik, would, new, list, need, detail, want, additional, inform, also, pleas,
> know, includ, fetch, api, product, could, nam, numb, articl, plann, data,
> specif, using, pric, avail, compan, match, provid, video

**API documentation** — the predicted *API* boilerplate, confirmed:

> search, valu, get, return, list, parameter, pag, result, api, nam, default,
> cod, endpoint, btc, numb, user, filter, countr, dat, data, loca, languag, typ,
> http, use, url, request, quer, retriev, tim

The expectation in the task brief holds on the documentation side exactly as
stated ("returns", "parameter", "default", "string"-adjacent terms dominate) and
holds on the query side in a different form: the noise is politeness and framing
("would", "pleas", "could", "want", "need", "also"), not API vocabulary.

### Stemmer damage

The suffix list ends in `e`, `y`, `s`, so stripping is aggressive. Over the
30,754 surface forms seen here it produces 17,421 stems, of which **9,813 merge
two or more distinct words**, often destructively:

| stem | merged surface forms |
|---|---|
| `stat` | stat, state, stated, statement, statements, states, static, statics, stating |
| `min` | min, mine, mineable, mined, miner, minify, mining, minor, minority |
| `serv` | serv, serve, served, server, servere, serverless, servers, serves, serving |
| `loc` | loc, local, locale, locales, locality, localize, locally, locals, locate |
| `pag` | pag, page, pageable, pageant, paged, pager, pages, pagesize, paging |

For a tool retriever, collapsing `state`/`static`/`statement` or
`mine`/`minor`/`minority` destroys exactly the distinctions the label depends on.

---

## Go / no-go per hierarchy level

### Category — **GO**

49 classes, effective 28.6, median 1,209 instances per class, zero starved
classes. Inside Rennie's regime. The label is single-valued for 99.66% of
instances. This is the safest level and the one the diagnostics actively support.

### Tool — **GO, with a quantified blind spot**

3,225 classes, effective 1,859, median 12. **This is A3's natural level**: the
label is exactly single-valued for 100% of G1 instances. The cost is the starved
tail — 1,528 classes (47.4%) at ≤3 instances, where CNB's `freq[k][a]` is noise
against `total[a]` and argmax is arbitrary. Those classes hold 5.1% of the data,
so the expected damage is bounded at roughly one query in twenty. Report
per-class results split at the support threshold; a single aggregate number will
hide this.

### API — **NO-GO as specified**

Not on support grounds — median 13, which passes. The blocker is structural:
**94.3% of G1 instances have 2+ ground-truth APIs**, so the class is a set and
the multi-class formulation does not apply. Recommend the next level up (tool),
per the decision rule's spirit. If API-level retrieval is wanted later it needs a
different formulation (ranking or multi-label), which is out of scope here.

### Cross-cutting — **NO-GO on any usability claim, at every level**

Diagnostic 2's 53.7% top-5 from a parameter-free ranker means results on this
data measure lexical leakage as much as tool selection. CNB-vs-BM25 remains a
valid *relative* comparison. An absolute claim ("this retriever would work in
deployment") requires a dataset whose queries were not generated from the
documentation being retrieved.

---

## Pipeline changes the numbers imply

Each tied to the statistic that motivates it. None of these are implemented —
this task is statistics only.

1. **Split identifiers before tokenizing** (`_`, `-`, `.`, `/`, digits, and
   camelCase humps). *Motivated by:* 4.04% of API names and 5.29% of tool names
   tokenize to the empty list under `\b[a-zA-Z]+\b`; 19.3% / 23.4% are damaged.
   Splitting drops the empty rate to 0.61% / 0.46%. Cheap, and mandatory before
   any real tool registry with snake_case names.

2. **Raise `the.textmine.top` from 100 to ~1,000.** *Motivated by:* mean non-zero
   features per query rises 9.78 → 16.91 while the zero-feature rate is already
   negligible; the gain is representation richness at 1,000, and returns flatten
   by 5,000 (19.13). Not a sparsity fix — sparsity is not the problem — but
   `top=100` was tuned for 250-word abstracts and is leaving signal on the table
   for 22-term queries.

3. **Replace the suffix stemmer**, or at minimum drop the `e`/`y`/`s` suffixes.
   *Motivated by:* 9,813 stems merge 2+ surface forms, including
   `state`/`static`/`statement` → `stat` and `mine`/`minor`/`minority` → `min`.
   Porter or Snowball (PyStemmer is already a pinned dependency) would preserve
   these distinctions.

4. **Extend the stop-word list with instruction-template and API-boilerplate
   terms.** *Motivated by:* the top-30 lists — 9 of the top 30 query terms are
   politeness/framing words EZR's list misses (`would`, `pleas`, `could`, `want`,
   `need`, `also`, `additional`, `know`, `lik`), and the documentation side is
   led by `search`, `valu`, `get`, `return`, `list`, `parameter`, `default`.
   These consume vocabulary budget at `top=100` without discriminating.

5. **Report tool-level results split by class support.** *Motivated by:* the
   bimodal histogram — 47.4% of classes at ≤3 instances holding 5.1% of labels.
   A single aggregate accuracy averages two populations with different expected
   behaviour.

6. **Do not index parameter defaults** (already the repo's convention; restated
   because this diagnostic re-derived the documentation concatenation).

---

## Scope compliance

No classifier was trained, no retriever was built, and `ezr.py` was not
imported. The overlap ranker in Diagnostic 2 is a statistic — a rank computed
from set intersections — not a retrieval implementation; it exists to measure the
confound and produces no run artifacts. The one blocker encountered is recorded
in Step 0: ToolBench's original instruction and answer files are not on disk, so
the 126,486 / 12,657 discrepancy could not be reconciled, only bounded.
