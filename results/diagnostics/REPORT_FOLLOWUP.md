# ToolBench follow-up diagnostics — hard negatives and test-set support

Second round. Both follow-ups change a verdict from
[round one](REPORT.md). Statistics only: no classifier, no retriever, no
`ezr.py` import.

| | verdict |
|---|---|
| **Follow-up A** — hard negatives | **Downgrade.** At equal pool size, same-category negatives drop top-5 to **73.6%**, *below* the size-matched random control's **84.5%**. The 53.7% figure measured pool size and cross-domain topicality, not copied wording. |
| **Follow-up B** — test-set support | **Bounded-damage claim confirmed** (≤3-support share is 3.5–8.0%, against the predicted 5.1%) — but the premise the task supplied is **wrong**: `test_G1_tool` and `test_G1_category` do **not** have zero support. Median support is 57 and 49; **zero test instances out of 600 have an unseen tool**. |

Follow-up B's investigation found something neither round asked for: this
release is a **random split with heavy train/test near-duplication** —
74.5% of the StableToolBench solvable queries have a training query at
Jaccard ≥ 0.5, and 10 are exact duplicates. That is reported in B.3.

Reproduce with `python scripts/run_diagnostics_followup.py` on a compute node;
raw numbers in [`followup.json`](followup.json).

---

## Follow-up A — leakage decomposition under hard negatives

> **Verdict: downgrade. The round-one leakage figure does not survive a
> size-matched control. Same-category negatives are *harder* than random ones,
> which is the opposite of what copied wording would produce.**

Everything except the candidate pool is held fixed from round one: EZR
tokenizer with identifier splitting, `api_name` + `api_description` + parameter
names and descriptions, `tool_name`/`category_name` excluded, defaults never
indexed, length-normalised Jaccard, best-matching ground-truth API.

### Ranks by pool

Chance is computed analytically per instance as
`1 − C(n−g, k)/C(n, k)` for a pool of `n` with `g` ground-truth entries —
without it, small pools look like triumphs.

| pool | instances | pool size (min/med/max) | median rank | top-1 | top-5 | chance top-5 | **lift** | norm. rank |
|---|---|---|---|---|---|---|---|---|
| **P0** all APIs | 88,995 | 10,355 / 10,355 / 10,355 | 4.0 | 30.5% | 53.7% | 0.11% | **500×** | 0.0004 |
| **P1** same category | 88,995 | 6 / 398 / 2,031 | 1.5 | 44.0% | 73.6% | 5.08% | **14.5×** | 0.0066 |
| **P2** same tool | 84,443 | 2 / 4 / 20 | 1.0 | 74.4% | 99.5% | 95.98% | **1.0×** | 0.3333 |
| **P3** size-matched random *(control)* | 88,995 | 6 / 398 / 2,031 | 1.0 | 59.1% | **84.5%** | 5.08% | **16.6×** | 0.0046 |

Optimistic tie-break, same ordering: P0 56.4%, P1 75.9%, P2 99.8%, P3 85.5%
top-5. The conclusion is identical under either tie convention.

Excluded for degenerate pools: **P2 4,552 instances** (single-API tools); P0,
P1, P3 zero — no instance had a category pool below 5.

[figures/d4_pool_ranks.png](figures/d4_pool_ranks.png)

### Reading it

**The P0 → P1 rise is pool size, not leakage.** Top-5 goes 53.7% → 73.6% when
the pool shrinks from 10,355 to a median of 398. The control does the same
shrink with *random* negatives and reaches **84.5%** — higher. So restricting to
one category makes the task **harder by 10.9 points at identical pool size**.

That is the opposite of the leakage signature. If the generator had copied
distinctive wording out of the ground-truth documentation, that wording would
still separate ground truth from its same-category neighbours, and P1 would
track P3. Instead the ranker's advantage largely evaporates once every candidate
shares the topic — which is what *topical relatedness*, the legitimate signal,
looks like.

### The overlap ratio decays the same way

Round one's headline was a 12.4× Jaccard ratio, also measured against uniformly
drawn negatives. Recomputed against harder ones:

| negative | mean Jaccard with query | ratio to ground truth |
|---|---|---|
| ground-truth API | 0.1634 | — |
| other API, **same tool** | 0.0786 | **2.08×** |
| random API, **same category** | 0.0222 | **7.35×** |
| random API, any category | 0.0132 | **12.37×** *(round one, reproduced)* |

[figures/d4_overlap_tiers.png](figures/d4_overlap_tiers.png)

Most of the 12.4× was cross-domain vocabulary separation. A residual 2.08×
advantage survives against sibling APIs inside the same tool.

**Limitation, stated rather than glossed.** This design bounds how much of the
round-one number was cross-domain topicality — most of it — but it **cannot
separate residual verbatim copying from legitimate semantic matching** at any
tier. A query asking to search products *should* match `Search` better than
`Get order`. The 2.08× and the P1 14.5× lift are consistent with either
explanation, and nothing here decides between them. **[inferred]** the direction
of the P1-vs-P3 gap makes copied wording the less likely of the two, but that is
an inference from the sign of one comparison, not a measurement.

### P2 — informational, as specified

**P2 is a near-random result and reinforces round one's API-level no-go.** Its
99.5% top-5 is an artifact: the median pool holds 4 APIs of which ~2 are ground
truth, so chance top-5 is **96.0%** and lift is **1.0×**. The meaningful numbers
are top-1 74.4% against **60.7% chance** (lift 1.2×) and a median normalised
rank of **0.333** — ground truth sits a third of the way down its own tool's
API list. Lexical features barely distinguish APIs within a tool. Round one
ruled out API-level A3 because the label is a set; this says that even if the
label were single-valued, the features would not support it.

---

## Follow-up B — test-set support distribution

> **Verdict: the bounded-damage claim is confirmed — but the task's premise about
> the test splits is wrong, and the investigation it mandated turned up
> train/test near-duplication that matters more than either.**

### B.1 — Lead with the contradiction: the test tools are not unseen

The task specified `test_G1_tool` → unseen tools → support **expected zero**, and
`test_G1_category` likewise, with the instruction: *"If support turns out to be
non-zero, that assumption is wrong and needs investigating."*

It is non-zero. Measured, training support of each test instance's ground-truth
tool:

| split | n | min | p25 | median | p75 | max | **zero support** | tools unseen in train |
|---|---|---|---|---|---|---|---|---|
| `test_G1_instruction` | 200 | 2 | 43 | **56** | 66 | 392 | **0.0%** | **0 of 184** |
| `test_G1_tool` | 200 | 2 | 42 | **57** | 68 | 141 | **0.0%** | **0 of 114** |
| `test_G1_category` | 200 | 1 | 38 | **49** | 67 | 92 | **0.0%** | **0 of 103** |

Category level: median support 3,369 / 3,758 / 3,851; zero instances at or below
3; **no unseen category in any split**.

[figures/d5_support_overlay.png](figures/d5_support_overlay.png)

**Investigation.** Three independent lines of evidence say this release is a
random split by query id, not ToolBench's generalisation splits:

1. **Query ids interleave.** Train spans ids 1–88,995; the three test splits
   scatter throughout it (`instruction` 577–88,193, `tool` 394–88,197,
   `category` 28–86,536). A held-out-tool split would not produce this.
2. **Adjacent ids share a tool across the boundary.** Test `qid=394` (tool
   `simple youtube search`, 74 training instances) sits beside training
   `qid=388`, `qid=389` on the same tool.
3. **Near-identical content crosses the boundary.** Test `qid=28` asks to track
   package `CA107308006SI`; training `qid=30` asks about **the same package
   ID**.

**[inferred]** The `_tool` and `_category` names are labels inherited from
ToolBench's *instruction* data, where those holdouts are defined. ToolGen's
retrieval repackaging appears to have pooled and re-split the queries, and the
names no longer describe the split. I could not confirm this against ToolBench's
own retrieval split — those files are not on disk (see Provenance).

**Consequence.** The scope statement the task intended to support — *"A3 cannot
answer on two of the three G1 test splits, by construction"* — **is not
supported by this data and must not appear in the writeup**. On this release, A3
can answer on all three splits, because all three test on tools it has seen.
The unseen-tool and unseen-category generalisation settings are simply **not
available** here; obtaining them requires ToolBench's original splits.

### B.2 — The bounded-damage claim survives

Round one predicted ~5.1% of test-time labels would fall in starved (≤3
training instances) classes. Measured:

| split | ≤3 support | ≤5 | ≤10 |
|---|---|---|---|
| `test_G1_instruction` | **3.5%** | 4.0% | 5.0% |
| `test_G1_tool` | **7.0%** | 7.5% | 7.5% |
| `test_G1_category` | **8.0%** | 8.0% | 8.0% |
| **StableToolBench solvable G1 (474)** | **5.5%** | 5.5% | 5.9% |

The evaluation set that downstream work actually reports on lands at **5.5%**,
against the 5.1% predicted from the training distribution. **Round one's
"damage bounded at roughly one query in twenty" is confirmed**, and needs no
revision.

The `_tool` and `_category` splits run modestly hotter (7.0%, 8.0%) — visible in
the overlay figure as a small excess in the low-support bins. **[inferred]** they
sample tools slightly more uniformly than training does, a residue of whatever
selection produced them. The effect is a couple of percentage points, not the
47% the task flagged as the risk.

**Recommendation unchanged from round one:** report tool-level results split at
the support threshold. At 5.5% the aggregate is not badly distorted, but the two
populations still behave differently.

### B.3 — StableToolBench matching, and what it exposed

All **474** G1 solvable queries matched the ToolGen G1 set **by query id, 100%**
— zero required text fallback, zero unmatched. None appears in the ToolGen
*training* portion by id or by exact text, so the evaluation set is properly
held out **as a set of ids**.

| file | n | match rate | by id | by text | unmatched | in ToolGen train |
|---|---|---|---|---|---|---|
| `G1_instruction` | 163 | 100% | 163 | 0 | 0 | 0 |
| `G1_tool` | 158 | 100% | 158 | 0 | 0 | 0 |
| `G1_category` | 153 | 100% | 153 | 0 | 0 | 0 |

**But id-level holdout is not content-level holdout.** Since B.1 showed the split
is random, I measured how close each test query's *nearest training query* is:

| set | median | p75 | p95 | max | **≥ 0.5** | ≥ 0.7 | exact duplicates |
|---|---|---|---|---|---|---|---|
| `test_G1_instruction` | 0.558 | 0.700 | 0.875 | 1.000 | 65.0% | 25.5% | 3 |
| `test_G1_tool` | 0.551 | 0.667 | 0.867 | 1.000 | 68.5% | 20.5% | 1 |
| `test_G1_category` | 0.600 | 0.735 | 1.000 | 1.000 | 73.0% | 28.0% | 7 |
| **StableToolBench solvable G1** | **0.583** | 0.714 | 0.905 | 1.000 | **74.5%** | **27.4%** | **10** |

[figures/d5_near_duplicates.png](figures/d5_near_duplicates.png)

**Three-quarters of the evaluation set has a training query sharing half its
vocabulary, and ten are exact duplicates after preprocessing.** Any model trained
on the ToolGen train split and scored on StableToolBench's solvable queries
inherits this. It does not affect a retrieval-only baseline that never trains on
queries, but it is decisive for A3, which does.

**This is now the strongest evaluation-validity concern in either round** — a
larger effect than the leakage Follow-up A was sent to measure, and it was found
only because the support numbers came back wrong.

---

## Round-one verdicts: confirmed, downgraded, strengthened

| round-one verdict | status | driving numbers |
|---|---|---|
| Category level **GO** | **confirmed** | Unchanged; nothing in this round bears on it. |
| Tool level **GO with a bounded blind spot (5.1%)** | **confirmed** | Test-time ≤3-support share is 3.5–8.0% by split, **5.5%** on the StableToolBench evaluation set. |
| API level **NO-GO** | **strengthened** | Still ruled out by the multi-label finding. Now also: within a tool, Jaccard ranks ground truth at normalised rank **0.333** with lift **1.2× over chance** — the features do not separate sibling APIs either. |
| Cross-cutting **NO-GO on any usability claim** ("parameter-free ranker solves it") | **downgraded** | P1 73.6% vs size-matched control P3 **84.5%**; overlap ratio decays 12.37× → 7.35× → 2.08×. The 53.7% was pool size plus cross-domain topicality. |
| *(new)* Train/test contamination | **new finding** | 74.5% of the evaluation set has a training near-duplicate at Jaccard ≥ 0.5; 10 exact duplicates. |
| *(new)* Unseen-tool / unseen-category settings | **new finding** | Not available in this release: 0 of 600 test instances has an unseen tool or category. |

### The downgraded verdict, restated

Round one said: *"an external, non-LLM-generated dataset is required for any
usability claim."* **That is now stated too strongly and is replaced by:**

> Reverse generation leaves a measurable lexical trace — ground truth still
> out-overlaps its own tool's siblings by 2.08× — but the round-one experiment
> did not isolate it, and the 53.7% figure is not evidence of it. Absolute
> retrieval numbers on ToolBench should be read as optimistic, chiefly because
> the candidate pool is dominated by topically unrelated APIs and because of the
> train/test near-duplication in B.3. A CNB-vs-BM25 comparison remains valid.
> An external dataset is **desirable** for a usability claim; it is no longer
> demonstrated to be **required** by the leakage evidence.

Note the concern did not disappear so much as move: the weaker leakage finding
is offset by a contamination finding that is larger and better evidenced.

---

## Revised go / no-go (supersedes round one)

| level | verdict | driving numbers |
|---|---|---|
| **Category** | **GO** | 49 classes, effective 28.6, median 1,209 training instances; test-time zero-support 0%, ≤3 0%. Inside Rennie's regime. |
| **Tool** | **GO** — A3's level | Single-valued label for 100% of G1 instances. Median test-time training support **56**; **5.5%** of the evaluation set in ≤3-support classes. Report results split at the support threshold. |
| **API** | **NO-GO** | Multi-label for 94.3% of instances; and lexically inseparable within a tool (normalised rank 0.333, lift 1.2×). |
| **Usability claim** | **CAUTION** (was NO-GO) | Leakage smaller than round one measured (2.08× against siblings). The binding constraint is now contamination: 74.5% of the evaluation set has a training near-duplicate. |
| **Generalisation to unseen tools/categories** | **NOT MEASURABLE HERE** | 0 of 600 test instances has an unseen tool or category. Needs ToolBench's original splits. |

### What this implies for the modelling work

1. **Train tool-level.** It is the only level where A3 is well-posed and
   supported.
2. **Deduplicate train against test before reporting anything**, at a Jaccard
   threshold of at least 0.7 (27.4% of the evaluation set) and preferably 0.5
   (74.5%). Report scores with and without. This is now the first thing to do,
   ahead of any tuning.
3. **Do not claim unseen-tool generalisation** from this release, and do not
   describe `test_G1_tool` / `test_G1_category` as unseen-tool/category settings.
4. Round one's five pipeline recommendations stand, unimplemented and unchanged.

---

## Provenance

* **Training data:** ToolGen repackaging of ToolBench's retrieval split —
  HuggingFace `reasonwang/ToolGen-Datasets`, `data.tar.gz`, blob
  `f80aa7422a39bc69b1d7c963f064fff8d94988f9dd153b12790087749cc2e7e0`, commit
  `9823c75ee3eabc790c11c1df924a084fe2fb5e53`. Path
  `data/raw/toolgen_data/data/retrieval/G1/`.
* **Evaluation set:** StableToolBench `solvable_queries` —
  `THUNLP-MT/StableToolBench` @ **`aa4ed9f4737ad98bd706663f01d63623c3427812`**.
  Path `data/raw/StableToolBench/solvable_queries/test_instruction/`, 474 G1
  queries.
* **The 126,486 / 12,657 instance-count discrepancy is closed as unresolved.**
  ToolBench's original instruction files are not available on disk. Stated once;
  not revisited.

One caveat this round inherits: B.1's conclusion that the split is random rests
on three lines of internal evidence, not on a comparison against ToolBench's own
retrieval split, which is unavailable for the same reason. The *measurements*
(zero unseen tools, the near-duplicate rates) stand regardless of why the split
looks this way.
