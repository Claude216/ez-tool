"""How much of the evaluation set survives each candidate dedup rule, per category.

Rules, applied to each StableToolBench-solvable G1 query against the ToolGen
train split (the 'stream' in a category-stream experiment):
  text50   drop if max Jaccard(token set) to ANY train query >= 0.5
  text70   same, threshold 0.7
  cat50    drop if max Jaccard to a SAME-CATEGORY train query >= 0.5
  label    drop if some train query has the IDENTICAL relevant-API set
  joint    drop if same-category Jaccard >= 0.5 AND identical relevant set
Also: within-category train redundancy (pairs at Jaccard>=0.5 with same label).
"""
import json, sys
from collections import Counter, defaultdict
import numpy as np
from ez_tool.diagnostics import textpipe, g1
from ez_tool.diagnostics.d2_leakage import _binary_matrix
from ez_tool.diagnostics.d5_support import STB_G1, STB_DIR
from ez_tool.diagnostics.g1 import G1_TEST_SUBSETS
from ez_tool.paths import REPO_ROOT

OUT = REPO_ROOT / "results" / "diagnostics"

data = g1.load()
inst = data.instances
cache = {}
sets = [set(textpipe.prepare(i.query, cache, split_identifiers=True)) for i in inst]
vocab = {t: k for k, t in enumerate(sorted(set().union(*sets)))}
M = _binary_matrix(sets, vocab).tocsr()
lens = np.array([len(s) for s in sets], dtype=np.float32)

def cat_of(i):
    for p in i.apis:
        if p in data.docs:
            return data.docs[p].category
    return "?"
cats = [cat_of(i) for i in inst]
labels = [frozenset(i.apis) for i in inst]

stb_ids = set()
for name in STB_G1:
    for rec in json.loads((STB_DIR / f"{name}.json").read_text()):
        stb_ids.add(rec.get("query_id"))

is_train = np.array([i.split == "train" for i in inst])
tr = np.flatnonzero(is_train)
te = np.flatnonzero(np.array([i.split in G1_TEST_SUBSETS and i.query_id in stb_ids for i in inst]))
print(f"train={len(tr)} test(stb G1)={len(te)}", file=sys.stderr)

def jac(rows, cols):
    inter = np.asarray((M[rows] @ M[cols].T).todense(), dtype=np.float32)
    union = lens[rows][:, None] + lens[cols][None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

J_all = jac(te, tr)                       # (n_te, n_tr)
best_all = J_all.max(axis=1)
tr_cat = np.array([cats[j] for j in tr])
tr_lab = [labels[j] for j in tr]
lab_index = defaultdict(list)
for k, j in enumerate(tr):
    lab_index[labels[j]].append(k)

rows = []
per_cat = defaultdict(lambda: Counter())
for r, i in enumerate(te):
    c = cats[i]
    same = np.flatnonzero(tr_cat == c)
    best_cat = J_all[r, same].max() if len(same) else 0.0
    same_label = lab_index.get(labels[i], [])
    joint = any(J_all[r, k] >= 0.5 for k in same_label)
    d = per_cat[c]
    d["n"] += 1
    d["text50"] += best_all[r] >= 0.5
    d["text70"] += best_all[r] >= 0.7
    d["cat50"] += best_cat >= 0.5
    d["label"] += bool(same_label)
    d["joint"] += joint

# within-category train redundancy
train_cat_n = Counter(tr_cat)
redund = {}
for c, n in train_cat_n.items():
    idx = tr[tr_cat == c]
    if n > 6000:
        idx = idx[:6000]
    Jc = jac(idx, idx)
    np.fill_diagonal(Jc, 0)
    lab = [labels[j] for j in idx]
    # a train query is redundant if some EARLIER one has J>=0.5 and same label
    red = 0
    for a in range(len(idx)):
        js = np.flatnonzero(Jc[a, :a] >= 0.5)
        if any(lab[b] == lab[a] for b in js):
            red += 1
    redund[c] = (len(idx), red)

tot = Counter()
for c, d in per_cat.items():
    tot.update(d)
def row(c, d, ntr, red):
    n = d["n"]
    surv = lambda k: f"{n - d[k]} ({100*(n-d[k])/n:.0f}%)"
    return f"| {c} | {ntr} | {red} | {n} | {surv('text50')} | {surv('text70')} | {surv('cat50')} | {surv('label')} | {surv('joint')} |"
out = ["| category | train n | train redundant (J>=.5 & same label) | test n | keep text<.5 | keep text<.7 | keep cat<.5 | keep no-label-twin | keep not-joint |", "|---|---|---|---|---|---|---|---|---|"]
for c, d in sorted(per_cat.items(), key=lambda kv: -kv[1]["n"]):
    ntr, red = redund.get(c, (0, 0))
    out.append(row(c, d, ntr, red))
out.append(row("**all**", tot, len(tr), sum(r for _, r in redund.values())))
text = "\n".join(out)
print(text)
open(OUT / "dedup_survey.md", "w").write(text + "\n")
json.dump({c: dict(d) for c, d in per_cat.items()} | {"_redundant": redund}, open(OUT / "dedup_survey.json", "w"), indent=1)
