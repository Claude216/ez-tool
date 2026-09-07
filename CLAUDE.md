# ez-tool

Research repo for evaluating tool-selection methods on StableToolBench.

## Hard rules

### NEVER execute anything on a login node

`login1` / `login2` are for editing and job submission ONLY. No installs, no
downloads, no indexing, no Python, no experiments. Every command runs on a
compute node.

Current interactive node: **c5**.

```bash
ssh c5 '<command>'                      # while an interactive job is alive
srun -p <part> --time=HH:MM:SS -n1 ...   # otherwise
```

Interactive jobs on this cluster are capped at 6 hours. When c5's job ends,
re-check with `squeue -u lli66` and update the node name above.

### NEVER write outside /home/lli66/ or /mnt/beegfs/lli66/

Those two trees are the only permitted write locations. In particular **never
write to `/tmp`** — it is shared on login and compute nodes, and filling it
can get the account flagged.

This includes indirect writes. `TMPDIR` is unset by default on this cluster,
so pip/uv build directories, Python `tempfile`, and `tar` scratch all land in
`/tmp` unless redirected. Every job and interactive command must export:

```bash
export TMPDIR=/home/lli66/tmp
```

Use `/home/lli66/tmp` for scratch files, never `/tmp/...`.

## Filesystem layout

| Path | Purpose | Limit |
|---|---|---|
| `/home/lli66/ez-tool` | Git repo: code, configs, small results | **40 GB quota, shared with all of $HOME** |
| `/mnt/beegfs/lli66/ez-tool` | Benchmark data, indexes, run outputs | beegfs, TBs free |
| `/mnt/beegfs/lli66/cache/huggingface` | Model weights (`HF_HOME`, already exported) | beegfs |
| `/mnt/beegfs/lli66/venvs/ez-tool` | uv-managed venv | beegfs |

`/mnt/beegfs` is **not mounted on login nodes**. It exists only on compute
nodes — another reason every command runs on c5.

Never write bulk data under `$HOME`. `data/` in the repo is a symlink to
beegfs.

## Environment

- uv + venv on beegfs. Dependencies pinned in `pyproject.toml` / `uv.lock`,
  both committed.
- System python is 3.6.8; the system `conda` is 4.10.3. Use neither.
- Compute nodes DO have network access. (A note in the older `aise26` scripts
  claims otherwise — that is stale, verified 2026-08-31 on c23 and c58.)

## Experiment

Phase 1 — BM25 baseline, retrieval-only. No LLM in the loop.
Metrics: Recall@k / NDCG@k against annotated relevant APIs.

Retrieval unit: one document per callable (`tool_name` + `api_name` always).
Whether the argument schema is indexed is an ablation, not a fixed choice —
**disclosure granularity is not index granularity**. Anthropic's tool search
defers a tool's schema from the model's context but still indexes "tool names,
descriptions, argument names, and argument descriptions", so `with_parameters`
mirrors a deployed retriever while `strict` indexes only what the model is
shown up front. Measured: `with_parameters` gains +0.028 R@5 / +0.021 NDCG@10.
Parameter *defaults* are never indexed (base64 payloads).

Model for later phases: `Qwen/Qwen3.5-4B` (9.34 GB bf16, multimodal
`Qwen3_5ForConditionalGeneration`, hybrid linear attention, 262K context).
Supported by vLLM >= 0.28.0. Weights only for now — do not start a server.
