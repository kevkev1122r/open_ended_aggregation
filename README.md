# Open-ended LLM answer aggregation

Aggregating **open-ended, set-valued** answers from heterogeneous LLM agents by merging
atomic claims, rather than selecting one agent's whole response.

Extends Ai, Pan, Simchi-Levi, Tambe & Xu, *Beyond Majority Voting: LLM Aggregation by
Leveraging Higher-Order Information* (ICML 2026, [arXiv:2510.01499](https://arxiv.org/abs/2510.01499)),
whose §7 asks how to derive optimal weights for open-ended questions.

## Headline

Two set-valued benchmarks, eight agents, judge-free programmatic grading.

| | QAMPARI (n=777) | QUEST (n=297) |
|---|---|---|
| best single agent | 27.92 | 10.86 |
| MV — strict majority | worse than best single | worse than best single |
| OW — response selection | reproduces best single **exactly** | reproduces best single **exactly** |
| MA-count filter | **+1.92 \*** | −0.10 (ns) |
| **MA-count + OW (ours)** | +2.00 \* | **+1.17 \*** |
| weighting gain (OW − count) | +0.08 (ns) | **+1.47 \*** |

`*` = 95% paired-bootstrap CI excludes zero.

**The two benchmarks disagree about which component works, and that turns out to be
predictable.** Sweeping all 56 five-of-eight agent subsets on both benchmarks, the
weighting gain tracks **relative dominance** — `best precision / mean(rest)` — at
r = +0.452 (QAMPARI) and +0.473 (QUEST). QUEST's best agent is 2.08× the rest; QAMPARI's
is 1.43×. Reliability weighting pays off when one agent clearly dominates, and adds
nothing when the pool is bunched.

The absolute gap `best − mean(rest)` does *not* explain it: that number is larger on
QAMPARI (0.085 vs 0.063), pointing the wrong way.

## What is and is not established

**Established** — merging atomic claims beats selecting a whole response; majority voting
is actively worse than the best single agent on set-valued answers; the published OW rule
is structurally inert here (with global weights and open-ended answers no two responses
coincide, so its argmax is always the highest-weighted agent — it reproduces best-single
with a zero-width CI); relative dominance predicts when weighting helps, on both benchmarks.

**Not established** — that any of this works label-free (all weights are supervised
cross-fitted per-agent precision); published ASC as a baseline (see below).

## Quick start

```bash
python3 -m venv venv && ./venv/bin/pip install -e .
cp .env.example .env      # add your Azure endpoint + key

./venv/bin/python -m open_ended_aggregation.analysis.compare_qampari
./venv/bin/python -m open_ended_aggregation.analysis.compare_quest
./venv/bin/python -m open_ended_aggregation.analysis.subset_sweep
```

Analysis needs no API calls — every model response is cached in `data/` (gitignored,
~250 MB, ~$130 of Azure spend). Generation resumes from that cache and only fills gaps.

See [docs/getting-started.md](docs/getting-started.md).

## Layout

```
open_ended_aggregation/
  backends/      azure, openrouter, judge — transport only
  benchmarks/    qampari, quest, asqa, facts — loader + prompt + parser + metrics
  methods/       kernel (KWA), asc (paper-exact), llm_cluster (variant)
  analysis/      per-benchmark comparisons, subset sweep, independent verification
  generation/    resumable backfill drivers
configs/pools.yaml   agent pools and their quirks
docs/                method, benchmarks, results, troubleshooting, handoffs
archive/             superseded one-off scripts, kept for provenance
```

Each benchmark is **fully separate** — own loader, prompt, gold format and metrics.
Coupling them is what produced the earlier "ASQA cannot replicate" confusion.

## A naming warning

No arm in this repo is published ASC. `MA-count` is a **multi-agent adaptation** of ASC's
count filter: published ASC ([arXiv:2405.13131](https://arxiv.org/abs/2405.13131)) draws
m=50 stochastic samples from **one** model, tunes Θ on a validation set, and composes
survivors with an LLM. Counting over agents instead of samples measures inter-model
agreement, not self-consistency — a different estimand. See
[docs/method.md](docs/method.md).

## Read before trusting a number

[docs/troubleshooting.md](docs/troubleshooting.md) lists the silent-failure modes found so
far — several produced wrong headline numbers before being caught. Every figure here was
re-derived independently from raw generations, and the uniform/shuffled controls behave as
predicted.
