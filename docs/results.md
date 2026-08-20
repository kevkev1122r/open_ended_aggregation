# Results

All figures: judge-free programmatic grading, paired bootstrap over questions (10k
resamples), `*` = 95% CI excludes zero.

## QAMPARI — 8 agents, n=777

| method | F1 | vs best single |
|---|---|---|
| mean single agent | 19.06 | −7.12 \* |
| **best single (grok-4.3)** | **27.92** | reference |
| OW — response selection | 27.92 | +0.00 (zero-width CI) |
| union / no filter | 21.26 | −4.93 \* |
| MV — strict majority | 22.80 | −3.38 \* |
| MA-count filter (θ=2) | 28.10 | **+1.92 \*** |
| MA-count + OW | 28.19 | **+2.00 \*** |
| oracle selection | 35.51 | +9.33 \* |

`MA-count+OW − MA-count = +0.08 [−0.55, +0.76]` — a tight null. Counting does the work.

The margin decayed monotonically as n grew: +1.12 → +0.45 → +0.37 → +0.08, while the
unweighted filter crossed into significance. The contribution relocated from the weighting
to the merging.

## QUEST — 8 agents, n=297

| method | F1 | vs best single |
|---|---|---|
| mean single agent | 6.52 | −4.34 \* |
| **best single (grok-4.3)** | **10.86** | reference |
| OW — response selection | 10.86 | +0.00 |
| union / no filter | 3.97 | −6.88 \* |
| MV — strict majority | 6.56 | −4.30 \* |
| MA-count filter (θ=2) | 10.75 | −0.10 (ns) |
| **MA-count + OW** | **12.03** | **+1.17 \*** |
| oracle selection | 16.90 | +6.04 \* |

`MA-count+OW − MA-count = +1.27 [+0.62, +1.92] *` — and **+1.47 [+0.96, +1.99] \*** at
7 agents / n=386.

**The benchmarks invert.** On QAMPARI counting does everything and weighting adds nothing;
on QUEST counting alone does nothing and weighting is the entire gain.

## Why they invert — 56 five-of-eight subsets, both benchmarks

Correlation with weighting gain:

| predictor | QAMPARI | QUEST | consistent |
|---|---|---|---|
| **best / mean(rest)** (relative) | **+0.452** | **+0.473** | ✓ |
| # agents below half the best | +0.372 | +0.411 | ✓ |
| best − mean(rest) (absolute) | +0.427 | +0.382 | ✓ within, ✗ across |
| coefficient of variation | +0.342 | +0.324 | ✓ |
| best / weakest | +0.276 | +0.093 | ✗ |

Benchmark-level: QAMPARI relative dominance **1.43×**, QUEST **2.08×** — ordering the
outcomes correctly. The absolute gap is *larger* on QAMPARI (0.085 vs 0.063) and so cannot
explain the flip.

Correlation with **aggregation** gain runs the other way (−0.27 QAMPARI, −0.41 QUEST for
CV): weighting earns its keep exactly when merging-vs-best-single stops paying.

## Not established

- **Label-free operation.** All weights are supervised cross-fitted per-agent precision.
- **Published ASC as a baseline.** The composition step has never been run; all ASC-shaped
  numbers are steps 1–5 only, and the pool-size curve was still rising at 5 samples where
  ASC uses 50 — so the real baseline is probably stronger.
- **LLM-judged clustering.** Loses ~6 F1 at every model size tested.
