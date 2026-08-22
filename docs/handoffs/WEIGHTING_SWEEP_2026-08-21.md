# Weighting mechanisms — full sweep, 21 August 2026

~60 aggregation rules cross-fitted on QAMPARI (777 q, 8 agents, 110,801 candidate
claims) and the survivors ported to ASQA (400 q, 6 agents, 6,015 sentence
clusters). Cached generations only — **zero API spend**.

Protocol throughout: 5-fold cross-fitted over questions (2-fold on ASQA, where
each threshold evaluation costs a ROUGE-L pass). Weights, likelihood tables,
logistic and GBM coefficients, calibration maps, ridge coefficients, blend
coefficients and keep thresholds are fitted on training folds only. Bootstrap
95% CIs resample questions.

---

## 1. The result

```
  arm                     F1        vs best single           vs MA-count
  best single (Kimi)    27.40           --                      -1.95
  MA-count              29.35     +1.95  ( +7.1%)*            (reference)
  pattern  [CEILING]    29.60     +2.20  ( +8.0%)*    +0.25  ( +0.8%)  ns
  BLEND + budget        32.01     +4.61  (+16.8%)*    +2.66  ( +9.1%)*
```

**+16.8% over the best single agent** (CI [+3.64, +5.58]) and **+9.1% over
MA-count** (CI [+1.86, +3.47]).

Seed stability — three independent fold assignments:

```
  seed        F1     vs best single     vs count
    0       32.01       +16.8%           +9.1%
    1       32.04       +16.9%           +9.2%
    2       32.11       +17.2%           +9.4%
```

Spread 0.10 F1 across seeds, against CI half-widths near 1.0. This is not noise.

The bar this has to clear is `pattern`, the **best rule expressible from the
support pattern alone** — i.e. the ceiling on all reliability weighting,
including correlation-aware and omission-aware forms. It is +0.8% and not
significant. Everything above it comes from information reliability weighting
cannot see.

### 1.1 What `BLEND + budget` is

1. Per-claim features outside the support pattern: rank of the claim inside each
   supporter's own list; how much each supporter said on this question; **how
   much each SILENT agent said**; per-agent rank buckets; the pattern
   probability stacked in as a feature.
2. Two scorers over those features — an L2 logistic and a histogram GBM
   (`analysis/gbm.py`, ~200 lines, no new dependencies) — combined by a
   meta-logistic fitted on a held-out slice of train.
3. A **per-question keep budget**: `k*(q) = argmax_k 2·(Σ_{i≤k} p_i)/(k + Ĝ_q)`,
   with `Ĝ` a ridge-predicted gold count.

---

## 2. Decomposition — three axes, and only one is "weighting"

| axis | vs count | note |
|---|---|---|
| who asserted (`pattern`, the ceiling) | +0.25 **ns** | this is what the project called weighting |
| **rank / verbosity / omission** | **+1.44** * | a global threshold, no budget |
| **per-question budget** | **+1.19** * | `adapt-count`: **no model identity at all** |
| nonlinear + blend | +0.27 | GBM alone is worse; it only pays under a blend or budget |
| all together | **+2.66** * | |

The two large axes are near-independent, and **neither is reliability
weighting.** Model identity is worth ~+0.3 on top of either. That inverts the
project's framing and is the finding to build the paper on.

### 2.1 Biggest single new feature: omission strength

Every prior rule treated silence as one thing — `nb-full` gave every silent agent
the same `log((1−TPR)/(1−FPR))`, the logistic gave it zero. But QAMPARI list
lengths run from a median of 1 (gpt-5.4-nano) to a mean of 71 (Kimi):

> an agent that emitted 3 items and did not mention X has barely spoken;
> an agent that emitted 50 and did not mention X has said a great deal.

Per-agent `log(list length)` for the **silent** agents was the largest single
ablation win: **+1.65 over counting on its own.**

### 2.2 It holds at every ensemble size — 210 of 210

`analysis/rank_crossover.py`, all 210 ensembles of size 3–6, 2-fold cross-fitted,
same folds and questions as `ensemble_sizes.py`.

```
                              GAIN OVER COUNTING
  size   n   best single  count   weighted  pattern    rank   rank+budget
    3   56      24.72     22.15     +2.25    +2.73    +4.12     +4.43  (+20.0%)
    4   70      26.00     23.69     +1.90    +2.57    +4.16     +4.53  (+19.1%)
    5   56      26.77     26.17     +0.40    +1.13    +2.74     +3.13  (+12.0%)
    6   28      27.18     27.74     -0.36    +0.36    +1.86     +2.31  ( +8.3%)

  ensembles where the arm SIGNIFICANTLY beats counting
    3            39/56    45/56    56/56    56/56
    4            39/70    44/70    70/70    70/70
    5            20/56    23/56    50/56    56/56
    6             7/28     7/28    21/28    28/28
  total         105/210  119/210  197/210  210/210
```

**`rank+budget` beats counting in every one of the 210 ensembles, significantly,
at every size** — while reliability weighting manages 105/210 and goes *negative*
by n=6. The relative gain is largest exactly where the pool is small: **+20.0% at
n=3**.

The `count`, `best single`, `weighted` and `pattern` columns reproduce
`ensemble_sizes.py` and `crossover_controls.py` byte-for-byte, so this is two
independent harnesses agreeing.

This also settles the 20 Aug crossover: **it is a property of support-pattern
rules, not of aggregation.** Pattern-based weighting crosses over at n≈5–6; rank
and rank+budget never cross over anywhere we can measure.

### 2.3 The control that proves budgets are about adaptivity

`top-K` — the same budget idea with one k for every question, cross-fitted, picks
k=8 — scores **29.41, no gain over counting**. The value is in the budget
*varying* per question, not in keeping fewer things.

---

## 3. Everything tested

### Works

| rule | vs count | |
|---|---|---|
| BLEND + budget | **+2.66** * | champion |
| GBM(bagged) + budget | +2.47 * | |
| STACK + budget (8 scorers) | +2.27 * | stacking did **not** beat the 2-way blend |
| LEAN + budget | +2.39 * | linear, much simpler, 90% of the gain |
| v3 +omission | +1.65 * | best single feature block |
| LEAN (logistic) | +1.44 * | |
| nb-rank | +1.34 * | rank-conditioned NB, nonparametric in rank |
| adapt-count | +1.19 * | budget with zero model identity |
| RRF(K=10) | +0.84 * | parameter-free IR fusion — **report as a baseline** |
| 2D `P(correct \| count, rank bucket)` | +0.71 * | |
| count × rank (two integers) | +0.63 * | the one-line version |
| pattern (ceiling) | +0.25 ns | |

### Does not work — each a usable negative

| rule | result | why it matters |
|---|---|---|
| error-correlation / diversity weighting | **λ=0 in 5/5 folds** | the supervisor's #4; it is a function of the support pattern, so the ceiling caps it |
| pairwise co-assertion (28 indicators) | +0.05 | same conclusion reached discriminatively, without assuming a functional form |
| rank agreement (spread/range) | +0.07 | agents agreeing on *order* carries nothing extra |
| question-domain weights `w_i(q)` | +0.00 | QAMPARI's 5 sources carry nothing; "condition on the question" ≠ topic |
| Borda / CombSUM / wBorda | −0.42 to −0.67 | **below counting**; only RRF among fusion rules survives |
| isotonic calibration | −0.03 | logistic already calibrated (3.55% predicted vs 3.55% actual) |
| within-question z-norm / percentile | −0.94 / −2.56 | destroys the cross-question information a global threshold needs |
| learned per-question k | −3.71 | k* is essentially unpredictable from question features |
| budget objective: fitted saturation | −0.02 | **fitted β = 1.10, i.e. no saturation** — recall does not saturate here |
| budget objective: precision scale c | −0.12 | plain `2S/(k+G)` is already right |
| kitchen-sink 115 features (linear) | below its own ablations | overfits; LEAN subset is better |
| GBM alone | −1.01 vs LEAN | best ranking (AP 0.397→0.418), worst cross-question calibration |
| alias merging (6 thresholds) | −0.02 to −1.30 | see below |

### 3.1 Alias merging — a negative with a clean mechanism

Merging string-similar candidates before scoring should consolidate split
support. It does not, at any threshold:

```
   tau   candidates   good/bad merges   LEAN+budget    vs no-merge
  1.00      110,801        0/0             31.74       (reference)
  0.95      108,706        0/1             31.72        -0.02
  0.90      106,167        7/10            31.68        -0.07
  0.85      102,765        7/40            31.50        -0.24
  0.80       95,090        8/105           31.53        -0.21
  0.70       82,227       14/413           30.44        -1.30 *
```

**Bad merges outnumber good ones at every threshold.** String-similar QAMPARI
entities are usually genuinely *different* entities (sequels, same-surname
people), not aliases — and a false merge deletes a correct answer outright. This
independently explains why LLM-judged clustering lost ~6 F1 earlier, and the
β=1.10 saturation result says the same thing from the other direction: duplicate
spellings of one gold answer are rare in this data.

### 3.2 `oracle-k` = 39.90 is not available headroom

Keeping the best prefix per question, on the ordering the model already produces,
scores 39.90 (+36%). That is the maximum of ~143 noisy prefix scores per
question and is heavily selection-biased. Every honest attempt to collect it
failed — learned k 25.64, constant k 29.41, the plug-in budget collects 2.4 of a
nominal 10.5, and three separate objective fixes each moved it by ≤0.18.
**Do not put 39.90 in the paper as a target.**

---

## 4. ASQA — the recipe does not generalise

```
  arm                  DR*   STR-EM  ROUGE-L    vs best single      vs count     vs ceiling
  count              24.77    37.38    24.51    -2.27 ( -8.4%)*   (reference)      -2.44 *
  weighted           27.45    46.01    22.04    +0.42 ( +1.6%)* +2.69 (+10.9%)*    +0.25 *
  pattern [CEILING]  27.20    45.36    22.05    +0.17 ( +0.6%)  +2.44 ( +9.8%)*   (ceiling)
  LEAN (dr-label)    26.88    48.25    20.19    -0.15 ( -0.6%)  +2.12 ( +8.5%)*     -0.32
  GBM (dr-label)     26.42    47.23    20.03    -0.61 ( -2.3%)  +1.65 ( +6.7%)*     -0.78
```

Includes the label fix flagged on 20 Aug (fitting to `DR(S+i) > DR(S)` rather
than to the recall-flavoured "covers an interpretation"). It helped by +0.07 and
did not change the verdict.

**On ASQA, plain reliability weighting is the best arm** — +10.9% over counting,
and the only arm beating the best single agent. Nothing outside the support
pattern helps, and the GBM early-stopped at 26–28 trees because 6,015 clusters is
far too little data.

This is consistent with the solo-ratio finding (20 Aug):

- **QAMPARI** — solo ratio 0.06, 110,801 candidates. Counting already filters
  well, weighting adds nothing, the wins come from rank + budgets.
- **ASQA** — solo ratio 0.47, 6,015 clusters. Counting cannot filter at all,
  weighting is the whole game, and there is not enough data to learn anything
  richer.

**The honest claim: which mechanism wins is a property of the answer format and
the candidate volume, not a universal aggregation rule.**

---

## 5. Files added

| file | what |
|---|---|
| `analysis/weighting_v2.py` | rank-conditioned NB, pattern⊕rank, budgets + the `adapt-count` control |
| `analysis/weighting_v3.py` | omission strength, rank buckets, question context, claim content; full ablation |
| `analysis/weighting_v4.py` | decision rule: isotonic calibration, 3 normalisations, top-K control, oracle bounds |
| `analysis/weighting_v5.py` | GBM scoring, logistic/GBM blend, budgets — **the champion**, `--seed` for stability |
| `analysis/weighting_v6.py` | RRF / Borda / CombSUM / CombMNZ, pairwise co-assertion, 2D lookups |
| `analysis/weighting_v7.py` | alias merging at 6 thresholds, with good/bad merge diagnostics |
| `analysis/weighting_v8.py` | 8-scorer stacked ensemble, bagged GBM |
| `analysis/weighting_v9.py` | budget objective: fitted saturation and precision scale |
| `analysis/gbm.py` | histogram gradient boosting, logistic loss, no new dependencies |
| `analysis/asqa_rank2.py`, `analysis/asqa_v5.py` | the recipe ported to ASQA |

---

## 6. Next

1. **Headline `BLEND + budget` vs best-single (+16.8%)**, MA-count (+9.1%) as
   secondary, `pattern` (+0.8%, ns) as the foil that makes both interesting.
   Report `RRF` (+2.9%) as the off-the-shelf baseline — a reviewer will ask.
2. **`LEAN + budget` is the version to put in the paper's method section**:
   +15.8% / +8.1%, linear, no boosting, far easier to describe. The GBM blend
   buys the last +0.3.
3. ~~Re-run the ensemble-size sweep with the budget arm.~~ **Done — §2.2.**
   210/210 ensembles, every size. This is probably the paper's central figure:
   it shows the mechanism holding across the whole ensemble-size range while the
   thing the project set out to test (reliability weighting) dies at n=6.
4. **ASQA's binding constraint is data volume, not method** — 6,015 clusters.
   Concrete argument for spending the $30 cap on more ASQA questions rather than
   more QAMPARI agents.
5. `az login` and the gpt-5.5 / gpt-5.6-sol temperature decision are still open
   from 20 Aug.
