# Robustness: overfitting, how k is chosen, and what predicts the gain

Answers the three concerns raised at the 21 Aug meeting. All cached data, zero
API spend. Code: `analysis/loso.py`, `analysis/gain_predictors.py`.

---

## 1. Overfitting — leave-one-source-out

**The concern.** "We are remembering patterns and that should artificially boost
performance, unless we have a separate training set and a separate testing
set... it might be relying on the specific characteristics of the immediately
present dataset."

**What was already in place.** Every number in the project is cross-fitted *by
question*, 5-fold: the pattern lookup, the logistic and GBM coefficients, the
gold-count regression and the keep threshold are all fitted on training folds and
scored on a held-out fold. So the literal version of the concern was already
handled.

**The sharper version was not**, and it is the right one to worry about: random
folds draw train and test from the *same* question distribution, so a rule
exploiting something peculiar to QAMPARI-as-a-whole would still look fine.

**The harder test.** QAMPARI ships five structurally different question sources.
Train on four, test on the fifth, five times — nothing about the held-out
question type is seen during fitting.

```
  arm                 random 5-fold      LOSO     shift     LOSO vs count
  count                       29.35     29.35     +0.00      (reference)
  pattern                     29.60     29.37     -0.23     +0.02 ( +0.1%)  ns
  LEAN                        30.79     30.73     -0.05     +1.38 ( +4.7%) *
  LEAN + budget               31.74     31.48     -0.26     +2.13 ( +7.3%) *
  BLEND + budget              32.02     31.44     -0.58     +2.09 ( +7.1%) *
```

**The method survives.** The distribution-shift penalty is 0.05–0.58 F1 against a
gain of 2.1–2.4, so roughly 80–90% of the gain survives a complete change of
question type. It remains significant against counting.

Per held-out source, gains are positive on all five:

```
  source                   n     count   BLEND+budget    gain
  wikidata_simple        327     25.13         26.41    +1.28
  wikidata_intersection  164     39.39         41.22    +1.84
  wikidata_comp          149     24.64         27.75    +3.11
  wikitables_composition  84     38.21         41.75    +3.55
  wikitables_simple       53     23.59         26.23    +2.64
```

### 1.1 Two things worth reporting from this

**The test has teeth — it catches the arm that *does* memorise.** `pattern` is a
2⁸-cell lookup table, exactly the kind of thing the concern describes. Under
random folds it is +0.25; under LOSO it collapses to **+0.02, ns**. So the
protocol detects distribution-dependent memorisation, and our features are not
what it flags.

**The GBM transfers worst.** `BLEND + budget` loses 0.58 under shift while
`LEAN + budget` loses 0.26 and ends up *ahead* of it (31.48 vs 31.44). The extra
capacity buys +0.3 in-distribution and costs more than that out of it.
**Recommendation: make `LEAN + budget` the headline method** — a plain L2
logistic plus the budget. Simpler to describe, one fewer moving part, and more
robust.

---

## 2. How k is chosen

**The concern.** "How is K calculated for step two? There's no way you can know
K — for QAMPARI, K is one of the unknowns."

Correct — we never know it. We *predict* it, and the method turns out not to need
the prediction to be good.

`Ĝ_q` is a ridge regression fitted **on training questions only**, using
question-level features available at inference: number of candidate claims, the
maximum support count on that question, how many candidates have ≥2 supporters,
the sum of predicted probabilities, and the maximum predicted probability. Then

```
k*(q) = argmax_k  2 · (Σ_{i≤k} p_i) / (k + Ĝ_q)
```

which is plug-in expected F1.

**Two measurements that answer the concern:**

| | |
|---|---|
| Ĝ accuracy | mean absolute error **4.62** answers, on a gold mean of 10.2 (range 5–40) |
| substituting the **true** gold count | 32.30 vs 31.75 — **only +0.55** |
| one **constant** k for every question, cross-fitted (picks k=8) | 29.41 vs count 29.35 — **no gain at all** |

So the estimate is poor and it barely matters; but the cut must *vary*, or the
whole mechanism is worth nothing. The claim to make is **"the keep-count must be
question-dependent, and even a noisy estimate of how it should vary captures most
of the value"** — not "we estimate k accurately."

---

## 3. Secondary analysis — what predicts the gain

**The ask.** "Given a set of models with these accuracies and how their
performance varies at a per-question level, can we predict how much improvement
we get?"

Yes. Regressing the gain on properties computable *before* running the method,
over all 210 QAMPARI ensembles:

```
  gain over BEST SINGLE   mean +2.30   range [+0.01, +4.79]
  gain over COUNTING      mean +3.84   range [+0.97, +9.74]

  multivariate fit, gain vs best single:  R² = 0.841,  leave-one-out R² = 0.817
```

Univariate correlations:

| predictor | vs best single | vs counting |
|---|---|---|
| **dominance** (best / mean of rest) | **−0.620** | **+0.687** |
| solo ratio | −0.595 | +0.328 |
| mean F1 | +0.345 | −0.460 |
| CV of F1 | −0.334 | +0.411 |
| ensemble size | +0.314 | −0.407 |
| spread (max − min) | −0.184 | +0.170 |
| per-question SD of member F1 | +0.101 | −0.129 |
| mean pairwise Jaccard of claim sets | −0.068 | −0.416 |
| union recall − best agent's recall | −0.050 | −0.344 |

**Dominance flips sign between the two tables, and that explains why they look so
different.** One dominant model makes counting bad — so we beat counting easily
(+0.687) — while simultaneously making the best single agent hard to beat
(−0.620). Table 1 and Table 2 are not two views of the same thing; they are
driven by the same variable in opposite directions.

Parsimonious model for the paper, three interpretable quantities:

```
  size + dominance + (union recall − best agent recall)     LOO R² = 0.642
  dominance alone                                           LOO R² = 0.368
```

---

## 4. Still open

- **Paper-exact ASC baseline.** Published ASC (arXiv:2405.13131) samples m≈50
  responses from ONE model, clusters atomic units, keeps clusters above a
  validation-tuned Θ, then composes survivors with an LLM. Our `count` differs on
  two of three counts (multi-model rather than multi-sample; no composition step)
  and matches on the third (Θ is cross-fitted). **It cannot be run on cached data
  — it needs new generation**, ~777 × m calls on one model. At m=20 with a cheap
  model this fits inside the $30 cap.
- **ASQA metric.** Currently DR* = √(ROUGE-L × STR-EM), a proxy. A faithful DR
  needs the RoBERTa-SQuADv2 reader for Disambig-F1. Flagged since 18 Aug.
- **Literature check** — whether the support-pattern/rank formulation duplicates
  existing work.
