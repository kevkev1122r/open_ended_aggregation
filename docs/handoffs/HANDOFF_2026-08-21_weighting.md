# Weighting mechanisms — overnight run, 21 August 2026

Continues `HANDOFF_2026-08-20.md`. Cached generations only, zero API spend.
Every number is 5-fold cross-fitted over questions on QAMPARI (777 q, 8 agents)
or 2-fold on ASQA (400 q, 6 agents); weights, likelihood tables, logistic and
GBM coefficients, calibration maps, ridge coefficients and keep thresholds are
all fitted on training folds only.

---

## 1. The headline

**QAMPARI, best method vs the two baselines that matter:**

```
  arm                        F1        vs best single          vs MA-count
  count (MA-count)        29.35      +1.95  ( +7.1%)*          (reference)
  pattern (CEILING)       29.60      +2.20  ( +8.0%)*   +0.25  ( +0.8%)  ns
  LEAN (logistic)         30.79      +3.39  (+12.4%)*   +1.44  ( +4.9%)*
  LEAN + budget           31.74      +4.34  (+15.8%)*   +2.39  ( +8.1%)*
  GBM                     29.78      +2.37  ( +8.7%)*   +0.43  ( +1.4%)  ns
  GBM + budget            31.56      +4.16  (+15.2%)*   +2.21  ( +7.5%)*
  BLEND                   30.60      +3.20  (+11.7%)*   +1.25  ( +4.3%)*
  BLEND + budget          32.01      +4.61  (+16.8%)*   +2.66  ( +9.1%)*
  ORACLE selection        59.34     +31.94 (+116.5%)   +29.99 (+102.2%)
```

`* = bootstrap 95% CI over questions excludes 0.`
**BLEND + budget: +16.8% over the best single agent (CI [+3.64, +5.58]) and
+9.1% over MA-count (CI [+1.86, +3.47]).** Both intervals clear zero by a wide
margin.

For scale: the `pattern` arm is the **best rule expressible from the support
pattern alone**, and it is +0.8% and not significant. Everything above it comes
from information that reliability weighting cannot see.

---

## 2. What actually produces the gain

Three axes, and only one of them is what this project has been calling
"weighting".

| axis | mechanism | worth (vs count) |
|---|---|---|
| who asserted | `pattern`, the ceiling on all support-pattern rules | +0.25, **ns** |
| **rank/verbosity/omission** | where the claim sat in each agent's own output, how much each agent said, how much each SILENT agent said | **+1.44** * |
| **per-question budget** | choose how many claims to keep per question, not a global threshold | **+1.19** * on its own |
| nonlinear scoring | GBM over the same features, blended with the logistic | +0.27 over the logistic |

The two big axes are close to independent — they combine to +2.66 — and
**neither is reliability weighting.** Model identity is worth about +0.3 on top
of either. That is the finding to build the paper on, and it is the opposite of
the framing we started with.

### 2.1 The single most valuable new feature: omission strength

Every rule before this treated silence as one thing. `nb-full` gave every silent
agent the same `log((1-TPR)/(1-FPR))`; the logistic gave it zero. But QAMPARI's
list lengths run from a median of 1 (gpt-5.4-nano) to a mean of 71 (Kimi), so:

> an agent that emitted 3 items and did not mention X has barely spoken;
> an agent that emitted 50 and did not mention X has said a great deal.

Adding per-agent `log(list length)` for the silent agents was the largest single
ablation win, +1.65 over counting on its own (`v3 +omission`, 31.01).

### 2.2 Per-question budgets, and the control that proves it is adaptivity

`k*(q) = argmax_k 2·(Σ_{i≤k} p_i)/(k + Ĝ_q)`, with `Ĝ` a ridge-predicted gold
count. Worth +1.19 **with no model identity at all** (`adapt-count`, on
`P(correct | k)` probabilities only).

The control: `top-K`, the same budget idea with one k for every question
(cross-fitted, picks k=8), scores **29.41 — no gain over counting**. So the
value is in the budget VARYING per question, not in keeping fewer things.

---

## 3. Things that did not work — all of them cheap negatives worth keeping

| idea | result | why it matters |
|---|---|---|
| error-correlation / diversity weighting | **λ = 0 selected in all 5 folds** | the supervisor's #4; it is a function of the support pattern, so the ceiling already caps it |
| question-domain weights `w_i(q)` | **+0.00** | QAMPARI's 5 question sources carry nothing; "condition on the question" ≠ topic |
| isotonic calibration of the score | 31.68 vs 31.71 | the logistic was already calibrated (3.55% predicted vs 3.55% actual) |
| within-question z-normalisation | **28.41, worse than counting** | destroys the cross-question information a global threshold needs |
| within-question percentile | **26.79, much worse** | same |
| learned per-question k (ridge on k*) | **25.64, much worse** | k* is essentially unpredictable from question features |
| kitchen-sink feature set (115 features, linear) | 30.60, **below its own ablations** | overfits; the LEAN subset is better |
| GBM alone | 29.78, below the logistic | better ranking (AP 0.397→0.418) but worse cross-question calibration; it only pays off blended or under a budget |

### 3.1 `oracle-k` = 39.90 is NOT available headroom

An earlier diagnostic reported that keeping the best prefix per question, using
the ordering the model already produces, scores 39.90 (+36%). **That is the
maximum of ~143 noisy prefix scores per question and is heavily
selection-biased.** Every honest attempt to collect it failed: a learned k
scored 25.64, a constant k scored 29.41, and the plug-in budget collects 2.4 of
the nominal 10.5. Do not put 39.90 in the paper as a target.

`ORACLE selection` (59.34) is a genuine ceiling but includes the ~7 F1 the
alias-poor gold costs every arm uniformly.

---

## 4. ASQA: the recipe does NOT generalise

Same features, same protocol, sentence-cluster unit, DR* scoring. Includes the
label fix flagged on 20 Aug — fitting to `DR(S+i) > DR(S)` rather than to "does
this cluster cover an interpretation", which was recall-flavoured.

```
  arm                     DR*  STR-EM  ROUGE-L    vs best single      vs count    vs ceiling
  count                 24.77   37.38    24.51    -2.27 ( -8.4%)*    (reference)   -2.44 *
  weighted              27.45   46.01    22.04    +0.42 ( +1.6%)*  +2.69 (+10.9%)* +0.25 *
  pattern  (CEILING)    27.20   45.36    22.05    +0.17 ( +0.6%)   +2.44 ( +9.8%)* (ceiling)
  LEAN (cover-label)    26.81   47.15    20.51    -0.22 ( -0.8%)   +2.05 ( +8.3%)* -0.39
  LEAN (dr-label)       26.88   48.25    20.19    -0.15 ( -0.6%)   +2.12 ( +8.5%)* -0.32
  GBM (dr-label)        26.42   47.23    20.03    -0.61 ( -2.3%)   +1.65 ( +6.7%)* -0.78
```

The label fix helped (+0.07) and did not change the verdict. **On ASQA, plain
reliability weighting is the best arm** — +10.9% over counting, and the only arm
that beats the best single agent. Nothing outside the support pattern helps, and
the GBM early-stopped at 26–28 trees because 6,015 clusters is far too little
data for it.

So the two benchmarks want opposite methods, which is consistent with the solo-
ratio finding from 20 Aug:

- **QAMPARI** (solo ratio 0.06, 110,801 candidates): counting is already a good
  filter, weighting adds nothing, and the wins come from rank + budgets.
- **ASQA** (solo ratio 0.47, 6,015 clusters): counting cannot filter at all,
  weighting is the whole game, and there is not enough data to learn anything
  richer.

**Claim it this way:** the mechanism that wins is a property of the task's answer
format and candidate volume, not a universal aggregation rule. That is a more
honest and more interesting paper than "our weighting scheme wins".

---

## 5. Files

| file | what |
|---|---|
| `analysis/weighting_v2.py` | rank-conditioned Naive Bayes, pattern⊕rank, per-question budgets + the `adapt-count` control |
| `analysis/weighting_v3.py` | omission strength, per-agent rank buckets, within-question context, claim content; full ablation |
| `analysis/weighting_v4.py` | the decision rule: isotonic calibration, three normalisations, top-K control, oracle diagnostics |
| `analysis/weighting_v5.py` | GBM scoring, logistic/GBM blend, budgets on top |
| `analysis/gbm.py` | histogram gradient boosting, logistic loss, ~200 lines, no new dependencies |
| `analysis/asqa_rank2.py` | ASQA with the DR-marginal label, budgets, learned budgets |
| `analysis/asqa_v5.py` | the full recipe ported to ASQA |

---

## 6. Next

1. **Report `BLEND + budget` against best-single as the headline** (+16.8%), with
   MA-count (+9.1%) as the secondary comparison. Both are what the paper needs;
   the pattern ceiling (+0.8%, ns) is the foil that makes them interesting.
2. **Re-run the ensemble-size sweep with the budget arm.** The 20 Aug crossover
   was measured on pattern rules; rank did not cross over, and budgets may not
   either.
3. **ASQA needs more data before anything rich can be tested there** — 6,015
   clusters is the binding constraint, not the method. This is a concrete reason
   to spend the $30 generation cap on ASQA rather than on more QAMPARI agents.
4. The `az login` and gpt-5.5/5.6-sol decisions from 20 Aug are still open.
