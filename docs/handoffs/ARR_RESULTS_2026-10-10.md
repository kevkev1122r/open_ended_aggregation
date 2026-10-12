# ARR results — final numbers for the paper (10 Oct 2026)

Every number below was produced on cached generations (no API spend) with the
**log-scale answer-count estimate** now used everywhere. All list-benchmark
numbers come from one script, `analysis/weighting_v5.py --bench <name>`, so the
main table, the ablation and the subset sweeps can no longer disagree.
`*` = paired bootstrap 95% CI over questions excludes 0 (10,000 resamples).

## 1. Main table

| Method | QAMPARI (set F1) | CWQ (set F1) | WebQSP (set F1) | ASQA (DR*) |
|---|---|---|---|---|
| Best single agent | 27.40 | 20.92 | 30.45 | 27.03 |
| Counting | 29.21 | 21.81 | 31.72 | 24.77 |
| Reliability weighting | 29.47 | 20.78 | 31.61 | **27.45** |
| Pattern lookup (ceiling on weighting) | 29.50 | 21.52 | 31.73 | 27.20 |
| ASC (published pipeline) | 28.43 | 21.62 | 29.41 | – |
| **Behavioural aggregation** | **31.54** | **23.35** | **32.75** | 26.81 |
| vs counting | +2.33 [+1.51, +3.14]* | +1.55 [+0.83, +2.30]* | +1.03 [+0.28, +1.85]* | +2.05 [+0.92, +3.22]* |
| vs best single | +4.14 [+3.19, +5.10]* | +2.44 [+1.38, +3.52]* | +2.31 [+1.35, +3.27]* | −0.22 (ns) |

- Questions / agents: QAMPARI 777 / 8, CWQ 415 / 5, WebQSP 375 / 5, ASQA 400 / 6.
- **ASQA, ours − reliability weighting: −0.64 [−1.31, +0.02], not significant**
  (reviewer KQ24). Story: wins on three list benchmarks; on ASQA it beats
  counting but is statistically tied with reliability weighting.
- Reliability weighting is the Naive Bayes rule over agents (assertions and
  omissions), as before; it reproduces the FAST paper's 29.47 / 20.78 exactly.
- ASC is paper-exact (edit-distance clustering, τ and Θ tuned on F1-5 on a
  validation half, as published), rebuilt on post-parsing-fix data; it sits
  below counting on all three list benchmarks. Tuning on F1-5 handicaps it under
  set F1 — say so in the caption. The old handoff's "22.21" was a 3-agent
  subset, not the full pool. Script: `asc_ensembles.py --bench qampari|cwq|webqsp`.

### Single-answer format: TriviaQA (2,757 questions × 6 agents, accuracy)

| Method | Accuracy | vs majority vote | vs reliability weighting |
|---|---|---|---|
| Best single agent | 86.29 | +1.31* | +0.36 |
| Majority vote | 84.98 | – | −0.94* |
| Reliability weighting (OW log-odds, paper-exact) | 85.93 | +0.94* | – |
| **Behaviour signals (self-consistency + verbosity)** | **86.69** | **+1.70 [+0.94, +2.50]*** | **+0.76 [+0.11, +1.45]*** |

Self-consistency = does the agent's terse answer appear in its own natural
answer; verbosity = length of the natural answer (2 words → 91% correct,
33 words → 65%). Script: `analysis/triviaqa_selfcons.py`.

## 2. Budget ablation (reviewer 5uMH: "show the budget on a non-behavioural scorer")

Gain over counting, same folds, same budget rule for every scorer:

| Scorer | QAMPARI | CWQ | WebQSP |
|---|---|---|---|
| Counting + budget | −0.90 | +0.24 | −0.51 |
| Reliability weighting + budget | −1.13* | −0.28 | −0.92* |
| Pattern lookup + budget | +1.39* | +0.59* | +0.31 |
| Behavioural scorer, no budget | +1.47* | +1.65* | +0.54 |
| **Behavioural scorer + budget** | **+2.33*** | **+1.55*** | **+1.03*** |

Reading: the budget alone does not produce the gain (it hurts counting and
weighting); the behavioural scorer is the main source, and the budget adds to it
on QAMPARI and WebQSP. On QAMPARI the budget also helps the pattern lookup.

## 3. Generality

**Unseen agents, no target labels** (`analysis/pool_transfer.py`, QAMPARI,
scorer trained on one agent group, applied to a disjoint group on different
questions):

| Setting | F1 | vs counting |
|---|---|---|
| 4 agents → other 4 (70 splits) | 27.63 | +4.91, significant 70/70 |
| 3 agents → other 5 (56 splits) | 28.14 | +4.57, significant 46/56 |
| 5 agents → other 3 (56 splits) | 26.29 | +6.46, significant 56/56 |
| Paper scorer trained on the target agents' gold (4 / 5 / 3 agents) | 28.13 / 29.21 / 26.54 | reference |

Script: `analysis/pool_size_transfer.py` → `results/pool_size_transfer.json`
(all three rows; the 3→5 and 5→3 rows were previously unsaved terminal output).
Counting fitted on the target agents' gold: 23.94 / 26.19 / 22.57; the
transferred scorer beats it in 70/70, 55/56, 56/56 splits.

**Unseen question types** (leave-one-source-out, QAMPARI): 31.10 vs 31.54 with
random folds; **+1.89 [+1.02, +2.73]* over counting, 81% of the gain retained**.
(FAST paper said 92% / 31.49 with the old count estimate — update the text.)

**Regime diagnostic — solo ratio** P(correct | 1 agent) / P(correct | ≥2):
CWQ 0.05, QAMPARI 0.06, WebQSP 0.09, ASQA 0.47, TriviaQA 0.75. Low ratio
(lists): counting already filters, weighting adds little, behaviour adds.
High ratio (ASQA, TriviaQA): weighting matters; on TriviaQA behaviour still
beats it.

## 4. Agent-subset sweeps (replaces `tab:ens-gain` and `tab:ens-sig`)

Mean gain over counting per size (significant / subsets).
Script: `analysis/subset_sweep_v5.py --bench <name>`; full-pool rows equal §1.

| Size | n | Best single | Counting | Weighting | Pattern | Beh. scorer | **Beh. aggregation** |
|---|---|---|---|---|---|---|---|
| *QAMPARI* | | | | | | | |
| 3 | 56 | 24.74 | 22.66 | +1.96 (36) | +2.33 (42) | +3.82 (56) | **+4.20 (56)** |
| 4 | 70 | 26.00 | 24.04 | +1.85 (40) | +2.41 (45) | +4.00 (70) | **+4.43 (70)** |
| 5 | 56 | 26.75 | 26.24 | +0.86 (21) | +1.30 (26) | +2.83 (56) | **+3.31 (56)** |
| 6 | 28 | 27.17 | 27.73 | +0.38 (7) | +0.67 (8) | +2.05 (28) | **+2.64 (28)** |
| total | 210 | | | 104/210 | 121/210 | 210/210 | **210/210** |
| *CWQ* | | | | | | | |
| 3 | 10 | 21.53 | 22.07 | +0.21 (2) | +0.28 (2) | +1.44 (10) | **+1.13 (6)** |
| 4 | 5 | 21.30 | 21.77 | +0.23 (1) | +0.33 (2) | +1.91 (5) | **+1.63 (4)** |
| 5 | 1 | 20.92 | 21.81 | −1.02 (0) | −0.29 (0) | +1.65 (1) | **+1.55 (1)** |
| total | 16 | | | 3/16 | 4/16 | 16/16 | **11/16** |
| *WebQSP* | | | | | | | |
| 3 | 10 | 30.40 | 30.42 | +0.08 (3) | +0.48 (5) | +1.33 (9) | **+1.48 (8)** |
| 4 | 5 | 30.45 | 31.33 | −0.00 (0) | −0.15 (0) | +0.70 (2) | **+1.12 (4)** |
| 5 | 1 | 30.45 | 31.72 | −0.11 (0) | +0.00 (0) | +0.54 (0) | **+1.03 (1)** |
| total | 16 | | | 3/16 | 5/16 | 11/16 | **13/16** |

Behavioural aggregation beats counting in **every** subset of all three
benchmarks (210/210, 16/16, 16/16), significantly in 210, 11 and 13.
(FAST paper said 13/16 significant on CWQ; it is 11/16 with one script.)

## 5. Appendix tables, regenerated on post-fix data

**Learning curve** (`analysis/learning_curve.py`, fixed 250-question test set):
gain over counting +0.46 at 50 training questions, +1.04 at 200, +1.64 at 350,
+1.61 with all 527.

**Position effect** (`analysis/rank_effect.py --bench`), solo assertions,
Spearman(rank, correct) within list-length bands, 1st vs last quintile:

| List length | QAMPARI | CWQ | WebQSP |
|---|---|---|---|
| 1–5 | +0.075 (3.1% vs 5.7%) | −0.005 | +0.132 |
| 6–20 | −0.056 (5.8% vs 3.0%) | −0.068 (9.2% vs 6.4%) | +0.017 (5.0% vs 5.1%) |
| 21–100 | −0.103 (5.5% vs 1.2%) | −0.087 (5.2% vs 2.6%) | −0.055 (16.5% vs 8.0%) |
| 101+ | −0.069 (0.9% vs 0.1%) | −0.081 (1.2% vs 0.1%) | −0.154 (4.1% vs 1.5%) |

Same pattern on all three: no signal (or reversed) on short lists, later items
worse on long lists. P(correct | first position) by band is flat on QAMPARI
(3.1, 5.6, 4.8, 4.4%), so the effect is positional, not verbosity.

**Alias merging** (`analysis/weighting_v7.py`), behavioural aggregation F1:

| τ | Candidates | Good / bad merges | F1 | vs no merge |
|---|---|---|---|---|
| 1.00 | 110,136 | 0 / 0 | 31.54 | reference |
| 0.95 | 108,039 | 0 / 1 | 31.49 | −0.05 |
| 0.90 | 105,495 | 7 / 10 | 31.45 | −0.09 |
| 0.85 | 102,086 | 7 / 41 | 31.29 | −0.25 |
| 0.80 | 94,390 | 8 / 106 | 31.34 | −0.20 |
| 0.70 | 81,547 | 15 / 414 | 30.46 | −1.08 |

**Rules that did not work** (`tab:negative`, QAMPARI, vs counting unless noted):

| Rule | Result | Source |
|---|---|---|
| error-correlation weighting | λ = 0 in 5/5 folds | beyond_pattern |
| question-domain features | +0.11 (ns) | beyond_pattern |
| rank agreement across agents | +0.09 over the behavioural scorer (ns) | weighting_v6 |
| pairwise co-assertion | −0.04 over the behavioural scorer | weighting_v6 |
| Borda / CombSUM / weighted Borda | −0.40 to −0.53 (only RRF survives, +0.84*) | weighting_v6 |
| within-question z-normalisation | −1.57* | weighting_v4 |
| within-question percentile norm. | −2.93* | weighting_v4 |
| learned per-question k | −3.26* | weighting_v3 |
| single constant k (picks 8–9) | +0.25 (ns) | weighting_v4 |
| alias merging, six thresholds | −0.05 to −1.08 | weighting_v7 |
| isotonic calibration | +0.12 over the budget (ns) — **sign changed from −0.03; update the wording** | weighting_v4 |

## 6. Text changes this implies (FAST draft → ARR)

- QAMPARI 31.70 → **31.54**; +2.50 → **+2.33 [+1.51, +3.14]**; vs best single +4.30 → **+4.14**.
- CWQ 23.65 → **23.35** in every table; +1.85 → **+1.55 [+0.83, +2.30]**; vs best single +2.74 → **+2.44**.
- Add WebQSP (32.75, +1.03*) and TriviaQA (§1).
- LOSO: 31.49 / ~92% → **31.10 / 81%**.
- CWQ sweep: 13/16 → **11/16** significant (16/16 positive); add WebQSP 13/16.
- Method section: the answer count is estimated on a log scale ("gold answer
  counts are heavy-tailed; WebQSP averages ~76").
- Drop every "pending regeneration" note.

## 7. Reproduce

```
./venv/bin/python -m open_ended_aggregation.analysis.weighting_v5 --bench qampari|cwq|webqsp   # main + ablation (+ boosted arms; --no-gbm to skip)
./venv/bin/python -m open_ended_aggregation.analysis.subset_sweep_v5 --bench qampari|cwq|webqsp
./venv/bin/python -m open_ended_aggregation.analysis.asqa_v5          # ASQA column + ours vs weighting
./venv/bin/python -m open_ended_aggregation.analysis.triviaqa_selfcons
./venv/bin/python -m open_ended_aggregation.analysis.loso
./venv/bin/python -m open_ended_aggregation.analysis.learning_curve
./venv/bin/python -m open_ended_aggregation.analysis.asc_ensembles --bench qampari|cwq|webqsp
./venv/bin/python -m open_ended_aggregation.analysis.rank_effect --bench qampari|cwq|webqsp
./venv/bin/python -m open_ended_aggregation.analysis.weighting_v7      # alias table
```

Logs: `results/main_*_final.log`, `results/sweep_*.log`, `results/regen_*.log`,
`results/loso_logG.log`, `results/learning_curve_logG.log`.
