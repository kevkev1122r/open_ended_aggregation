# Open-ended LLM answer aggregation — Kernel-Weighted Aggregation (KWA)

Pilot study answering the open question in Ai, Pan, Simchi-Levi, Tambe & Xu,
*Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information*
(ICML 2026, arXiv:2510.01499), §7: **"How to derive optimal weights for open-ended
questions?"**

> **Real-data status** (FINDINGS.md §0, §0.5, §0.6). Replicated on **3 independent corpora
> × 3 encoders = 9 cells** with bootstrap CIs. The *effect* is bulletproof: 9/9 cells,
> Cohen's d 1.43–3.14, placebo passes 9/9. The *log-linear form* is **refuted in 6/9 cells**
> with disjoint CIs — and the split is by corpus, not encoder: it fails for model-generated
> errors, holds only for hand-written exam distractors. β is not a constant (2.6–20.6).
> **The kernel must be curved. All accuracy numbers remain synthetic.**

## The idea in three lines

```
  their OW:   argmax_s  Σ_j  w_j · 1{a_j = s}          needs "all wrong answers equally likely"
  our KWA:    argmax_s  Σ_j  β_j · sim(a_j, s)          replaces the indicator with a kernel
  reduction:  sim = exact match  ⟹  KWA IS OW           verified numerically, 1.0000 at every K
```

β is estimated from unlabelled data by EM with the truth as a latent variable.

## Headline results

| | |
|---|---|
| Reduces exactly to OW on multiple choice | 1.0000 agreement, K ∈ {2,3,4,6,10} |
| Label-free β recovery | corr 0.98 (strongest agent underestimated ~20%, does not improve with data) |
| vs cluster-then-vote, oracle support | 96.1% vs 92.0% |
| **vs cluster-then-vote, fully deployable** | **94.7% vs 93.1%** ← the honest number |
| Correct on questions where *every* agent was wrong | **34.1%** (vote-based methods: 0% by construction) |
| Errors land near the truth | **9/9 cells**, Cohen's d 1.43–3.14 |
| Placebo (shuffled labels) | passes 9/9, max \|β\| 0.277 vs real β 2.6–20.6 |
| **Log-LINEAR form** | **REFUTED 6/9, CIs disjoint — fails exactly where errors are model-made** |
| β a constant? | **No — 2.6 to 20.6** across corpora/encoders/pool spread |
| vs *tuned* baseline under real measured geometry | **93.3% vs 88.3% (+5.0)** |

The main hypothesis was **refuted**: the gain is not from pooling votes split across
paraphrases (it's largest when there are *zero* paraphrases). The mechanism is
triangulation — wrong answers collectively point at the truth.

## Files

| file | what |
|---|---|
| `PLAN.docx` | the research plan, with pilot results and figures |
| `FINDINGS.md` | full write-up, including what broke |
| `kernel_agg.py` | library: generative model, aggregators, EM estimator |
| `experiments.py` | the ten experiments (E1–E10) |
| `make_figures.py` | summary figures from results.json |
| `robustness_test.py` | the three stress tests (encoder / task / control design) |
| `control_gradient.py` | the control-hardness sweep that broke the linear claim |
| `triple_replication.py` | 3 corpora × 3 encoders × bootstrap CIs — the reliability run |
| `real_data_test.py` | the real-data validation (TruthfulQA + HaluEval + MiniLM) |
| `explain_figure.py` | the dartboard explainer diagram |
| `results/results.json` | raw numbers |
| `results/full_run.log` | complete console transcript |
| `results/figures/` | six PNGs |

## Run it

```bash
./venv/bin/python experiments.py           # everything, ~200s
./venv/bin/python experiments.py E4 E9     # selected experiments
./venv/bin/python make_figures.py
node build_plan.js                         # rebuild PLAN.docx
```

## Next, in priority order

1. **Replace the linear kernel with a curved one.** §0.5 shows `β·sim` is the wrong form exactly
   where the aggregator operates; a quadratic captures 98% of the variance there. Fit a monotone
   `g` and re-run the whole pilot with `β·g(sim)`.
2. **Chase the triangulation result** — build a real candidate generator, see if 34% survives.
3. **Add confidence-weighting baselines** (CISC, inverse-entropy voting) — the real incumbents.
4. Fix strong-agent identifiability (prior on β, or multiple samples per agent).
5. Diversity-aware extension — correlated agents are the main threat.
