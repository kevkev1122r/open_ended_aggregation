# Code map

3,001 lines of Python. Read in this order.

## Core (read these three to understand the project)

| file | lines | what |
|---|---|---|
| `kernel_agg.py` | 357 | **The library.** Generative model, all aggregators, the label-free EM estimator for β, and `test_reduction()` which proves KWA == the paper's OW when the kernel is exact match. |
| `generate.py` | 199 | **Real data collection.** Queries 6 models via OpenRouter on TriviaQA, two prompt settings, resumable, incremental JSONL, spend tracking. Reads the key from `.env` (not included). |
| `analyze_real.py` | 244 | **The real experiment.** Builds per-question candidate pools from actual generations, runs every aggregator, scores against TriviaQA alias lists. |

## Synthetic pilot

| file | lines | what |
|---|---|---|
| `experiments.py` | 495 | Experiments E1–E11: correctness, β recovery, support bias, main comparison, vote splitting, distractor proximity, misspecification, ceiling, triangulation, deployable pipeline, real-geometry rerun. |
| `weak_signal.py` | 71 | Whether a weak per-answer signal is still aggregatable. **Contains a bug** — its AUC measure indexes the wrong pool; superseded by `gap_sweep.py`. |
| `gap_sweep.py` | 60 | The corrected version: sweeps the within-question gap that actually drives aggregation. |

## Validating the assumption on real text

| file | lines | what |
|---|---|---|
| `real_data_test.py` | 229 | First real-data test (TruthfulQA + HaluEval + MiniLM): do errors cluster near the truth, is the law log-linear. |
| `robustness_test.py` | 259 | Three stress tests: encoder robustness, task family, and the control-design attack that broke the headline claim. |
| `control_gradient.py` | 59 | Sweeps control hardness continuously — showed most of R²=0.99 was topic-matching. |
| `triple_replication.py` | 214 | Same protocol on 3 independent corpora × 3 encoders with bootstrap CIs. |
| `regression.py` | 165 | Proper logistic regression with question-clustered SEs and LR tests. **Corrects two earlier conclusions.** |
| `separation.py` | 48 | AUC — how well similarity actually separates real errors from controls. |

## Figures

`make_figures.py`, `real_figures.py`, `regression_figure.py`, `replication_figure.py`,
`gradient_figure.py`, `explain_figure.py`, `teach_figure.py`, `teach_bins.py`

## Documents

`build_plan.js` → `PLAN.docx` · `build_report.js` → `REPORT.docx` · `FINDINGS.md` · `README.md`

## Run it

```bash
python3 -m venv venv && ./venv/bin/pip install numpy scipy matplotlib pandas pyarrow \
    requests sentence-transformers statsmodels
echo 'OPENROUTER_API_KEY=sk-or-...' > .env      # only needed for generate.py

./venv/bin/python experiments.py          # synthetic suite, ~200s
./venv/bin/python real_data_test.py       # assumption check on real text
./venv/bin/python triple_replication.py   # 3 corpora x 3 encoders, ~12 min
./venv/bin/python regression.py           # the proper statistics
./venv/bin/python generate.py pilot       # 50 questions, ~$0.01
./venv/bin/python analyze_real.py full    # the headline experiment
```

## Known defects

- `weak_signal.py` measures the wrong AUC (see above).
- Grading is substring alias matching; **12.8% of full-sentence responses are truncated** by the token cap, and false negatives on single-alias questions are unquantified. See REPORT.docx §7.
- `analyze_real.py` passes `round(K_eff)` to the accuracy estimator but unrounded `K_eff` to the weights — a minor inconsistency.
