# HANDOFF — current state, 16 August 2026 (evening)

**This supersedes `HANDOFF_2026-08-16.md`, which was written earlier the same day and
concluded the method does not work. That conclusion was based on two bugs in my own
analysis code. It is now partially reversed. Read this file, not that one.**

`HANDOFF.md` §1 (the method) and §2 (TriviaQA findings) remain accurate.

---

## 1. THE HEADLINE, and it changed late

On QAMPARI, **weighted atomic merging beats the best single model by +1.90 F1, and the
confidence interval excludes zero.** That is the first significant result in favour of the
method in this project.

```
  n = 198 complete questions, 5 Azure models, judge-free set-F1 grading

  best single model (grok-4.3)            F1 28.60
  ASC count filter        (theta=2)       F1 29.54    +0.94  [-1.28, +3.13]
  WEIGHTED filter         (theta=0.28)    F1 30.51    +1.90  [+0.34, +3.56]  *
  union, no filter                        F1 22.80    -5.82

  CONTROLS (this is what makes it credible)
  UNIFORM weights, no information         F1 29.54    identical to the count filter
  SHUFFLED weights, wrong models          F1 27.47

  WEIGHTED - UNIFORM     +0.96  [-0.21, +2.32]     <- the reliability effect
  WEIGHTED - SHUFFLED    +3.04  [+1.57, +4.69]  *  <- assignment carries real signal
  UNIFORM  - count(2)    +0.00  [ 0.00,  0.00]  *  <- fractional thresholds buy NOTHING
```

**How to read this.** Uniform weights reproduce the count filter exactly, so the gain is
not from having a continuous threshold instead of an integer one. Shuffling the same
weight *values* onto the wrong models costs 3.04 points, so the gain is not from the
magnitudes either — it is from **which model gets which weight**. That is a genuine
information effect.

**What is NOT established.** `WEIGHTED - ASC count = +0.96` still spans zero. So
"weighting beats counting" is unproven. Only "weighted merging beats the best single
model" clears. Do not overstate this.

---

## 2. The two bugs that hid it — both mine, both in `analyze_merge.py`

Fixed as of this file. Documented because they are the reason the earlier handoff is wrong.

**Bug A — no within-model deduplication.** `merged()` added a model's weight once per
*occurrence* of an item in that model's list. Models repeat items; each repeat voted
again. Support must be counted once per MODEL — the premise is "how many distinct models
assert this". Cost: ~1 F1 point, suppressing the weighted filter.

**Bug B — threshold sweep too coarse.** The grid stepped ~0.09 and jumped over the optimum
at theta=0.28, evaluating 0.26 and 0.35 and reporting 29.3 instead of 30.5. Sweep
resolution must be finer than the gap between the strongest solo weight (0.344) and the
weakest pair sum (0.298) — a 0.047-wide band, which is the only region where weighting can
behave differently from counting at all.

Both now fixed, with the reasoning in code comments. `analyze_merge.py` also now runs the
uniform and shuffled controls automatically.

**Note the direction.** Three earlier bugs this session all *flattered* the method. These
two *suppressed* it. So "my errors all favour the method" was itself wrong.

---

## 3. Exactly how to reproduce

```bash
cd ~/Documents/open-ended-aggregation
./venv/bin/python analyze_merge.py          # QAMPARI, ~4 min, prints sweeps + controls
```

Expect: best single 28.60, count 29.54, weighted 30.51, uniform 29.54, shuffled 27.47.
If uniform does not exactly equal count(2), the dedup fix has been lost.

---

## 4. NEXT STEPS, in priority order

### 4.1 — Replicate on a second benchmark. THIS IS THE BLOCKER.

The +1.90 is one benchmark, n=198. Everything else in this project that looked real at one
sample size flipped at another. `analyze_merge_asqa.py` and `analyze_facts.py` were written
**before** the dedup fix and therefore carry Bug A, and their sweeps carry Bug B.

```bash
# 1. port the two fixes into analyze_merge_asqa.py:
#    - add `seen=set()` per model inside the support loop (Bug A)
#    - replace the theta grid with np.linspace(0.10, 0.60, 26) scaled to that
#      benchmark's weight range (Bug B)
#    - add the uniform + shuffled controls
./venv/bin/python analyze_merge_asqa.py
```

**Prediction to write down before running:** ASQA will NOT replicate. Its EM/skill signal
is degenerate there (Spearman rho = +0.000 despite 18 points of capability spread) because
prose responses share almost no surface overlap. If it replicates anyway, that is
informative and means the mechanism is not what §1 claims.

Also rerun FACTS the same way.

### 4.2 — Extend QAMPARI to n=1000

Only 198 of 300 attempted questions came back complete; QAMPARI dev has 1,000. More n is
the cheapest way to move +0.96 (weighted vs count) off zero. Azure grant, roughly $25.

```bash
./venv/bin/python run_qampari.py 1000     # resumes; ~3-4 hours
./venv/bin/python analyze_merge.py
```

### 4.3 — Make the weights LABEL-FREE

Every number above uses **supervised** per-model precision, cross-fitted. That is not the
deployable configuration and a reviewer will say so immediately.

The good news: QAMPARI is the one benchmark where the label-free EM estimator works —
**Spearman +0.975, p=0.005, and it picks the correct best model with zero labels.** So
substituting EM betas for supervised precision is a direct test, and the estimator's
preconditions hold here.

Watch the sign convention: EM returns concentration betas, not precisions. They must be
mapped to a non-negative scale before use in a threshold filter (see 4.5).

### 4.4 — Cross-fit theta

Theta is currently swept on the evaluation set for every arm. It is applied identically to
all arms and the full curve is printed, so the comparison is fair — but the absolute
numbers are optimistic. Split: choose theta on 80%, report on the held-out 20%, rotate.

### 4.5 — The weight-form problem is unfinished

`logit(accuracy)` — the OW log-odds form used everywhere else in this project — goes
NEGATIVE whenever accuracy < 50%. On QAMPARI every model's precision is 8–35%, so a
Σβ ≥ Θ filter returned the **empty set at every theta** on the first run. We switched to
raw precision as a non-negative weight. That works but is not principled: the right form
is log-odds relative to the *actual base rate* that a proposed item is correct, not
relative to 50%. Deriving that is open.

---

## 5. What remains dead — do not revisit

- **Response-level weighting (KWA as originally specified).** Four benchmarks, never beat
  the best single model.
- **Per-question / per-domain weights.** Ours failed; a learned router failed at 67-model
  scale (arXiv 2606.27288 captures ~9% of available gain, CI spans zero).
- **Bayes-optimality of the KWA rule.** `agg_kernel` drops the logZ term `_agent_logp`
  computes. The correct normalised rule was implemented and tested: worse (−3.61 label-free
  on v2, CI excludes zero).
- **The pairwise error-correlation diagnostic (our 215–621x figure).** arXiv 2606.27288
  proves ρ cannot identify the quantity that actually binds.
- **Label-free model selection as the paper framing.** Occupied by the CAI Ratio, ICLR 2025
  workshop, arXiv 2509.08809.

---

## 6. Why the QAMPARI result is not contradicted by the co-failure ceiling paper

arXiv 2606.27288 (67 models, June 2026) proves accuracy ≤ 1−β for **"any selection
policy... whose output is almost surely one of the members' answers."** Merging assembles a
NEW answer from fragments across models, so it is **formally outside that theorem**. The
paper defines the boundary and stops there.

This is the strongest framing available: a rigorous, recent bound on everything selection-
based, with an explicitly delimited region where our result lives.

That paper's limitations section also states *"open-ended quality would reintroduce judge
bias"* — they restrict to programmatically-graded tasks. QAMPARI is programmatically graded,
so our result sits inside their evaluation regime, not outside it. That is a strength.

---

## 7. Documents that are now WRONG and need regenerating after 4.1

- `HANDOFF_2026-08-16.md` — concludes the method fails. Superseded by this file.
- `MENTOR_BRIEF.docx` / `MENTOR_BRIEF.js` — §5A says "our weights make it slightly worse".
  Now false: +1.90 over best single, CI excludes zero. **Do not present this version.**
- `ASC_WEIGHTED_REPORT.docx` / `build_asc_report.js` — same, reports weighted at −0.20.

Regenerate all three only after 4.1 tells you whether the effect replicates. If it does
not replicate on ASQA or FACTS, the honest headline is "one benchmark, unreplicated".

---

## 8. Environment, unchanged

- **OpenRouter is spent** (~$7.5 of $8) and the key needs rotating — it appeared in a prior
  transcript in plaintext.
- **Azure**: ~$1000 grant, ~$55 used. Quota, not money, is the constraint.
- **Deployed and reliable**: `grok-4.3`, `Kimi-K2.5`, `Cohere-command-a-plus-05-2026`,
  `MAI-Thinking-1`, `DeepSeek-V4-Flash`.
- **Deploymentless, works but vanished once** — judge only, never in the pool:
  `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.6-sol`.
- **Zero quota**: Anthropic, Mistral, Meta, and every newer variant.
- `gpt-5.5` / `gpt-5.6-*` reject `temperature=0`. `Cohere-command-a-plus` emits a literal
  `<|END_TEXT|>` token on ~100% of responses.

The ten silent-failure traps in `HANDOFF_2026-08-16.md` §5 are all still valid and worth
reading — they are the most reusable thing in that file. Two more from this session:
within-model duplicate counting, and threshold-sweep resolution finer than the
reordering window.

---

## 9. What to distrust

Five analysis bugs in one session. Three inflated the method, two suppressed it. The
lesson is not "the bugs go one way" — it is that **every headline number in this project
has been wrong at least once**, and the ones that survived did so because a control was
run, not because the code looked right.

Before anything external: re-derive the QAMPARI +1.90 from `data/qampari_gen.jsonl`
without reading `analyze_merge.py`. If it does not reproduce, this file is wrong too.

Specifically unverified: `Q.norm()` strips articles and punctuation aggressively. Nobody
has checked that it never merges two genuinely different QAMPARI entities. If it does,
precision is overstated for every arm including the baselines.
