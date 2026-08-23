# Paper skeleton — open-ended aggregation

> **How to use this.** Section order follows the Algoverse guide, including its
> preference for Related Work as the **second-to-last** section. Blockquoted notes
> are instructions from the guide or open decisions — delete them as you write.
> Bracketed `[...]` items are placeholders. Everything not bracketed is settled
> content whose numbers are cross-fitted and reproduce from cached generations.

> **Guide rules that apply everywhere:**
> - Every claim needs a citation, including broad framing claims.
> - No logical leaps — each step must follow immediately from the previous one.
> - Do not exaggerate novelty. Reviewers see through it.
> - Every statement about the literature must be *true*; verify, don't assume.
> - Figure and table captions must be **self-contained** and state the takeaway.

---

## Title

`[Working title — needs to name the mechanism, not the benchmark]`

Candidates:
- Reading How an Agent Answered, Not Just Which Agent Answered
- Support Patterns Are Not Enough: Output-Structure Signals for Multi-Model Claim Aggregation
- A Ceiling on Reliability Weighting for Set-Valued LLM Aggregation

---

## Abstract

> Structure: (a) topic + motivation, (b) background/context, (c) method,
> (d) results, (e) takeaway. Roughly one to two sentences each.

- **(a) Topic + motivation.** Ensembling multiple LLMs is standard, but aggregation still uses majority voting, designed for single-label prediction. Most LLM output is set-valued.
- **(b) Context.** `[Optimal-weight aggregation is defined for discrete answers; atomic self-consistency handles sets but only within one model.]`
- **(c) Method.** `[Score each claim using the structure of each agent's own output — its position in that agent's list, and how much each silent agent said — then choose a per-question keep-count.]`
- **(d) Results.** `[+15.8% over the best single agent, +8.1% over multi-agent counting; significant in 210/210 ensembles; 80–90% of the gain survives leave-one-source-out.]`
- **(e) Takeaway.** `[Reliability weighting is provably capped on set-valued aggregation; output structure is what carries the remaining signal.]`

---

## 1 Introduction

> Must follow the five-part structure. The end must explicitly list contributions.

### 1.1 Motivation
- Multi-model ensembles are standard practice; the aggregation step is still majority voting `[cite]`.
- Real LLM outputs are frequently **sets of claims** — entity lists, long-form answers with many separable facts `[cite QAMPARI, ASQA]`.
- Voting on sets is actively harmful: strict majority scores 23.11 against 27.40 for the best single model on QAMPARI. Naive aggregation is worse than not aggregating.
- Practical stake: 8× the inference cost for a loss.

### 1.2 Context and the gap
- **Weighted voting** derives provably optimal weights, but for binary/categorical predictions `[cite 2510.01499]`.
  - Not merely weak on set-valued answers — **undefined**. Its weight `log(p/(1−p))` requires per-model accuracy > 0.5; per-*claim* precision here is 5.8–28%, so every weight goes negative and the rule inverts.
- **Atomic self-consistency** handles set-valued answers, but draws repeated samples from one model, so model identity does not exist as a variable `[cite 2405.13131]`.
- **Response-level selection/fusion** degenerates: with open-ended answers no two responses coincide, so selection reproduces best-single exactly (measured: zero-width CI) `[cite LLM-Blender]`.
- **The gap:** no principled weighting scheme for heterogeneous models producing set-valued answers.

### 1.3 Insights and research questions
- **RQ1.** Can reliability weighting improve multi-model aggregation of set-valued answers?
- **RQ2.** If not, what information does help — and where exactly is the limit?
- **RQ3.** Can we predict, before deploying an ensemble, how much aggregation will help?

Insights:
- **A provable ceiling.** Every weighting rule is a function of the support pattern, so a direct lookup of `P(correct | exactly this subset asserted)` upper-bounds all of them — including correlation- and omission-aware variants, since the pattern also determines who stayed silent. Measured: **+0.25, n.s.**
- **The ceiling binds only one class of rule.** It says nothing about an agent's own output structure.
- **The decision rule is a separate axis.** Gold counts vary 5–40 per question, so one global threshold is wrong nearly everywhere.

### 1.4 Method summary
- `[2–3 sentences: score claims on support pattern + position + omission strength; then choose k per question by plug-in expected F1.]`

### 1.5 Contributions
> Guide: explicitly list contributions at the end of the introduction.

1. A ceiling result establishing that reliability weighting on set-valued aggregation is capped at +0.25 (n.s.), proven by construction rather than observed as a failure.
2. An aggregation method that clears the ceiling: +15.8% over the best single agent and +8.1% over multi-agent counting, using only a logistic regression and a closed-form keep-count.
3. Evidence across all 210 agent ensembles of size 3–6, significant in every one.
4. A robustness result: the gain survives leave-one-source-out, while the ceiling arm collapses under the same shift.
5. A predictor of the gain from pool properties alone (leave-one-out R² = 0.817).
6. A cheap dataset screen — the solo ratio — that predicts which regime a benchmark is in.

---

## 2 Background

> Optional per the guide, but we need it: this is where **all notation** is
> introduced, and the Method section may not use notation defined anywhere else.

- **Task.** `[n agents, question q, each returns an ordered list; pool distinct normalised claims into candidate set C(q).]`
- **Notation to define:** `[agents m ∈ M; claim c; assertion indicator z_m(c); support count; support pattern S(c) ⊆ M; position r_m(c); list length L_m(q); gold set G(q); set F1.]`
- **Metrics.** `[QAMPARI set F1; ASQA DR; TriviaQA accuracy. Define DR and its proxy explicitly.]`
- **Baselines defined here or in Experiments:** multi-agent count, atomic self-consistency, best single agent.

---

## 3 Method

> Guide: procedural and precise. No metaphors, no unsubstantiated intuition —
> that belongs in the Introduction. Readers spend most of their time here.

### 3.1 The support pattern and its ceiling
- Definition of the support pattern.
- Statement: any rule reading only the support pattern is bounded by the pattern lookup.
- `[Short argument — one paragraph, no hand-waving. This is a construction, state it as one.]`

### 3.2 Scoring claims from output structure
- Feature groups: support pattern; position within each asserting agent's own list, with per-agent position bands; **omission strength** (each silent agent weighted by `log` of how much it said); the pattern probability stacked in as a feature.
- Model: L2-regularised logistic regression. `[State the objective and regularisation.]`
- `[State that gradient boosting over the same features was tested and REJECTED: it gains +0.3 in-distribution but loses 0.58 under distribution shift, so the plain logistic wins out of distribution. Report as an ablation, not as part of the method.]`
- `[Note explicitly: verbosity of ASSERTING agents is excluded — it is redundant with position and contributes +0.01. Include the controlled table in the appendix.]`

### 3.3 Per-question keep-count
- `k*(q) = argmax_k 2·(Σ_{i≤k} p_i) / (k + Ĝ_q)` — plug-in expected F1.
- `Ĝ_q` is a ridge regression on question-level features, fitted on training questions only.
- `[State that accuracy of Ĝ is not required — MAE 4.62 against a gold mean of 10.2, and the oracle count gains only +0.55 — but that the cut must vary; a constant k gains nothing.]`

### 3.4 Cross-fitting protocol
- All coefficients, the gold-count regression and every threshold are fitted on training folds and scored on held-out questions.
- `[Say why this matters here: in-sample threshold tuning flips a headline from +0.08 to −1.25 on this data.]`

> **Figure 1 — method diagram.** Needed. Suggested content: one question, a few
> agents' ordered lists, the pooled candidate set, the two signals highlighted
> (position, omission), then the per-question cut. Tools the guide suggests:
> draw.io, vecta.io.

---

## 4 Experiments

> Guide: describe the "what" in detail, not the basics of the "how".

### 4.1 Benchmarks
- **QAMPARI** — 777 questions, list-of-entities answers, set F1. `[source, gold format, alias caveat]`
- **ASQA** — 400 questions, long-form; sentence-cluster atoms. `[state DR proxy and its limitation explicitly]`
- **TriviaQA** — 2,757 questions, single short answer. `[note position and budget are undefined here]`

### 4.2 Agent pool
- 8 agents on QAMPARI, 6 on ASQA and TriviaQA. `[table of models and per-agent scores]`
- `[Disclose: single sample per agent, temperature setting, and that all generations are cached.]`

### 4.3 Baselines
- **Best single agent** — the strongest pool member.
- **Multi-agent count** — keep a claim if ≥ θ agents assert it, θ cross-fitted. Our proposed naive method, not a published one.
- **Atomic self-consistency** — published pipeline; edit-distance clustering, member-count strength, θ tuned for F1-at-recall-5. `[Disclose the one substitution: agents replace stochastic samples, because the cache holds one response per agent. Also disclose the omitted LLM composition step.]`
- **Reliability weighting family** — marginal, solo, Naive Bayes with and without silence, correlation-aware, and the pattern lookup as the ceiling.
- **Rank fusion** — Reciprocal Rank Fusion, Borda, CombSUM.

### 4.4 Evaluation protocol
- 5-fold cross-fitting over questions (2-fold for per-ensemble sweeps).
- Paired bootstrap over questions, 10k resamples, 95% CI.
- Ensemble sweep: all 210 subsets of sizes 3–6.

---

## 5 Results and Analysis

> Guide: answer the RQs directly, don't number-dump, make every caption
> self-contained and state its takeaway.

### 5.1 RQ1 — Does reliability weighting help? (No, and here is the bound)
- Main table: best single / atomic self-consistency / count / the weighting family / the pattern ceiling.
- Takeaway: the ceiling is +0.25, n.s. `[State this as the answer to RQ1.]`

### 5.2 RQ2 — What does help?
- Main results table: our method vs both baselines, all three benchmarks.
- Ablation: position, omission strength, position bands, stacking, and gradient boosting (rejected — see §5.3).
- Ensemble sweep by size, n = 3 to 6, plus the significant-win counts (210/210).
- `[Include the crossover reading: weighting decays and goes negative by n=6; ours does not.]`

### 5.3 Robustness — is this overfitting?
- Leave-one-source-out table, with the shift column.
- Key point: the pattern lookup collapses to +0.02 n.s. under the same shift, so the test detects memorisation and our features are not what it flags.
- `[Decide: does this go here or in its own section? It answers a reviewer question, so probably here.]`

### 5.4 RQ3 — Predicting the gain
- Regression of gain on pool properties; leave-one-out R² = 0.817.
- Dominance flips sign between the two comparisons — the reason the two tables differ.

### 5.5 When does this transfer? The solo ratio
- Solo ratio across QAMPARI 0.06, TriviaQA 0.24, ASQA 0.47, monotone with weighting's value.
- `[Strongest form of this, added 23 Aug: the per-ensemble sweeps are mirror images. On QAMPARI our method beats weighting in 210/210 ensembles; on ASQA weighting beats ours at every size and the gap widens with n. Same two methods, opposite verdicts, predicted in advance by the solo ratio.]`
- `[Be careful here: three points is an observation, not an established law. Say so.]`

### 5.6 Negative results
- Error-correlation weighting (λ = 0 in 5/5 folds); question-domain weights (+0.00); pairwise co-assertion (+0.05); within-question normalisation; learned per-question k; alias merging.

---

## 6 Related Work

> Guide: 10–20 papers, general → specific → more specific, three axes, and a
> brief statement of how ours differs. Placed second-to-last deliberately.
> Avoid laundry lists; don't use citations as nouns; be respectful.

### 6.1 Aggregating outputs from multiple models
- Trajectory: majority voting → reliability-weighted voting → optimal weights from higher-order information; parallel branch selects or fuses whole responses.
- **Contrast: discrete vs set-valued.** `[Ours takes this line's open question but the estimand must change.]`

### 6.2 Decomposing long-form answers into atomic claims
- Trajectory: self-consistency → universal self-consistency → atomic self-consistency → atomic decomposition as general infrastructure for factuality evaluation and calibration.
- **Contrast: one model vs many.** `[Identity becomes available for the first time — and is worth less than expected.]`

### 6.3 Estimating per-item reliability without labels
- Trajectory: crowdsourcing annotator models → ported to deep ensembles → label-free behavioural signals in LLMs (length, perturbation stability, input-side position effects) → conformal adaptive set size.
- **Contrast: response-level vs claim-level, and asserters vs abstainers.**
- `[Credit Dawid–Skene explicitly and present our pattern/Naive-Bayes arms as ITS baselines, not our contributions.]`

---

## 7 Conclusion

> Guide: just summarise takeaways, specific → broad. No future work, no
> limitations here — they belong below.

- `[Specific: the ceiling, then the method that clears it, then the breadth of the ensemble evidence.]`
- `[Broad: when the unit of aggregation becomes the claim, who said it matters less than how they said it.]`

---

## Limitations

> Required as a separate section for *ACL venues; sits below the Conclusion and
> does not count toward the page limit.

- The method's full form is demonstrated on one benchmark; the per-question budget does not transfer to ASQA, and position/budget are undefined for single-answer tasks.
- Atomic self-consistency is run with agents substituted for stochastic samples; the single-model form is not evaluated.
- The ASQA metric is a proxy for the published DR; a faithful score needs the RoBERTa-SQuADv2 reader.
- QAMPARI gold ships no aliases for most entries, which depresses every arm roughly uniformly.
- The solo-ratio screen rests on three benchmarks.
- All agents are proprietary API models at a single sampling setting.

---

## Appendix

- A. Full per-ensemble results, all 210 QAMPARI and 42 ASQA ensembles.
- B. Feature definitions and the controlled length-vs-position analysis.
- C. Per-agent statistics: scores, list lengths, solo precision.
- D. Prompts used for generation.
- E. Hyperparameters: regularisation, folds, bootstrap settings, thresholds.
- F. Extended negative results.

---

## Pre-submission checklist

> From the guide's final review section.

- [ ] Line numbers enabled
- [ ] Anonymous submission; no acknowledgements section
- [ ] GitHub/HuggingFace links anonymised (`anonymous.4open.science`)
- [ ] Conference style template applied; page limit respected
- [ ] Every table/figure caption self-contained with a stated takeaway
- [ ] Every claim carries a citation that actually supports it
- [ ] All literature statements verified — 15 of 23 references still need direct reads
- [ ] Mentor review, then P.I. review booked at least one week before deadline

---

## Open decisions

1. ~~Headline method.~~ **Settled 23 Aug: rank scorer + budget, no boosting.** Leave-one-source-out showed the boosted blend transfers worse; boosting is now reported as a rejected ablation.
2. **Lead with the ceiling or the method?** The ceiling is the more novel framing.
3. **Conference target**, which fixes the template, page limit and whether a Limitations section is required.
4. **Does Background stay?** Only if the notation load justifies it.
