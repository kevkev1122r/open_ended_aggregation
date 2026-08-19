# RESULTS — the niche-specialisation experiment

Run 2026-08-14. This is the experiment §3 of `HANDOFF.md` describes as "designed, built,
and unrun". It is now run.

**One-line answer: specialisation is real. Weighting helps. The kernel adds little once
weights are present, and the label-free estimator — the part that makes any of this
deployable — performs worse than plain majority voting.**

> ### Correction, and it matters
>
> The first version of this document reported that weighting cannot beat the best single
> model, based on `analyze_domains.aggregation()`. **That function never runs the method.**
> It compares answers by exact string identity of the last 120 characters and never imports
> `kernel_agg`. On this data 94.8% of questions have all seven answers as distinct strings, so
> every "majority vote" was a 7-way tie broken at random and every "weighted vote" collapsed
> to picking the single highest-weighted model:
>
> ```
>   "majority vote"      81.13  ==  mean model accuracy (random pick)
>   "global weights"     83.61  ==  always pick the globally best model
>   "per-domain weights" 84.33  ==  pick the best model per domain
> ```
>
> No consensus information appears anywhere in those numbers, and no kernel. §4 below is the
> corrected comparison, run by `analyze_kernel.py`. Treat any figure from
> `analyze_domains.aggregation()` as measuring model *selection*, not aggregation.

> ### Second correction — the weights were supervised
>
> `analyze_kernel.py` estimated beta from `piv.mean()`, i.e. accuracy measured against the
> judge's labels. **That is not the method.** KWA's beta is supposed to come from
> `kernel_agg.em_estimate_beta`, which infers it label-free from the agreement structure and is
> the project's headline deliverable. Re-run with the real estimator:
>
> ```
>   KWA, supervised beta (first reported)   84.85   +0.62 vs best single
>   KWA, LABEL-FREE beta (EM)               82.58   -1.65 vs best single
>   medoid / cluster, no weights at all     82.99   -1.24
>   best single model                       84.23
>
>   label-free - supervised   -2.27  [-4.02, -0.52]   significant
> ```
>
> The label-free weights are worse than **no weights**. Every KWA figure in §4 below is the
> supervised variant and overstates the deployable method.

---

## 0.  THIRD CORRECTION — read before any number below

Two measurement bugs of mine inflated everything in section 4, and fixing them reverses the
headline.

1. **The majority-vote baseline was random selection.** `agg_majority_exact` was called with
   `ans = arange(7)`, giving every answer its own candidate index, so every question was a
   7-way tie broken at random. The "+1.65, the kernel earns its place" was measured against
   that. Identical answers must share an index.
2. **Answers were never deduplicated or normalised.** `Cohere-command-a-plus` emits a literal
   `<|END_TEXT|>` control token on 964 of its 970 answers, so it could never match anyone.
   Markdown bold (474), trailing periods (967) and LaTeX delimiters (263) fragment the rest.

With both fixed — normalise, dedupe so identical answers share a candidate, then aggregate:

```
  majority vote (real, deduped)     82.99   -1.24 vs best single
  medoid / cluster (kernel only)    83.51   -0.72
  OW exact (weights only)           84.74   +0.52   <- BEST
  KWA supervised beta               84.54   +0.31
  KWA label-free beta (the method)  82.58   -1.65   <- below plain majority voting
  best single model                 84.23
  ceiling                           94.02
```

**Consequences.** The kernel is worth +0.52 over a real majority vote, not +1.65. **OW exact
beats KWA**, so section 2's "KWA is never better than the better single ingredient" DOES
reproduce — my earlier claim that it did not was an artifact of the broken baseline. And the
label-free method scores below plain majority voting.

Normalisation also collapsed spurious clusters: among questions where all seven models are
correct, the share forming a single cluster rose from 5.4% to 39.7%. It did **not** rescue the
EM estimator (Spearman +0.473 raw, +0.445 deduped, +0.178 cleaned-without-dedup) — the
catch-22 in section 9 stands.

---

## 1. What was run

Seven models, seven labs, one model per lab, across two providers because Azure had zero
quota for Anthropic, OpenAI, Mistral and Meta.

| model | lab | provider |
|---|---|---|
| `anthropic/claude-sonnet-5` | Anthropic | OpenRouter |
| `openai/gpt-5.4` | OpenAI | OpenRouter |
| `grok-4.3` | xAI | Azure |
| `Kimi-K2.5` | Moonshot | Azure |
| `Cohere-command-a-plus-05-2026` | Cohere | Azure |
| `DeepSeek-V4-Flash` | DeepSeek | Azure |
| `MAI-Thinking-1` | Microsoft | Azure |

996 questions over 4 benchmarks (TriviaQA, GSM8K, MedQA, MMLU-Pro split into 6 subject
domains). **6,939 responses**, graded by `google/gemini-2.5-flash-lite` — a lab with no
model in the pool. 6,938 of 6,939 judged; 1 fell back to string matching.

**970 of 996 questions** have an answer from every model and form the paired analysis set.
The 26 dropped are cells no model-specific budget could recover (see §5).

Cost: **$7.51 OpenRouter**, ~$10.5 Azure grant credit (approximate — `MAI-Thinking-1` has
no published price meter).

---

## 2. Are the models matched? — YES

| model | accuracy | 95% CI |
|---|---:|---|
| `anthropic/claude-sonnet-5` | **84.2%** | [81.9, 86.5] |
| `grok-4.3` | 83.9% | [81.6, 86.2] |
| `openai/gpt-5.4` | 82.8% | [80.4, 85.2] |
| `Kimi-K2.5` | 82.5% | [80.1, 84.9] |
| `MAI-Thinking-1` | 80.8% | [78.3, 83.3] |
| `DeepSeek-V4-Flash` | 80.3% | [77.8, 82.8] |
| `Cohere-command-a-plus-05-2026` | 76.9% | [74.3, 79.6] |

Spread best-to-worst 7.3 points; **first to second, 0.3 points**. The premise the experiment
needs — models of comparable overall strength — holds. This is not the TriviaQA regime where
one model dominated and weighting merely rediscovered it.

---

## 3. Is there niche specialisation? — YES, and this is the real finding

Model × domain accuracy, after removing model and domain main effects, leaves an interaction
with **sd = 3.35 accuracy points**, max **10.56 points** (`MAI-Thinking-1` on MMLU-Pro
psychology).

Different models win different domains, and not randomly:

| domain | best model |
|---|---|
| GSM8K | `Kimi-K2.5` |
| MedQA | `grok-4.3` |
| MMLU-Pro business | `claude-sonnet-5` |
| MMLU-Pro chemistry | `MAI-Thinking-1` |
| MMLU-Pro health | `grok-4.3` |
| MMLU-Pro math | `Kimi-K2.5` |
| MMLU-Pro physics | `claude-sonnet-5` |
| MMLU-Pro psychology | `grok-4.3` |
| TriviaQA | `grok-4.3` |

**Permutation test: observed 3.35 vs null mean 2.46 (95th percentile 2.90), p < 0.001.**

Note the null mean is 2.46, not zero — with ~110 questions per domain, sampling noise alone
manufactures apparent specialisation of that size. The genuine excess is roughly **0.9 points
of interaction sd**. Real, but modest. Anyone eyeballing a model × domain table without this
null will badly overread it.

---

## 4. Can aggregation exploit it? — the corrected comparison

`analyze_kernel.py`. Answers embedded with `all-mpnet-base-v2` (final answer only, not the
reasoning chain), cosine similarity, aggregators taken directly from `kernel_agg`. All weights
5-fold cross-validated. n = 970, paired bootstrap, 10,000 resamples.

| method | accuracy | vs best single | 95% CI |
|---|---:|---:|---|
| majority (exact) | 81.86 | −2.37 | [−4.54, −0.21] * |
| medoid / cluster — *kernel alone* | 83.51 | −0.72 | [−2.78, +1.24] |
| OW exact — *weights alone* | 83.51 | −0.72 | [−2.06, +0.62] |
| **KWA (global β)** | **84.95** | **+0.72** | [−1.13, +2.58] |
| KWA (per-domain β) | 84.74 | +0.52 | [−1.34, +2.37] |
| best single model | 84.23 | — | |
| ceiling (any model correct) | 94.02 | +9.79 | |

`*` = CI excludes zero.

**KWA vs its own two ingredients** — the §2 refutation, retested:

| comparison | diff | 95% CI |
|---|---:|---|
| KWA − medoid/cluster (kernel alone) | +1.24 | [−0.52, +2.99] |
| KWA − OW exact (weights alone) | +1.24 | [−0.62, +3.09] |

### What this supports

- **The kernel earns its place.** 81.86 → 83.51 is +1.65 points from letting near-identical
  answers cluster instead of requiring string identity. This is the open-ended generalisation
  doing precisely what it was designed for, and it is the single largest effect in the table.
- **§2's refutation does not reproduce.** The handoff states *"KWA is never better than the
  better single ingredient."* Here it beats both by +1.24. That finding came from TriviaQA;
  on a matched, lab-diverse pool it does not hold.
- **KWA edges past the best single model for the first time** (+0.72).

### What this does not support

- **Nothing clears significance at n = 970.** Every CI above spans zero. The honest claim is:
  KWA leads on every comparison, by margins consistent with a real effect of ~1 point, and
  this experiment cannot establish it.
- **The per-question hypothesis specifically fails.** Per-domain β (84.74) is *worse* than
  global β (84.95). The prediction that a scalar weight cannot express specialisation is not
  supported — even though §3 shows the specialisation is genuinely there. Whatever KWA gains,
  it does not gain from per-domain weighting.

### Power

Margins are ~1.2 points against CIs of ~±1.8. Resolving KWA-vs-ingredients needs roughly
**2,000–2,500 questions**; establishing a win over the best single model needs closer to
**6,000**. The run is resumable, so this is a matter of credit, not rework.

### The methodological point that decides this

Fitting and scoring per-domain weights on the same questions gives:

| | in-sample | cross-validated | optimism |
|---|---:|---:|---:|
| global weights | 84.33 | 83.61 | 0.72 |
| per-domain weights | **86.19** | **84.33** | **1.86** |

In-sample, per-domain weights beat the best single model by +1.96 and the headline would read
*"per-question weighting clears the best single model — the prediction is confirmed."* That
result is entirely an artifact of fitting 63 parameters (7 models × 9 domains) on 970
questions and scoring them on the same 970. Cross-validation removes it. **Report the CV
numbers; the in-sample ones are shown only to size the illusion.**

---

## 5. Limitations, stated plainly

- **26 of 996 questions (2.6%)** are excluded because at least one model never produced an
  answer. `Cohere-command-a-plus` loops inside `reasoning_content` on some TriviaQA items and
  emits nothing at any budget to 24,000 tokens; `MAI-Thinking-1` returns `finish=stop` with
  every field empty on some GSM8K items despite billing output tokens (probably a content
  filter). These are recorded as **missing, not wrong** — scoring them zero would fabricate a
  capability gap.
- **~110 questions per domain.** The interaction test is powered for effects around 3 points
  of sd and no finer. A smaller true specialisation would not have been detected.
- **Domain labels come from the benchmark, not inferred.** This measures whether
  specialisation is exploitable *in principle*, not what a deployed system achieves — a real
  system must classify the question first, and that error propagates.
- **Two providers.** Sampling regime was held identical (temperature 0, same system prompt,
  same budgets); the vendor is not the variable under study, but it is not a controlled one
  either.
- **`Kimi-K2.5`, not K2.6** — one generation behind, because that is what got deployed.

---

## 6. Three bugs found during this run, all of the same shape

Recorded because each would have produced a confident, publishable-looking, wrong number.

1. **Generation wrote empty responses as rows.** `resp=""` grades as incorrect and is
   indistinguishable in the file from a model that answered wrongly. This is exactly how run
   v1 manufactured its fake 67-point spread. Now an empty response is left unwritten so the
   cell is missing and gets retried.
2. **The judge graded everything INCORRECT — 0 of 6,606** — while string matching on the same
   rows said 65.8%. Cause: the judge was switched to a thinking model while `max_tokens` stayed
   at 5, so it returned `content: null`, and `("" ).startswith("CORRECT")` → `False` converted
   every silent failure into a confident wrong grade. Fixed: budget raised, and an unparseable
   verdict now returns `None` (ungraded) rather than `False`.
3. **The permutation test could never reject.** It permuted rows within a domain block and then
   took the column mean — but a mean over rows is invariant to row order, so every replicate
   reproduced the observed matrix and `p` was pinned at 1.0000 by construction. The corrected
   test shuffles domain labels across questions and returns p < 0.001.

The common thread: **an absent or unmeasured value being silently coerced into a confident
one.** Worth treating as the default failure mode of this pipeline rather than three accidents.

---

## 7. Where this leaves the research question

The most defensible deliverables remain the label-free skill estimator and the exact reduction
to OW. To those, this run adds:

- **A clean positive:** niche specialisation across frontier labs is real and measurable, with
  a correctly-calibrated null showing most of a raw interaction table is noise.
- **A qualified positive:** the similarity kernel is worth +1.65 points over exact matching,
  and KWA leads every comparison including the best single model — but underpowered at n=970.
- **A clean negative:** per-domain weights do not beat global weights. The specific hypothesis
  this experiment was built to test is unsupported, in the regime built to favour it.

The ceiling remains the interesting number: **94.02% of questions are answered correctly by at
least one model, against 84.23% for the best single model.** KWA claims 0.72 of those 9.79
points. The rest sit unclaimed.

Two next steps, in order of value:

1. **Extend to ~2,500 questions.** Every effect here is ~1 point against CIs of ~±1.8. This is
   the cheapest way to convert four suggestive results into established ones, and the run
   resumes from what exists.
2. **Then logprob aggregation.** A vote discards everything except each model's argmax, which
   is why 9 points sit above a method that already uses similarity and calibrated weights.

---

## 8. Reproducing

```
data/v2.jsonl          6,939 generations   — VALID
data/v2_judged.jsonl   6,938 judgements    — VALID (gemini-2.5-flash-lite)
results/domain_analysis.json
results_run{,2,3}.log  the three passes; run 1 and 2 contain the broken judge

data/v2_judged_BROKEN_gemini37_maxtok5.jsonl   — INVALID, kept as evidence, see its README
data/v2_aborted_openrouter_pool.jsonl          — INVALID, pre-Azure pool, see its README
```

```bash
./venv/bin/python run_all.py 200 6.50
```


---

## 9.  The catch-22 (added after the label-free re-run)

`em_estimate_beta` scored Spearman **+0.986** against true accuracy on the TriviaQA pool. On
this pool it scores **+0.473, p = 0.28**, with four of seven models pinned at the optimiser's
lower bound — including `grok-4.3`, genuinely the second-best model, ranked last.

The cause is structural, not a bug. EM identifies beta from disagreement patterns; it needs
models to differ enough for the agreement structure to reveal who is sharp. Seven models
spanning 7.3 points and clustering near 82% do not provide that, so the optimiser degenerates.

```
  matched pool  ->  specialisation question is live, but label-free beta has no signal
  spread pool   ->  label-free beta works, but weights merely track capability (settled in §2)
```

**The estimator works where it is not needed and fails where it is.** This is a sharper
obstacle than the aggregation result itself, because label-free skill estimation was the most
practical standalone deliverable. It should be re-tested on a deliberately spread pool to
confirm the mechanism, and any claim for it must state the capability-spread precondition.
