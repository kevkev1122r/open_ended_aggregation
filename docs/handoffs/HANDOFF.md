> ## ⚠ SUPERSEDED — read `HANDOFF_2026-08-16.md` FIRST
>
> That file has the current state: four benchmarks run, the method refuted at three
> granularities, the measurement findings that survived, the full trap list, and the
> Azure/OpenRouter environment. §1 (the method) and §2 (the TriviaQA findings) below
> are still accurate. **Everything about experiment state below is stale.**

# HANDOFF — open-ended LLM answer aggregation

Written to let a fresh session or a teammate continue without the original conversation.
Everything below is either measured or explicitly flagged as untested.

---

## 1. What this project is

Extending Ai, Pan, Simchi-Levi, Tambe & Xu, *Beyond Majority Voting: LLM Aggregation by
Leveraging Higher-Order Information* (ICML 2026, arXiv:2510.01499) from multiple choice to
open-ended answers. That extension is the paper's own closing question.

**Their method (OW).** Weighted vote, weight = `log((K−1)·x/(1−x))` where `x` is a model's
accuracy and `K` the number of options. Provably Bayes-optimal for multiple choice. It needs
two facts, both manufactured by randomly shuffling the answer labels:

```
  (1)  P(model j says s  | truth = s)  =  x_j
  (2)  P(model j says s' | truth = s)  =  (1 − x_j)/(K − 1)   for EVERY s' ≠ s
```

Fact (2) — all wrong answers equally likely — is what factors a constant out of the likelihood
product, leaving `weight × indicator`, i.e. a vote. Open-ended generation destroys it: there is
no label set to shuffle, and "Sydney" is a far likelier error than "photosynthesis" when the
truth is "Canberra".

**Our method (KWA).** Replace the exact-match indicator with a similarity kernel:

```
  pick s maximising   Σ_j  β_j · sim(answer_j, s)
```

`β_j` is how sharply model j concentrates near the truth, estimated label-free by EM.
Set `sim` to exact match and it becomes OW exactly — **verified numerically at 1.0000 for
K = 2, 3, 4, 6, 10** (`kernel_agg.test_reduction`). It is a strict generalisation, not a rival.

---

## 2. What is established

### Validated

- **The reduction to OW is exact.** 1.0000 agreement at every K.
- **Errors cluster near the truth on real data.** 9/9 cells (3 corpora × 3 encoders),
  Cohen's d 1.43–3.14, scrambled-label placebo passes everywhere (|β| ≤ 0.277).
- **Label-free skill estimation works where inter-model agreement exists.** Spearman **+0.986**
  against true accuracy on real models; correctly identified the best model with zero labels.
- **Similarity-aware selection beats exact-match voting on open-ended answers.**
  +3.2 points on short answers, +6.5 on verbose ones, p < 1e-10, n = 2,757.

### Refuted

- **`log P = α + β·sim` is the wrong functional form.** Logistic regression with
  question-clustered SEs: curvature required in **9/9 cells**, p < 1e-16. The bend goes
  *opposite* directions — positive (accelerating) for model-made errors, negative (saturating)
  for hand-written distractors. So no single fixed curve serves both.
- **β is not a constant.** It ranges 2.6 to 20.6 across corpora, encoders and candidate-pool
  spread. Must be estimated per deployment, never assumed.
- **The combination does not beat its own parts.** Ablation (`majority vote → medoid → KWA`):
  the kernel contributes everything on verbose answers and nothing on short ones; the weights
  do the reverse. **KWA is never better than the better single ingredient.** On verbose answers
  a plain unweighted medoid (3 lines, no estimation) matches or beats it.

### The structural ceiling — applies to everyone, not just us

On TriviaQA with 6 models, errors correlated **215–621× more** than conditional independence
predicts. Consequence:

```
  all 6 right   42.6%   → aggregation irrelevant
  all 6 wrong    7.0%   → aggregation impossible
  mixed         50.3%   → the only battleground
  on the battleground, the best single model is already right 86.7% of the time
  TOTAL headroom for ANY aggregator: 6.67 points (short) / 4.32 (verbose)
```

And on winnable questions the truth is held by a mean of **1.88 of 6 models** — a minority. To
win them you must overrule a 4–2 majority, which no label-free weight estimate is confident
enough to do. This also explains the paper's own MMLU result, where their best single model
(91.02%) beat every method they proposed (90.37%).

---

## 3. The open question, and why the last experiment was attempted

**If all models are frontier and roughly equal overall, but each is better in a niche — what
happens?**

Answer from theory: both the paper's method and ours **collapse into majority voting**, because
each represents a model as a single number. Equal overall accuracy ⇒ equal weights ⇒ plain
vote. Corollary 3.3 says so directly.

But that setting is the *best case for aggregation* and the *worst case for scalar weights*:
errors decorrelate (specialists fail on different things, which is exactly the condition the
theory needs), and the minority holding the truth becomes **predictable** rather than random —
on a chemistry question, trust the chemistry model.

The fix is per-question weights `β_j(question)` instead of a global `β_j`. Prediction to test:

```
  global weights      ≈ majority voting     (a scalar cannot express specialisation)
  per-domain weights  > both, and may clear the best single model
                        — which never once happened on TriviaQA
```

**Accepted limitation:** domain labels come from the benchmark, not inferred. So this measures
whether specialisation is *exploitable in principle*, not what a deployed system achieves. A
real system would classify or cluster questions first and that error would propagate.

---

## 3b. THE EXPERIMENT IS RUN — see `RESULTS_v2.md` (2026-08-14)

Headline: **specialisation is real and the kernel earns its place — but the LABEL-FREE weight
estimator fails on a matched pool, and the deployable method loses to the best single model.**

⚠ **The label-free EM estimator does not survive this pool.** Spearman vs true accuracy is
**+0.473 (p = 0.28)** here against **+0.986** on TriviaQA, with 4 of 7 models pinned at the
optimiser's floor — including grok-4.3, actually the #2 model, ranked last. Scoring KWA with
real EM betas instead of judge-derived accuracy:

```
  KWA supervised beta       84.85   +0.62 vs best single   <- NOT the method
  KWA LABEL-FREE beta (EM)  82.58   -1.65 vs best single   <- the method
  medoid, no weights        82.99   -1.24                  <- beats the label-free weights
  best single model         84.23
  label-free - supervised   -2.27  [-4.02, -0.52]  significant
```

**The catch-22:** EM identifies beta from disagreement patterns, so it needs capability
spread. A matched pool — required for the specialisation question — starves it. A spread pool
feeds it, but then weights just track capability, which §2 already settled. The estimator
works where it is not needed and fails where it is. Any claim for the label-free estimator
must state the capability-spread precondition.

- Models matched: 76.9–84.2%, 0.3 points between #1 and #2. Premise holds.
- Specialisation real: interaction sd 3.35 vs null 2.46 (95th pct 2.90), **p < 0.001**.
  The null is *not* zero — ~110 questions/domain manufactures 2.46 points of apparent
  specialisation from noise alone. Genuine excess ≈ 0.9 points of sd.
- Aggregation (`analyze_kernel.py`, mpnet embeddings, 5-fold CV weights, n=970):

  ```
    majority (exact)                  81.86   -2.37 vs best single  (CI excludes 0)
    medoid/cluster  (kernel only)     83.51   -0.72
    OW exact        (weights only)    83.51   -0.72
    KWA global beta                   84.95   +0.72   [-1.13, +2.58]
    KWA per-domain beta               84.74   +0.52   [-1.34, +2.37]
    best single model                 84.23
    ceiling                           94.02   +9.79
    KWA - kernel-only  +1.24 [-0.52, +2.99]   KWA - weights-only  +1.24 [-0.62, +3.09]
  ```

- **§2's "KWA never beats the better ingredient" does NOT reproduce** — it beats both by
  +1.24 here. That refutation was TriviaQA-specific.
- Prediction scorecard: per-domain β **worse** than global β, so "a scalar cannot express
  specialisation" is unsupported even though the specialisation is real.
- Underpowered: ~1.2-point effects against ~±1.8 CIs. **~2,500 questions** would settle
  KWA-vs-ingredients; ~6,000 for a clean win over the best single model. Run is resumable.

⚠ **`analyze_domains.aggregation()` DOES NOT RUN THE METHOD.** It compares answers by exact
string identity of the last 120 chars and never imports `kernel_agg`. 94.8% of questions have
all 7 answers distinct, so its "majority vote" is a random pick and its "weighted vote" is
just picking the highest-weighted model. Its numbers measure model *selection*, not
aggregation. Use `analyze_kernel.py`. The first draft of RESULTS_v2.md drew the wrong
conclusion from it.

---

## 3c. FACTS GROUNDING — the first viable testbed (2026-08-15)

`run_facts.py`, `analyze_facts.py`. Pool = the 5 DEPLOYED Azure models
(xAI, Moonshot, Cohere, Microsoft, DeepSeek). Judge = `gpt-5.4-mini` — OpenAI has no
model in the pool. **Zero OpenRouter**: that balance is spent.

Why this benchmark: the v2 suite could not test KWA. Median answer was 4 words with a
canonical form, so exact matching is already the *correct* similarity function and the
kernel can only add noise; 62% of questions were incapable of showing an effect. HaluEval
summarisation was tried and scored 4%, because a 28-word reference is one arbitrary
selection of facts — reference matching measures agreement with that choice, not truth.
FACTS grades "is every sentence grounded in the provided document", which two
differently-worded correct answers both pass, and ships DeepMind's validated rubrics.

```
                       v2 suite      FACTS
  answer length         6 words     214 words
  contested               32.7%        61.1%
  headroom             9.8 pts      19.4 pts
```

### ⚠ GRADING GRANULARITY IS THE BINDING CONSTRAINT — the transferable finding

Same responses, same models, same benchmark. Only how the judge scores them changes:

```
  binary response-level   unanimous 80.0%   best single 97.1%   headroom  2.9 pts
  sentence-level          unanimous 30.3%   best single 75.8%   headroom 21.2 pts
```

Collapsing a 214-word answer to one accurate/inaccurate bit destroys the signal the
benchmark contains. **Every benchmark rejected during the search was judged binary.**
Before rejecting a benchmark as "too easy", check whether it is the grading, not the task,
that is saturated.

The same error repeats one level down. Scoring a response correct only if EVERY sentence
is grounded (`THRESH = 1.0`) gives:

```
  r(centrality, groundedness score) = +0.419  p<0.001   <- real signal
  r(centrality, binary correct)     = +0.038  p=0.62    <- destroyed by the threshold
```

So weights estimated from binary accuracy are noise, and KWA (kernel + those weights)
scores *below* the unweighted medoid. For long-form the outcome is continuous — score
aggregators on the mean groundedness of what they select, not on a thresholded bit.

### Pilot result, n=36, CONTINUOUS metric — all CIs span zero

```
  medoid / cluster (kernel alone)   93.43%   +0.91 vs best single   <- only method above it
  best single (grok-4.3)            92.52%
  majority vote                     91.67%   -0.85
  OW exact (weights only)           90.76%   -1.76
  KWA supervised beta               90.28%   -2.24
  KWA label-free beta (EM)          89.00%   -3.52
  oracle (best per item)            98.95%   +6.43
```

**§2's refutation reproduces a third time**: KWA loses to the better of its ingredients.
And EM went *negative* here — Spearman **−0.205** — ranking the best model (DeepSeek,
69.4%) at the floor and a near-worst model highest. Responses to one document sit at 0.890
mean pairwise similarity, so the kernel is discriminating inside a very narrow band.

Full 300-item run in progress; these intervals should narrow ~3x.

---

## 4. State of the earlier runs

Designed and built, not finished. Two runs attempted:

**Run v1 — INVALID, do not use `data/full_run.jsonl` for capability claims.**
`max_tokens` was sized for answer length (60–400). Reasoning models spend that budget on
internal reasoning tokens and return `content=""`. Six of ten models came back 38–88% empty,
producing a **fake 67-point capability spread** and invalidating the capability ranking, the
specialisation matrix and the aggregation comparison. The judged grades in
`data/full_run_judged.jsonl` are still valid *as grades*; the accuracy comparison is not.

**Run v2 — ABORTED on budget.** Fixed budgets (800–1500 tokens, uniform, reasoning left ON so
reasoning models keep their GSM8K advantage). Pre-flight passed for 9/10 models.
`thinkingmachines/inkling` dropped — empty on medqa and mmlupro even at 8× budget.
Stopped at 177 rows: cost per call was **$0.00403, 6× the v1 estimate**, because v1 was cheap
precisely *because* it was broken. OpenRouter budget exhausted ($6.67 of $9).

**Now unblocked:** $1000 Azure startup grant. The full run is ~$110 of that.

---

## 5. Next steps, in order

1. ~~**Adapt to Azure.**~~ **DONE** (2026-08-13) — but it was not the 15 lines predicted. See
   `azure_backend.py`; `run_v2.py` now switches on `BACKEND=azure|openrouter` (default azure).
   Three differences from OpenRouter, all of which reproduce v1's silent-failure shape:
   - `max_tokens` is rejected by gpt-5.x — it wants `max_completion_tokens`.
   - `temperature=0` is rejected by gpt-5.5 and gpt-5.6-*. Silently dropping it runs those
     models at temperature 1 next to peers at 0 — a confound on the measured quantity. The
     pool uses gpt-5.4 for this reason. Quirks are auto-detected from the 400 and cached.
   - **No usage endpoint.** OpenRouter's `GET /api/v1/key` was what the hard spend cap read.
     Foundry has no data-plane equivalent, so the cap is reconstructed locally from reported
     token usage × `AZ.PRICE`. `PRICE` is deliberately empty — a guessed price makes the guard
     lie, and this project already lost a run to an estimate wrong by 6×.

   Upside: Azure reports `finish_reason` and `completion_tokens_details.reasoning_tokens`,
   so **the v1 bug is now directly detectable** rather than inferred from an empty string.

2. **BLOCKED — provision the deployments.** The Foundry project has **zero** deployments.
   Only gpt-5.x answers off the bare project endpoint; every other model returns
   `400 "does not support deploymentless inference"`. Catalogue presence ≠ callable — this is
   the same trap as v1's blocked models, in new clothing. Creating deployments is a
   control-plane operation: the API key in `.env` cannot do it, and no `az` CLI is installed.
   Do it in the Foundry portal (Models → Deploy) or install `az` and authenticate.

3. **Fill in `AZ.PRICE`** from the Azure pricing page for the resource's region, one
   `(prompt, completion)` pair per 1M tokens per model. `preflight` refuses to pass without it.

4. **Run the pre-flight** (`run_v2.py preflight`). Now checks, in order: deployments present →
   PRICE present → per-model reachability → per-model × per-domain non-empty. This session
   lost time to three separate silent failures — blocked models, an invalid model ID, empty
   responses — and the pre-flight catches all three shapes for cents.

5. **Run the full experiment.** 9 models × 3,000 questions × 4 benchmarks. The ~$110 estimate
   was for the OpenRouter pool and does **not** transfer: re-derive it from `AZ.PRICE` and the
   token counts preflight reports before committing.
6. **Judge-grade it** (`judge.py`). Judge must stay from a lab with **no model in the pool**.
   **Google is held out of the pool on purpose** and is the judge. Gemini is not on Azure, so
   judging runs off-Azure on a separate key — `judge.py` still points at OpenRouter, where
   ~$2.30 remains. Price the judge pass against 27k responses before assuming that covers it.
7. **Analyse** (`analyze_domains.py`) — capability match → specialisation permutation test →
   global vs per-domain weights.

### The pool (settled 2026-08-14): seven peer-tier models, seven labs, two providers

| model | lab | provider | $/call |
|---|---|---|---|
| `anthropic/claude-sonnet-5` | Anthropic | OpenRouter | 0.00396 |
| `openai/gpt-5.4` | OpenAI | OpenRouter | 0.00198 |
| `grok-4.3` | xAI | Azure | 0.00381 |
| `Kimi-K2.5` | Moonshot | Azure | 0.00292 |
| `Cohere-command-a-plus-05-2026` | Cohere | Azure | 0.00239 |
| `DeepSeek-V4-Flash-2026-04-23` | DeepSeek | Azure | 0.00036 |
| `MAI-Thinking-1` | Microsoft | Azure | **unpriced** |

Anthropic and OpenAI have **zero Azure quota** and route through OpenRouter. Dispatch is by ID
shape: `"/"` means OpenRouter, otherwise an Azure deployment name. Mixing providers is sound so
long as the sampling regime matches (temperature 0, same system prompt, same token budget) —
the vendor is not the variable under study.

**Judge: `google/gemini-3.7-flash` via OpenRouter.** Was `deepseek-v3.2`, which the quota cuts
silently invalidated by forcing DeepSeek into the pool. $0.000057/judgement — grading the whole
experiment costs under $1.50. Google is held out deliberately and cannot be on Azure at all
(closed weights, competitor); 0 of 217 catalogue entries.

**The budget is bound by OpenRouter, not by the grant.** Azure's share is ~$8 against ~$1000 of
credit, because quota — not money — is what limits Azure. `run()` takes a separate `or_cap` for
the OpenRouter half; one combined cap is dominated by the plentiful side and would let the
scarce one run dry unnoticed.

Deployment names must match these strings exactly — the deployment name is what gets called,
not the model name. `Kimi-K2.5` and `MAI-Thinking-1` are named for what was actually deployed
(K2.6 and a dated MAI variant were intended; K2.5 is a generation behind).

One model per lab is load-bearing, not cosmetic: same-lab models share training data,
tokenizer and RLHF lineage, and the 215–621× error correlation in §2 is what a low-diversity
pool buys you.

**Anthropic, Mistral and Meta were cut for zero TPM quota, not for design reasons.** Quota is
per-model per-region, and the newest models are rationed hardest — `claude-sonnet-5`,
`mistral-medium-3-5` and `Llama-4-Maverick` all came back 0/0 on the grant subscription.

Older siblings (`Llama-3.3-70B`, `mistral-medium-2505`, `claude-haiku-4-5`) almost certainly
have quota and were **deliberately not substituted in**. The premise of this experiment is
models roughly *equal* overall but specialised by niche — the one regime where a scalar weight
provably fails and a per-question weight might win. A weaker model restores a capability gap,
the weights collapse to tracking overall skill, and the run re-derives Corollary 3.3 at full
price. Six peer-tier models beat nine mismatched ones.

Six is also the size of the completed TriviaQA experiment, so the error-correlation and
minority-holds-truth numbers in §2 stay directly comparable rather than needing a rerun.

Still worth checking if quota appears: `claude-sonnet-4-6` or `claude-opus-4-5` (true Anthropic
peers, unlike haiku), and `DeepSeek-V4-Pro` (Flash is the value tier among five flagships).

Only 4 of the original 9 had an Azure peer on the per-token surface (claude-haiku-4.5,
gpt-5.4-mini, o4-mini, kimi-k2). Qwen appears there only as `qwen3-32b`, not a frontier peer,
which would import a capability gap into an experiment whose premise is models of *equal*
overall strength.

**Two serving surfaces, and only one is visible to the API.** `/openai/v1/models` returns 399
entries and is the per-token serverless catalogue. The portal also lists a Hugging Face
catalogue that endpoint never returns — `zai-org--glm-5.2-fp8` and `qwen--qwen3.6-27b` are in
the portal and invisible to an API probe. They fail with `404 "The API deployment for this
resource does not exist"` rather than the `400 "deploymentless"` the serverless models give;
the two errors mean different things.

HF models deploy to **managed compute**: a GPU VM billed by the hour, running whether or not
you call it. `AZ.PRICE` is per-token and cannot bound them. `glm-5.2-fp8` is also quantised and
on a different serving stack from the API-served models it would sit beside — comparable only
with care. GLM is therefore an available swap, not a default.

Gemini is genuinely absent (Microsoft does not serve Google's models) and is the judge.
**MiniMax is open-weight and may be in the HF catalogue — unverified.** Search the portal
before concluding any model is unavailable; an API probe answers a narrower question.

**Longer term, the highest-value untested direction:** KWA's entire gain came from questions
where the vote was *tied*; it never overturned a strict plurality. That is a hard ceiling. The
fix is to stop treating models as casting hard votes and aggregate their full **logprob
distributions** instead. Needs local weights or an API exposing logprobs — untestable so far.

---

## 6. Traps to avoid (all hit in this session)

- **Empty responses.** Always check. A reasoning model with a small `max_tokens` returns `""`
  and silently scores zero, which looks exactly like incompetence.
- **Model IDs.** OpenRouter renders aliases with a `~` prefix; stripping it yields an invalid
  ID and 750 failed calls. Validate every ID with one call first.
- **Blocked models.** Some need account-level settings (data policy, 18+ attestation). Those
  are the account owner's to change.
- **Catalogue presence ≠ callable (Azure).** `/openai/v1/models` lists 140 chat models the
  region knows about. Eight of nine in our pool return 400 until a deployment exists, and
  `qwen3-32b` / `gpt-4.1` / `gpt-oss-20b` additionally reject the `GlobalStandard` SKU. Always
  check `AZ.deployments()` — an empty list means *nothing* runs — before reading a table of
  zeros as a capability result.
- **The portal's "Available in my project" means available to DEPLOY, not deployed.** It is
  not contradicting the API when the API returns errors; they answer different questions.
- **Reasoning tokens are billed but sit OUTSIDE `completion_tokens`.** Measured on grok-4.3:
  `total=207, prompt=21, completion=26, reasoning=160`. Costing a call as
  `prompt + completion` undercounts a reasoning model by ~8×, and undercounting is the
  direction that lets a spend cap silently permit an overrun. Derive billable output as
  `total_tokens − prompt_tokens`, which is correct whether or not reasoning is nested.
- **Deploymentless inference is not capacity you own.** gpt-5.4 answered with no deployment,
  then began failing `deployment_disabled` / "Insufficient quota available for instant
  inference" once other models were deployed. It draws on a shared instant pool. Never build a
  run on it.
- **Azure prices mix per-1K and per-1M units in the same API.** Grok, DeepSeek and Kimi publish
  per 1K; Cohere and OpenAI per 1M. Read `unitOfMeasure`, never just `retailPrice` — the error
  is 1000×.
- **One API probe does not enumerate the platform.** `/openai/v1/models` covers the per-token
  surface only and silently omits the entire Hugging Face catalogue. A model missing from it
  may still be one portal click away. This produced a wrong "GLM is absent from Azure" claim
  in the first pass of this port.
- **Provider-specific parameter names.** `max_tokens` → `max_completion_tokens` on gpt-5.x,
  and `temperature=0` is refused outright by gpt-5.5/5.6. A port that drops the offending
  parameter to make the call succeed silently changes the experiment's sampling regime.
- **Cost estimates from a broken run.** v1's cheapness was a *symptom*. Re-estimate after any
  fix that changes output length.
- **Option-dependent benchmark questions.** "Which of the following…" is meaningless without
  options. MedQA is 85% such — but they are mechanically rewritable ("*What* is the most
  likely diagnosis?"), recovering 164 → 670 items. MMLU-Pro needs a harsher filter: 47% dropped
  for phrasing/blanks/multi-part, then **another 32% because a distractor shares the correct
  answer's numeric value** ("5 mm" vs "5 cm"). 4,337 of 12,032 survive.
- **String grading.** Judge audit, n=7,500: 92.4% agreement, but string matching **understates
  MMLU-Pro by 6.4 points**. TriviaQA net bias was only +0.2 (errors cancelled), so the earlier
  TriviaQA result stands.

---

## 7. Files

```
  kernel_agg.py        the method: aggregators, label-free EM for β, test_reduction()
  benchmarks.py        4 benchmarks with filtering, option-hiding, per-domain grading
  run_v2.py            generation with PRE-FLIGHT + spend cap  ← use this, not generate.py
                       BACKEND=azure (default) | BACKEND=openrouter
  deploy_models.py     emits the `az` deployment commands for the pool (prints, never runs)
                       handles the catalogue-id vs model-name/version split
  azure_backend.py     Foundry transport: quirk auto-detection, reasoning-token and
                       finish_reason capture, local spend reconstruction, PRICE table
                       ./venv/bin/python azure_backend.py catalogue   # what the region lists
                       ./venv/bin/python azure_backend.py <model>...  # what actually answers
  judge.py             LLM judge + string-vs-judge audit
  analyze_domains.py   capability → specialisation → per-domain weights
  analyze_real.py      the TriviaQA-only analysis (completed experiment)
  generate.py          v1 generator, TriviaQA only — superseded by run_v2.py

  REPORT.docx          completed TriviaQA experiment write-up
  FINDINGS.md          synthetic + proxy work preceding it
  PLAN.docx            original research plan
  CODE_MAP.md          full file index

  data/gen_full.jsonl  36,000 TriviaQA responses (the completed experiment) — VALID
  data/full_run.jsonl  v1 multi-benchmark — INVALID for capability, see §4
  data/v2.jsonl        177 rows, aborted
```

**Credentials** are in `.env` (chmod 600, gitignored): `OPENROUTER_API_KEY` (~$2.30 left,
**should be rotated** — it appeared in plaintext in the originating conversation),
`AZURE_ENDPOINT`, `AZURE_API_KEY`.

---

## 8. Honest summary

The theory is sound and the reduction to OW is exact. The label-free skill estimator works and
is arguably the most practical deliverable. But **the central claim — that weighted similarity
beats both plain voting and plain similarity — is not supported by the data**, and the
structural ceiling from correlated errors caps *any* aggregator at a few points on the
benchmarks tested.

The niche-specialisation experiment is the live question and the one thing that could change
that verdict, because it is the one regime where a scalar weight provably cannot work and a
per-question weight might. It is designed, built, and unrun.
