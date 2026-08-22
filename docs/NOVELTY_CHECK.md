# Novelty check — 22 August 2026

Action item from the 21 Aug meeting: *"analyse this, make sure you properly
separate this out from existing literature... just check if this is not similar
to something that's already been done."*

25+ arXiv sources surveyed. **Verdict: the combination is novel, but three of the
four individual signals have prior art that must be cited.** The framing needs to
change from "we found these signals" to "we are the first to use output-structure
signals for multi-model *set* aggregation."

Confidence marks: **[read]** = fetched and read the paper/abstract directly;
**[search]** = seen only in search-result summaries, needs a direct read before
citation.

---

## 1. What we claim

| # | component | status |
|---|---|---|
| 1 | rank of a claim **within the generating model's own output list** as a per-claim reliability signal | **no prior art found** |
| 2 | **omission strength** — a silent model's evidence weighted by how much it said | **no prior art found** |
| 3 | per-question **keep-count** chosen by plug-in expected F1 over aggregated claims | **no direct prior art found** |
| 4 | verbosity (list length) as a per-claim signal | **PRIOR ART EXISTS** — §3.1 |
| 5 | the four combined, for multi-model set-valued QA | **no prior art found** |

---

## 2. Direct lineage — cite prominently

**Atomic Self-Consistency**, arXiv:2405.13131, EMNLP 2024 **[read]**. The direct
parent. Draws m stochastic samples from ONE model, clusters atomic units, keeps
clusters above a validation-tuned Θ, composes with an LLM. No reliability
weighting, no notion of which model said what, no rank, no omission, no
per-question budget. Evaluated on ASQA, QAMPARI, QUEST, ELI5.

**Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order
Information**, arXiv:2510.01499 **[read]**. The OW / ISP paper we extend.
**Confirmed to target binary/categorical predictions, not set-valued or
long-form generation.** This is favourable: it is the precise statement of the
gap, and it corroborates our empirical finding that OW's `log(p/(1−p))` weight
inverts when per-claim precision < 0.5 (ours run 5.8–28%).

**Universal Self-Consistency**, arXiv:2311.17311 **[search]** — LLM picks the
most consistent *whole response*; ASC's own baseline.

**More Agents Is All You Need**, arXiv:2402.05120 **[read]** — sampling and
majority voting scales with agent count. Response-level, not claim-level.

**LLM-Blender**, arXiv:2306.02561 **[read]** — pairwise ranking + generative
fusion over whole responses. Different granularity.

---

## 3. Overlaps we must acknowledge

### 3.1 Verbosity is NOT a novel signal

- **The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs**,
  arXiv:2508.08285 **[search]** — reports that *response length alone* is a
  powerful hallucination detector, often matching more sophisticated methods.
- **Verbosity ≠ Veracity**, arXiv:2411.07858 **[search]** — verbosity
  compensation behaviour; models are more uncertain when they pad.

**Consequence.** We cannot present "long lists are less accurate" as a
discovery. Our defensible increment is narrower and should be stated that way:
using per-question list length as a **per-claim weight inside multi-model
aggregation**, and — the part with no prior art — using it to **weight a model's
silence**.

### 3.2 Dawid–Skene is the classical form of our `pattern` and `nb-full` arms

- **Fast Dawid-Skene**, arXiv:1803.02781 **[search]**
- **Crowd-Certain: Label Aggregation in Crowdsourced and Ensemble Learning**,
  arXiv:2310.16293 **[search]**
- **Improving Deep Ensembles by Estimating Confusion Matrices** (soft
  Dawid-Skene for deep ensembles), arXiv:2503.07119 **[search]**

**This is the single biggest reviewer risk.** Our `nb-full` (per-model TPR/FPR
with omission as evidence) and `pattern` (`P(correct | exact subset asserting)`)
are, in substance, a Dawid–Skene annotator model applied to LLM agents. We must
cite this family and present those arms explicitly as **baselines/ceilings drawn
from the crowdsourcing literature**, not as our contribution. Doing so actually
*strengthens* the paper: the headline becomes "even the classical
annotator-reliability ceiling is +0.25 ns, and output-structure signals beat it."

### 3.3 Paraphrase consistency predicting correctness

- **Is This Your Final Answer? Cross-Contextual Consistency as a Measure of LLM
  Credibility**, arXiv:2608.10315 **[read]** — single-model consistency under
  perturbed prompts; answers with smaller shifts are more likely correct.
- **Prompt-Reverse Inconsistency**, arXiv:2504.01282 **[search]**; self-consistency
  under prompt transformations, arXiv:2305.14279 **[search]**.

**Consequence.** Our TriviaQA terse-vs-natural self-consistency signal (86.5% vs
29.0% correct) is essentially C3. Cite it. Our increment is using it for
**multi-model answer selection**, where C3 is single-model credibility scoring.

### 3.4 Cross-model claim filtering already exists

- **Zero-knowledge LLM hallucination detection through fine-grained cross-model
  consistency** (Finch-Zk), arXiv:2508.14314 **[read]** — segments responses into
  semantic blocks, evaluates each against samples from multiple models, weighted
  aggregation with a threshold. Confirmed: **no position/rank signal, no
  omission-as-evidence, no per-question claim budget, and evaluated on FELM and
  GPQA-diamond, not list QA.**
- **MAD-Fact**, arXiv:2510.22967 **[search]**; **FaStfact**, arXiv:2510.12839
  **[search]**; **VeriFact**, arXiv:2505.09701 **[search]** — multi-agent /
  atomic-claim long-form factuality *evaluation*, not aggregation into a better
  answer.
- **Atomic Calibration of LLMs in Long-Form Generations**, arXiv:2410.13246
  **[read]** — confirmed single-model atomic-level calibration. No multi-model
  aggregation, no rank, no verbosity, no omission, no budget.

So multi-model claim-level filtering by agreement is prior art — that is our
`count`/`weighted` baselines, correctly. What has no prior art is the signals we
add on top.

### 3.5 Position effects are about INPUT position, not generated position

- **Serial Position Effects of Large Language Models**, arXiv:2406.15981 —
  **[read, inconclusive]** the abstract does not settle whether it studies input
  or output position. **Someone must read the full paper before we claim
  novelty here.** This is the one unresolved risk to claim #1.
- **The Position Curse: LLMs Struggle to Locate the Last Few Items in a List**,
  arXiv:2605.07127 **[search]** — locating items in a *given* list (input side).
- **Lost in the Middle** follow-up, arXiv:2510.10276 **[search]** — context
  position.
- **LLM-based Listwise Reranking under Positional Bias**, arXiv:2604.03642
  **[search]**; **EviRank**, arXiv:2606.04727 **[search]** — positional bias when
  the model *reranks supplied candidates*, and position-aware confidence for
  ranking. Both concern ranking items given to the model, not the reliability of
  items the model itself enumerates.

The distinction to state explicitly in the paper: **prior work studies how
position in the model's INPUT affects behaviour; we use position in the model's
OWN OUTPUT as a correctness signal.**

### 3.6 Adaptive set size exists, but for coverage not F1

- **Adaptive Coverage Policies in Conformal Prediction**, arXiv:2510.04318
  **[search]** — tailors miscoverage per test sample and can optimise for a
  target expected set size. Conformal coverage guarantees on classification, not
  an F1-optimal keep count over aggregated multi-model claims.
- Abstention / selective prediction (arXiv:2607.08456, arXiv:2502.09054
  **[search]**) — whether to answer, not how many items to return.

Closest genuine competitor to claim #3, and worth one paragraph of positioning.

### 3.7 Rank fusion classics

RRF, Borda, CombSUM/CombMNZ **[search]** are standard IR fusion over retrieval
lists. We already implement all four as baselines (RRF is the only one that beats
counting: +0.84). Cite the IR lineage; it strengthens the baseline table.

### 3.8 Also checked, not close

**[search]** unless noted: Power and Limitations of Aggregation in Compound AI
Systems (arXiv:2602.21556, theoretical, **[read]**); Harnessing Collective
Intelligence for Biomedical QA (arXiv:2508.01480 — plain *union* for list
questions, **[read]**); Mixture of Complementary Agents (arXiv:2605.24048);
Rethinking Mixture-of-Agents (arXiv:2502.00674); Best-of-∞ ensembling
(arXiv:2509.21091); Judging Is Not Enumerating (arXiv:2608.01000 — omission as a
*failure mode* to detect, not as evidence); Where Facts Go Missing
(arXiv:2607.22448); Consensus is Not Verification (arXiv:2603.06612); QAMPARI
(arXiv:2205.12665).

---

## 4. What to do before the paper

1. **Read arXiv:2406.15981 in full.** It is the only source that could
   invalidate claim #1. Highest priority.
2. **Cite Dawid–Skene** and relabel `pattern`/`nb-full` as classical
   annotator-reliability baselines rather than our own schemes.
3. **Soften the verbosity claim** to its multi-model, per-claim, omission-weighted
   form, citing arXiv:2508.08285 and arXiv:2411.07858.
4. **Cite C3** (arXiv:2608.10315) for the TriviaQA self-consistency signal.
5. **Position against Finch-Zk** (arXiv:2508.14314) as the nearest multi-model
   claim-filtering system, on the three axes it lacks.
6. Replace every **[search]** with a direct read before it enters the
   bibliography. Search-result summaries are not a citable source.

---

## 5. Bottom line

The **rank signal**, **omission strength**, and **per-question F1 budget** each
appear unclaimed, and the combination applied to multi-model set-valued QA
appears unclaimed. The result is defensible.

But the paper should not claim four novel signals. It should claim: *reliability
weighting over agents — including its classical Dawid–Skene ceiling — is
provably capped on set-valued open-ended QA, and signals from the structure of
each model's own output break that cap.* That claim is both novel and exactly
what the 210-ensemble evidence supports.
