# Novelty check — 22 August 2026, revised 26 August

Action item from the 21 Aug meeting: *"analyse this, make sure you properly
separate this out from existing literature... just check if this is not similar
to something that's already been done."*

> **REVISION, 26 August.** A broader literature review surfaced closer prior art
> for three of the four signals this document previously recorded as unclaimed.
> **Claims 1, 2 and 3 in §1 are downgraded.** The individual-signal novelty is
> weaker than the 22 Aug verdict stated; the combination and the empirical
> contribution are unaffected. New material is in §3.9 to §3.13, and §5 is
> rewritten. Nothing here has been read, and most of it arrived without an
> identifier — see the provenance warning below.

25+ arXiv sources surveyed on 22 Aug. **Verdict as revised: the combination and
the information-decomposition framing are defensible, but rank, omission and the
F1 keep-count each have closer prior art than first recorded and must be cited
rather than claimed.**

Confidence marks: **[read]** = fetched and read the paper/abstract directly;
**[search]** = seen only in search-result summaries, needs a direct read before
citation; **[reported]** = named by a third-party language model summarising the
literature, never searched for or opened by us.

> **[reported] is the weakest mark in this document and must be treated as a
> lead, not a source.** Items carrying it may have inaccurate titles, wrong
> years, or may not exist. Most arrived with no arXiv identifier or DOI, so the
> first task for each is to locate it at all. Do not cite, paraphrase, or
> describe any [reported] item in the paper until it has been found and read.

---

## 1. What we claim

| # | component | status (22 Aug) | status (26 Aug, revised) |
|---|---|---|---|
| 1 | rank of a claim **within the generating model's own output list** as a per-claim reliability signal | no prior art found | **DOWNGRADED.** Generated-position work exists, §3.9. Surviving increment: position as *cross-model aggregation evidence*, not as single-model calibration |
| 2 | **omission strength** — a silent model's evidence weighted by how much it said | no prior art found | **DOWNGRADED.** Informative-stopping models exist in the partial-ranking literature, §3.10. Surviving increment: the LLM-aggregation application |
| 3 | per-question **keep-count** chosen by plug-in expected F1 | no direct prior art found | **DOWNGRADED.** F1-optimal set decisions are established, §3.11. Present as the decision layer, not an algorithmic contribution |
| 4 | verbosity (list length) as a per-claim signal | PRIOR ART EXISTS — §3.1 | unchanged |
| 5 | the four combined, for multi-model claim aggregation | no prior art found | **holds.** No equivalent construction found |
| 6 | the information decomposition $\|S\|$ → $S$ → $(S,B)$, and the finding that the middle channel saturates | not assessed | **the strongest remaining claim**, §5 |

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

- **Serial Position Effects of Large Language Models**, arXiv:2406.15981 (Guo &
  Vosoughi, Dartmouth) — **[RESOLVED, full text read 22 Aug]. Input-side.**
  Figure 1 plots "the distribution of label selections across 42 labels, with the
  x-axis representing label positions": it measures which of the *supplied*
  labels a model picks, and finds primacy/recency bias in that choice. The paper
  itself frames prior work as "choice re-ranking methodologies... restricted to
  single-label selections" and positions its own contribution as a fuller view of
  "model focus across complete **inputs**."

  So it is selection bias over a sequence given TO the model. It says nothing
  about the correctness of items a model enumerates at position k of its own
  free-form output. ~~**Claim #1 stands.**~~ **Superseded 26 Aug: this paper is
  still a valid contrast, but it was never the threat. See §3.9 for work on
  position within the model's own generated output, which is.** Cite it as the
  input-versus-output contrast only.
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


### 3.9 Generated-list position — claim #1 is NOT unclaimed

This is the most consequential downgrade. §3.5 established that *input*-side
position work does not threaten us, and that remains true. It is the wrong place
to have stopped looking, because there is work on position within a model's own
**generated** output.

- **"Single Answer is Not Enough: On Generating Ranked Lists with Medical
  Reasoning Models"**, reported as 2025 **[reported]**. Studies model-generated
  ranked lists directly. Evaluates the position of the correct generated item
  using MRR and average correct-answer position, and reportedly describes
  correct-answer position as reflecting confidence calibration. If accurate, this
  is the same underlying observation as our rank feature.
- **Long-form factuality decay** **[reported]**. Factuality reportedly declines
  in later sentences and unsupported claims increase later in generation. Note
  this may be the same phenomenon as, or a duplicate of, arXiv:2505.15291 already
  logged in the morning handoff as an unread lead.

**Consequence.** We cannot write "we are the first to observe that earlier items
in a model's generated list are more trustworthy." Our defensible increment is
narrower: *the positions at which several heterogeneous models independently
generated the same claim, used as evidence by an aggregator*. Prior work asks
whether one model ranks its own correct answer highly. We ask what the ranks
across asserting models tell a third party deciding whether to keep the claim.

Note also that this reframes the SLAQ tension in §7 of the evening handoff. If
generated-position decay is already documented, our flat result on short lists is
the more surprising finding, not the gradient on long ones.

### 3.10 Omission strength — the statistical idea has a literature

- **"Statistical Models of Top-k Partial Orders"**, reported as 2024
  **[reported]**. Reportedly argues that conventional partial-ranking models
  overlook the information carried by list length, and develops models that
  jointly represent the ranking and the decision to stop.

If that description is accurate it is close to our intuition, in a different
application: the meaning of an item's absence depends on how far the ranker
continued before stopping. That is omission strength, stated for partial orders
rather than for LLM claims.

**Consequence, and it is not all bad.** We stop saying "no prior art" and start
saying "a novel adaptation of informative truncation to claim-level evidence in
LLM aggregation." That is a smaller novelty claim and a **better framing**: it
turns a hand-engineered feature into an instance of a known statistical idea.
"We treat free-form enumeration as a partial ranking with informative stopping"
is a stronger sentence than "we invented this feature," and it gives Proposition
2 in the paper's theory section a natural home in existing work.

### 3.11 The F1 keep-count — decision theory, not a new algorithm

- **Lipton, Elkan & Narayanaswamy**, on F1-optimal decision rules
  **[reported]**. Reportedly studies F1-optimal thresholding for binary and
  multilabel classification, shows that whether an item should be predicted
  positive depends on the probability scores of *other items in the same set*,
  and develops plug-in rules for F1. Likely findable under a title concerning
  optimal thresholding of classifiers to maximise F1; year and venue unconfirmed.

**Consequence.** The batch dependence of F1-optimal decisions is established, so
"the right number of positives depends on the other scores in the set" is not
ours. What remains is the application: a *dynamically constructed* candidate pool
from heterogeneous open-ended responses, with a noisy per-question estimate of
answer-set cardinality. Present it as the decision layer of the framework. The
paper's Proposition 3 should cite this rather than derive the plug-in rule as
though it were new; the closed-form stopping characterisation
($p_{k+1} > \tfrac{1}{2}\hat{F_1}(k)$) is worth keeping as exposition either way.

### 3.12 Predicting when aggregation helps — the solo ratio has neighbours

§3.8 already logs the co-failure ceiling (arXiv:2606.27288, **[read]**). Add:

- **Disagreement-based analysis of when ensembles help**, reported as NeurIPS
  2023 **[reported]**. Reportedly relates model disagreement to ensemble error.

**Consequence.** Do not claim "we are the first to predict whether aggregation
helps." Claim instead: *for claim-level aggregation, singleton reliability is a
simple diagnostic for whether model identity retains information beyond agreement
count.* That is narrower and it is what our three benchmarks actually support.
This also strengthens the case for a fourth benchmark run prospectively: compute
the ratio, predict, then test.

### 3.13 Multi-label crowdsourcing already covers set-valued labels

- **Multi-label extensions of annotator-reliability models** **[reported]**.
  Reportedly the crowdsourcing literature includes explicit multi-label variants
  where an item may carry several simultaneously correct labels.

**Consequence.** "Ours is set-valued rather than categorical" does not by itself
escape the Dawid–Skene family. This is a further argument for the framing change
already made in the paper: the distinction that does work is *votes versus
observations*, not *categorical versus set-valued*. §3.2 remains the biggest
reviewer risk and this widens it slightly.

---

## 4. What to do before the paper

0. **Locate and read every [reported] item in §3.9 to §3.13, before anything
   else.** They have no identifiers, they may be misdescribed, and two of them
   directly contradict claims this document previously made. Until they are
   found, the §1 table's revised column is itself unverified. `paper-lookup`
   needs no API key.
1. ~~Read arXiv:2406.15981 in full.~~ **Done 22 Aug — it is input-side, and it
   remains a valid contrast for input versus output position.** But see §3.9:
   input-side work was never the threat to claim #1, and stopping there was the
   error.
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

**Revised 26 August.** The 22 Aug verdict, that the rank signal, omission
strength and the F1 keep-count each appear unclaimed, does not survive. Each has
closer prior art, recorded in §3.9 to §3.11.

What survives, in descending order of strength:

1. **The information decomposition and the saturation finding.** Aggregation
   methods have grown more sophisticated about one observation, which agents
   asserted a claim. Separating that from what each agent's generation reveals,
   and showing the first channel can be nearly exhausted while the second is not,
   is the contribution no neighbouring work was reported to make.
2. **The empirical contribution.** The 210-ensemble sweep, the identity-channel
   measurement against counting, the ablations, the source-shift protocol, and a
   benchmark where our own method fails for a measured reason. This is more
   substantial than any individual feature.
3. **The combination.** No reported work pools claims across heterogeneous models
   and scores each using supporter identity, generated position, informative
   silence, and a variable-cardinality set decision together.
4. **The individual signals.** Moderate at best. Rank and omission are
   adaptations of ideas with precedent; the keep-count is an application of
   established decision theory.

The paper should therefore be framed around 1 and 2, present 3 as the mechanism
demonstrating them, and cite prior art for each element of 4 rather than claiming
it. The current draft's framing already does this. What must change is the
related-work section, which still owes citations for §3.9 to §3.11, and any
sentence anywhere that says a signal has no prior art.

**One sentence to delete wherever it appears:** that we found no prior art for
rank, omission, or the keep-count. It was true of the 22 Aug survey and is not
true now.
