# Iteration 2

Written 26 August 2026 against `main1.tex`. Covers the reframe, the theory
section, the prior-art revision, the power analysis, and the ASQA story.

**What this iteration did.** Reframed the paper from a set-valued aggregation
method to an argument about *what an aggregator observes*, added a theory section
with three propositions, reduced one overclaim, and downgraded three novelty
claims against prior art we had missed.

**Two things it cost.** The paper now claims less about the ceiling and less
about the individual signals than it did. Both reductions are correct.

---

## 1. The reframe: votes versus observations

### The problem with the old framing

The paper was titled around *set-valued answers*, and the mentor's note was that
this reads as narrower than the work is. That is right, and the generality was
already in the paper:

- the ceiling construction in §3.1 never mentions sets, only $S(c)$;
- three answer formats are already run, entity lists, prose, short answers;
- the reason for using set-valued benchmarks is **methodological**, not
  conceptual: a filter trades precision against recall, so only a metric scoring
  both can adjudicate one.

That last point belongs in Experiments as a measurement constraint. It was doing
duty as the paper's identity.

### The new spine

For a candidate claim $c$, write $S(c)$ for the set of agents that asserted it,
and $B(c)$ for what those agents' own generations reveal about it. The literature
is a hierarchy on that observation:

| level | reads | methods |
|---|---|---|
| count | $\|S(c)\|$ | majority voting |
| identity | $S(c)$ | reliability weighting, correlation-aware weighting, Dawid–Skene, optimal weights |
| behaviour | $(S(c), B(c))$ | ours |

The generality argument is now structural rather than a claim about answer types:
**an agent that generates does not cast a vote, it produces a structured
artifact, and reducing that artifact to an assertion indicator discards the
rest.**

### What changed in the file

| location | change |
|---|---|
| title | `Beyond Votes: Generation Behaviour as Evidence in Multi-Agent Aggregation` |
| abstract | rebuilt around the $\|S\| \to S \to (S,B)$ hierarchy |
| §1.1 | opens on the observation an aggregator uses; two-candidate example replaces the set-valued motivation |
| §1.3 | states the hierarchy explicitly; RQs reframed around which channel has headroom |
| §1.5 | contributions cut from six to four |
| §2 | `B(c)` defined next to `S(c)` |
| §4.1 | new "Why these benchmarks" paragraph, set-valued data as measurement constraint |
| §6.1, §7 | related work and conclusion aligned; old overclaim removed |

"Set-valued" is gone from title and abstract. It survives twice in the body, both
times describing someone else's work.

The title deliberately echoes *Beyond Majority Voting*, which is the direct
lineage. Proposition 2 below makes that lineage formal.

---

## 2. The theory section, §3.2

Three propositions. The first two are elementary and the section says so; their
job is to name the quantity the experiments must measure.

**Proposition 1, information dominance.** $R^*(S,B) \leq R^*(S)$, with equality
when $Y_c \perp B(c) \mid S(c)$. One-line proof: any $S$-only rule is an $(S,B)$
rule that ignores its second argument.

Paired immediately with the finite-sample caveat, because **the ASQA result is
the worked example**: adding features that carry no signal costs estimation
variance, and our scorer lands below the identity-only rule it builds on. Theory
and negative result now explain each other.

**Proposition 2, behaviour as likelihood evidence.** Under conditional
independence,

$$\mathrm{logit}\, P(Y_c=1 \mid X) = \mathrm{logit}\, \pi + \sum_i \log \frac{P(X_{ic} \mid Y_c=1)}{P(X_{ic} \mid Y_c=0)}$$

Collapse $X_{ic}$ to the assertion indicator and each agent contributes
$\log(\mathrm{TPR}_i/\mathrm{FPR}_i)$ when asserting and
$\log((1-\mathrm{TPR}_i)/(1-\mathrm{FPR}_i))$ when silent, which is an additive
score over agents, which is a weighted vote. Add symmetric errors and you recover
$\log(x_i/(1-x_i))$, the optimal weight of Beyond Majority Voting.

**Reliability weighting is therefore nested inside the framework, not opposed to
it.** This is the single most useful thing the section does for the generality
question.

The section states plainly that the independence assumption is false here, and
gives two reasons the decomposition still earns its place: it fixes the
functional form the scorer fits, and it explains why we fit it discriminatively.
The sweep supports that choice, since correlation-aware weighting selected
$\lambda=0$ in 5/5 folds and pairwise co-assertion gained $+0.05$.

**Proposition 3, the metric-matched decision.** $F_1 = 2H/(k+G)$ exactly. The
denominator is constant in $H$, so for known $G$ the expectation passes through
the ratio exactly and **the only approximation is substituting $\hat{G}_q$**.

Deriving this produced a closed form the paper did not have. The objective
improves from $k$ to $k+1$ exactly when

$$p_{(k+1)} > \frac{C(k)}{k + \hat{G}_q} = \tfrac{1}{2}\widehat{F_1}(k)$$

**Keep taking candidates while the next probability exceeds half the expected F1
so far.** Verified against both worked examples in Figure 3; it reproduces
$k^*=7$ in each. It also explains why $\hat{G}_q$ accuracy barely matters: the
estimate enters only through a threshold that moves slowly in it, which is why an
MAE of 4.62 against a gold mean of 10.2 costs so little and the true count gains
only $+0.55$.

---

## 3. The overclaim we removed

§3.1 previously said the pattern lookup bounds every reliability-weighting rule
by construction. Two problems:

1. **The lookup is a finite-sample, shrunk estimator**, not the population
   quantity $P(\mathrm{correct} \mid S(c))$. It is an empirical benchmark, not an
   attained bound.
2. **Set F1 is not decomposable over candidates.** A rule making joint decisions
   across all candidates in a question could in principle beat any rule that
   scores candidates independently and thresholds, which is the form every arm
   here takes.

§3.1 now has a paragraph headed "What this does and does not establish" naming
both, and the claim is narrowed to *the strongest claim-local, identity-only rule
we can estimate*. Related Work and Conclusion carried the old wording and were
updated to match.

That is still enough for the job it does, which is bounding a design space we
would otherwise search by trial and error.

---

## 4. Prior art: citation ledger

Provenance marks, weakest last:

- **[read]** — fetched and read directly.
- **[search]** — seen in search-result summaries only. Not citable yet.
- **[reported]** — named by a language model summarising the literature. Never
  searched for or opened by us. **Treat as a lead, not a source.** Titles, years
  and existence are all unconfirmed, and most arrived with no identifier.

### 4.1 Direct lineage

| work | id | mark | role |
|---|---|---|---|
| Atomic Self-Consistency | arXiv:2405.13131, EMNLP 2024 | [read] | the direct parent; list pipeline has no composition step |
| Beyond Majority Voting | arXiv:2510.01499 | [read] | the OW/ISP paper we extend; Proposition 2 nests it |
| LLM-Blender | arXiv:2306.02561 | [read] | response-level ranking and fusion |
| More Agents Is All You Need | arXiv:2402.05120 | [search] | voting scales with agent count |
| Universal Self-Consistency | arXiv:2311.17311 | [search] | response selection; ASC's own baseline |
| Self-Consistency (CoT) | arXiv:2203.11171 | [search] | majority voting over sampled paths |

### 4.2 Benchmarks and metrics

| work | id | mark |
|---|---|---|
| ASQA | arXiv:2204.06092, EMNLP 2022 | [read] — DR definition verified |
| QAMPARI | arXiv:2205.12665 | [search] — **we use this dataset; the citation must be right** |

### 4.3 Annotator reliability, the classical form of our ceiling arm

| work | id | mark |
|---|---|---|
| Dawid & Skene (1979) | no id recorded | [search] — bibliographic details unconfirmed |
| Fast Dawid-Skene | arXiv:1803.02781 | [search] |
| Crowd-Certain | arXiv:2310.16293 | [search] |
| Deep ensembles via confusion matrices | arXiv:2503.07119 | [search] |
| Multi-label extensions of annotator models | **no id** | **[reported]** — §3.13 of the novelty check |

The last row matters: it means "ours is set-valued rather than categorical" never
escaped the Dawid–Skene family. The distinction that works is votes versus
observations.

### 4.4 Generated position — the downgrade that hurts most

Our §3.5 established that *input*-side position work does not threaten the rank
claim, and that is correct. It was the wrong place to stop looking.

| work | id | mark |
|---|---|---|
| Single Answer is Not Enough: On Generating Ranked Lists with Medical Reasoning Models | **no id**, reported 2025 | **[reported]** |
| Long-form factuality decay across sentences | **no id** | **[reported]**; may duplicate arXiv:2505.15291, already an unread lead |
| SLAQ, short- vs long-form factual alignment | arXiv:2510.11218 | [read] — 51.3% at first fact to 30.1% at fifth |
| Serial Position Effects of LLMs | arXiv:2406.15981 | [read] — **input-side**; cite as contrast only |
| The Position Curse | arXiv:2605.07127 | [search] — input-side |
| Lost in the Middle follow-up | arXiv:2510.10276 | [search] — input-side |
| Listwise reranking under positional bias | arXiv:2604.03642 | [search] — input-side |
| EviRank | arXiv:2606.04727 | [search] — input-side |

**We can no longer write that no prior art exists for generated-position as a
reliability signal.** The surviving increment is narrower: the positions at which
several *heterogeneous* models independently generated the same claim, used as
*aggregation* evidence. Prior work asks whether one model ranks its own correct
answer highly.

Second-order consequence: if generated-position decay is documented, then **our
flat result on short lists is the more surprising finding**, not the gradient on
long ones. That inverts which half of the rank table is the contribution.

### 4.5 Informative stopping — omission strength has a statistical home

| work | id | mark |
|---|---|---|
| Statistical Models of Top-k Partial Orders | **no id**, reported 2024 | **[reported]** |
| The Illusion of Progress (length as hallucination detector) | arXiv:2508.08285 | [search] |
| Verbosity is not Veracity | arXiv:2411.07858 | [search] |

Reportedly argues that partial-ranking models overlook the information in list
length, and models the ranking jointly with the decision to stop. That is
omission strength, stated for partial orders.

**This downgrade improves the paper.** "We treat free-form enumeration as a
partial ranking with informative stopping" is a stronger sentence than "we
invented this feature," and it gives Proposition 2 a home in existing statistics.

### 4.6 F1-optimal set decisions

| work | id | mark |
|---|---|---|
| Lipton, Elkan & Narayanaswamy, F1-optimal thresholding | **no id**; likely findable under a title on optimal thresholding of classifiers to maximise F1 | **[reported]** |
| Adaptive Coverage Policies in Conformal Prediction | arXiv:2510.04318 | [search] — coverage, not F1 |

Batch dependence of F1-optimal decisions is established. Present Proposition 3 as
the decision layer with this citation, not as new decision theory. The closed-form
stopping characterisation stays as exposition.

### 4.7 Claim-level and cross-model checking

| work | id | mark |
|---|---|---|
| Finch-Zk, cross-model consistency | arXiv:2508.14314 | [read] — no position, no omission, no budget |
| Atomic Calibration | arXiv:2410.13246 | [read] — single-model, atomic-level |
| C3, cross-contextual consistency | arXiv:2608.10315 | [read] — cite for the TriviaQA self-consistency signal |
| MAD-Fact / FaStfact / VeriFact | arXiv:2510.22967 / 2510.12839 / 2505.09701 | [search] — evaluation, not aggregation |
| Prompt-Reverse Inconsistency | arXiv:2504.01282 | [search] |
| Self-consistency under prompt transformations | arXiv:2305.14279 | [search] |

### 4.8 When does aggregation help

| work | id | mark |
|---|---|---|
| Power and Limitations of Aggregation in Compound AI Systems | arXiv:2602.21556 | [read] |
| Co-failure ceiling | arXiv:2606.27288 | [read] |
| Disagreement-based analysis of ensemble usefulness | **no id**, reported NeurIPS 2023 | **[reported]** |
| Harnessing Collective Intelligence for Biomedical QA | arXiv:2508.01480 | [read] — plain union for list questions |

Do not claim to be first at predicting whether aggregation helps. Claim instead
that singleton reliability is a diagnostic for whether identity retains
information beyond agreement count.

### 4.9 Rank fusion baselines

| work | id | mark |
|---|---|---|
| Reciprocal Rank Fusion (Cormack, Clarke & Buettcher) | no id recorded, SIGIR 2009 | [search] |

---

## 5. What this obliges, in order

1. **Locate and read every [reported] item.** Six items, most with no identifier,
   two of which directly contradict claims the novelty check previously made.
   Until they are found, the downgrades in §4.4 to §4.6 are themselves
   unverified. `paper-lookup` needs no API key.
2. **Related Work owes citations** for generated position, informative stopping,
   and F1 decision theory. It currently has none of them.
3. **One sentence in §3.3 of `main1.tex` is now closer to wrong than right**:
   that we are not aware of prior work using graded omission and make no claim
   prior work supports it. It was carefully hedged and needs revisiting once the
   §4.5 lead is located.
4. **Delete any sentence claiming no prior art** for rank, omission, or the
   keep-count. True of the 22 Aug survey, not true now.
5. **15 references remain unread** and 52 in-text markers point at them.

---

## 6. Extra work: is the identity channel's null a bound or an underpowered miss?

The self-review flagged this as optional and it was never done. It is worth
doing, because a reviewer reading "+0.30, interval includes zero" has two
readings available and they have opposite consequences for contribution 1.

**What was computed.** A sensitivity analysis at the achieved sample size, using
the same paired bootstrap over 777 questions the paper already uses. Not observed
power, which is a deterministic function of the $p$-value and adds nothing; the
`statistical-power` skill is explicit that it should not be reported.

New module `analysis/power_check.py`, output `results/power_check.json`. It
required patching `weighting_v5.py` to dump per-question scores, which it
computes and previously discarded. That rerun also **reproduced the headline
exactly** (29.21 / 29.50 / 30.68 / 31.70), which is an independent
reproducibility check we did not have.

| comparison | observed | SE | 95% CI | MDE at 80% | MDE at 90% | TOST margin |
|---|---|---|---|---|---|---|
| pattern vs count | +0.30 | 0.328 | [−0.35, +0.92] | **0.92** | 1.06 | **±0.83** |
| scorer vs count | +1.47 | 0.348 | [+0.79, +2.17] | 0.98 | 1.13 | ±2.07 |
| scorer + budget vs count | +2.50 | 0.402 | [+1.69, +3.28] | 1.13 | 1.30 | ±3.14 |

**The answer is: a bound.**

- The design detects effects of **0.92 F1 or larger** at 80% power.
- A two one-sided tests procedure establishes **equivalence to zero within
  ±0.83 F1** at $\alpha = 0.05$. That is a formal claim, and it is stronger than
  quoting a confidence interval that happens to contain zero.
- So effects above roughly 0.9 F1 are excluded, while +0.30 itself cannot be
  separated from zero at this $n$.

**The contrast is what makes it persuasive.** The same design resolves the full
method's +2.50 against an MDE of 1.13, comfortably. This is a small effect
measured precisely, not a large one measured badly. A reviewer who suspects the
study is simply weak has that answered in one line.

Added to §5.1 of the paper as a paragraph headed "The null is a bound, not an
underpowered miss," registered as claim C051 against E127.

The bootstrap CI here, [−0.35, +0.92], differs from the paper's [−0.34, +0.90] in
the second decimal because the resampling seed differs. Worth reconciling to one
seed before submission so the two numbers match.

## 7. The ASQA reframe

### What the paper said

> On ASQA the ordering reverses. Our method beats counting everywhere but loses
> to reliability weighting at every size.

True, but it buries the fact that counting is a terrible reference point on this
benchmark.

### What the numbers actually say

Full six-agent pool, DR\* proxy, 2-fold cross-fitted. Source: `asqa_v5.log`.

| arm | DR\* | vs best single agent | vs counting |
|---|---|---|---|
| best single agent (`gpt-5.4`) | 27.03 | (reference) | +2.27 |
| multi-agent counting | 24.77 | **−2.27 [−3.52, −1.11]** \* | (reference) |
| reliability weighting | **27.45** | +0.42 [+0.02, +0.86] \* | +2.69 \* |
| pattern lookup (ceiling) | 27.20 | +0.17 [−0.30, +0.66] ns | +2.44 \* |
| ours, cover-label | 26.81 | −0.22 [−0.96, +0.54] ns | +2.05 \* |
| ours, DR-label | 26.88 | −0.15 [−1.11, +0.83] ns | +2.12 \* |

`*` = bootstrap interval excludes zero.

**Counting is significantly worse than just using the best agent.** Everything
else is within noise of it. Reliability weighting clears it by 0.42, which is
significant but small.

### Per-ensemble, the same picture

Source: `results/asqa_ensembles.json` and `asqa_ensembles.log`, all 42 subsets.

| ensemble size | counting below best single | counting *significantly* above best single | mean gap |
|---|---|---|---|
| 3 | 17/20 | 0/20 | −0.70 |
| 4 | 15/15 | 0/15 | −1.30 |
| 5 | 6/6 | 0/6 | −1.88 |
| 6 | 1/1 | 0/1 | −2.26 |
| **total** | **39/42** | **0/42** | |

The deficit **widens monotonically with pool size**. That is not noise, it is a
mechanism, and section 2 explains it.

### The sentence this licenses

> On ASQA the entire family of claim-level filters, ours included, is competing
> for the right to match one agent rather than to improve on an ensemble.

---

## 8. The mechanism, as a chain

Each link is measured, not inferred.

**1. The solo ratio is 0.47 on ASQA against 0.06 on QAMPARI.**
A claim asserted by exactly one agent is 47% as likely to be correct as one
asserted by two or more. On QAMPARI it is 6%.

**2. So a count threshold deletes good content.**
Counting drops solo claims. On ASQA roughly half of what it drops is correct.
On QAMPARI almost none of it is. This is why counting loses to the best single
agent, and why it loses *more* as the pool grows: a larger pool produces more
unshared content, so the filter has more correct material available to discard.

**3. Weighting recovers because it can cut selectively.**
Per-agent weights let a strong agent's unsupported claims survive a cut that a
count threshold applies uniformly. That buys back most of the gap, ending at
+0.42 over the best single agent.

**4. Our confidence slot has nothing in it.**
On ASQA stage 3 is filled with sentence position. ASQA responses average 3.5
sentences and 79% are 3 or 4 sentences, which sits inside the band where the
QAMPARI measurement shows position is flat. Measured directly on ASQA:
Spearman ρ = **−0.029** in the 3-to-4 sentence band (n = 4,064) and **−0.047**
pooled over solo claims. Source: `asqa_rank_effect.json`.

**5. Therefore the scorer adds variance, not signal.** See section 3.

---

## 9. The sharpest single piece of evidence

This is the part worth keeping, because it converts an assertion into a
measurement.

Our scorer takes the pattern probability as one of its input features and adds
the confidence signal on top. So its position relative to the pattern lookup is
diagnostic:

- where the confidence signal carries information, the scorer should sit **above**
  the pattern lookup;
- where it does not, the scorer should sit **below** it, by whatever the extra
  parameters cost in estimation variance.

| benchmark | pattern lookup | our scorer | difference |
|---|---|---|---|
| QAMPARI | 29.50 | 30.68 | **+1.18** |
| ASQA | 27.20 | 26.81 | **−0.39** (interval includes zero) |

That is what an empty feature block looks like from the outside. The model pays
for the parameters and gets nothing back. It is a much stronger claim than
"the signal is absent," because it is visible in the arm ordering itself.

---

## 10. Two things that cut against us, stated rather than buried

**Reliability weighting beats the ceiling on ASQA.** 27.45 against the pattern
lookup's 27.20, interval excluding zero. A constrained rule beating its own
population bound is the finite-sample case §3.1 sets aside: 6,015 clusters over
2^6 cells on ASQA, against 110,136 claims over 2^8 cells on QAMPARI. Reported as
a limit on the empirical bound, not on the construction. It is also a second
reading of the same data shortage that leaves the confidence slot empty.

**The ASQA pipeline is missing a feature block the QAMPARI pipeline has.**
`asqa_ours_ensembles.py` does not include the stacked pattern probability that
`weighting_v5.py:123` carries as an explicit block. On ASQA the pattern arm is
the strongest non-weighted baseline, so that omission plausibly costs the ASQA
arm. This was previously only in `UNRESOLVED.md` §7.1; it is now stated in the
paper. **It should be closed before the ASQA numbers are called final.**

---

## 11. What changed in §5.5

Only §5.5. Backup at `main1.tex.bak`.

| before | after |
|---|---|
| one paragraph, "we can say why it happens" | five headed paragraphs |
| — | **On ASQA, aggregating by agreement barely works at all** (the reframe) |
| — | **The solo ratio said so in advance** (the mechanism) |
| "The confidence slot is empty on ASQA, and this was predicted before it was measured" | **Our confidence slot is empty, so the scorer adds variance rather than signal** (adds the below-ceiling measurement) |
| buried in a "two further observations" paragraph | **The decision slot is not the problem** (own paragraph) |
| same paragraph | **One reading that cuts against us** (own paragraph) |
| — | **What this does and does not establish** |

New claims and evidence registered:

| id | fact | source |
|---|---|---|
| C048 | counting below best single in 39/42, deficit widens with pool size | E125, E126 |
| C049 | scorer sits +1.18 above the ceiling on QAMPARI and −0.39 below on ASQA | E125, E100 |
| C050 | the ASQA pipeline is missing the pattern-stack feature block | E103 |
| E125 | `asqa_v5.log`, full-pool arms with the vs-ceiling column | new |
| E126 | `results/asqa_ensembles.json`, counting vs best single per ensemble | new |

Registry state: 50 of 50 claims mapped, `check_consistency` passes,
`check_references` passes.

---

## 12. Still open on ASQA

Ranked by how much each could move the numbers.

1. ~~Add the pattern-stack feature and rerun.~~ **Closed 26 Aug: tested, worth
   $+0.01$ DR* over 42 ensembles, negative at size 6. The scorer still sits
   below the pattern lookup, so the gap is not caused by the omission. Not
   added; code reverted.**
2. **Resolve the cross-harness pattern discrepancy.** `asqa_dr2.log` reports the
   pattern ceiling at **27.37**; `asqa_rank2.log` and `asqa_v5.log` report
   **27.20**. The "weighting beats the ceiling by +0.25" claim depends on which
   is right. At 27.37 the gap is +0.08 and the point largely dissolves.
3. **`marginal` and `solo` weighting return byte-identical DR\*, STR-EM and
   ROUGE-L** on ASQA (27.45 / 46.01 / 22.04). Two different rules collapsing to
   the same decision needs an explanation before a reviewer asks for one.
4. **Single-linkage clustering.** `asqa_ensembles.py:114` is union-find over a
   cosine threshold, which is agglomerative clustering with single linkage, the
   one linkage that chains A→B→C when A and C are unrelated. ASC uses ordinary
   agglomerative at d = 0.15. This is the most obvious reviewer attack on the
   ASQA pipeline and the fix is small.
5. **`sentences()` drops sentences under 3 words**; ASC has no such filter.
6. **DR\* is a proxy.** A faithful DR needs the RoBERTa-SQuADv2 reader.

Items 3 to 5 do not change the story in this document, but each is a way a
reviewer could argue the ASQA pipeline is not a fair instantiation of the
template, which would weaken the "the slot is empty" explanation by offering a
competing one.

---

## 13. Why this is presentable

The paper now says: the template has a stated precondition, the precondition is
checkable from cached generations before any implementation work, and on ASQA the
precondition fails. The failure is located in exactly one of the two slots, and
the reason is measured three independent ways — the solo ratio predicting it in
advance, the position correlation being flat at ASQA's response lengths, and the
scorer landing below the ceiling it stacks on.

A method with a known domain of validity and a cheap test for it is a stronger
contribution than a method that claims to work everywhere. The remaining risk is
not the loss itself but item 1 above: if the missing feature block is what causes
the loss, the mechanism story is wrong and the fix is cheap. Run it first.

---

## 14. State of the paper after this iteration

| | |
|---|---|
| title, abstract, §1 | reframed around votes versus observations |
| §3.2 theory | new; three propositions |
| §3.1 ceiling | claim reduced to claim-local, identity-only |
| §5.1 | sensitivity paragraph added; null shown to be a bound |
| §5.5 ASQA | rewritten around best-single-agent as the yardstick |
| figures | 8, in `figur/` |
| tables | 12 |
| claims registered | 51 of 51 mapped |
| `check_consistency` | pass |
| `audit_claims` | fail by design; 23 claims rest on pre-fix or unread evidence |
| page count | untested since the theory section was added; likely worse |

New modules this iteration: `analysis/power_check.py`,
`analysis/pattern_cells.py`, `analysis/make_figures.py`. `weighting_v5.py`
patched to dump per-question scores; the rerun reproduced the headline exactly.

Backups: `main1.tex.preframing`, `main1.tex.pretheory`, `main1.tex.bak`,
`docs/NOVELTY_CHECK.md.bak`.
