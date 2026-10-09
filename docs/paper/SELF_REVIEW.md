# Peer-review working draft — REVIEW-LOCAL-OEA-001

> Private working document. This is an **author self-review**, not an independent
> review, and it does not substitute for the mentor and P.I. reviews the
> programme writing guide requires. Human verification of every location and
> figure is still required.
>
> **Locations.** Written against `PAPER_DRAFT.md` and re-checked against
> `main.tex`, which is now the source of record. Section numbering is unchanged
> through Section 7; Limitations became Section 8. Three items are resolved in
> `main.tex` and are marked below: M1, m4, m6.

## Intake record

- Reviewer capacity: `author_requested_reader` (self-review; conflict disclosed)
- Peer-review model declared: `open` — this describes **this** internal,
  non-anonymised pre-submission read. The target venue is undecided, so no venue
  review model has been checked.
- Declared processing plan: `local_deterministic_tools`, local-only, no external
  service, no network call.
- Intake validator: `READY_FOR_LOCAL_REVIEW`.
- Claim-evidence matrix: `claim_evidence_matrix.csv`,
  `VALID_WITH_ALIGNMENT_GAPS` — 7 supported, 10 partly supported, 1 not
  assessed.

---

# Comments to authors

## Evidence-bounded summary

The manuscript studies aggregation of set-valued answers produced by several
different language models, where the unit that can be right or wrong is an
individual claim rather than a whole response. It makes one structural argument
and one empirical one. The structural argument is that any aggregation rule whose
decision depends on a claim only through the set of agents that asserted it is
bounded by a direct estimate of the probability of correctness conditional on that
set; the class named includes reliability weighting, correlation-aware weighting,
and annotator-reliability models. The empirical argument is that on an entity-list
benchmark this bound is small and not distinguishable from zero, and that signals
drawn from the structure of each agent's own output — where a claim sits in that
agent's list, and how much each agent that omitted it said instead — together with
a per-question keep-count, clear the bound by a margin whose bootstrap interval
excludes zero. A second benchmark is presented where the same template loses to
reliability weighting, together with a statistic intended to predict which regime a
benchmark is in.

Principal claims and their evidence mapping are in `claim_evidence_matrix.csv`.

## Strengths

- **The central negative result is stated as a construction rather than a search
  outcome** (§3.1). This is the right form for the claim: it converts "the
  weighting schemes we tried did not help" into a bounded quantity, and it is
  independently corroborated from below by three different parameterisations that
  each land at or under the bound (§5.1).
- **The method is reported at a lower number than the best number measured**
  (§5.2, §5.4). The boosted blend scores 32.06 in distribution and is excluded
  because it transfers worse. Papers rarely do this and reviewers notice when they
  do.
- **The benchmark where the method loses is given a section, a mechanism, and a
  prediction made before the measurement** (§5.5). Reporting the ASQA sweep as a
  mirror image of the QAMPARI sweep is a stronger result than the QAMPARI sweep
  alone.
- **Limits are attached to the signals rather than to the paper's conclusion.**
  §5.5 states that the position signal is flat on short lists and that the
  QAMPARI gain depends on 70% of solo assertions sitting in very long lists. That
  is the kind of boundary a reader can use.
- **A previously circulated claim is retracted in the text** (§5.4, the pattern
  lookup does not collapse under distribution shift). Correcting an internal
  result before submission is the cheapest place to do it.

## Major comments

### Major comment M1

*(Resolved in `main.tex`.)*

- Location: §5.1, final paragraph — "reliability weighting is worth approximately nothing"; and Abstract.
- Observation: The pattern arm's gain over counting is +0.30 with a bootstrap interval of [−0.34, +0.90]. The text converts that non-significant interval into a statement of no value.
- Evidence or criterion: A confidence interval containing zero bounds an effect; it does not establish equivalence. The manuscript's own framing elsewhere is careful about this distinction.
- Why it matters: This is contribution 1. A reviewer who reads a null result presented as a zero result will discount the rest of the claim-handling in the paper, and the correction available is stronger than the current wording rather than weaker — the interval's upper limit is itself the result.
- Requested action: Replace the equivalence phrasing with the bound: at 95% the data rule out gains above roughly +0.9 F1 for any rule in the class. Optionally add an explicit minimum-detectable-effect statement for this comparison.

### Major comment M2

- Location: §5.5, final paragraph before "Three points are an observation" — ASQA reliability weighting 27.45 against the pattern lookup's 27.20.
- Observation: On one of the two benchmarks where the ceiling is testable, the constrained rule exceeds the ceiling by +0.25 with an interval excluding zero. The manuscript attributes this to finite-sample estimation on thin data but reports no quantity that lets a reader check the attribution.
- Evidence or criterion: §3.1 states the bound applies to the population optimum, and that the empirical lookup estimates up to 2^n cells. The claim that ASQA is data-limited is currently an assertion.
- Why it matters: This is the most likely single point of reviewer attack on contribution 1, because it reads as the central construction failing on half the benchmarks where it could fail.
- Requested action: Report cell occupancy for both benchmarks — number of occupied support patterns and claims per occupied pattern — so the reader can see the difference directly. Add a forward reference from §5.1 to the finite-sample caveat in §3.1 so the caveat is visible before the exception appears.

### Major comment M3

- Location: §5.3, Table 3; and §5.7, Table 10.
- Observation: Both tables are computed from the pre-fix candidate pool. Table 3 supports contribution 4, which is a headline claim.
- Evidence or criterion: §4.4 documents that the normalisation fix changed candidate counts and merged previously split candidates; the arms where it has been rerun moved by at most 0.14 F1, but that has not been checked for these two tables.
- Why it matters: A headline claim resting on a computation the paper itself describes as superseded is a correctness risk, not only a presentation risk.
- Requested action: Rerun `analysis/rank_crossover.py` and the weighting sweep on the fixed pool and update both tables. Keep the stale markers in the captions until that is done.

### Major comment M4

- Location: §5.4, the per-source paragraph following Table 4.
- Observation: The five per-source gains describe the logistic/GBM blend, which the same section excludes from the method, rather than the method as reported.
- Evidence or criterion: The source log reports the per-source breakdown only for that arm; the manuscript states this in the text.
- Why it matters: The paragraph is offered as evidence that gains are positive on every held-out question type for the proposed method. As written it evidences that for a different arm.
- Requested action: Regenerate the per-source breakdown for the scorer-plus-budget arm and replace the numbers. If that is not possible before submission, move the paragraph under the boosting ablation where it belongs.

### Major comment M5

- Location: §5.4, Table 4, "random 5-fold" column, blend row.
- Observation: Two harnesses disagree on this arm — 32.06 in the main run log and 32.29 in the leave-one-source-out log — under a protocol described identically in both. Every other row in that column agrees to the second decimal.
- Evidence or criterion: The count, pattern, scorer and scorer-plus-budget rows reproduce exactly across the two harnesses, which isolates the discrepancy to the boosted arm.
- Why it matters: The decision to exclude boosting from the method is made by comparing this arm's in-distribution and shifted scores. A discrepancy in the in-distribution term is a discrepancy in the input to that decision.
- Requested action: Diagnose the difference — the likely cause is boosting seed or early-stopping split handling — and report one number, or report both with the reason for the difference.

### Major comment M6

- Location: §5.4, Table 5.
- Observation: The full-data gain over counting on a held-out test set is +1.45 and +1.28 on two splits, against a cross-fitted headline gain of +2.50 in §5.1. The caption flags the discrepancy but does not resolve it.
- Evidence or criterion: The two protocols differ in training-pool size (527 against roughly 622) and test-set size (250 against 777), which plausibly accounts for part of the gap but has not been shown to account for all of it.
- Why it matters: A reader can read Table 5 as an independent estimate of the headline that comes in 40% lower. That reading needs to be pre-empted in the paper, not left to the reviewer.
- Requested action: Run two further test seeds. If the gap persists, decompose it into the pool-size and test-set-size components and state what remains unexplained.

### Major comment M7

- Location: §5.6, Table 9.
- Observation: The calibration table was measured without a saved artifact or a module in the analysis package. It is the only table in the manuscript whose provenance is an internal document rather than a rerunnable command.
- Evidence or criterion: §4.2 states that every result in the paper is computed from the cache with no further model calls, which implies rerunnability.
- Why it matters: The table supports a specific mechanical claim — that the keep-count is fed by overconfident probabilities in exactly the region it reads — that a reader may want to check, and the paper's reproducibility posture is otherwise unusually strong.
- Requested action: Add `analysis/calibration.py`, regenerate, and write the output to `results/`. If that is not done, remove the table and state the overconfidence qualitatively without the bins.

### Major comment M8

- Location: §6 throughout; References.
- Observation: 52 in-text citation markers point at 15 sources that have not been read directly. Among them is the annotator-reliability attribution used in §2 and §6.3 to characterise the pattern lookup as a Dawid–Skene model, whose bibliographic details are also unconfirmed.
- Evidence or criterion: The claim "our pattern arm is in substance a classical annotator model" is a statement about the literature, and the manuscript's own framing depends on it: presenting that arm as a borrowed baseline rather than a contribution is what makes the ceiling result a statement about prior work rather than about our own arms.
- Why it matters: A mis-supported citation is grounds for rejection independently of the result, and this particular one carries framing weight.
- Requested action: Read all 15 before they enter the bibliography, starting with the annotator-reliability family, the dataset papers, and the two verbosity references that bound how much of the verbosity finding may be claimed as new.

## Minor comments

### Minor comment m1

- Location: §1.1, the majority-vote and response-selection figures.
- Observation: These come from an earlier agent pool and the pre-fix parse, while every other QAMPARI number in the paper uses the current 777-question, 8-agent, post-fix protocol.
- Evidence or criterion: The source document for these numbers reports a different best single agent than §5.1 does.
- Why it matters: The motivating example uses a different experimental setup from the paper it motivates, which an attentive reader will notice on the first page.
- Requested action: Regenerate both figures under the current protocol, or state the motivating point without specific numbers.

### Minor comment m2

- Location: §4.2.
- Observation: The QAMPARI and ASQA agent rosters are listed in full; the TriviaQA roster is given only as "six on TriviaQA".
- Evidence or criterion: The manuscript lists rosters for the other two benchmarks, so the omission is asymmetric rather than a stated scope decision.
- Why it matters: TriviaQA carries one of the three points in the screen of §5.5, and a reader cannot assess pool comparability without the roster.
- Requested action: List the six TriviaQA agents, or state why the roster is not reported.

### Minor comment m3

- Location: Abstract; §1.5; §5.1.
- Observation: Gains are reported both in absolute F1 and as percentage improvements over the comparator.
- Evidence or criterion: The programme writing guide cautions against phrasing that inflates a contribution.
- Why it matters: A +2.50 absolute gain rendered as "+8.5%" reads larger, and both numbers appearing together invites the reader to ask which one the paper prefers.
- Requested action: Choose absolute F1 as the primary form throughout and keep percentages to at most one place per claim.

### Minor comment m4

*(Resolved in `main.tex`: Tables 3 and 4 have separate captions.)*

- Location: §5.3, the two tables sharing one caption.
- Observation: A single caption serves both the per-size gain table and the significant-win-count table.
- Evidence or criterion: The writing guide requires each table caption to be self-contained and to state its own takeaway.
- Why it matters: Under a venue template these will be typeset as two floats and the second will lose its caption.
- Requested action: Split into two captions, each stating its own takeaway, or merge into a single table.

### Minor comment m5

- Location: §3.2.
- Observation: The pipeline figure is present only as a Mermaid sketch serving as a build note.
- Evidence or criterion: The writing guide asks for a first-page method diagram and names tools for producing one.
- Why it matters: The two-slot framing is the paper's organising idea and is currently carried entirely by prose and one table.
- Requested action: Draw Figure 1 — one question, several agents' ordered lists, the pooled candidate set, the two signals highlighted, then the per-question cut.

### Minor comment m6

*(Resolved in `main.tex`.)*

- Location: §5.2, "That ordering is the opposite of what the project set out to test".
- Observation: First-person reference to the research programme's history rather than to the literature.
- Evidence or criterion: Anonymous submission; the sentence describes internal project history that a reader cannot verify.
- Why it matters: It reads as narrative rather than result, and it is not needed — the ordering stands on the ablation numbers in the same paragraph.
- Requested action: Rephrase to state the ordering against the prior literature's expectation, or cut the sentence.

### Minor comment m7

- Location: §2, §4.1, §5.5 — the `DR*` definition.
- Observation: The proxy is defined once and referenced three times with slightly varying phrasing.
- Evidence or criterion: The manuscript is consistent on the substance but not on the wording.
- Why it matters: Under a page limit this is redundancy that can be reclaimed.
- Requested action: Define once in §2 and reference by name thereafter.

## Methods, statistics, and reproducibility

**Assessed and sound.** The cross-fitting protocol is stated precisely and applies
to every fitted object including thresholds; the manuscript gives a concrete
reason for the strictness (an in-sample threshold moved a measured gain from
+0.08 to −1.25). Uncertainty is reported as paired bootstrap intervals over
questions with a stated resample count, which is the right pairing for
per-question metrics. The distribution-shift protocol holds out a whole question
source rather than random questions, which is the stronger of the two available
tests. The exhaustive ensemble sweep removes the "favourable pool" objection in a
way that a single result cannot.

**Assessed and requiring action.** Treating a non-significant interval as an
absence of effect (M1). Attribution of the ASQA ceiling inversion to data volume
without reporting the quantity that would show it (M2). Two harnesses disagreeing
on the arm that a stated design decision rests on (M5). An independent-looking
estimate of the headline that comes in substantially lower and is not reconciled
(M6).

**Not assessed.** The analysis pipeline was not independently re-executed for this
pass, so every number here is checked for internal consistency against the run
logs and for arithmetic coherence between sections, not reproduced. Seed
stability post-fix is unestablished — one fold assignment has been rerun on the
fixed pool and two have not.

**Reproducibility posture.** Strong apart from Table 9 (M7): all other results are
described as computable from the cached generations with no model calls, which is
an unusually checkable claim for this kind of paper and is worth preserving
intact.

## Ethics, transparency, figures, tables, and citations

No human-subjects, animal, consent, or privacy considerations arise; the study
uses public benchmarks and cached model outputs. No integrity concerns are
identified.

Transparency gaps are reporting gaps rather than integrity issues: the
availability statements are unwritten, the code link is not yet anonymised, and
the appendix is listed but unwritten. The manuscript discloses its two most
significant methodological substitutions — agents standing in for stochastic
samples in the ASC baseline, and a proxy for the published ASQA metric — in both
the Experiments section and the Limitations section, which is the right treatment.

Citations: see M8. The manuscript includes an explicit do-not-cite note
distinguishing input-side from output-side position effects; that note should
survive into the submitted version, because it is the distinction the novelty of
the position signal rests on.

## Limitations of this review

- This is a self-review by an author. It cannot substitute for independent review
  and does not clear the mentor and P.I. reviews the programme requires.
- The analysis pipeline was not re-executed. Numbers were checked against run
  logs for internal consistency only.
- Statements about the literature could not be checked for the 15 references that
  have not been read; M8 records this rather than resolving it.
- Venue formatting, page limits, and template compliance are out of scope because
  the target venue is undecided.
- Figure 1 does not yet exist and could not be assessed.

---

# Confidential comments to editor

> Kept separate per the skill's channel rule. In this project the equivalent
> channel is the mentor. Nothing here is ordinary scientific criticism; all of
> that is above.

## Reviewer disclosures

- Conflicts and editor clearance: The reviewer is an author of the manuscript.
  This is a self-review, disclosed as such, and is not offered as independent
  assessment.
- Competence limits or specialist review needed: No independent re-execution of
  the analysis pipeline was performed. Statements about unread literature cannot
  be assessed. Mentor review and P.I. review are both still required.
- Assistance or tools used and required disclosure: Local deterministic
  validators only — intake validation, claim-evidence matrix validation, and
  review lint. No external service, no network call, no manuscript text sent
  anywhere.
- Confidentiality or retention issue: None. The manuscript, registries, and this
  review are retained in the project repository, which is the authors' own.

## Editorial-process or integrity concerns

None. Two process notes for the mentor rather than concerns:

1. The programme writing guide requires the P.I. review to be booked at least one
   week before the deadline, and requires mentor feedback first. The items in
   `UNRESOLVED.md` sections 1 through 6 are the work that should land before that
   booking, since several of them can change table values.
2. The target venue is undecided, which currently blocks template selection, the
   page limit, and whether a separate Limitations section is required. That
   decision gates the final formatting pass.
