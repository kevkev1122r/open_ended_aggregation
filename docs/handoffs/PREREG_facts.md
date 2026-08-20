# Pre-registration — FACTS Grounding aggregation run

Written **2026-08-15, while generation was still running**, before any 300-item result
was seen. The 36-item pilot had been seen; everything below that the pilot informs is
marked as such. The point is that the primary comparison and the success criterion are
fixed in advance, because this session has already produced several findings that came
from choosing an analysis after seeing which direction helped.

---

## Why this run exists

Three previous attempts could not test the method:

| attempt | why it could not answer the question |
|---|---|
| v2 suite (TriviaQA/GSM8K/MedQA/MMLU-Pro) | median answer 4 words with a canonical form; exact matching is already the *correct* similarity function, so the kernel can only add noise. 62% of items structurally incapable of showing an effect. |
| HaluEval summarisation | scored 4%: a 28-word reference is one arbitrary selection of facts from a 388-word article. Reference matching measures agreement with that choice, not correctness. |
| FACTS with binary response-level grading | 97.1% best single, 2.9 pts headroom. Collapsing a 214-word answer to one bit destroys the signal. |

FACTS Grounding with **sentence-level** grading is the first configuration where all
preconditions hold at once: 214-word answers, no canonical form, verification independent
of wording, 61.1% contested, 19.4 pts headroom.

---

## Primary comparison — fixed in advance

**Metric.** Mean groundedness of the selected response — the fraction of checkable
sentences that are `supported`. Continuous, not thresholded.

Justification, measured on the pilot and stated as the reason for the choice:
`r(centrality, groundedness) = +0.419, p<0.001` but `r(centrality, binary@1.0) = +0.038,
p=0.62`. A threshold at 1.0 demands every sentence be grounded and destroys the signal
the benchmark carries. For long-form generation the outcome is continuous; thresholding
it is a modelling choice that has to be justified, and here it cannot be.

**Primary hypothesis.** The similarity kernel beats exact matching when answers are long:

> `medoid/cluster` > `majority vote`, and `KWA` > `OW exact`

**Success criterion.** A gain whose 95% paired-bootstrap CI excludes zero. Anything whose
interval spans zero is reported as unresolved, not as a direction.

**Secondary, pre-specified.**
1. `KWA` vs `medoid/cluster` — does adding weights to the kernel help? §2 says KWA never
   beats the better of its ingredients; this has now reproduced twice. Pilot suggests it
   reproduces a third time (KWA 90.28 vs medoid 93.43).
2. `KWA label-free` vs `KWA supervised` — how much does EM cost versus knowing the truth?
3. EM's Spearman against true accuracy. Pilot: **−0.205**, i.e. anti-correlated.
4. Every method vs the best single model — the only bar that matters for deployment.

---

## What would falsify the method

If, on 300 items, `KWA` does not beat `OW exact` with a CI excluding zero, the method does
not work. This is the regime it was designed for — long free-form answers, no canonical
form, exact matching 97% blind — and no benchmark-shape objection remains. That conclusion
should be recorded rather than followed by a fourth benchmark.

## What is NOT being claimed

- n=300 gives roughly ±3–4 points on these comparisons. Effects near 1 point stay
  unresolvable. A null is "underpowered", not "no effect".
- Weights come from binary accuracy, which the pilot shows is a poor signal here.
  Estimating them from continuous groundedness is a **change proposed after seeing pilot
  data** and is therefore explicitly exploratory. It is reported separately and is not
  the primary comparison.
- The judge is a single model (`gpt-5.4-mini`); DeepMind uses three and averages. Absolute
  numbers are not comparable to their leaderboard. Relative comparisons between our five
  models are the object of study.
- Documents filtered to ≤1500 words (728 of 860), so this is not the full public set.

## Analysis is frozen

`analyze_facts.py` is written and dry-run on the pilot. The continuous-metric variant is
specified above. No further aggregator, threshold, encoder or filter will be added after
the 300-item numbers are seen; if something new is tried, it gets reported as exploratory
under its own heading.
