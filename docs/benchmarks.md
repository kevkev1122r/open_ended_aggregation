# Benchmarks

A merging filter trades **precision against recall**. A benchmark can only adjudicate it
if its metric scores both. Of the four run here, two can.

| benchmark | metric | scores | can adjudicate a filter? |
|---|---|---|---|
| **QAMPARI** | set-F1 | precision + recall | **yes** |
| **QUEST** | P/R/F1/Recall-5 | precision + recall | **yes** |
| ASQA | STR-EM | recall only | no |
| FACTS | groundedness | precision only | no |

## QAMPARI

Entity-set QA — "what manga did Ryoichi Ikegami draw?" → 8 titles. Each list item is
natively an atomic fact, so no sentence splitter and **no summariser** is needed: the
output is the surviving item list, and nothing can hallucinate content absent from a
source response. Graded locally against gold aliases, no judge.

**Caveat: 97.2% of gold entries ship no aliases beyond the answer text.** Grading is
therefore near-exact string match, and correct answers phrased differently score zero
("Essendon" vs "Essendon Football Club"). A strict-vs-relaxed sensitivity analysis puts
this at roughly **−7 F1 across every arm**, with no conclusion changed — the penalty is
near-uniform. Report it; do not silently relax it.

## QUEST

Entity-seeking queries with implicit set operations — "Philippine remakes of South Korean
films or 2010s prison dramas". Gold is a set of Wikipedia titles, median 11 per query.
1,727 test examples, the same split ASC evaluated.

Much harder than QAMPARI (best single 10.86 vs 27.92) and its best agent is more dominant
(2.08× the rest vs 1.43×), which is why the weighting works here and not there.

**Cost warning:** QUEST cost ~$115 for 400 queries vs ~$13 for QAMPARI's 800. Cohere burns
the full 12k budget and returns empty on ~99 queries, and every backfill pass retries them.
Drop Cohere — coverage goes from 297/400 to **386/400** and the analysis is better powered.

## ASQA — cannot adjudicate

STR-EM is "fraction of interpretations whose short answer appears anywhere in the
response". Pure recall: no precision term, no length penalty. So concatenating every
agent is optimal by construction and the count filter's optimum is θ=1:

```
θ=1  53.70%   ← keep everything, best
θ=2  35.85%
θ=3  28.36%
θ=4  21.82%
θ=5  15.80%
```

Every arm collapses to union. Union "beats" the best single agent by +7.13, which is worth
nothing. This is a property of the metric, not of the method.

## FACTS — cannot adjudicate

`grounded` = fraction of checkable claims that are grounded. Precision-only; there is no
coverage term in the data at all. Biased the opposite way to ASQA.

## Adding one

Prefer set-valued benchmarks with an F1-style metric. ELI5 is ASC's fourth dataset but the
original release is defunct (Reddit API) and it is free-form prose, so the ASQA
recall-only trap likely recurs.
