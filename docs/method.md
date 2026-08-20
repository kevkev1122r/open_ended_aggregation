# Method

## The setting

An open-ended question has a **set** of correct answers. Each agent returns a list. The
aggregator must output a new list scored by set precision, recall and F1.

Selection methods pick one agent's whole list. Merging assembles a new list from fragments
across agents — so it is not bounded by any single agent's output, and it sits formally
outside the co-failure ceiling of arXiv:2606.27288, which bounds any policy whose output
is almost surely one of the members' answers.

## The arms

All arms operate on the **same parsed item sets**. Only the keep-rule differs.

| arm | rule |
|---|---|
| best single | one agent's list, the best by mean F1 |
| union | keep every item any agent proposed (θ=1) |
| MV | keep items ⌈n/2⌉+1 agents assert |
| MA-count | keep items ≥2 agents assert |
| **MA-count + OW** | keep items where Σ weights ≥ Θ, weights = cross-fitted per-agent precision |
| OW — response selection | the published Optimal-Weights rule over whole responses |
| oracle selection | best single response per question; ceiling for anything that picks |

## Why OW degenerates

With global weights and open-ended answers, no two responses coincide, so the weighted
vote's argmax is always the highest-weighted agent. It reproduces best-single **exactly**
— +0.00 with a zero-width CI. That is a structural property, not a weak result.

The log-odds weight form fails for a second reason: `log(x/(1−x))` goes negative whenever
precision < 0.5, and every agent's precision here is well under 0.5. Every weight is then
negative, so *more* agreement means *less* support, and the optimal threshold is the one
that switches the filter off entirely.

## What the weighted filter actually computes

With n agents an item's support depends only on **which subset** asserts it, so a
threshold on subset sums has at most 2ⁿ distinct behaviours. Enumerating subset-sum
breakpoints therefore finds the exact optimum — no grid, no resolution artifact.

On QAMPARI at Θ*, the weighted filter collapses to a two-term discrete rule:

> keep an item if **≥2 agents** assert it, **OR** if **the best agent** asserts it

Substituting any other agent into that slot loses ground. So the mechanism is *admit the
best agent's solo claims on top of consensus*, and the risk is misidentifying that agent.

## Why weighting helps on one benchmark and not the other

A weight only earns its keep when an agent is worth trusting **alone**. Measured on
QAMPARI:

| # agents asserting | P(correct) |
|---|---|
| 1 | 1.8% |
| 2 | 20.4% |
| 3 | 32.0% |
| 4 | 49.4% |
| 5 | 59.3% |

Every agent's **solo** precision is below the 20.4% that *any* two agreeing agents give
(the best agent: 28.7% overall but only **9.6%** alone). Marginal precision is therefore
the wrong conditional: the decision weighting actually changes needs
`P(correct | this agent asserts AND no one else does)`, not `P(correct | this agent asserts)`.

Across 56 five-of-eight subsets on both benchmarks, weighting gain tracks **relative
dominance** `best / mean(rest)` (r = +0.452 QAMPARI, +0.473 QUEST) and correlates
*negatively* with aggregation gain. Weighting pays off exactly when merging-vs-best-single
stops paying.

## Relationship to ASC — read this before naming anything

**No arm here is published ASC.** Published ASC
([arXiv:2405.13131](https://arxiv.org/abs/2405.13131), EMNLP 2024):

| | published ASC | `MA-count` here |
|---|---|---|
| pool | m=50 stochastic samples from **one** model | one response per **different** agent |
| clustering | normalised edit distance (list mode) | same |
| strength | count of cluster members | same |
| Θ | tuned on a validation set | swept on eval (except where stated) |
| final step | LLM composes survivors (`P_combine`) | **absent** |

Counting over agents rather than samples measures **inter-model agreement**, not
**self-consistency**. Same arithmetic, different estimand. `methods/asc.py` implements the
paper faithfully with agents substituted for samples; the composition step has not yet
been run, so every ASC-shaped number so far is steps 1–5 only.

ASC published on QAMPARI *and* QUEST, so a reviewer will ask for it.

## LLM-judged clustering — a negative result

`methods/llm_cluster.py` replaces edit-distance clustering with an LLM grouping call
(one call per question, not O(n²)). It **loses ~6 F1** regardless of model size
(gpt-5.5 24.80, gpt-5.4-mini 24.94, vs edit distance 31.18) because it merges
distinct entities — `['UMass Amherst', 'University of Nebraska']` — and only the cluster
representative survives, so a false merge deletes a correct answer.

Note the axis confusion this exposed: alias poverty in gold is a **prediction↔gold**
problem; clustering operates on **prediction↔prediction**. Merging candidates cannot fix
gold that ships no aliases.
