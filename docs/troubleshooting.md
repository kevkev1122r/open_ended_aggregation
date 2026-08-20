# Silent-failure modes

Every headline number in this project has been wrong at least once. The ones that survived
did so because a **control** was run, not because the code looked right. These are the
failure modes found so far — most produce plausible numbers rather than errors.

## Generation

**A long-lived generator silently drops ~50% of calls.** Measured: a fresh process
succeeds 36/36 on identical prompts, budget and worker count, while a long-running one
drops about half. `generate()` writes a row only `if txt:`, so every dropped call vanishes
leaving no trace — the `(qid, model)` cell is simply absent and the progress line, which
prints only spend and `empty`, does not move. **Run generation in passes and print
coverage after each.**

**Throttling is invisible.** The progress line never printed `usage["throttled"]`. A 429
storm and a healthy run look identical in the log.

**"Empty" is not one thing.** A reasoning model that spends its whole budget and returns
`finish="length"` yields empty content with no error. Alternate fields
(`reasoning_content`, `thinking`) are only trustworthy when `finish="stop"` — on
truncation they hold reasoning cut mid-sentence, and treating that as an answer invents a
wrong answer rather than recording a missing one.

**A too-small `max_out` looks like model failure.** Kimi and MAI-Thinking-1 return empty at
64 tokens and fine at 2000. Probe with a realistic budget or you will wrongly conclude a
model is dead.

**Spend is understated.** Several models have no `PRICE` entry, so `AZ.spend()` silently
omits them. Check `usage["unpriced"]` before quoting a cost.

## Grading

**Alias-poor gold penalises models unevenly.** 97.2% of QAMPARI gold entries have no real
aliases, so grading is near-exact match — and near-miss rates differ by model (11.7% vs
2.3%), because naming convention varies. Since per-model precision *is* the weight vector,
naming style leaks into the weights.

**Gold entries can be duplicated.** 3.4% of QAMPARI gold entries repeat within a question,
so recall is weighted by multiplicity rather than by distinct entity.

**Normalisation collisions are structurally harmless — verify rather than assume.** Grading
normalises both sides before matching, so two surface forms sharing a key always get the
same verdict; a collision cannot turn a wrong item right. Zero distinct gold answers
collide within a question.

## Analysis

**Threshold-sweep resolution.** A coarse grid can step over the optimum. With n agents,
support depends only on which subset asserts an item, so enumerate the ≤2ⁿ subset-sum
breakpoints and get the exact optimum instead of sweeping.

**A regression check that cannot fire.** "Uniform weights must equal count(θ=2), else the
dedup fix is lost" — with zero within-model duplicates that identity holds either way, so
the check cannot detect what it was written to detect.

**Caching a config-dependent step on the wrong key.** Generations are config-independent
and cache on `(qid, model)`. Composition depends on which representatives survived the
filter, so it must key on the representative set too — otherwise changing Θ silently
returns an answer composed under the old Θ.

**Inherited labels.** An arm annotated `[published]` in the code was a multi-agent
adaptation, not the published method, and the label propagated into reports. Grep for a
citation before putting a named method in a results table.

**Piping a long run through `tail` hides it.** Buffering means no output until completion;
a 57-minute run looked identical to a hung one. Use `python -u` and redirect.
