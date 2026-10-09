# HANDOFF — 23 August 2026, evening

**Read `HANDOFF_2026-08-23.md` (morning) first, then this.** The morning handoff's
framing of the open problem still stands and is not repeated here. This handoff
**supersedes its §1 numbers** and **retracts one of its §2 claims**, because a
parsing bug was found and fixed and every QAMPARI number moved.

**Nothing is committed.** `git status` at time of writing:

```
 M open_ended_aggregation/benchmarks/qampari.py     <- the fix
 M results/loso.json
 M results/weighting_v5_seed0.json
?? open_ended_aggregation/analysis/asqa_rank_effect.py
?? open_ended_aggregation/analysis/budget_trace.py
?? open_ended_aggregation/analysis/learning_curve.py
?? open_ended_aggregation/analysis/omission_effect.py
?? open_ended_aggregation/analysis/rank_effect.py
?? results/{asqa_rank_effect,learning_curve,learning_curve_test1,omission_effect,rank_effect}.json
```

Untracked logs also written to the repo root: `v5_seed0_fixed.log`,
`loso_fixed.log`, `learning_curve.log`, `learning_curve_seed1.log`,
`rank_effect.log`, `asqa_rank_effect.log`, `omission_effect.log`.

---

## PART I — THE SCIENCE

## 1. The parsing bug, which is why everything else moved

### What it was

`<|START_TEXT|>` and `<|END_TEXT|>` leaked into generations — **1,519
occurrences, every single one from `Cohere-command-a-plus`.** No other agent
leaked. They are baked into the already-parsed `items` arrays inside
`data/qampari_asc800.jsonl`, so a fix at generation-parse time alone would never
have reached them without regenerating.

The damage came from `QA.norm`, which maps every non-alphanumeric to a space:

```
"<|START_TEXT|>Harvard University"  ->  "start text harvard university"
```

That is a candidate string that can never match gold. Measured before the fix:
**1,244 of 110,801 candidates (1.12%), of which ZERO matched gold.** Pure noise
padding the precision denominator of every arm.

### The fix

`strip_special()` added to `open_ended_aggregation/benchmarks/qampari.py`,
called from **both**:

- `norm()` — must run BEFORE `_PUNC`, and this is the call that reaches the
  cached generations, so it is the one that actually mattered;
- `parse_list()` — keeps any future generation clean at parse time.

```python
_SPECIAL = re.compile(r"<\|[^|>]{0,40}\|>")

def strip_special(s):
    return _SPECIAL.sub(" ", str(s))
```

Verified after: candidates 110,801 -> **110,136**, zero leaked tokens remaining.

### Why counting moved but best-single did not

Because merging matters more than deletion here. When Cohere emitted
`<|START_TEXT|>Harvard University` and another agent emitted
`Harvard University`, those were **two candidates with support 1 each** — so
counting at theta=2 dropped both. After the fix they are **one candidate with
support 2**, which counting keeps. **The bug was systematically excluding Cohere
from consensus** on roughly 665 candidates. Best-single is unchanged because
Kimi never leaked.

---

## 2. New headline numbers — replaces §1 of the morning handoff

QAMPARI, 777 q, 8 agents, set F1, 5-fold cross-fitted by question.
Source: `v5_seed0_fixed.log`.

```
  arm                     F1     was      vs count
  best single (Kimi)    27.40   27.40         --
  count                 29.21   29.35    (reference)
  pattern               29.50   29.60    +0.30      ns
  rank scorer           30.68   30.79    +1.47       *
  rank scorer + budget  31.70   31.74    +2.50       *
```

Every arm fell by <=0.14, no significance verdict flipped, and **the method's
margin over counting GREW, +2.39 -> +2.50.** The fix is hygiene, not a result —
which is the good outcome, because it means the headline was not resting on a
parser bug.

Underlying data after the fix: **110,136 candidate claims, 3.57% correct,
96,548 (87.6%) asserted by exactly one agent.**

`P(correct | exactly k agents assert)`, measured pre-fix (regenerate post-fix
before quoting): 1.2% / 11.1% / 17.4% / 27.2% / 36.5% / 46.3% / 51.9% / 56.5%
for k = 1..8.

### Leave-one-source-out — `loso_fixed.log`

Trains on 4 of QAMPARI's 5 structurally different question sources, tests on the
fifth, so the model has never seen that *kind* of question.

```
  arm                    random    LOSO   shift   LOSO vs count
  count                   29.21   29.21   +0.00     (reference)
  pattern                 29.50   29.44   -0.06   +0.24      ns
  rank scorer             30.68   30.57   -0.11   +1.36       *
  rank scorer + budget    31.70   31.49   -0.21   +2.29       *
  BLEND + budget          32.29   31.43   -0.86   +2.22       *

  per-source gain, each source held out entirely (BLEND+budget in the log):
    wikidata_simple          n=327  +1.41
    wikidata_intersection    n=164  +2.26
    wikidata_comp            n=149  +2.80
    wikitables_composition   n= 84  +3.31
    wikitables_simple        n= 53  +3.73
```

Headline holds and strengthens under shift: **31.49, +2.29 over counting**, up
from +2.13. All five sources gain.

`* = bootstrap 95% CI over questions excludes 0.`

---

## 3. A CLAIM THAT BROKE — do not repeat it

Every prior write-up leaned on: *the pattern lookup collapses under LOSO
(+0.25 -> +0.02), so the protocol detects memorisation and the rank features
survive it.*

**That was partly an artifact of the parsing bug.** Post-fix `pattern` goes
+0.30 -> +0.24, a shift of −0.06 rather than −0.23. It does not collapse.

Worse, on re-reading: **`pattern` was never significantly better than counting
under either protocol**, before or after the fix. Its CI included zero every
time. The "collapse" was movement between two non-significant numbers — far
weaker evidence than it was presented as.

**What DOES survive** as a genuine LOSO discriminator is boosting: `BLEND +
budget` degrades −0.86 against rank+budget's −0.21, and that gap **widened**
after the fix. So LOSO still earns its place in the paper — it justifies dropping
gradient boosting. It does **not** indict the pattern lookup. Rewrite §10 of the
explainer and any paper draft accordingly.

---

## 4. Five new analysis modules

All run from cached data, zero API calls. All added under
`open_ended_aggregation/analysis/`.

### 4.1 `rank_effect.py` — rank is LENGTH-CONDITIONAL, and this is the real limit

Solo assertions only (removes the support confound), stratified by the asserting
agent's own list length (removes the verbosity confound):

```
  list length   1st quintile   last quintile   Spearman(rank, correct)
  1-5 items         2.2%           4.6%              +0.084
  6-20              5.2%           2.8%              -0.048
  21-100            5.4%           1.2%              -0.102
  101+              0.9%           0.1%              -0.069
  pooled solo                                        -0.121
```

**On short lists there is NO usable rank signal** — it is flat, or mildly
reversed. Excluding 1-item lists (which come disproportionately from the two
weakest agents at 0.3% correct), positions 1 / 2-3 / 4-5 in the 2-5 band score
3.8% / 5.4% / 5.1%. Still flat.

The feature works on QAMPARI because **70% of solo assertions sit in lists of
101+ items.**

Control that matters: `P(correct | first position)` by length band is
**2.2 / 4.1 / 4.4 / 4.3%** — flat. So the effect is genuinely positional and not
verbosity in disguise, which independently supports the v3 decision to ablate
`verb` out.

The pooled unstratified table (29.2% at position 1 falling to 0.2% at 71+) is
mostly the support confound. **Do not quote it.**

### 4.2 `asqa_rank_effect.py` — explains the ASQA defeat mechanistically

ASQA responses average 3.5 sentences; **79% are 3-4 sentences**, squarely inside
QAMPARI's flat band. Prediction was made in advance and confirmed:

```
  response length   1st    2nd    3rd    4-5    Spearman     n
  1-2 sentences      --   22.4%    --     --     -0.035     203
  3-4 (79% of data) 23.8%  25.9%  22.4%  22.8%   -0.029   4,064
  5-7                5.2%  29.1%  14.5%  14.2%   -0.022     947
  pooled solo                                    -0.047
```

`ours` loses to reliability weighting on ASQA because **the confidence-signal
slot is empty there**, not because the method is wrong. Report it as a mechanism.
It agrees with the solo-ratio screen from the other direction, and it turns an
unexplained defeat into a predicted one.

**Unaffected by the parsing fix** — ASQA does not route through `QA.norm`.

### 4.3 `omission_effect.py` — real, but the code overstates it

Solo claims only, stratified by the asserter's own list length. Columns are
increasing mean log(list length) of the seven silent agents:

```
  asserter len   silent wrote least ------------> silent wrote most
  1-20 items          5.7%   3.7%   2.5%   1.0%
  21-100              3.7%   2.3%   2.5%   1.8%
  101+                0.3%   0.2%   0.2%   0.7%

  pooled Spearman(silent-agent volume, correct) = -0.048   n=96,548
```

Correctly signed, holds in the first two bands only, **about half the strength of
rank**.

Also confirmed the list-length spread the feature relies on:

```
  Kimi-K2.5           median 13   mean 72.2   max 2431
  DeepSeek-V4-Flash   median  7   mean 28.5   max 2398
  gpt-5.4-nano        median  1   mean 24.3   max 1067
  gpt-5.4             median  6   mean 18.9   max  562
  MAI-Thinking-1      median  8   mean 15.3   max  658
  grok-4.3            median  7   mean 10.4   max  400
  Cohere-command-a+   median  3   mean  6.7   max  140
  gpt-5.4-mini        median  3   mean  4.9   max   77
```

So "median 1 to mean 71" in the docstring checks out.

**Three corrections needed:**
1. `weighting_v3.py`'s docstring calls this "a large, entirely unmodelled
   signal." **Large is not supported.** Say "real but small."
2. A confound nothing here removes: questions where every agent writes a lot are
   plausibly broader questions, so silent-agent volume partly proxies question
   difficulty rather than evidence about the specific claim.
3. **No external citation exists** — it is our own idea. `NOVELTY_CHECK.md`
   supports "nobody else uses it," which is a *novelty* claim, not a *support*
   claim. The paper must say "we introduce" and never imply prior work backs it.

### 4.4 `learning_curve.py` — the supervision cost, and why fitting transfers

Fixed held-out test set, training subsets of increasing size drawn from the
remainder, five independent draws each, run on two different test splits:

```
  N train   split A   split B   spread of rank+budget across draws
     25      -1.10     +0.18                 +-1.64
     50      +0.62     -0.20                 +-0.22
    100      +0.97     +0.54                 +-0.21
    200      +1.10     +1.00                 +-0.34
    350      +1.55     +1.11                 +-0.15
    527      +1.45     +1.28                   --
```

Two readings, both load-bearing:

- It **saturates** — ~77% of the gain by 200 questions, flat after. That is the
  shape you would NOT see if per-question heterogeneity were the binding
  constraint.
- The draw-to-draw spread **collapses** from +-1.64 to +-0.15. This is the direct
  answer to *"open-ended questions are all different, so how can coefficients
  transfer"*: by N~200 it no longer matters **which** questions you trained on.
  Because the features are normalised by their own question's quantities, the
  coefficients describe how LLMs emit lists, not the subject matter.

Also: at N=25 the method **loses** to counting. There is a real floor around 50.

**UNRESOLVED — chase before publishing.** The full-data gain here (+1.45, +1.28)
is well below the cross-fitted headline (+2.50), on both splits. Part is the
smaller training pool (527 vs ~622) and a 250-question test set instead of 777,
and on split A the method reproduced almost exactly (31.68 vs 31.74) while
counting came in high. But it is consistently lower and should be understood, not
written off. **Run a third and fourth `--test-seed`.**

### 4.5 `budget_trace.py` — pedagogy, not a result

Prints the expected-F1 curve candidate by candidate for one held-out question, so
the argmax is visible rather than asserted. Built because "how is k derived" was
the question the 22 Aug meeting could not answer from the code. Example output:

```
  question: which cities or municipalities are located in province of basilan
  35 candidates, true G = 12, predicted Ghat = 11.64

   k  candidate            p     sum p   2*sum/(k+Ghat)  correct?  true F1
   1  isabela city      0.780     0.78       0.1235       yes      0.154
  12  al barka          0.503     7.48       0.6329       yes      0.833
  13  akbar             0.470     7.95      >0.6454<      yes      0.880  <- CHOSEN
  14  lamitan           0.062     8.01       0.6250       yes      0.923  <- true optimum
  15  basilan province  0.054     8.07       0.6056        -       0.889

  chosen k=13 -> F1 88.00 | oracle k=14 -> 92.31 | keep-everything k=35 -> 51.06
```

Hand this to anyone onboarding. Run with `--qid <id>` to inspect any question.

**This example also exposes the alias problem concretely:** `lamitan city` at
row 10 is graded WRONG and `lamitan` at row 14 is graded CORRECT, while
`isabela city` at row 1 is CORRECT and `isabela` at row 16 is WRONG. Identical
surface variation, graded inconsistently in opposite directions on the same
question. That is the 97.2%-no-aliases issue costing real F1.

### 4.6 Calibration (run inline, NOT yet a module — worth writing one)

Cross-fitted, post-fix, 110,136 claims. Mean predicted 3.58% vs base rate 3.57%.

```
  predicted p       n     mean pred   actual   gap
  0.00-0.02    81,541        0.4%      0.4%    0.0
  0.02-0.05    12,278        3.2%      4.0%   +0.8
  0.05-0.10     6,925        7.1%      7.5%   +0.5
  0.10-0.20     3,876       13.8%     13.4%   -0.5
  0.20-0.35     2,401       26.9%     26.1%   -0.9
  0.35-0.50     1,602       42.1%     39.5%   -2.6
  0.50-0.70     1,323       58.7%     55.1%   -3.6
  0.70-1.00       190       74.8%     68.9%   -5.8
```

Calibration is excellent where the mass is and **overconfident by up to 6 points
at the top — exactly where the budget operates.** Since the budget substitutes
`sum(p)` for the unknown hit count `H`, it therefore **overstates expected hits
for the kept set**, biasing `k` toward keeping too much.

This also reframes the v4 note that "isotonic changed nothing": **74% of claims
sit in the bottom bin**, so a globally-fitted isotonic curve is dominated by a
region that is already perfect and barely moves the 3,115 claims above 0.35 that
actually get kept. It would correctly report no change while leaving the only
region that matters untouched.

**Untried, cheap, no generation:** recalibrate on the training fold weighted
toward (or restricted to) the high-p region, then rerun the budget. Cannot
threaten the headline either way — ordering is calibration-invariant, so
`rank scorer` at 30.68 stands regardless; only the +1.02 the budget adds rests on
the sums being right. `scikit-learn` (now installed, see Part II) has
`IsotonicRegression` and `CalibratedClassifierCV`.

---

## 5. The generality problem (morning handoff §3) — where it landed

The three pipelines already share a five-stage skeleton. **Stages 2 and 4 are
literally the same code** in all three files: `frozenset` of asserting agents,
then `fit_logistic(..., ridge=1.0)` imported from `beyond_pattern`.

Only two stages vary, **and they vary for two different, declarable reasons:**

| stage | QAMPARI | ASQA | TriviaQA | determined by |
|---|---|---|---|---|
| 1 pool -> units | normalised entity strings | sentence clusters | distinct terse answers | answer format |
| 2 support pattern | identical | identical | identical | — |
| **3 confidence signal** | list rank + omission | sentence position | self-consistency | **answer format** |
| 4 cross-fitted logistic | identical | identical | identical | — |
| **5 decision rule** | per-question budget | swept threshold | argmax | **metric shape** |

**The reframe to pitch: one method, two typed slots, each keyed to a stated
property of the task.** This is better than the morning handoff's framing A,
which lumped both into "the signal slot" — naming *two* slots and saying which
task property fixes each is what turns a bag of tricks into a method.

Plus a **precondition**, which is a strength rather than a hedge: the stage-3
slot only pays when the ceiling binds, and the **solo ratio predicts that in
advance from cached data** (QAMPARI 0.06, ASQA 0.47, TriviaQA 0.24).
Contribution = template + a screen that says whether the expensive slot is worth
filling.

**Do not oversell stage 5.** The metric-matched budget for ASQA was already
derived and tried — the `adapt-*` and `learned-budget` arms in `asqa_rank2.log`
score 24.98-26.70, all below the swept threshold. The ASQA failure is upstream in
stage 3, not in stage 5.

**Free experiment the unification surfaces:** `asqa_ours_ensembles.py` has **no
pattern-probability stack feature**, while on QAMPARI that is an explicit feature
block (`blk("stack", 1)`, filled at `weighting_v5.py:123`). Yet on ASQA
`pattern` is the strongest non-weighted arm (27.20 vs count's 24.77). The one
benchmark we lose on is missing the feature block carrying most of the signal
there. **Add it and rerun before writing the ASQA table.**

---

## 6. Errors found in the repo — verify, then fix

1. **`docs/method.md`'s support-count table is stale.** It prints 1.8 / 20.4 /
   32.0 / 49.4 / 59.3. Measured pre-fix on the current pool: 1.2 / 11.1 / 17.4 /
   27.2 / 36.5. Regenerate post-fix.

2. **ASC has NO composition step for list datasets.** Verified by reading the
   EMNLP PDF, section "Adapting ASC to list style datasets": *"The final answer is
   just the list of selected cluster representatives."* But `methods/asc.py`
   lists composition as step 6, and `method.md`'s comparison table marks its
   absence as a gap in our arm. **Our `--no-compose` run IS the paper's list
   pipeline**, so the QAMPARI ASC baseline is closer to paper-exact than the docs
   claim, and one entry in the morning handoff's §5 blocked list disappears.
   Check §A.1 for whether the quoted `P_combine` is the prose prompt adapted for
   lists.

3. **`method.md`'s LLM-clustering numbers have no backing.** It reports −6 F1
   (24.80 / 24.94 vs 31.18). The `asc_llmcluster*.log` files contain only the
   clustering step, not downstream scoring, and there is no `results/` entry. The
   24.80 that grep finds is an unrelated single-agent score in
   `asc_nocompose.log`. Re-run or soften.
   *The qualitative finding IS well supported by the logs:* it correctly merges
   `['Warner Bros', 'Warner Bros. Pictures']`, `['MGM', 'Metro-Goldwyn-Mayer']`,
   `['University of Oxford', 'Oxford University']` — but also merges
   `['Yale Law School', 'University of Michigan Law School']`,
   `['UMass Amherst', 'University of Nebraska']`, at a 7.6% merge rate. The cost
   is **asymmetric**: only the representative survives, so a false merge
   *deletes* a correct answer.

4. **`weighting_v3.py`'s omission docstring** — see §4.3.

5. **`asqa_ensembles.py:114` is single-linkage.** Union-find over a cosine
   threshold *is* agglomerative clustering with single linkage, the one linkage
   that chains A->B->C when A and C are unrelated. ASC uses ordinary agglomerative
   at d=0.15 over roberta-large SimCSE. Switching linkage is a small change worth
   testing and is the most obvious reviewer attack on the ASQA pipeline. Also
   `sentences()` drops sentences under 3 words; ASC has no such filter.

6. **The headline is 31.70, not 32.01.** All of 32.01 / 32.06 / 32.29 are
   `BLEND + budget`, the boosted arm the morning handoff removes.

---

## 7. Literature — one verified citation, one trap

**CITE THIS for the rank feature:** Islam, Lauscher & Glavaš, *The Curious Case
of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers*,
[arXiv:2510.11218](https://arxiv.org/abs/2510.11218). SLAQ framework, 16 models:
accuracy falls **51.3% for the first requested fact to 30.1% for the fifth**,
~5% per position, strictly linear, attributed to cumulative retrieval load. They
explicitly contrast it with U-shaped *input*-position effects. Verified by
reading the paper, not a search summary.

**DO NOT CITE** lost-in-the-middle, LLM-as-judge position bias, listwise
reranking bias, or recommender candidate position bias. Those are position in the
**input**. Our feature is position in the model's **own generated output**.
Different phenomenon; a reviewer will catch it immediately.

Unread leads, still `[search]`-only: arXiv 2505.17510 (models emit the
most-likely label first), 2605.12446 (ORCE — generation order reflects verbalised
confidence), 2505.15291 (hallucinations cluster at the end of long outputs). The
literature is **not unanimous**; at least one line argues hallucinations appear
*earlier*.

**Tension to own rather than be caught on:** SLAQ finds a clean decline across
positions 1-5; we find none across positions 1-5. Likely because SLAQ *requests*
five facts so later ones are dredged under pressure, while our short lists are
self-truncated — the model stopped because it ran out. In our long lists the
model *is* under exhaustive-listing pressure and the gradient reappears. That
reconciles both and is testable.

**ASC's actual method**, verified from the EMNLP PDF for the paper-exactness rule:

| | prose (ASQA, ELI5) | lists (QAMPARI, QUEST) |
|---|---|---|
| unit | individual sentences, NLTK | each list item directly |
| clustering | agglomerative over roberta-large SimCSE, **d=0.15** | normalised edit distance |
| representative | longest sentence | first item |
| filter | count < Theta, tuned on validation | same |
| final step | LLM summarise with `P_combine` | **none** |

§3.1 concedes the unit problem outright: LLM decomposition "can produce better
quality atomic facts" but was too expensive. **So ASC does not atomize prose
despite the name** — it has the same unit mismatch we do, and proper
FActScore/SAFE-style decomposition is the thing ASC wanted and could not afford.
8,393 ASQA sentences with a small model is well inside budget.

---

## 8. What must be rerun — nothing below reflects the parsing fix

| module | why | cost |
|---|---|---|
| `rank_crossover.py` | 210-ensemble claim, a headline result | medium |
| `learning_curve.py` | ran pre-fix, cited in the explainer | slow (2 splits x 5 draws) |
| `rank_effect.py` | ran pre-fix, cited in the explainer | fast |
| `omission_effect.py` | ran pre-fix, cited in the explainer | fast |
| `asc_ensembles.py` | routes through `Q.norm`; edge cache must be rebuilt | slowest |
| `gain_predictors.py` | routes through `Q.norm` | medium |
| `weighting_v2..v9` | the ~60-rule sweep, if any of it is quoted | slow |

**Already post-fix and correct:** `weighting_v5` (headline), `loso`,
`budget_trace`, the calibration check.

**Unaffected entirely:** everything ASQA (`asqa_*`), everything TriviaQA
(`triviaqa_*`).

Expect small movements — 1.12% of candidates, concentrated in one weak agent —
but "expect" is not "checked."

**Also stale: the method explainer artifact** at
`https://claude.ai/code/artifact/a88dbf9c-a15d-4cec-9017-bdc61f5a265d`. Twelve
sections, every number carrying a provenance chip (code / log / measured /
paper). It has pre-fix numbers throughout and, more seriously, **§10 still tells
the retracted pattern-collapse story from §3 above.** Regenerate or annotate
before anyone reads it as current. To update it in a new session, pass that URL
as the `url` parameter to the Artifact tool.

---

# PART II — THE TWO SKILL REPOS

Two K-Dense skill collections were installed this session. Everything is
symlinked, so `git pull` in the source checkout updates every linked skill at
once. **41 skills are currently active.**

## 9. Where things live

```
~/.agents/plugins/claude-scientific-writer/     26 skills, v2.21.0, commit 0c72606
~/.agents/plugins/scientific-agent-skills/     163 skills, v2.64.0, commit 390f514
~/.claude/skills/                              41 symlinks -> the above
```

`claude-scientific-writer` pins a **subset of** `scientific-agent-skills` — all
26 of its skills also exist in the 163. They were installed first, so they are
the ones currently linked; the 13 added from the larger repo are all names the
writer repo does not carry.

Installed manually rather than via `/plugin marketplace add` or
`npx skills add`, because those need an interactive terminal. Manual symlinks do
**not** register slash commands — notably
`/claude-scientific-writer:scientific-writer-init`, which writes a `CLAUDE.md`
of scientific-writing instructions into a project. If you want that, run the
`/plugin` install from an interactive `claude` terminal, then remove these
symlinks so nothing appears twice.

## 10. Exactly what is active

**From `claude-scientific-writer` (26):**
citation-management, clinical-decision-support, clinical-reports, docx,
generate-image, hypothesis-generation, infographics, latex-posters,
literature-review, market-research-reports, markitdown, parallel-web, pdf,
peer-review, pptx, pptx-posters, research-grants, research-lookup,
scholar-evaluation, scientific-critical-thinking, scientific-schematics,
scientific-slides, scientific-writing, treatment-plans, venue-templates, xlsx

**From `scientific-agent-skills` (13), chosen for this project:**
statistical-analysis, statistical-power, experimental-design, scikit-learn,
statsmodels, shap, scientific-visualization, matplotlib, seaborn, paper-lookup,
exa-search, markdown-mermaid-writing, arbor

**Pre-existing (2):** autoresearch, orx

Only 13 of the 163 were linked, deliberately. The larger repo's own README says
*"Do not install everything at once... consider installing a topical subset"* —
163 skills is a lot of standing context and most of it is
genomics/chemistry/clinical/astronomy, irrelevant here.

## 11. SECURITY — read before adding more

`~/.agents/plugins/scientific-agent-skills/docs/security-report.md` is the
maintainers' own LLM-based scan (cisco-ai-skill-scanner 2.0.13). Headline:
**34 CRITICAL, 8 HIGH, 147/162 rated Safe.**

**Eight CRITICAL skills are currently active:**
`citation-management`, `research-lookup`, `literature-review`,
`scientific-slides`, `scientific-schematics`, `latex-posters`, `infographics`,
`xlsx`.

**What the findings actually say.** Almost every CRITICAL is the same string:

> "Environment variable access with network calls in `scripts/*.py`" — category
> `data_exfiltration`

That is a scanner pattern-matching *reads an env var* + *makes an HTTP request*.
For these skills that is **what they are for**: `research-lookup` reads
`PARALLEL_API_KEY` and calls Parallel; `infographics` and `scientific-schematics`
read `OPENROUTER_API_KEY` and call an image API. True positive on the pattern,
false positive on intent.

`xlsx` is different — "code execution and system commands in
`scripts/recalc.py`", which is the LibreOffice recalculation shelling out. Note
`xlsx` is one of four skills **vendored from Anthropic's own repo**, so that
CRITICAL is on Anthropic-authored code.

**Assessment: not evidence of malice, but real evidence that these skills read
your environment and transmit outward.** Right now that is inert — no
`PARALLEL_API_KEY` or `OPENROUTER_API_KEY` is set, so those paths cannot fire.
The risk that matters is *upstream modification*, which is why the pinned clone
matters and why `git pull` should be a deliberate act, not a reflex.

All 13 newly added skills scan LOW / MEDIUM / SAFE (`shap` is clean).

**Recommended hardening:** unlink what you will not use. You are drafting a paper
in Overleaf, not making posters or decks. Removing `infographics`,
`latex-posters`, `scientific-slides`, `scientific-schematics`, `generate-image`,
`pptx-posters`, and the four clinical/treatment skills removes most of the
flagged surface plus a chunk of standing context, and costs nothing currently in
use.

## 12. Operating instructions

```bash
# ---- ADD one skill from either repo
ln -s ~/.agents/plugins/scientific-agent-skills/skills/<name> ~/.claude/skills/<name>
ln -s ~/.agents/plugins/claude-scientific-writer/skills/<name> ~/.claude/skills/<name>

# ---- REMOVE a skill (safe: only deletes the symlink, not the checkout)
rm ~/.claude/skills/<name>

# ---- SEE what is active and where each came from
ls -l ~/.claude/skills

# ---- BROWSE what is available but not linked
ls ~/.agents/plugins/scientific-agent-skills/skills

# ---- READ a skill before linking it (do this)
less ~/.agents/plugins/scientific-agent-skills/skills/<name>/SKILL.md

# ---- UPDATE (deliberate act; re-read the security report after)
cd ~/.agents/plugins/scientific-agent-skills && git pull && \
  head -10 docs/security-report.md

# ---- SCAN a skill yourself before trusting it
uv pip install cisco-ai-skill-scanner
skill-scanner scan ~/.agents/plugins/scientific-agent-skills/skills/<name> --use-behavioral
```

**NOTE on zsh:** `for s in $VAR` does **not** word-split in zsh the way it does in
bash. Loop over an explicit list or use an array, or you will silently link
nothing. This bit once already.

Newly linked skills are **live immediately** — no session restart needed.

## 13. Prerequisites still missing

| missing | consequence |
|---|---|
| `pdflatex` / `latexmk` | no local PDF builds. Probably fine — Overleaf compiles server-side. MacTeX is ~4GB. |
| `PARALLEL_API_KEY` | `research-lookup` and `parallel-web` are dormant |
| `OPENROUTER_API_KEY` | `generate-image`, `infographics`, `scientific-schematics` are dormant |
| `scientific-writer` pip package | skills work without it; the standalone CLI and `generate_paper()` API do not |

Four names collide with Anthropic built-ins (`docx`, `pdf`, `pptx`, `xlsx`). The
built-ins are namespaced `anthropic-skills:*`, so they coexist — but each appears
twice in the skill list.

## 14. Which skills are actually useful for THIS project

| skill | why, specifically |
|---|---|
| `paper-lookup` | 11 literature APIs, **no key needed** — go finish the 15 `[search]`-only refs in `NOVELTY_CHECK.md` |
| `scikit-learn` | `IsotonicRegression` / `CalibratedClassifierCV` for the §4.6 recalibration experiment |
| `statistical-analysis` | bootstrap CIs, effect sizes, APA reporting for the results tables |
| `statistical-power` | the `pattern` arm is ns — this says whether it is underpowered or genuinely null |
| `experimental-design` | vocabulary for defending the cross-fitting and LOSO protocols in review |
| `scientific-visualization` | the paper's figures; matplotlib/seaborn sit under it |
| `shap` | 92 logistic coefficients — a principled way to report which features carry the gain |
| `venue-templates` | has a NeurIPS 2026 wrapper; **you supply `neurips_2026.sty`** |
| `scientific-writing` | evidence-provenance drafting; offline, no key |
| `peer-review` + `scholar-evaluation` | self-review the draft before Sneheel sees it |
| `citation-management` | BibTeX generation (CRITICAL-flagged; inert without keys) |
| `arbor` | iterative optimize-against-an-evaluator with a held-out gate — close to what `autoresearch` already does |

---

## 15. Suggested order for the next session

1. **Commit.** Nothing from this session is committed.
2. Rerun `rank_crossover.py` — it is a paper claim and it is stale.
3. Rerun the three pre-fix diagnostics (fast), then regenerate the explainer
   artifact — it currently carries a retracted claim.
4. Add the pattern-stack feature to `asqa_ours_ensembles.py` and rerun. Free, and
   the one thing that could move the ASQA verdict.
5. Try the high-p recalibration (§4.6). Free.
6. Run a third and fourth `--test-seed` on `learning_curve.py` to resolve the
   +1.45-vs-+2.50 discrepancy.
7. Fix the six repo errors in §6.
8. Optionally unlink the unused flagged skills (§11).

Everything above is reproducible from `data/` with no API calls.
