# Unresolved — blocks submission of `main.tex`

Generated alongside the first full draft, 23 August 2026. Nothing here is prose
work; every item is a measurement, a read, or a human decision.

**Source of record is `main.tex`** (NeurIPS 2026 template). `PAPER_DRAFT.md` is
the earlier Markdown version, kept for diffing only. `build_registries.py` reads
`main.tex`, rebuilds the registries, and writes `audit_view.md`, a generated
verbatim copy that exists only because the skill's auditors accept Markdown
suffixes. Never edit `audit_view.md`.

Audit state at time of writing:

| check | status |
|---|---|
| `validate_manifest.py source_manifest.json --kind source` | pass (24 warnings, all "no DOI/PMID") |
| `check_references.py` | pass |
| `check_consistency.py` | pass |
| `audit_claims.py audit_view.md` | **fail** — 47/47 claims mapped, but 23 rest on unverified evidence and 23 citation markers point at unread sources |
| `lint_manuscript.py audit_view.md` | **fail** — two manifest gates only a human can clear; one false positive (the word "proprietary" describing the model pool) |
| `lint_review.py SELF_REVIEW.md` | pass, 15 structured comments |
| `validate_claim_evidence.py` | valid with alignment gaps: 7 supported, 10 partly, 1 not assessed |

## 0 LaTeX and venue blockers, new with `main.tex`

1. **`neurips_2026.sty` and `checklist.tex` are not in this repository.** Both
   must come from neurips.cc. The file will not compile without them, and
   `\input{checklist.tex}` is the last line before `\end{document}`.
2. **No local LaTeX toolchain.** `pdflatex` and `latexmk` are not installed on
   this machine, so the file has **never been compiled**. It passes a brace,
   math-mode, environment, cross-reference and column-count check only. Compile
   on Overleaf before trusting it.
3. **Page limit.** NeurIPS allows nine content pages. The draft carries eleven
   tables and one figure. It is very likely over and will need trimming; the
   negative-results table and the calibration table are the first candidates for
   the appendix.
4. **The NeurIPS paper checklist is unwritten.** It does not count toward the
   page limit but is required.
5. **Track not selected.** The preamble is on the default (main track,
   double-blind) option. Confirm before submission.

---

## 1 Numbers in the draft that predate the parsing fix

Every one of these is reported in the draft with an explicit stale marker. They
must be rerun before submission; the expectation is small movement (1.12% of
candidates, concentrated in one agent) but expectation is not measurement.

| draft location | what | module to rerun |
|---|---|---|
| Table 3 (210 ensembles) | the breadth result — a headline claim | `analysis/rank_crossover.py` |
| Table 5 (learning curve) | supervision cost | `analysis/learning_curve.py` |
| Table 8 (position by list length) | the mechanism behind the ASQA loss | `analysis/rank_effect.py` |
| Table 1 (per-agent list lengths) | backs the omission feature | `analysis/omission_effect.py` |
| Table 10 (negative results) | the whole ablation sweep | `analysis/weighting_v2..v9.py` |
| §5.1 support-count profile | `P(correct \| k assert)`, k = 1..8 | regenerate from the fixed pool |
| §1.1 majority-vote and selection-degeneracy numbers | uses the older pool and pre-fix parse | regenerate on the current 777q / 8-agent protocol |
| §3.4 `Ĝ` MAE 4.62, constant-`k` 29.41 | backs "the cut must vary" | `analysis/loso.py` / `gain_predictors.py` |
| §3.6 solo ratios (0.06 / 0.24 / 0.47) | the screen, a contribution | recompute post-fix |

Already post-fix and safe to quote: Table 2, Table 4, and every ASQA and TriviaQA
number (neither routes through the QAMPARI normaliser).

## 2 Seed stability is not yet reproduced post-fix

The draft does not currently claim seed stability. Pre-fix, three fold
assignments gave 31.74 / 31.63 / 31.63. Only seed 0 has been rerun post-fix.
**Run seeds 1 and 2**, then add a stability sentence to §4.4 — a reviewer will
ask whether 31.70 is a fold-assignment artefact.

## 3 Two internal inconsistencies to resolve

1. **`BLEND + budget` under random folds disagrees between harnesses**: 32.06 in
   `v5_seed0_fixed.log`, 32.29 in `loso_fixed.log`, which is the "random 5-fold"
   column of Table 4. Same arm, same data, same protocol description. Every other
   row agrees exactly (29.21 / 29.50 / 30.68 / 31.70), so this is isolated to the
   boosted arm and is most likely GBM seeding. Diagnose before Table 4 ships.
2. **The learning curve's full-data gain (+1.45, +1.28) is well below the
   cross-fitted headline (+2.50)** on both splits. Partly explained by the
   smaller training pool (527 vs ~622) and a 250-question test set, but it is
   consistently lower. **Run a third and fourth `--test-seed`.** Table 5 flags
   this in its own caption, which is honest but is not a resolution.

## 4 Per-source LOSO gains are reported for the wrong arm

§5.4 reports the five per-source gains from `loso_fixed.log`, which breaks them
down for `BLEND + budget` — the arm the paper explicitly rejects. Rerun the
per-source breakdown for `LEAN + budget` so the table matches the method as
reported. The draft states this gap in the text; that is a stopgap.

## 5 Fifteen references have never been read

52 in-text citation markers point at sources marked ‡. Search-result summaries
are not a citable source, and the writing guide treats a mis-supported citation
as grounds for rejection. `paper-lookup` needs no API key and can close these.

Highest priority, because a claim in the draft leans on them:

- **E017 Dawid & Skene (1979)** — the draft presents the pattern lookup as
  "in substance a Dawid–Skene annotator model". That is a statement about the
  literature and it is currently unverified. Bibliographic details also
  unconfirmed.
- **E013 QAMPARI** — we use the dataset; the citation must be right.
- **E015 / E016 verbosity** — these bound how much of the verbosity finding we
  are allowed to claim as ours.
- **E031 RRF** — cited as a baseline we ran.
- **E003 / E032 / E014** — the majority-voting lineage in §6.1.

## 6 The calibration table has no reproducible artifact

Table 9 was measured inline during a session and never written to a module or a
`results/` entry. It is currently the only table in the paper whose provenance is
a handoff document. **Write `analysis/calibration.py`** and regenerate, or cut
the table. Everything else in the paper reproduces from `data/` with no API
calls, and that property is worth keeping intact.

## 7 Experiments that would change what the paper can claim

1. **Add the pattern-probability stack feature to `asqa_ours_ensembles.py`.**
   On QAMPARI it is an explicit feature block; on ASQA it is missing, yet
   `pattern` is the strongest non-weighted ASQA arm. Free, no generation, and it
   is the one change that could move the ASQA verdict in Table 7. Do this before
   the ASQA table is considered final.
2. **High-`p` recalibration** (§5.6). Free. Cannot threaten the scorer's 30.68 —
   ordering is calibration-invariant — but could improve the +1.02 the budget
   adds.
3. **A fourth dataset for the solo-ratio screen.** Three points support an
   ordering, not a law, and the draft says so. A fourth would upgrade
   contribution 6 from an observation to a validated screen.
4. **Multi-sample generation on QAMPARI or ASQA**, so self-consistency — the
   signal that fills the slot on TriviaQA — becomes available on a set-valued
   task. This is the single experiment that most directly tests the two-slot
   claim, because it ports a signal *across* answer formats. Needs generation:
   777 × m calls, inside the $30 cap at m = 20 with a cheap model.

## 8 Baselines still blocked

- **Paper-exact single-model ASC** needs generation (777 × m calls on one model).
  The draft discloses the agents-for-samples substitution in §4.3 and again in
  Limitations. Note that one prior blocker has cleared: ASC's list pipeline has
  no composition step, so `--no-compose` *is* the paper's list configuration.
- **A faithful ASQA DR** needs the RoBERTa-SQuADv2 reader. The draft labels
  `DR*` a proxy everywhere it appears.
- **`az login` is interactive**, still blocking the OW paper's small model pool.

## 9 Missing artifacts and decisions

- **Figure 1.** The draft carries a Mermaid pipeline sketch as a build note. The
  writing guide wants a real first-page diagram (draw.io / vecta.io suggested):
  one question, a few agents' ordered lists, the pooled candidate set, the two
  signals highlighted, then the per-question cut.
- **Conference target undecided.** Fixes the template, the page limit, whether
  the Limitations section is required (it is at *ACL venues), and whether Related
  Work stays second-to-last.
- **Appendix content is listed but not written** (A–G in the draft).
- **Author list, affiliations, and the anonymised code link**
  (`anonymous.4open.science`). No acknowledgements section — the guide notes a
  prior desk rejection for a non-anonymised link.
- **Title not chosen.** Three candidates are at the top of the draft.

## 10 Human gates in the manuscript manifest

`lint_manuscript.py` fails until an accountable author sets these, and only an
author may:

- `confidentiality_review.completed` / `.policy_checked`
- `human_verification.completed`
- the eight `required_statements` (code and data availability, funding,
  conflicts, AI disclosure, author contributions)
- `submission_ready`, and removal of the draft banner

## 11 Items from the self-review not already listed above

Full text in `SELF_REVIEW.md`; three items it raised were prose-level and are
already applied to the draft (the bound-not-null rewrite of §5.1, the forward
reference to the finite-sample caveat, and the removal of the project-history
sentence in §5.2).

| id | item | where it lands |
|---|---|---|
| M2 | Report support-pattern cell occupancy — occupied cells and claims per occupied cell — for QAMPARI and ASQA, so the "ASQA is data-limited" attribution in §5.5 is checkable rather than asserted | new measurement, cheap, from cache |
| M4 | Per-source LOSO gains are reported for the rejected boosted arm, not the method | §4 above |
| m2 | The TriviaQA agent roster is never given, while QAMPARI's and ASQA's are | §4.2 of the draft |
| m3 | Gains appear in both absolute F1 and percentage form; pick absolute as primary | editorial pass |
| m5 | Figure 1 does not exist; `main.tex` holds a sized `\fbox` and a caption | §9 above |
| m7 | `DR*` is redefined three times | editorial pass |

Resolved in `main.tex`: **M1** (§5.1 now states a bound rather than a null),
**m4** (the two ensemble tables have separate captions, Tables 3 and 4),
**m6** (the project-history sentence is gone).

## 12 Process requirements from the writing guide

- Mentor review first, **then** a P.I. review booked **at least one week before
  the deadline**. Papers are not submitted without that review.
- On Overleaf, do not resolve reviewer comments — reply "done" and leave them
  visible.
- Final checklist: line numbers on, anonymous submission, self-contained captions
  with a stated takeaway, every claim carrying a citation that actually supports
  it.
