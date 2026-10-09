#!/usr/bin/env python3
"""Generate the evidence registries for PAPER_DRAFT.md.

Claim text is taken as the prose immediately preceding each inline
`<!-- [claim:Cxxx] [evidence:Exxx,...] -->` marker, so the recorded hash always
matches what is actually written. Re-run after editing the draft.

    python3 build_registries.py
"""
import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The manuscript has been called main.tex and main1.tex at different times,
# because Overleaf projects already ship a main.tex. Take whichever is present.
# The manuscript keeps getting renamed (main.tex, main1.tex, iteration3.tex...),
# so find it by content instead: the newest .tex carrying claim markers.
_cands = sorted((f for f in HERE.glob("*.tex")
                 if "[claim:C" in f.read_text(errors="ignore")),
                key=lambda f: f.stat().st_mtime, reverse=True)
DRAFT = _cands[0] if _cands else HERE / "PAPER_DRAFT.md"

# Markers are LaTeX comments in main.tex and HTML comments in PAPER_DRAFT.md.
# Both forms are matched so either file can be the source of record.
MARKER = re.compile(
    r"(?:<!--|%)\s*\[claim:(C\d{3})\]\s*\[evidence:((?:E\d{3})(?:\s*,\s*E\d{3})*)\]"
)
HEADING = re.compile(
    r"^(?:#{1,4}\s+(.*)$|\\(?:sub)*section\*?\{([^}]*)\})"
)

# ---------------------------------------------------------------- sources ----
# "verified" means: the artifact exists AND the number it supports reflects the
# current post-parsing-fix pipeline. Pre-fix artifacts are recorded as
# unverified even though the file exists, because the claim they back is stale.
SOURCES = [
    # -- literature, read directly ------------------------------------------
    ("E001", "Atomic Self-Consistency for Better Long Form Generations", 2024,
     "arXiv:2405.13131", "method and list-pipeline configuration", True),
    ("E002", "Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information",
     2025, "arXiv:2510.01499", "weight derivation; target prediction type", True),
    ("E003", "More Agents Is All You Need", 2024, "arXiv:2402.05120",
     "sampling-and-voting scaling result", False),
    ("E004", "LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion",
     2023, "arXiv:2306.02561", "pairwise ranking and fusion over whole responses", True),
    ("E005", "Serial Position Effects of Large Language Models", 2024,
     "arXiv:2406.15981", "Figure 1; input-side label-selection framing", True),
    ("E006", "Zero-knowledge LLM hallucination detection through fine-grained cross-model consistency (Finch-Zk)",
     2025, "arXiv:2508.14314", "segmentation and cross-model scoring; evaluation suites", True),
    ("E007", "Atomic Calibration of LLMs in Long-Form Generations", 2024,
     "arXiv:2410.13246", "single-model atomic-level calibration", True),
    ("E008", "Is This Your Final Answer? Cross-Contextual Consistency as a Measure of LLM Credibility",
     2026, "arXiv:2608.10315", "perturbation consistency as credibility", True),
    ("E009", "Power and Limitations of Aggregation in Compound AI Systems", 2026,
     "arXiv:2602.21556", "bound on member-answer selection policies", True),
    ("E010", "Harnessing Collective Intelligence for Biomedical Question Answering",
     2025, "arXiv:2508.01480", "union aggregation for list questions", True),
    ("E011", "The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers",
     2025, "arXiv:2510.11218", "SLAQ; 51.3% at first fact to 30.1% at fifth, 16 models", True),
    ("E012", "ASQA: Factoid Questions Meet Long-Form Answers", 2022,
     "arXiv:2204.06092", "DR definition; statement that STR-EM and Disambig-F1 measure the same aspect", True),
    # -- literature, search-summary only ------------------------------------
    ("E013", "QAMPARI", 2022, "arXiv:2205.12665", "dataset description", False),
    ("E014", "Universal Self-Consistency", 2023, "arXiv:2311.17311", "response selection", False),
    ("E015", "The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs",
     2025, "arXiv:2508.08285", "response length as hallucination detector", False),
    ("E016", "Verbosity is not Veracity", 2024, "arXiv:2411.07858", "verbosity compensation", False),
    ("E017", "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm",
     1979, "", "annotator confusion-matrix model; bibliographic details unconfirmed", False),
    ("E018", "Fast Dawid-Skene", 2018, "arXiv:1803.02781", "aggregation algorithm", False),
    ("E019", "Crowd-Certain: Label Aggregation in Crowdsourced and Ensemble Learning",
     2023, "arXiv:2310.16293", "label aggregation", False),
    ("E020", "Improving Deep Ensembles by Estimating Confusion Matrices", 2025,
     "arXiv:2503.07119", "soft Dawid-Skene for deep ensembles", False),
    ("E021", "Prompt-Reverse Inconsistency", 2025, "arXiv:2504.01282", "prompt-transform consistency", False),
    ("E022", "Self-consistency under prompt transformations", 2023, "arXiv:2305.14279",
     "prompt-transform consistency", False),
    ("E023", "MAD-Fact", 2025, "arXiv:2510.22967", "multi-agent long-form factuality evaluation", False),
    ("E024", "FaStfact", 2025, "arXiv:2510.12839", "atomic-claim factuality evaluation", False),
    ("E025", "VeriFact", 2025, "arXiv:2505.09701", "atomic-claim factuality evaluation", False),
    ("E026", "The Position Curse: LLMs Struggle to Locate the Last Few Items in a List",
     2026, "arXiv:2605.07127", "input-side item location", False),
    ("E027", "Lost in the Middle follow-up", 2025, "arXiv:2510.10276", "context position", False),
    ("E028", "LLM-based Listwise Reranking under Positional Bias", 2026,
     "arXiv:2604.03642", "positional bias when reranking supplied candidates", False),
    ("E029", "EviRank", 2026, "arXiv:2606.04727", "position-aware confidence for ranking", False),
    ("E030", "Adaptive Coverage Policies in Conformal Prediction", 2025,
     "arXiv:2510.04318", "per-sample miscoverage and expected set size", False),
    ("E031", "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods",
     2009, "", "RRF definition; bibliographic details unconfirmed", False),
    ("E032", "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
     2022, "arXiv:2203.11171", "majority voting over sampled reasoning paths", False),
    # -- internal artifacts, current --------------------------------------
    ("E100", "Run log: weighting_v5 seed 0, post parsing fix", 2026,
     "v5_seed0_fixed.log", "QAMPARI headline table, all arms, bootstrap CIs", True),
    ("E101", "Run log: leave-one-source-out, post parsing fix", 2026,
     "loso_fixed.log", "LOSO vs random-fold table; per-source gains", True),
    ("E103", "Run log: ASQA rank/budget arms", 2026, "asqa_rank2.log",
     "ASQA DR* arm table incl. adapt-* and learned-budget arms", True),
    ("E104", "Result file: ASQA per-ensemble sweep", 2026,
     "results/asqa_ours_ensembles.json", "all 42 ASQA ensembles; agent roster", True),
    ("E106", "Run log: ASQA rank effect", 2026, "asqa_rank_effect.log",
     "sentence-position correlation by response length", True),
    ("E113", "Source: analysis/weighting_v3.py, build_features", 2026,
     "open_ended_aggregation/analysis/weighting_v3.py:97", "feature block definitions and sizes", True),
    ("E114", "Source: analysis/weighting_v5.py", 2026,
     "open_ended_aggregation/analysis/weighting_v5.py", "cross-fitting protocol; ridge=1.0; arm definitions", True),
    ("E115", "Config: model pools", 2026, "configs/pools.yaml", "QAMPARI 8-agent roster", True),
    ("E116", "Source: analysis/asqa_metrics.py", 2026,
     "open_ended_aggregation/analysis/asqa_metrics.py", "DR* proxy definition and its justification", True),
    ("E117", "Source: benchmarks/qampari.py, strip_special", 2026,
     "open_ended_aggregation/benchmarks/qampari.py", "special-token normalisation fix; candidate counts", True),
    ("E119", "Doc: benchmarks.md", 2026, "docs/benchmarks.md",
     "QAMPARI alias coverage and the strict-vs-relaxed sensitivity analysis", True),
    ("E120", "ASC EMNLP paper, section 'Adapting ASC to list style datasets'", 2024,
     "arXiv:2405.13131", "list pipeline has no composition step", True),
    ("E123", "Run logs: LLM-judged clustering", 2026, "asc_llmcluster*.log",
     "qualitative merge examples and 7.6% merge rate", True),
    ("E124", "Result file: support-pattern cell occupancy", 2026,
     "results/pattern_cells.json, from analysis/pattern_cells.py",
     "occupied cells, claims per cell, and the correctness-rate spread "
     "within each support count; post parsing fix", True),
    ("E125", "Run log: ASQA full-pool arms with the vs-ceiling column", 2026,
     "asqa_v5.log", "best single 27.03, count 24.77, pattern 27.20, weighted "
     "27.45, LEAN 26.81/26.88; all comparisons against best single and ceiling", True),
    ("E126", "Result file: ASQA per-ensemble sweep, counting vs best single", 2026,
     "results/asqa_ensembles.json", "counting below best single in 39 of 42 "
     "ensembles; mean gap -0.70 at n=3 widening to -2.26 at n=6", True),
    ("E127", "Result file: sensitivity at the achieved sample size", 2026,
     "results/power_check.json, from analysis/power_check.py",
     "MDE 0.92 at 80% power and 1.13 for the method arm; TOST equivalence "
     "within +-0.83 F1; bootstrap SE 0.328; post parsing fix", True),
    ("E128", "Run log: ASQA per-ensemble sweep with the stacked pattern "
     "probability added as a feature", 2026, "asqa_stack.log",
     "stacking is worth +0.01 DR* mean over 42 ensembles, significant in 5 and "
     "negative in 1; the scorer remains below the pattern lookup", True),
    # -- internal artifacts, stale or unbacked ------------------------------
    ("E102", "Run log: rank_crossover, 210 ensembles (PRE parsing fix)", 2026,
     "rank_crossover.log", "per-size gains and significant-win counts; PENDING RERUN", False),
    ("E105", "Run log: rank effect (PRE parsing fix)", 2026, "rank_effect.log",
     "length-stratified position analysis; PENDING RERUN", False),
    ("E107", "Run log: omission effect (PRE parsing fix)", 2026, "omission_effect.log",
     "per-agent list-length distribution; PENDING RERUN", False),
    ("E108", "Run logs: learning curve, two test splits (PRE parsing fix)", 2026,
     "learning_curve.log, learning_curve_seed1.log", "gain vs training-set size; PENDING RERUN", False),
    ("E110", "Calibration measurement, run inline with no saved artifact", 2026,
     "HANDOFF_2026-08-23_evening.md section 4.6", "predicted-vs-actual bins; NO REPRODUCIBLE MODULE", False),
    ("E111", "Doc: weighting sweep, ~60 rules (PRE parsing fix)", 2026,
     "docs/handoffs/WEIGHTING_SWEEP_2026-08-21.md", "ablation and negative-result tables; PENDING RERUN", False),
    ("E112", "Doc: ROBUSTNESS.md (PRE parsing fix)", 2026, "docs/ROBUSTNESS.md",
     "gold-count statistics, Ghat MAE, constant-k control; PENDING RERUN", False),
    ("E118", "Solo-ratio measurements, 20 August 2026 (PRE parsing fix)", 2026,
     "docs/handoffs/WEIGHTING_SWEEP_2026-08-21.md section 4", "solo ratios and pool CVs; PENDING RERUN", False),
    ("E121", "Doc: results.md (PRE parsing fix, earlier agent pool)", 2026,
     "docs/results.md", "majority-vote and response-selection degeneracy; PENDING REGENERATION", False),
    ("E122", "Support-count profile (PRE parsing fix)", 2026,
     "docs/handoffs/HANDOFF_2026-08-23_evening.md section 2", "P(correct | k agents assert); PENDING RERUN", False),
]

# ------------------------------------------------------------------ claims ----
# claim_id -> (kind, uncertainty, analysis_intent)
CLAIM_META = {
    "C001": ("numeric", "moderate", "confirmatory"),
    "C002": ("numeric", "moderate", "confirmatory"),
    "C003": ("numeric", "not_estimated", "descriptive"),
    "C004": ("numeric", "low", "descriptive"),
    "C005": ("numeric", "not_estimated", "descriptive"),
    "C006": ("result", "moderate", "confirmatory"),
    "C007": ("result", "moderate", "confirmatory"),
    "C008": ("method", "not_applicable", "not_applicable"),
    "C009": ("interpretive", "high", "exploratory"),
    "C010": ("method", "not_applicable", "not_applicable"),
    "C011": ("numeric", "not_estimated", "descriptive"),
    "C012": ("method", "not_applicable", "not_applicable"),
    "C013": ("numeric", "not_estimated", "descriptive"),
    "C014": ("numeric", "not_estimated", "confirmatory"),
    "C015": ("numeric", "not_estimated", "descriptive"),
    "C016": ("numeric", "not_estimated", "descriptive"),
    "C017": ("numeric", "moderate", "descriptive"),
    "C018": ("factual", "not_applicable", "descriptive"),
    "C019": ("factual", "not_applicable", "descriptive"),
    "C020": ("numeric", "not_estimated", "descriptive"),
    "C021": ("factual", "not_applicable", "not_applicable"),
    "C022": ("numeric", "low", "descriptive"),
    "C023": ("numeric", "low", "descriptive"),
    "C024": ("result", "moderate", "confirmatory"),
    "C025": ("numeric", "moderate", "exploratory"),
    "C026": ("numeric", "not_estimated", "descriptive"),
    "C027": ("result", "moderate", "confirmatory"),
    "C028": ("result", "moderate", "exploratory"),
    "C029": ("result", "moderate", "exploratory"),
    "C030": ("result", "moderate", "exploratory"),
    "C031": ("result", "moderate", "confirmatory"),
    "C032": ("interpretive", "moderate", "exploratory"),
    "C033": ("result", "moderate", "confirmatory"),
    "C034": ("numeric", "moderate", "exploratory"),
    "C035": ("interpretive", "high", "exploratory"),
    "C036": ("result", "high", "exploratory"),
    "C037": ("numeric", "high", "exploratory"),
    "C038": ("result", "moderate", "confirmatory"),
    "C039": ("numeric", "moderate", "exploratory"),
    "C040": ("numeric", "moderate", "confirmatory"),
    "C041": ("numeric", "moderate", "exploratory"),
    "C042": ("interpretive", "high", "exploratory"),
    "C043": ("numeric", "moderate", "descriptive"),
    "C044": ("result", "moderate", "exploratory"),
    "C045": ("numeric", "not_estimated", "descriptive"),
    "C046": ("numeric", "not_estimated", "descriptive"),
    "C047": ("numeric", "high", "descriptive"),
    "C048": ("result", "low", "confirmatory"),
    "C049": ("result", "moderate", "confirmatory"),
    "C050": ("interpretive", "moderate", "exploratory"),
    "C051": ("numeric", "low", "confirmatory"),
    "C048": ("numeric", "low", "descriptive"),
}

# ------------------------------------------------- repeated numeric facts ----
NUMERIC_FACTS = [
    ("N001", "QAMPARI method set F1 (scorer + budget), 5-fold cross-fitted",
     "5 Results", 31.70, "set F1 x100", 777, "all QAMPARI questions", ["E100"]),
    ("N002", "QAMPARI multi-agent counting set F1", "5 Results", 29.21,
     "set F1 x100", 777, "all QAMPARI questions", ["E100"]),
    ("N003", "QAMPARI best single agent set F1 (Kimi-K2.5)", "5 Results", 27.40,
     "set F1 x100", 777, "all QAMPARI questions", ["E100"]),
    ("N004", "QAMPARI pattern-lookup ceiling set F1", "5 Results", 29.50,
     "set F1 x100", 777, "all QAMPARI questions", ["E100"]),
    ("N005", "QAMPARI method gain over counting", "5 Results", 2.50,
     "set F1 x100", 777, "all QAMPARI questions", ["E100"]),
    ("N006", "QAMPARI method gain over best single agent", "5 Results", 4.30,
     "set F1 x100", 777, "all QAMPARI questions", ["E100"]),
    ("N007", "QAMPARI method set F1 under leave-one-source-out", "5 Results",
     31.49, "set F1 x100", 777, "all QAMPARI questions", ["E101"]),
    ("N008", "QAMPARI candidate claims after normalisation fix", "5 Results",
     110136, "claims", 777, "all QAMPARI questions", ["E100", "E117"]),
    ("N009", "QAMPARI candidate claims asserted by exactly one agent",
     "5 Results", 95772, "claims", 777, "all QAMPARI questions",
     ["E100", "E124"]),
    ("N010", "Features used by the scorer", "3 Method", 92, "features", None,
     "not applicable", ["E113", "E114"]),
    ("N011", "ASQA reliability weighting DR* at ensemble size 6", "5 Results",
     27.01, "DR* x100", 400, "all ASQA questions", ["E104"]),
    ("N012", "ASQA method DR* at ensemble size 6", "5 Results", 26.35,
     "DR* x100", 400, "all ASQA questions", ["E104"]),
]

METHODS = [
    ("M001", "5-fold cross-fitted aggregation with paired bootstrap over questions",
     "confirmatory", "prespecified", ["O001", "O002"]),
    ("M002", "Leave-one-source-out cross-fitting over QAMPARI question sources",
     "confirmatory", "prespecified", ["O001"]),
    ("M003", "Exhaustive per-ensemble sweep over agent subsets of size 3-6",
     "confirmatory", "prespecified", ["O001", "O002"]),
    ("M004", "Length-stratified correlation of within-output position with correctness",
     "exploratory", "post_hoc", ["O003"]),
]

RESULTS = [
    ("R001", "M001", "O001", "confirmatory", 777, ["E100"], ["5 Results"]),
    ("R002", "M002", "O001", "confirmatory", 777, ["E101"], ["5 Results"]),
    ("R003", "M003", "O001", "confirmatory", 777, ["E102"], ["5 Results"]),
    ("R004", "M003", "O002", "confirmatory", 400, ["E104"], ["5 Results"]),
    ("R005", "M004", "O003", "exploratory", 777, ["E105"], ["5 Results"]),
    ("R006", "M004", "O003", "exploratory", 400, ["E106"], ["5 Results"]),
    ("R007", "M001", "O002", "confirmatory", 400, ["E103"], ["5 Results"]),
]


def main() -> None:
    text = DRAFT.read_text(encoding="utf-8")

    # The skill's auditors accept Markdown suffixes only. Emit a verbatim
    # generated view of main.tex so audit_claims.py and lint_manuscript.py can
    # read it. Never edit this file; it is overwritten on every run.
    (HERE / "audit_view.md").write_text(
        "<!-- GENERATED FROM main.tex BY build_registries.py. DO NOT EDIT. -->\n"
        + text,
        encoding="utf-8")

    verified = {s[0]: s[5] for s in SOURCES}

    sources = []
    for eid, title, year, ident, locator, ok in SOURCES:
        is_arxiv = ident.startswith("arXiv:")
        sources.append({
            "authors": [],
            "confidentiality": "public",
            "evidence_id": eid,
            "identifiers": {
                "doi": "",
                "isbn": "",
                "pmcid": "",
                "pmid": "",
                "url": f"https://arxiv.org/abs/{ident.split(':', 1)[1]}" if is_arxiv else "",
            },
            "locator": locator,
            "source_type": "preprint" if is_arxiv else "other",
            "title": title,
            "verification": {
                "source_opened": ok,
                "status": "verified" if ok else "unverified",
                "verified_by": "project author" if ok else "",
                "verified_on": "2026-08-23" if ok else "",
            },
            "year": year,
        })
    (HERE / "source_manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "sources": sources}, indent=2) + "\n",
        encoding="utf-8")

    # ---- claims, hashed from the prose that precedes each marker ----------
    section = "front matter"
    rows, seen = [], []
    prev_prose = ""
    lines = text.splitlines()
    for raw in lines:
        head = HEADING.match(raw.strip())
        if head:
            section = (head.group(1) or head.group(2) or "").strip()
        stripped = raw.strip()
        has_marker = MARKER.search(raw)
        for m in MARKER.finditer(raw):
            cid = m.group(1)
            eids = [e.strip() for e in m.group(2).split(",")]
            # Inline marker: claim text is what precedes it on the line.
            # Own-line marker (the LaTeX form): use the last prose line seen.
            claim_text = " ".join(raw[:m.start()].split()).lstrip("% ").strip()
            if not claim_text:
                claim_text = prev_prose
            if not claim_text:
                claim_text = f"(table or display claim in section: {section})"
            digest = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
            kind, unc, intent = CLAIM_META.get(cid, ("factual", "not_estimated", "descriptive"))
            status = "verified" if all(verified.get(e) for e in eids) else "unverified"
            rows.append({
                "claim_id": cid,
                "section": section,
                "claim_kind": kind,
                "claim_text_sha256": digest,
                "evidence_ids": ";".join(eids),
                "verification_status": status,
                "uncertainty": unc,
                "analysis_intent": intent,
            })
            seen.append(cid)
        if stripped and not has_marker and not stripped.startswith(("%", "\\", "#")):
            prev_prose = " ".join(stripped.split())

    fields = ["claim_id", "section", "claim_kind", "claim_text_sha256",
              "evidence_ids", "verification_status", "uncertainty", "analysis_intent"]
    with (HERE / "claims.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- consistency registry --------------------------------------------
    consistency = {
        "schema_version": "1.0",
        "numeric_facts": [
            {
                "analysis_set": aset, "concept": concept, "denominator": None,
                "evidence_ids": eids, "fact_id": fid, "numerator": None,
                "sample_size": nsize, "section": sect, "unit": unit, "value": val,
            }
            for fid, concept, sect, val, unit, nsize, aset, eids in NUMERIC_FACTS
        ],
        "methods": [
            {"analysis_intent": intent, "method_id": mid, "name": name,
             "outcome_ids": outs, "protocol_status": proto}
            for mid, name, intent, proto, outs in METHODS
        ],
        "results": [
            {"analysis_intent": intent, "evidence_ids": eids, "method_id": mid,
             "outcome_id": oid, "reported_sections": sects, "result_id": rid,
             "sample_size": nsize}
            for rid, mid, oid, intent, nsize, eids, sects in RESULTS
        ],
    }
    (HERE / "consistency_manifest.json").write_text(
        json.dumps(consistency, indent=2) + "\n", encoding="utf-8")

    manuscript = {
        "schema_version": "1.0",
        "document_id": "open-ended-aggregation-paper",
        "draft_status": "draft",
        "study_design": "computational_benchmark_study",
        "submission_ready": False,
        "confidentiality_review": {
            "completed": False,
            "external_services_authorized": False,
            "policy_checked": False,
            "verified_by": "",
            "verified_on": "",
        },
        "human_verification": {"completed": False, "verified_by": "", "verified_on": ""},
        "registries": {
            "authorship_manifest": "authorship.json",
            "claim_evidence": "claims.csv",
            "consistency_manifest": "consistency_manifest.json",
            "reporting_coverage": "reporting_coverage.json",
            "source_manifest": "source_manifest.json",
        },
        "reporting_guidelines": [],
        "required_statements": {
            "ai_disclosure": "missing",
            "author_contributions": "missing",
            "code_availability": "missing",
            "conflicts": "missing",
            "consent": "not_applicable",
            "data_availability": "missing",
            "ethics": "not_applicable",
            "funding": "missing",
        },
    }
    (HERE / "manuscript_manifest.json").write_text(
        json.dumps(manuscript, indent=2) + "\n", encoding="utf-8")

    missing = sorted(set(CLAIM_META) - set(seen))
    unver = [r["claim_id"] for r in rows if r["verification_status"] == "unverified"]
    print(f"claims found in draft : {len(rows)}")
    print(f"declared but not found: {missing or 'none'}")
    print(f"sources               : {len(sources)} "
          f"({sum(verified.values())} verified, {len(sources) - sum(verified.values())} unverified)")
    print(f"claims resting on unverified evidence ({len(unver)}): {', '.join(unver)}")


if __name__ == "__main__":
    main()
