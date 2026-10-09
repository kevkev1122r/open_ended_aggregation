"""
LLM judges, used for EVALUATION only (never inside the method).

JUDGE MODEL. Default google/gemini-3.8-flash: outside every agent pool and not
the synthesizer's vendor, so it cannot favour its own outputs (gpt-5.4 judging
gpt-5.4 composer answers would bias toward the LLM arms). Stronger options:
anthropic/claude-sonnet-5, google/gemini-3.1-pro-preview.

LIST AUDIT  (plan action 6: how incomplete is the gold?)
  For output items that do NOT match a QAMPARI gold alias, the judge says
  whether each is nevertheless a correct answer. Gives a corrected precision
  per arm, and the share of "wrong" items that were right but missing from gold.

LONG-FORM FACT PRECISION  (the long-form metric in the plan)
  The judge splits a candidate answer into atomic claims and labels each against
  ASQA's references (disambiguated sub-questions with short answers, and the
  human long answers): supported / contradicted / unverifiable.
  precision = supported / (supported + contradicted). Recall stays STR-EM.
"""
import json, re

import pandas as pd

from open_ended_aggregation.paths import DATA

JUDGE_MODEL = "google/gemini-3.8-flash"

LIST_JUDGE = """You are checking answers to a list question. The gold list below may be
INCOMPLETE. For each numbered candidate, decide whether it is a correct answer
to the question, using the gold list and your own knowledge.
label: "correct", "incorrect", or "unsure".

Return JSON only: {"items": [{"i": 1, "label": "correct"}, ...]}
"""

LONG_JUDGE = """You are checking the factual accuracy of an answer to an ambiguous question.
References: the question's interpretations with their short answers, and human-
written long answers. Split the CANDIDATE into atomic factual claims (skip hedges
and advice). Label each claim against the references and your own knowledge:
"supported", "contradicted", or "unverifiable".

Return JSON only: {"claims": [{"claim": "...", "label": "supported"}, ...]}
"""


def _parse(txt, key):
    m = re.search(r"\{.*\}", txt or "", re.S)
    try:
        v = json.loads(m.group(0))[key] if m else None
        return v if isinstance(v, list) else None
    except Exception:
        return None


# ------------------------------------------------------------------ lists
def list_judge_prompt(question, gold_answers, candidates):
    g = "\n".join(f"- {a}" for a in gold_answers)
    c = "\n".join(f"{i}. {x}" for i, x in enumerate(candidates, 1))
    return LIST_JUDGE + f"\nQuestion: {question}\n\nGold list (may be incomplete):\n{g}\n\nCandidates:\n{c}"


def list_judge_labels(txt, n):
    """{index (0-based): label}; missing or malformed entries are absent."""
    out = {}
    for it in _parse(txt, "items") or []:
        try:
            i = int(it.get("i")) - 1
        except (TypeError, ValueError):
            continue
        lab = str(it.get("label", "")).lower()
        if 0 <= i < n and lab in ("correct", "incorrect", "unsure"):
            out[i] = lab
    return out


# ------------------------------------------------------------------ long-form
def asqa_references():
    """{qid: {"sub": [(disambiguated question, [short answers])], "long": [..]}}"""
    df = pd.read_parquet(DATA / "asqa_dev.parquet")
    out = {}
    for i, row in df.iterrows():
        out[f"asqa:{i:04d}"] = {
            "sub": [(p["question"], list(p["short_answers"])) for p in row["qa_pairs"]],
            "long": [a["long_answer"] for a in row["annotations"] if a.get("long_answer")]}
    return out


def long_judge_prompt(question, ref, candidate):
    sub = "\n".join(f"- {q}  ->  {' / '.join(a)}" for q, a in ref["sub"])
    lng = "\n\n".join(f"[{j}] {t}" for j, t in enumerate(ref["long"], 1))
    return (LONG_JUDGE + f"\nQuestion: {question}\n\nInterpretations and short answers:\n{sub}"
            f"\n\nHuman long answers:\n{lng}\n\nCANDIDATE:\n{candidate}")


def long_judge_scores(txt):
    """(precision or None, n_supported, n_contradicted, n_unverifiable)."""
    cl = _parse(txt, "claims")
    if cl is None:
        return None, 0, 0, 0
    lab = [str(c.get("label", "")).lower() for c in cl]
    s, c, u = lab.count("supported"), lab.count("contradicted"), lab.count("unverifiable")
    return (s / (s + c) if s + c else None), s, c, u
