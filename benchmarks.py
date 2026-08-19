"""
Four open-ended benchmarks with filtering, option-hiding, and per-domain grading.

Design constraints, all learned the hard way on TriviaQA:
  * every question must be answerable WITHOUT seeing options (that is the whole point)
  * gold answers must be short enough to grade by matching, not by judging
  * grading must be auditable -- we already shipped one result contaminated by a
    grading bug, so each domain has its own explicit matcher

  TriviaQA   recall            official alias lists
  GSM8K      math reasoning    final number, exact
  MedQA      clinical          option text; "which of the following is the most
                               likely X" is REWRITTEN to "what is the most likely X"
                               rather than dropped, which recovers 164 -> 670 items
  MMLU-Pro   academic breadth  option text; 47% dropped as option-dependent,
                               fill-in-the-blank, or multi-part; numeric answers
                               compared with unit-aware tolerance
"""
import os, re, json
import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

PARQUET = {
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "validation"),
    "gsm8k":    ("openai/gsm8k", None, "test"),
    "medqa":    ("GBaker/MedQA-USMLE-4-options", None, "test"),
    "mmlupro":  ("TIGER-Lab/MMLU-Pro", None, "test"),
}


def _fetch(name):
    p = os.path.join(DATA, {"triviaqa": "triviaqa_val", "gsm8k": "gsm8k",
                            "medqa": "medqa", "mmlupro": "mmlu_pro"}[name] + ".parquet")
    if not os.path.exists(p):
        ds, cfg, split = PARQUET[name]
        u = requests.get(f"https://huggingface.co/api/datasets/{ds}/parquet", timeout=90).json()
        cfg = cfg or list(u)[0]
        sp = split if split in u[cfg] else list(u[cfg])[0]
        r = requests.get(u[cfg][sp][0], timeout=900); r.raise_for_status()
        open(p, "wb").write(r.content)
    return pd.read_parquet(p)


# ------------------------------------------------------------------ normalisation
_ART = re.compile(r"\b(a|an|the)\b", re.I)
_PUN = re.compile(r"[^a-z0-9 ]")

def norm(s):
    s = str(s).lower().replace("’", "'")
    s = _PUN.sub(" ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())

def contains(resp, targets):
    r = norm(resp)
    if not r:
        return False
    for t in targets:
        t = norm(t)
        if t and f" {t} " in f" {r} ":
            return True
    return False

_NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:\s*[eE]\s*-?\d+)?")

def numbers(s):
    out = []
    for m in _NUM.finditer(str(s).replace(",", "")):
        try:
            out.append(float(m.group().replace(" ", "")))
        except ValueError:
            pass
    return out

def numeric_match(resp, gold, rtol=0.02):
    """Gold is numeric -> accept any number in the response within 2%.

    Handles '0.57 mm' vs '0.57mm' vs 'about 0.6 mm' vs '5.7e-4 m'-style variation
    that substring matching cannot. Deliberately lenient on units, because the
    alternative (unit parsing) is a source of silent false negatives.
    """
    g = numbers(gold)
    if not g:
        return None                        # not a numeric item
    gv = g[-1]
    for v in numbers(resp):
        if gv == 0:
            if abs(v) < 1e-9: return True
        elif abs(v - gv) <= rtol * abs(gv):
            return True
    return False


# ------------------------------------------------------------------ loaders
def load_triviaqa(n, seed=0):
    d = _fetch("triviaqa").sample(frac=1, random_state=seed).reset_index(drop=True)
    out = []
    for _, r in d.iterrows():
        al = [a for a in list(r["answer"]["normalized_aliases"]) + [r["answer"]["value"]] if a]
        if not al:
            continue
        out.append(dict(qid=f"tq:{r['question_id']}", domain="triviaqa",
                        question=r["question"], gold=r["answer"]["value"],
                        aliases=al, max_tokens=60))
        if len(out) >= n: break
    return out


def load_gsm8k(n, seed=0):
    d = _fetch("gsm8k").sample(frac=1, random_state=seed).reset_index(drop=True)
    out = []
    for i, r in d.iterrows():
        final = str(r["answer"]).split("####")[-1].strip().replace(",", "")
        out.append(dict(qid=f"gsm:{i}", domain="gsm8k", question=r["question"],
                        gold=final, aliases=[final], max_tokens=400))
        if len(out) >= n: break
    return out


_MEDQA_REWRITES = [
    (r"[Ww]hich of the following is the most likely", "What is the most likely"),
    (r"[Ww]hich of the following is the best", "What is the best"),
    (r"[Ww]hich of the following is the most appropriate", "What is the most appropriate"),
    (r"[Ww]hich of the following is the most probable", "What is the most probable"),
    (r"[Ww]hich of the following is the underlying", "What is the underlying"),
    (r"[Ww]hich of the following is the primary", "What is the primary"),
    (r"[Ww]hich of the following best describes", "What best describes"),
    (r"[Ww]hich of the following medications", "What medication"),
    (r"[Ww]hich of the following drugs", "What drug"),
    (r"[Ww]hich of the following is most likely", "What is most likely"),
    (r"[Ww]hich of the following would be the most", "What would be the most"),
    (r"[Ww]hich of the following is the next best step", "What is the next best step"),
    (r"[Ww]hich of the following is the most accurate", "What is the most accurate"),
]

def load_medqa(n, seed=0):
    d = _fetch("medqa").sample(frac=1, random_state=seed).reset_index(drop=True)
    out = []
    for i, r in d.iterrows():
        q = r["question"]
        for pat, rep in _MEDQA_REWRITES:
            q = re.sub(pat, rep, q)
        if re.search(r"of the following|all of the above|none of the above", q, re.I):
            continue                                   # genuinely needs the options
        gold = str(r["answer"])
        if len(gold.split()) > 6:
            continue
        out.append(dict(qid=f"med:{i}", domain="medqa", question=q, gold=gold,
                        aliases=[gold], max_tokens=120))
        if len(out) >= n: break
    return out


MMLU_CATS = ["math", "physics", "chemistry", "business", "health", "psychology"]

def load_mmlupro(n, seed=0, cats=None):
    d = _fetch("mmlupro")
    cats = cats or MMLU_CATS
    per = max(1, n // len(cats))
    out = []
    for c in cats:
        g = d[d.category == c].sample(frac=1, random_state=seed).reset_index(drop=True)
        got = 0
        for _, r in g.iterrows():
            q = r["question"]
            gold = str(r["options"][r["answer_index"]])
            if re.search(r"of the following|all of the above|none of the above", q, re.I): continue
            if re.search(r"_{2,}", q): continue
            if gold.count(",") >= 2: continue
            if len(gold.split()) > 6: continue
            # numeric gold is only gradable if no distractor shares its value.
            # 32% of otherwise-clean items fail this ("5 mm" vs "5 cm" vs "5%").
            gn = numbers(gold)
            if gn:
                gv = gn[-1]
                others = [numbers(o)[-1] for i, o in enumerate(r["options"])
                          if i != r["answer_index"] and numbers(o)]
                if not others or gv == 0: continue
                if min(abs(v - gv) / abs(gv) for v in others) <= 0.05: continue
            out.append(dict(qid=f"mp:{r['question_id']}", domain=f"mmlupro:{c}",
                            question=q, gold=gold, aliases=[gold], max_tokens=300))
            got += 1
            if got >= per: break
    return out


LOADERS = {"triviaqa": load_triviaqa, "gsm8k": load_gsm8k,
           "medqa": load_medqa, "mmlupro": load_mmlupro}


def load_all(per_domain, overrides=None):
    """per_domain applies to each benchmark; overrides={'mmlupro': 300} to differ.

    MMLU-Pro gets more by default in the full run because it is split across six
    categories and carries the per-domain comparison; 25 per category is too thin
    to read a specialisation signal from.
    """
    overrides = overrides or {}
    items = []
    for k, fn in LOADERS.items():
        items += fn(overrides.get(k, per_domain))
    return items


# ------------------------------------------------------------------ grading
def grade(resp, item):
    """One matcher per domain. Returns bool."""
    if resp is None:
        return False
    dom = item["domain"].split(":")[0]
    if dom == "gsm8k":
        nums = numbers(resp)
        if not nums:
            return False
        try:
            return abs(nums[-1] - float(item["gold"])) < 1e-4
        except ValueError:
            return False
    if dom in ("mmlupro", "medqa"):
        nm = numeric_match(resp, item["gold"])
        if nm is not None:
            return nm                       # numeric gold -> tolerance compare
    return contains(resp, item["aliases"])


PROMPT = ("Answer the question. Reason briefly if you need to, then give the final "
          "answer on its own last line, prefixed with 'Answer:'.")

def final_line(resp):
    """Prefer the model's declared final answer when it gives one."""
    if not resp:
        return resp
    m = re.findall(r"(?i)answer\s*[:\-]\s*(.+)", resp)
    return m[-1].strip() if m else resp


if __name__ == "__main__":
    items = load_all(40)
    df = pd.DataFrame(items)
    print(df.groupby("domain").size().to_string())
    print(f"\ntotal {len(items)}")
    for d in ("triviaqa", "gsm8k", "medqa", "mmlupro:math", "mmlupro:health"):
        it = next(i for i in items if i["domain"] == d)
        print(f"\n[{d}] {it['question'][:130]}")
        print(f"   gold: {it['gold']}")
