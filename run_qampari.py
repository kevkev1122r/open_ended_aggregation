"""
QAMPARI: the cleanest test this project can run.

WHY IT IS CLEAN
  Answers are SETS of entities ("what manga did Ryoichi Ikegami draw?" -> 8
  titles, median). That gives three properties nothing else here had at once:

    1. The answer decomposes natively. Each list item IS an atomic fact, so
       ASC needs no sentence splitter and -- critically -- NO SUMMARISER. The
       output is the surviving item list. Nothing can hallucinate content that
       was not in a source response.
    2. Grading is set precision/recall/F1 with gold aliases. Local,
       deterministic, free. NO JUDGE. The v2 judge contradicted itself on 7% of
       identical answers; there is nothing here to contradict.
    3. Recall is first-class. ASC trades precision for recall, and F1 scores
       both. FACTS groundedness could only ever measure precision.

  So the only moving part left is the weights. That is the point.

WHAT IS COMPARED (all on the same parsed item sets)
    single model            each model's own list
    union of all models     the recall ceiling
    MA-count filter         keep items >= THETA MODELS asserted
                            (ours, ASC-style; NOT published ASC, which samples
                             one model m=50 times -- arXiv:2405.13131)
    WEIGHTED filter         keep items with sum(beta) >= THETA   (the contribution)

  ASC's own list mode is the baseline: "we directly used each item in the list
  as an atomic fact ... surface-form based clustering ... Theta threshold
  (tuned on a validation set) based filtering".

Usage:  ./venv/bin/python run_qampari.py [n]
"""
import os, sys, json, re, time, zipfile, threading, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import azure_backend as AZ
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = ["grok-4.3", "Kimi-K2.5", "Cohere-command-a-plus-05-2026",
        "MAI-Thinking-1", "DeepSeek-V4-Flash"]
GEN_PATH = f"{HERE}/data/qampari_gen.jsonl"
GEN_BUDGET = 12000
MAXW = 8
_lk = threading.Lock()

PROMPT = ("List every correct answer to the question, one per line, as a bare "
          "list. No numbering, no commentary, no explanation. If there are many, "
          "list as many as you can. Output only the list.")

_PUNC = re.compile(r"[^a-z0-9 ]")
_ART = re.compile(r"\b(a|an|the)\b")


def norm(s):
    s = str(s).lower().replace("’", "'")
    s = _PUNC.sub(" ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def parse_list(txt):
    """Pull list items out of a model response. Models ignore formatting
    instructions often enough that this must be forgiving."""
    out = []
    for line in str(txt).splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^\s*[-*•]\s*", "", s)          # bullets
        s = re.sub(r"^\s*\d+[.)]\s*", "", s)             # numbering
        s = re.sub(r"\s*\((?:19|20)\d\d\)\s*$", "", s)   # trailing year
        s = s.strip(" .;,")
        if not s or len(s) > 120:
            continue
        if re.match(r"(?i)^(here|the following|answers?|list|sure|note|these)\b", s):
            continue
        out.append(s)
    seen, uniq = set(), []
    for s in out:
        k = norm(s)
        if k and k not in seen:
            seen.add(k); uniq.append(s)
    return uniq


def score_set(pred_items, gold):
    """Precision / recall / F1 against gold answers, matching on any alias.

    A predicted item counts correct if it normalises to any gold alias. A gold
    answer counts covered if any predicted item matches one of its aliases.
    """
    gold_sets = [{norm(a) for a in ([g["answer_text"]] + list(g.get("aliases") or []))
                  if str(a).strip()} for g in gold]
    P = [norm(p) for p in pred_items]
    hit_gold = sum(1 for gs in gold_sets if any(p in gs for p in P))
    hit_pred = sum(1 for p in P if any(p in gs for gs in gold_sets))
    rec = hit_gold / max(1, len(gold_sets))
    prec = hit_pred / max(1, len(P))
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def load_items(n):
    z = zipfile.ZipFile(f"{HERE}/data/qampari.zip")
    with z.open("qampari_data/dev_data.jsonl") as f:
        rows = [json.loads(l) for l in f]
    rows = [r for r in rows if 5 <= len(r.get("answer_list") or []) <= 40][:n]
    return [dict(qid=r["qid"], question=r["question_text"],
                 gold=r["answer_list"]) for r in rows]


def done_keys(path):
    s = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l); s.add((r["qid"], r["model"]))
            except Exception: pass
    return s


def generate(items):
    done = done_keys(GEN_PATH)
    jobs = [(it, m) for it in items for m in POOL if (it["qid"], m) not in done]
    if not jobs:
        print(f"  generation complete ({len(done)} rows)", flush=True); return
    print(f"  {len(items)} questions x {len(POOL)} models — "
          f"{len(done)} cached, {len(jobs)} to run", flush=True)
    f = open(GEN_PATH, "a"); t0 = time.time(); n = 0
    def one(a):
        it, m = a
        txt, _ = AZ.chat(m, PROMPT, it["question"], GEN_BUDGET, temp=0)
        return it, m, txt
    with ThreadPoolExecutor(max_workers=MAXW) as ex:
        for it, m, txt in ex.map(one, jobs):
            n += 1
            if txt:
                items_ = parse_list(txt)
                p, r_, f1 = score_set(items_, it["gold"])
                with _lk:
                    f.write(json.dumps(dict(qid=it["qid"], model=m, resp=txt,
                                            items=items_, n_gold=len(it["gold"]),
                                            prec=p, rec=r_, f1=f1)) + "\n")
                    f.flush()
            if n % 100 == 0 or n == len(jobs):
                print(f"    gen {n}/{len(jobs)}  {time.time()-t0:5.0f}s  "
                      f"${AZ.spend():.3f}  empty {AZ.usage['empty']}", flush=True)
    f.close()


def viability(items):
    gold = {it["qid"]: it["gold"] for it in items}
    rows = [json.loads(l) for l in open(GEN_PATH)]
    by = collections.defaultdict(dict)
    for r in rows: by[r["qid"]][r["model"]] = r
    full = [q for q in by if len(by[q]) == len(POOL)]
    if not full:
        print("  no complete rows"); return
    print(f"\n  VIABILITY on {len(full)} complete questions")
    print(f"    {'model':<34}{'P':>7}{'R':>7}{'F1':>7}{'items':>7}")
    for m in POOL:
        P = np.mean([by[q][m]["prec"] for q in full])
        R = np.mean([by[q][m]["rec"] for q in full])
        F = np.mean([by[q][m]["f1"] for q in full])
        n = np.mean([len(by[q][m]["items"]) for q in full])
        print(f"    {m:<34}{100*P:6.1f}%{100*R:6.1f}%{100*F:6.1f}%{n:7.1f}")
    best = max(np.mean([by[q][m]["f1"] for q in full]) for m in POOL)
    # union ceiling: every item any model produced
    uP, uR, uF = [], [], []
    for q in full:
        allit = []
        for m in POOL: allit += by[q][m]["items"]
        p, r_, f1 = score_set(list(dict.fromkeys(allit)), gold[q])
        uP.append(p); uR.append(r_); uF.append(f1)
    print(f"    {'UNION of all 5 (no filter)':<34}{100*np.mean(uP):6.1f}%"
          f"{100*np.mean(uR):6.1f}%{100*np.mean(uF):6.1f}%")
    print(f"    best single F1 {100*best:.1f}%   union recall {100*np.mean(uR):.1f}%"
          f"   -> merging headroom in RECALL: {100*(np.mean(uR)-max(np.mean([by[q][m]['rec'] for q in full]) for m in POOL)):.1f} pts")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    items = load_items(n)
    print("=" * 72); print(f"  QAMPARI  n={len(items)}  pool={len(POOL)}"); print("=" * 72, flush=True)
    generate(items)
    viability(items)
    print(f"\n  Azure spend this process ${AZ.spend():.3f}", flush=True)
