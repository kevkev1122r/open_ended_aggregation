"""
QUEST — the second set-valued benchmark, and the one that can actually adjudicate.

WHY THIS DATASET
  PROGRESS_2026-08-17.docx claims "no second benchmark can replicate this". That
  is WRONG and this file is the correction. ASQA is recall-only and FACTS is
  precision-only, so neither can score a precision/recall filter -- but QUEST is
  set-valued entity retrieval with Precision/Recall/F1/Recall-5, exactly the
  metric family QAMPARI uses.

  ASC itself evaluated QUEST alongside QAMPARI (arXiv:2405.13131), on the same
  1,727-example test split loaded here. So a reviewer will ask why we skipped it.

WHAT THE DATA IS
  Entity-seeking queries with implicit set operations -- "Philippine remakes of
  South Korean films or 2010s prison dramas" -- whose gold answer is a SET of
  Wikipedia article titles. Median 11 answers per query.
  Malaviya et al., QUEST, ACL 2023. https://storage.googleapis.com/gresearch/quest/

SEPARATE FROM QAMPARI AND ASQA BY DESIGN
  Its own loader, its own prompt, its own generation file, its own metrics. No
  shared driver -- the three datasets have different answer shapes and different
  metric properties, and coupling them is what produced the "ASQA cannot
  replicate" confusion in the first place.

CAVEAT INHERITED FROM QAMPARI
  Gold is a bare list of article titles with NO aliases, so grading is exact
  normalised match. On QAMPARI that understates every arm by ~7 F1 (measured via
  the strict-vs-relaxed sensitivity) without changing any comparison. Expect the
  same here; report it, do not silently relax it.

Usage:  ./venv/bin/python run_quest.py [n]
"""
import os, sys, json, re, time, threading, collections

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
from open_ended_aggregation.backends import azure as AZ
from concurrent.futures import ThreadPoolExecutor

POOL = ["Cohere-command-a-plus-05-2026", "DeepSeek-V4-Flash", "Kimi-K2.5",
        "MAI-Thinking-1", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "grok-4.3"]
GEN_PATH = f"{HERE}/data/quest_gen.jsonl"
GEN_BUDGET = 12000
MAXW = 8
_lk = threading.Lock()

PROMPT = ("List every entity that satisfies the query, one per line, as a bare "
          "list of names. No numbering, no commentary, no explanation. If there "
          "are many, list as many as you can. Output only the list.")

_PUNC = re.compile(r"[^a-z0-9 ]")
_ART = re.compile(r"\b(a|an|the)\b")


def norm(s):
    s = str(s).lower().replace("’", "'")
    s = _PUNC.sub(" ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def parse_list(txt):
    out = []
    for line in str(txt).splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^\s*[-*•]\s*", "", s)
        s = re.sub(r"^\s*\d+[.)]\s*", "", s)
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
    """QUEST/QAMPARI metrics. gold is a list of article titles (no aliases)."""
    gold_sets = [{norm(g)} for g in gold if str(g).strip()]
    P = [norm(p) for p in pred_items]
    hit_gold = sum(1 for gs in gold_sets if any(p in gs for p in P))
    hit_pred = sum(1 for p in P if any(p in gs for gs in gold_sets))
    prec = hit_pred / max(1, len(P))
    rec = hit_gold / max(1, len(gold_sets))
    rec5 = min(hit_gold, 5) / max(1, min(len(gold_sets), 5))
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    f15 = 0.0 if prec + rec5 == 0 else 2 * prec * rec5 / (prec + rec5)
    return prec, rec, f1, rec5, f15


def load_items(n, split="test"):
    """ASC evaluated the 1,727-example test split. Same 5..40 answer filter as
    run_qampari, so the two datasets are directly comparable."""
    rows = [json.loads(l) for l in open(f"{HERE}/data/quest_{split}.jsonl") if l.strip()]
    rows = [r for r in rows if 5 <= len(r.get("docs") or []) <= 40][:n]
    return [dict(qid=f"quest:{split}:{i:05d}", question=r["query"], gold=list(r["docs"]))
            for i, r in enumerate(rows)]


def done_keys(path):
    s = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l); s.add((r["qid"], r["model"]))
            except Exception:
                pass
    return s


def generate(items):
    done = done_keys(GEN_PATH)
    jobs = [(it, m) for it in items for m in POOL if (it["qid"], m) not in done]
    if not jobs:
        print(f"  generation complete ({len(done)} rows)", flush=True); return
    print(f"  {len(items)} queries x {len(POOL)} models — "
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
                its = parse_list(txt)
                p, r_, f1, r5, f15 = score_set(its, it["gold"])
                with _lk:
                    f.write(json.dumps(dict(qid=it["qid"], model=m, resp=txt, items=its,
                                            n_gold=len(it["gold"]), prec=p, rec=r_,
                                            f1=f1, rec5=r5, f15=f15)) + "\n")
                    f.flush()
            if n % 100 == 0 or n == len(jobs):
                print(f"    gen {n}/{len(jobs)}  {time.time()-t0:5.0f}s  "
                      f"${AZ.spend():.3f}  empty {AZ.usage['empty']}", flush=True)
    f.close()


def coverage(items):
    have = collections.defaultdict(set)
    for l in open(GEN_PATH) if os.path.exists(GEN_PATH) else []:
        try:
            r = json.loads(l); have[r["qid"]].add(r["model"])
        except Exception:
            pass
    got = sum(len(have[it["qid"]] & set(POOL)) for it in items)
    full = sum(1 for it in items if len(have[it["qid"]] & set(POOL)) == len(POOL))
    return got, len(items) * len(POOL), full


def viability(items):
    import statistics
    rows = [json.loads(l) for l in open(GEN_PATH)]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["qid"]][r["model"]] = r
    full = [q for q in by if len(by[q]) == len(POOL)]
    if not full:
        print("  no complete rows"); return
    print(f"\n  VIABILITY on {len(full)} complete queries")
    print(f"    {'model':<34}{'P':>7}{'R':>7}{'F1':>7}{'R-5':>7}{'items':>7}")
    for m in sorted(POOL, key=lambda m: -statistics.mean(by[q][m]["f1"] for q in full)):
        print(f"    {m:<34}"
              f"{100*statistics.mean(by[q][m]['prec'] for q in full):6.1f}%"
              f"{100*statistics.mean(by[q][m]['rec'] for q in full):6.1f}%"
              f"{100*statistics.mean(by[q][m]['f1'] for q in full):6.1f}%"
              f"{100*statistics.mean(by[q][m]['rec5'] for q in full):6.1f}%"
              f"{statistics.mean(len(by[q][m]['items']) for q in full):7.1f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    items = load_items(n)
    print("=" * 74)
    print(f"  QUEST  n={len(items)}  agents={len(POOL)}")
    print("=" * 74, flush=True)
    prev = -1
    for p in range(1, 15):
        got, want, full = coverage(items)
        if got >= want:
            print(f"  pass {p}: COMPLETE ({got}/{want}, {full} full)", flush=True); break
        if got == prev:
            print(f"  pass {p}: no progress -- stopping at {got}/{want}", flush=True); break
        prev, before = got, got
        try:
            generate(items)
        except Exception as e:
            print(f"  pass {p} raised {type(e).__name__}: {e}", flush=True)
        got, want, full = coverage(items)
        print(f"  pass {p}: +{got-before} -> {got}/{want} ({100*got/want:.1f}%), "
              f"{full} full queries", flush=True)
    viability(items)
    print(f"\n  Azure spend this process ${AZ.spend():.3f}", flush=True)
