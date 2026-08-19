"""
ASQA: the first benchmark in this project with an automatic RECALL metric.

WHY ASQA
  Every previous attempt failed on the grader, not the method:
    v2 suite   - median answer 4 words; exact match is already the correct
                 similarity function, so the kernel had nothing to do on 62%.
    HaluEval   - 4%: a 28-word reference is ONE arbitrary selection of facts.
    FACTS      - fixed the grader (surface-independent, sentence-level) but the
                 metric is groundedness, i.e. PRECISION only. ASC trades
                 precision for recall, so FACTS cannot score it fairly.

  ASQA questions are AMBIGUOUS on purpose. Each has ~3 disambiguated
  interpretations, each with its own short answer. The metric (STR-EM) asks:
  of those short answers, how many appear in the generated long answer?

  That is a recall metric computed by string matching -- no judge, no embedding
  threshold, nothing to tune. The judge contradicted itself on 7% of identical
  answers in the v2 run; here there is no judge to contradict anything.

  Ambiguity, which made ASQA wrong for SELECTION (argmax_s P(s|answers) assumes
  one truth), is exactly what makes it right for MERGING: no single response can
  cover every interpretation, so combining them is the point.

DESIGN
  Pool  : the 5 DEPLOYED Azure models. gpt-5.x is deploymentless -- it vanished
          mid-run once already, so it is not trusted in the pool.
  Grader: STR-EM, local, deterministic, free.
  No judge, no OpenRouter.

Usage:  ./venv/bin/python run_asqa.py [n]
"""
import os, sys, json, time, re, threading, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import azure_backend as AZ
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = ["grok-4.3", "Kimi-K2.5", "Cohere-command-a-plus-05-2026",
        "MAI-Thinking-1", "DeepSeek-V4-Flash"]
GEN_PATH = f"{HERE}/data/asqa_gen.jsonl"
GEN_BUDGET = 12000
MAXW = 8
_lk = threading.Lock()

PROMPT = ("Answer the question. It may be ambiguous and have several valid "
          "interpretations -- cover all of them. Write a single paragraph of "
          "about 80 words. Do not ask for clarification.")

# ------------------------------------------------------------------ STR-EM
_PUNC = re.compile(r"[^a-z0-9 ]")
_ART = re.compile(r"\b(a|an|the)\b")


def norm(s):
    s = str(s).lower().replace("’", "'")
    s = _PUNC.sub(" ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def str_em(response, short_answer_sets):
    """ASQA's metric: fraction of disambiguated interpretations whose short
    answer appears in the response. Each interpretation supplies several
    aliases; matching ANY alias counts that interpretation as covered.

    This is RECALL over interpretations -- the quantity ASC optimises and the
    quantity FACTS groundedness could not express.
    """
    r = " " + norm(response) + " "
    hit = 0
    for aliases in short_answer_sets:
        if any((" " + norm(a) + " ") in r for a in aliases if str(a).strip()):
            hit += 1
    return hit / max(1, len(short_answer_sets))


def load_items(n):
    df = pd.read_parquet(f"{HERE}/data/asqa_dev.parquet").head(n).reset_index(drop=True)
    items = []
    for i, row in df.iterrows():
        sets = [list(p["short_answers"]) for p in row["qa_pairs"]]
        items.append(dict(qid="asqa:%04d" % i,
                          question=row["ambiguous_question"],
                          short_sets=sets,
                          n_interp=len(sets)))
    return items


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
            if txt:                       # empty stays MISSING, never wrong
                with _lk:
                    f.write(json.dumps(dict(qid=it["qid"], model=m, resp=txt,
                                            strem=str_em(txt, it["short_sets"]),
                                            n_interp=it["n_interp"])) + "\n")
                    f.flush()
            if n % 100 == 0 or n == len(jobs):
                print(f"    gen {n}/{len(jobs)}  {time.time()-t0:5.0f}s  "
                      f"${AZ.spend():.3f}  empty {AZ.usage['empty']}", flush=True)
    f.close()


def viability(items):
    rows = [json.loads(l) for l in open(GEN_PATH)]
    by = collections.defaultdict(dict)
    for r in rows: by[r["qid"]][r["model"]] = r
    full = [q for q in by if len(by[q]) == len(POOL)]
    if not full:
        print("  no complete rows"); return
    M = np.array([[by[q][m]["strem"] for m in POOL] for q in full])
    wl = np.mean([len(by[q][m]["resp"].split()) for q in full for m in POOL])
    print(f"\n  VIABILITY on {len(full)} complete questions")
    print(f"    mean response length      {wl:8.0f} words")
    print(f"    interpretations/question  {np.mean([by[q][POOL[0]]['n_interp'] for q in full]):8.1f}")
    print(f"    {'model':<34}{'STR-EM':>9}")
    for j, m in enumerate(POOL):
        print(f"      {m:<32}{100*M[:, j].mean():8.1f}%")
    best = M.mean(axis=0).max()
    print(f"    best single model         {100*best:8.1f}%")
    print(f"    union ceiling (any model covers it) {100*M.max(axis=1).mean():6.1f}%")
    print(f"    ORACLE MERGE ceiling      {100*np.mean([ _union(by[q]) for q in full]):8.1f}%"
          f"   <- what merging could reach")
    print(f"    headroom over best single {100*(np.mean([_union(by[q]) for q in full])-best):8.1f} pts")


def _union(d):
    """If you could take the best interpretation coverage from every model at
    once, what fraction of interpretations would be covered? This is the ceiling
    a MERGING method aims at, and it is strictly above any single response."""
    covered = None
    for m, r in d.items():
        v = np.array(_covered_vec(r))
        covered = v if covered is None else np.maximum(covered, v)
    return float(covered.mean()) if covered is not None else 0.0


_ITEMS = {}
def _covered_vec(r):
    sets = _ITEMS[r["qid"]]["short_sets"]
    rr = " " + norm(r["resp"]) + " "
    return [1.0 if any((" " + norm(a) + " ") in rr for a in al if str(a).strip()) else 0.0
            for al in sets]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    items = load_items(n)
    _ITEMS.update({it["qid"]: it for it in items})
    print("=" * 72); print(f"  ASQA  n={len(items)}  pool={len(POOL)}"); print("=" * 72, flush=True)
    generate(items)
    viability(items)
    print(f"\n  Azure spend this process ${AZ.spend():.3f}", flush=True)
