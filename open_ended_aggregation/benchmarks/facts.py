"""
FACTS Grounding run: the long-form, surface-independently-graded benchmark the
kernel was actually built for.

WHY THIS BENCHMARK AND NOT THE LAST ONE
  The v2 suite failed to test KWA for two measured reasons: median answer was 4
  words (so 62% of questions could not show a kernel effect at all), and 61.3% of
  questions were unanimous (so aggregation was irrelevant on two thirds of them).
  Attempting HaluEval summarisation instead produced 4% accuracy, because a
  28-word reference summary is ONE arbitrary selection of facts from a 388-word
  article -- reference matching measures agreement with that choice, not truth.

  FACTS Grounding grades differently: is every claim in the response supported by
  the provided document? Two correct answers worded completely differently both
  pass. That is verification independent of surface form, which is the property
  the kernel needs in order to be given a fair test.

  It also ships DeepMind's own validated judge rubrics, so the grader is not
  something we invent and mis-specify -- which cost three attempts last run.

DESIGN
  Pool   : the five DEPLOYED Azure models. No OpenRouter: that balance is spent.
  Judge  : gpt-5.4-mini. OpenAI has no model in the pool, so independence holds.
           NOTE it runs deploymentless, which vanished mid-run once before --
           the run is resumable, so a disappearance costs only a restart.
  Grading: SENTENCE-LEVEL (json_alt rubric). Binary response-level grading gave
           97.1% best single and 2.9 pts of headroom -- unusable. See judge().

  Everything is resumable on (qid, model). Empty responses are NOT written, so an
  unanswered cell stays missing instead of silently grading as wrong.

Usage:  ./venv/bin/python run_facts.py [n_pilot] [n_full]
"""
import os, sys, json, time, re, threading, collections
import numpy as np, pandas as pd
from open_ended_aggregation.backends import azure as AZ
from concurrent.futures import ThreadPoolExecutor

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
POOL = ["grok-4.3", "Kimi-K2.5", "Cohere-command-a-plus-05-2026",
        "MAI-Thinking-1", "DeepSeek-V4-Flash"]
JUDGE = "gpt-5.4-mini"
GEN_PATH = f"{HERE}/data/facts_gen.jsonl"
JUD_PATH = f"{HERE}/data/facts_judged.jsonl"
THRESH = 1.0          # a response counts correct only if EVERY checkable sentence is grounded
MAX_CTX_WORDS = 1500          # keeps prompts sane; 728 of 860 qualify
GEN_BUDGET = 12000    # Kimi hits finish="length" at 4000 on ~1100-word contexts
MAXW = 8
_lk = threading.Lock()


def load_items(n):
    df = pd.read_parquet(f"{HERE}/data/facts_grounding.parquet")
    df = df[df.context_document.str.split().str.len() <= MAX_CTX_WORDS].reset_index(drop=True)
    df = df.head(n).copy()
    df["qid"] = ["facts:%04d" % i for i in range(len(df))]
    return df


def done_keys(path):
    s = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l); s.add((r["qid"], r["model"]))
            except Exception: pass
    return s


def generate(df):
    done = done_keys(GEN_PATH)
    jobs = [(i, m) for i in range(len(df)) for m in POOL
            if (df.qid[i], m) not in done]
    if not jobs:
        print(f"  generation already complete ({len(done)} rows)", flush=True)
        return
    print(f"  {len(df)} items x {len(POOL)} models — {len(done)} cached, {len(jobs)} to run",
          flush=True)
    f = open(GEN_PATH, "a"); t0 = time.time(); n = 0
    def one(a):
        i, m = a
        txt, meta = AZ.chat(m, None, df.full_prompt[i], GEN_BUDGET, temp=0)
        return i, m, txt, meta
    with ThreadPoolExecutor(max_workers=MAXW) as ex:
        for i, m, txt, meta in ex.map(one, jobs):
            n += 1
            if txt:                       # empty stays MISSING, never a wrong answer
                with _lk:
                    f.write(json.dumps(dict(qid=df.qid[i], model=m, resp=txt)) + "\n")
                    f.flush()
            if n % 100 == 0 or n == len(jobs):
                print(f"    gen {n}/{len(jobs)}  {time.time()-t0:5.0f}s  "
                      f"${AZ.spend():.3f}  empty {AZ.usage['empty']}", flush=True)
    f.close()


def judge(df):
    """Sentence-level groundedness, not one bit per response.

    MEASURED on the 40-item pilot, same responses both ways:
        binary response_level : unanimous 80.0%, best single 97.1%, headroom  2.9
        sentence-level json_alt: unanimous 30.3%, best single 75.8%, headroom 21.2

    Collapsing a 209-word answer to a single accurate/inaccurate bit destroys the
    signal the benchmark contains. Score = fraction of checkable sentences that
    are `supported`; sentences labelled `no_rad` (opinions, greetings) are not
    checkable and are excluded. A response is correct iff that fraction >= THRESH.
    """
    rub = json.load(open(f"{HERE}/data/facts_eval_prompts.json"))["json_alt"]
    gen = [json.loads(l) for l in open(GEN_PATH)]
    # Only judge qids belonging to THIS df. The file accumulates across runs, so
    # after a restart the 40-item pilot would otherwise hit rows from the earlier
    # 300-item pass and die on idx[qid]. That crash killed the previous run.
    mine = set(df.qid)
    by = {(r["qid"], r["model"]): r["resp"] for r in gen if r["qid"] in mine}
    done = done_keys(JUD_PATH)
    todo = [k for k in by if k not in done]
    if not todo:
        print(f"  judging already complete ({len(done)} rows)", flush=True)
        return
    print(f"  judging {len(todo)} responses sentence-by-sentence with {JUDGE}", flush=True)
    idx = {q: i for i, q in enumerate(df.qid)}
    f = open(JUD_PATH, "a"); t0 = time.time(); n = 0
    def one(k):
        qid, m = k
        i = idx[qid]
        p = (rub + f"\n\n**Context:**\n{df.context_document[i]}\n\n"
                   f"**Response:**\n{by[k]}\n\n"
                   "Return ONLY a JSON array of objects with keys 'sentence' and 'label'.")
        txt, _ = AZ.chat(JUDGE, None, p, 6000, temp=0)
        mm = re.search(r"\[.*\]", txt or "", re.S)
        if not mm:
            return qid, m, None, None
        try:
            arr = json.loads(mm.group(0))
        except Exception:
            return qid, m, None, None
        lab = [str(x.get("label", "")).lower() for x in arr if isinstance(x, dict)]
        need = [l for l in lab if l != "no_rad"]
        score = 1.0 if not need else sum(l == "supported" for l in need) / len(need)
        return qid, m, score, len(need)
    with ThreadPoolExecutor(max_workers=MAXW) as ex:
        for qid, m, sc, ns in ex.map(one, todo):
            n += 1
            if sc is not None:        # unparseable stays UNGRADED, never False
                with _lk:
                    f.write(json.dumps(dict(qid=qid, model=m, grounded=sc,
                                            n_checkable=ns,
                                            judged=bool(sc >= THRESH))) + "\n")
                    f.flush()
            if n % 100 == 0 or n == len(todo):
                print(f"    judge {n}/{len(todo)}  {time.time()-t0:5.0f}s  "
                      f"${AZ.spend():.3f}", flush=True)
    f.close()


def viability(df):
    """The checks that would have saved the previous run, run BEFORE scaling."""
    gen = [json.loads(l) for l in open(GEN_PATH)]
    jud = {(r["qid"], r["model"]): r["judged"] for r in
           (json.loads(l) for l in open(JUD_PATH))}
    by = collections.defaultdict(dict)
    for r in gen: by[r["qid"]][r["model"]] = r["resp"]
    full = [q for q in by if len(by[q]) == len(POOL) and
            all((q, m) in jud for m in POOL)]
    if not full:
        print("  no complete rows yet"); return None
    wl = np.mean([np.mean([len(by[q][m].split()) for m in POOL]) for q in full])
    acc = {m: 100*np.mean([jud[(q, m)] for q in full]) for m in POOL}
    nc = [sum(jud[(q, m)] for m in POOL) for q in full]
    uni = 100*np.mean([c == len(POOL) for c in nc])
    zero = 100*np.mean([c == 0 for c in nc])
    ceil = 100*np.mean([c > 0 for c in nc])
    print(f"\n  VIABILITY on {len(full)} complete items")
    print(f"    mean response length      {wl:8.0f} words   (want > 60)")
    print(f"    unanimous-correct         {uni:8.1f}%       (want < 60)")
    print(f"    nobody correct            {zero:8.1f}%")
    print(f"    contested                 {100-uni-zero:8.1f}%")
    print(f"    ceiling                   {ceil:8.1f}%")
    print(f"    best single model         {max(acc.values()):8.1f}%")
    print(f"    headroom                  {ceil-max(acc.values()):8.1f} pts   (want > 8)")
    for m, a in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"      {m:<34}{a:6.1f}%")
    ok = wl > 60 and uni < 60 and (ceil - max(acc.values())) > 5
    print(f"    -> {'VIABLE, scaling up' if ok else 'NOT viable on these thresholds'}")
    return ok


if __name__ == "__main__":
    n_pilot = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_full  = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    print("="*74); print(f"  PILOT  n={n_pilot}"); print("="*74, flush=True)
    df = load_items(n_pilot)
    generate(df); judge(df)
    ok = viability(df)
    if ok:
        print("\n" + "="*74); print(f"  FULL RUN  n={n_full}"); print("="*74, flush=True)
        df2 = load_items(n_full)
        generate(df2); judge(df2)
        viability(df2)
    print(f"\n  Azure spend this process ${AZ.spend():.3f}"
          f"  (unpriced: {sorted(AZ.usage['unpriced'])})", flush=True)
