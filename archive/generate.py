"""
REAL GENERATION RUN -- TriviaQA via OpenRouter.

This is the first time anything in this project touches real multi-model output.

Design decisions and why:
  * TriviaQA rc.nocontext -- genuinely free-form (no options offered, no K), with
    official alias lists so grading is automatic string matching. No judge model,
    so no judge cost and no judge noise.
  * TWO PROMPT CONDITIONS. 'terse' pushes models toward a canonical short string;
    'natural' lets them answer however they like. The kernel's whole claimed
    advantage is pooling surface variants, so the advantage should be LARGER in
    'natural'. If it is not, my explanation of the mechanism is wrong.
  * temperature 0 -- we want each model's characteristic answer, not sampling noise.
  * 6 models spanning a deliberate skill range, 3 of them Llamas, so same-family
    error correlation can be measured on live models.

Everything is written incrementally to JSONL and the run is resumable, so a crash
or a rate-limit storm never loses paid work.

Usage:
    ./venv/bin/python generate.py pilot          # 50 questions, ~$0.01
    ./venv/bin/python generate.py full 3000      # the real run
    ./venv/bin/python generate.py cost           # what has been spent so far
"""
import os, sys, json, time, re, threading
import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = None
for line in open(os.path.join(HERE, ".env")):
    if line.startswith("OPENROUTER_API_KEY="):
        KEY = line.strip().split("=", 1)[1]
URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "meta-llama/llama-3.3-70b-instruct",   # strong
    "openai/gpt-4o-mini",                  # strong, different family
    "microsoft/phi-4",                     # mid
    "google/gemma-3-12b-it",               # mid
    "meta-llama/llama-3.1-8b-instruct",    # weak
    "meta-llama/llama-3.2-3b-instruct",    # weakest
]
PROMPTS = {
    "terse":   ("Answer the trivia question with just the answer and nothing else.",  16),
    "natural": ("Answer the trivia question.",                                        64),
}
MAXW = 12
_lock = threading.Lock()
_spend = {"prompt": 0, "completion": 0, "calls": 0, "errors": 0}


# ---------------------------------------------------------------- grading
_ART = re.compile(r"\b(a|an|the)\b", re.I)
_PUN = re.compile(r"[^a-z0-9 ]")

def norm(s):
    s = s.lower().replace("’", "'")
    s = _PUN.sub(" ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())

def is_correct(resp, aliases):
    """An answer counts if any normalized alias appears as a token-substring.

    Substring (not equality) because 'natural' answers embed the entity in a
    sentence: 'The man behind The Chipmunks was David Seville.'
    """
    r = norm(resp)
    if not r:
        return False
    for a in aliases:
        a = norm(a)
        if a and (f" {a} " in f" {r} "):
            return True
    return False


# ---------------------------------------------------------------- api
def ask(model, question, style, retries=4):
    instr, mx = PROMPTS[style]
    body = {"model": model, "temperature": 0, "max_tokens": mx,
            "messages": [{"role": "system", "content": instr},
                         {"role": "user", "content": question}]}
    for a in range(retries):
        try:
            r = requests.post(URL, timeout=90,
                              headers={"Authorization": f"Bearer {KEY}",
                                       "Content-Type": "application/json"},
                              json=body)
            if r.status_code == 429:
                time.sleep(2 ** a + 1); continue
            r.raise_for_status()
            j = r.json()
            if "choices" not in j:
                time.sleep(1.5 ** a); continue
            txt = j["choices"][0]["message"]["content"] or ""
            u = j.get("usage", {}) or {}
            with _lock:
                _spend["prompt"] += u.get("prompt_tokens", 0)
                _spend["completion"] += u.get("completion_tokens", 0)
                _spend["calls"] += 1
            return txt.strip()
        except Exception:
            time.sleep(1.5 ** a)
    with _lock:
        _spend["errors"] += 1
    return None


# ---------------------------------------------------------------- runner
def load_questions(n, seed=0):
    d = pd.read_parquet(os.path.join(HERE, "data", "triviaqa_val.parquet"))
    d = d.sample(frac=1.0, random_state=seed).reset_index(drop=True).iloc[:n]
    out = []
    for _, r in d.iterrows():
        al = list(r["answer"]["normalized_aliases"]) + [r["answer"]["value"]]
        out.append(dict(qid=r["question_id"], q=r["question"],
                        gold=r["answer"]["value"], aliases=[a for a in al if a]))
    return out


def run(tag, n):
    qs = load_questions(n)
    path = os.path.join(HERE, "data", f"gen_{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line); done.add((r["qid"], r["model"], r["style"]))
            except Exception:
                pass
    jobs = [(q, m, s) for q in qs for m in MODELS for s in PROMPTS
            if (q["qid"], m, s) not in done]
    print(f"  {len(qs)} questions x {len(MODELS)} models x {len(PROMPTS)} styles")
    print(f"  {len(done)} already done, {len(jobs)} to run\n")
    if not jobs:
        print("  nothing to do"); return path

    t0 = time.time()
    f = open(path, "a")
    with ThreadPoolExecutor(max_workers=MAXW) as ex:
        futs = {ex.submit(ask, m, q["q"], s): (q, m, s) for q, m, s in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            q, m, s = futs[fu]
            txt = fu.result()
            if txt is not None:
                with _lock:
                    f.write(json.dumps(dict(qid=q["qid"], model=m, style=s, resp=txt,
                                            correct=is_correct(txt, q["aliases"]),
                                            gold=q["gold"])) + "\n")
                    f.flush()
            if i % 200 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"    {i}/{len(jobs)}  {el:5.0f}s  "
                      f"tokens {_spend['prompt']+_spend['completion']:,}  errors {_spend['errors']}")
    f.close()
    return path


def report(path):
    rows = [json.loads(l) for l in open(path)]
    d = pd.DataFrame(rows)
    print(f"\n  responses: {len(d):,}")
    print(f"\n  {'model':<38}{'terse':>9}{'natural':>10}")
    print("  " + "-" * 57)
    for m in MODELS:
        s = d[d["model"] == m]
        t = 100 * s[s["style"] == "terse"].correct.mean() if len(s[s["style"] == "terse"]) else float("nan")
        nn = 100 * s[s["style"] == "natural"].correct.mean() if len(s[s["style"] == "natural"]) else float("nan")
        print(f"  {m:<38}{t:>8.1f}%{nn:>9.1f}%")
    for style in PROMPTS:
        sub = d[d["style"] == style]
        piv = sub.pivot_table(index="qid", columns="model", values="resp", aggfunc="first")
        piv = piv.dropna()
        uniq = piv.apply(lambda r: len(set(norm(x) for x in r)), axis=1)
        cor = sub.pivot_table(index="qid", columns="model", values="correct", aggfunc="first").dropna()
        print(f"\n  [{style}] distinct answer strings per question: mean {uniq.mean():.2f} of {len(MODELS)}")
        print(f"  [{style}] all 6 models agree exactly: {100*(uniq==1).mean():.1f}% of questions")
        print(f"  [{style}] at least one model correct: {100*(cor.sum(axis=1)>0).mean():.1f}%")
        print(f"  [{style}] every model wrong:          {100*(cor.sum(axis=1)==0).mean():.1f}%")


def cost():
    r = requests.get("https://openrouter.ai/api/v1/key",
                     headers={"Authorization": f"Bearer {KEY}"}, timeout=30).json()["data"]
    print(f"  total spent on this key so far: ${r['usage']:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if cmd == "cost":
        cost()
    elif cmd == "pilot":
        cost(); p = run("pilot", 50); report(p); cost()
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
        cost(); p = run("full", n); report(p); cost()
