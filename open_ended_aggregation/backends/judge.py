"""
LLM judge for grading, plus an audit of the string grader against it.

Why a judge. String matching against a reference has known, measured failure modes:
alias gaps produce false negatives, substring hits produce false positives
("not Paris, but London" matches London), and numeric answers with different units
or rounding are unmatchable. On the earlier TriviaQA run those defects were visible
but unquantified. A judge fixes the grading AND, by disagreeing with the string
grader in specific places, finally measures how wrong the string grader was.

Two design choices that matter:

  * INDEPENDENCE. The judge's lab must have NO model in the candidate pool, so
    there is no path for a judge to favour its own family.

    CHANGED 2026-08-14: was deepseek-v3.2, which became INVALID when the Azure
    quota cuts forced DeepSeek-V4-Flash into the pool. A judge grading its own
    lab's outputs is exactly the bias this rule exists to prevent, and nothing in
    the code would have flagged it -- the run would have completed and the
    numbers would have looked fine.

    Now Google, which is held out of the pool deliberately and is unavailable on
    Azure anyway. If the pool ever gains a Google model, this must change again.
    Pool labs as of this writing: Anthropic, OpenAI, xAI, Moonshot, Cohere,
    Microsoft, DeepSeek.
  * BLINDNESS. The judge never sees which model produced a response. It sees the
    question, the reference answer, and the response. Nothing else.

The judge grades against a REFERENCE answer, which is a much more mechanical task
than open-ended quality judging and correspondingly less prone to bias -- but the
audit below still checks for lab-level favouritism rather than assuming it away.
"""
import os, sys, json, time, threading, re
import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
KEY = [l.split("=", 1)[1].strip() for l in open(f"{HERE}/.env")
       if l.startswith("OPENROUTER_API_KEY=")][0]
URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE = "google/gemini-2.5-flash-lite"   # see INDEPENDENCE note; flash-lite chosen on budget
MAXW = 12

SYS = """You grade whether a candidate answer is correct, given a reference answer.

Rules:
- Judge ONLY the factual content of the final answer. Ignore style, verbosity,
  reasoning shown, and formatting.
- Accept different surface forms of the same answer (abbreviations, alternative
  names, different units, reasonable rounding, extra qualifying words).
- Reject if the candidate hedges without committing, refuses, contradicts itself,
  or is cut off before stating an answer.
- Reject if the candidate states the reference answer only to dismiss it.

Reply with exactly one word: CORRECT or INCORRECT."""

_lk = threading.Lock()
_sp = {"in": 0, "out": 0, "err": 0}


def judge_one(question, reference, response, retries=4):
    if response is None or not str(response).strip():
        return False
    user = (f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
            f"Candidate answer:\n{response}\n\nCORRECT or INCORRECT?")
    # max_tokens was 5, which is enough for the literal word CORRECT and NOT
    # enough for a thinking model to reach it. Measured 2026-08-14 on
    # gemini-3.7-flash: at 5 tokens it returns content=null, finish="length",
    # having spent the budget on reasoning; at 2000 it answers after ~33
    # reasoning tokens. The old value silently graded EVERY response INCORRECT
    # -- 0/6606 -- while string matching said 65.8%. Any judge swap must re-check
    # this: deepseek-v3.2 needed 5 tokens, its replacement needs hundreds.
    body = {"model": JUDGE, "temperature": 0, "max_tokens": 2000,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": user}]}
    for a in range(retries):
        try:
            r = requests.post(URL, timeout=90,
                              headers={"Authorization": f"Bearer {KEY}"}, json=body)
            if r.status_code == 429:
                time.sleep(2 ** a + 1); continue
            r.raise_for_status(); j = r.json()
            if "choices" not in j:
                time.sleep(1.5 ** a); continue
            u = j.get("usage", {}) or {}
            with _lk:
                _sp["in"] += u.get("prompt_tokens", 0)
                _sp["out"] += u.get("completion_tokens", 0)
            txt = (j["choices"][0]["message"]["content"] or "").strip().upper()
            # An ABSENT verdict is not a verdict of INCORRECT. The old code did
            # `"".startswith("CORRECT")` -> False, which turned every silent
            # judge failure into a confident wrong grade. Return None instead:
            # the caller records it as ungraded and the analysis falls back to
            # string matching, which is visible in the graded_by breakdown.
            if txt.startswith("CORRECT"):
                return True
            if txt.startswith("INCORRECT"):
                return False
            with _lk:
                _sp["unparsed"] = _sp.get("unparsed", 0) + 1
            return None
        except Exception:
            time.sleep(1.5 ** a)
    with _lk:
        _sp["err"] += 1
    return None


def spend():
    r = requests.get("https://openrouter.ai/api/v1/key",
                     headers={"Authorization": f"Bearer {KEY}"}, timeout=30).json()["data"]
    return r["usage"]


def run(src, out, question_map, limit=None):
    rows = [json.loads(l) for l in open(src)]
    if limit:
        rows = rows[:limit]
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                r = json.loads(l); done.add((r["qid"], r["model"]))
            except Exception:
                pass
    todo = [r for r in rows if (r["qid"], r["model"]) not in done]
    print(f"  {len(rows):,} responses, {len(done):,} already judged, {len(todo):,} to go")
    if not todo:
        return out
    st = spend(); t0 = time.time()
    f = open(out, "a")
    with ThreadPoolExecutor(max_workers=MAXW) as ex:
        futs = {ex.submit(judge_one, question_map.get(r["qid"], ""), r["gold"], r["resp"]): r
                for r in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            r = futs[fu]; v = fu.result()
            if v is not None:
                with _lk:
                    f.write(json.dumps(dict(qid=r["qid"], model=r["model"],
                                            domain=r["domain"], judged=bool(v),
                                            string=bool(r["correct"]))) + "\n")
                    f.flush()
            if i % 300 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}  {time.time()-t0:5.0f}s  "
                      f"tok {_sp['in']+_sp['out']:,}  err {_sp['err']}")
    f.close()
    print(f"  judge cost: ${spend()-st:.4f}")
    return out


def audit(path):
    """Where does the string grader disagree with the judge, and how badly?"""
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    n = len(d)
    agree = (d.judged == d.string).mean()
    fp = ((d.string) & (~d.judged)).mean()       # string said correct, judge says no
    fn = ((~d.string) & (d.judged)).mean()       # string said wrong, judge says yes
    print(f"\n  string grader vs judge, n={n:,}")
    print(f"    agreement            {100*agree:6.2f}%")
    print(f"    string FALSE POSITIVE {100*fp:6.2f}%   (string said correct, judge disagrees)")
    print(f"    string FALSE NEGATIVE {100*fn:6.2f}%   (string said wrong, judge disagrees)")
    print(f"    net bias on accuracy  {100*(d.string.mean()-d.judged.mean()):+6.2f} pts "
          f"(string {100*d.string.mean():.2f} vs judge {100*d.judged.mean():.2f})")
    print(f"\n  by domain:")
    print(f"    {'domain':<20}{'n':>6}{'agree':>9}{'str FP':>9}{'str FN':>9}{'str-judge':>11}")
    print("    " + "-" * 64)
    d["dom"] = d.domain.str.split(":").str[0]
    for dom, g in d.groupby("dom"):
        print(f"    {dom:<20}{len(g):>6}{100*(g.judged==g.string).mean():>8.1f}%"
              f"{100*((g.string)&(~g.judged)).mean():>8.1f}%"
              f"{100*((~g.string)&(g.judged)).mean():>8.1f}%"
              f"{100*(g.string.mean()-g.judged.mean()):>+10.1f}")
    # lab-level favouritism check: is the judge kinder to any one lab?
    d["lab"] = d.model.str.split("/").str[0]
    print(f"\n  judge leniency by lab (judge% minus string%) -- large spread would "
          f"suggest favouritism:")
    for lab, g in d.groupby("lab"):
        print(f"    {lab:<22}{100*(g.judged.mean()-g.string.mean()):>+7.2f} pts   n={len(g)}")
    return d


if __name__ == "__main__":
    import benchmarks as B
    src = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/data/pilot_cal.jsonl"
    out = src.replace(".jsonl", "_judged.jsonl")
    items = B.load_all(40)
    qmap = {i["qid"]: i["question"] for i in items}
    run(src, out, qmap)
    audit(out)
