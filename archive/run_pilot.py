"""Calibration pilot: measure ACTUAL capability of candidate models before committing.

The point is not to test aggregation. It is to find out which candidates are
genuinely matched in general capability, since the niche-specialisation question
is only meaningful among equals. Selecting on price or parameter count is a guess;
this measures it.
"""
import os, sys, json, time, threading
import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import benchmarks as B

HERE=os.path.dirname(os.path.abspath(__file__))
KEY=[l.split("=",1)[1].strip() for l in open(f"{HERE}/.env") if l.startswith("OPENROUTER_API_KEY=")][0]
URL="https://openrouter.ai/api/v1/chat/completions"

CANDIDATES=[
 "anthropic/claude-haiku-4.5",
 "openai/gpt-5.4-mini",
 "openai/o4-mini",
 "qwen/qwen3.7-max",
 "qwen/qwen3-max-thinking",
 "z-ai/glm-5-turbo",
 "thinkingmachines/inkling",
 # replacements for two candidates blocked by account settings
 # (sakana: data-policy guardrail; meta/muse: 18+ attestation) -- both are
 # settings only the account owner can change, so different labs were used instead.
 "google/gemini-3.7-flash",
 "moonshotai/kimi-k2",
 "minimax/minimax-m2",
]
MAXW=10
_lk=threading.Lock(); _sp={"in":0,"out":0,"err":0}

def ask(model,item,retries=4):
    body={"model":model,"temperature":0,"max_tokens":item["max_tokens"],
          "messages":[{"role":"system","content":B.PROMPT},
                      {"role":"user","content":item["question"]}]}
    for a in range(retries):
        try:
            r=requests.post(URL,timeout=120,headers={"Authorization":f"Bearer {KEY}"},json=body)
            if r.status_code==429: time.sleep(2**a+1); continue
            r.raise_for_status(); j=r.json()
            if "choices" not in j: time.sleep(1.5**a); continue
            u=j.get("usage",{}) or {}
            with _lk:
                _sp["in"]+=u.get("prompt_tokens",0); _sp["out"]+=u.get("completion_tokens",0)
            return (j["choices"][0]["message"]["content"] or "").strip()
        except Exception: time.sleep(1.5**a)
    with _lk: _sp["err"]+=1
    return None

def cost():
    r=requests.get("https://openrouter.ai/api/v1/key",headers={"Authorization":f"Bearer {KEY}"},timeout=30).json()["data"]
    return r["usage"]

if __name__=="__main__":
    per=int(sys.argv[1]) if len(sys.argv)>1 else 40
    items=B.load_all(per, overrides={"mmlupro": per*2})
    print(f"  {len(items)} questions across {len(set(i['domain'].split(':')[0] for i in items))} benchmarks")
    print(f"  {len(CANDIDATES)} candidate models -> {len(items)*len(CANDIDATES):,} calls")
    start=cost(); print(f"  spend before: ${start:.4f}\n")
    path=f"{HERE}/data/full_run.jsonl"
    done=set()
    if os.path.exists(path):
        for l in open(path):
            try: r=json.loads(l); done.add((r["qid"],r["model"]))
            except: pass
    jobs=[(it,m) for it in items for m in CANDIDATES if (it["qid"],m) not in done]
    print(f"  {len(done)} cached, {len(jobs)} to run")
    f=open(path,"a"); t0=time.time()
    with ThreadPoolExecutor(max_workers=MAXW) as ex:
        futs={ex.submit(ask,m,it):(it,m) for it,m in jobs}
        for i,fu in enumerate(as_completed(futs),1):
            it,m=futs[fu]; txt=fu.result()
            if txt is not None:
                with _lk:
                    f.write(json.dumps(dict(qid=it["qid"],domain=it["domain"],model=m,
                        resp=txt,correct=B.grade(B.final_line(txt),it),gold=it["gold"]))+"\n"); f.flush()
            if i%250==0 or i==len(jobs):
                print(f"    {i}/{len(jobs)}  {time.time()-t0:5.0f}s  tok {_sp['in']+_sp['out']:,}  err {_sp['err']}")
    f.close()
    end=cost(); print(f"\n  spend after: ${end:.4f}   (this run: ${end-start:.4f})")
