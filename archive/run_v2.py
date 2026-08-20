"""
Run v2 -- fixes the reasoning-token failure that invalidated run v1.

WHAT WENT WRONG IN V1
    max_tokens was sized for ANSWER length (60-400). Reasoning models spend that
    budget on internal reasoning tokens and return content="". Six of ten models
    came back 38-88% empty, which produced a fake 67-point capability spread and
    invalidated every downstream result. Empty responses were never checked for.

WHAT IS DIFFERENT HERE
  1. PRE-FLIGHT. Every model is tested on every domain before any bulk spending.
     Any model that returns empty is reported and excluded rather than silently
     scoring zero.
  2. GENEROUS UNIFORM BUDGETS. Same generous max_tokens for every model, so no
     model is handicapped. Reasoning is NOT disabled -- that would strip reasoning
     models of their advantage on GSM8K and be unfair in the opposite direction.
  3. EMPTY-RESPONSE RETRY. An empty result is retried at double the budget before
     being recorded.
  4. HARD SPEND GUARD. The run aborts if it crosses the cap, instead of discovering
     the overrun afterwards.

Usage:  ./venv/bin/python run_v2.py preflight
        ./venv/bin/python run_v2.py run 50 2.40      # 50 q/domain, abort at $2.40
"""
import os, sys, json, time, threading
import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import benchmarks as B

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("BACKEND", "azure")      # azure | openrouter
if BACKEND == "openrouter":
    KEY = [l.split("=", 1)[1].strip() for l in open(f"{HERE}/.env")
           if l.startswith("OPENROUTER_API_KEY=")][0]
    URL = "https://openrouter.ai/api/v1/chat/completions"
else:
    import azure_backend as AZ
    import openrouter_backend as OR      # mixed pool -- see ask()

OPENROUTER_MODELS = [
    "anthropic/claude-haiku-4.5", "openai/gpt-5.4-mini", "openai/o4-mini",
    "qwen/qwen3.7-max", "qwen/qwen3-max-thinking", "z-ai/glm-5-turbo",
    "google/gemini-3.7-flash",
    "moonshotai/kimi-k2", "minimax/minimax-m2",
    # thinkingmachines/inkling DROPPED: returned empty on medqa and mmlupro even
    # at 8x budget in pre-flight. A model scoring fake zeros on half the domains
    # would poison the specialisation matrix -- exactly the v1 failure.
]

# Azure catalogue probe, 2026-08-13. Only 4 of the 9 above have a Foundry peer:
#   claude-haiku-4.5   -> claude-haiku-4-5
#   gpt-5.4-mini       -> gpt-5.4-mini          (callable today, no deployment)
#   o4-mini            -> o4-mini
#   kimi-k2            -> Kimi-K2.5 / Kimi-K2.6-2026-04-20   (newer, not the same model)
# Not on the per-token surface: qwen3.7-max, qwen3-max-thinking (only qwen3-32b,
# not a frontier peer), glm-5-turbo, gemini-3.7-flash, minimax-m2.
#
# CAVEAT, and it bit us once already: that list is what /openai/v1/models
# reports, which is the per-token serverless surface ONLY. Foundry also carries
# a Hugging Face catalogue that endpoint never returns -- zai-org--glm-5.2-fp8
# and qwen--qwen3.6-27b are both in the portal and both invisible to the API
# probe. Those deploy to MANAGED COMPUTE: a GPU VM billed by the hour, running
# whether or not you call it, so AZ.PRICE (per-token) cannot bound them and they
# need their own guardrail. Gemini is genuinely absent (Microsoft does not serve
# Google's models); MiniMax is open-weight and may well be in the HF catalogue.
# Search the portal, not this comment, before concluding a model is unavailable.
# Substitutes that keep lab diversity -- the property the specialisation
# experiment actually depends on: grok-4.3 / grok-4-20-reasoning (xAI),
# DeepSeek-V4-Pro-2026-04-23, Mistral-Large-3, cohere-command-a,
# Llama-4-Maverick-17B-128E-Instruct-FP8, MAI-Thinking-1, gpt-oss-120b.
#
# EXCLUDE any model that rejects temperature=0 (gpt-5.5, gpt-5.6-*): running one
# model at temperature 1 while its peers run at 0 puts a confound directly on
# the quantity being measured. preflight flags this per model.
# Newest best-value mid-frontier model from each of nine labs. One model per lab
# is the point: the specialisation experiment needs errors to decorrelate, and
# same-lab models share training data, tokenizer and RLHF lineage -- the measured
# 215-621x error correlation on TriviaQA is what a low-diversity pool buys you.
#
# Google is held out ON PURPOSE. It is the judge (§5 step 4: the judge's lab must
# have no model in the pool), and Gemini is not on Azure anyway -- judging runs
# off-Azure against a separate key.
#
# gpt-5.4 rather than 5.5/5.6-*: the newer OpenAI models reject temperature=0,
# and one model sampling at temperature 1 while eight sample at 0 is a confound
# on the exact quantity being measured. Verified deploymentless, 2026-08-13.
AZURE_MODELS = [
    "anthropic/claude-sonnet-5",                # Anthropic  via OpenRouter, not Azure
    "openai/gpt-5.4",                           # OpenAI     via OpenRouter -- 0 Azure quota
    "grok-4.3",                                 # xAI
    "DeepSeek-V4-Flash",                        # DeepSeek   (deployment name, not catalogue id)
    "Kimi-K2.5",                                # Moonshot   (K2.6 preferred; K2.5 is what got deployed)
    "Cohere-command-a-plus-05-2026",            # Cohere
    "MAI-Thinking-1",                           # Microsoft
]
# CUT 2026-08-13 for zero quota, not for design reasons:
#   claude-sonnet-5 (Anthropic), mistral-medium-3-5 (Mistral),
#   Llama-4-Maverick-17B-128E-Instruct-FP8 (Meta)
#
# Older siblings of all three almost certainly DO have quota -- newest models are
# rationed hardest. Deliberately not taking them. The premise of this experiment
# is models roughly EQUAL overall but specialised by niche; that is the one
# regime where a scalar weight provably fails and a per-question weight might
# win. Putting Llama-3.3-70B next to gpt-5.4 restores a capability gap, at which
# point the weights just track overall skill and the run re-derives Corollary 3.3
# at full price. Six peer-tier models beat nine mismatched ones -- and six is the
# size of the completed TriviaQA experiment, so the correlation and
# minority-holds-truth numbers stay directly comparable.
# EVERY entry above needs (a) a Foundry deployment and (b) an AZ.PRICE entry.
#
# gpt-5.4 briefly answered with NO deployment, and that was a trap. Once the
# other models were deployed it began returning "Insufficient quota available
# for instant inference" / deployment_disabled. Deploymentless inference draws
# on a shared instant pool that is itself quota-backed, and provisioning real
# deployments appears to consume the headroom it was using. Treat "it answered
# without a deployment" as a coincidence, never as capacity you have.

MODELS = AZURE_MODELS if BACKEND == "azure" else OPENROUTER_MODELS
# generous, and identical for every model
#
# RAISED 2026-08-14 after preflight caught Kimi-K2.5 returning EMPTY on medqa and
# mmlupro even after the automatic 2x retry. Measured: at 1000 tokens it returns
# finish="length" with the full budget spent and content="" -- the v1 failure
# exactly. It needs ~2200 (medqa) and ~2700 (mmlupro) to produce an answer.
#
# Raising a cap costs nothing for models that do not reach it: billing is on
# tokens used, not tokens allowed. Claude (365 out) and gpt-5.4 (116 out) are
# unaffected, so the OpenRouter budget math is unchanged. Only Kimi's Azure cost
# moves, and that is grant credit.
#
# RAISED AGAIN 2026-08-14: Kimi-K2.5 still hit finish="length" on hard MedQA
# vignettes at 4000 (and at 12000). It answers at ~8,600 completion tokens, so
# medqa needs real headroom. Cohere and MAI empties are NOT budget-related --
# see the alternate-field recovery in azure_backend.chat.
BUDGET = {"triviaqa": 4000, "medqa": 16000, "gsm8k": 8000, "mmlupro": 8000}
MAXW = 10
_lk = threading.Lock()
_st = {"empty_retried": 0, "still_empty": 0, "err": 0, "truncated": 0, "aborted": False}


def spend():
    """Azure has no usage endpoint -- AZ.spend() reconstructs from token counts.

    Mixed pool: Azure's figure starts at 0 for this process, OpenRouter's is
    lifetime account usage. The sum is still a valid cap basis because run()
    only ever compares it against its own starting value.
    """
    if BACKEND == "azure":
        return AZ.spend() + (OR.spend() if any("/" in m for m in MODELS) else 0.0)
    return requests.get("https://openrouter.ai/api/v1/key",
                        headers={"Authorization": f"Bearer {KEY}"},
                        timeout=30).json()["data"]["usage"]


def ask(model, item, mult=1, retries=3):
    dom = item["domain"].split(":")[0]
    mt = BUDGET[dom] * mult

    if BACKEND == "azure":
        # mixed pool: "/" in the id means an OpenRouter model, everything else is
        # an Azure deployment name. Azure has no quota for Anthropic/Mistral/Meta.
        mod = OR if "/" in model else AZ
        txt, meta = mod.chat(model, B.PROMPT, item["question"], mt, temp=0,
                             retries=retries)
        if txt is None:
            with _lk: _st["err"] += 1
            return None
        # v1's failure had to be inferred from an empty string. Azure names it:
        # finish_reason=="length" with reasoning tokens spent means the budget
        # went to internal reasoning and never reached an answer.
        if not txt and mult == 1:
            with _lk:
                _st["empty_retried"] += 1
                if meta.get("finish") == "length":
                    _st["truncated"] += 1
            return ask(model, item, mult=2, retries=2)
        if not txt:
            with _lk: _st["still_empty"] += 1
        return txt

    body = {"model": model, "temperature": 0, "max_tokens": mt,
            "messages": [{"role": "system", "content": B.PROMPT},
                         {"role": "user", "content": item["question"]}]}
    for a in range(retries):
        try:
            r = requests.post(URL, timeout=180,
                              headers={"Authorization": f"Bearer {KEY}"}, json=body)
            if r.status_code == 429:
                time.sleep(2 ** a + 1); continue
            r.raise_for_status(); j = r.json()
            if "choices" not in j:
                time.sleep(1.5 ** a); continue
            txt = (j["choices"][0]["message"].get("content") or "").strip()
            if not txt and mult == 1:                 # empty -> one retry at 2x
                with _lk: _st["empty_retried"] += 1
                return ask(model, item, mult=2, retries=2)
            if not txt:
                with _lk: _st["still_empty"] += 1
            return txt
        except Exception:
            time.sleep(1.5 ** a)
    with _lk: _st["err"] += 1
    return None


def preflight(per_dom_sample=1):
    """Every model x every domain must return non-empty content before we spend."""
    if BACKEND == "azure":
        # Gate 0, new on Azure: catalogued != deployed != priced. A model can be
        # listed in the region, have no deployment, and fail every call -- which
        # looks exactly like incompetence in the results table.
        dep = AZ.deployments()
        print(f"  deployments provisioned: {dep if dep else 'NONE'}")
        # OpenRouter models are priced from its own API, not AZ.PRICE
        missing = [m for m in MODELS if "/" not in m and m not in AZ.PRICE]
        missing += [m for m in MODELS if "/" in m and m not in OR.prices()]
        if missing:
            print(f"  !! no PRICE entry for {missing} -- the spend cap cannot bound them")
        print("  reachability:")
        az_only = [m for m in MODELS if "/" not in m]
        or_only = [m for m in MODELS if "/" in m]
        rows = AZ.reachable(az_only)
        for m in or_only:                       # OpenRouter half of a mixed pool
            txt, meta = OR.chat(m, None, "What is the capital of Australia? "
                                          "Answer with just the city.", 400)
            rows.append({"model": m, "ok": bool(txt),
                         "words": len(txt.split()) if txt else 0,
                         "reasoning": meta.get("reasoning"), "finish": meta.get("finish"),
                         "temp0": meta.get("temp0"), "error": meta.get("error")})
        for r in rows:
            flag = "" if r["ok"] else "   <-- NOT CALLABLE (needs a deployment)"
            # only meaningful for a model that answered -- an unreachable model
            # reports temp0=None, which must not be read as a temperature refusal
            t0 = ("   <-- REJECTS temperature=0, exclude from pool"
                  if r["ok"] and not r["temp0"] else "")
            print(f"    {r['model']:<40}{str(r['ok']):>6}{flag}{t0}")
            if r["error"]:
                print(f"        {r['error']}")
        print()

    items = B.load_all(6, overrides={"mmlupro": 12})
    doms = sorted({i["domain"].split(":")[0] for i in items})
    probe = {d: next(i for i in items if i["domain"].startswith(d)) for d in doms}
    print(f"  probing {len(MODELS)} models x {len(doms)} domains = "
          f"{len(MODELS)*len(doms)} calls\n")
    print(f"  {'model':<32}" + "".join(f"{d[:9]:>11}" for d in doms))
    print("  " + "-" * (32 + 11 * len(doms)))
    bad = []
    for m in MODELS:
        cells = []
        with ThreadPoolExecutor(max_workers=len(doms)) as ex:
            futs = {ex.submit(ask, m, probe[d]): d for d in doms}
            got = {futs[f]: f.result() for f in as_completed(futs)}
        for d in doms:
            t = got.get(d)
            cells.append("EMPTY" if not t else f"{len(t.split())}w")
            if not t: bad.append((m, d))
        print(f"  {m:<32}" + "".join(f"{c:>11}" for c in cells))
    if bad:
        print(f"\n  !! {len(bad)} model/domain cells returned EMPTY:")
        for m, d in bad: print(f"       {m}  /  {d}")
        print("  These would score a fake zero. Exclude them or raise their budget.")
    else:
        print("\n  all cells non-empty -- safe to run")
    return bad


def run(per, cap, or_cap=None):
    """cap bounds TOTAL spend; or_cap separately bounds the OpenRouter half.

    These are different currencies in practice. Azure is grant credit (~$1000
    available); OpenRouter is real money (~$10). A single combined cap is
    dominated by the plentiful side and would let the scarce one run dry
    unnoticed -- so the OpenRouter delta is tracked and enforced on its own.
    """
    items = B.load_all(per, overrides={"mmlupro": per * 2})
    path = f"{HERE}/data/v2.jsonl"
    done = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l); done.add((r["qid"], r["model"]))
            except Exception: pass
    jobs = [(it, m) for it in items for m in MODELS if (it["qid"], m) not in done]
    st = spend()
    or_st = OR.spend() if BACKEND == "azure" else 0.0
    print(f"  {len(items)} questions x {len(MODELS)} models")
    print(f"  {len(done)} cached, {len(jobs)} to run")
    print(f"  spend now ${st:.4f}, hard cap ${st+cap:.4f}")
    if or_cap is not None:
        print(f"  OpenRouter now ${or_st:.4f}, its own cap +${or_cap:.2f} "
              f"(real money, not grant credit)")
    print()
    f = open(path, "a"); t0 = time.time(); limit = st + cap
    with ThreadPoolExecutor(max_workers=MAXW) as ex:
        futs = {ex.submit(ask, m, it): (it, m) for it, m in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            it, m = futs[fu]; txt = fu.result()
            # `if txt` not `if txt is not None` -- an empty response must be a
            # MISSING cell, never a wrong one. Writing resp="" grades as
            # incorrect, which is indistinguishable in the output file from a
            # model that answered and got it wrong. That is precisely how v1
            # manufactured a 67-point capability spread out of nothing.
            # Left unwritten, the cell is retried on resume, and if it never
            # resolves complete_matrix() drops the question from the paired
            # comparison instead of scoring the model zero for it.
            if txt:
                with _lk:
                    f.write(json.dumps(dict(qid=it["qid"], domain=it["domain"], model=m,
                        resp=txt, correct=B.grade(B.final_line(txt), it),
                        gold=it["gold"]))+"\n"); f.flush()
            # check often when real money is on the line: 200 calls is ~$1.20 of
            # OpenRouter spend, too coarse a grain against a $10 balance
            every = 25 if or_cap is not None else 200
            if i % every == 0 or i == len(jobs):
                cur = spend()
                or_used = (OR.spend() - or_st) if BACKEND == "azure" else 0.0
                thr = (f"  throttled {AZ.usage['throttled']}  dropped {AZ.usage['dropped']}"
                       if BACKEND == "azure" else "")
                ors = f"  OR +${or_used:.2f}" if or_cap is not None else ""
                print(f"    {i}/{len(jobs)}  {time.time()-t0:5.0f}s  ${cur:.3f}  "
                      f"empty-retried {_st['empty_retried']}  still-empty {_st['still_empty']}"
                      + ors + thr)
                if or_cap is not None and or_used > or_cap:
                    print(f"    !! OPENROUTER CAP HIT (+${or_used:.2f} > +${or_cap:.2f}) "
                          f"-- stopping. Azure rows already written are still valid; "
                          f"re-run after topping up and it resumes.")
                    _st["aborted"] = True
                    for g in futs: g.cancel()
                    break
                if cur > limit:
                    print(f"    !! SPEND CAP HIT (${cur:.3f} > ${limit:.3f}) -- stopping")
                    _st["aborted"] = True
                    for g in futs: g.cancel()
                    break
    f.close()
    print(f"\n  final spend ${spend():.4f}   aborted={_st['aborted']}")
    return path


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preflight"
    if cmd == "preflight":
        preflight()
    else:
        # run <per-domain> <total-cap> [openrouter-cap]
        per = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        cap = float(sys.argv[3]) if len(sys.argv) > 3 else 2.40
        or_cap = float(sys.argv[4]) if len(sys.argv) > 4 else None
        run(per, cap, or_cap)
