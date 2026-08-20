"""
Azure AI Foundry transport, drop-in for the OpenRouter one in run_v2.py.

WHY THIS IS NOT THE 15-LINE SWAP THE HANDOFF PREDICTED
  Foundry speaks an OpenAI-shaped API, but three things differ in ways that
  reproduce, exactly, the class of silent failure that killed run v1:

  1. DEPLOYMENTLESS INFERENCE IS OPENAI-ONLY. Probed 2026-08-13: gpt-5.x answers
     straight off the project endpoint. Claude, Grok, DeepSeek, Kimi, Mistral,
     Cohere, Llama, qwen, o-series, gpt-4.1 and gpt-oss all return
     400 "does not support deploymentless inference" until a deployment exists.
     The project currently has ZERO deployments. Catalogue presence != callable.

  2. max_tokens IS REJECTED by gpt-5.x -- it wants max_completion_tokens. A
     naive port raises 400 on every call, which at least fails loudly. Worse is
     the near-miss: reasoning models still burn the budget on reasoning tokens
     and return content="". That is v1 again. Azure, unlike OpenRouter, reports
     usage.completion_tokens_details.reasoning_tokens and finish_reason, so the
     failure is now DETECTABLE rather than inferred. We check both, every call.

  3. temperature=0 IS REJECTED by gpt-5.5 and gpt-5.6-*. Silently dropping it
     would run those models at temperature 1 while their peers run at 0 -- a
     confound sitting directly on top of the quantity being measured. We record
     it and let the caller refuse the model instead.

  Quirks are DISCOVERED, not hardcoded: a 400 naming a parameter is parsed and
  retried once with that parameter adjusted, and the resolution is cached per
  model. Azure changes this surface faster than any table we would maintain.

NO SPEND ENDPOINT
  OpenRouter had GET /api/v1/key, which is what run_v2's hard cap read. Foundry
  has no data-plane equivalent -- billing is subscription-level and lagged. The
  cap here is reconstructed locally from reported token usage x PRICE. A model
  with no PRICE entry cannot be run: unpriced tokens would make the guard a
  decoration, and the last run died on a 6x cost surprise.
"""
import os, json, time, threading
import requests

from open_ended_aggregation.paths import ROOT as _ROOT
HERE = str(_ROOT)
def _env(key):
    for line in open(f"{HERE}/.env"):
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{key} missing from .env")


# project endpoint -> OpenAI-shaped data plane on the same host
BASE = _env("AZURE_ENDPOINT").split("/api/projects/")[0].rstrip("/") + "/openai/v1"
KEY = _env("AZURE_API_KEY")
URL = f"{BASE}/chat/completions"

# $ per 1M tokens (prompt, completion). Reasoning tokens bill as completion.
#
# Pulled 2026-08-13 from the Azure Retail Prices API (public, no auth):
#   https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Foundry Models'
# GlobalStandard meters only -- NOT batch, NOT cached-input, NOT long-context,
# NOT provisioned. Re-pull rather than trusting these if the run is months later.
#
# UNITS ARE NOT UNIFORM AND THIS IS A TRAP. Grok, DeepSeek and Kimi publish per
# 1K tokens; Cohere and OpenAI per 1M. Reading the raw retailPrice without
# checking unitOfMeasure gives a spend guard wrong by 1000x -- in the direction
# that lets a run overspend silently. Everything below is normalised to $/1M.
PRICE = {
    "grok-4.3":                      (1.25,  2.50),   # 0.00125/0.0025 per 1K
    "DeepSeek-V4-Flash-2026-04-23":  (0.19,  0.51),   # 0.00019/0.00051 per 1K
    "Kimi-K2.5":                     (0.60,  3.00),   # 0.0006/0.003 per 1K
    "Cohere-command-a-plus-05-2026": (0.80,  4.00),   # already per 1M
    "gpt-5.4":                       (2.50, 15.00),   # already per 1M -- priciest in the pool
    # MAI-Thinking-1: NO published text-token meter in the retail API. The only
    # MAI entries are image models and MAI-DS-R1. Possibly free in preview, but
    # unverified -- left out on purpose so preflight refuses to pass. Check the
    # "View pricing" link on the model card in the portal before running it.
}

_lk = threading.Lock()
_quirk = {}                      # model -> {"budget": str, "temp0": bool}
usage = {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0, "cost": 0.0,
         "empty": 0, "truncated": 0, "throttled": 0, "throttle_wait": 0.0,
         "dropped": 0, "recovered_alt": 0, "unpriced": set()}


def _body(model, system, user, max_out, temp):
    q = _quirk.get(model, {})
    b = {"model": model,
         "messages": ([{"role": "system", "content": system}] if system else [])
                     + [{"role": "user", "content": user}],
         q.get("budget", "max_completion_tokens"): max_out}
    if temp is not None and q.get("temp0", True):
        b["temperature"] = temp
    return b


def _adapt(model, err):
    """Parse a 400 into a cached quirk. Returns True if worth retrying."""
    msg = (err.get("message") or "").lower()
    param = (err.get("param") or "").lower()
    with _lk:
        q = _quirk.setdefault(model, {})
        if "max_tokens" in param or "max_completion_tokens" in msg:
            q["budget"] = "max_completion_tokens"
            return True
        if "max_completion_tokens" in param or "'max_tokens' instead" in msg:
            q["budget"] = "max_tokens"
            return True
        if "temperature" in param or "temperature" in msg:
            q["temp0"] = False          # caller decides whether that is acceptable
            return True
    return False


def chat(model, system, user, max_out, temp=0, retries=3, throttle_retries=8):
    """Returns (text, meta). text is None only on hard failure.

    meta carries the things v1 had no way to see: reasoning token spend and
    whether the response was cut off mid-generation.

    429s get their OWN retry budget, deliberately much deeper than the error
    budget. Partner models on a grant subscription land at single-digit-thousand
    TPM, where sustained throttling is the normal operating condition rather than
    a blip. A shallow backoff turns that into dropped calls -- missing (qid,
    model) cells that look like nothing at all in the output file, which is the
    v1 failure wearing a different hat. Re-running `run` does backfill them, but
    only if you notice.
    """
    a = t = adapts = 0
    meta = {"error": "no attempt made"}
    while a < retries and t < throttle_retries:
        try:
            r = requests.post(URL, timeout=300,
                              headers={"Authorization": f"Bearer {KEY}",
                                       "Content-Type": "application/json"},
                              json=_body(model, system, user, max_out, temp))
            if r.status_code == 429:
                # Azure states the wait; guessing it is how you get a 429 storm
                wait = float(r.headers.get("Retry-After") or min(60, 2 ** t + 1))
                with _lk:
                    usage["throttled"] += 1
                    usage["throttle_wait"] += wait
                t += 1
                time.sleep(wait)
                continue
            j = r.json()
            # bounded: a quirk that "adapts" but keeps 400ing must not spin forever
            if r.status_code == 400 and adapts < 3 and _adapt(model, j.get("error", {})):
                adapts += 1
                continue
            if r.status_code != 200:
                return None, {"error": str(j.get("error", j))[:200]}

            ch = (j.get("choices") or [{}])[0]
            m_ = ch.get("message", {}) or {}
            txt = (m_.get("content") or "").strip()

            # Some Foundry models put the answer somewhere other than `content`.
            # Measured 2026-08-14: Cohere-command-a-plus returns content="" with
            # 4,976 chars sitting in reasoning_content; MAI-Thinking-1 exposes
            # `thinking` and `reasoning` fields. Reading only `content` scored
            # those as silence -- a fake zero, which is the v1 failure.
            #
            # ONLY fall back when the model actually FINISHED. On finish="length"
            # the alternate field holds reasoning truncated mid-sentence, with no
            # answer in it; treating that as a response would invent a wrong
            # answer rather than record a missing one.
            if not txt and ch.get("finish_reason") == "stop":
                for alt in ("reasoning_content", "thinking", "reasoning"):
                    if (m_.get(alt) or "").strip():
                        txt = m_[alt].strip()
                        with _lk:
                            usage["recovered_alt"] += 1
                        break

            u = j.get("usage", {}) or {}
            det = u.get("completion_tokens_details") or {}
            pt = u.get("prompt_tokens", 0)
            ct = u.get("completion_tokens", 0)
            rt = det.get("reasoning_tokens", 0) or 0

            # BILLABLE OUTPUT != completion_tokens on this platform.
            # Measured on grok-4.3: total=207, prompt=21, completion=26,
            # reasoning=160 -- reasoning sits OUTSIDE completion_tokens and bills
            # as output. Summing completion alone undercounts a reasoning model
            # by ~8x, which is a spend guard that silently permits an overrun.
            # Deriving from total_tokens is correct under BOTH conventions: if
            # reasoning is already inside completion, total-prompt == completion.
            tt = u.get("total_tokens")
            out = (tt - pt) if (tt and tt > pt) else (ct + rt)

            meta = {"prompt": pt, "completion": ct, "reasoning": rt,
                    "billable_out": out,
                    "finish": ch.get("finish_reason"),
                    "temp0": _quirk.get(model, {}).get("temp0", True)}

            p = PRICE.get(model)
            with _lk:
                usage["calls"] += 1
                usage["prompt"] += meta["prompt"]
                usage["completion"] += meta["completion"]
                usage["reasoning"] += meta["reasoning"]
                if p:
                    usage["cost"] += (meta["prompt"] * p[0]
                                      + meta["billable_out"] * p[1]) / 1e6
                else:
                    usage["unpriced"].add(model)
                if not txt:
                    usage["empty"] += 1
                if meta["finish"] == "length":
                    usage["truncated"] += 1
            return txt, meta
        except Exception as e:
            meta = {"error": repr(e)[:200]}
            a += 1                      # the while loop will not do this for us
            time.sleep(1.5 ** a)
    if t >= throttle_retries:
        meta = {"error": f"gave up after {t} consecutive 429s -- quota too tight"}
    with _lk:
        usage["dropped"] += 1
    return None, meta


def spend():
    """Local reconstruction. Foundry exposes no usage endpoint -- see docstring."""
    return usage["cost"]


def require_prices(models):
    missing = [m for m in models if m not in PRICE]
    if missing:
        raise SystemExit("  no PRICE entry for: " + ", ".join(missing)
                         + "\n  the spend cap cannot bound an unpriced run -- add them first")


def deployments():
    """What is actually callable. Empty list means nothing has been provisioned."""
    ep = _env("AZURE_ENDPOINT").rstrip("/")
    r = requests.get(f"{ep}/deployments?api-version=2025-05-01",
                     headers={"api-key": KEY}, timeout=60)
    return [d.get("name") for d in r.json().get("value", [])]


def catalogue(chat_only=True):
    """Models the region lists. Presence here does NOT imply callable -- see #1."""
    r = requests.get(f"{BASE}/models",
                     headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
    out = []
    for m in r.json().get("data", []):
        if m.get("lifecycle_status") == "deprecated":
            continue
        if chat_only and not (m.get("capabilities") or {}).get("chat_completion"):
            continue
        out.append(m["id"])
    return sorted(set(out))


def reachable(models, probe="What is the capital of Australia? Answer with just the city."):
    """One cheap call each. Separates 'deployed' from 'catalogued' before spending.

    This is the check that would have caught all three of v1/v2's silent
    failures -- blocked models, invalid IDs, empty responses -- in one pass.
    """
    rows = []
    for m in models:
        txt, meta = chat(m, None, probe, 400)
        rows.append({"model": m,
                     "ok": bool(txt),
                     "words": len(txt.split()) if txt else 0,
                     "reasoning": meta.get("reasoning"),
                     "finish": meta.get("finish"),
                     "temp0": meta.get("temp0"),
                     "error": meta.get("error")})
    return rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "catalogue":
        for m in catalogue():
            print(" ", m)
    else:
        d = deployments()
        print(f"  deployments: {d if d else 'NONE — nothing is provisioned'}")
        pool = sys.argv[1:] or ["gpt-5.4-mini"]
        print(f"  {'model':<30}{'ok':>5}{'words':>7}{'reason':>8}{'finish':>10}{'temp0':>7}")
        print("  " + "-" * 67)
        for r in reachable(pool):
            print(f"  {r['model']:<30}{str(r['ok']):>5}{r['words']:>7}"
                  f"{str(r['reasoning']):>8}{str(r['finish']):>10}{str(r['temp0']):>7}"
                  + (f"\n      {r['error']}" if r['error'] else ""))
        print(f"\n  reconstructed spend ${spend():.4f} over {usage['calls']} calls"
              + (f"   UNPRICED: {sorted(usage['unpriced'])}" if usage['unpriced'] else ""))
        if usage["throttled"] or usage["dropped"]:
            print(f"  throttled {usage['throttled']}x "
                  f"({usage['throttle_wait']:.0f}s waiting), dropped {usage['dropped']}")
