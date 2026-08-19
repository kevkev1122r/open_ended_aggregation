"""
OpenRouter transport, same interface as azure_backend.chat().

WHY BOTH AT ONCE
  The pool does not have to be single-provider, and after the Azure quota cuts it
  cannot be: Anthropic, Mistral and Meta are all 0/0 TPM there. Models are routed
  by ID shape -- anything containing "/" is an OpenRouter ID
  ("anthropic/claude-sonnet-5"), everything else is an Azure deployment name.

  Mixing providers is methodologically fine here because what must match across
  models is the SAMPLING REGIME (temperature 0, same system prompt, same token
  budget), not the vendor. Anthropic's own serving is arguably a cleaner source
  for Claude than a third-party copy.

ONE REAL ADVANTAGE OVER AZURE
  OpenRouter publishes per-model pricing on the same API it serves from, so the
  spend guard is exact and self-updating. Azure's prices live in a separate
  retail-pricing API keyed on meter names ("4.3 Outp Glbl Tokens") that must be
  matched by hand and normalised across per-1K and per-1M units -- see AZ.PRICE.
"""
import os, time, threading
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = [l.split("=", 1)[1].strip() for l in open(f"{HERE}/.env")
       if l.startswith("OPENROUTER_API_KEY=")][0]
URL = "https://openrouter.ai/api/v1/chat/completions"

_lk = threading.Lock()
_price = {}
usage = {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0, "cost": 0.0,
         "empty": 0, "truncated": 0, "throttled": 0, "throttle_wait": 0.0,
         "dropped": 0, "unpriced": set()}


def prices():
    """$/token (prompt, completion), straight from OpenRouter. Cached per process."""
    if not _price:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=60)
        for m in r.json()["data"]:
            p = m.get("pricing") or {}
            try:
                _price[m["id"]] = (float(p.get("prompt", 0)), float(p.get("completion", 0)))
            except (TypeError, ValueError):
                pass
    return _price


def chat(model, system, user, max_out, temp=0, retries=3, throttle_retries=8):
    """Mirrors azure_backend.chat -- returns (text, meta)."""
    body = {"model": model, "temperature": temp, "max_tokens": max_out,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": user}]}
    a = t = 0
    meta = {"error": "no attempt made"}
    while a < retries and t < throttle_retries:
        try:
            r = requests.post(URL, timeout=300,
                              headers={"Authorization": f"Bearer {KEY}"}, json=body)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After") or min(60, 2 ** t + 1))
                with _lk:
                    usage["throttled"] += 1
                    usage["throttle_wait"] += wait
                t += 1
                time.sleep(wait)
                continue
            j = r.json()
            if r.status_code != 200 or "choices" not in j:
                meta = {"error": str(j.get("error", j))[:200]}
                a += 1
                time.sleep(1.5 ** a)
                continue

            ch = j["choices"][0]
            txt = (ch.get("message", {}).get("content") or "").strip()
            u = j.get("usage", {}) or {}
            det = u.get("completion_tokens_details") or {}
            pt = u.get("prompt_tokens", 0)
            ct = u.get("completion_tokens", 0)
            rt = det.get("reasoning_tokens", 0) or 0
            # see azure_backend: reasoning tokens may sit outside completion_tokens
            # and still bill as output. total-prompt is correct either way.
            tt = u.get("total_tokens")
            out = (tt - pt) if (tt and tt > pt) else (ct + rt)
            meta = {"prompt": pt, "completion": ct, "reasoning": rt,
                    "billable_out": out,
                    "finish": ch.get("finish_reason"), "temp0": True}

            p = prices().get(model)
            with _lk:
                usage["calls"] += 1
                usage["prompt"] += meta["prompt"]
                usage["completion"] += meta["completion"]
                usage["reasoning"] += meta["reasoning"]
                if p:
                    usage["cost"] += meta["prompt"] * p[0] + meta["billable_out"] * p[1]
                else:
                    usage["unpriced"].add(model)
                if not txt:
                    usage["empty"] += 1
                if meta["finish"] == "length":
                    usage["truncated"] += 1
            return txt, meta
        except Exception as e:
            meta = {"error": repr(e)[:200]}
            a += 1
            time.sleep(1.5 ** a)
    with _lk:
        usage["dropped"] += 1
    return None, meta


def spend():
    """Authoritative, unlike Azure: OpenRouter reports account usage directly."""
    try:
        return requests.get("https://openrouter.ai/api/v1/key",
                            headers={"Authorization": f"Bearer {KEY}"},
                            timeout=30).json()["data"]["usage"]
    except Exception:
        return usage["cost"]
