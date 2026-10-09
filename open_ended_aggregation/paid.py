"""
One safety layer for every paid LLM call made by the new modules.

RULES (Kevin, 22 Sept 2026: "ask me first and think it through twice")
  - Nothing is sent unless the caller was created with confirm=True, which the
    runners set only from an explicit --confirm flag.
  - Every paid response is appended to a cache file and reused on any rerun, so
    a crash, an abort or a rescoring never pays twice. Cache files are a paid
    cache: never delete them to "start clean".
  - A hard cap (USD, new calls this process) aborts BEFORE the next call once
    reached.
  - estimate() prints a cost from live OpenRouter prices before anything runs,
    using the token rate measured on the QAMPARI pilot.

Only OpenRouter model ids ("vendor/model") are allowed here. The Azure
deployments bill a personal card and are not routed through this layer.
"""
import hashlib
import json
import threading

from open_ended_aggregation.backends import openrouter

TOK_PER_CHAR = 0.292     # measured on the pilot prompts (chars/4 undercounts by ~17%)
OUT_TYPICAL = 800        # measured mean billable output per call at effort=low
MAX_OUT = 3000
BROKEN = ("error", "length")   # a partial answer is never cached or reused


class NotConfirmed(RuntimeError):
    pass


class CapReached(RuntimeError):
    pass


def key_of(model, effort, msg, extra=None):
    blob = f"{model}\x00{effort}\x00{msg}"
    if extra:
        blob += "\x00" + json.dumps(extra, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class PaidCaller:
    def __init__(self, cache_path, model, effort="low", cap=1.0, confirm=False,
                 json_mode=False, max_out=MAX_OUT):
        if "/" not in model:
            raise ValueError(f"{model}: only OpenRouter ids are allowed in PaidCaller")
        self.cache_path, self.model, self.effort = cache_path, model, effort
        self.cap, self.confirm, self.max_out = cap, confirm, max_out
        self.json_mode = json_mode
        self.cache = {}
        if cache_path.exists():
            for ln in open(cache_path):
                try:
                    o = json.loads(ln)
                except Exception:
                    continue
                if (o.get("meta") or {}).get("finish") in BROKEN:
                    continue              # kept in the file, never reused
                self.cache[o["key"]] = o
        self.n_cached = self.n_new = self.n_broken = 0
        self.cost = 0.0
        self._lk = threading.Lock()   # callers may run questions in threads

    # -------------------------------------------------------------- estimate
    def price(self):
        p = openrouter.prices().get(self.model)
        if p is None:
            raise SystemExit(f"  {self.model} is not a known OpenRouter model id -- nothing run.")
        return p

    def estimate(self, msgs, label=""):
        """Cost of the msgs NOT yet cached, at typical and worst-case output.
        Each msg is a user string or a (system, user) pair."""
        pin, pout = self.price()
        pairs = [(None, m) if isinstance(m, str) else m for m in msgs]
        todo = [(sy, u) for sy, u in pairs if self._key(u, sy) not in self.cache]
        tin = sum(len(u) + len(sy or "") for sy, u in todo) * TOK_PER_CHAR
        typ = tin * pin + len(todo) * OUT_TYPICAL * pout
        worst = tin * pin + len(todo) * self.max_out * pout
        print(f"  ESTIMATE {label}: {len(todo)} new calls ({len(msgs) - len(todo)} cached), "
              f"~{tin/1e6:.2f}M input tokens -> ~${typ:.2f} typical, ${worst:.2f} worst case; "
              f"cap ${self.cap:.2f}; {self.model}, effort {self.effort}")
        return typ, worst

    # -------------------------------------------------------------- call
    def _extra(self):
        e = {"reasoning": {"effort": self.effort}}
        if self.json_mode:
            e["response_format"] = {"type": "json_object"}
        return e

    def _key(self, msg, system=None):
        extra = {"json": self.json_mode}
        if system:
            extra["system"] = system
        return key_of(self.model, self.effort, msg, extra)

    def __call__(self, msg, tag=None, system=None):
        """Returns (text or None, meta). Cached responses are free. `system` is
        sent as a system message and is part of the cache key."""
        k = self._key(msg, system)
        if k in self.cache:
            self.n_cached += 1
            return self.cache[k]["text"], self.cache[k].get("meta", {})
        if not self.confirm:
            raise NotConfirmed("paid call attempted without --confirm; nothing was sent")
        if self.cost >= self.cap:
            raise CapReached(f"spend cap ${self.cap:.2f} reached "
                             f"(${self.cost:.2f}); paid responses are cached "
                             f"in {self.cache_path.name}")
        txt, meta = openrouter.chat(self.model, system, msg, self.max_out, extra=self._extra())
        pin, pout = self.price()
        with self._lk:   # every billed call counts, usable or not
            self.cost += (meta.get("prompt") or 0) * pin + (meta.get("billable_out") or 0) * pout
        if txt and meta.get("finish") in BROKEN:
            self.n_broken += 1
            return None, meta             # truncated or errored: not usable, not cached
        if txt:
            rec = {"key": k, "model": self.model, "effort": self.effort, "tag": tag,
                   "text": txt, "meta": meta}
            with self._lk:
                self.cache[k] = rec
                with open(self.cache_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                self.n_new += 1
        return txt, meta

    def spent(self):
        """USD billed by THIS caller in this process (a composer and a judge in
        one process each have their own cap)."""
        return self.cost
