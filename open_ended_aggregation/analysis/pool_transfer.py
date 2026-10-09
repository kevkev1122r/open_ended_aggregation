"""
Does the scorer transfer to agents it has never seen?   (HANDOFF_2026-09-06 Q1)

THE QUESTION
  The paper's scorer is parameterised by agent identity: ~81 of 92 features are
  per-agent slots (indicators, 8x7 rank buckets, a 2^8 pattern lookup). Swap the
  pool and it is undefined. Every transfer result so far (LOSO, the 210 subsets)
  kept the same eight agents, so none of them answers this.

  This file replaces every identity-indexed feature with a pool-invariant one
  and asks two separate questions:

    cost of invariance   invariant scorer fitted IN-POOL vs the paper scorer
                         fitted in-pool. What does dropping identity cost?
    cost of transfer     invariant scorer fitted on a DISJOINT pool (and, in
                         part 2, a different dataset) vs the same scorer fitted
                         in-pool. What does never seeing these agents cost?

  Zero-label on the target is the standard the final method must meet, so
  every "transfer" arm uses no target gold for anything: not the weights, not
  the threshold, not the gold-count predictor, not agent reliability.

POOL-INVARIANT FEATURES (per claim; k = pool size)
  support        |S|/k, and |S| / max support on this question
  rank           log rank and rank-as-fraction-of-list, min and mean over supporters
  verbosity      mean log list length of the supporters
  omission       silent agents' log list lengths: mean over the pool, and max
  question       log #candidates, mean support fraction on this question
  reliability    (optional) supporters' mean and max relative reliability, and
                 reliability-weighted silence. Reliability is log(r_j / mean r),
                 so its scale is the same for any pool.

RELIABILITY, three ways
  none   no reliability features at all
  agr    LABEL-FREE: mean fraction of the other agents that also assert agent j's
         claims, over every target question. No gold.
  sup    supervised precision on the training questions (DEV-ONLY scaffold:
         on the target it uses target labels)
  ds     LABEL-FREE two-coin Dawid-Skene log-likelihood ratio, one feature
  self   LABEL-FREE: no reliability at all; rank and verbosity measured against
         each agent's own median list length. Best mode (cut_rules.py E2).

DECISION RULES, all zero-label on the target
  thr        global probability threshold fitted on the source
  bud-src    per-question budget k*(q) with G-hat from a ridge fitted on the source
  bud-sum    per-question budget with G-hat = sum of p on the question -- the
             plug-in expected-F1 rule, no second model, needs only calibrated p

PART 1  QAMPARI, all 35 splits of the 8 agents into two disjoint pools of 4,
        both directions (70 transfers), 2-fold over questions so the source and
        target never share a question either.
PART 2  across datasets: QAMPARI -> CWQ and -> WebQSP, from (a) the 5 Azure agents
        (dataset shift only), (b) the 3 gpt-5.4 agents (dataset AND pool shift,
        pools fully disjoint), (c) all 8. CWQ -> QAMPARI(Azure 5) reversed.

Cached generations only -- CPU, zero API spend.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.pool_transfer
"""
import json, math, random, argparse, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.beyond_pattern import fit_logistic
from open_ended_aggregation.analysis.rank_crossover import Scorer, run_ensemble, tensors
from open_ended_aggregation.analysis.weighting_v2 import fit_gold_ridge, adaptive_keep
from open_ended_aggregation.benchmarks import qampari as QA
from open_ended_aggregation.paths import DATA, RESULTS

SEED = 0
AZURE = ["Cohere-command-a-plus-05-2026", "DeepSeek-V4-Flash", "Kimi-K2.5",
         "MAI-Thinking-1", "grok-4.3"]
GPT = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]

DATASETS = {
    "qampari": ("qampari_asc800.jsonl", None),
    "cwq":     ("cwq_gen.jsonl", "cwq_items.json"),
    "webqsp":  ("webqsp_gen.jsonl", "webqsp_items.json"),
}


# ------------------------------------------------------------------ data
def load(name, pool, with_keys=False):
    """Same construction as beyond_pattern.load, for any list dataset.

    One row per (question, distinct normalised claim) with per-model rank and
    per-model list length. A question is kept only if every model in `pool`
    answered it, so pools are compared on identical questions."""
    gen, items_file = DATASETS[name]
    byq = collections.defaultdict(dict)
    for l in open(DATA / gen, errors="ignore"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        byq[r["qid"]].setdefault(r["model"], r)
    if items_file is None:
        items = {it["qid"]: it for it in QA.load_items(100000)}
    else:
        items = {it["qid"]: it for it in json.load(open(DATA / items_file))}
    models = sorted(pool)
    qids = sorted(q for q, d in byq.items() if set(models) <= set(d) and q in items)

    cand, ngold, nlist, keys = {}, {}, {}, {}
    for q in qids:
        gold = [{QA.norm(a) for a in [g["answer_text"]] + list(g.get("aliases") or [])
                 if str(a).strip()} for g in items[q]["gold"]]
        ngold[q] = len(gold)
        rank, L = collections.defaultdict(dict), {}
        for m in models:
            seen = 0
            for it in byq[q][m].get("items") or []:
                k = QA.norm(it)
                if not k:
                    continue
                if m not in rank[k]:
                    rank[k][m] = seen
                seen += 1
            L[m] = max(1, seen)
        nlist[q] = L
        cand[q] = [(frozenset(rk), next((i for i, gs in enumerate(gold) if k in gs), None),
                    dict(rk)) for k, rk in rank.items()]
        keys[q] = list(rank)
    if with_keys:
        return models, qids, cand, ngold, nlist, keys
    return models, qids, cand, ngold, nlist


def view(T, S):
    """Restrict the full tensors to the agents in S (column indices)."""
    Z, RK, LL, rowq, rowgi, ng = T
    z = Z[:, S]
    keep = z.any(1)
    return dict(z=z[keep], rk=RK[np.ix_(keep, S)].astype(float),
                ll=LL[np.ix_(keep, S)].astype(float), rq=rowq[keep],
                rg=rowgi[keep], y=(rowgi[keep] >= 0).astype(float),
                ng=ng, nq=len(ng), k=len(S))


# ------------------------------------------------------------------ reliability
def rel_agreement(v):
    """LABEL-FREE: for each agent, the mean fraction of the OTHER agents that
    also assert its claims. Uses no gold at all."""
    z, k = v["z"], v["k"]
    others = (z.sum(1, keepdims=True) - 1) / max(1, k - 1)
    r = np.array([others[z[:, j], 0].mean() if z[:, j].any() else 0.0
                  for j in range(k)])
    return np.maximum(r, 1e-3)


def rel_supervised(v, qmask):
    """DEV-ONLY: smoothed claim precision on the questions in qmask."""
    z, y = v["z"], v["y"]
    m = qmask[v["rq"]]
    return np.array([((y[m] * z[m, j]).sum() + 1) / (z[m, j].sum() + 2)
                     for j in range(v["k"])])


def ds_llr(v, iters=50):
    """LABEL-FREE two-coin Dawid-Skene over claims. Each agent j asserts a true
    claim with rate a_j and a false one with rate b_j; EM on the target's own
    outputs. Returns one log-likelihood ratio per claim. Its meaning is the same
    for any pool, so a scorer trained on 4 agents cannot use it to memorise
    which agent is which -- the failure of the 3-feature reliability block."""
    z = v["z"].astype(float)
    post = z.mean(1)
    for _ in range(iters):
        pt, pf = post.sum() + 1, (1 - post).sum() + 1
        a = np.clip((post[:, None] * z).sum(0) / pt, 1e-3, 1 - 1e-3)
        b = np.clip(((1 - post)[:, None] * z).sum(0) / pf, 1e-3, 1 - 1e-3)
        llr = z @ np.log(a / b) + (1 - z) @ np.log((1 - a) / (1 - b))
        pi = np.clip(post.mean(), 1e-3, 1 - 1e-3)
        new = 1 / (1 + np.exp(-np.clip(np.log(pi / (1 - pi)) + llr, -30, 30)))
        if np.max(np.abs(new - post)) < 1e-6:
            post = new
            break
        post = new
    return llr


def label_shift(p, pi_s, iters=100):
    """LABEL-FREE prior-shift correction (Saerens, Latinne & Decaestecker 2002).
    Re-estimates the target base rate by EM on the target's own predictions and
    rescales every posterior to it. Needs only the source base rate."""
    pi_t = pi_s
    for _ in range(iters):
        r1, r0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
        q = r1 * p / (r1 * p + r0 * (1 - p))
        new = float(np.clip(q.mean(), 1e-4, 1 - 1e-4))
        if abs(new - pi_t) < 1e-7:
            break
        pi_t = new
    r1, r0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
    return r1 * p / (r1 * p + r0 * (1 - p)), pi_t


# ------------------------------------------------------------------ features
def self_norm(v):
    """LABEL-FREE per-agent verbosity baseline: each agent's median log list
    length over every question in the view. Not a reliability estimate -- it
    lets 'rank 12' mean 'deep' for a terse agent and 'shallow' for a verbose
    one, which is part of what the identity features were buying."""
    rq, ll, nq = v["rq"], v["ll"], v["nq"]
    first = np.unique(rq, return_index=True)[1]          # one row per question
    return np.median(ll[first], axis=0)                  # (k,)


def inv_features(v, rel=None, llr=None, med=None):
    z, rk, ll, rq, k, nq = v["z"], v["rk"], v["ll"], v["rq"], v["k"], v["nq"]
    cnt = z.sum(1).astype(float)
    sil = ~z
    lenm = np.maximum(np.exp(ll), 1.0)
    big = 1e9
    lr = np.log1p(np.maximum(rk, 0))
    relr = rk / lenm
    ncs = np.maximum(cnt, 1)

    # per-question context: candidates, mean support, max support
    ncand = np.bincount(rq, minlength=nq).astype(float)
    msup = np.bincount(rq, weights=cnt / k, minlength=nq) / np.maximum(ncand, 1)
    mx = np.zeros(nq)
    np.maximum.at(mx, rq, cnt)

    cols = [np.ones(len(z)),
            cnt / k,
            cnt / np.maximum(mx[rq], 1),
            np.min(np.where(z, lr, big), 1) / 5.0,
            np.where(z, lr, 0).sum(1) / ncs / 5.0,
            np.min(np.where(z, relr, big), 1),
            np.where(z, relr, 0).sum(1) / ncs,
            np.where(z, ll, 0).sum(1) / ncs / 5.0,
            np.where(sil, ll, 0).sum(1) / k / 5.0,
            np.where(sil, ll, 0).max(1) / 5.0,
            np.log1p(ncand[rq]) / 5.0,
            msup[rq]]
    if rel is not None:
        R = np.log(rel / rel.mean())
        cols += [(z * R).sum(1) / ncs,
                 np.max(np.where(z, R, -big), 1),
                 (sil * R * ll).sum(1) / k / 5.0]
    if llr is not None:
        cols.append(np.clip(llr, -30, 30) / 10.0)
    if med is not None:
        mlen = np.exp(med)[None, :]
        dev = ll - med[None, :]                          # verbosity vs own norm
        rel_own = rk / mlen                              # rank vs own typical depth
        cols += [np.min(np.where(z, rel_own, big), 1),
                 np.where(z, rel_own, 0).sum(1) / ncs,
                 np.where(z, dev, 0).sum(1) / ncs,
                 np.where(sil, dev, 0).sum(1) / k,
                 np.where(sil, med[None, :], 0).sum(1) / k / 5.0]
    return np.column_stack(cols)


def qfeat(v, p):
    """Question-level features for the gold-count ridge. Pool-invariant."""
    z, rq, nq, k = v["z"], v["rq"], v["nq"], v["k"]
    cnt = z.sum(1)
    ncand = np.bincount(rq, minlength=nq).astype(float)
    mx = np.zeros(nq)
    np.maximum.at(mx, rq, cnt / k)
    multi = np.bincount(rq, weights=(cnt >= 2).astype(float), minlength=nq)
    ps = np.bincount(rq, weights=p, minlength=nq)
    return np.column_stack([np.ones(nq), np.log1p(ncand) / 5.0, mx,
                            np.log1p(multi) / 3.0, np.log1p(ps) / 3.0])


def sig(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))


def ece(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return 100 * e


# ------------------------------------------------------------------ fit / apply
class Fitted:
    """Everything learned on the source. Applied to a target it touches no gold."""

    def __init__(self, v, qmask, relmode):
        self.relmode = relmode
        rel = self._rel(v, qmask)
        F = inv_features(v, rel, ds_llr(v) if relmode == "ds" else None,
                         self_norm(v) if relmode == "self" else None)
        m = qmask[v["rq"]]
        self.pi_s = float(v["y"][m].mean())
        self.w = fit_logistic(F[m], v["y"][m])
        p = sig(F @ self.w)
        sc = Scorer(v["rq"], v["rg"], v["ng"], v["nq"])
        self.t = sc.best_threshold(p, qmask)
        self.t_cnt = sc.best_threshold(v["z"].sum(1) / v["k"], qmask)
        qi = np.nonzero(qmask)[0]
        # log-space G-hat: gold counts are heavy-tailed, and a linear ridge on
        # them is what the handoff's "log-space G-hat patch" fixes.
        self.beta = fit_gold_ridge(qfeat(v, p)[qi], np.log(v["ng"][qi]))

    def _rel(self, v, qmask):
        if self.relmode in ("none", "ds", "self"):
            return None
        if self.relmode == "agr":
            return rel_agreement(v)
        return rel_supervised(v, qmask)

    def apply(self, v, qmask_target_labels=None):
        """qmask_target_labels is only consulted by relmode='sup' (dev-only)."""
        if self.relmode == "sup":
            rel = rel_supervised(v, qmask_target_labels)
        else:
            rel = self._rel(v, None)
        p = sig(inv_features(v, rel, ds_llr(v) if self.relmode == "ds" else None,
                             self_norm(v) if self.relmode == "self" else None) @ self.w)
        sc = Scorer(v["rq"], v["rg"], v["ng"], v["nq"])
        out = {"thr": sc.f1_at(p, self.t),
               "cnt-thr": sc.f1_at(v["z"].sum(1) / v["k"], self.t_cnt)}
        G_src = np.exp(qfeat(v, p) @ self.beta)
        self.bud_mask = adaptive_keep(p, v["rq"], v["nq"], G_src)
        out["bud-src"] = sc.f1_of_mask(self.bud_mask)
        G_sum = np.maximum(np.bincount(v["rq"], weights=p, minlength=v["nq"]), 1.0)
        out["bud-sum"] = sc.f1_of_mask(adaptive_keep(p, v["rq"], v["nq"], G_sum))
        # label-shift corrected p: same scorer, base rate re-estimated on target
        q, self.pi_t = label_shift(p, self.pi_s)
        G_ls = np.exp(qfeat(v, q) @ self.beta)
        out["ls-bud-src"] = sc.f1_of_mask(adaptive_keep(q, v["rq"], v["nq"], G_ls))
        G_lsum = np.maximum(np.bincount(v["rq"], weights=q, minlength=v["nq"]), 1.0)
        out["ls-bud-sum"] = sc.f1_of_mask(adaptive_keep(q, v["rq"], v["nq"], G_lsum))
        out["ls-thr"] = sc.f1_at(q, self.t)
        self.q = q
        return out, p


def majority(v):
    sc = Scorer(v["rq"], v["rg"], v["ng"], v["nq"])
    return sc.f1_of_mask(v["z"].sum(1) / v["k"] > 0.5)


def boot(a, b, n=4000, seed=0):
    """Paired percentile CI on mean(a)-mean(b), resampling questions (x100)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    reps = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.quantile(reps, 0.025), 100 * np.quantile(reps, 0.975)


# ------------------------------------------------------------------ part 1
def part1(args):
    models, qids, cand, ngold, nlist = load("qampari", AZURE + GPT)
    T = tensors(models, qids, cand, ngold, nlist)
    nq = len(qids)
    print(f"\n  PART 1  QAMPARI {nq} q, pool of {len(models)}, disjoint 4|4 splits",
          flush=True)
    rng = random.Random(SEED)
    order = list(range(nq)); rng.shuffle(order)
    h = nq // 2
    folds = [(order[:h], order[h:]), (order[h:], order[:h])]

    sc_full = Scorer(T[3], T[4], T[5], nq)
    singles = {j: sc_full.f1_at(T[0][:, j].astype(float), 1.0) for j in range(len(models))}

    splits = []
    for A in itertools.combinations(range(len(models)), 4):
        B = tuple(j for j in range(len(models)) if j not in A)
        splits.append((A, B))
    if args.max_splits:
        splits = splits[:args.max_splits]

    rows = []
    for si, (A, B) in enumerate(splits):
        vA, vB = view(T, list(A)), view(T, list(B))
        ref = run_ensemble(list(B), *T, nq, folds)          # in-pool, uses B's gold
        arms = {}
        cal = collections.defaultdict(list)
        for tr, te in folds:
            trm = np.zeros(nq, bool); trm[tr] = True
            tem = np.zeros(nq, bool); tem[te] = True
            for relmode in ("none", "agr", "ds", "sup"):
                # in-pool invariant: fitted on B's training half (uses B gold)
                fin = Fitted(vB, trm, relmode)
                o_in, _ = fin.apply(vB, trm)
                # transfer: fitted on A's training half, applied to B's test half.
                # Pools disjoint, questions disjoint, no B gold unless relmode=sup.
                ftr = Fitted(vA, trm, relmode)
                o_tr, p = ftr.apply(vB, trm)
                m = tem[vB["rq"]]
                cal[relmode].append((p[m], vB["y"][m]))
                for dec, f in o_in.items():
                    arms.setdefault(f"in/{relmode}/{dec}", np.zeros(nq))[te] = f[te]
                for dec, f in o_tr.items():
                    arms.setdefault(f"tr/{relmode}/{dec}", np.zeros(nq))[te] = f[te]
        arms["majority"] = majority(vB)
        for k2, v2 in ref.items():
            arms[f"paper/{k2}"] = v2
        bj = max(B, key=lambda j: singles[j].mean())
        arms["best-single(oracle)"] = singles[bj]
        row = dict(A=[models[j] for j in A], B=[models[j] for j in B],
                   **{k2: 100 * v2.mean() for k2, v2 in arms.items()})
        for rm in ("none", "agr", "ds", "sup"):
            P = np.concatenate([c[0] for c in cal[rm]]); Y = np.concatenate([c[1] for c in cal[rm]])
            row[f"ece/{rm}"] = ece(P, Y)
        for k2 in ("tr/agr/bud-sum", "tr/agr/bud-src", "tr/none/bud-sum", "tr/none/bud-src",
                   "tr/ds/bud-src", "tr/ds/ls-bud-src", "tr/none/ls-bud-src",
                   "paper/rank+budget"):
            d, lo, hi = boot(arms[k2], arms["paper/count"])
            row[f"{k2}_vs_cnt"] = d; row[f"{k2}_sig"] = bool(lo > 0)
        d, lo, hi = boot(arms["tr/none/bud-src"], arms["paper/rank+budget"])
        row["tr_vs_paper"] = d; row["tr_vs_paper_ci"] = [lo, hi]
        rows.append(row)
        if (si + 1) % 10 == 0 or si + 1 == len(splits):
            print(f"    {si+1}/{len(splits)} splits", flush=True)

    # agreement vs supervised reliability, on the full pool: is the label-free
    # estimate even ordering the agents correctly?
    vall = view(T, list(range(len(models))))
    ra = rel_agreement(vall); rs = rel_supervised(vall, np.ones(nq, bool))
    rho = spearman(ra, rs)
    return rows, dict(models=models, agr=list(ra), sup=list(rs), spearman=rho)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------ part 2
def part2():
    print("\n  PART 2  across datasets (fit on ALL source q, test on ALL target q)",
          flush=True)
    cache = {}

    def get(name, pool):
        key = (name, tuple(sorted(pool)))
        if key not in cache:
            models, qids, cand, ngold, nlist = load(name, pool)
            T = tensors(models, qids, cand, ngold, nlist)
            cache[key] = (models, qids, T)
        return cache[key]

    jobs = [("qampari", AZURE, "cwq", AZURE, "dataset shift only"),
            ("qampari", GPT, "cwq", AZURE, "dataset + pool shift (disjoint)"),
            ("qampari", AZURE + GPT, "cwq", AZURE, "8 -> 5"),
            ("qampari", AZURE, "webqsp", AZURE, "dataset shift only"),
            ("qampari", GPT, "webqsp", AZURE, "dataset + pool shift (disjoint)"),
            ("qampari", AZURE + GPT, "webqsp", AZURE, "8 -> 5"),
            ("cwq", AZURE, "qampari", AZURE, "dataset shift only (reverse)"),
            ("cwq", AZURE, "webqsp", AZURE, "KG -> KG")]
    out = []
    for sname, spool, tname, tpool, label in jobs:
        sm, sq, ST = get(sname, spool)
        tm, tq, TT = get(tname, tpool)
        vS, vT = view(ST, list(range(len(sm)))), view(TT, list(range(len(tm))))
        nqT = len(tq)
        rng = random.Random(SEED)
        order = list(range(nqT)); rng.shuffle(order)
        h = nqT // 2
        folds = [(order[:h], order[h:]), (order[h:], order[:h])]
        ref = run_ensemble(list(range(len(tm))), *TT, nqT, folds)
        sc = Scorer(TT[3], TT[4], TT[5], nqT)
        best = max(sc.f1_at(TT[0][:, j].astype(float), 1.0).mean() for j in range(len(tm)))
        row = dict(src=sname, src_pool=len(sm), tgt=tname, tgt_pool=len(tm), what=label,
                   nq_src=len(sq), nq_tgt=nqT, best_single_oracle=100 * best,
                   majority=100 * majority(vT).mean(),
                   **{f"paper/{k}": 100 * v.mean() for k, v in ref.items()})
        for relmode in ("none", "agr", "ds"):
            f = Fitted(vS, np.ones(len(sq), bool), relmode)
            o, p = f.apply(vT)
            for dec, v in o.items():
                row[f"tr/{relmode}/{dec}"] = 100 * v.mean()
            row[f"ece/{relmode}"] = ece(p, vT["y"])
            row[f"ece-ls/{relmode}"] = ece(f.q, vT["y"])
            row[f"meanp_vs_rate/{relmode}"] = [float(p.mean()), float(f.q.mean()),
                                               float(vT["y"].mean())]
            for dec in ("bud-src", "ls-bud-src", "thr"):
                d, lo, hi = boot(o[dec], ref["count"])
                row[f"tr/{relmode}/{dec}_vs_cnt"] = [d, lo, hi]
                d, lo, hi = boot(o[dec], o["cnt-thr"])
                row[f"tr/{relmode}/{dec}_vs_cnt0"] = [d, lo, hi]
                d, lo, hi = boot(o[dec], ref["rank+budget"])
                row[f"tr/{relmode}/{dec}_vs_paper"] = [d, lo, hi]
        out.append(row)
        print(f"    {sname}({len(sm)}) -> {tname}({len(tm)})  {label}", flush=True)
    return out


# ------------------------------------------------------------------ export
def zero_label_p(target, source="qampari", src_pool=None, tgt_pool=None, relmode="self"):
    """Per-claim p on `target` from a scorer fitted on `source` only.

    Returns ({qid: {normalised claim: p}}, {qid: set of claims the source-fitted
    budget keeps}). No target gold is read for either: the target tensors carry
    gold only because the loader always builds it, and Fitted.apply never
    consults it for relmode none/agr/ds."""
    src_pool = src_pool or AZURE + GPT
    tgt_pool = tgt_pool or AZURE
    r = load(source, src_pool)
    vS = view(tensors(*r), list(range(len(r[0]))))
    f = Fitted(vS, np.ones(len(r[1]), bool), relmode)
    models, qids, cand, ngold, nlist, keys = load(target, tgt_pool, with_keys=True)
    vT = view(tensors(models, qids, cand, ngold, nlist), list(range(len(models))))
    _, p = f.apply(vT)
    G = np.exp(qfeat(vT, p) @ f.beta)
    kept = adaptive_keep(p, vT["rq"], vT["nq"], G)
    flat = [(q, k) for q in qids for k in keys[q]]
    assert len(flat) == len(p)
    P, K = collections.defaultdict(dict), collections.defaultdict(set)
    for (q, k), pi, ki in zip(flat, p, kept):
        P[q][k] = float(pi)
        if ki:
            K[q].add(k)
    return P, K


def zero_label_p_pool(src_pool=None, tgt_pool=None, relmode="self"):
    """QAMPARI, zero-label on a NEW POOL (the definition chosen 22 Sept).

    The scorer is fitted on the answers of `src_pool` (default: the 3 gpt-5.4
    agents) to the TRAIN half of the questions, and applied to the answers of
    `tgt_pool` (default: the 5 Azure agents) to the TEST half. Pools share no
    agent and the halves share no question. No gold of the target pool, nor of
    any test question, is read. The split is the same seeded half used by part 1.

    Returns ({qid: {claim: p}}, {qid: set kept by the source-fitted budget},
    list of test-half qids)."""
    src_pool = src_pool or GPT
    tgt_pool = tgt_pool or AZURE
    allp = sorted(set(src_pool) | set(tgt_pool))
    models, qids, cand, ngold, nlist, keys = load("qampari", allp, with_keys=True)
    T = tensors(models, qids, cand, ngold, nlist)
    nq = len(qids)
    rng = random.Random(SEED)
    order = list(range(nq)); rng.shuffle(order)
    trm = np.zeros(nq, bool); trm[order[:nq // 2]] = True
    S = [models.index(m) for m in sorted(src_pool)]
    Tg = [models.index(m) for m in sorted(tgt_pool)]
    f = Fitted(view(T, S), trm, relmode)
    vT = view(T, Tg)
    _, p = f.apply(vT)
    kept = f.bud_mask
    flat = [(q, k) for q in qids for k in keys[q]]
    keepT = T[0][:, Tg].any(1)
    flat = [fk for fk, kk in zip(flat, keepT) if kk]
    assert len(flat) == len(p)
    P, K = collections.defaultdict(dict), collections.defaultdict(set)
    test = {qids[i] for i in range(nq) if not trm[i]}
    for (q, k), pi, ki in zip(flat, p, kept):
        if q not in test:
            continue
        P[q][k] = float(pi)
        if ki:
            K[q].add(k)
    return P, K, sorted(test)


# ------------------------------------------------------------------ part 3
def paper_scores(v, qmask):
    """The paper's identity-indexed rank scorer (rank_crossover.run_ensemble's F),
    fitted in-pool with target gold. Reference ranking only."""
    z, rk, ll, k = v["z"], v["rk"], v["ll"], v["k"]
    cnt = z.sum(1).astype(float)
    lenm = np.maximum(np.exp(ll), 1.0)
    lr_ = np.log1p(np.where(z, rk, 0.0))
    relr = rk / lenm
    minr = np.min(np.where(z, rk, np.inf), axis=1)
    F = np.column_stack([np.ones(len(z)), z.astype(float), cnt / k, np.log1p(minr) / 5.0,
                         (lr_.sum(1) / np.maximum(cnt, 1)) / 5.0,
                         np.min(np.where(z, relr, np.inf), axis=1),
                         np.where(z, relr, 0.0).sum(1) / np.maximum(cnt, 1), lr_ / 5.0])
    m = qmask[v["rq"]]
    return F @ fit_logistic(F[m], v["y"][m])


def oracle_cut(v, scores):
    """Per question, the best F1 over every realisable cutoff of this ranking
    (a cutoff may only close a tie group). Measures RANKING quality alone, with
    the keep decision handed to an oracle."""
    rq, rg, ng, nq = v["rq"], v["rg"], v["ng"], v["nq"]
    out = np.zeros(nq)
    order = np.lexsort((-scores, rq))
    qs = rq[order]
    start = np.searchsorted(qs, np.arange(nq), "left")
    end = np.searchsorted(qs, np.arange(nq), "right")
    for q in range(nq):
        idx = order[start[q]:end[q]]
        if not len(idx):
            continue
        g, s = rg[idx], scores[idx]
        seen, hit, best = set(), 0, 0.0
        for i in range(len(idx)):
            if g[i] >= 0:
                hit += 1; seen.add(int(g[i]))
            if i + 1 < len(idx) and s[i + 1] == s[i]:
                continue
            pr, rc = hit / (i + 1), len(seen) / ng[q]
            if pr + rc > 0:
                best = max(best, 2 * pr * rc / (pr + rc))
        out[q] = best
    return out


def part3():
    """Split every transfer loss into RANKING loss and CUTOFF loss."""
    print("\n  PART 3  ranking vs cutoff: realised F1 / oracle-cutoff F1", flush=True)
    res = []
    # (a) within QAMPARI, first 10 directed 4|4 splits, fold 0 test half
    models, qids, cand, ngold, nlist = load("qampari", AZURE + GPT)
    T = tensors(models, qids, cand, ngold, nlist)
    nq = len(qids)
    rng = random.Random(SEED)
    order = list(range(nq)); rng.shuffle(order)
    tr = np.zeros(nq, bool); tr[order[:nq // 2]] = True
    te = ~tr
    acc = collections.defaultdict(list)
    for A in list(itertools.combinations(range(8), 4))[:35]:
        B = [j for j in range(8) if j not in A]
        vA, vB = view(T, list(A)), view(T, B)
        f_in, f_tr = Fitted(vB, tr, "none"), Fitted(vA, tr, "none")
        o_in, p_in = f_in.apply(vB)
        o_tr, p_tr = f_tr.apply(vB)
        ps = paper_scores(vB, tr)
        cntv = vB["z"].sum(1).astype(float)
        for name, sc_, real in (("count", cntv, None),
                                ("paper in-pool", ps, None),
                                ("invariant in-pool", p_in, o_in["bud-src"]),
                                ("invariant TRANSFER", p_tr, o_tr["bud-src"])):
            acc[name + "/oracle"].append(oracle_cut(vB, sc_)[te].mean() * 100)
            if real is not None:
                acc[name + "/real"].append(real[te].mean() * 100)
    print("    QAMPARI 4|4, 35 splits, held-out half:")
    for k in sorted(acc):
        print(f"      {k:<28}{statistics.mean(acc[k]):7.2f}")
    res.append({"qampari_4x4": {k: statistics.mean(v) for k, v in acc.items()}})

    # (b) across datasets
    for sname, spool, tname, tpool in (("qampari", AZURE, "cwq", AZURE),
                                       ("qampari", AZURE, "webqsp", AZURE),
                                       ("qampari", AZURE + GPT, "webqsp", AZURE),
                                       ("cwq", AZURE, "webqsp", AZURE),
                                       ("cwq", AZURE, "qampari", AZURE)):
        sm, sq, ST = (lambda r: (r[0], r[1], tensors(*r)))(load(sname, spool))
        tm, tq, TT = (lambda r: (r[0], r[1], tensors(*r)))(load(tname, tpool))
        vS, vT = view(ST, list(range(len(sm)))), view(TT, list(range(len(tm))))
        nqT = len(tq)
        rng = random.Random(SEED)
        o2 = list(range(nqT)); rng.shuffle(o2)
        trT = np.zeros(nqT, bool); trT[o2[:nqT // 2]] = True
        teT = ~trT
        f = Fitted(vS, np.ones(len(sq), bool), "none")
        o, p = f.apply(vT)
        ps = paper_scores(vT, trT)
        cntv = vT["z"].sum(1).astype(float)
        row = dict(src=sname, tgt=tname, src_pool=len(sm))
        for name, sc_ in (("count", cntv), ("paper in-pool", ps), ("invariant TRANSFER", p)):
            row[name + "/oracle"] = oracle_cut(vT, sc_)[teT].mean() * 100
        row["invariant TRANSFER/real thr"] = o["thr"][teT].mean() * 100
        row["invariant TRANSFER/real bud-src"] = o["bud-src"][teT].mean() * 100
        print(f"    {sname}({len(sm)}) -> {tname}({len(tm)}), held-out half:  " +
              "  ".join(f"{k} {v:.2f}" for k, v in row.items() if isinstance(v, float)))
        res.append(row)
    return res


# ------------------------------------------------------------------ report
def report1(rows, relinfo):
    def m(k):
        return statistics.mean(r[k] for r in rows)
    n = len(rows)
    print(f"\n  PART 1 RESULT  mean F1 over {n} directed 4|4 transfers (target pool B)")
    print("  " + "-" * 74)
    print(f"  {'best single (oracle pick, needs labels)':<46}{m('best-single(oracle)'):>8.2f}")
    print(f"  {'majority vote (label-free)':<46}{m('majority'):>8.2f}")
    print(f"  {'count, in-pool threshold (uses B gold)':<46}{m('paper/count'):>8.2f}")
    print(f"  {'paper scorer rank+budget, in-pool (B gold)':<46}{m('paper/rank+budget'):>8.2f}")
    print()
    print(f"  {'':<22}{'in-pool (B gold)':>18}{'TRANSFER (no B gold)':>24}")
    for rm, lab in (("none", "no reliability"), ("agr", "agreement rel."),
                    ("ds", "Dawid-Skene LLR"), ("sup", "supervised rel.*")):
        for dec in ("thr", "bud-src", "bud-sum", "ls-thr", "ls-bud-src", "ls-bud-sum"):
            print(f"  {lab + ' / ' + dec:<30}{m(f'in/{rm}/{dec}'):>10.2f}"
                  f"{m(f'tr/{rm}/{dec}'):>20.2f}")
    print(f"  {'count / source threshold':<30}{'':>10}{m('tr/none/cnt-thr'):>20.2f}")
    print("  * supervised reliability on the target uses target labels: dev-only.")
    print()
    for k in ("tr/none/bud-src", "tr/none/ls-bud-src", "tr/ds/bud-src", "tr/ds/ls-bud-src",
              "tr/agr/bud-src", "tr/none/bud-sum", "paper/rank+budget"):
        wins = sum(1 for r in rows if r[f"{k}_sig"] and r[f"{k}_vs_cnt"] > 0)
        print(f"  {k:<22} vs in-pool count  {m(f'{k}_vs_cnt'):+6.2f}   "
              f"significant wins {wins}/{n}")
    worse = sum(1 for r in rows if r["tr_vs_paper_ci"][1] < 0)
    print(f"  transfer(none,bud-src) - paper in-pool  {m('tr_vs_paper'):+6.2f}   "
          f"significantly worse in {worse}/{n}")
    print(f"  ECE of transferred p   none {m('ece/none'):.2f}  agr {m('ece/agr'):.2f}"
          f"  ds {m('ece/ds'):.2f}  sup {m('ece/sup'):.2f}")
    print(f"\n  label-free reliability vs supervised precision, 8 agents: "
          f"Spearman {relinfo['spearman']:+.3f}")
    for mm, a, s in zip(relinfo["models"], relinfo["agr"], relinfo["sup"]):
        print(f"    {mm:<32} agr {a:.3f}   sup {s:.3f}")


def report2(out):
    print("\n  PART 2 RESULT  (F1 on target; transfer arms use no target gold)")
    for r in out:
        print(f"\n  {r['src']}({r['src_pool']}) -> {r['tgt']}({r['tgt_pool']})  "
              f"[{r['what']}]  n={r['nq_tgt']}")
        print(f"    best single (oracle) {r['best_single_oracle']:6.2f}   majority "
              f"{r['majority']:6.2f}   count(in-pool) {r['paper/count']:6.2f}   "
              f"paper rank+budget(in-pool) {r['paper/rank+budget']:6.2f}")
        print(f"    zero-label count (source threshold) {r['tr/none/cnt-thr']:6.2f}")
        for rm in ("none", "agr", "ds"):
            mp, mq, rate = r[f"meanp_vs_rate/{rm}"]
            print(f"    transfer/{rm:<4} thr {r[f'tr/{rm}/thr']:6.2f}  bud-src "
                  f"{r[f'tr/{rm}/bud-src']:6.2f}  bud-sum {r[f'tr/{rm}/bud-sum']:6.2f}  "
                  f"| label-shift: thr {r[f'tr/{rm}/ls-thr']:6.2f}  bud-src "
                  f"{r[f'tr/{rm}/ls-bud-src']:6.2f}  bud-sum {r[f'tr/{rm}/ls-bud-sum']:6.2f}"
                  f"   ECE {r[f'ece/{rm}']:.1f} -> {r[f'ece-ls/{rm}']:.1f}"
                  f"  (mean p {mp:.3f} -> {mq:.3f}, rate {rate:.3f})")
            for dec in ("bud-src", "ls-bud-src"):
                d, lo, hi = r[f"tr/{rm}/{dec}_vs_cnt"]
                d0, lo0, hi0 = r[f"tr/{rm}/{dec}_vs_cnt0"]
                d2, lo2, hi2 = r[f"tr/{rm}/{dec}_vs_paper"]
                print(f"        {dec:<11} vs count(in-pool) {d:+.2f} [{lo:+.2f},{hi:+.2f}]"
                      f"  vs count(zero-label) {d0:+.2f} [{lo0:+.2f},{hi0:+.2f}]"
                      f"  vs paper(in-pool) {d2:+.2f} [{lo2:+.2f},{hi2:+.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-splits", type=int, default=0)
    ap.add_argument("--part", default="1,2,3")
    a = ap.parse_args()
    res = {}
    if "1" in a.part:
        rows, relinfo = part1(a)
        report1(rows, relinfo)
        res["part1"] = rows; res["reliability"] = relinfo
    if "2" in a.part:
        out = part2()
        report2(out)
        res["part2"] = out
    if "3" in a.part:
        res["part3"] = part3()
    json.dump(res, open(RESULTS / "pool_transfer.json", "w"), indent=1)
    print("\n  wrote results/pool_transfer.json")


if __name__ == "__main__":
    main()
