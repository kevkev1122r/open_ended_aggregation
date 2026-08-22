"""
Third round. Everything outside the support pattern, at once, then ablated.

WHAT v1/v2 ESTABLISHED
  count 29.35 | pattern (ceiling for global-threshold who-said-what rules) 29.60
  lr+rank 30.52 | nb-rank 30.69 | adapt-rank 31.32

  Two axes, roughly independent, neither of which is reliability weighting:
  per-question BUDGETS (+1.19 with no model identity at all) and RANK (+1.16
  with a global threshold). Model identity is worth about +0.3 on top of either.

WHAT IS STILL UNEXPLOITED, AND WHY EACH SHOULD MATTER

  OMISSION STRENGTH.  Every rule so far treats silence as one thing. nb-full
  gives every silent agent the same log((1-TPR)/(1-FPR)); the logistic gives it
  zero. But an agent that emitted 3 items and did not mention X has barely said
  anything, while an agent that emitted 50 and did not mention X has said a
  great deal. QAMPARI's list lengths run from a median of 1 (gpt-5.4-nano) to a
  mean of 71 (Kimi), so this is a large, entirely unmodelled signal.

  NONPARAMETRIC PER-AGENT RANK.  lr+rank spends ONE linear coefficient per agent
  on log-rank. nb-rank is nonparametric in rank but purely additive across
  agents. Per-agent rank-bucket indicators inside a discriminative model is
  strictly more expressive than either.

  WITHIN-QUESTION CONTEXT.  A support count of 2 means something different on a
  question where the maximum is 2 than on one where the maximum is 8. The
  candidate's percentile within its own question is what lets a GLOBAL threshold
  imitate a per-question budget.

  CLAIM CONTENT.  Length in words, whether it contains a digit, and whether the
  string already appears in the question. That last one is a classic
  false-positive: models echo the question back as an answer.

  STACKING.  The pattern probability is the best available summary of the
  support pattern, so feed it in as a feature rather than making the linear
  model rediscover it from 2^n indicators.

ABLATIONS
  each block is added to the rank model alone, then all together, then with the
  per-question budget on top. Everything 5-fold cross-fitted over questions.

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.weighting_v3
"""
import json, math, random, argparse, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.benchmarks import qampari as QA
from open_ended_aggregation.analysis.beyond_pattern import (
    make_eval, curves, sweep, apply_th, fit_logistic, fit_pattern,
    SEED, FOLDS, GRID)
from open_ended_aggregation.analysis.weighting_v2 import (
    bucket, NB_BUCKETS, fit_gold_ridge, adaptive_keep)
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import DATA, RESULTS

GEN = "qampari_asc800.jsonl"


def load_rich(pool=None):
    recs = [json.loads(l) for l in open(DATA / GEN)]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    models = sorted({r["model"] for r in recs}) if pool is None else sorted(pool)
    items = {it["qid"]: it for it in QA.load_items(100000)}
    qids = sorted(q for q, d in byq.items() if set(models) <= set(d) and q in items)

    cand, ngold, nlist, qtype, qtext = {}, {}, {}, {}, {}
    for q in qids:
        gold = [{QA.norm(a) for a in [g["answer_text"]] + list(g.get("aliases") or [])
                 if str(a).strip()} for g in items[q]["gold"]]
        ngold[q] = len(gold)
        qtype[q] = q.split("__", 1)[1]
        qtext[q] = QA.norm(items[q]["question"])
        rank, L = collections.defaultdict(dict), {}
        for m in models:
            seen = 0
            for it in byq[q][m]["items"]:
                k = QA.norm(it)
                if not k:
                    continue
                if m not in rank[k]:
                    rank[k][m] = seen
                seen += 1
            L[m] = max(1, seen)
        nlist[q] = L
        rows = []
        for k, rk in rank.items():
            gi = next((i for i, gs in enumerate(gold) if k in gs), None)
            rows.append((frozenset(rk), gi, dict(rk), k))
        cand[q] = rows
    return models, qids, cand, ngold, nlist, qtype, qtext


def build_features(models, qids, cand, nlist, qtype, qtext):
    n = len(models)
    mi = {m: i for i, m in enumerate(models)}
    types = sorted(set(qtype.values()))
    blocks, cols, c = {}, [], 0

    def blk(name, k):
        nonlocal c
        blocks.setdefault(name, []).extend(range(c, c + k)); c += k

    blk("base", 1 + n + 1)
    blk("rank", 6 + n)
    blk("rankbuckets", n * NB_BUCKETS)
    blk("omit", 3 + n)
    blk("verb", 2 + n)
    blk("qctx", 4)
    blk("content", 4)
    blk("qtype", len(types))
    blk("stack", 1)
    D = c

    X, y, rq, pats, gis, RK = [], [], [], [], [], []
    for qi, q in enumerate(qids):
        rws = cand[q]
        ncand = len(rws)
        counts = [len(ms) for ms, _, _, _ in rws]
        mx = max(counts) if counts else 1
        n2 = sum(1 for c_ in counts if c_ >= 2)
        srt = sorted(counts)
        for ms, gi, rk, key in rws:
            x = np.zeros(D); j = 0
            x[j] = 1.0; j += 1
            for m in ms:
                x[j + mi[m]] = 1.0
            j += n
            x[j] = len(ms) / n; j += 1

            rs = [math.log1p(rk[m]) for m in ms]
            rl = [rk[m] / nlist[q][m] for m in ms]
            x[j] = min(rs) / 5.0
            x[j + 1] = statistics.mean(rs) / 5.0
            x[j + 2] = max(rs) / 5.0
            x[j + 3] = (statistics.pstdev(rs) if len(rs) > 1 else 0.0) / 5.0
            x[j + 4] = min(rl)
            x[j + 5] = statistics.mean(rl)
            j += 6
            for m in ms:
                x[j + mi[m]] = math.log1p(rk[m]) / 5.0
            j += n

            for m in ms:                                   # per-agent rank bucket
                x[j + mi[m] * NB_BUCKETS + int(bucket(np.array([rk[m]]))[0])] = 1.0
            j += n * NB_BUCKETS

            # OMISSION STRENGTH: how much did each silent agent actually say?
            sil = [m for m in models if m not in ms]
            sl = [math.log(nlist[q][m]) for m in sil]
            x[j] = (sum(sl) / 5.0) / n
            x[j + 1] = (max(sl) / 5.0) if sl else 0.0
            x[j + 2] = sum(1 for m in sil if nlist[q][m] >= 20) / n
            j += 3
            for m in sil:
                x[j + mi[m]] = math.log(nlist[q][m]) / 5.0
            j += n

            lv = [math.log(nlist[q][m]) for m in ms]
            x[j] = statistics.mean(lv) / 5.0
            x[j + 1] = min(lv) / 5.0
            j += 2
            for m in ms:
                x[j + mi[m]] = math.log(nlist[q][m]) / 5.0
            j += n

            x[j] = math.log1p(ncand) / 5.0
            x[j + 1] = mx / n
            x[j + 2] = n2 / 20.0
            # percentile of this candidate's support within its own question
            x[j + 3] = np.searchsorted(srt, len(ms), side="left") / max(1, ncand)
            j += 4

            w = key.split()
            x[j] = math.log1p(len(w)) / 2.0
            x[j + 1] = 1.0 if any(ch.isdigit() for ch in key) else 0.0
            x[j + 2] = 1.0 if key and key in qtext[q] else 0.0
            x[j + 3] = min(len(key), 40) / 40.0
            j += 4

            x[j + types.index(qtype[q])] = 1.0
            j += len(types)
            # j is the stack slot, filled per fold

            X.append(x); y.append(gi is not None); rq.append(qi)
            pats.append(ms); gis.append(gi)
            r = np.full(n, -1)
            for m in ms:
                r[mi[m]] = rk[m]
            RK.append(r)
    return (np.array(X), np.array(y, dtype=float), np.array(rq), pats, gis,
            np.array(RK), blocks, D - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="")
    a = ap.parse_args()
    pool = [x for x in a.pool.split(",") if x] or None

    print("=" * 100)
    print("  WEIGHTING v3 -- omission strength, per-agent rank buckets, "
          "within-question context, claim content")
    print("=" * 100)
    models, qids, cand, ngold, nlist, qtype, qtext = load_rich(pool)
    X, y, rowq, pats, gis, RK, blocks, STACK = build_features(
        models, qids, cand, nlist, qtype, qtext)
    nq, n = len(qids), len(models)
    ev = make_eval(qids, ngold, rowq, gis)
    ng_arr = np.array([ngold[q] for q in qids], dtype=float)
    cnt = np.array([len(p) for p in pats], dtype=float)
    print(f"\n  {nq} questions, {n} agents, {len(y):,} claims, "
          f"{100*y.mean():.1f}% correct, {X.shape[1]} features")

    spans = []
    for qi in range(nq):
        w = np.nonzero(rowq == qi)[0]
        spans.append((int(w[0]), int(w[-1]) + 1) if len(w) else (0, 0))
    C = lambda s: curves(qids, ngold, gis, spans, s)

    tix = sorted(set(qtype.values()))
    QF = np.zeros((nq, 4 + len(tix)))
    for qi, q in enumerate(qids):
        w = np.nonzero(rowq == qi)[0]
        QF[qi, :4] = [1.0, math.log1p(len(w)) / 5.0,
                      (cnt[w].max() if len(w) else 0) / n, (cnt[w] >= 2).sum() / 20.0]
        QF[qi, 4 + tix.index(qtype[q])] = 1.0

    R = ["base", "rank", "rankbuckets"]
    ABL = {
        "lr+rank":            ["base", "rank"],
        "v3 +buckets":        R,
        "v3 +omission":       R + ["omit"],
        "v3 +verbosity":      R + ["verb"],
        "v3 +qctx":           R + ["qctx"],
        "v3 +content":        R + ["content"],
        "v3 +stack":          R + ["stack"],
        "v3 ALL":             R + ["omit", "verb", "qctx", "content", "qtype", "stack"],
        # ALL loses to its own ablations, so the kitchen sink is overfitting.
        # LEAN keeps only the blocks that helped on their own.
        "v3 LEAN":            R + ["omit", "stack"],
    }
    BUDGETS = ["LEAN + budget", "LEAN + learned-k",
               "LEAN + budget (oracleG)", "LEAN + oracle-k",
               "ORACLE selection"]
    ARMS = ["count", "pattern"] + list(ABL) + BUDGETS
    per = {a: np.zeros(nq) for a in ARMS}
    lam_pick, ghat_err = [], []

    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}
    qfold = np.array([fold[q] for q in qids])

    for f in range(FOLDS):
        tr_q = np.nonzero(qfold != f)[0]
        te_q = np.nonzero(qfold == f)[0]
        tm = np.zeros(nq, dtype=bool); tm[tr_q] = True
        trm = tm[rowq]

        plut, cr = fit_pattern([p for p, m in zip(pats, trm) if m], y[trm])
        pat_p = np.array([plut.get(p, cr.get(len(p), .05)) for p in pats])
        X[:, STACK] = np.log(pat_p / (1 - pat_p)) / 5.0

        # ridge strength chosen on a held-out SLICE OF TRAIN, never on test
        inner = tr_q[: int(0.8 * len(tr_q))]
        held = tr_q[int(0.8 * len(tr_q)):]
        im = np.isin(rowq, inner)
        lean_ix = sorted(set(i for g in ABL["v3 LEAN"] for i in blocks[g]))
        best_lam = None
        for lam in (0.3, 1.0, 3.0, 10.0, 30.0):
            wl = fit_logistic(X[np.ix_(im, lean_ix)], y[im], ridge=lam)
            s = X[:, lean_ix] @ wl
            S, F = C(s)
            g = np.unique(s[im])
            if len(g) > GRID:
                g = np.quantile(g, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, list(inner), g)
            got = apply_th(S, F, list(held), th)
            v = statistics.mean(got.values())
            if best_lam is None or v > best_lam[0]:
                best_lam = (v, lam)
        lam = best_lam[1]
        lam_pick.append(lam)

        scores = {"count": cnt, "pattern": pat_p}
        for arm, cfg in ABL.items():
            cix = sorted(set(i for g in cfg for i in blocks[g]))
            w = fit_logistic(X[np.ix_(trm, cix)], y[trm],
                             ridge=lam if arm == "v3 LEAN" else 1.0)
            scores[arm] = X[:, cix] @ w

        for arm, s in scores.items():
            S, F = C(s)
            g = np.unique(s[trm])
            if len(g) > GRID:
                g = np.quantile(g, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, list(tr_q), g)
            got = apply_th(S, F, list(te_q), th)
            for qi in te_q:
                per[arm][qi] = got[qi]

        p_lean = 1.0 / (1.0 + np.exp(-np.clip(scores["v3 LEAN"], -30, 30)))

        # G_hat: the 4 structural features were weak (|err| 4.8 on a mean of
        # 10.2). Sum of predicted probabilities over the question's candidates is
        # a direct estimate of how many correct answers are even on the table.
        QG = np.zeros((nq, QF.shape[1] + 3))
        QG[:, :QF.shape[1]] = QF
        for qi in range(nq):
            w_ = np.nonzero(rowq == qi)[0]
            QG[qi, QF.shape[1]:] = [p_lean[w_].sum() / 10.0,
                                    p_lean[w_][cnt[w_] >= 2].sum() / 10.0,
                                    float(p_lean[w_].max()) if len(w_) else 0.0]
        beta = fit_gold_ridge(QG[tr_q], ng_arr[tr_q])
        Ghat = np.maximum(QG @ beta, 1.0)
        ghat_err.append(float(np.abs(Ghat[te_q] - ng_arr[te_q]).mean()))

        S, F = C(scores["v3 LEAN"])
        for arm, G in (("LEAN + budget", Ghat),
                       ("LEAN + budget (oracleG)", ng_arr)):
            fa, _ = ev(adaptive_keep(p_lean, rowq, nq, G), list(tr_q))
            for qi in te_q:
                per[arm][qi] = fa[qi]

        # learned-k: regress the F1-optimal prefix length on question features,
        # rather than plugging G_hat into the F1 surrogate
        kstar = np.array([int(np.argmax(F[qi])) for qi in range(nq)], dtype=float)
        bk = fit_gold_ridge(QG[tr_q], kstar[tr_q])
        khat = np.clip(np.round(QG @ bk), 0, None).astype(int)
        for qi in te_q:
            per["LEAN + learned-k"][qi] = float(F[qi][min(khat[qi], len(F[qi]) - 1)])
            per["LEAN + oracle-k"][qi] = float(F[qi].max())

        # absolute ceiling for ANY filter: keep exactly the correct candidates
        fa, _ = ev(np.array([g is not None for g in gis]), list(tr_q))
        for qi in te_q:
            per["ORACLE selection"][qi] = fa[qi]
        print(f"    fold {f+1}/{FOLDS} done (ridge={lam})", flush=True)

    singles = {}
    for j, m in enumerate(models):
        fa, _ = ev(RK[:, j] >= 0, list(range(nq)))
        singles[m] = fa
    bm = max(models, key=lambda m: singles[m].mean())
    best = list(singles[bm])
    ref_c = list(per["count"]); ref_p = list(per["pattern"])

    print(f"\n  best single = {bm} ({100*singles[bm].mean():.2f})")
    print(f"  ridge picked per fold: {lam_pick}")
    print(f"  mean |G_hat - G| = {statistics.mean(ghat_err):.2f} "
          f"(gold mean {ng_arr.mean():.1f})")
    print(f"\n  {'arm':<28}{'F1':>7}{'vs best single':>24}{'vs count':>24}"
          f"{'vs ceiling':>22}")
    print("  " + "-" * 105)

    def cell(v, ref, w=24):
        d, lo, hi = bootstrap(v, ref)
        return f"{d:+6.2f} ({100*d/(100*statistics.mean(ref)):+5.1f}%)" \
               f"[{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}"

    res = {}
    for arm in ARMS:
        v = list(per[arm])
        res[arm] = 100 * float(np.mean(v))
        c0 = cell(v, best)
        c1 = f"{'(reference)':>24}" if arm == "count" else cell(v, ref_c)
        d, lo, hi = bootstrap(v, ref_p)
        c2 = f"{'(ceiling)':>22}" if arm == "pattern" else \
             f"{d:+7.2f} [{lo:+5.2f},{hi:+5.2f}]{'*' if (lo>0 or hi<0) else ' '}"
        print(f"  {arm:<28}{res[arm]:7.2f}{c0:>24}{c1:>24}{c2:>22}")
    print("\n  * = bootstrap 95% CI over questions excludes 0. "
          "(%) is relative to that column's baseline.")

    json.dump(res, open(RESULTS / "weighting_v3.json", "w"), indent=2)
    print("  wrote results/weighting_v3.json")


if __name__ == "__main__":
    main()
