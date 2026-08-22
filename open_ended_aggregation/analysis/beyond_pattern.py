"""
Weighting rules that read information the support pattern does not contain.

WHY THIS FILE EXISTS
  `pattern` -- the shrunk lookup of P(correct | exact subset asserting) -- upper
  bounds every rule that reads only WHO asserted a claim. That includes
  correlation-aware and omission-aware weighting, because the pattern already
  determines who agreed and who stayed silent. On QAMPARI at n=8 that ceiling is
  +0.25 over counting and not significant, so no re-weighting of the support
  pattern can rescue the method at full pool size. The only way up is to
  condition on something outside the pattern.

  Three such signals are already sitting in the cached generations, free:

    RANK       QAMPARI answers are ORDERED lists, and precision falls
               monotonically with position: 29.2% at rank 1, 15.8% at rank 11,
               6.3% at rank 26.
    VERBOSITY  How much the model said FOR THIS QUESTION. Items from a model
               that listed 5-9 of them are 34.2% correct; items from a model
               that listed 50+ are 1.9% correct. Kimi's median list is 13 items
               and its longest is 2431 -- a single global weight averages those
               together and reports one number for both.
    SOURCE     QAMPARI ships five question sources (wikidata_simple /
               _intersection / _comp, wikitables_simple / _composition). That is
               a question-dependent weight w_i(q) with no new labelling.

  All three are per (model, question, claim). None is a function of the support
  pattern, so none is bounded by `pattern`.

ARMS
  count          keep if >= theta assert                        [baseline]
  pattern        shrunk P(correct | exact subset)               [pattern ceiling]
  corr           sum_m w_m z_m  -  lambda * sum_{m<m'} rho_mm' z_m z_m'
                 diversity-aware: rho is the error correlation, estimated from
                 co-assertion on WRONG claims only, so two models that
                 habitually hallucinate together stop counting as two votes.
                 Still a function of the pattern, so `pattern` bounds it -- kept
                 because it spends 1 free parameter where the ceiling spends
                 2^n, and may cross-fit better than the ceiling does.
  lr-pattern     logistic on assert indicators + count           [should ~= pattern]
  lr+qtype       + question-source one-hots                      [w_i(q), coarse]
  lr+qstat       + how contested this question is
  lr+verbosity   + how much each supporter said here
  lr+rank        + where the claim sat in each supporter's list
  lr-full        everything

  Reading it: lr-pattern is the sanity check that the logistic can reproduce the
  ceiling. Any arm that clears `pattern` is doing so on information the pattern
  does not have, which is the whole point.

HONESTY
  5-fold cross-fitted over QUESTIONS. Weights, rho, lambda, every logistic
  coefficient and the keep threshold are fitted on 4 folds and scored on the
  fifth. Nothing is tuned on what it is scored on -- on this dataset that
  distinction has already turned a +0.08 headline into -1.25.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.beyond_pattern
"""
import json, math, random, argparse, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.benchmarks import qampari as QA
from open_ended_aggregation.analysis.verify_qampari import bootstrap
from open_ended_aggregation.paths import DATA, RESULTS

SEED = 0
FOLDS = 5
RIDGE = 1.0
GRID = 600          # candidate thresholds per arm per fold
GEN = "qampari_asc800.jsonl"


# ------------------------------------------------------------------ data
def load(pool=None):
    """One row per (question, distinct normalised claim), with the per-model
    rank and per-model list length attached."""
    recs = [json.loads(l) for l in open(DATA / GEN)]
    byq = collections.defaultdict(dict)
    for r in recs:
        byq[r["qid"]].setdefault(r["model"], r)
    models = sorted({r["model"] for r in recs}) if pool is None else sorted(pool)
    items = {it["qid"]: it for it in QA.load_items(100000)}
    qids = sorted(q for q, d in byq.items() if set(models) <= set(d) and q in items)

    cand, ngold, nlist, qtype = {}, {}, {}, {}
    for q in qids:
        gold = [{QA.norm(a) for a in [g["answer_text"]] + list(g.get("aliases") or [])
                 if str(a).strip()} for g in items[q]["gold"]]
        ngold[q] = len(gold)
        qtype[q] = q.split("__", 1)[1]
        rank = collections.defaultdict(dict)     # claim -> model -> rank
        L = {}
        for m in models:
            seen = 0
            for it in byq[q][m]["items"]:
                k = QA.norm(it)
                if not k:
                    continue
                if m not in rank[k]:
                    rank[k][m] = seen            # first occurrence wins
                seen += 1
            L[m] = max(1, seen)
        nlist[q] = L
        rows = []
        for k, rk in rank.items():
            gi = next((i for i, gs in enumerate(gold) if k in gs), None)
            rows.append((frozenset(rk), gi, dict(rk)))
        cand[q] = rows
    return models, qids, cand, ngold, nlist, qtype


# ------------------------------------------------------------------ features
def featurize(models, qids, cand, nlist, qtype):
    """X (N x D), y (N,), question index (N,), plus named column groups."""
    n = len(models)
    mi = {m: i for i, m in enumerate(models)}
    types = sorted(set(qtype.values()))
    ti = {t: i for i, t in enumerate(types)}

    cols, groups, c = [], {}, 0

    def block(name, k):
        nonlocal c
        groups.setdefault(name, []).extend(range(c, c + k))
        c += k

    block("base", 1 + n + 1)                     # bias, z_m, count
    block("qtype", len(types))
    block("qstat", 2)
    block("verb", 2 + n)
    block("rank", 4 + n)
    D = c

    rowsX, rowsY, rowsQ, pats, rowsG, rowsR = [], [], [], [], [], []
    for qi, q in enumerate(qids):
        rws = cand[q]
        ncand = len(rws)
        mx = max((len(ms) for ms, _, _ in rws), default=1)
        for ms, gi, rk in rws:
            x = np.zeros(D)
            j = 0
            x[j] = 1.0; j += 1
            for m in ms:
                x[j + mi[m]] = 1.0
            j += n
            x[j] = len(ms) / n; j += 1
            x[j + ti[qtype[q]]] = 1.0; j += len(types)
            x[j] = math.log1p(ncand) / 5.0
            x[j + 1] = mx / n; j += 2
            # verbosity: how much each supporter said on THIS question
            ls = [math.log(nlist[q][m]) for m in ms]
            x[j] = statistics.mean(ls) / 5.0
            x[j + 1] = min(ls) / 5.0
            j += 2
            for m in ms:
                x[j + mi[m]] = math.log(nlist[q][m]) / 5.0
            j += n
            # rank: where the claim sat in each supporter's list
            rs = [math.log1p(rk[m]) for m in ms]
            rl = [rk[m] / nlist[q][m] for m in ms]
            x[j] = min(rs) / 5.0
            x[j + 1] = statistics.mean(rs) / 5.0
            x[j + 2] = min(rl)
            x[j + 3] = statistics.mean(rl)
            j += 4
            for m in ms:
                x[j + mi[m]] = math.log1p(rk[m]) / 5.0
            j += n
            rowsX.append(x); rowsY.append(gi is not None)
            rowsQ.append(qi); pats.append(ms); rowsG.append(gi)
            rk_row = np.full(n, -1)
            for m in ms:
                rk_row[mi[m]] = rk[m]
            rowsR.append(rk_row)
    return (np.array(rowsX), np.array(rowsY, dtype=float),
            np.array(rowsQ), pats, rowsG, np.array(rowsR), groups, types)


# ------------------------------------------------------------------ direct mask scoring
def make_eval(qids, ngold, rowq, rowgi):
    """mean per-question F1 of an arbitrary keep-mask, over a subset of questions.

    Needed for the CONTROLS, which are conjunctions (count AND rank) rather than
    thresholds on one score, so the prefix trick in curves() does not apply."""
    nq = len(qids)
    ng = np.array([max(1, ngold[q]) for q in qids], dtype=float)
    ok = np.array([g is not None for g in rowgi])
    gidx = np.array([-1 if g is None else g for g in rowgi])
    span = int(gidx.max()) + 2
    uid = rowq.astype(np.int64) * span + gidx           # (question, gold slot)

    def ev(mask, qs):
        n_kept = np.bincount(rowq[mask], minlength=nq).astype(float)
        m2 = mask & ok
        n_hit = np.bincount(rowq[m2], minlength=nq).astype(float)
        u = np.unique(uid[m2])
        n_gold_hit = np.bincount((u // span).astype(int), minlength=nq).astype(float)
        p = np.divide(n_hit, n_kept, out=np.zeros(nq), where=n_kept > 0)
        r = n_gold_hit / ng
        f = np.divide(2 * p * r, p + r, out=np.zeros(nq), where=(p + r) > 0)
        return f, float(f[qs].mean())
    return ev


# ------------------------------------------------------------------ per-question F1 curve
def curves(qids, ngold, rowgi, spans, scores):
    """For each question, scores sorted descending and the F1 of every prefix.
    A global threshold keeps a prefix per question, so one pass per question
    prices every threshold at once."""
    S, F = [], []
    for qi, q in enumerate(qids):
        a, b = spans[qi]
        idx = a + np.argsort(-scores[a:b], kind="stable")
        f = np.zeros(b - a + 1)
        c = 0; hit = set()
        for j, i in enumerate(idx, 1):
            gi = rowgi[i]
            if gi is not None:
                c += 1; hit.add(gi)
            p = c / j
            r = len(hit) / max(1, ngold[q])
            f[j] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        S.append(-scores[idx])          # ascending, for searchsorted
        F.append(f)
    return S, F


def sweep(S, F, qs, grid):
    """Best threshold on the questions `qs`, and the mean F1 it achieves."""
    tot = np.zeros(len(grid))
    neg = -grid
    for qi in qs:
        j = np.searchsorted(S[qi], neg, side="right")
        tot += F[qi][j]
    k = int(np.argmax(tot))
    return grid[k], tot[k] / len(qs)


def apply_th(S, F, qs, t):
    return {qi: float(F[qi][int(np.searchsorted(S[qi], -t, side="right"))]) for qi in qs}


# ------------------------------------------------------------------ estimators
def fit_logistic(X, y, ridge=RIDGE, iters=30):
    d = X.shape[1]
    w = np.zeros(d)
    R = ridge * np.eye(d); R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + R @ w
        s = np.clip(p * (1 - p), 1e-6, None)
        H = X.T @ (X * s[:, None]) + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def fit_pattern(pats, y, alpha=20.0):
    bypat = collections.defaultdict(lambda: [0, 0])
    bycnt = collections.defaultdict(lambda: [0, 0])
    for ms, ok in zip(pats, y):
        bypat[ms][0] += ok; bypat[ms][1] += 1
        bycnt[len(ms)][0] += ok; bycnt[len(ms)][1] += 1
    cr = {k: (h + 1) / (t + 2) for k, (h, t) in bycnt.items()}
    pat = {p: (h + alpha * cr.get(len(p), .05)) / (t + alpha)
           for p, (h, t) in bypat.items()}
    return pat, cr


def fit_corr(pats, y, models):
    """w_m = P(correct | m asserts alone); rho = error correlation, measured as
    the phi coefficient of co-assertion restricted to WRONG claims."""
    sn = collections.Counter(); sc = collections.Counter()
    for ms, ok in zip(pats, y):
        if len(ms) == 1:
            m = next(iter(ms)); sn[m] += 1; sc[m] += ok
    w = {m: (sc[m] + 1) / (sn[m] + 2) for m in models}

    wrong = [ms for ms, ok in zip(pats, y) if not ok]
    N = max(1, len(wrong))
    p1 = {m: sum(1 for ms in wrong if m in ms) / N for m in models}
    rho = {}
    for a, b in itertools.combinations(models, 2):
        p11 = sum(1 for ms in wrong if a in ms and b in ms) / N
        den = math.sqrt(max(1e-9, p1[a] * (1 - p1[a]) * p1[b] * (1 - p1[b])))
        rho[(a, b)] = (p11 - p1[a] * p1[b]) / den
    return w, rho


# ------------------------------------------------------------------ main
def run(models, qids, cand, ngold, nlist, qtype, label):
    X, y, rowq, pats, rowgi, RK, groups, types = featurize(
        models, qids, cand, nlist, qtype)
    n = len(models)
    spans = []
    for qi in range(len(qids)):
        w = np.nonzero(rowq == qi)[0]
        spans.append((int(w[0]), int(w[-1]) + 1) if len(w) else (0, 0))
    C = lambda s: curves(qids, ngold, rowgi, spans, s)
    ev = make_eval(qids, ngold, rowq, rowgi)
    cnt_arr = np.array([len(p) for p in pats])
    minrank = np.where(RK >= 0, RK, 10 ** 6).min(axis=1)
    KGRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 10 ** 6]
    print(f"\n  {label}   n={len(qids)} questions, {n} agents, "
          f"{len(y):,} candidate claims, {100*y.mean():.1f}% correct")
    print(f"  question sources: " + ", ".join(f"{t.split('__')[0]}" for t in types))

    rng = random.Random(SEED)
    order = qids[:]; rng.shuffle(order)
    fold = {q: i % FOLDS for i, q in enumerate(order)}
    qfold = np.array([fold[q] for q in qids])

    ARMS = {
        "count":        ("fixed", None),
        "pattern":      ("pattern", None),
        "corr":         ("corr", None),
        "lr-pattern":   ("lr", ["base"]),
        "lr+qtype":     ("lr", ["base", "qtype"]),
        "lr+qstat":     ("lr", ["base", "qstat"]),
        "lr+verbosity": ("lr", ["base", "verb"]),
        "lr+rank":      ("lr", ["base", "rank"]),
        "lr-full":      ("lr", ["base", "qtype", "qstat", "verb", "rank"]),
    }
    ORDER = list(ARMS) + ["single-trunc", "count x rank"]
    per_arm = {a: {} for a in ORDER}
    coefs = {}

    for f in range(FOLDS):
        tr_q = [i for i in range(len(qids)) if qfold[i] != f]
        te_q = [i for i in range(len(qids)) if qfold[i] == f]
        trm = np.isin(rowq, tr_q)

        for arm, (kind, cfg) in ARMS.items():
            if kind == "fixed":
                sc = np.array([float(len(p)) for p in pats])
            elif kind == "pattern":
                pat, cr = fit_pattern([p for p, m in zip(pats, trm) if m], y[trm])
                sc = np.array([pat.get(p, cr.get(len(p), .05)) for p in pats])
            elif kind == "corr":
                w, rho = fit_corr([p for p, m in zip(pats, trm) if m], y[trm], models)
                uniq = {p for p in pats}
                best = None
                for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
                    tbl = {p: sum(w[m] for m in p) -
                              lam * sum(rho[(a, b)] for a, b in
                                        itertools.combinations(sorted(p), 2))
                           for p in uniq}
                    s = np.array([tbl[p] for p in pats])
                    Sl, Fl = C(s)
                    g = np.unique(s)
                    if len(g) > GRID:
                        g = np.quantile(g, np.linspace(0, 1, GRID))
                    t, v = sweep(Sl, Fl, tr_q, g)
                    if best is None or v > best[0]:
                        best = (v, s, t, lam)
                _, sc, th, lam = best
                S, F = C(sc)
                per_arm[arm].update(apply_th(S, F, te_q, th))
                coefs.setdefault("corr_lambda", []).append(lam)
                continue
            else:
                cix = sorted(set(i for gname in cfg for i in groups[gname]))
                w = fit_logistic(X[np.ix_(trm, cix)], y[trm])
                sc = X[:, cix] @ w
                if f == 0:
                    coefs[arm] = {"cols": cix, "w": [round(float(v), 4) for v in w]}

            S, F = C(sc)
            g = np.unique(sc[trm])
            if len(g) > GRID:
                g = np.quantile(g, np.linspace(0, 1, GRID))
            th, _ = sweep(S, F, tr_q, g)
            per_arm[arm].update(apply_th(S, F, te_q, th))

        # ---- CONTROLS. The rank gain is worthless if either of these matches it.
        # single-trunc: ONE model's list, cut at k. If truncation alone gets there,
        # the ensemble is doing no work and the finding is "short lists score
        # better under F1", not "aggregation exploits rank".
        best = None
        for mi_, m in enumerate(models):
            for k in KGRID:
                _, v = ev((RK[:, mi_] >= 0) & (RK[:, mi_] < k), tr_q)
                if best is None or v > best[0]:
                    best = (v, mi_, k)
        _, mi_, k = best
        f_all, _ = ev((RK[:, mi_] >= 0) & (RK[:, mi_] < k), tr_q)
        per_arm["single-trunc"].update({qi: float(f_all[qi]) for qi in te_q})
        coefs.setdefault("single_trunc", []).append((models[mi_], k))

        # count x rank: plain counting AND a global rank cut. Two integers, no
        # per-model weights at all -- the interpretable version of lr+rank.
        best = None
        for th_ in range(1, n + 1):
            for R in KGRID:
                _, v = ev((cnt_arr >= th_) & (minrank < R), tr_q)
                if best is None or v > best[0]:
                    best = (v, th_, R)
        _, th_, R = best
        f_all, _ = ev((cnt_arr >= th_) & (minrank < R), tr_q)
        per_arm["count x rank"].update({qi: float(f_all[qi]) for qi in te_q})
        coefs.setdefault("count_x_rank", []).append((th_, R))
        print(f"    fold {f+1}/{FOLDS} done", flush=True)

    # best single agent on the same questions
    singles = {}
    for m in models:
        v = []
        for qi, q in enumerate(qids):
            kept = [gi for ms, gi, _ in cand[q] if m in ms]
            if not kept:
                v.append(0.0); continue
            c = sum(1 for gi in kept if gi is not None)
            h = len({gi for gi in kept if gi is not None})
            p = c / len(kept); r = h / max(1, ngold[q])
            v.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
        singles[m] = v
    bm = max(models, key=lambda m: statistics.mean(singles[m]))
    best = singles[bm]

    idx = list(range(len(qids)))
    cnt = [per_arm["count"][i] for i in idx]
    pat_v = [per_arm["pattern"][i] for i in idx]

    print(f"\n  best single = {bm} ({100*statistics.mean(best):.2f})")
    print(f"  {'arm':<15}{'F1':>8}{'vs count':>23}{'vs pattern ceiling':>23}")
    print("  " + "-" * 69)
    res = {}
    for arm in ORDER:
        v = [per_arm[arm][i] for i in idx]
        res[arm] = 100 * statistics.mean(v)
        if arm == "count":
            c1 = f"{'(reference)':>23}"
        else:
            d, l, h = bootstrap(v, cnt)
            c1 = f"{d:+7.2f} [{l:+6.2f},{h:+6.2f}]{'*' if (l>0 or h<0) else ' '}"
        if arm == "pattern":
            c2 = f"{'(ceiling)':>23}"
        else:
            d, l, h = bootstrap(v, pat_v)
            c2 = f"{d:+7.2f} [{l:+6.2f},{h:+6.2f}]{'*' if (l>0 or h<0) else ' '}"
        print(f"  {arm:<15}{res[arm]:8.2f}{c1:>23}{c2:>23}")
    print(f"\n  * = bootstrap 95% CI over questions excludes 0")
    if "corr_lambda" in coefs:
        print(f"  corr lambda selected per fold: {coefs['corr_lambda']}")
    return res, coefs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="", help="comma-separated subset of agents")
    a = ap.parse_args()
    pool = [x for x in a.pool.split(",") if x] or None

    print("=" * 78)
    print("  BEYOND THE PATTERN CEILING -- QAMPARI, 5-fold cross-fitted, cached data")
    print("=" * 78)
    models, qids, cand, ngold, nlist, qtype = load(pool)
    res, coefs = run(models, qids, cand, ngold, nlist, qtype, "QAMPARI")
    json.dump({"qampari": res, "models": models, "coefs": coefs},
              open(RESULTS / "beyond_pattern.json", "w"), indent=2)
    print("\n  wrote results/beyond_pattern.json")


if __name__ == "__main__":
    main()
