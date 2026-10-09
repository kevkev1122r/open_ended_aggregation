"""
The four "not yet positive" items from 22 Sept, on cached data.

E1  SIZE FROM THE CROWD, CONTENT FROM THE EVIDENCE.
    In every cross-dataset transfer the count threshold fitted on the source
    equalled the one fitted on the target (pool_transfer part 2) -- the count
    cutoff transfers, the p cutoff does not. So: per question keep exactly as
    many claims as the source-fitted count rule keeps, but choose WHICH ones by
    the transferred scorer. No parameters, no target gold. Any gain over count
    is ranking alone, at identical output size.
    (Motivated by part 2's numbers, so it is one rule stated in advance of this
    file, not a search over rules. All variants tried are printed.)

E2  SELF-NORMALISED VERBOSITY ("self"): rank measured against each agent's own
    typical list length, label-free. Does it recover the 1.5 identity points?

E3  MULTI-SOURCE: leave one dataset out, train on the other two (Azure-5 pool
    on all three, so the views stack).

E4  RUNWAY: across the 70 QAMPARI transfers, does the gain over count shrink as
    the target pool's best single agent gets stronger?

Usage:  ./venv/bin/python -u -m open_ended_aggregation.analysis.cut_rules
"""
import json, random, itertools, collections, statistics

import numpy as np

from open_ended_aggregation.analysis.pool_transfer import (
    load, view, Fitted, AZURE, GPT, SEED, boot, oracle_cut, majority)
from open_ended_aggregation.analysis.rank_crossover import Scorer, run_ensemble, tensors
from open_ended_aggregation.paths import RESULTS


def topk_mask(v, p, K):
    """Per question keep the K[q] highest-p claims."""
    m = np.zeros(len(p), bool)
    o = np.lexsort((-p, v["rq"]))
    qs = v["rq"][o]
    st = np.searchsorted(qs, np.arange(v["nq"]), "left")
    en = np.searchsorted(qs, np.arange(v["nq"]), "right")
    for q in range(v["nq"]):
        m[o[st[q]:en[q]][:int(K[q])]] = True
    return m


def rules(v, f, p):
    """Every zero-label decision rule, given a scorer fitted on the source."""
    sc = Scorer(v["rq"], v["rg"], v["ng"], v["nq"])
    cnt = v["z"].sum(1) / v["k"]
    cmask = cnt >= f.t_cnt - 1e-12
    K = np.bincount(v["rq"], weights=cmask.astype(float), minlength=v["nq"])
    o, _ = f.apply(v)
    return {
        "count(src thr)": sc.f1_of_mask(cmask),
        "scorer bud-src": o["bud-src"],
        "scorer thr": o["thr"],
        "SIZE=count, CONTENT=scorer": sc.f1_of_mask(topk_mask(v, p, K)),
        "SIZE=max(count,1)": sc.f1_of_mask(topk_mask(v, p, np.maximum(K, 1))),
        "SIZE=mean(count,budget)": sc.f1_of_mask(topk_mask(
            v, p, np.round((K + np.bincount(v["rq"], weights=f.bud_mask.astype(float),
                                            minlength=v["nq"])) / 2))),
    }


def e1_e2_e4():
    models, qids, cand, ngold, nlist = load("qampari", AZURE + GPT)
    T = tensors(models, qids, cand, ngold, nlist)
    nq = len(qids)
    rng = random.Random(SEED)
    order = list(range(nq)); rng.shuffle(order)
    h = nq // 2
    folds = [(order[:h], order[h:]), (order[h:], order[:h])]
    sc_full = Scorer(T[3], T[4], T[5], nq)
    singles = {j: sc_full.f1_at(T[0][:, j].astype(float), 1.0) for j in range(8)}

    rows = []
    for A in itertools.combinations(range(8), 4):
        B = [j for j in range(8) if j not in A]
        vA, vB = view(T, list(A)), view(T, B)
        arms = {}
        for relmode in ("none", "self"):
            for tr, te in folds:
                trm = np.zeros(nq, bool); trm[tr] = True
                f = Fitted(vA, trm, relmode)
                _, p = f.apply(vB)
                for k, val in rules(vB, f, p).items():
                    arms.setdefault(f"{relmode}/{k}", np.zeros(nq))[te] = val[te]
                if relmode == "self":
                    fin = Fitted(vB, trm, "self")
                    o_in, _ = fin.apply(vB)
                    arms.setdefault("self/IN-POOL bud-src", np.zeros(nq))[te] = o_in["bud-src"][te]
        ref = run_ensemble(B, *T, nq, folds)
        arms["paper in-pool"] = ref["rank+budget"]
        arms["count in-pool"] = ref["count"]
        row = {k: 100 * v.mean() for k, v in arms.items()}
        row["best_single"] = 100 * max(singles[j].mean() for j in B)
        for k in ("none/SIZE=count, CONTENT=scorer", "self/SIZE=count, CONTENT=scorer",
                  "none/scorer bud-src", "self/scorer bud-src"):
            d, lo, hi = boot(arms[k], arms["none/count(src thr)"])
            row[k + " vs count0"] = [d, lo, hi]
        rows.append(row)
    return rows


def e3():
    """Leave one dataset out, Azure-5 pool everywhere so the views stack."""
    V = {}
    for name in ("qampari", "cwq", "webqsp"):
        r = load(name, AZURE)
        V[name] = view(tensors(*r), list(range(5)))

    def stack(vs):
        out = {k: [] for k in ("z", "rk", "ll", "rq", "rg", "y", "ng")}
        off = 0
        for v in vs:
            for k in ("z", "rk", "ll", "rg", "y"):
                out[k].append(v[k])
            out["rq"].append(v["rq"] + off)
            out["ng"].append(v["ng"])
            off += v["nq"]
        s = {k: np.concatenate(x) for k, x in out.items()}
        s["nq"], s["k"] = off, 5
        return s

    res = []
    for tgt in V:
        src = [n for n in V if n != tgt]
        vS = stack([V[n] for n in src])
        vT = V[tgt]
        row = {"target": tgt, "sources": src}
        for label, srcv in [("both", vS)] + [(n, V[n]) for n in src]:
            for relmode in ("none", "self"):
                f = Fitted(srcv, np.ones(srcv["nq"], bool), relmode)
                _, p = f.apply(vT)
                R = rules(vT, f, p)
                for k, val in R.items():
                    row[f"{label}/{relmode}/{k}"] = 100 * val.mean()
                d, lo, hi = boot(R["SIZE=count, CONTENT=scorer"], R["count(src thr)"])
                row[f"{label}/{relmode}/SIZE=count vs count0"] = [d, lo, hi]
                d, lo, hi = boot(R["scorer bud-src"], R["count(src thr)"])
                row[f"{label}/{relmode}/bud-src vs count0"] = [d, lo, hi]
        res.append(row)
    return res


def main():
    rows = e1_e2_e4()
    n = len(rows)
    m = lambda k: statistics.mean(r[k] for r in rows)
    print(f"\n  E1/E2  QAMPARI, {n} directed 4|4 transfers, no target gold unless marked")
    for k in ("count in-pool", "paper in-pool", "self/IN-POOL bud-src"):
        print(f"    {k + '  (target gold)':<44}{m(k):7.2f}")
    for rm in ("none", "self"):
        for k in ("count(src thr)", "scorer thr", "scorer bud-src",
                  "SIZE=count, CONTENT=scorer", "SIZE=max(count,1)", "SIZE=mean(count,budget)"):
            print(f"    {rm + '/' + k:<44}{m(rm + '/' + k):7.2f}")
    for k in ("none/SIZE=count, CONTENT=scorer", "self/SIZE=count, CONTENT=scorer",
              "none/scorer bud-src", "self/scorer bud-src"):
        ds = [r[k + " vs count0"] for r in rows]
        print(f"    {k:<40} vs zero-label count {statistics.mean(d[0] for d in ds):+5.2f}"
              f"   sig wins {sum(1 for d in ds if d[1] > 0)}/{n}"
              f"   sig losses {sum(1 for d in ds if d[2] < 0)}/{n}")

    # E4 runway
    bs = np.array([r["best_single"] for r in rows])
    for k in ("self/scorer bud-src", "self/SIZE=count, CONTENT=scorer", "paper in-pool"):
        g = np.array([r[k] - r["count in-pool"] for r in rows])
        print(f"\n  E4  gain of {k} over in-pool count vs best single: "
              f"corr {np.corrcoef(bs, g)[0,1]:+.2f}, slope {np.polyfit(bs, g, 1)[0]:+.2f} per F1 point")
        for lo, hi in ((0, 25), (25, 27), (27, 100)):
            sel = (bs >= lo) & (bs < hi)
            if sel.any():
                print(f"      best single in [{lo},{hi}): n={sel.sum():2d}  gain {g[sel].mean():+.2f}")

    res3 = e3()
    print("\n  E3  leave one dataset out (Azure-5 pool), zero-label on target")
    for r in res3:
        print(f"\n    target {r['target']}  count(src thr) {r['both/none/count(src thr)']:.2f}")
        for label in ["both"] + r["sources"]:
            for rm in ("none", "self"):
                d, lo, hi = r[f"{label}/{rm}/SIZE=count vs count0"]
                d2, lo2, hi2 = r[f"{label}/{rm}/bud-src vs count0"]
                print(f"      from {label:<8}{rm:<5} SIZE=count {r[f'{label}/{rm}/SIZE=count, CONTENT=scorer']:6.2f}"
                      f" ({d:+.2f} [{lo:+.2f},{hi:+.2f}])   bud-src "
                      f"{r[f'{label}/{rm}/scorer bud-src']:6.2f} ({d2:+.2f} [{lo2:+.2f},{hi2:+.2f}])")
    json.dump({"e1_e2_e4": rows, "e3": res3}, open(RESULTS / "cut_rules.json", "w"), indent=1)
    print("\n  wrote results/cut_rules.json")


if __name__ == "__main__":
    main()
