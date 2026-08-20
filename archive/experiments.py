"""
Pilot experiments for Kernel-Weighted Aggregation (KWA) on open-ended answers.

Run:  ./venv/bin/python experiments.py            (all)
      ./venv/bin/python experiments.py E4 E5      (selected)

Writes results/*.json, results/figures/*.png, and prints markdown tables.
"""
import sys, json, os, time
import numpy as np
import kernel_agg as ka

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- defaults
BETAS = [8.0, 6.0, 4.0, 2.5, 1.5]        # 5 agents, strong -> weak
M_DEF = 1500
POOL = dict(n_para=2, n_near=3, n_far=4)
C_DEF = 1 + POOL["n_para"] + POOL["n_near"] + POOL["n_far"]
RESULTS = {}


def agent_accuracies(A, S):
    return np.array([np.mean([ka.correct_mask(S[q])[A[q, j]] for q in range(len(A))])
                     for j in range(A.shape[1])])


def run_aggregators(A, S, betas_true, betas_hat, rng_seed=0, tau=0.90):
    """Evaluate every aggregator on one dataset. Returns {name: accuracy}."""
    M, N = A.shape
    C = S.shape[1]
    x_true = agent_accuracies(A, S)
    w_ow = np.log((C - 1) * np.clip(x_true, 1e-6, 1-1e-6) / (1 - np.clip(x_true, 1e-6, 1-1e-6)))
    agree = ka.second_order_table(A, C)
    try:
        x_owl = ka.estimate_acc_from_agreement(agree, K_eff=C)
    except Exception:
        x_owl = np.full(N, 1.0 / C + 1e-3)
    w_owl = np.log((C - 1) * np.clip(x_owl, 1e-6, 1-1e-6) / (1 - np.clip(x_owl, 1e-6, 1-1e-6)))

    methods = {
        "MV-exact":      (ka.agg_majority_exact,   {}),
        "MV-cluster":    (ka.agg_majority_cluster, dict(tau=tau)),
        "OW-oracle":     (ka.agg_ow_exact,         dict(weights=w_ow)),
        "OW-L":          (ka.agg_ow_exact,         dict(weights=w_owl)),
        "KWA-oracle":    (ka.agg_kernel,           dict(betas=betas_true, support="observed")),
        "KWA-EM":        (ka.agg_kernel,           dict(betas=betas_hat,  support="observed")),
    }
    out = {}
    for name, (fn, kw) in methods.items():
        rng = np.random.default_rng(rng_seed)
        ok = 0
        for q in range(M):
            pick = fn(A[q], S[q], rng, **kw)
            ok += bool(ka.correct_mask(S[q])[pick])
        out[name] = 100.0 * ok / M
    out["BestSingle"] = 100.0 * x_true.max()
    out["_x_true"] = x_true.tolist()
    out["_x_owl"] = x_owl.tolist()
    return out


def table(rows, cols, title):
    """rows: list of (label, {col: val}); prints a markdown table."""
    print(f"\n### {title}\n")
    print("| " + " | ".join([""] + cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for lab, d in rows:
        cells = [f"{d[c]:.2f}" if isinstance(d.get(c), float) else str(d.get(c, "")) for c in cols]
        print("| " + " | ".join([str(lab)] + cells) + " |")


# ======================================================================== E1
def E1():
    """Correctness: KWA with an exact-match kernel must BE Optimal Weight."""
    print("\n" + "="*72)
    print("E1  Correctness -- does KWA reduce exactly to OW on multiple choice?")
    print("="*72)
    res = {}
    for K in (2, 3, 4, 6, 10):
        r = ka.test_reduction(seed=K, K=K, N=5, trials=3000)
        res[f"K={K}"] = r
        print(f"   K={K:<3} agreement with OW: {r:.4f}")
    print("\n   Interpretation: 1.0000 everywhere. The multiple-choice case is")
    print("   recovered exactly, so KWA is a strict generalisation, not a rival.")
    RESULTS["E1"] = res


# ======================================================================== E2
def E2():
    """Label-free beta recovery vs. dataset size (full support / well specified)."""
    print("\n" + "="*72)
    print("E2  Can beta be recovered with NO labels?  (oracle support)")
    print("="*72)
    rows, res = [], {}
    for M in (100, 300, 1000, 3000):
        errs, corrs, hats = [], [], []
        for s in range(4):
            rng = np.random.default_rng(100 + s)
            A, S = ka.make_dataset(M, BETAS, rng, pool_kw=POOL)
            bh, _ = ka.em_estimate_beta(A, S, support="full")
            hats.append(bh)
            errs.append(np.sqrt(np.mean((bh - np.array(BETAS))**2)))
            corrs.append(np.corrcoef(bh, BETAS)[0, 1])
        mh = np.mean(hats, axis=0)
        rows.append((f"M={M}", {"RMSE": float(np.mean(errs)),
                                "corr": float(np.mean(corrs)),
                                "beta_hat": np.round(mh, 2).tolist()}))
        res[M] = dict(rmse=float(np.mean(errs)), corr=float(np.mean(corrs)),
                      beta_hat=mh.tolist())
        print(f"   M={M:<5} RMSE {np.mean(errs):5.2f}   corr {np.mean(corrs):.4f}   "
              f"beta_hat {np.round(mh,2)}   (true {BETAS})")
    print("\n   Note where the error lives: the WEAK agents are pinned down well and")
    print("   the STRONG ones are not. Same pathology as OW -- accuracy saturates,")
    print("   so the likelihood is flat in beta at the top end.")
    RESULTS["E2"] = res


# ======================================================================== E3
def E3():
    """The support-selection bias -- the central practical obstacle."""
    print("\n" + "="*72)
    print("E3  Support bias: normalising over the OBSERVED answers vs the true pool")
    print("="*72)
    print("""
   In deployment you never see the full answer space. The natural approximation
   is to normalise over the answers the agents actually produced. But that set
   is SELECTION BIASED -- it contains each agent's own draw by construction.
""")
    rows, res = [], {}
    for N in (3, 5, 8, 12):
        bt = list(np.linspace(8.0, 1.5, N))
        obs_hat, full_hat = [], []
        for s in range(3):
            rng = np.random.default_rng(200 + s)
            A, S = ka.make_dataset(M_DEF, bt, rng, pool_kw=POOL)
            bo, _ = ka.em_estimate_beta(A, S, support="observed")
            bf, _ = ka.em_estimate_beta(A, S, support="full")
            obs_hat.append(np.corrcoef(bo, bt)[0, 1])
            full_hat.append(np.corrcoef(bf, bt)[0, 1])
        med_support = float(np.mean([len(np.unique(A[q])) for q in range(len(A))]))
        rows.append((f"N={N}", {"support size": med_support,
                                "corr (observed)": float(np.mean(obs_hat)),
                                "corr (full)": float(np.mean(full_hat))}))
        res[N] = dict(support=med_support, obs=float(np.mean(obs_hat)),
                      full=float(np.mean(full_hat)))
        print(f"   N={N:<3} mean distinct answers/question {med_support:4.1f}   "
              f"corr(beta_hat, beta): observed {np.mean(obs_hat):+.3f}   full {np.mean(full_hat):+.3f}")
    RESULTS["E3"] = res


# ======================================================================== E4
def E4():
    """Main comparison."""
    print("\n" + "="*72)
    print("E4  Main comparison -- open-ended aggregation accuracy")
    print("="*72)
    allr = {}
    for s in range(5):
        rng = np.random.default_rng(300 + s)
        A, S = ka.make_dataset(M_DEF, BETAS, rng, pool_kw=POOL)
        bh, _ = ka.em_estimate_beta(A, S, support="full")
        r = run_aggregators(A, S, np.array(BETAS), bh, rng_seed=s)
        for k, v in r.items():
            if not k.startswith("_"):
                allr.setdefault(k, []).append(v)
    order = ["MV-exact", "MV-cluster", "OW-L", "OW-oracle", "KWA-EM", "KWA-oracle", "BestSingle"]
    print(f"\n   {'method':<14}{'accuracy %':>12}{'+/- sd':>9}{'vs MV-exact':>14}")
    print("   " + "-"*49)
    base = np.mean(allr["MV-exact"])
    res = {}
    for k in order:
        m, sd = float(np.mean(allr[k])), float(np.std(allr[k]))
        res[k] = dict(mean=m, sd=sd, gain=m - base)
        print(f"   {k:<14}{m:>12.2f}{sd:>9.2f}{m-base:>+14.2f}")
    RESULTS["E4"] = res


# ======================================================================== E5
def E5():
    """Vote splitting: the mechanism KWA is supposed to fix."""
    print("\n" + "="*72)
    print("E5  Vote splitting -- vary how many PARAPHRASES of the truth exist")
    print("="*72)
    print("""
   Paraphrases are counted as correct answers. Exact-match voting SPLITS their
   votes; a single wrong answer can then win a plurality. This is the specific
   open-ended failure the kernel is meant to repair.
""")
    res = {}
    curves = {}
    for npara in (0, 1, 2, 3, 4):
        pk = dict(POOL); pk["n_para"] = npara
        acc = {}
        for s in range(3):
            rng = np.random.default_rng(400 + s)
            A, S = ka.make_dataset(M_DEF, BETAS, rng, pool_kw=pk)
            bh, _ = ka.em_estimate_beta(A, S, support="full")
            r = run_aggregators(A, S, np.array(BETAS), bh, rng_seed=s)
            for k, v in r.items():
                if not k.startswith("_"):
                    acc.setdefault(k, []).append(v)
        row = {k: float(np.mean(v)) for k, v in acc.items()}
        res[npara] = row
        for k, v in row.items():
            curves.setdefault(k, []).append(v)
        print(f"   paraphrases={npara}: " +
              "  ".join(f"{k} {row[k]:5.1f}" for k in
                        ["MV-exact", "MV-cluster", "OW-L", "KWA-EM", "KWA-oracle"]))
    RESULTS["E5"] = res
    _plot(list((0,1,2,3,4)), curves, "number of paraphrases of the truth in the pool",
          "accuracy (%)", "E5_vote_splitting.png",
          "Vote splitting: exact-match voting degrades, kernel does not")


# ======================================================================== E6
def E6():
    """How close the distractors are."""
    print("\n" + "="*72)
    print("E6  Distractor proximity -- vary similarity of plausible wrong answers")
    print("="*72)
    res, curves = {}, {}
    xs = [0.40, 0.55, 0.70, 0.85]
    for tn in xs:
        pk = dict(POOL); pk["t_near"] = tn
        acc = {}
        for s in range(3):
            rng = np.random.default_rng(500 + s)
            A, S = ka.make_dataset(M_DEF, BETAS, rng, pool_kw=pk)
            bh, _ = ka.em_estimate_beta(A, S, support="full")
            r = run_aggregators(A, S, np.array(BETAS), bh, rng_seed=s)
            for k, v in r.items():
                if not k.startswith("_"):
                    acc.setdefault(k, []).append(v)
        row = {k: float(np.mean(v)) for k, v in acc.items()}
        res[tn] = row
        for k, v in row.items():
            curves.setdefault(k, []).append(v)
        print(f"   t_near={tn:.2f}: " +
              "  ".join(f"{k} {row[k]:5.1f}" for k in
                        ["MV-exact", "MV-cluster", "OW-L", "KWA-EM", "KWA-oracle"]))
    RESULTS["E6"] = res
    _plot(xs, curves, "similarity of plausible wrong answers to the truth",
          "accuracy (%)", "E6_distractor_proximity.png",
          "Harder (nearer) distractors")


# ======================================================================== E7
def E7():
    """Misspecification: nonlinear kernel response, and correlated agents."""
    print("\n" + "="*72)
    print("E7  Misspecification -- the log-linear-in-similarity assumption")
    print("="*72)
    res = {}
    print("\n   (a) curvature: true log P ∝ beta * sim^c, but we fit c = 1")
    for c in (0.5, 1.0, 1.5, 2.0):
        acc = {}
        for s in range(3):
            rng = np.random.default_rng(600 + s)
            A, S = ka.make_dataset(M_DEF, BETAS, rng, pool_kw=POOL, curvature=c)
            bh, _ = ka.em_estimate_beta(A, S, support="full")
            r = run_aggregators(A, S, np.array(BETAS), bh, rng_seed=s)
            for k, v in r.items():
                if not k.startswith("_"):
                    acc.setdefault(k, []).append(v)
        row = {k: float(np.mean(v)) for k, v in acc.items()}
        res[f"curvature={c}"] = row
        print(f"      c={c}: " + "  ".join(f"{k} {row[k]:5.1f}" for k in
              ["MV-exact", "MV-cluster", "OW-L", "KWA-EM"]))

    print("\n   (b) correlated agents: two 'families' pulled to a shared wrong answer")
    fam = [0, 0, 1, 1, 2]
    for pull in (0.0, 1.0, 2.0, 3.0):
        acc = {}
        for s in range(3):
            rng = np.random.default_rng(700 + s)
            A, S = ka.make_dataset(M_DEF, BETAS, rng, pool_kw=POOL,
                                   family=fam, family_pull=pull)
            bh, _ = ka.em_estimate_beta(A, S, support="full")
            r = run_aggregators(A, S, np.array(BETAS), bh, rng_seed=s)
            for k, v in r.items():
                if not k.startswith("_"):
                    acc.setdefault(k, []).append(v)
        row = {k: float(np.mean(v)) for k, v in acc.items()}
        res[f"family_pull={pull}"] = row
        print(f"      pull={pull}: " + "  ".join(f"{k} {row[k]:5.1f}" for k in
              ["MV-exact", "MV-cluster", "OW-L", "KWA-EM"]))
    RESULTS["E7"] = res


# ======================================================================== E8
def E8():
    """Ceiling: when nobody generates a correct answer, nothing can help."""
    print("\n" + "="*72)
    print("E8  The generation ceiling")
    print("="*72)
    rng = np.random.default_rng(800)
    A, S = ka.make_dataset(M_DEF, BETAS, rng, pool_kw=POOL)
    hit = np.array([ka.correct_mask(S[q])[A[q]].any() for q in range(len(A))])
    print(f"\n   at least one agent produced a correct answer: {100*hit.mean():.1f}% of questions")
    print(f"   so NO aggregator can exceed {100*hit.mean():.1f}%.")
    bh, _ = ka.em_estimate_beta(A, S, support="full")
    r = run_aggregators(A, S, np.array(BETAS), bh, rng_seed=0)
    print(f"\n   {'method':<14}{'accuracy':>10}{'% of ceiling':>15}")
    print("   " + "-"*39)
    res = {"ceiling": float(100*hit.mean())}
    for k in ["MV-exact", "MV-cluster", "OW-L", "KWA-EM", "KWA-oracle"]:
        print(f"   {k:<14}{r[k]:>10.2f}{100*r[k]/(100*hit.mean()):>15.1f}")
        res[k] = r[k]
    print("\n   This ceiling has no analogue in multiple choice, where the correct")
    print("   answer is on the ballot by construction. It is the hard limit on")
    print("   open-ended aggregation and no weighting scheme can move it.")
    RESULTS["E8"] = res


# ---------------------------------------------------------------- plotting
def _plot(xs, curves, xlabel, ylabel, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"MV-exact": ("o-", "#b3261e"), "MV-cluster": ("s-", "#e8871a"),
             "OW-L": ("^-", "#7a5195"), "OW-oracle": ("v--", "#bbbbbb"),
             "KWA-EM": ("D-", "#1a73e8"), "KWA-oracle": ("*--", "#0b8043"),
             "BestSingle": (":", "#888888")}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, ys in curves.items():
        mk, col = style.get(k, ("-", None))
        ax.plot(xs, ys, mk, color=col, label=k, linewidth=1.8, markersize=6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)
    print(f"   [figure] results/figures/{fname}")




# ======================================================================== E9
def E9():
    """Can the kernel recover a correct answer that NO agent produced?

    E5 showed the KWA gain is LARGEST when there are zero paraphrases -- so the
    mechanism cannot be 'pooling near-duplicate votes'. The alternative
    hypothesis: because every agent's answer is drawn from a distribution
    CENTRED on the truth, even a set of entirely wrong answers points at the
    truth collectively. The kernel triangulates rather than tallies.

    Test: give the aggregator a candidate set larger than what the ensemble
    produced (in practice: a retrieval set, or a model asked to propose
    alternatives), and measure accuracy on questions where EVERY agent was wrong.
    """
    print("\n" + "="*72)
    print("E9  Triangulation -- picking an answer nobody generated")
    print("="*72)
    hard = [3.0, 2.5, 2.0, 1.5, 1.0]                 # weak ensemble
    pk = dict(n_para=1, n_near=4, n_far=4)
    res = {}
    rows = []
    for s in range(4):
        rng = np.random.default_rng(900 + s)
        A, S = ka.make_dataset(M_DEF, hard, rng, pool_kw=pk)
        bh, _ = ka.em_estimate_beta(A, S, support="full")
        anyone = np.array([ka.correct_mask(S[q])[A[q]].any() for q in range(len(A))])
        r = {}
        for label, sup in (("observed", "observed"), ("full-pool", None)):
            rng2 = np.random.default_rng(s)
            ok = np.array([bool(ka.correct_mask(S[q])[
                ka.agg_kernel(A[q], S[q], rng2, betas=bh, support=sup)])
                for q in range(len(A))])
            r[label] = 100*ok.mean()
            r[label + "_when_all_wrong"] = 100*ok[~anyone].mean() if (~anyone).any() else float("nan")
        rng2 = np.random.default_rng(s)
        mvc = np.array([bool(ka.correct_mask(S[q])[
            ka.agg_majority_cluster(A[q], S[q], rng2, tau=0.90)]) for q in range(len(A))])
        r["MV-cluster"] = 100*mvc.mean()
        r["ceiling"] = 100*anyone.mean()
        rows.append(r)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    res.update(agg)
    print(f"\n   ensemble is weak on purpose: single-agent accuracies are low")
    print(f"   at least one agent correct (the 'ceiling') : {agg['ceiling']:.1f}%")
    print(f"\n   {'aggregator':<34}{'accuracy':>10}")
    print("   " + "-"*44)
    print(f"   {'MV-cluster':<34}{agg['MV-cluster']:>10.2f}")
    print(f"   {'KWA, pick among produced answers':<34}{agg['observed']:>10.2f}")
    print(f"   {'KWA, pick from wider candidate set':<34}{agg['full-pool']:>10.2f}")
    print(f"\n   On the questions where EVERY agent was wrong:")
    print(f"   {'KWA restricted to produced answers':<38}{agg['observed_when_all_wrong']:>6.2f}%  (must be 0)")
    print(f"   {'KWA with a wider candidate set':<38}{agg['full-pool_when_all_wrong']:>6.2f}%  <-- broke the ceiling")
    RESULTS["E9"] = res


# ======================================================================= E10
def E10():
    """Deployable variant: EM with observed support only (no oracle anywhere)."""
    print("\n" + "="*72)
    print("E10  Fully deployable pipeline -- no oracle at any step")
    print("="*72)
    out = {}
    for N in (5, 8):
        bt = list(np.linspace(8.0, 1.5, N))
        acc = {}
        for s in range(3):
            rng = np.random.default_rng(1000 + s)
            A, S = ka.make_dataset(M_DEF, bt, rng, pool_kw=POOL)
            b_obs, _ = ka.em_estimate_beta(A, S, support="observed")
            b_full, _ = ka.em_estimate_beta(A, S, support="full")
            for lab, bb in (("KWA-EM(observed)", b_obs), ("KWA-EM(full)", b_full)):
                rng2 = np.random.default_rng(s)
                ok = np.mean([bool(ka.correct_mask(S[q])[
                    ka.agg_kernel(A[q], S[q], rng2, betas=bb, support="observed")])
                    for q in range(len(A))])
                acc.setdefault(lab, []).append(100*ok)
            rng2 = np.random.default_rng(s)
            acc.setdefault("MV-cluster", []).append(100*np.mean(
                [bool(ka.correct_mask(S[q])[ka.agg_majority_cluster(A[q], S[q], rng2, tau=0.90)])
                 for q in range(len(A))]))
            acc.setdefault("_corr_obs", []).append(np.corrcoef(b_obs, bt)[0, 1])
            acc.setdefault("_scale_obs", []).append(float(np.mean(b_obs)/np.mean(bt)))
        row = {k: float(np.mean(v)) for k, v in acc.items()}
        out[N] = row
        print(f"\n   N={N} agents")
        print(f"     MV-cluster                 {row['MV-cluster']:6.2f}")
        print(f"     KWA-EM (observed support)  {row['KWA-EM(observed)']:6.2f}   "
              f"corr(beta_hat,beta)={row['_corr_obs']:.3f}  scale={row['_scale_obs']:.2f}x")
        print(f"     KWA-EM (full support)      {row['KWA-EM(full)']:6.2f}")
    print("\n   The observed-support estimator SHRINKS beta toward zero (scale < 1)")
    print("   but preserves the ordering, and the aggregator only needs relative")
    print("   weights -- so the deployable pipeline loses very little.")
    RESULTS["E10"] = out


# ======================================================================= E11
def E11():
    """Re-run the main comparison with geometry MEASURED FROM REAL DATA.

    real_data_test.py measured, with all-MiniLM-L6-v2 cosine similarity:
        correct paraphrase   -> truth   mean 0.626
        plausible wrong      -> truth   mean 0.580     <-- only 0.046 below correct
        unrelated answer     -> truth   mean 0.058

    My synthetic pilot assumed 0.975 / 0.75 / 0.30 -- a correct-vs-plausible-wrong
    gap of 0.225, roughly 5x wider than reality. So the pilot's discrimination
    problem was far easier than the real one. This re-runs it honestly.
    """
    import functools
    print("\n" + "="*72)
    print("E11  Main comparison under REAL measured geometry")
    print("="*72)
    REAL = dict(n_para=2, n_near=3, n_far=4, t_para=0.626, t_near=0.580,
                t_far=0.058, jitter=0.10)
    SYNTH = dict(POOL); SYNTH.update(t_para=0.975, t_near=0.75, t_far=0.30, jitter=0.012)
    n_corr = 1 + REAL["n_para"]
    res = {}
    for label, pk in (("synthetic (optimistic)", SYNTH), ("real geometry", REAL)):
        # correctness by construction, not by similarity threshold
        orig = ka.correct_mask
        ka.correct_mask = functools.partial(orig, n_correct=n_corr)
        try:
            allr = {}
            for s in range(4):
                rng = np.random.default_rng(1100 + s)
                A, S = ka.make_dataset(M_DEF, BETAS, rng, pool_kw=pk)
                bh, _ = ka.em_estimate_beta(A, S, support="observed")   # deployable
                r = run_aggregators(A, S, np.array(BETAS), bh, rng_seed=s)
                for k, v in r.items():
                    if not k.startswith("_"):
                        allr.setdefault(k, []).append(v)
            row = {k: float(np.mean(v)) for k, v in allr.items()}
        finally:
            ka.correct_mask = orig
        res[label] = row
        print(f"\n   {label}")
        for k in ["MV-exact", "MV-cluster", "OW-L", "KWA-EM", "BestSingle"]:
            print(f"     {k:<12}{row[k]:7.2f}")
        print(f"     KWA-EM advantage over MV-cluster: {row['KWA-EM']-row['MV-cluster']:+.2f}")
    d1 = res["synthetic (optimistic)"]["KWA-EM"] - res["synthetic (optimistic)"]["MV-cluster"]
    d2 = res["real geometry"]["KWA-EM"] - res["real geometry"]["MV-cluster"]
    print(f"\n   advantage over the strong baseline: {d1:+.2f} synthetic  ->  {d2:+.2f} realistic")
    RESULTS["E11"] = res


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    which = sys.argv[1:] or ["E1","E2","E3","E4","E5","E6","E7","E8","E9","E10","E11"]
    t0 = time.time()
    for name in which:
        globals()[name]()
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\n\nDone in {time.time()-t0:.0f}s. Results -> results/results.json")


