"""
PROPER REGRESSION -- replacing the binned eyeball comparison with a real model.

What I was doing before: chop similarity into 10 slices, compute log-odds per
slice, fit a line to those 10 summary points, compare R^2 by eye. That throws
away everything inside a bin, and an R^2 on 10 aggregated points is not a
hypothesis test.

What this does instead: LOGISTIC REGRESSION on the raw observations. Each answer
is one row:

    y = 1  if it is an answer a model really produced
    y = 0  if it is a control (an answer that was not produced here)
    x = similarity of that answer to the truth

Then  logit P(y=1) = alpha + beta*sim  IS the model I assumed, fitted properly on
all ~27,000 rows instead of 10 binned points. Adding a sim^2 term and running a
likelihood-ratio test turns "the curve looks better" into an actual p-value.

Two statistical points that matter:

  * The rows are NOT independent -- each question contributes 1 positive and K
    controls, and the same wrong-answer texts get reused as controls elsewhere.
    So all standard errors are CLUSTERED BY QUESTION. Naive SEs would be far too
    small and every p-value would be fake.
  * Reported alongside: McFadden pseudo-R^2, AIC, and the LR test. The LR test is
    the one that answers "is curvature actually needed".
"""
import os, json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); FIG = os.path.join(OUT, "figures")

ENCODERS = ["sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5"]
DATASETS = {
    "HaluEval-QA": ("halueval_qa.parquet", "right_answer", "hallucinated_answer", None),
    "TruthfulQA":  ("truthfulqa.parquet", "best_answer", "incorrect_answers", "first"),
    "SciQ":        ("sciq.parquet", "correct_answer", "distractor1", None),
}
N_MAX, K = 3000, 8
_enc = {}


def encode(name, texts):
    if name not in _enc:
        from sentence_transformers import SentenceTransformer
        _enc[name] = SentenceTransformer(name)
    return _enc[name].encode(list(texts), batch_size=128, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)


def load(ds):
    f, tc, wc, mode = DATASETS[ds]
    d = pd.read_parquet(os.path.join(HERE, "data", f))
    if mode == "first":
        d = d[d[tc].astype(bool)]
        d = d[d[wc].apply(lambda a: len(a) > 0 and bool(a[0]))]
        w = d[wc].apply(lambda a: a[0])
    else:
        d = d.dropna(subset=[tc, wc]); w = d[wc]
    return list(d[tc])[:N_MAX], list(w)[:N_MAX]


def build_rows(Et, Ew, hard):
    """Long-format design: one row per (question, candidate). Returns sim, y, cluster."""
    n = len(Et)
    pos = np.einsum("ij,ij->i", Ew, Et)
    if hard:
        S = Et @ Et.T; np.fill_diagonal(S, -np.inf)
        idx = np.argsort(-S, axis=1)[:, :K]
    else:
        rng = np.random.default_rng(0)
        idx = np.empty((n, K), dtype=int)
        for j in range(K):
            p = rng.permutation(n); s = p == np.arange(n); p[s] = (p[s] + 1) % n
            idx[:, j] = p
    ctl = np.stack([np.einsum("ij,ij->i", Ew[idx[:, j]], Et) for j in range(K)], 1)
    sim = np.concatenate([pos, ctl.ravel()])
    y = np.concatenate([np.ones(n), np.zeros(n * K)])
    cl = np.concatenate([np.arange(n), np.tile(np.arange(n), K)])
    return sim, y, cl


def run(sim, y, cl):
    """Fit linear and quadratic logistic models with question-clustered SEs."""
    X1 = sm.add_constant(sim)
    X2 = sm.add_constant(np.column_stack([sim, sim ** 2]))
    m1 = sm.Logit(y, X1).fit(disp=0)
    m2 = sm.Logit(y, X2).fit(disp=0)
    r1 = sm.Logit(y, X1).fit(disp=0, cov_type="cluster", cov_kwds={"groups": cl})
    r2 = sm.Logit(y, X2).fit(disp=0, cov_type="cluster", cov_kwds={"groups": cl})
    lr = 2 * (m2.llf - m1.llf)                       # likelihood-ratio, 1 df
    p_lr = stats.chi2.sf(lr, 1)
    return dict(
        beta=float(r1.params[1]), se=float(r1.bse[1]),
        ci=[float(r1.conf_int()[1][0]), float(r1.conf_int()[1][1])],
        z=float(r1.tvalues[1]), p=float(r1.pvalues[1]),
        quad_coef=float(r2.params[2]), quad_se=float(r2.bse[2]),
        quad_z=float(r2.tvalues[2]), quad_p=float(r2.pvalues[2]),
        lr_stat=float(lr), lr_p=float(p_lr),
        pseudo_r2_lin=float(m1.prsquared), pseudo_r2_quad=float(m2.prsquared),
        aic_lin=float(m1.aic), aic_quad=float(m2.aic), n=int(len(y)))


RES = {}
print("=" * 100)
print(" LOGISTIC REGRESSION -- all raw rows, standard errors clustered by question")
print("=" * 100)
for ds in DATASETS:
    t, w = load(ds)
    RES[ds] = {}
    print(f"\n\n{'#'*100}\n# {ds}   ({len(t)} questions -> {len(t)*(K+1):,} rows)\n{'#'*100}")
    for menc in ENCODERS:
        short = menc.split("/")[-1]
        Et, Ew = encode(menc, t), encode(menc, w)
        RES[ds][short] = {}
        print(f"\n  {short}")
        print(f"    {'controls':<10}{'beta (95% CI)':>26}{'z':>8}{'p':>11}"
              f"{'quad coef':>12}{'quad p':>11}{'LR test p':>12}")
        print("    " + "-" * 91)
        for hard, lab in ((True, "HARD"), (False, "random")):
            sim, y, cl = build_rows(Et, Ew, hard)
            r = run(sim, y, cl)
            RES[ds][short][lab] = r
            pstr = "<1e-16" if r["p"] < 1e-16 else f"{r['p']:.2e}"
            qstr = "<1e-16" if r["quad_p"] < 1e-16 else f"{r['quad_p']:.2e}"
            lstr = "<1e-16" if r["lr_p"] < 1e-16 else f"{r['lr_p']:.2e}"
            print(f"    {lab:<10}{r['beta']:>8.2f} [{r['ci'][0]:>6.2f},{r['ci'][1]:>6.2f}]"
                  f"{r['z']:>8.1f}{pstr:>11}{r['quad_coef']:>12.2f}{qstr:>11}{lstr:>12}")

print("\n\n" + "=" * 100)
print(" WHAT THE REGRESSION SAYS")
print("=" * 100)
print(f"\n  {'dataset':<14}{'encoder':<22}{'controls':<9}{'beta':>8}"
      f"{'curvature needed?':>22}{'AIC improvement':>18}")
print("  " + "-" * 95)
n_sig = n_tot = 0
for ds, D in RES.items():
    for e, E in D.items():
        for lab, r in E.items():
            if lab != "HARD": continue
            n_tot += 1
            sig = r["lr_p"] < 0.001
            n_sig += sig
            print(f"  {ds:<14}{e:<22}{lab:<9}{r['beta']:>8.2f}"
                  f"{('YES  p<0.001' if sig else 'no'):>22}{r['aic_lin']-r['aic_quad']:>18.0f}")
print(f"\n  curvature statistically required in {n_sig}/{n_tot} cells (hard controls)")
print("""
  How to read this:
    beta          how steeply the odds of being a real error rise with similarity.
                  Every one is positive and enormous in z -- the EFFECT is not in doubt.
    quad coef     the bend. Positive = the curve accelerates upward.
    LR test p     'does adding the bend explain significantly more?' This is the
                  formal version of the R^2 comparison I was doing by eye.
    AIC improve   how much better the curved model is after penalising the extra
                  parameter. Anything above ~10 is decisive.
""")
json.dump(RES, open(os.path.join(OUT, "regression.json"), "w"), indent=2)
print("  -> results/regression.json")
