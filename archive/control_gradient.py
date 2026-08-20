"""How much of R^2=0.99 was just 'positives are on-topic, controls are off-topic'?

Sweep control hardness continuously: use the r-th nearest OTHER question's
hallucinated answer as the control, for r from 1 (hardest) to random (easiest).
If the log-linear law is real at fine scale, beta and R^2 should be stable.
If the first result was a topic-matching artefact, both collapse as r -> 1.
"""
import os, json, numpy as np, pandas as pd, requests
HERE = os.path.dirname(os.path.abspath(__file__))
from robustness_test import load, get_encoder, fit_loglinear, ENCODERS, SPLITS

N = 3000
enc = get_encoder(ENCODERS[0])
rows = {}
for cfg in ("qa", "dialogue"):
    tc, wc = SPLITS[cfg]
    d = load(cfg).dropna(subset=[tc, wc]).reset_index(drop=True).iloc[:N]
    Et, Ew = enc(d[tc]), enc(d[wc])
    pos = np.einsum("ij,ij->i", Ew, Et)
    sim = Et @ Et.T; np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)                       # nearest -> farthest questions
    print(f"\n=== {cfg}  (n={len(d)}, mean sim of a real error to its truth = {pos.mean():.3f})")
    print(f"  {'control = r-th nearest Q':>26}{'ctl mean sim':>14}{'beta':>9}{'R2':>8}{'R2 quad':>10}")
    print("  " + "-" * 67)
    out = []
    for r in [1, 2, 5, 10, 50, 200, 1000, "random"]:
        if r == "random":
            rng = np.random.default_rng(0); K = 8; c = []
            for _ in range(K):
                p = rng.permutation(len(d)); s = p == np.arange(len(d)); p[s] = (p[s]+1) % len(d)
                c.append(np.einsum("ij,ij->i", Ew[p], Et))
            ctl = np.concatenate(c)
        else:
            K = 8
            idx = order[:, (r-1):(r-1+K)]
            ctl = np.concatenate([np.einsum("ij,ij->i", Ew[idx[:, j]], Et) for j in range(idx.shape[1])])
        f = fit_loglinear(pos, ctl, K)
        lab = f"r={r}" if r != "random" else "random"
        if f.get("failed"):
            print(f"  {lab:>26}{ctl.mean():>14.3f}{'--':>9}{'--':>8}{'--':>10}")
        else:
            print(f"  {lab:>26}{ctl.mean():>14.3f}{f['slope']:>9.2f}{f['r2']:>8.3f}{f['r2_quad']:>10.3f}")
            out.append(dict(r=str(r), ctl_mean=float(ctl.mean()), **{k: f[k] for k in ("slope","r2","r2_quad")}))
    rows[cfg] = out

# contamination check: are the hardest controls actually near-correct answers?
tc, wc = SPLITS["qa"]
d = load("qa").dropna(subset=[tc, wc]).reset_index(drop=True).iloc[:N]
Et = enc(d[tc]); sim = Et @ Et.T; np.fill_diagonal(sim, -np.inf)
nn1 = sim.max(axis=1)
print(f"\n=== contamination check (qa)")
print(f"  similarity between a question's TRUTH and its NEAREST other question's TRUTH:")
print(f"    mean {nn1.mean():.3f}   frac > 0.8 : {100*(nn1>0.8).mean():.1f}%   frac > 0.9 : {100*(nn1>0.9).mean():.1f}%")
print("  If that fraction is large, the 'hard controls' include near-duplicate questions")
print("  whose answers are effectively CORRECT -- which would flatten beta artificially.")
json.dump({"gradient": rows, "nn_truth_sim_mean": float(nn1.mean()),
           "frac_gt_0.8": float((nn1>0.8).mean()), "frac_gt_0.9": float((nn1>0.9).mean())},
          open(os.path.join(HERE,"results","control_gradient.json"),"w"), indent=2)
print("\n  -> results/control_gradient.json")
