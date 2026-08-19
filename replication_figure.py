"""Figure: does the pattern replicate across three independent datasets?"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{HERE}/results/triple_replication.json"))
GRID = ["1", "5", "50", "1000", "random"]
LBL = ["1\n(hardest)", "5", "50", "1000", "random\n(easiest)"]
RED, GREEN, BLUE = "#b3261e", "#0b8043", "#1a73e8"

fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharex=True)
for col, (ds, D) in enumerate(R.items()):
    encs = list(D["encoders"])
    # --- top row: R^2 straight vs curved, averaged over encoders, with spread
    ax = axes[0, col]
    for key, colr, lab, mk in [("r2", RED, "straight line", "o-"),
                               ("r2_quad", GREEN, "allowing curvature", "s-")]:
        M, LO, HI = [], [], []
        for g in GRID:
            vals = [D["encoders"][e]["grid"][g][key] for e in encs if g in D["encoders"][e]["grid"]]
            cis = [D["encoders"][e]["grid"][g]["ci_r2" if key == "r2" else "ci_r2q"]
                   for e in encs if g in D["encoders"][e]["grid"]]
            if not vals: M.append(np.nan); LO.append(np.nan); HI.append(np.nan); continue
            M.append(np.mean(vals)); LO.append(np.mean([c[0] for c in cis])); HI.append(np.mean([c[1] for c in cis]))
        x = np.arange(len(GRID))
        ax.plot(x, M, mk, color=colr, lw=2, ms=6, label=lab)
        ax.fill_between(x, LO, HI, color=colr, alpha=0.15)
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.25)
    ax.set_title(f"{ds}\n{D['provenance']}  (n={D['n']})", fontsize=10.5, fontweight="bold")
    if col == 0:
        ax.set_ylabel("goodness of fit  (R²)")
        ax.legend(fontsize=8.5, loc="lower right")
    ax.axvspan(-0.4, 0.5, color=RED, alpha=0.07)

    # --- bottom row: fitted beta
    ax = axes[1, col]
    for e in encs:
        ys = [D["encoders"][e]["grid"][g]["slope"] if g in D["encoders"][e]["grid"] else np.nan for g in GRID]
        ax.plot(np.arange(len(GRID)), ys, "^--", lw=1.4, ms=5, label=e.replace("all-", "").replace("-en-v1.5", ""))
    ax.set_xticks(np.arange(len(GRID))); ax.set_xticklabels(LBL, fontsize=8.5)
    ax.set_xlabel("controls = r-th nearest other question")
    ax.grid(alpha=0.25); ax.axvspan(-0.4, 0.5, color=RED, alpha=0.07)
    if col == 0: ax.set_ylabel("fitted β")
    ax.legend(fontsize=7.5)

fig.suptitle("Same protocol, three independent datasets, three encoders each\n"
             "Where the wrong answers COME FROM decides whether a straight line works — "
             "it fails for model-made errors, holds for hand-written exam distractors",
             fontsize=12.5, fontweight="bold")
axes[0, 2].text(0.5, 0.12, "EXCEPTION: straight line fits fine here\n(distractors written independently,\nnot engineered to be confusable)",
                transform=axes[0, 2].transAxes, ha="center", fontsize=8.5, color="#0b8043",
                bbox=dict(boxstyle="round,pad=0.4", fc="#e6f4ea", ec="#0b8043", lw=0.8))
for ax in (axes[0, 0], axes[0, 1]):
    ax.text(0.5, 0.12, "curvature REQUIRED\n(bootstrap CIs disjoint)", transform=ax.transAxes,
            ha="center", fontsize=8.5, color="#b3261e",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fce8e6", ec="#b3261e", lw=0.8))
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(f"{HERE}/results/figures/R3_triple_replication.png", dpi=150, facecolor="white")
print("wrote R3_triple_replication.png")
