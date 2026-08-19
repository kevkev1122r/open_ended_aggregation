"""Build the summary figures from results/results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "results.json")))
FIG = os.path.join(HERE, "results", "figures")

BLUE, GREEN, RED, ORANGE, PURPLE, GREY = (
    "#1a73e8", "#0b8043", "#b3261e", "#e8871a", "#7a5195", "#9aa0a6")


# ---------------------------------------------------------------- fig 1
def fig_main():
    order = ["MV-exact", "OW-L", "OW-oracle", "BestSingle", "MV-cluster", "KWA-EM", "KWA-oracle"]
    vals = [R["E4"][k]["mean"] for k in order]
    sds = [R["E4"][k]["sd"] for k in order]
    cols = [RED, PURPLE, PURPLE, GREY, ORANGE, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    b = ax.bar(order, vals, yerr=sds, color=cols, capsize=3, width=0.62)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 1.1, f"{v:.1f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(R["E4"]["MV-exact"]["mean"], ls=":", color=RED, lw=1)
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 106)
    ax.set_title("Open-ended aggregation: kernel weighting vs vote-based baselines\n"
                 "(5 agents, 1500 questions, 5 seeds)", fontsize=11)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F1_main_comparison.png"), dpi=150)
    plt.close(fig); print("F1_main_comparison.png")


# ---------------------------------------------------------------- fig 2
def fig_beta():
    Ms = sorted(int(k) for k in R["E2"])
    true = [8.0, 6.0, 4.0, 2.5, 1.5]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.1))
    a1.plot(Ms, [R["E2"][str(m)]["corr"] for m in Ms], "o-", color=BLUE, lw=2)
    a1.set_xscale("log"); a1.set_xlabel("labelled questions used: ZERO\n(M = unlabelled questions)")
    a1.set_ylabel("corr(beta_hat, beta_true)"); a1.set_ylim(0.9, 1.005)
    a1.grid(alpha=0.25); a1.set_title("Label-free recovery of agent skill", fontsize=10)

    bh = R["E2"][str(Ms[-1])]["beta_hat"]
    idx = np.arange(len(true)); w = 0.38
    a2.bar(idx - w/2, true, w, label="true beta", color=GREEN)
    a2.bar(idx + w/2, bh, w, label="EM estimate (no labels)", color=BLUE)
    a2.set_xticks(idx); a2.set_xticklabels([f"agent {i+1}" for i in idx])
    a2.set_ylabel("beta"); a2.legend(fontsize=9); a2.grid(axis="y", alpha=0.25)
    a2.set_title("Ordering is recovered; the STRONGEST agent is\nsystematically under-estimated",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F2_beta_recovery.png"), dpi=150)
    plt.close(fig); print("F2_beta_recovery.png")


# ---------------------------------------------------------------- fig 3
def fig_ceiling():
    e9 = R["E9"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    names = ["MV-cluster", "KWA\nproduced\nanswers", "KWA\nwider\ncandidate set"]
    vals = [e9["MV-cluster"], e9["observed"], e9["full-pool"]]
    a1.bar(names, vals, color=[ORANGE, BLUE, GREEN], width=0.6)
    a1.tick_params(axis="x", labelsize=8.5)
    a1.axhline(e9["ceiling"], ls="--", color=RED, lw=1.5)
    a1.text(0.02, e9["ceiling"] + 1.2, f"ceiling for vote-based methods: {e9['ceiling']:.1f}%",
            color=RED, fontsize=9)
    for i, v in enumerate(vals):
        a1.text(i, v + 1.2, f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
    a1.set_ylabel("accuracy (%)"); a1.set_ylim(0, 105)
    a1.set_title("Weak ensemble, deliberately", fontsize=10); a1.grid(axis="y", alpha=0.25)

    names2 = ["KWA\n(produced answers)", "KWA\n(wider candidate set)"]
    vals2 = [e9["observed_when_all_wrong"], e9["full-pool_when_all_wrong"]]
    a2.bar(names2, vals2, color=[BLUE, GREEN], width=0.5)
    for i, v in enumerate(vals2):
        a2.text(i, v + 1.0, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)
    a2.set_ylabel("accuracy (%)"); a2.set_ylim(0, 50)
    a2.set_title("On questions where EVERY agent was wrong\n"
                 "(any vote-based method scores 0 here by construction)", fontsize=10)
    a2.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F3_triangulation.png"), dpi=150)
    plt.close(fig); print("F3_triangulation.png")


# ---------------------------------------------------------------- fig 4
def fig_robust():
    e7 = R["E7"]
    pulls = [0.0, 1.0, 2.0, 3.0]
    keys = ["MV-exact", "MV-cluster", "OW-L", "KWA-EM"]
    cols = {"MV-exact": RED, "MV-cluster": ORANGE, "OW-L": PURPLE, "KWA-EM": BLUE}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.1))
    for k in keys:
        a1.plot(pulls, [e7[f"family_pull={p}"][k] for p in pulls], "o-",
                color=cols[k], label=k, lw=1.9)
    a1.set_xlabel("strength of shared same-family error"); a1.set_ylabel("accuracy (%)")
    a1.set_title("Correlated agents (violates conditional independence)", fontsize=10)
    a1.legend(fontsize=8); a1.grid(alpha=0.25)

    cs = [0.5, 1.0, 1.5, 2.0]
    for k in keys:
        a2.plot(cs, [e7[f"curvature={c}"][k] for c in cs], "o-",
                color=cols[k], label=k, lw=1.9)
    a2.axvline(1.0, ls=":", color="k", lw=1)
    a2.text(1.02, 62, "model correctly\nspecified here", fontsize=8)
    a2.set_xlabel("curvature of the true log-prob response"); a2.set_ylabel("accuracy (%)")
    a2.set_title("Kernel misspecification (we always fit c = 1)", fontsize=10)
    a2.legend(fontsize=8); a2.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4_robustness.png"), dpi=150)
    plt.close(fig); print("F4_robustness.png")


if __name__ == "__main__":
    fig_main(); fig_beta(); fig_ceiling(); fig_robust()
    print("all figures ->", FIG)
