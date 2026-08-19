"""One picture explaining the whole project, using darts."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

rng = np.random.default_rng(3)
fig, axes = plt.subplots(1, 3, figsize=(15, 6.4))
COL = ["#1a73e8", "#0b8043", "#e8871a", "#b3261e"]
NAMES = ["great aim", "good aim", "ok aim", "wild aim"]
SPREAD = [0.14, 0.28, 0.50, 0.85]

# ---------------- panel 1
ax = axes[0]
ax.set_title("MULTIPLE CHOICE\nyou only learn: right or wrong", fontsize=12.5, fontweight="bold", pad=14)
for i in range(4):
    for k, lab in enumerate("ABCD"):
        hit = (k == [2, 2, 0, 1][i])
        ax.add_patch(plt.Rectangle((k*1.1, -i*1.05), 0.92, 0.72,
                     facecolor=COL[i] if hit else "#f1f3f4", edgecolor="#9aa0a6", lw=1))
        ax.text(k*1.1+0.46, -i*1.05+0.36, lab, ha="center", va="center",
                fontsize=11, color="white" if hit else "#5f6368", fontweight="bold")
    ax.text(-0.30, -i*1.05+0.36, f"friend {i+1}", ha="right", va="center", fontsize=10, color=COL[i])
ax.text(2.25, -4.85, "the right answer is C", ha="center", fontsize=10.5, style="italic")
ax.text(2.25, -5.55, "friends 1 and 2 hit   ·   friends 3 and 4 missed", ha="center", fontsize=10.5)
ax.text(2.25, -7.15, "That is ALL you get.\nA miss is just a miss — you never\nlearn how CLOSE the miss was.",
        ha="center", va="center", fontsize=11, color="#b3261e", linespacing=1.5)
ax.set_xlim(-1.9, 4.9); ax.set_ylim(-8.4, 0.9); ax.axis("off")

# ---------------- panel 2
ax = axes[1]
ax.set_title("OPEN-ENDED\nyou see WHERE each answer landed", fontsize=12.5, fontweight="bold", pad=14)
for r, a in [(1.35, 0.07), (0.90, 0.11), (0.45, 0.16)]:
    ax.add_patch(Circle((0, 0), r, facecolor="#5f6368", alpha=a, edgecolor="none"))
for i in range(4):
    pts = rng.normal(0, SPREAD[i], size=(3, 2))
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = np.where(n > 1.5, pts/n*1.5, pts)          # keep everything on screen
    ax.scatter(pts[:, 0], pts[:, 1], s=70, color=COL[i], edgecolor="white", lw=1.3,
               zorder=4, label=f"friend {i+1} — {NAMES[i]}")
ax.plot(0, 0, "*", ms=26, color="#fbbc04", markeredgecolor="#202124", mew=1.3, zorder=6)
ax.annotate("the true answer", xy=(0, 0), xytext=(0.05, -1.78), ha="center", fontsize=10, style="italic",
            arrowprops=dict(arrowstyle="->", color="#5f6368", lw=1))
ax.text(0, 1.80, "tight cluster = good aim\nscattered = bad aim", ha="center", fontsize=11,
        color="#0b8043", linespacing=1.5)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), fontsize=9, frameon=False, ncol=2,
          handletextpad=0.4, columnspacing=1.2)
ax.set_xlim(-2.0, 2.0); ax.set_ylim(-3.0, 2.35); ax.set_aspect("equal"); ax.axis("off")

# ---------------- panel 3
ax = axes[2]
ax.set_title("TRIANGULATION\nnobody hit it — but they still point at it", fontsize=12.5, fontweight="bold", pad=14)
ax.add_patch(Circle((0, 0), 1.25, facecolor="#5f6368", alpha=0.07, edgecolor="none"))
darts = np.array([[0.42, 0.46], [-0.50, 0.40], [0.16, -0.62], [-0.90, -0.78]])
w = np.array([4.0, 3.0, 2.0, 0.6]); w = w / w.sum()
for i, (p, c) in enumerate(zip(darts, COL)):
    ax.add_patch(FancyArrowPatch(p, (0.02, 0.02), arrowstyle="-|>", mutation_scale=12,
                 color=c, lw=0.9 + 3.4*w[i], alpha=0.5, zorder=3,
                 shrinkA=9, shrinkB=16))
    ax.scatter(*p, s=105, color=c, edgecolor="white", lw=1.5, zorder=5)
ax.plot(0, 0, "*", ms=26, color="#fbbc04", markeredgecolor="#202124", mew=1.3, zorder=6)
est = (darts * w[:, None]).sum(0)
ax.plot(*est, "o", ms=17, mfc="none", mec="#0b8043", mew=2.8, zorder=7)
ax.annotate("what we compute", xy=est, xytext=(1.05, 1.32), fontsize=10.5, color="#0b8043",
            fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color="#0b8043", lw=1.4))
ax.text(0, -2.05, "Every dart missed the bullseye.\nBut they scatter AROUND it, so together\nthey still reveal roughly where it is.",
        ha="center", fontsize=11, color="#0b8043", linespacing=1.5)
ax.text(0, -2.78, "thicker arrow = better thrower = pulls harder", ha="center", fontsize=9.5, style="italic")
ax.set_xlim(-2.0, 2.0); ax.set_ylim(-3.0, 2.35); ax.set_aspect("equal"); ax.axis("off")

fig.suptitle("Why open-ended answers need a different kind of voting",
             fontsize=15, fontweight="bold", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("results/figures/EXPLAIN_darts.png", dpi=150, facecolor="white")
print("ok")
