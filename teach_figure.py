"""Two teaching figures: how to READ the graphs, using the real data."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results", "figures")
BLUE, GREEN, RED, GREY, ORANGE = "#1a73e8", "#0b8043", "#b3261e", "#9aa0a6", "#e8871a"

# ---------------------------------------------------------------- real data
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
d = pd.read_parquet(f"{HERE}/data/halueval_qa.parquet").dropna(
    subset=["right_answer", "hallucinated_answer"]).reset_index(drop=True).iloc[:3000]
E = lambda t: m.encode(list(t), batch_size=128, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
Et, Ew = E(d["right_answer"]), E(d["hallucinated_answer"])
pos = np.einsum("ij,ij->i", Ew, Et)
sim = Et @ Et.T; np.fill_diagonal(sim, -np.inf); order = np.argsort(-sim, axis=1)
K = 8
def block(r):
    if r == "random":
        rng = np.random.default_rng(0); cols = []
        for _ in range(K):
            p = rng.permutation(len(Et)); s = p == np.arange(len(Et)); p[s] = (p[s]+1) % len(Et)
            cols.append(np.einsum("ij,ij->i", Ew[p], Et))
        return np.stack(cols, 1).ravel()
    idx = order[:, (r-1):(r-1+K)]
    return np.stack([np.einsum("ij,ij->i", Ew[idx[:, j]], Et) for j in range(K)], 1).ravel()

def bins(pos, ctl, nb=10):
    e = np.quantile(np.concatenate([pos, ctl]), np.linspace(0.02, 0.98, nb+1))
    c, lo, np_, nc_ = [], [], [], []
    for a, b in zip(e[:-1], e[1:]):
        p = int(((pos>=a)&(pos<b)).sum()); q = int(((ctl>=a)&(ctl<b)).sum())
        if p>=5 and q>=5:
            c.append((a+b)/2); lo.append(np.log(p/q)+np.log(K)); np_.append(p); nc_.append(q)
    return np.array(c), np.array(lo), np_, nc_

c_hard, lo_hard, nph, nch = bins(pos, block(1))
c_easy, lo_easy, npe, nce = bins(pos, block("random"))

# ============================================================ FIGURE 1
fig = plt.figure(figsize=(17.5, 6.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.95, 1.25], wspace=0.42)

# --- A: what we measure
ax = fig.add_subplot(gs[0]); ax.axis("off")
ax.set_title("STEP 1.  What we measure", fontsize=13, fontweight="bold", pad=22)
ax.text(0.5, 0.94, 'The right answer is  "Canberra"', ha="center", fontsize=11.5,
        style="italic", transform=ax.transAxes)
rows = [("Sydney", 0.62, 0.80, ORANGE, "an Australian city —\nvery easy to confuse"),
        ("Melbourne", 0.55, 0.45, ORANGE, "also a city, a bit\nless tempting"),
        ("photosynthesis", 0.04, 0.02, GREY, "nothing to do with it —\nnobody says this")]
for i, (word, s, freq, col, note) in enumerate(rows):
    y = 0.665 - i*0.225
    ax.text(0.03, y+0.045, f'"{word}"', fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.03, y-0.055, note, fontsize=8.4, color="#5f6368", transform=ax.transAxes, linespacing=1.4)
    ax.add_patch(FancyBboxPatch((0.46, y+0.025), 0.5*s, 0.045, boxstyle="round,pad=0.004",
                 fc=col, ec="none", transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((0.46, y-0.045), 0.5*freq, 0.045, boxstyle="round,pad=0.004",
                 fc=col, ec="none", alpha=0.5, transform=ax.transAxes))
    ax.text(0.98, y+0.045, f"{s:.2f}", fontsize=8.5, ha="right", transform=ax.transAxes)
ax.add_patch(FancyBboxPatch((0.46, 0.862), 0.05, 0.028, boxstyle="round,pad=0.003", fc=ORANGE, ec="none", transform=ax.transAxes))
ax.text(0.53, 0.862, "how SIMILAR to the right answer", fontsize=8.6, color="#202124", transform=ax.transAxes)
ax.add_patch(FancyBboxPatch((0.46, 0.815), 0.05, 0.028, boxstyle="round,pad=0.003", fc=ORANGE, alpha=0.5, ec="none", transform=ax.transAxes))
ax.text(0.53, 0.815, "how OFTEN a model actually says it", fontsize=8.6, color="#5f6368", transform=ax.transAxes)
ax.text(0.5, 0.05, "The two bars move together.\nThat is the whole idea we are testing.",
        ha="center", fontsize=10.5, color=GREEN, transform=ax.transAxes, fontweight="bold", linespacing=1.5)

# --- B: how one dot is made
ax = fig.add_subplot(gs[1]); ax.axis("off")
ax.set_title("STEP 2.  How ONE dot is made", fontsize=13, fontweight="bold", pad=22)
k = 7
ax.text(0.5, 0.90, f"Take every answer whose similarity\nis about {c_hard[k]:.2f}", ha="center",
        fontsize=10.5, transform=ax.transAxes, linespacing=1.5)
ax.add_patch(FancyBboxPatch((0.06, 0.55), 0.38, 0.22, boxstyle="round,pad=0.012",
             fc="#e8f0fe", ec=BLUE, lw=1.4, transform=ax.transAxes))
ax.text(0.25, 0.70, f"{nph[k]}", ha="center", fontsize=20, fontweight="bold", color=BLUE, transform=ax.transAxes)
ax.text(0.25, 0.60, "were REALLY\nsaid by a model", ha="center", fontsize=9, transform=ax.transAxes, linespacing=1.4)
ax.add_patch(FancyBboxPatch((0.56, 0.55), 0.38, 0.22, boxstyle="round,pad=0.012",
             fc="#f1f3f4", ec=GREY, lw=1.4, transform=ax.transAxes))
ax.text(0.75, 0.70, f"{nch[k]}", ha="center", fontsize=20, fontweight="bold", color="#5f6368", transform=ax.transAxes)
ax.text(0.75, 0.60, "were NOT\n(our comparison group)", ha="center", fontsize=9, transform=ax.transAxes, linespacing=1.4)
ax.text(0.5, 0.45, f"divide:   {nph[k]} ÷ {nch[k]}  →  a ratio", ha="center", fontsize=10.5,
        transform=ax.transAxes, fontfamily="monospace")
ax.text(0.5, 0.355, "then take the log of it, which just\nsquashes big numbers so they plot nicely",
        ha="center", fontsize=9, color="#5f6368", transform=ax.transAxes, linespacing=1.4)
ax.annotate("", xy=(0.5, 0.20), xytext=(0.5, 0.30), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color=BLUE))
ax.plot([0.5], [0.13], "o", ms=15, color=BLUE, transform=ax.transAxes, clip_on=False)
ax.text(0.5, 0.03, "ONE DOT on the next graph", ha="center", fontsize=11,
        fontweight="bold", color=BLUE, transform=ax.transAxes)

# --- C: the real dots
ax = fig.add_subplot(gs[2])
ax.set_title("STEP 3.  All the dots, and two\nways to draw through them",
             fontsize=13, fontweight="bold", pad=10)
ax.plot(c_hard, lo_hard, "o", ms=11, color=BLUE, zorder=5, label="the real dots")
sl, ic = np.polyfit(c_hard, lo_hard, 1)
q = np.polyfit(c_hard, lo_hard, 2)
xs = np.linspace(c_hard.min(), c_hard.max(), 200)
sst = ((lo_hard-lo_hard.mean())**2).sum()
r2l = 1-((lo_hard-(sl*c_hard+ic))**2).sum()/sst
r2q = 1-((lo_hard-np.polyval(q,c_hard))**2).sum()/sst
ax.plot(xs, sl*xs+ic, "-", lw=2.6, color=RED, label=f"a STRAIGHT line — misses a lot (score {r2l:.2f})")
ax.plot(xs, np.polyval(q, xs), "-", lw=2.6, color=GREEN, label=f"a CURVE — hugs the dots (score {r2q:.2f})")
ax.plot([c_hard[k]], [lo_hard[k]], "o", ms=17, mfc="none", mec=BLUE, mew=2.6, zorder=6)
ax.annotate("the dot we just built", xy=(c_hard[k], lo_hard[k]), xytext=(0.30, 0.16),
            textcoords="axes fraction", fontsize=9.5, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
ax.set_xlabel("→  more similar to the right answer", fontsize=10.5)
ax.set_ylabel("→  more likely to really be said", fontsize=10.5)
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
ax.text(0.98, 0.03, "score = how much of the pattern the line captures\n1.00 = perfect,  0 = useless",
        transform=ax.transAxes, ha="right", fontsize=8.5, color="#5f6368", style="italic", linespacing=1.5)
fig.suptitle("How to read these graphs", fontsize=15, fontweight="bold", y=0.995)
fig.subplots_adjust(left=0.035, right=0.975, top=0.80, bottom=0.10)
fig.savefig(f"{FIG}/T1_how_to_read.png", dpi=150, facecolor="white")
plt.close(fig); print("T1_how_to_read.png")

# ============================================================ FIGURE 2
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
for ax, (c, lo, ttl, sub) in zip(axes[:2], [
    (c_easy, lo_easy, "EASY test", "compare against answers to totally\nunrelated questions"),
    (c_hard, lo_hard, "HARD test", "compare against answers to very\nSIMILAR questions")]):
    sl, ic = np.polyfit(c, lo, 1); q = np.polyfit(c, lo, 2)
    xs = np.linspace(c.min(), c.max(), 200)
    sst = ((lo-lo.mean())**2).sum()
    r2l = 1-((lo-(sl*c+ic))**2).sum()/sst; r2q = 1-((lo-np.polyval(q,c))**2).sum()/sst
    ax.plot(c, lo, "o", ms=10, color=BLUE, zorder=5)
    ax.plot(xs, sl*xs+ic, "-", lw=2.6, color=RED)
    ax.plot(xs, np.polyval(q, xs), "-", lw=2.2, color=GREEN, alpha=0.9)
    ax.set_title(f"{ttl}\n{sub}", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("more similar →"); ax.grid(alpha=0.25)
    ax.text(0.03, 0.95, f"straight line score: {r2l:.2f}", transform=ax.transAxes,
            fontsize=11, color=RED, fontweight="bold", va="top")
    ax.text(0.03, 0.86, f"curve score:  {r2q:.2f}", transform=ax.transAxes,
            fontsize=11, color=GREEN, fontweight="bold", va="top")
axes[0].set_ylabel("more likely to really be said →")
axes[0].text(0.5, 0.06, "straight line works fine here", transform=axes[0].transAxes,
             ha="center", fontsize=10.5, color=GREEN, fontweight="bold")
axes[1].text(0.5, 0.06, "straight line clearly misses", transform=axes[1].transAxes,
             ha="center", fontsize=10.5, color=RED, fontweight="bold")

ax = axes[2]
ax.set_title("STEP 4.  The summary graph\neach dot here = one WHOLE graph on the left",
             fontsize=11.5, fontweight="bold")
xs_ = [0, 1]
ax.plot(xs_, [0.685, 0.991], "o-", color=RED, lw=2.6, ms=13, label="straight-line score")
ax.plot(xs_, [0.987, 0.991], "s-", color=GREEN, lw=2.6, ms=11, label="curve score")
ax.set_xticks(xs_); ax.set_xticklabels(["HARD\ntest", "EASY\ntest"], fontsize=11)
ax.set_ylim(0, 1.08); ax.set_ylabel("score (1.00 = perfect fit)")
ax.grid(alpha=0.25); ax.legend(fontsize=9.5, loc="lower right")
ax.text(0.52, 0.42, "The green line stays high.\nThe red line falls off a cliff\nwhen the test gets hard.\n\nThat is the whole result.",
        transform=ax.transAxes, ha="center", fontsize=11, color="#202124",
        fontweight="bold", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5", fc="#fef7e0", ec="#f9ab00", lw=1.2))
fig.suptitle("Why the test had to get harder — and what broke when it did",
             fontsize=15, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{FIG}/T2_easy_vs_hard.png", dpi=150, facecolor="white")
print("T2_easy_vs_hard.png")
