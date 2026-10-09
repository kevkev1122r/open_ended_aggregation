"""
Paper figures. Reproducible from results/*.json, no API calls, no re-fitting.

Built with the scientific-visualization skill's style presets, Okabe-Ito
palette and provenance-tracking exporter, so every PDF ships a sidecar
manifest naming the source file and the transformations applied.

MAIN BODY
  fig0_hierarchy        two real claims that |S(c)| and S(c) cannot tell apart and
                        that B(c) separates. The decomposition, on measured data.
  fig1_worked_example   one held-out question end to end. Defines C(q), S(c),
                        r_m(c) and L_m(q) against something concrete.
  fig2_ceiling          the support-pattern partition, on real data. Spread
                        within a support count is what weighting can add over
                        counting; the right panel shows where the claims are.
  fig3_budget           how k is chosen, on the same question as fig1, beside
                        a question where the same rule misses.
  fig4_mirror           every ensemble on both benchmarks.

APPENDIX
  figA1_position, figA2_calibration, figA3_learning_curve

REFERENCE ARM. Multi-agent counting is the baseline and is drawn as the zero
line or reference wherever a comparison is made. Reliability weighting is a
member of the class bounded by the support-pattern ceiling, not the yardstick.

Figure text is descriptive. Takeaways belong in the LaTeX captions.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.make_figures
Output: docs/paper/figur/*.pdf (+ .png, + .manifest.json)
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

SKILL = Path.home() / ".claude/skills/scientific-visualization"
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL / "assets"))
from style_presets import style_context          # noqa: E402
from figure_export import export_figure          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "paper" / "figur"

FULL = 5.5                       # NeurIPS text block width, inches

METHOD = "output-structure scorer + budget"
CEILING = "support-pattern ceiling"
WEIGHT = "reliability weighting"
BASE = "multi-agent counting"

C = {"base": "#000000", "weight": "#D55E00", "ceiling": "#CC79A7",
     "method": "#0072B2", "grid": "#BBBBBB", "slot": "#56B4E9",
     "fixed": "#EEEEEE", "note": "#6E6E6E", "good": "#009E73"}
STALE = "pre-parsing-fix input; pending regeneration"

SHORT = {"Cohere-command-a-plus-05-2026": "Cohere", "DeepSeek-V4-Flash": "DeepSeek",
         "Kimi-K2.5": "Kimi", "MAI-Thinking-1": "MAI", "gpt-5.4": "gpt-5.4",
         "gpt-5.4-mini": "mini", "gpt-5.4-nano": "nano", "grok-4.3": "grok"}


def _save(fig, name, provenance):
    OUT.mkdir(parents=True, exist_ok=True)
    export_figure(fig, str(OUT / name), formats=["pdf", "png"], dpi=600,
                  bbox_inches=None, facecolor="white", font_mode="truetype",
                  overwrite=True, mkdir=True, provenance=provenance,
                  write_manifest=True)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --------------------------------------------------------------- figure 0 ---
def fig_hierarchy():
    """Two real QAMPARI claims every published rule must treat identically, and
    that generation behaviour separates. Data: results/hierarchy_pair.json."""
    d = json.loads((RESULTS / "hierarchy_pair.json").read_text())
    A, B = d["claims"]
    pool = d["pool"]
    cols = d["col_order"]          # fixed across both blocks: same column = same agent
    n = len(pool)

    def kfmt(v):
        return f"{v/1000:.1f}k" if v >= 1000 else str(v)

    LAB, GAP = 4.7, 1.6
    W = LAB + n + GAP + n                     # 22.3
    H = 6.70
    ROWY = (4.20, 2.70, 1.05)                 # count, identity, behaviour
    HB = 0.42                                 # half box height
    YSEP = (ROWY[1] - HB + ROWY[2] + HB) / 2  # between identity and behaviour

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, ax = plt.subplots(figsize=(FULL, FULL * H / W), layout="constrained")
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_aspect("equal"); ax.axis("off")

        for c, X in ((A, LAB), (B, LAB + n + GAP)):
            ok = c["correct"]
            col = C["good"] if ok else C["weight"]
            ax.text(X, H - 0.55, f'claim {c["label"]}:  "{c["name"]}"',
                    fontsize=6.6, fontweight="bold", color=col)
            ax.text(X + n, H - 0.55, "correct" if ok else "wrong", fontsize=6.0,
                    ha="right", color=col, fontweight="bold")
            ax.text(X, H - 1.10, f'"{c["question"][:46]}"', fontsize=5.0,
                    color=C["note"], style="italic")
            for j, m in enumerate(cols):
                ax.text(X + j + 0.5, ROWY[1] + HB + 0.30, SHORT[m], fontsize=4.2,
                        ha="center", va="center", color=C["note"], rotation=38)

            for kind, y in zip(("count", "ident", "behav"), ROWY):
                if kind == "count":
                    ax.add_patch(FancyBboxPatch(
                        (X + 0.06, y - HB), 0.88, 2 * HB,
                        boxstyle="round,pad=0.02", linewidth=0.7,
                        edgecolor=C["base"], facecolor=C["fixed"]))
                    ax.text(X + 0.5, y, "1", fontsize=7.5, ha="center",
                            va="center", fontweight="bold")
                    continue
                for j, m in enumerate(cols):
                    x, a = X + j, (m == c["asserter"])
                    ax.add_patch(FancyBboxPatch(
                        (x + 0.06, y - HB), 0.88, 2 * HB,
                        boxstyle="round,pad=0.02", linewidth=0.7,
                        edgecolor=C["base"] if a else C["grid"],
                        facecolor=C["slot"] if a else "white"))
                    if kind == "ident":
                        if a:
                            ax.text(x + 0.5, y, "yes", fontsize=5.0,
                                    ha="center", va="center")
                    elif a:
                        ax.text(x + 0.5, y + 0.11, str(c["pos"]), fontsize=6.4,
                                ha="center", va="center", fontweight="bold")
                        ax.text(x + 0.5, y - 0.25, f'of {c["L"]}', fontsize=4.4,
                                ha="center", va="center", color=C["method"])
                    else:
                        ax.text(x + 0.5, y + 0.11, "·", fontsize=6,
                                ha="center", va="center", color=C["note"])
                        ax.text(x + 0.5, y - 0.25, kfmt(c["lengths"][m]),
                                fontsize=4.4, ha="center", va="center",
                                color=C["note"])

        for (lead, sym), y in zip((
                ("majority voting reads", r"$|S(c)|$"),
                ("reliability weighting reads", r"$S(c)$"),
                ("we read", r"$(S(c),\,B(c))$")), ROWY):
            ax.text(LAB - 0.4, y + 0.26, lead, fontsize=5.4, ha="right",
                    va="center", color=C["note"])
            ax.text(LAB - 0.4, y - 0.32, sym, fontsize=6.6, ha="right",
                    va="center")

        ax.plot([0.2, W - 0.2], [YSEP, YSEP], linewidth=0.6, color=C["base"],
                linestyle=(0, (4, 3)))

        _save(fig, "fig0_hierarchy", {
            "source": "results/hierarchy_pair.json",
            "derived_from": "data/qampari_asc800.jsonl, 8 agents, 777 full-pool questions",
            "transform": "two solo claims asserted by the same agent; positions and "
                         "per-agent output lengths parsed after the special-token "
                         "fix of Section 4.4",
        })


# --------------------------------------------------------------- figure 2 ---
def fig_ceiling_v2():
    """Two bar charts on one axis. Where the claims are, the headroom is not."""
    d = json.loads((RESULTS / "pattern_cells.json").read_text())
    ps = {r["k"]: r for r in d["per_support_count"]}
    ks = sorted(ps)
    claims = [ps[k]["claims"] for k in ks]
    head = [ps[k]["spread_ge30"] or 0.0 for k in ks]
    total = sum(claims)
    cw = {int(k): v for k, v in
          json.loads((RESULTS / "cwq_ceiling.json").read_text()).items()}
    kw = sorted(cw)
    cw_claims = [cw[k][0] for k in kw]
    cw_head = [cw[k][2] for k in kw]

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(FULL, 3.0), layout="constrained",
            gridspec_kw=dict(height_ratios=[1, 1], hspace=0.10))

        ax1.bar([k - 0.17 for k in ks], claims, width=0.32, color=C["base"],
                zorder=2, label="QAMPARI")
        ax1.bar([k + 0.17 for k in kw], cw_claims, width=0.32, color=C["good"],
                zorder=2, label="CWQ")
        ax1.legend(loc="upper right", fontsize=5.6, frameon=False)
        ax1.set_yscale("log"); ax1.set_ylim(100, 4e5)
        ax1.set_ylabel("claims\n(log scale)")
        for k, v in zip(ks, claims):
            ax1.text(k - 0.17, v * 1.5, f"{v:,}", fontsize=4.4, ha="center",
                     color=C["note"])
        for k, v in zip(kw, cw_claims):
            ax1.text(k + 0.17, v * 1.5, f"{v:,}", fontsize=4.4, ha="center",
                     color=C["note"])


        ax2.bar([k - 0.17 for k in ks], head, width=0.32, color=C["ceiling"],
                zorder=2)
        ax2.bar([k + 0.17 for k in kw], cw_head, width=0.32, color=C["good"],
                zorder=2)
        ax2.set_ylim(0, 72); ax2.set_yticks([0, 20, 40, 60])
        ax2.set_ylabel("headroom over counting\n(percentage points)")
        ax2.set_xlabel("support count $|S(c)|$")
        ax2.set_xticks(ks)
        for k, v in zip(ks, head):
            if v:
                ax2.text(k - 0.17, v + 2.2, f"{v:.0f}", fontsize=4.6,
                         ha="center", color=C["note"])
        for k, v in zip(kw, cw_head):
            if v:
                ax2.text(k + 0.17, v + 2.2, f"{v:.0f}", fontsize=4.6,
                         ha="center", color=C["note"])
        for ax in (ax1, ax2):
            ax.spines[["top", "right"]].set_visible(False)
        _save(fig, "fig2_ceiling", {
            "source": "results/pattern_cells.json; results/cwq_ceiling.json",
            "transform": "upper: claims per support count. lower: max minus min "
                         "correctness over support patterns holding >=30 claims "
                         "at that support count, which is what a rule reading "
                         "S(c) can exploit over one reading |S(c)| alone.",
        })


# --------------------------------------------------------------- figure 4 ---
def fig_regimes():
    """Counting against the best single agent on both benchmarks, with the
    statistic that predicts the sign before either method is built."""
    q = json.loads((RESULTS / "ensemble_sizes.json").read_text())
    a = json.loads((RESULTS / "asqa_ours_ensembles.json").read_text())
    w = json.loads((RESULTS / "cwq_kg_ensembles.json").read_text())

    def series(rows, f):
        by = collections.defaultdict(list)
        for r in rows:
            by[r["k"]].append(f(r))
        ks = sorted(by)
        return ks, [float(np.mean(by[k])) for k in ks], by

    kq, vq, byq = series(q, lambda r: r["cnt_vs_best"])
    ka, va, bya = series(a, lambda r: r["count"] - r["best"])
    kw, vw, byw = series(w, lambda r: r["cnt_vs_best"])

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, ax = plt.subplots(figsize=(FULL, 2.6), layout="constrained")
        ax.axhline(0, color=C["base"], linewidth=0.9, linestyle="--")
        ax.text(6.42, 0.12, "best single agent", fontsize=5.4, color=C["base"],
                ha="right")
        for ks, vs, by, col, mk, lbl in (
                (kq, vq, byq, C["method"], "o", "QAMPARI"),
                (kw, vw, byw, C["good"], "^", "CWQ"),
                (ka, va, bya, C["weight"], "s", "ASQA")):
            for k in ks:
                ax.scatter([k] * len(by[k]), by[k], s=5, color=col, alpha=0.20,
                           edgecolors="none", zorder=1)
            ax.plot(ks, vs, color=col, marker=mk, markersize=4, linewidth=1.5,
                    zorder=3, label=lbl)
        ax.set_xlabel("agents in the ensemble")
        ax.set_ylabel("multi-agent counting\nminus best single agent")
        ax.set_xticks([3, 4, 5, 6])
        ax.legend(loc="upper left", fontsize=6.0, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, "fig4_regimes", {
            "source": "results/ensemble_sizes.json (QAMPARI, 210 ensembles), "
                      "results/asqa_ours_ensembles.json (ASQA, 42 ensembles), "
                      "results/cwq_kg_ensembles.json (CWQ, 16 ensembles)",
            "transform": "per ensemble, counting minus the best single agent in "
                         "that ensemble; line is the mean over ensembles of each "
                         "size. Solo ratios are Table 9. QAMPARI arm predates the "
                         "parsing fix of Section 4.4.",
        })


# --------------------------------------------------------------- figure 1 ---
def fig_template():
    """The five-stage pipeline. Fills the figure slot the paper already has,
    whose caption describes a template diagram rather than a worked example."""
    stages = [("1\npool to\ncandidates", False), ("2\nsupport\npattern", False),
              ("3\nconfidence\nsignal", True), ("4\ncross-fitted\nlogistic", False),
              ("5\ndecision\nrule", True)]
    fills = {2: ("determined by answer format",
                 [("QAMPARI", "list rank, omission"), ("ASQA", "sentence position"),
                  ("TriviaQA", "self-consistency")]),
             4: ("determined by metric shape",
                 [("QAMPARI", "per-question budget"), ("ASQA", "swept threshold"),
                  ("TriviaQA", "argmax")])}

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, ax = plt.subplots(figsize=(FULL, 2.35), layout="constrained")
        ax.set_xlim(0, 10.6)
        ax.set_ylim(1.15, 5.35)
        ax.axis("off")

        bw, bh, y0 = 1.72, 0.95, 4.05
        for i, (label, is_slot) in enumerate(stages):
            x = 0.15 + i * 2.08
            ax.add_patch(FancyBboxPatch(
                (x, y0), bw, bh, boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.5 if is_slot else 1.0,
                linestyle="--" if is_slot else "-",
                edgecolor=C["slot"] if is_slot else "#444444",
                facecolor="#EAF4FB" if is_slot else C["fixed"], zorder=2))
            ax.text(x + bw / 2, y0 + bh / 2, label, ha="center", va="center",
                    fontsize=7.2, zorder=3,
                    fontweight="bold" if is_slot else "normal")
            if i < len(stages) - 1:
                ax.add_patch(FancyArrowPatch(
                    (x + bw + 0.04, y0 + bh / 2), (x + 2.04, y0 + bh / 2),
                    arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
                    color="#444444", zorder=1))

        ax.text(0.15, y0 + bh + 0.20, "identical across all three benchmarks",
                fontsize=6.6, color="#444444")

        for i, (heading, rows) in fills.items():
            x = 0.15 + i * 2.08
            ax.plot([x + bw / 2] * 2, [y0 - 0.05, 3.55], color=C["slot"],
                    linewidth=0.9, linestyle=":", zorder=1)
            ax.text(x + bw / 2, 3.44, heading, ha="center", va="top",
                    fontsize=6.3, color=C["slot"], fontweight="bold")
            for j, (bench, filling) in enumerate(rows):
                ax.text(x + bw / 2, 3.06 - j * 0.42, bench, ha="center",
                        va="center", fontsize=6.1, color="#666666")
                ax.text(x + bw / 2, 2.88 - j * 0.42, filling, ha="center",
                        va="center", fontsize=6.5)

        ax.text(0.15, 1.52,
                "Stage 3 is worth filling only where the support-pattern "
                "ceiling binds. The solo ratio estimates", fontsize=6.6)
        ax.text(0.15, 1.28,
                "that from cached generations, before stage 3 is implemented.",
                fontsize=6.6)

        _save(fig, "fig1_template", {
            "raw_data": "none; schematic",
            "transformations": ["none"],
            "uncertainty": "not applicable",
            "missing_data": "not applicable"})


def _raw_list_lengths(qid):
    """True per-agent output lengths. weighting_v3.load_rich clamps these to a
    minimum of 1 so that log(L) stays finite in the omission feature, which
    would display an agent that returned nothing as though it emitted one
    item. The figure shows the truth and marks the clamp."""
    import collections
    from open_ended_aggregation.benchmarks import qampari as QA
    from open_ended_aggregation.paths import DATA
    out = collections.Counter()
    for line in open(DATA / "qampari_asc800.jsonl"):
        r = json.loads(line)
        if r["qid"] != qid:
            continue
        out[r["model"]] = sum(1 for it in r["items"] if QA.norm(it))
    return out


def fig_worked_example():
    d = json.loads((RESULTS / "worked_example_hit.json").read_text())
    rows, models = d["rows"], d["models"]
    L = _raw_list_lengths(d["qid"])
    cols = sorted(models, key=lambda m: (-L[m], m))
    nr = len(rows)

    XN, X0, XW = 0.5, 21.0, 5.4          # name col, first agent col, col width
    XS = X0 + len(cols) * XW + 1.2       # |S|
    XP = XS + 3.4                        # p bar
    PW = 11.0
    XC = XP + PW + 5.6                   # gold marker

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, ax = plt.subplots(figsize=(FULL, 3.5), layout="constrained")
        ax.set_xlim(-0.6, XC + 9.4)
        ax.set_ylim(-3.5, nr + 2.6)
        ax.axis("off")
        ax.invert_yaxis()

        ax.text(XN - 0.6, -3.15, f'"{d["question"]}"', fontsize=7.0,
                fontweight="bold")
        ax.text(XN - 0.6, -2.45,
                f'{d["n_candidates"]} pooled candidates from {len(models)} agents, '
                f'{d["gold_count"]} gold answers; held-out fold {d["fold"]}',
                fontsize=6.0, color=C["note"])

        # -- header
        ax.text(XN - 0.6, -1.05, "candidate $c$", fontsize=6.4,
                fontweight="bold")
        ax.annotate("", xy=(X0 - 0.4, -1.72), xytext=(X0 + len(cols) * XW, -1.72),
                    arrowprops=dict(arrowstyle="-", linewidth=0.7,
                                    color=C["method"]))
        ax.text(X0 + len(cols) * XW / 2, -1.95,
                "position $r_m(c)$ within that agent's own output",
                fontsize=6.0, ha="center", color=C["method"])
        for j, m in enumerate(cols):
            x = X0 + j * XW + XW / 2
            ax.text(x, -1.05, SHORT[m], fontsize=5.7, ha="center")
            ax.text(x, -0.42, f"$L_m$={L[m]}", fontsize=5.3, ha="center",
                    color=C["note"])
        ax.text(XS, -1.05, "$|S|$", fontsize=6.4, ha="center", fontweight="bold")
        ax.text(XP + PW / 2, -1.05, "score $p$", fontsize=6.4, ha="center",
                fontweight="bold")
        ax.text(XC, -1.05, "gold", fontsize=6.4, ha="center", fontweight="bold")

        flagged = {"barchana", "barachana"}
        for i, r in enumerate(rows):
            y = i + 0.95
            if r["k"] % 2 == 0:
                ax.add_patch(Rectangle((XN - 0.6, y - 0.42), XC + 2.6, 0.84,
                                       facecolor="#F6F6F6", edgecolor="none",
                                       zorder=0))
            nm = r["candidate"][:22]
            ax.text(XN - 0.6, y, nm, fontsize=6.2, va="center",
                    color=C["weight"] if nm in flagged else "black",
                    fontweight="bold" if nm in flagged else "normal")
            for j, m in enumerate(cols):
                x = X0 + j * XW + XW / 2
                if m in r["asserted"]:
                    rk = r["asserted"][m]
                    ax.add_patch(FancyBboxPatch(
                        (x - 1.7, y - 0.31), 3.4, 0.62,
                        boxstyle="round,pad=0.02,rounding_size=0.12",
                        facecolor="#DCEAF5", edgecolor="#9FC6DF",
                        linewidth=0.5, zorder=1))
                    ax.text(x, y, str(rk), fontsize=5.9, ha="center",
                            va="center", zorder=2)
                else:
                    ax.text(x, y, "·", fontsize=6.5, ha="center",
                            va="center", color="#C8C8C8")
            ax.text(XS, y, str(r["support"]), fontsize=6.2, ha="center",
                    va="center")
            ax.add_patch(Rectangle((XP, y - 0.26), PW * r["p"], 0.52,
                                   facecolor=C["method"], alpha=0.75,
                                   edgecolor="none", zorder=1))
            ax.text(XP + PW + 0.4, y, f'{r["p"]:.2f}', fontsize=5.8,
                    va="center")
            ax.plot([XC], [y], marker="o" if r["correct"] else "o",
                    markersize=4.0,
                    markerfacecolor=C["good"] if r["correct"] else "white",
                    markeredgecolor=C["good"] if r["correct"] else "#B0B0B0",
                    markeredgewidth=0.9, zorder=2)

        # -- the cut
        kc = d["k_chosen"]
        ax.plot([XN - 0.6, XC + 2.0], [kc + 0.45] * 2, color=C["weight"],
                linewidth=1.5, zorder=4)
        ax.text(XC + 2.6, kc + 0.45, f"$k^*(q)={kc}$", fontsize=6.6,
                color=C["weight"], ha="left", va="center", zorder=5)
        ax.annotate("", xy=(XC + 3.2, 0.7), xytext=(XC + 3.2, kc + 0.15),
                    arrowprops=dict(arrowstyle="->", linewidth=0.7,
                                    color=C["note"]))
        ax.text(XC + 3.9, kc / 2, "kept", fontsize=5.9, color=C["note"],
                ha="left", va="center")
        ax.annotate("", xy=(XC + 3.2, nr + 0.2), xytext=(XC + 3.2, kc + 0.75),
                    arrowprops=dict(arrowstyle="->", linewidth=0.7,
                                    color=C["note"]))
        ax.text(XC + 3.9, (kc + nr) / 2 + 0.5, "dropped", fontsize=5.9,
                color=C["note"], ha="left", va="center")

        _save(fig, "fig1_worked_example", {
            "raw_data": "results/worked_example_hit.json, from "
                        "analysis/budget_trace.py --qid "
                        "68__wikitables_composition__dev --json",
            "transformations": ["candidates ordered by the scorer's predicted "
                                "probability", "agent columns ordered by "
                                "list length", "candidate strings truncated "
                                "to 22 characters for display"],
            "uncertainty": "not applicable; one question shown in full",
            "missing_data": "a dot marks an agent that did not assert that "
                            "candidate; Cohere returned nothing at all"})


# --------------------------------------------------------------- figure 2 ---
def fig_ceiling():
    d = json.loads((RESULTS / "pattern_cells.json").read_text())
    cells = [c for c in d["cells"] if c["n"] >= 30]
    per_k = d["per_support_count"]
    ks = [r["k"] for r in per_k]
    nmax = max(c["n"] for c in cells)
    rng = np.random.default_rng(20260823)

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(FULL, 2.75), layout="constrained",
            gridspec_kw={"width_ratios": [1.5, 1.0]}, sharex=True)

        for c in cells:
            jit = (rng.random() - 0.5) * 0.30
            ax1.scatter(c["k"] + jit, c["rate"], s=4 + 90 * (c["n"] / nmax) ** 0.5,
                        color=C["ceiling"], alpha=0.45, linewidths=0, zorder=2)
        ax1.plot(ks, [r["rate"] for r in per_k], color=C["base"], linewidth=1.5,
                 marker="s", markersize=3.6, zorder=4,
                 label=f"what {BASE} sees:\n$P(\\mathrm{{correct}}\\,|\\,|S(c)|=k)$")
        ax1.scatter([], [], s=30, color=C["ceiling"], alpha=0.55, linewidths=0,
                    label="one support pattern $s$ with $\\geq$30 claims,\n"
                          "area $\\propto$ claims in that cell")
        ax1.set_xticks(ks)
        ax1.set_xlabel("support count $|S(c)|$")
        ax1.set_ylabel("claims correct (%)")
        ax1.set_title("Every rule in the class decides per cell;\n"
                      "counting can only decide per column", fontsize=8)
        ax1.legend(loc="upper left", fontsize=5.6, frameon=False,
                   labelspacing=0.55, borderaxespad=0.2)
        ax1.grid(color=C["grid"], linewidth=0.4, alpha=0.6)
        ax1.set_axisbelow(True)
        ax1.set_ylim(-4, 108)
        ax1.annotate("spread within a column\nis what weighting adds",
                     xy=(5.2, 42), xytext=(8.5, 92), fontsize=5.8,
                     color=C["note"], ha="right",
                     arrowprops=dict(arrowstyle="->", linewidth=0.6,
                                     color=C["note"]))
        ax1.text(8.5, 5.0,
                 f'{d["cells_occupied"]} of {d["cells_possible"]} patterns occupied\n'
                 f'median {d["median_claims_per_occupied_cell"]} claims per cell',
                 fontsize=5.4, color=C["note"], ha="right", va="center")

        ax2.bar(ks, [r["claims"] for r in per_k], color=C["method"],
                width=0.68, zorder=2)
        ax2.set_yscale("log")
        ax2.set_ylim(100, 4e5)
        ax2.set_xlabel("support count $|S(c)|$")
        ax2.set_ylabel("claims")
        ax2.set_title("but 87% of claims sit at $|S(c)|=1$,\n"
                      "where the spread is 5.5 points", fontsize=8)
        ax2.grid(axis="y", color=C["grid"], linewidth=0.4, alpha=0.6)
        ax2.set_axisbelow(True)
        for r in per_k:
            ax2.text(r["k"], r["claims"] * 1.35, f'{r["claims"]:,}',
                     fontsize=5.0, ha="center", rotation=90, va="bottom")

        _save(fig, "fig2_ceiling", {
            "raw_data": "results/pattern_cells.json, from "
                        "analysis/pattern_cells.py",
            "transformations": ["cells with fewer than 30 claims omitted from "
                                "the left panel, since their rate is not "
                                "estimable; all claims retained in the right "
                                "panel and in the black reference line",
                                "horizontal jitter for legibility only, "
                                "seed 20260823"],
            "uncertainty": "not estimated per cell; marker area encodes the "
                           "claim count each rate is computed from",
            "missing_data": "5 of 256 support patterns never occur"})


# --------------------------------------------------------------- figure 3 ---
def fig_budget():
    hit = json.loads((RESULTS / "worked_example_hit.json").read_text())
    miss = json.loads((RESULTS / "worked_example_miss.json").read_text())

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.65),
                                 layout="constrained", sharey=True)

        for ax, d, title in ((axes[0], hit, "the rule lands on the optimum"),
                             (axes[1], miss, "the rule cuts too early")):
            ks = [r["k"] for r in d["rows"]]
            plug = [100 * r["plugin"] for r in d["rows"]]
            true = [100 * r["true_f1"] for r in d["rows"]]
            ax.plot(ks, true, color=C["note"], linewidth=1.3, linestyle="--",
                    marker="o", markersize=3.0, zorder=2,
                    label="true F1 (needs the labels)")
            ax.plot(ks, plug, color=C["method"], linewidth=1.6, marker="D",
                    markersize=3.2, zorder=3,
                    label="plug-in expected F1 (what the rule maximises)")
            kc, ko = d["k_chosen"], d["k_oracle"]
            ax.axvline(kc, color=C["weight"], linewidth=1.2, zorder=1)
            ax.text(kc + 0.22, 103, f"$k^*$={kc}", fontsize=6.0,
                    color=C["weight"])
            if ko != kc:
                ax.axvline(ko, color=C["good"], linewidth=1.0,
                           linestyle=":", zorder=1)
                ax.text(ko + 0.22, 103, f"oracle={ko}", fontsize=6.0,
                        color=C["good"])
            ax.scatter([len(ks)], [100 * d["rows"][-1]["true_f1"]], s=26,
                       marker="v", color=C["base"], zorder=4)
            ax.text(len(ks) - 0.35, 100 * d["rows"][-1]["true_f1"] - 7,
                    f'keep all\n{d["f1_keep_all"]:.1f}', fontsize=5.6,
                    ha="right", color=C["base"])
            ax.set_xlabel("candidates kept, $k$")
            ax.set_title(f'{title}\n$G$={d["gold_count"]}, '
                         f'$\\hat{{G}}_q$={d["ghat"]:.2f}, '
                         f'chosen F1 {d["f1_chosen"]:.1f}', fontsize=7.4)
            ax.set_xticks(range(2, len(ks) + 1, 2))
            ax.set_ylim(-4, 116)
            ax.grid(color=C["grid"], linewidth=0.4, alpha=0.6)
            ax.set_axisbelow(True)

        axes[0].set_ylabel("F1 (%)")
        axes[0].legend(loc="lower left", fontsize=5.7, frameon=False,
                       borderaxespad=0.25)

        _save(fig, "fig3_budget", {
            "raw_data": "results/worked_example_hit.json (question 68) and "
                        "results/worked_example_miss.json (question 4), both "
                        "from analysis/budget_trace.py --json",
            "transformations": ["none; both curves are printed as computed",
                                "the true-F1 curve is never seen by the rule "
                                "and is drawn only to show where the argmax "
                                "lands"],
            "uncertainty": "not applicable; two individual held-out questions",
            "missing_data": "none"})


# --------------------------------------------------------------- figure 4 ---
def fig_mirror():
    qam = json.loads((RESULTS / "rank_crossover.json").read_text())
    asqa = json.loads((RESULTS / "asqa_ours_ensembles.json").read_text())

    panels = [
        ("QAMPARI, set F1, 210 ensembles", qam,
         [(WEIGHT, "weighted_vs_cnt", "weighted_sig", C["weight"], "o"),
          (CEILING, "pattern_vs_cnt", "pattern_sig", C["ceiling"], "^"),
          (METHOD, "rank+budget_vs_cnt", "rank+budget_sig", C["method"], "D")],
         True),
        ("ASQA, DR*, 42 ensembles", asqa,
         [(WEIGHT, "wgt_vs_cnt", "wgt_sig", C["weight"], "o"),
          ("output-structure scorer", "ours_vs_cnt", "ours_sig", C["method"], "D")],
         False),
    ]

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.95), layout="constrained")
        rng = np.random.default_rng(20260823)

        for ax, (title, rows, series, stale) in zip(axes, panels):
            sizes = sorted({r["k"] for r in rows})
            span = 0.46
            for si, (label, key, sigkey, colour, marker) in enumerate(series):
                off = -span / 2 + span * si / max(1, len(series) - 1)
                nbet = nwor = tot = 0
                for k in sizes:
                    grp = [r for r in rows if r["k"] == k]
                    vals = np.array([r[key] for r in grp])
                    # the stored flag is direction-agnostic; splitting it stops
                    # a significant LOSS being counted as a win
                    nbet += sum(1 for r in grp if r[sigkey] and r[key] > 0)
                    nwor += sum(1 for r in grp if r[sigkey] and r[key] < 0)
                    tot += len(vals)
                    jit = ((rng.random(len(vals)) - 0.5) * 0.16
                           if len(vals) > 1 else np.zeros(1))
                    ax.scatter(k + off + jit, vals, s=5.5, alpha=0.42,
                               color=colour, marker=marker, linewidths=0, zorder=2)
                    ax.plot([k + off - 0.11, k + off + 0.11],
                            [np.median(vals)] * 2, color=colour, linewidth=1.8,
                            solid_capstyle="butt", zorder=3)
                tag = f"{nbet}/{tot} above baseline"
                if nwor:
                    tag += f", {nwor} below"
                ax.scatter([], [], s=15, color=colour, marker=marker,
                           label=f"{label}  ({tag})")

            ax.axhline(0, color=C["base"], linewidth=0.9, linestyle="--", zorder=1)
            ax.set_xticks(sizes)
            ax.set_xlabel("agents in the ensemble")
            ax.set_title(title, fontsize=8)
            ax.legend(loc="upper left", fontsize=5.7, frameon=False,
                      handletextpad=0.3, borderaxespad=0.15, labelspacing=0.32)
            ax.grid(axis="y", color=C["grid"], linewidth=0.4, alpha=0.6)
            ax.set_axisbelow(True)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.34 * (hi - lo))
            ax.text(sizes[0] - 0.46, 0.04 * (hi - lo),
                    f"baseline: {BASE}", fontsize=5.9, color=C["base"], va="bottom")
            if stale:
                ax.text(0.985, 0.015, STALE, transform=ax.transAxes,
                        fontsize=5.4, color=C["note"], style="italic", ha="right")

        axes[0].set_ylabel(f"gain over {BASE}")
        axes[1].annotate("n = 1", xy=(6, 2.74), xytext=(5.62, 1.75),
                         fontsize=6.0, color=C["note"], ha="center",
                         arrowprops=dict(arrowstyle="->", linewidth=0.6,
                                         color=C["note"]))
        axes[1].text(0.985, 0.015, "per-ensemble ceiling not computed on ASQA",
                     transform=axes[1].transAxes, fontsize=5.4,
                     color=C["note"], style="italic", ha="right")

        _save(fig, "fig4_mirror", {
            "raw_data": "results/rank_crossover.json (PRE parsing fix); "
                        "results/asqa_ours_ensembles.json",
            "transformations": ["per-ensemble gain over multi-agent counting, "
                                "as stored",
                                "horizontal jitter for legibility only, "
                                "seed 20260823", "median bar per group"],
            "uncertainty": "every ensemble is plotted; legend counts are "
                           "per-ensemble paired bootstrap, 10k resamples, "
                           "split by direction",
            "missing_data": "support-pattern ceiling absent from the ASQA "
                            "per-ensemble sweep; ASQA size 6 has one ensemble"})


# ------------------------------------------------------------- appendix A1 ---
def fig_position():
    q = json.loads((RESULTS / "rank_effect.json").read_text())
    a = json.loads((RESULTS / "asqa_rank_effect.json").read_text())
    w = json.loads((RESULTS / "cwq_rank_effect.json").read_text())
    bands = ["1-5", "6-20", "21-100", "101+"]

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(FULL, 3.15), layout="constrained",
            gridspec_kw={"width_ratios": [1.0, 1.18]})

        for b in bands:
            dd = q[b]
            colour = C["weight"] if dd["last"] > dd["first"] else C["method"]
            ax1.plot([0, 1], [dd["first"], dd["last"]], color=colour,
                     linewidth=1.5, marker="o", markersize=3.6, zorder=2)
            ax1.annotate(f"{b} items  (n={dd['n']:,})", xy=(1, dd["last"]),
                         xytext=(1.07, dd["last"]), fontsize=6.1, va="center")
        ax1.set_xlim(-0.10, 2.30)
        ax1.set_ylim(-0.45, 6.3)
        ax1.set_xticks([0, 1], ["first quintile", "last quintile"])
        ax1.set_ylabel("claims correct (%)")
        ax1.grid(axis="y", color=C["grid"], linewidth=0.4, alpha=0.6)
        ax1.set_axisbelow(True)

        labels = [f"QAMPARI, {b} items" for b in bands] + \
                 [f"CWQ, {b} items" for b in bands] + \
                 [f"ASQA, {b} sentences" for b in ["1-2", "3-4", "5-7"]]
        rhos = [q[b]["rho"] for b in bands] + \
               [w[b]["rho"] for b in bands] + \
               [a[b]["rho"] for b in ["1-2", "3-4", "5-7"]]
        ns = [q[b]["n"] for b in bands] + [w[b]["n"] for b in bands] + \
             [a[b]["n"] for b in ["1-2", "3-4", "5-7"]]
        ypos = np.arange(len(labels))[::-1]
        cols = [C["method"]] * 4 + [C["good"]] * 4 + [C["ceiling"]] * 3

        ax2.axvspan(-0.05, 0.05, color=C["grid"], alpha=0.30, zorder=0)
        ax2.hlines(ypos, 0, rhos, color=cols, linewidth=1.5, zorder=2)
        ax2.scatter(rhos, ypos, s=20, color=cols, zorder=3, marker="D",
                    linewidths=0)
        ax2.axvline(0, color="#000000", linewidth=0.9, zorder=1)
        for y, n in zip(ypos, ns):
            ax2.text(0.187, y, f"n={n:,}", fontsize=5.8, va="center", ha="right")
        ax2.text(0.0, -0.92, r"$|\rho| < 0.05$", fontsize=6.0, ha="center",
                 color=C["note"])
        ax2.set_yticks(ypos, labels, fontsize=6.4)
        ax2.set_ylim(-1.35, len(labels) - 0.4)
        ax2.set_xlim(-0.135, 0.195)
        ax2.set_xticks([-0.10, -0.05, 0.0, 0.05])
        ax2.set_xlabel(r"Spearman $\rho$ (position, correct)")
        ax2.grid(axis="x", color=C["grid"], linewidth=0.4, alpha=0.6)
        ax2.set_axisbelow(True)

        _save(fig, "figA1_position", {
            "raw_data": "results/rank_effect.json (PRE parsing fix); "
                        "results/asqa_rank_effect.json; "
                        "results/cwq_rank_effect.json",
            "transformations": ["solo assertions only, removing the support "
                                "confound",
                                "stratified by the asserting agent's own "
                                "output length, removing the verbosity confound"],
            "uncertainty": "not estimated; n reported per band",
            "missing_data": "none"})


# ------------------------------------------------------------- appendix A2 ---
def fig_calibration():
    bins = [("0.00-0.02", 81541, 0.4, 0.4, 0.0),
            ("0.02-0.05", 12278, 3.2, 4.0, +0.8),
            ("0.05-0.10", 6925, 7.1, 7.5, +0.5),
            ("0.10-0.20", 3876, 13.8, 13.4, -0.5),
            ("0.20-0.35", 2401, 26.9, 26.1, -0.9),
            ("0.35-0.50", 1602, 42.1, 39.5, -2.6),
            ("0.50-0.70", 1323, 58.7, 55.1, -3.6),
            ("0.70-1.00", 190, 74.8, 68.9, -5.8)]
    n = np.array([b[1] for b in bins], float)
    gap = np.array([b[4] for b in bins])
    read = np.array([b[2] >= 35 for b in bins])
    y = np.arange(len(bins))[::-1]
    cols = [C["weight"] if r else C["method"] for r in read]

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(FULL, 2.45), layout="constrained", sharey=True,
            gridspec_kw={"width_ratios": [1.25, 1.0]})

        ax1.barh(y, gap, color=cols, height=0.66, zorder=2)
        ax1.axvline(0, color="#000000", linewidth=0.9, zorder=3)
        for yy, g in zip(y, gap):
            ax1.text(g + (0.18 if g >= 0 else -0.18), yy, f"{g:+.1f}",
                     fontsize=6.0, va="center",
                     ha="left" if g >= 0 else "right")
        ax1.set_yticks(y, [b[0] for b in bins], fontsize=6.4)
        ax1.set_ylabel("predicted probability bin")
        ax1.set_xlim(-7.4, 2.6)
        ax1.set_xlabel("observed minus predicted (percentage points)")
        ax1.set_title("Calibration error by bin", fontsize=8)
        ax1.grid(axis="x", color=C["grid"], linewidth=0.4, alpha=0.6)
        ax1.set_axisbelow(True)
        ax1.text(-7.15, 1.35, "bins the keep-count\nreads", fontsize=6.0,
                 color=C["weight"], va="center")

        ax2.barh(y, n, color=cols, height=0.66, zorder=2)
        ax2.set_xscale("log")
        ax2.set_xlim(100, 2.2e5)
        ax2.set_xlabel("claims in bin")
        ax2.set_title("Claims per bin", fontsize=8)
        ax2.grid(axis="x", color=C["grid"], linewidth=0.4, alpha=0.6, which="major")
        ax2.set_axisbelow(True)
        for yy, v in zip(y, n):
            ax2.text(v * 1.25, yy, f"{int(v):,}", fontsize=5.9, va="center")

        _save(fig, "figA2_calibration", {
            "raw_data": "measured inline, cross-fitted, 110,136 claims; "
                        "NO REPRODUCIBLE MODULE, see UNRESOLVED.md section 6",
            "transformations": ["binned by predicted probability; edges on the "
                                "y axis",
                                "gap taken as recorded, not recomputed from "
                                "the rounded columns"],
            "uncertainty": "not estimated; bin counts in the right panel",
            "missing_data": "none"})


# ------------------------------------------------------------- appendix A3 ---
def fig_learning_curve():
    files = [("test split A", "learning_curve.json"),
             ("test split B", "learning_curve_test1.json")]
    series, spread = [], {}
    for label, fname in files:
        p = RESULTS / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ns = sorted(int(k) for k in d)
        series.append((label, ns,
                       [d[str(k)]["rank + budget"][0] - d[str(k)]["count"][0]
                        for k in ns]))
        spread[label] = (ns, [d[str(k)]["rank + budget"][1] for k in ns])

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.45),
                                       layout="constrained")
        ticks = [25, 50, 100, 200, 350, 527]

        for (label, ns, gain), mk, col in zip(series, ["o", "s"],
                                              [C["method"], C["ceiling"]]):
            ax1.plot(ns, gain, marker=mk, markersize=3.6, linewidth=1.4,
                     color=col, label=label, zorder=3)
        ax1.axhline(0, color=C["base"], linewidth=0.9, linestyle="--", zorder=1)
        ax1.text(62, 0.08, f"baseline: {BASE}", fontsize=5.9, color=C["base"])
        ax1.set_xscale("log")
        ax1.set_xticks(ticks, [str(t) for t in ticks])
        ax1.minorticks_off()
        ax1.set_xlabel("labelled training questions")
        ax1.set_ylabel(f"gain over {BASE} (set F1)")
        ax1.set_title("Gain against training-set size", fontsize=8)
        ax1.legend(loc="lower right", fontsize=6.2, frameon=False)
        ax1.grid(color=C["grid"], linewidth=0.4, alpha=0.6)
        ax1.set_axisbelow(True)

        for (label, (ns, sv)), mk, col in zip(spread.items(), ["o", "s"],
                                              [C["method"], C["ceiling"]]):
            # the stored 0.0 at the largest size is "one draw", not "no spread"
            ns_m = [x for x in ns if x != max(ns)]
            ax2.plot(ns_m, sv[:len(ns_m)], marker=mk, markersize=3.6,
                     linewidth=1.4, color=col, label=label, zorder=3)
        ax2.set_xscale("log")
        ax2.set_xticks(ticks, [str(t) for t in ticks])
        ax2.minorticks_off()
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel("labelled training questions")
        ax2.set_ylabel("range across 5 training draws (F1)")
        ax2.set_title("Sensitivity to which questions were used", fontsize=8)
        ax2.legend(loc="upper right", fontsize=6.2, frameon=False)
        ax2.grid(color=C["grid"], linewidth=0.4, alpha=0.6)
        ax2.set_axisbelow(True)
        ax2.annotate("single draw at 527;\nrange undefined",
                     xy=(527, 0.02), xytext=(300, 0.62), fontsize=5.8,
                     color=C["note"], ha="center",
                     arrowprops=dict(arrowstyle="->", linewidth=0.6,
                                     color=C["note"]))

        fig.text(0.005, 0.008, STALE, fontsize=5.4, color=C["note"],
                 style="italic")

        _save(fig, "figA3_learning_curve", {
            "raw_data": "results/learning_curve.json, "
                        "results/learning_curve_test1.json (both PRE parsing fix)",
            "transformations": ["gain = (output-structure scorer + budget) "
                                "minus multi-agent counting, fixed held-out "
                                "test set",
                                "5 independent training draws per size"],
            "uncertainty": "right panel is the observed range across the 5 "
                           "draws, not a confidence interval",
            "missing_data": "N=527 exhausts the training pool, so only one "
                            "draw exists and the range is undefined"})


# --------------------------------------------------------------- pipeline ---
def fig_pipeline():
    """One question end to end: agent outputs, the two channels read from them,
    and the kept set. Data: results/pipeline_example.json.

    Compact layout: the question banner is one line, near-identical agents are
    folded into shared rows, and each channel box states its reading in three
    rows instead of five. Same content, ~25% less page.
    """
    d = json.loads((RESULTS / "pipeline_example.json").read_text())
    A = d["agents"]
    NAVY, GREEN, GFILL, BFILL = "#1F3864", "#548235", "#E9F2E2", "#DEEAF6"
    LH, GAP, FS = 0.23, 0.20, 6.2

    def chip(ax, x, y, text):
        ax.add_patch(FancyBboxPatch(
            (x, y - 0.13), 0.12 + 0.082 * len(text), 0.26,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0, facecolor=NAVY, zorder=3))
        ax.text(x + 0.06, y, text, fontsize=5.8, color="white",
                fontweight="bold", va="center", zorder=4)

    def box(ax, x, top, w, title, rows, lab, fill=GFILL, edge=GREEN):
        h = 0.52 + (len(rows) - 1) * LH + 0.13
        ax.add_patch(FancyBboxPatch(
            (x, top - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=2))
        chip(ax, x + 0.16, top - 0.25, title)
        for i, (k, v) in enumerate(rows):
            yy = top - 0.52 - i * LH
            if k:
                ax.text(x + 0.22, yy, k, fontsize=FS, va="center",
                        fontweight="bold", zorder=4)
            ax.text(x + 0.22 + lab, yy, v, fontsize=FS, va="center", zorder=4)
        return top - h

    def down(ax, x, y):
        ax.add_patch(FancyArrowPatch((x, y), (x, y - GAP + 0.03),
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.85, color=NAVY, zorder=1))

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, ax = plt.subplots(figsize=(FULL, 2.62), layout="constrained")
        ax.set_xlim(0, 10); ax.axis("off")
        top = 14.0

        # -- question banner: chip and question on one line, counts beneath --
        BH = 0.62
        ax.add_patch(FancyBboxPatch(
            (0.1, top - BH), 9.8, BH,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0, facecolor=NAVY, zorder=2))
        ax.add_patch(FancyBboxPatch(
            (0.26, top - 0.37), 0.78, 0.26,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0.9, edgecolor="white", facecolor=NAVY, zorder=3))
        ax.text(0.33, top - 0.24, "Question", fontsize=5.8, color="white",
                fontweight="bold", va="center", zorder=4)
        ax.text(1.18, top - 0.24, d["question"], fontsize=6.6, color="white",
                va="center", zorder=4)
        ax.text(0.31, top - 0.48,
                f"{d['n_gold']} gold answers  \u00b7  {len(A)} agents  "
                f"\u00b7  {d['n_candidates']} pooled candidates",
                fontsize=5.9, color="#C9D4E4", va="center", zorder=4)
        y = top - BH
        down(ax, 5.0, y)

        # -- agent outputs: the two agents the example turns on, then folds --
        def items(m, n):
            return " \u00b7 ".join(A[m]["items"][:n]) + (
                " \u00b7 \u2026" if A[m]["L"] > n else "")

        rows = [
            (SHORT["Kimi-K2.5"], f"L=3     {items('Kimi-K2.5', 3)}"),
            (SHORT["DeepSeek-V4-Flash"], f"L=31    {items('DeepSeek-V4-Flash', 3)}"),
            ("grok, gpt-5.4", f"L=3 each    {A['grok-4.3']['items'][0]} \u00b7 "
                              f"{A['gpt-5.4']['items'][0]} \u00b7 \u2026"),
            ("4 others", "L=1 each    one item each"),
        ]
        y = box(ax, 0.1, y - GAP, 9.8, "Agent outputs", rows, 1.50)

        down(ax, 2.55, y); down(ax, 7.45, y)
        ytop, sup = y - GAP, d["supported"]

        box(ax, 0.1, ytop, 4.75, "Support pattern S(c) \u2014 what counting reads",
            [("\u2265 2 asserters", ", ".join(s["name"] for s in sup)
                                    + "  \u2014  both wrong"),
             ("all 3 correct", "asserted by one agent, |S(c)| = 1"),
             ("", "also enters the scorer as features")], 1.25)

        y = box(ax, 5.15, ytop, 4.75,
                "Generation behaviour B(c) \u2014 what we add",
                [("all 3 correct", "positions 1\u20133 of Kimi's 3-item list"),
                 ("silent agents", "four at L=1; DeepSeek at L=31"),
                 ("", "position and graded omission")], 1.25)

        down(ax, 2.55, y); down(ax, 7.45, y)

        ht = 0.98
        ax.add_patch(FancyBboxPatch(
            (1.5, y - GAP - ht), 7.0, ht,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0, facecolor=BFILL, zorder=2))
        chip(ax, 1.66, y - GAP - 0.25, "Claim Scorer \u2192 Keep Budget")
        ax.text(1.72, y - GAP - 0.50, "92 features over S(c) and B(c)  "
                                      "$\\rightarrow$  probability $p_c$",
                fontsize=FS, va="center", zorder=4)
        ax.text(1.72, y - GAP - 0.82,
                r"$k^{*}(q)=\arg\max_{k}\ 2\sum_{i\leq k} p_i / (k+\hat{G}_q)$",
                fontsize=6.5, va="center", zorder=4)
        y = y - GAP - ht
        down(ax, 5.0, y)

        y = box(ax, 0.1, y - GAP, 9.8, "Kept set",
                [("kept", " \u00b7 ".join(c["name"] for c in d["correct"])
                          + f"   \u2014   all {d['n_correct']} correct "
                            "candidates in the pool")], 1.05)
        ax.set_ylim(y - 0.14, top + 0.18)

        _save(fig, "fig5_pipeline", {
            "source": "results/pipeline_example.json",
            "derived_from": f"QAMPARI question {d['qid']}, 8 agents, "
                            "from analysis/weighting_v3.load_rich",
            "transform": "one question end to end; support counts, within-output "
                         "positions and per-agent output lengths parsed after the "
                         "special-token fix of Section 4.4",
        })


def main():
    print("building figures ->", OUT)
    mpl.rcParams["pdf.fonttype"] = 42
    # fig_ceiling and fig_mirror are superseded by fig_ceiling_v2 and
    # fig_regimes; kept in the module for reference, not built.
    for fn in (fig_hierarchy, fig_worked_example, fig_ceiling_v2,
               fig_regimes, fig_position, fig_pipeline):
        fn()
    print("done")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------- figure 6 ---
def fig_contrast():
    """One compact panel: two claims with identical support, separated by
    behaviour. Data: results/hierarchy_pair.json.

    The support pattern is a column rather than a second panel, so the reader
    sees "identical here, different there" in one row-wise sweep.
    """
    d = json.loads((RESULTS / "hierarchy_pair.json").read_text())
    A, B = d["claims"]
    n = len(d["col_order"])
    NAVY, BLUE, GREEN, RED, GREY = ("#1F3864", "#4E86C6", "#2E7D32",
                                    "#B4472F", "#6E6E6E")
    FS = 6.4
    XC, XP, XS, XV = 2.05, 4.85, 6.85, 9.72      # cells, position, silence, verdict

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, ax = plt.subplots(figsize=(FULL, 1.28), layout="constrained")
        ax.set_xlim(0, 10); ax.set_ylim(0, 2.30); ax.axis("off")

        ax.add_patch(FancyBboxPatch(
            (0.08, 0.16), 9.84, 1.98,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.3, edgecolor=BLUE, facecolor="white", zorder=1))

        for x, t in ((XC, "asserted by"), (XP, "position in its list"),
                     (XS, "longest silent agent")):
            ax.text(x, 1.83, t, fontsize=FS - 1.0, color=GREY, va="center")

        for c, y, col, mark in ((A, 1.33, GREEN, "keep"), (B, 0.83, RED, "drop")):
            sil = max(v for k, v in c["lengths"].items() if k != c["asserter"])
            ax.text(0.32, y, c["label"], fontsize=FS, va="center",
                    fontweight="bold", zorder=4)
            ax.text(0.62, y, f'"{c["name"]}"', fontsize=FS, va="center",
                    style="italic", color=NAVY, zorder=4)
            for i in range(n):
                ax.add_patch(FancyBboxPatch(
                    (XC + i * 0.28, y - 0.10), 0.22, 0.20,
                    boxstyle="round,pad=0.005,rounding_size=0.02",
                    linewidth=0.6, edgecolor="#9AA5B1",
                    facecolor=NAVY if i == 0 else "white", zorder=3))
            ax.text(XP, y, f"item {c['pos'] + 1} of {c['L']}", fontsize=FS,
                    va="center", zorder=4)
            ax.text(XS, y, f"wrote {sil:,} items", fontsize=FS, va="center",
                    zorder=4)
            ax.text(XV, y, mark, fontsize=FS, color=col, ha="right",
                    va="center", fontweight="bold", zorder=4)

        ax.text(XC, 0.38, "identical", fontsize=FS - 1.0, color=GREY,
                va="center")
        ax.text(XP, 0.38, "what we read instead", fontsize=FS - 1.0,
                color=NAVY, va="center")

        _save(fig, "fig5_contrast", {
            "source": "results/hierarchy_pair.json",
            "derived_from": "two QAMPARI claims asserted by Kimi-K2.5 alone",
            "transform": "support pattern beside within-output position and "
                         "longest silent-agent length, same claim pair",
        })
