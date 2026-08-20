const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageOrientation, LevelFormat, convertInchesToTwip,
} = require("docx");
const fs = require("fs");

const W = 9360;                       // content width, US Letter, 1" margins
const GREY = "F2F2F2", HDR = "1F3864", OK = "1E6B3A", BAD = "9C2A2A";

const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, before: o.before ?? 0, line: 276 },
  alignment: o.align,
  indent: o.indent,
  children: [new TextRun({ text, bold: o.bold, italics: o.italics,
                           color: o.color, size: o.size ?? 21, font: o.font })],
});

const Bullet = (text, o = {}) => new Paragraph({
  numbering: { reference: "bul", level: 0 },
  spacing: { after: 80, line: 276 },
  children: [new TextRun({ text, size: 21, bold: o.bold, color: o.color })],
});

// rich paragraph: array of [text, opts]
const RP = (runs, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: 276 },
  numbering: o.bullet ? { reference: "bul", level: 0 } : undefined,
  children: runs.map(([t, x = {}]) => new TextRun({
    text: t, bold: x.b, italics: x.i, color: x.c, size: 21,
    font: x.mono ? "Consolas" : undefined,
  })),
});

const H = (text, lvl) => new Paragraph({
  heading: lvl, spacing: { before: 280, after: 140 },
  children: [new TextRun({ text, bold: true, color: HDR,
                           size: lvl === HeadingLevel.HEADING_1 ? 30 : 24 })],
});

function cell(text, w, o = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({
      alignment: o.align,
      spacing: { after: 0, line: 240 },
      children: [new TextRun({
        text, bold: o.bold, size: o.size ?? 19, color: o.color,
        font: o.mono ? "Consolas" : undefined,
      })],
    })],
  });
}

function table(widths, header, rows, opts = {}) {
  const hr = new TableRow({
    tableHeader: true,
    children: header.map((h, i) =>
      cell(h, widths[i], { bold: true, fill: HDR, color: "FFFFFF",
                           align: i ? AlignmentType.CENTER : undefined })),
  });
  const body = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => {
      const spec = typeof c === "object" ? c : { t: c };
      return cell(spec.t, widths[i], {
        bold: spec.b, color: spec.c, mono: i > 0 && !opts.noMono,
        fill: spec.fill ?? (ri % 2 ? GREY : undefined),
        align: i ? AlignmentType.CENTER : undefined,
      });
    }),
  }));
  return new Table({
    columnWidths: widths,
    width: { size: W, type: WidthType.DXA },
    rows: [hr, ...body],
  });
}

const rule = () => new Paragraph({
  spacing: { before: 60, after: 160 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "BFBFBF" } },
  children: [],
});

// ─────────────────────────────────────────────────────────── content
const doc = new Document({
  numbering: {
    config: [{
      reference: "bul",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 220 } } },
      }],
    }],
  },
  styles: { default: { document: { run: { font: "Calibri", size: 21 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children: [
      // ── title
      new Paragraph({
        spacing: { after: 60 },
        children: [new TextRun({ text: "Open-Ended LLM Answer Aggregation",
                                 bold: true, size: 40, color: HDR })],
      }),
      P("Results and progress report — 17 August 2026", { italics: true, size: 22, after: 40 }),
      P("Extending Ai et al., Beyond Majority Voting: LLM Aggregation by Leveraging "
        + "Higher-Order Information (ICML 2026, arXiv:2510.01499) from multiple choice "
        + "to open-ended answers.", { size: 20, after: 200 }),
      rule(),

      // ── executive summary
      H("Summary", HeadingLevel.HEADING_1),
      RP([["Merging beats selection. ", { b: true }],
          ["On QAMPARI, assembling an answer from atomic claims across five models beats the "
           + "best single model by "],
          ["+2.00 F1 [+1.11, +2.96]", { b: true }],
          [", CI excluding zero. This has held at every sample size tested, from n=198 to n=535."]],
        { bullet: true }),
      RP([["The reliability weighting contributes nothing. ", { b: true, c: BAD }],
          ["Weighted merging beats the unweighted count filter by "],
          ["+0.08 [−0.55, +0.76]", { b: true }],
          [" — a tight null, not an underpowered one. The margin has collapsed "
           + "monotonically as n grew: +1.12 → +0.45 → +0.37 → +0.08."]],
        { bullet: true }),
      RP([["Majority voting actively hurts. ", { b: true }],
          ["Strict majority scores 3.38 points "],
          ["below", { i: true }],
          [" the best single model. The standard rule does not transfer to set-valued answers."]],
        { bullet: true }),
      RP([["The published OW rule is structurally inert here. ", { b: true }],
          ["With global weights and open-ended answers no two responses coincide, so its "
           + "weighted vote always returns the highest-weighted model. It reproduces "
           + "best-single exactly (+0.00, zero-width CI)."]],
        { bullet: true }),
      RP([["No second benchmark can replicate this. ", { b: true, c: BAD }],
          ["ASQA's metric is recall-only and FACTS's is precision-only; neither can score the "
           + "precision/recall tradeoff a merging filter operates on. This is a property of "
           + "the metrics, not of the method."]],
        { bullet: true }),

      // ── headline table
      H("1. Method comparison", HeadingLevel.HEADING_1),
      P("QAMPARI, set-F1, 5 Azure models, judge-free grading, n=535 complete questions. "
        + "All arms operate on the same parsed item sets; only the keep-rule differs. "
        + "Paired bootstrap over questions, 10,000 resamples.", { size: 20 }),
      table([2740, 900, 1200, 1760, 2760],
        ["method", "F1", "Δ vs best", "95% CI", "note"],
        [
          ["mean single model", "19.06", { t: "−7.12", c: BAD }, "[−8.39, −5.87] *", "no aggregation"],
          [{ t: "best single (grok-4.3)", b: true }, { t: "26.18", b: true }, "—", "reference", "no aggregation"],
          ["OW — response selection", "26.18", "+0.00", "[+0.00, +0.00]", "degenerates to best model"],
          ["OW — log-odds item filter", "21.26", { t: "−4.93", c: BAD }, "[−6.59, −3.30] *", "optimum = filter off"],
          ["union / no filter", "21.26", { t: "−4.93", c: BAD }, "[−6.59, −3.30] *", "θ = 1"],
          ["MV — strict majority", "22.80", { t: "−3.38", c: BAD }, "[−4.84, −1.96] *", "θ = 3"],
          ["ASC — count filter", "28.10", { t: "+1.92", c: OK }, "[+0.69, +3.20] *", "θ = 2, tuned"],
          [{ t: "ASC + OW (ours)", b: true }, { t: "28.19", b: true }, { t: "+2.00", b: true, c: OK }, "[+1.11, +2.96] *", "θ = 0.285"],
          ["oracle selection", "35.51", "+9.33", "[+7.89, +10.83] *", "needs labels; selection ceiling"],
        ]),
      P("* = 95% CI excludes zero.", { size: 18, italics: true, after: 60 }),
      RP([["Head-to-head: "],
          ["ASC+OW − ASC = +0.08 [−0.55, +0.76]", { b: true, mono: true }],
          ["   ·   ASC+OW − MV = +5.38 * ·   ASC − MV = +5.30 *", { mono: true }]],
        { after: 60 }),

      // ── trajectory
      H("2. How the result moved as n grew", HeadingLevel.HEADING_1),
      P("Generation is still running toward n=931. Snapshots taken along the way. This is the "
        + "single most informative table in the report: the headline is stable across a 2.7× "
        + "growth in sample, while the weighting effect decays to zero and the unweighted count "
        + "filter crosses into significance.", { size: 20 }),
      table([760, 1500, 1240, 1500, 2180, 2180],
        ["n", "best single", "ASC", "ASC+OW", "ASC+OW − best", "ASC+OW − ASC"],
        [
          ["198", "28.60", "29.54", "30.66", { t: "+2.06 *", c: OK }, "+1.12 [−0.05, +2.46]"],
          ["247", "28.67", "30.10", "30.55", { t: "+1.88 *", c: OK }, "+0.45 [−0.56, +1.59]"],
          ["275", "27.74", "29.31", "29.68", { t: "+1.94 *", c: OK }, "+0.37 [−0.59, +1.44]"],
          [{ t: "535", b: true }, "26.18", "28.10", "28.19", { t: "+2.00 *", b: true, c: OK },
           { t: "+0.08 [−0.55, +0.76]", b: true, c: BAD }],
        ]),
      P("ASC alone against the best single model over the same span: +0.94 → +1.42 → "
        + "+1.57 → +1.92 *. It has now cleared on its own.", { size: 20 }),
      RP([["Interpretation. ", { b: true }],
          ["The contribution has relocated from the weighting to the merging. What survives is "
           + "that merging beats the best single model; what does not survive is that our "
           + "reliability weights add anything over simply counting how many models assert a claim. "
           + "The null is now tight enough to state affirmatively rather than as a failure to detect."]]),

      // ── what the method is
      H("3. What the weighted rule actually does", HeadingLevel.HEADING_1),
      P("At the optimal threshold the weighted filter is not performing continuous weighting. "
        + "It reduces to a two-term discrete rule:", { size: 20 }),
      new Paragraph({
        spacing: { before: 60, after: 120 },
        indent: { left: 360 },
        children: [new TextRun({
          text: "keep an item if ≥ 2 models assert it, OR if grok-4.3 asserts it",
          font: "Consolas", size: 20, bold: true })],
      }),
      P("That rule scores 30.51 at n=198 — identical to the tuned weighted filter. "
        + "Substituting any other model into the trusted slot loses ground against plain "
        + "counting: Cohere −3.59, Kimi −2.77, DeepSeek −2.35, MAI −2.11. "
        + "The shuffled-weights control (27.43) is exactly the “trust the wrong model” value.",
        { size: 20 }),
      RP([["So the mechanism is: admit the best model's solo claims on top of consensus. "
           + "The risk is misidentifying which model that is. This is a two-parameter heuristic, "
           + "not reliability-weighted aggregation — and it is the most likely thing a "
           + "reviewer will name."]]),

      // ── verification
      H("4. Verification", HeadingLevel.HEADING_1),
      P("The n=198 result was re-derived from the raw generation file without reading the "
        + "original analysis code, per the prior handoff's own instruction.", { size: 20 }),
      Bullet("Grading contract: re-grading each model's own list reproduces the stored "
             + "generation-time F1 to |Δ| = 0.0000 for all five models."),
      Bullet("Every reported figure reproduced: best single 28.60, count 29.54, "
             + "weighted 30.66, uniform 29.54, shuffled 27.43."),
      Bullet("Uniform weights reproduce the count filter exactly (Δ = 0.00, zero-width CI), "
             + "so the gain never came from having a continuous threshold."),
      Bullet("Shuffling weights onto the wrong models costs 3.23 points, so model identity "
             + "carried real signal at n=198 — though at n=535 that signal no longer "
             + "translates into a margin over counting."),
      Bullet("Cross-fitting the threshold (choose on 80%, report on held-out 20%, rotate) "
             + "holds the headline: 30.66, +2.06, CI still excluding zero. All five folds "
             + "independently selected the same threshold."),

      // ── corrections
      H("5. Corrections to the previous handoff", HeadingLevel.HEADING_1),
      table([2500, 6860],
        ["claim", "finding"],
        [
          ["“Bug A cost ~1 F1 point”",
           "Never fired. Zero within-model duplicates in 22,131 items; dedup on/off is identical at every threshold. The parser already deduplicated by the same normaliser. Threshold-sweep resolution alone explains the recovery."],
          ["“Uniform must equal count(2), else the dedup fix is lost”",
           "Vacuous. With zero duplicates the identity holds whether or not the fix is present, so the check cannot detect what it was written to detect."],
          ["“The ASQA script carries Bug A”",
           "It does not — it already deduplicates one vote per model per cluster."],
          ["“Normalisation may merge distinct entities and overstate precision”",
           "Cannot happen. Zero distinct gold answers collide within a question, and grading normalises both sides before matching, so a collision always yields the same verdict for both surface forms."],
          ["§4.4 cross-fitted threshold — open",
           "Done. Result holds at 30.66, +2.06, CI excluding zero."],
        ], { noMono: true }),

      // ── replication
      H("6. Why ASQA and FACTS cannot replicate this", HeadingLevel.HEADING_1),
      P("A merging filter trades precision against recall. A benchmark can only adjudicate it "
        + "if its metric scores both. Of the three benchmarks run, only QAMPARI does.", { size: 20 }),
      table([1500, 2200, 2200, 3460],
        ["benchmark", "metric", "scores", "consequence for a filter"],
        [
          ["QAMPARI", "set-F1", "precision + recall", "Tradeoff exists; the method is testable."],
          ["ASQA", "STR-EM", "recall only", "Optimum is θ=1, keep everything (53.70%), decaying monotonically 35.85 / 28.36 / 21.82 / 15.80. Every arm collapses to union."],
          ["FACTS", "groundedness", "precision only", "No coverage term exists in the data; biased the opposite way."],
        ], { noMono: true }),
      RP([["This reframes the replication blocker. ", { b: true }],
          ["It is not that the effect failed to replicate — it is that no second benchmark "
           + "currently available can confirm or refute it. What is needed is a set-valued "
           + "benchmark with an F1-style metric, not further work on the existing scripts."]]),

      // ── standing
      H("7. What is and is not established", HeadingLevel.HEADING_1),
      P("Established", { bold: true, color: OK, after: 60 }),
      Bullet("Merging atomic claims across models beats the best single model on QAMPARI, "
             + "+2.00 [+1.11, +2.96], stable across four sample sizes."),
      Bullet("Majority voting and the published OW selection rule both fail to beat the best "
             + "single model on open-ended set-valued answers — OW provably so, since it "
             + "degenerates to picking one model."),
      Bullet("The result sits formally outside the co-failure ceiling of arXiv 2606.27288, "
             + "which bounds any policy whose output is almost surely one of the members' answers. "
             + "Merging assembles a new answer from fragments."),
      Bullet("QAMPARI is programmatically graded, so the result sits inside that paper's own "
             + "evaluation regime rather than outside it."),
      P("Not established", { bold: true, color: BAD, before: 120, after: 60 }),
      Bullet("That reliability weighting beats counting. Currently +0.08 [−0.55, +0.76]; "
             + "the evidence now points at no effect."),
      Bullet("That any of this works without labels. Every weight is supervised cross-fitted "
             + "per-model precision. The label-free configuration is untested."),
      Bullet("Replication. One benchmark, and no second benchmark capable of adjudicating."),

      // ── questions
      H("8. Open questions for discussion", HeadingLevel.HEADING_1),
      RP([["1. ", { b: true }],
          ["If the weighting adds nothing and the unweighted count filter now clears on its own, "
           + "where does the contribution sit — and is “merging beats selection, "
           + "weighting is a flat ablation” a paper?"]], { after: 80 }),
      RP([["2. ", { b: true }],
          ["Is the collapse to “consensus OR trust the best model” a publishable "
           + "negative result about scalar reliability weights, or evidence the method was never "
           + "doing what we claimed?"]], { after: 80 }),
      RP([["3. ", { b: true }],
          ["Is there a second set-valued benchmark with an F1-style metric? Adding a precision "
           + "term to ASQA would work mechanically but may read as metric-shopping."]], { after: 80 }),
      RP([["4. ", { b: true }],
          ["Does “formally outside the theorem” land as a genuine gap, or as exploiting "
           + "a definition?"]], { after: 80 }),
      RP([["5. ", { b: true }],
          ["Since the rule reduces to identifying the best model, and the EM estimator picks the "
           + "correct model with zero labels (Spearman +0.975), is label-free "],
          ["identification", { i: true }],
          [" sufficient, or must the weights themselves be label-free?"]], { after: 80 }),
      RP([["6. ", { b: true }],
          ["Venue and scope, given one benchmark and a null on the novel component."]], { after: 80 }),

      // ── appendix
      H("Appendix — reproduction", HeadingLevel.HEADING_1),
      P("./venv/bin/python verify_qampari_independent.py     # independent re-derivation, n=198",
        { font: "Consolas", size: 18, after: 40 }),
      P("./venv/bin/python compare_methods.py                # method comparison, current n",
        { font: "Consolas", size: 18, after: 40 }),
      P("./venv/bin/python compare_methods.py data/qampari_gen.jsonl.n198.bak   # frozen state",
        { font: "Consolas", size: 18, after: 40 }),
      P("./venv/bin/python analyze_merge_asqa_fixed.py       # ASQA, exact threshold enumeration",
        { font: "Consolas", size: 18, after: 140 }),
      P("Environment. Five Azure models: grok-4.3, Kimi-K2.5, Cohere-command-a-plus-05-2026, "
        + "MAI-Thinking-1, DeepSeek-V4-Flash. Generation toward n=931 was in flight at time of "
        + "writing (throttled at roughly 3.7 rows/min); the tables above are a snapshot and "
        + "compare_methods.py gives a current read. Requires the public QAMPARI release for gold "
        + "answers.", { size: 19 }),
      P("Caveat carried forward. Five analysis bugs have been found in this project to date, "
        + "three inflating the method and two suppressing it. The figures above are trusted "
        + "because the headline was re-derived independently from raw data, the grading contract "
        + "matches to 0.0000, and the uniform and shuffled controls behave as predicted — "
        + "not because the code looks correct.", { size: 19, italics: true }),
    ],
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("/Users/kevinchou/Documents/open-ended-aggregation/PROGRESS_2026-08-17.docx", b);
  console.log("wrote PROGRESS_2026-08-17.docx");
});
