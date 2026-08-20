const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  LevelFormat, PageOrientation, PageBreak,
} = require("docx");
const fs = require("fs");

const W = 9360;
const NAVY = "1F3864", GREY = "595959", RED = "9C0006", GREEN = "1E6B34", AMB = "8A6D1F";

const P = (t, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: 276 }, alignment: o.align,
  children: [new TextRun({ text: t, bold: o.bold, italics: o.italics, color: o.color,
    size: o.size ?? 21, font: "Calibri" })],
});
const PR = (runs, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: 276 },
  children: runs.map(([t, f = {}]) => new TextRun({ text: t, bold: f.bold,
    italics: f.italics, color: f.color, font: f.mono ? "Consolas" : "Calibri",
    size: f.size ?? 21 })),
});
const H = (t, lv) => new Paragraph({
  heading: lv, spacing: { before: lv === HeadingLevel.HEADING_1 ? 320 : 240, after: 140 },
  children: [new TextRun({ text: t, bold: true, font: "Calibri",
    size: lv === HeadingLevel.HEADING_1 ? 30 : 24,
    color: lv === HeadingLevel.HEADING_1 ? NAVY : "2E5496" })],
});
const BUL = (t, o = {}) => new Paragraph({
  numbering: { reference: "b", level: 0 }, spacing: { after: 90, line: 276 },
  children: [new TextRun({ text: t, font: "Calibri", size: 21, bold: o.bold, color: o.color })],
});
const MONO = (t) => new Paragraph({
  spacing: { after: 100, line: 240 },
  shading: { type: ShadingType.CLEAR, fill: "F4F4F4" },
  children: [new TextRun({ text: t, font: "Consolas", size: 17 })],
});
const CALL = (title, body, color) => new Table({
  width: { size: W, type: WidthType.DXA }, columnWidths: [W],
  borders: {
    top: { style: BorderStyle.SINGLE, size: 2, color }, bottom: { style: BorderStyle.SINGLE, size: 2, color },
    left: { style: BorderStyle.SINGLE, size: 18, color }, right: { style: BorderStyle.SINGLE, size: 2, color },
  },
  rows: [new TableRow({ children: [new TableCell({
    width: { size: W, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: "F7F7F7" },
    margins: { top: 140, bottom: 140, left: 200, right: 200 },
    children: [P(title, { bold: true, color, after: 80 }), ...body.map(b => P(b, { after: 60 }))],
  })] })],
});
function table(headers, rows, widths) {
  const hdr = new TableRow({ tableHeader: true, children: headers.map((h, i) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: NAVY },
    margins: { top: 90, bottom: 90, left: 120, right: 120 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
      children: [new TextRun({ text: h, bold: true, color: "FFFFFF", size: 19, font: "Calibri" })] })],
  })) });
  const body = rows.map((r, ri) => new TableRow({ children: r.map((c, i) => {
    const isStr = typeof c === "string", txt = isStr ? c : c.t;
    return new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: (!isStr && c.hi) ? "FDF2E9" : (ri % 2 ? "F2F2F2" : "FFFFFF") },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
        children: [new TextRun({ text: txt, size: 19, font: "Calibri",
          bold: !isStr && c.bold, color: !isStr ? c.color : undefined })] })],
    });
  }) }));
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows: [hdr, ...body] });
}

const doc = new Document({
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 220 } } } }] }] },
  styles: { default: { document: { run: { font: "Calibri", size: 21 } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
      margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: [
      new Paragraph({ spacing: { after: 60 }, children: [new TextRun({
        text: "Fact-Level Weighted Aggregation", bold: true, size: 40, color: NAVY, font: "Calibri" })] }),
      new Paragraph({ spacing: { after: 40 }, children: [new TextRun({
        text: "ASC + optimal weights — methodology and first results", size: 26, color: GREY, font: "Calibri" })] }),
      new Paragraph({ spacing: { after: 280 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 8 } },
        children: [new TextRun({ text: "16 August 2026   ·   final   ·   ASQA n=280, QAMPARI n=198   ·   1,392 + 1,391 generations, Azure only",
          size: 19, color: GREY, font: "Calibri" })] }),

      CALL("Bottom line", [
        "Moving the weights from whole responses down to individual facts does not rescue them. On both benchmarks the weighted filter is at or below a plain count filter: ASQA −0.90 (CI excludes zero), QAMPARI −0.20 (spans zero).",
        "Merging itself is mildly promising — both filters edge past the best single model on QAMPARI (+0.94 count, +0.74 weighted) — but neither margin clears zero at this sample size.",
        "One ASQA result looked like a significant win and was a metric artifact. It is written up in full in §4 because it is the most reusable thing here.",
      ], RED),

      new Paragraph({ spacing: { after: 200 }, children: [] }),

      H("1.  Why we changed the unit", HeadingLevel.HEADING_1),
      P("Our method (KWA) weights each model by an estimated reliability β and selects one existing response:"),
      MONO("    ŝ = argmax_s  Σ_j  β_j · sim(a_j , s)"),
      P("Across four benchmark configurations this never beat simply picking the strongest model in advance. The diagnosis was granularity. 71% of losses were questions where only one to three of seven models held the truth — and at response level, keeping that minority's contribution means accepting their entire answer, which no weight was ever confident enough to justify."),
      P("Atomic Self-Consistency (Thirukovalluru et al., 2024) changes the unit. It splits each response into atomic facts, clusters equivalent facts, keeps clusters that enough samples support, and reassembles. Its filter uses a raw count:"),
      MONO("    ASC:       keep fact if  count(f) = Σ_j 1[model j asserted f]  ≥ Θ"),
      P("Our contribution is a one-line substitution — the same β, applied per fact instead of per response:"),
      MONO("    ours:      keep fact if  support(f) = Σ_j β_j · 1[model j asserted f]  ≥ Θ"),
      PR([["Nothing else changes. ", {}],
          ["The intuition: one reliable model asserting a fact should outweigh two unreliable ones — and unlike response-level weighting, you can take the good parts of a bad response.", { bold: true }]]),
      P("ASC is also cross-model here where the original is same-model (m stochastic samples from one LLM). That combination — different models, fact-level, reliability-weighted — is unoccupied in the literature."),

      H("2.  Setup", HeadingLevel.HEADING_1),
      P("Five models, five labs, all deployed on Azure. OpenAI models run on shared instant-inference quota that vanished mid-run once, so they are used only as a judge where a judge is needed, never in the pool."),
      table(["Model", "Lab"], [
        ["grok-4.3", "xAI"], ["Kimi-K2.5", "Moonshot"],
        ["Cohere-command-a-plus-05-2026", "Cohere"], ["MAI-Thinking-1", "Microsoft"],
        ["DeepSeek-V4-Flash", "DeepSeek"]], [5600, 3760]),
      new Paragraph({ spacing: { after: 140 }, children: [] }),
      P("Temperature 0, one sample per model per question. Empty responses are never written, so an unanswered cell stays MISSING rather than silently grading as wrong — that specific bug produced a fake 67-point capability spread in an earlier run."),

      H("Benchmarks, and why these two", HeadingLevel.HEADING_2),
      P("Three earlier attempts failed on the grader rather than the method: answers of median four words left the similarity kernel nothing to do; reference-matching a 28-word summary measured agreement with one arbitrary selection of facts; and grading a 214-word answer as a single correct/incorrect bit destroyed the signal. Both benchmarks below were chosen to be graded automatically, with no LLM judge anywhere in the pipeline."),
      table(["", "ASQA", "QAMPARI"], [
        ["task", "ambiguous questions, ~3.4 valid interpretations each", "set-valued answers, median 8 entities"],
        ["output", "prose, ~80 words", "a list of items"],
        ["metric", "STR-EM — fraction of interpretations covered", "set precision / recall / F1 with gold aliases"],
        ["judge", "none", "none"],
        ["summariser", "none — see §4", "none — ASC list mode"]], [1700, 3900, 3760]),

      new Paragraph({ children: [new PageBreak()] }),

      H("3.  Result — QAMPARI (primary)", HeadingLevel.HEADING_1),
      P("n = 198 complete questions. Θ swept identically for both filters; the full sweep is printed in the logs rather than hidden."),
      table(["method", "P", "R", "F1", "vs best single"], [
        [{ t: "ASC count filter (θ=2)", bold: true }, "33.2%", "34.4%", { t: "29.5%", bold: true }, "+0.94"],
        ["WEIGHTED filter (θ=0.44)", "33.2%", "33.6%", "29.3%", "+0.74"],
        ["best single model (grok-4.3)", "—", "—", "28.6%", "—"],
        ["union, no filter", "17.5%", "48.8%", "22.8%", "−5.82"]], [3400, 1300, 1300, 1300, 2060]),
      new Paragraph({ spacing: { after: 140 }, children: [] }),
      MONO("    WEIGHTED − ASC count      −0.20   [−0.93, +0.41]     not significant"),
      MONO("    ASC count − best single   +0.94   [−1.28, +3.13]     not significant"),
      CALL("The interim number flipped, and that is the story", [
        "At n=141 the weighted filter led counting by +0.52. At n=198 it trails by −0.20. A 40% increase in sample reversed the sign.",
        "This is the third time in this project a ranking has flipped with more data or a different implementation — medoid reversed between n=36 and n=259 on another benchmark, and the label-free method reversed between two good-faith implementations of the same procedure.",
        "Treat any single ordering at this sample size as unresolved. The effects being chased are ~0.5 points; the instability is ~0.7.",
      ], AMB),
      BUL("Merging does edge past the best single model — but +0.94 with an interval of ±2 is not a result."),
      BUL("The union scoring WORST (−5.82) confirms F1 penalises padding correctly here. That is precisely what ASQA's metric failed to do; see §4."),
      BUL("Recall headroom is real: the best model reaches 30.0% recall, the union 48.8%. No single response contains the answer set — median 8 gold entities against 6–56 items listed per model."),

      H("4.  Result — ASQA, and the artifact", HeadingLevel.HEADING_1),
      P("n = 280 complete questions. The headline looked excellent and is not real."),
      table(["method", "STR-EM", "vs best single"], [
        ["best single model (Kimi-K2.5)", "46.57%", "—"],
        ["WEIGHTED filter (θ=0.30)", "52.79%", "+6.22 *"],
        ["ASC count filter (θ=1)", "53.70%", "+7.12 *"],
        [{ t: "naive concatenation — paste all 5 responses, no method", hi: true, bold: true },
         { t: "58.48%", hi: true, bold: true }, { t: "+11.91", hi: true, bold: true, color: RED }]], [5200, 2080, 2080]),
      new Paragraph({ spacing: { after: 140 }, children: [] }),
      CALL("The artifact, stated plainly", [
        "STR-EM asks only whether each interpretation's short answer appears somewhere in the text. It has no precision penalty. Concatenating all five responses produces 4.9× more text (393 words versus 81) and scores 58.48% — beating every method, with no method involved.",
        "Measured against that baseline, both filters LOSE: ASC by 4.78 points, weighted by 5.69. The apparent \"+7.12, significant\" was measuring text volume.",
        "The tell was visible and we nearly missed it: ASC's best threshold was Θ=1, which means \"filter nothing\". When the optimal setting of a filter is to not filter, the metric is rewarding something other than the method.",
      ], RED),
      P("The literature pairs STR-EM with ROUGE precisely to stop this; ROUGE penalises padding. Running STR-EM alone made a degenerate strategy optimal. QAMPARI does not have this problem because F1 penalises precision loss directly — which is why the union scores worst there (24.5%) and best there (47.8%) on recall alone."),
      PR([["This is the third time in this project that ", {}],
          ["the grader, not the method, was the finding", { bold: true }],
          [". It is the most transferable result we have: coarse or one-sided metrics on long-form output produce effects larger than anything the methods themselves move.", {}]]),

      H("5.  What is firm", HeadingLevel.HEADING_1),
      BUL("Weighting is at or below plain counting at fact level on BOTH benchmarks: ASQA −0.90 (CI [−1.60, −0.31], significant), QAMPARI −0.20 (CI [−0.93, +0.41]). Both filters see identical inputs, so the ASQA length artifact does not favour either — those comparisons are clean.", { bold: true }),
      BUL("That is five independent tests of the weighting idea across three granularities — response, sentence, and list item — and none supports it."),
      BUL("The label-free estimator has now degenerated on three datasets: it collapses to putting all weight on one model and outputting that model's answer verbatim — 259/259 on one benchmark. Every result above therefore uses SUPERVISED weights, which is not the deployable configuration."),
      BUL("Merging beats selection on QAMPARI (union recall 47.8% vs 31.4% best model) but barely on ASQA, where a per-question oracle selection already captures 9.65 of the 11.91 available points and merging adds only 2.26."),

      H("6.  Open, and what we would run next", HeadingLevel.HEADING_1),
      BUL("Finish QAMPARI to n=300 and rerun. Current interval is ±1.9 points against an effect of +0.5."),
      BUL("Add ROUGE alongside STR-EM on ASQA so padding is penalised, then rerun both filters."),
      BUL("A weight form appropriate to set tasks. The OW log-odds form logit(p) assumes a 50% prior and goes negative when precision is below 50% — on QAMPARI every model's precision is 8–35%, so the first run returned the empty set at every threshold. We now use precision directly as a non-negative weight; a principled form calibrated to the actual base rate is unfinished work."),
      BUL("Cross-fit Θ on a held-out split rather than sweeping on the evaluation set."),

      H("Reproducing", HeadingLevel.HEADING_2),
      table(["file", "what it does"], [
        ["run_asqa.py", "generation + STR-EM (unit-tested on hand-computed cases)"],
        ["run_qampari.py", "generation + set P/R/F1 (6 unit tests: bullets, aliases, preambles, articles)"],
        ["analyze_merge.py", "ASC vs weighted filter on QAMPARI, with threshold sweeps"],
        ["analyze_merge_asqa.py", "the same on ASQA, sentence-level clustering"],
        ["verify_aggregation.py", "clean-room re-derivation of the selection methods, with unit tests"]],
        [3400, 5960]),
    ],
  }],
});

Packer.toBuffer(doc).then(b => { fs.writeFileSync("ASC_WEIGHTED_REPORT.docx", b);
  console.log("wrote ASC_WEIGHTED_REPORT.docx"); });
