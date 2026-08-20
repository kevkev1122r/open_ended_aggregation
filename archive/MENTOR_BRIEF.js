const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  LevelFormat, PageOrientation, PageBreak,
} = require("docx");
const fs = require("fs");
const W = 9360;
const NAVY = "1F3864", GREY = "595959", RED = "9C0006", GREEN = "1E6B34", AMB = "8A6D1F";

const P = (t, o = {}) => new Paragraph({ spacing: { after: o.after ?? 120, line: 276 },
  children: [new TextRun({ text: t, bold: o.bold, italics: o.italics, color: o.color,
    size: o.size ?? 21, font: "Calibri" })] });
const PR = (runs, o = {}) => new Paragraph({ spacing: { after: o.after ?? 120, line: 276 },
  children: runs.map(([t, f = {}]) => new TextRun({ text: t, bold: f.bold, italics: f.italics,
    color: f.color, font: f.mono ? "Consolas" : "Calibri", size: f.size ?? 21 })) });
const H = (t, lv) => new Paragraph({ heading: lv,
  spacing: { before: lv === HeadingLevel.HEADING_1 ? 300 : 220, after: 130 },
  children: [new TextRun({ text: t, bold: true, font: "Calibri",
    size: lv === HeadingLevel.HEADING_1 ? 28 : 23,
    color: lv === HeadingLevel.HEADING_1 ? NAVY : "2E5496" })] });
const BUL = (t, o = {}) => new Paragraph({ numbering: { reference: "b", level: 0 },
  spacing: { after: 85, line: 272 },
  children: [new TextRun({ text: t, font: "Calibri", size: 21, bold: o.bold, color: o.color })] });
const MONO = (t) => new Paragraph({ spacing: { after: 90, line: 240 },
  shading: { type: ShadingType.CLEAR, fill: "F4F4F4" },
  children: [new TextRun({ text: t, font: "Consolas", size: 17 })] });
const CALL = (title, body, color) => new Table({
  width: { size: W, type: WidthType.DXA }, columnWidths: [W],
  borders: { top: { style: BorderStyle.SINGLE, size: 2, color }, bottom: { style: BorderStyle.SINGLE, size: 2, color },
             left: { style: BorderStyle.SINGLE, size: 18, color }, right: { style: BorderStyle.SINGLE, size: 2, color } },
  rows: [new TableRow({ children: [new TableCell({
    width: { size: W, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: "F7F7F7" },
    margins: { top: 130, bottom: 130, left: 190, right: 190 },
    children: [P(title, { bold: true, color, after: 75 }), ...body.map(b => P(b, { after: 55 }))] })] })] });
function table(headers, rows, widths) {
  const hdr = new TableRow({ tableHeader: true, children: headers.map((h, i) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: NAVY },
    margins: { top: 85, bottom: 85, left: 115, right: 115 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
      children: [new TextRun({ text: h, bold: true, color: "FFFFFF", size: 18, font: "Calibri" })] })] })) });
  const body = rows.map((r, ri) => new TableRow({ children: r.map((c, i) => {
    const isStr = typeof c === "string", txt = isStr ? c : c.t;
    return new TableCell({ width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: (!isStr && c.hi) ? "E8F0E4" : (ri % 2 ? "F2F2F2" : "FFFFFF") },
      margins: { top: 75, bottom: 75, left: 115, right: 115 },
      children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
        children: [new TextRun({ text: txt, size: 18, font: "Calibri",
          bold: !isStr && c.bold, color: !isStr ? c.color : undefined })] })] }); }) }));
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows: [hdr, ...body] });
}

const doc = new Document({
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 340, hanging: 210 } } } }] }] },
  styles: { default: { document: { run: { font: "Calibri", size: 21 } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
      margin: { top: 1300, bottom: 1300, left: 1300, right: 1300 } } },
    children: [
      new Paragraph({ spacing: { after: 50 }, children: [new TextRun({
        text: "Open-Ended LLM Aggregation — status and where to go", bold: true, size: 36,
        color: NAVY, font: "Calibri" })] }),
      new Paragraph({ spacing: { after: 240 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 8 } },
        children: [new TextRun({ text: "Kevin Chou   ·   16 August 2026   ·   4 benchmarks, ~11,000 generations, ~$55 Azure grant",
          size: 19, color: GREY, font: "Calibri" })] }),

      CALL("The ask", [
        "The method does not work, and we now know why. A paper published in June (67 models, formal bounds) proves the ceiling that our results kept hitting.",
        "That paper's theorem is scoped to policies that OUTPUT ONE MODEL'S ANSWER, and it explicitly excludes open-ended judged tasks. Both exclusions are where our remaining work sits.",
        "I want your read on which of the two openings in §5 to take, or whether to write up the measurement results and stop.",
      ], AMB),

      new Paragraph({ spacing: { after: 180 }, children: [] }),

      H("1.  What we set out to do", HeadingLevel.HEADING_1),
      P("The ICML 2026 paper we extend (\"Beyond Majority Voting\", arXiv:2510.01499) weights each model by its accuracy and takes a vote. Provably optimal for multiple choice; breaks on open-ended answers, because it needs every wrong answer to be equally likely. Extending it to open-ended generation is that paper's own closing question."),
      P("Our method (KWA) replaces the exact-match indicator with a similarity kernel, keeping their weights. It reduces exactly to their rule when similarity is exact match — verified numerically at 1.0000."),

      H("2.  What we tested", HeadingLevel.HEADING_1),
      table(["benchmark", "answer form", "grading", "n"], [
        ["TriviaQA / GSM8K / MedQA / MMLU-Pro", "4 words median", "LLM judge", "970"],
        ["FACTS Grounding", "214 words", "sentence-level groundedness", "292"],
        ["ASQA", "80 words prose", "STR-EM, no judge", "280"],
        ["QAMPARI", "entity lists", "set F1, no judge", "198"]], [3900, 2000, 2200, 1260]),
      new Paragraph({ spacing: { after: 130 }, children: [] }),
      P("Five tests of the weighting idea across three granularities — whole response, sentence, list item. Weights estimated both label-free (their EM) and supervised."),
      PR([["Result: ", {}], ["the weighting never beat its unweighted counterpart with a confidence interval excluding zero, on any benchmark, at any granularity.", { bold: true }]]),

      H("3.  Why — and this is now settled externally", HeadingLevel.HEADING_1),
      PR([["arXiv 2606.27288, ", {}], ["\"When Does Combining Language Models Help? A Co-Failure Ceiling\"", { italics: true }],
          [" (June 2026, 67 frontier models, 21 providers).", {}]]),
      MONO("  Prop. 1: any SELECTION policy — router, weighted vote, or cascade,"),
      MONO("  whose output is almost surely one of the members' answers —"),
      MONO("  has accuracy at most 1 − β,  β = rate at which ALL models fail."),
      BUL("They prove the diagnostic the field reports (pairwise error correlation ρ) cannot identify β. Our own 215–621× correlation figure is exactly that statistic, so that finding of ours is both superseded and methodologically criticised."),
      BUL("They tested query-level routing — the more sophisticated version of our per-domain weights — with a learned router: it captured ~9% of available gain, CI spanning zero. \"Realizable routing gain is near zero on the 2026 frontier.\""),
      BUL("Our negatives are now expected rather than disappointing. We should cite this paper, not compete with it.", { bold: true }),

      H("4.  What we found that they did not", HeadingLevel.HEADING_1),
      P("Their limitations section states: \"Programmatic grading covers verifiable tasks only... open-ended quality would reintroduce judge bias.\" They stop where we spent four months. Six artifacts, all quantified from our own runs:"),
      table(["measurement artifact", "size"], [
        [{ t: "binary vs sentence-level grading of identical responses", bold: true }, { t: "2.9 → 21.2 pts headroom", bold: true }],
        ["a one-sided metric making \"no method at all\" optimal", "beats every method by 4.8"],
        ["in-sample weight fitting", "1.86 pts"],
        ["uncalibrated interaction null", "2.46 pts"],
        ["two good-faith implementations of the same table", "0.93 pts"],
        ["judge self-contradiction on identical answers", "7%"]], [6300, 3060]),
      new Paragraph({ spacing: { after: 130 }, children: [] }),
      PR([["Published gains in this literature are 1–3 points. ", {}],
          ["Every artifact above is that size or larger.", { bold: true }],
          [" Independently corroborated by arXiv 2504.18413, which finds automatic long-form metrics are biased by style and length and that fine-grained evaluation mitigates it.", {}]]),

      new Paragraph({ children: [new PageBreak()] }),

      H("5.  The two openings", HeadingLevel.HEADING_1),
      H("A.  Composition is outside their theorem", HeadingLevel.HEADING_2),
      P("The 1−β ceiling binds policies whose output IS one of the member answers. Assembling a new answer from fragments across models is not such a policy — it is formally outside the bound. We have the pipeline built (Atomic Self-Consistency + our weights, judge-free on QAMPARI)."),
      table(["method", "F1", "vs best single"], [
        ["ASC count filter (published baseline)", "29.5%", "+0.94"],
        ["ASC + our weights", "29.3%", "+0.74"],
        ["best single model", "28.6%", "—"]], [5000, 2180, 2180]),
      new Paragraph({ spacing: { after: 130 }, children: [] }),
      BUL("Merging edges past best-single; neither margin clears zero at n=198."),
      BUL("Our weights make it slightly worse, and we know the mechanism: with four models inside a 0.12 reliability band, weights cannot reorder decisions, only tighten the threshold — and the count threshold was already at its optimum.", { bold: true }),
      BUL("Honest read: the question \"does composition break the co-failure ceiling?\" is well posed and open. Our weights are not the way to answer it."),

      H("B.  The open-ended half they excluded", HeadingLevel.HEADING_2),
      P("They bound verifiable tasks and name judge bias as the reason for stopping. We have four benchmarks of open-ended data and the six artifacts above. \"What happens to the co-failure ceiling when grading is not programmatic?\" is motivated by their own limitations section."),
      BUL("This is the option I lean toward. It is a measurement paper, not a method paper — but it is supported by data we already have, needs no new API spend, and fills a gap a strong recent paper explicitly names."),

      H("6.  What is dead — so we do not revisit it", HeadingLevel.HEADING_1),
      BUL("Reliability weighting, at every granularity. Five tests, zero wins."),
      BUL("Bayes-optimality of our rule. The decision rule drops a normaliser the estimator computes; we implemented the correct version and it is worse (−3.61, CI excludes zero)."),
      BUL("Per-question / per-domain weights. Ours failed; theirs failed at 67-model scale."),
      BUL("Label-free model selection as a reframe — occupied by the CAI Ratio (ICLR 2025 workshop)."),
      BUL("The ρ diagnostic — proved unable to identify the quantity that actually binds."),

      H("7.  What I would want you to push back on", HeadingLevel.HEADING_1),
      BUL("Three measurement bugs in one session all favoured our method and none favoured the null. That is the signature of scrutinising the interesting number and trusting the boring ones. Every headline should be re-derived independently before it leaves the group.", { bold: true }),
      BUL("Rankings among our aggregators flipped three times — with sample size, and between two good-faith implementations of the same procedure. No ordering should be quoted as a result."),
      BUL("Whether a measurement paper is worth the group's time, or whether this should be written up as an internal negative result and closed."),
    ],
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync("MENTOR_BRIEF.docx", b);
  console.log("wrote MENTOR_BRIEF.docx"); });
