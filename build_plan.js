const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  LevelFormat, convertInchesToTwip, PageOrientation, ImageRun,
} = require("docx");

const DIR = "/Users/kevinchou/Documents/open-ended-aggregation";
const ACCENT = "1A73E8";
const GREY = "5F6368";

// ---------------------------------------------------------------- helpers
const P = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: opts.after ?? 120, line: 276 },
    alignment: opts.align,
    children: [new TextRun({ text, size: opts.size ?? 21, color: opts.color, bold: opts.bold, italics: opts.italics, font: opts.font })],
    ...(opts.border ? { border: opts.border } : {}),
  });

const Rich = (runs, opts = {}) =>
  new Paragraph({
    spacing: { after: opts.after ?? 120, line: 276 },
    children: runs.map(r => new TextRun({ size: 21, ...r })),
  });

const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 140 } });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 100 } });

const Bullet = (text, lvl = 0) =>
  new Paragraph({ numbering: { reference: "bul", level: lvl }, spacing: { after: 70, line: 264 },
    children: [new TextRun({ text, size: 21 })] });

const Num = (text) =>
  new Paragraph({ numbering: { reference: "num", level: 0 }, spacing: { after: 80, line: 264 },
    children: [new TextRun({ text, size: 21 })] });

const Code = (text) =>
  new Paragraph({
    spacing: { after: 60, before: 60 }, indent: { left: convertInchesToTwip(0.25) },
    children: [new TextRun({ text, size: 18, font: "Menlo", color: "202124" })],
  });

const Rule = () =>
  new Paragraph({ spacing: { after: 160, before: 40 }, children: [new TextRun({ text: "", size: 2 })],
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "DADCE0", space: 1 } } });

// table with dual widths (DXA everywhere)
const TOTAL = 9360;
function mkTable(header, rows, widths) {
  const cell = (txt, { bold = false, shade = null, align = AlignmentType.LEFT, w }) =>
    new TableCell({
      width: { size: w, type: WidthType.DXA },
      shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: "auto" } : undefined,
      margins: { top: 60, bottom: 60, left: 90, right: 90 },
      children: [new Paragraph({ alignment: align, spacing: { after: 0 },
        children: [new TextRun({ text: txt, size: 19, bold })] })],
    });
  return new Table({
    columnWidths: widths,
    width: { size: TOTAL, type: WidthType.DXA },
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) =>
        cell(h, { bold: true, shade: "E8F0FE", w: widths[i], align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) =>
        cell(String(c), { w: widths[i], shade: ri % 2 ? "F8F9FA" : null,
          bold: i === 0 && String(c).startsWith("KWA"),
          align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) })),
    ],
  });
}

function img(file, widthPx, heightPx) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(`${DIR}/results/figures/${file}`),
      transformation: { width: widthPx, height: heightPx } })],
  });
}
const Caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: t, size: 17, italics: true, color: GREY })] });

// ---------------------------------------------------------------- content
const doc = new Document({
  creator: "research plan",
  title: "Kernel-Weighted Aggregation for Open-Ended LLM Answers",
  numbering: {
    config: [
      { reference: "bul", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 240 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 860, hanging: 240 } } } },
      ]},
      { reference: "num", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 240 } } } },
      ]},
    ],
  },
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "202124" } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: ACCENT } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1080, bottom: 1080, left: 1260, right: 1260 } } },
    children: [
      // ---------------- title
      new Paragraph({ spacing: { after: 60 }, children: [new TextRun({
        text: "Kernel-Weighted Aggregation for Open-Ended LLM Answers", bold: true, size: 36, color: "202124" })] }),
      new Paragraph({ spacing: { after: 40 }, children: [new TextRun({
        text: "A research plan, with a synthetic pilot and a real-data validation of its core assumption", size: 22, color: GREY })] }),
      new Paragraph({ spacing: { after: 260 }, children: [new TextRun({
        text: "Draft — the effect is validated on real data; the log-LINEAR form is refuted and needs a curved kernel (Section 4.5–4.6)", size: 18, italics: true, color: "B3261E" })],
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "DADCE0", space: 6 } } }),

      // ---------------- 1
      H1("1. The opening"),
      Rich([
        { text: "Ai, Pan, Simchi-Levi, Tambe and Xu, " },
        { text: "Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information", italics: true },
        { text: " (ICML 2026, arXiv:2510.01499), derive a Bayes-optimal aggregator for K-way multiple choice and close with an explicit open question:" },
      ]),
      new Paragraph({
        spacing: { before: 100, after: 140 }, indent: { left: convertInchesToTwip(0.3) },
        border: { left: { style: BorderStyle.SINGLE, size: 14, color: ACCENT, space: 10 } },
        children: [new TextRun({ text: "“How to derive optimal weights for open-ended questions?”", italics: true, size: 22 })],
      }),
      P("This project answers that question, and argues along the way that it is subtly mis-posed."),

      H2("Why their derivation stops at multiple choice"),
      P("Optimal Weight (OW) needs exactly two facts, both manufactured by randomly shuffling the answer labels before querying:"),
      Code("(1)  P(agent j says s  | truth = s)  =  x_j"),
      Code("(2)  P(agent j says s' | truth = s)  =  (1 - x_j)/(K - 1)   for EVERY s' ≠ s"),
      P("Fact (2) — every wrong answer equally likely — is the load-bearing one. It makes a constant factor out of the likelihood product, and what remains is weight × indicator: a vote."),
      P("Open-ended generation destroys it. There is no label set to shuffle, and when the truth is “Canberra”, “Sydney” is a far likelier error than “photosynthesis”. Errors concentrate on plausible neighbours, which is precisely the structure fact (2) assumes away."),

      H2("The reframing"),
      P("Work out the general posterior without fact (2):"),
      Code("score(s) = Σ_j log P(a_j | truth = s)"),
      Code("         = Σ_{j: a_j = s} log x_j  +  Σ_{j: a_j ≠ s} log e_j(a_j, s)"),
      Code("                                        ↑ depends on WHICH wrong answer — will not factor out"),
      Rich([
        { text: "So for open-ended you do not need a weight per agent. " },
        { text: "You need a weight and a kernel.", bold: true },
      ], { after: 160 }),

      // ---------------- 2
      H1("2. Proposed method"),
      P("Model each agent's output distribution as decaying with semantic distance from the truth, at an agent-specific rate:"),
      Code("log P(a | truth = s)  =  α_j  +  β_j · sim(a, s)"),
      P("β_j is how sharply agent j concentrates its probability mass near the truth: a strong agent has large β and answers close to correct; a weak agent sprays. Since α_j does not depend on s, the Bayes rule is:"),
      Code("KWA:   argmax_s  Σ_j  β_j · sim(a_j, s)"),

      H2("It is a strict generalisation, not a rival"),
      P("Set sim(a, s) = 1{a = s}. Normalising over K options gives P(correct) = e^β / (e^β + K − 1) = σ_K(β) — exactly the paper's own σ_K parametrisation from their Section D.1 — and the rule collapses to OW. Multiple choice is the special case where similarity happens to be binary. This is verified numerically in the pilot (Section 4)."),

      H2("Estimating β with no labels"),
      P("EM with the truth as a latent variable ranging over the candidate pool:"),
      Bullet("E-step: posterior over which candidate is the truth, given the current β."),
      Bullet("M-step: separates across agents; each per-agent objective is linear minus log-sum-exp, hence concave, so each is a reliable one-dimensional maximisation."),
      P("This is Dawid–Skene generalised from a confusion matrix to a similarity kernel, and it is the direct analogue of the paper's OW-L (which inverts an agreement rate; this inverts an agreement distribution)."),

      // ---------------- 3
      H1("3. Hypotheses"),
      mkTable(
        ["#", "Hypothesis", "Status after pilot"],
        [
          ["H1", "KWA reduces exactly to OW when the kernel is exact match", "Confirmed"],
          ["H2", "β is recoverable from unlabelled data alone", "Confirmed, with a caveat"],
          ["H3", "KWA beats cluster-then-vote on open-ended answers", "Confirmed, margin smaller than hoped"],
          ["H4", "The gain comes from pooling votes split across paraphrases", "REFUTED — see Section 5"],
          ["H5", "KWA is robust to kernel misspecification", "Confirmed"],
          ["H7", "errors cluster near the truth, on real data", "CONFIRMED — 4 encoders, 3 tasks, placebo passes"],
          ["H9", "log P is LINEAR in similarity where it matters", "REFUTED in 9/9 by logistic regression"],
          ["H10", "the effect replicates across corpora and encoders", "CONFIRMED 9/9, d = 1.43–3.14, placebo passes"],
          ["H8", "the synthetic pool geometry resembles reality", "REFUTED — real gap is 5x narrower"],
          ["H6", "KWA can exceed the ‘someone must be right’ ceiling", "Confirmed, and it is the best result"],
        ],
        [620, 5340, 3400]),
      P("", { after: 160 }),

      // ---------------- 3.5  REAL DATA
      H1("4. The make-or-break test, on real data"),
      P("The whole method rests on one assumption: that log-probability decays linearly with similarity to the truth. Two things must hold. (G) errors must land near the truth in embedding space, or similarity carries no signal at all. (L) log-odds must fall linearly in similarity, or the aggregator is misspecified. Both were tested on real text with real embeddings. (G) survives every attack. (L) does not — it holds only when the candidate pool is topically wide, and fails in the narrow regime the aggregator actually operates in."),
      P("Data, nothing simulated: TruthfulQA (817 questions — a best answer, human-written correct paraphrases, and human-written plausible misconceptions) and HaluEval QA (10,000 questions — a right answer and a hallucinated answer generated by a real model). Embeddings from all-MiniLM-L6-v2, cosine similarity."),

      H2("4.1  Do real model errors land near the truth?  Yes."),
      mkTable(
        ["similarity to the right answer", "mean", "sd"],
        [["the model's OWN wrong answer", "0.302", "0.192"],
         ["another question's wrong answer", "0.079", "0.088"]],
        [5360, 2000, 2000]),
      P("", { after: 80 }),
      P("Cohen's d = 1.49, and the model's own error is nearer the truth than an unrelated error in 86.7% of questions. Model mistakes are not scattered — they cluster around the truth. There is signal to exploit."),

      H2("4.2  First pass: a straight line fits at R² = 0.99 — but see 4.5"),
      P("Case-control over 4,000 real model outputs against 32,000 controls, binned by similarity to the truth:"),
      Code("straight-line fit:  log-odds = 10.69 · sim − 1.86"),
      Code("  R² (linear)     0.990"),
      Code("  R² (quadratic)  0.991      ← curvature buys +0.001"),
      P("Ten bins, dead straight, and the quadratic term buys essentially nothing. Taken alone this looks like a clean confirmation, and the fitted slope would be β — measured at 10.69. Section 4.5 shows this particular design flatters the hypothesis, and the result does not survive a harder control set."),
      img("R1_real_data_test.png", 560, 224),
      Caption("Figure 1 — Left: real answer geometry. Correct paraphrases (green) and plausible wrong answers (orange) overlap almost completely. Right: the assumed law, confirmed."),

      H2("4.3  But the geometry is far harder than the pilot assumed"),
      mkTable(
        ["similarity to the correct answer", "measured", "pilot assumed"],
        [["correct paraphrase", "0.626", "0.975"],
         ["plausible WRONG answer", "0.580", "0.750"],
         ["unrelated answer", "0.058", "0.300"],
         ["correct-vs-plausible-wrong gap", "0.046", "0.225"]],
        [4760, 2300, 2300]),
      P("", { after: 80 }),
      P("Correct paraphrases and plausible wrong answers sit almost on top of each other. The synthetic pool made that discrimination roughly five times easier than it really is."),

      H2("4.4  Re-running with the measured geometry"),
      P("Fully deployable variant, no oracle anywhere, and with the baseline tuned to its own best clustering threshold so it is not a strawman:"),
      Code("MV-cluster, τ swept over [0.30 … 0.90], best τ = 0.60      88.27"),
      Code("KWA-EM (deployable)                                       93.27"),
      Code("                                             advantage    +5.00"),
      Rich([
        { text: "The advantage grew, from +1.6 synthetic to +5.0 realistic. ", bold: true },
        { text: "The overlap is the reason: when correct and plausible-wrong answers are 0.05 apart, no hard threshold can separate them — cluster too loosely and wrong answers merge in, too tightly and nothing merges. A soft kernel never has to make that binary call. Real geometry is precisely where hard clustering breaks and soft weighting wins." },
      ]),
      H2("4.5  Stress-testing that result — and it does not fully survive"),
      P("Three tests designed to break the claim rather than confirm it."),
      Rich([{ text: "T1 — Encoder robustness: passes. ", bold: true },
            { text: "Four unrelated encoders on HaluEval QA give R² between 0.959 and 0.993 (β 10.87 / 11.61 / 14.61 / 11.49). β moves because each encoder has its own similarity scale; R² does not. Not a MiniLM artefact." }]),
      Rich([{ text: "T2 — Task family: partly passes. ", bold: true },
            { text: "QA fits at R² 0.991, but dialogue is already visibly worse as a straight line (0.858) while staying excellent with curvature (0.989). Summarization could not be fit with random controls at all — positives sit at 0.73 and controls at 0.10, so the supports barely overlap." }]),
      Rich([{ text: "T3 — Control design: the headline claim breaks. ", bold: true, color: "B3261E" },
            { text: "The first pass drew controls from random other questions, so they were all off-topic — which alone can manufacture a linear-looking trend. Redrawing them from the NEAREST other questions:" }]),
      mkTable(
        ["task", "controls", "β", "R² linear", "verdict"],
        [["qa", "random", "10.96", "0.991", "holds"],
         ["qa", "hard (nearest)", "2.60", "0.685", "breaks"],
         ["dialogue", "random", "7.01", "0.858", "weak"],
         ["dialogue", "hard (nearest)", "2.33", "0.349", "breaks"],
         ["summarization", "hard (nearest)", "11.34", "0.977", "holds"]],
        [2060, 2500, 1400, 1700, 1700]),
      P("", { after: 90 }),
      P("The placebo passes everywhere: randomly relabelling which answers were actually produced collapses the slope to |β| ≤ 0.088 with R² ≤ 0.062. The pipeline is not inventing structure from nothing — the machinery is sound, and it is the functional form that fails."),

      H2("4.6  The gradient — what is actually true"),
      P("Sweeping control hardness continuously, from the nearest other question (hardest) out to random (easiest):"),
      Code("controls =    r=1     r=2     r=5    r=10    r=50   r=200  r=1000  random"),
      Code("β            2.60    2.84    3.32    3.81    5.36    7.45   10.57   10.87"),
      Code("R² straight  0.685   0.717   0.757   0.790   0.853   0.928   0.975   0.993"),
      Code("R² curved    0.987   0.979   0.982   0.982   0.985   0.972   0.983   0.994"),
      P("Read the last row. Allowing curvature, R² stays between 0.97 and 0.99 across the whole gradient, including the hardest regime. The relationship is real, smooth and highly predictable throughout. What collapses is specifically the assumption that it is a straight line."),
      img("R2_control_gradient.png", 560, 218),
      Caption("Figure 2 — Red: straight-line fit, collapses as controls get harder. Green: allowing curvature, flat and near-perfect throughout. Blue: the fitted β, which is not a constant."),
      P("Three-line verdict:"),
      Num("Errors cluster near the truth — bulletproof. Placebo passes, four encoders, three task families, Cohen's d between 1.5 and 5.2."),
      Num("β is not a constant. It runs from about 2.6 to about 11 depending on how topically spread the candidate pool is. Most of the original R² = 0.99 was a topic-matching effect."),
      Num("The log-linear form is misspecified in the regime that matters. The aggregator chooses among answers to the SAME question — all topic-matched — which is exactly the r=1 end where a straight line fits worst (R² 0.35–0.69) and curvature fits nearly perfectly (0.96–0.99)."),
      Rich([{ text: "One caveat cutting the other way: ", bold: true },
            { text: "in HaluEval QA, 30.2% of questions have a nearest neighbour whose correct answer is more than 0.8 similar (23.6% above 0.9) — the dataset is built from HotpotQA and contains many near-duplicate questions. The hard controls are therefore contaminated with answers that are effectively correct, which flattens β. So β = 2.60 is a biased-low bound, and the true hard-regime value lies somewhere between 2.6 and 11." }]),
      H2("4.7  Triple replication — three independent datasets, three encoders, bootstrap CIs"),
      P("The whole protocol re-run on three corpora sharing no construction pipeline, chosen to differ in where the wrong answers come from: HaluEval-QA (n=3000, wrong answers GENERATED BY A MODEL), TruthfulQA (n=817, written by EXPERTS as adversarial misconceptions), and SciQ (n=3000, written by CROWDWORKERS as exam distractors). Three encoders each, and bootstrap confidence intervals on every number."),
      Rich([{ text: "The effect is bulletproof — 9 of 9 cells. ", bold: true },
            { text: "Cohen's d runs from 1.43 to 3.14 across every dataset × encoder cell, and the placebo passes in all nine (max |β| = 0.277, max R² = 0.153, against real β of 2.6–20.6). Errors cluster near the truth: not a corpus quirk, not an encoder quirk, not an artefact of the pipeline." }]),
      Rich([{ text: "β is not a constant — it spans 2.6 to 20.6. ", bold: true },
            { text: "Within a single cell it moves four- to sevenfold as controls go from topic-matched to random. β is a property of the encoder, corpus and candidate-pool spread together, not of a model. Any method treating it as fixed is wrong." }]),
      P("At the hardest (topic-matched) controls, R² for a straight line against R² allowing curvature, with bootstrap 95% intervals:"),
      mkTable(
        ["dataset", "encoder", "straight", "curved", "CIs"],
        [["HaluEval-QA", "MiniLM-L6", "0.685 [0.58,0.76]", "0.987 [0.93,0.99]", "disjoint"],
         ["HaluEval-QA", "mpnet-base", "0.691 [0.60,0.76]", "0.970 [0.91,0.98]", "disjoint"],
         ["HaluEval-QA", "bge-small", "0.824 [0.73,0.89]", "0.991 [0.94,0.99]", "disjoint"],
         ["TruthfulQA", "MiniLM-L6", "0.441 [0.35,0.52]", "0.959 [0.90,0.98]", "disjoint"],
         ["TruthfulQA", "mpnet-base", "0.426 [0.34,0.50]", "0.956 [0.89,0.97]", "disjoint"],
         ["TruthfulQA", "bge-small", "0.527 [0.42,0.63]", "0.941 [0.86,0.98]", "disjoint"],
         ["SciQ", "MiniLM-L6", "0.983 [0.93,0.99]", "0.983 [0.93,0.99]", "overlap"],
         ["SciQ", "mpnet-base", "0.991 [0.95,0.99]", "0.994 [0.96,1.00]", "overlap"],
         ["SciQ", "bge-small", "0.962 [0.89,0.98]", "0.988 [0.94,0.99]", "overlap"]],
        [1900, 1700, 2280, 2280, 1200]),
      P("", { after: 90 }),
      Rich([{ text: "Curvature is required in 6 of 9 cells with non-overlapping bootstrap intervals, and the split is perfectly clean by dataset rather than by encoder. ", bold: true },
            { text: "All three encoders agree within each corpus." }]),
      P("The exception is the informative part. SciQ is the one corpus whose wrong answers were written independently of any model — crowdworkers inventing plausible science terms — and there a straight line fits fine even at the hardest controls. HaluEval's errors were generated by a model, and TruthfulQA's were engineered to fool models; in both, errors pile up at the top of the plausibility range, which saturates the response and bends the curve."),
      Rich([{ text: "That is the worst possible arrangement for the method's convenience: the two corpora that actually represent model behaviour are exactly the two where the linear form fails.", bold: true, color: "B3261E" }]),
      img("R3_triple_replication.png", 560, 306),
      Caption("Figure 3 — Top: straight-line fit (red) against curvature-allowed fit (green), shaded bands are bootstrap CIs. Bottom: fitted β. The encoder does not matter; the corpus does."),
      P("Verdict, with numbers behind every clause: (1) errors cluster near the truth — 9/9 cells, d = 1.43–3.14, placebo passes 9/9; (2) β is not a constant — 2.6 to 20.6, so it must be estimated per deployment and never assumed; (3) the log-linear form is refuted for model-generated errors — 6/9 cells, intervals disjoint, consistent across all three encoders; (4) a curved kernel suffices everywhere — R² above 0.94 in all nine cells including the three where linear already worked, so adopting curvature costs nothing and fixes the failures.", { after: 150 }),

      H2("4.8  Proper regression — which corrects two of the findings above"),
      P("Everything above binned similarity into ten slices, fit a line to those ten summary points, and compared R² by eye. That discards everything inside a bin and is not a hypothesis test. The correct analysis is logistic regression on the raw rows: y = 1 for an answer a model really produced, y = 0 for a control, x = similarity to the truth, so that logit P(y=1) = α + β·sim is exactly the assumed model fit on all ~27,000 rows. Adding a squared term and running a likelihood-ratio test converts “the curve looks better” into a p-value. Standard errors are clustered by question, since each question contributes one positive and eight controls and naive errors would be far too small."),
      mkTable(
        ["dataset", "encoder", "β (95% CI)", "bend", "LR test p", "AIC gain"],
        [["HaluEval-QA", "MiniLM-L6", "3.96 [3.69, 4.24]", "+5.63", "<1e-16", "109"],
         ["HaluEval-QA", "mpnet-base", "3.74 [3.47, 4.01]", "+4.58", "<1e-16", "74"],
         ["HaluEval-QA", "bge-small", "7.07 [6.63, 7.51]", "+17.15", "<1e-16", "133"],
         ["TruthfulQA", "MiniLM-L6", "8.23", "+18.49", "<1e-16", "384"],
         ["TruthfulQA", "mpnet-base", "8.55", "+18.95", "<1e-16", "380"],
         ["TruthfulQA", "bge-small", "14.20 [13.16, 15.24]", "+46.17", "<1e-16", "319"],
         ["SciQ", "MiniLM-L6", "2.44 [2.25, 2.63]", "−3.87", "<1e-16", "85"],
         ["SciQ", "mpnet-base", "2.48 [2.29, 2.67]", "−3.93", "<1e-16", "87"],
         ["SciQ", "bge-small", "5.12 [4.71, 5.53]", "−9.40", "8e-10", "36"]],
        [1700, 1500, 2260, 1300, 1300, 1300]),
      P("", { after: 90 }),
      Rich([{ text: "Correction 1 — curvature is required in 9 of 9 cells, not 6 of 9. ", bold: true, color: "B3261E" },
            { text: "SciQ looked linear under binning only because ten bins were too coarse to see its bend. The earlier “6 of 9” was an artefact of my own crude method, not a property of the data." }]),
      Rich([{ text: "Correction 2 — the bend goes in opposite directions, which the binned analysis could not have found. ", bold: true, color: "B3261E" },
            { text: "For HaluEval and TruthfulQA — model-made and model-targeting errors — the bend is positive and the curve accelerates: as answers get very close to the truth, the odds of being a real error rise faster than linearly. For SciQ — hand-written exam distractors — the bend is negative and the curve saturates, flattening off at high similarity. So “use a curve” is not sufficient guidance: the shape depends on where the errors come from, and the two regimes bend opposite ways. A single fixed correction will not serve both." }]),
      img("R4_regression_curves.png", 560, 181),
      Caption("Figure 4 — Fitted logistic curves. Blue dots are the binned observations shown only for reference; the lines are fit to all raw rows."),
      P("A third correction: the binned β estimates were biased low. For HaluEval with MiniLM under hard controls, binning gave 2.60 while the regression gives 3.96 [3.69, 4.24]. The regression estimate is the trustworthy one, and any β quoted in the preceding sections should be read as approximate."),
      H2("4.9  How much does similarity actually explain?"),
      P("With 27,000 rows a p-value below 1e-16 is cheap, so the effect size is what matters. McFadden pseudo-R², straight then curved, under hard controls:"),
      mkTable(
        ["dataset", "straight", "curve", "gain"],
        [["HaluEval-QA", "0.053", "0.059", "+0.006"],
         ["TruthfulQA", "0.379", "0.454", "+0.075"],
         ["SciQ", "0.030", "0.035", "+0.005"]],
        [3360, 2000, 2000, 2000]),
      P("", { after: 90 }),
      Rich([{ text: "Similarity is a strong discriminator on TruthfulQA and a weak one on HaluEval and SciQ. ", bold: true },
            { text: "McFadden values are not comparable to ordinary R² — 0.2 to 0.4 is considered a good fit — so 0.03 to 0.05 is modest but genuine. The effect itself is not in doubt anywhere: z-statistics run from 25 to 62 with question-clustered standard errors. But on some corpora similarity provides a gentle tilt rather than a decisive separation, so the aggregator should not be expected to work equally well in every setting." }], { after: 150 }),

      Rich([{ text: "What this changes about the method: ", bold: true },
            { text: "the kernel should not be β·sim but β·g(sim) for a monotone curved g — a quadratic already captures 98% of the variance. Section 7 tested exactly this misspecification synthetically and found KWA still led at every curvature from 0.5 to 2.0, so the method tolerates it; but the direction of the error is now known and can be corrected rather than tolerated. That is the top-priority change." }]),
      P("What none of this establishes: HaluEval hallucinations are one model's errors, deliberately elicited, and not a broad sample of behaviour across models; there is still no end-to-end run on genuine multi-model free-form outputs, because no local model or API key was available; and absolute similarity values are encoder-specific, so only the ordering should be trusted.", { after: 160 }),


      // ---------------- 4
      H1("5. Synthetic pilot study"),
      Rich([
        { text: "Everything in this section is synthetic, ", bold: true, color: "B3261E" },
        { text: "and its accuracy numbers should be discounted in light of Section 4.3. It tests whether the estimator and the aggregator work, not whether the model is true of real language models." },
      ]),
      P("Setup: agents draw from a pool of candidate answers laid out in a similarity geometry — the truth, some paraphrases of it (counted as correct), plausible-but-wrong near distractors, and implausible far ones. Agent j picks candidate c with probability proportional to exp(β_j · sim(c, truth)). Default configuration is 5 agents and 1,500 questions, averaged over 5 seeds."),

      H2("5.1  Correctness"),
      P("KWA with an exact-match kernel agrees with OW on 1.0000 of decisions at K = 2, 3, 4, 6 and 10. The multiple-choice case is recovered exactly."),

      H2("5.2  Main comparison"),
      mkTable(
        ["method", "accuracy %", "± sd", "vs MV-exact"],
        [
          ["MV-exact", "79.84", "0.86", "—"],
          ["OW-L  (paper's method, exact match)", "84.80", "1.12", "+4.96"],
          ["OW-oracle  (paper's method, true accuracies)", "85.64", "0.92", "+5.80"],
          ["Best single agent  (clairvoyant)", "85.83", "1.04", "+5.99"],
          ["MV-cluster  (Universal Self-Consistency)", "91.96", "0.47", "+12.12"],
          ["KWA-EM  (label-free)", "96.09", "0.57", "+16.25"],
          ["KWA-oracle  (true β)", "96.13", "0.49", "+16.29"],
        ],
        [4360, 1700, 1500, 1800]),
      P("", { after: 100 }),
      Bullet("Label-free costs essentially nothing: KWA-EM 96.09 against KWA-oracle 96.13."),
      Bullet("The paper's methods transplanted directly onto open-ended answers barely beat the best single model (84.8 / 85.6 against 85.8). That gap is what this project exists to close."),
      img("F1_main_comparison.png", 560, 300),
      Caption("Figure 5 — Open-ended aggregation accuracy on synthetic data. Kernel weighting versus vote-based baselines."),

      H2("5.3  The honest deployable number"),
      P("The table above lets the estimator normalise over the true answer pool. A real system cannot. With no oracle at any step — EM normalising only over answers the agents actually produced:"),
      mkTable(
        ["", "MV-cluster", "KWA-EM (deployable)", "KWA-EM (oracle support)"],
        [
          ["5 agents", "93.09", "94.69", "95.91"],
          ["8 agents", "97.04", "97.78", "99.16"],
        ],
        [2160, 2200, 2600, 2400]),
      P("", { after: 100 }),
      Rich([
        { text: "The real gain over the strong incumbent is roughly +1.6 points at 5 agents and +0.7 at 8 — not +4. ", bold: true },
        { text: "Any write-up should lead with this number." },
      ], { after: 160 }),

      // ---------------- 5
      H1("6. The pilot refuted its own main hypothesis"),
      P("H4 predicted KWA would win by pooling votes split across paraphrases of the truth. Varying how many paraphrases exist in the pool:"),
      mkTable(
        ["paraphrases in pool", "0", "1", "2", "3", "4"],
        [
          ["MV-exact", "64.3", "74.6", "80.0", "82.0", "83.2"],
          ["MV-cluster", "64.6", "84.9", "91.2", "94.9", "96.2"],
          ["KWA-EM", "83.5", "92.9", "96.2", "97.1", "98.2"],
          ["KWA − MV-exact", "+19.2", "+18.3", "+16.2", "+15.1", "+15.0"],
        ],
        [2960, 1280, 1280, 1280, 1280, 1280]),
      P("", { after: 100 }),
      P("The gap is largest at zero paraphrases — where vote-splitting cannot possibly be the mechanism, and where cluster-then-vote degenerates to plain majority voting (64.6 against 64.3) because there is nothing to cluster."),
      Rich([
        { text: "The actual mechanism is triangulation. ", bold: true },
        { text: "Every agent's answer is drawn from a distribution centred on the truth, so even a set of entirely wrong answers points at the truth collectively. KWA finds the point of maximum weighted similarity to all of them — closer to a weighted geometric median than to a tally." },
      ]),

      H2("6.1  This breaks a ceiling that binds every vote-based method"),
      P("Any method that outputs one of the agents' answers — majority voting, OW, ISP, Dawid–Skene, all of them — is capped by “at least one agent was right”. Give KWA a candidate set wider than what the ensemble produced and it is not. On a deliberately weak ensemble:"),
      Code("ceiling for any vote-based method              89.7%"),
      Code("MV-cluster                                    55.7%"),
      Code("KWA, restricted to produced answers           76.0%"),
      Code("KWA, wider candidate set                      80.4%"),
      Code(""),
      Code("on questions where EVERY agent was wrong:"),
      Code("  KWA restricted to produced answers           0.0%   (zero by construction)"),
      Code("  KWA with a wider candidate set              34.1%   ← ceiling broken"),
      P("", { after: 80 }),
      img("F3_triangulation.png", 560, 240),
      Caption("Figure 6 — Left: overall accuracy against the vote-based ceiling. Right: accuracy on the subset where every agent was wrong."),
      Rich([
        { text: "Caveat: ", bold: true },
        { text: "the wider candidate set is oracle-supplied in this pilot. In practice it requires a candidate generator (retrieval, or a model asked to propose alternatives), whose quality then becomes the binding constraint. Untested." },
      ], { after: 160 }),

      // ---------------- 6
      H1("7. Robustness"),
      P("Nearer distractors make the advantage larger — the method helps most exactly where the problem is hardest. As the similarity of wrong answers to the truth rises from 0.40 to 0.85, the KWA advantage over exact-match voting grows from +5.7 to +21.7 points."),
      P("Kernel misspecification is survivable: with the true log-probability response curved (β · sim^c for c between 0.5 and 2) while the fit always assumes c = 1, KWA leads at every curvature."),
      P("Correlated agents hurt everyone, and KWA least. Two “families” pulled toward a shared wrong answer, which violates conditional independence:"),
      mkTable(
        ["shared-error strength", "0.0", "1.0", "2.0", "3.0", "drop"],
        [
          ["MV-exact", "79.6", "77.5", "72.0", "60.0", "−19.6"],
          ["MV-cluster", "91.4", "89.9", "84.5", "73.3", "−18.1"],
          ["OW-L", "85.1", "84.1", "77.5", "55.8", "−29.3"],
          ["KWA-EM", "96.0", "95.2", "92.0", "83.2", "−12.8"],
        ],
        [2760, 1300, 1300, 1300, 1300, 1400]),
      P("", { after: 100 }),
      P("OW-L collapses hardest, consistent with agreement-based accuracy estimation being fooled when colluding agents agree for the wrong reason. KWA is the most robust of the four."),
      img("F4_robustness.png", 560, 230),
      Caption("Figure 7 — Left: correlated agents. Right: kernel misspecification."),

      // ---------------- 7
      H1("8. Known weaknesses"),
      Num("Entirely synthetic. Outside the misspecification experiments, the generative process and the fitted model are the same family. Nothing here is evidence about real language models."),
      Num("Similarity is a clean geometric object in the pilot. Real embedding similarity is non-metric, anisotropic, and its failure modes correlate with the very models being aggregated."),
      Num("The correctness threshold (similarity ≥ 0.95 counts as a correct paraphrase) does a great deal of work, and every headline number moves with it."),
      Num("Pool size is fixed at 10 candidates. Real effective answer-set size varies per question and correlates with difficulty."),
      Num("The headline triangulation result depends on a candidate generator that does not yet exist."),
      Num("No baseline from the confidence-weighting literature (CISC, inverse-entropy voting). That is the real incumbent on live tasks, and it is absent here."),
      Num("The strongest agent's β is systematically underestimated by about 20%, and the error does not shrink with more data — an identifiability limit, mirroring OW's weight error-bars blowing up as accuracy approaches 1."),

      // ---------------- 8
      H1("9. Next steps, in priority order"),
      Num("Replace the linear kernel with a curved one, and fit the curve per deployment rather than fixing it. Section 4.8 shows curvature is required in all nine cells, but that it accelerates for model-made errors and saturates for hand-written distractors — opposite directions — so no single fixed shape will serve both."),
      Num("Chase the triangulation result. Build a real candidate generator and see whether the 34% survives contact with reality. If it does, that is the paper — aggregation that can output what no member said is a genuinely new capability."),
      Num("Add the confidence-weighting baselines. KWA has to beat CISC and inverse-entropy voting, not just majority voting."),
      Num("Fix the strong-agent identifiability, either with a prior on β or by pooling multiple samples per agent — self-consistency inside each agent — to sharpen the likelihood at the top end."),
      Num("Build the diversity-aware extension. Correlated agents are the main threat; down-weight agreement between similar agents with a covariance term on top of β."),

      Rule(),
      Rich([
        { text: "Code and raw results: ", color: GREY, size: 18 },
        { text: "kernel_agg.py", font: "Menlo", size: 18 },
        { text: " (library), ", color: GREY, size: 18 },
        { text: "experiments.py", font: "Menlo", size: 18 },
        { text: " (study), ", color: GREY, size: 18 },
        { text: "FINDINGS.md", font: "Menlo", size: 18 },
        { text: " (full write-up), ", color: GREY, size: 18 },
        { text: "results/", font: "Menlo", size: 18 },
        { text: " (JSON, console log, figures). Full suite runs in about 200 seconds.", color: GREY, size: 18 },
      ]),
    ],
  }],
});

Packer.toBuffer(doc).then(b => { fs.writeFileSync(`${DIR}/PLAN.docx`, b); console.log("wrote PLAN.docx", b.length, "bytes"); });
