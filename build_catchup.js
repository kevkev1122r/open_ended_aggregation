const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  LevelFormat, PageOrientation, TableOfContents, PageBreak,
} = require("docx");
const fs = require("fs");

const CONTENT = 9360;                       // US Letter 12240 - 2*1440 margins
const NAVY = "1F3864", GREY = "595959", RED = "9C0006", GREEN = "1E6B34";

const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: 276 },
  alignment: o.align,
  children: [new TextRun({ text, bold: o.bold, italics: o.italics,
    color: o.color, size: o.size ?? 21, font: "Calibri" })],
});

// mixed-format paragraph: runs = [[text, {bold,color,...}], ...]
const PR = (runs, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: 276 },
  children: runs.map(([t, f = {}]) => new TextRun({
    text: t, bold: f.bold, italics: f.italics, color: f.color,
    font: f.mono ? "Consolas" : "Calibri", size: f.size ?? 21 })),
});

const H = (text, level) => new Paragraph({
  heading: level,
  spacing: { before: level === HeadingLevel.HEADING_1 ? 320 : 240, after: 140 },
  children: [new TextRun({ text, bold: true, font: "Calibri",
    size: level === HeadingLevel.HEADING_1 ? 30 : 24,
    color: level === HeadingLevel.HEADING_1 ? NAVY : "2E5496" })],
});

const BUL = (text, o = {}) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  spacing: { after: 90, line: 276 },
  children: [new TextRun({ text, font: "Calibri", size: 21,
    bold: o.bold, color: o.color })],
});

const CALLOUT = (title, body, color) => new Table({
  width: { size: CONTENT, type: WidthType.DXA },
  columnWidths: [CONTENT],
  borders: {
    top:    { style: BorderStyle.SINGLE, size: 2, color: color },
    bottom: { style: BorderStyle.SINGLE, size: 2, color: color },
    left:   { style: BorderStyle.SINGLE, size: 18, color: color },
    right:  { style: BorderStyle.SINGLE, size: 2, color: color },
  },
  rows: [new TableRow({ children: [new TableCell({
    width: { size: CONTENT, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: "F7F7F7" },
    margins: { top: 140, bottom: 140, left: 200, right: 200 },
    children: [
      P(title, { bold: true, color: color, after: 80 }),
      ...body.map((b) => P(b, { after: 60 })),
    ],
  })] })],
});

function table(headers, rows, widths, opts = {}) {
  const hdr = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: NAVY },
      margins: { top: 90, bottom: 90, left: 120, right: 120 },
      children: [new Paragraph({
        alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
        children: [new TextRun({ text: h, bold: true, color: "FFFFFF",
          size: 19, font: "Calibri" })],
      })],
    })),
  });
  const body = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => {
      const isStr = typeof c === "string";
      const txt = isStr ? c : c.t;
      return new TableCell({
        width: { size: widths[i], type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR,
                   fill: (!isStr && c.hi) ? "E8F0E4" : (ri % 2 ? "F2F2F2" : "FFFFFF") },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({
          alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
          children: [new TextRun({ text: txt, size: 19, font: "Calibri",
            bold: !isStr && c.bold, color: !isStr ? c.color : undefined })],
        })],
      });
    }),
  }));
  return new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: widths,
    rows: [hdr, ...body],
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 220 } } } }],
    }],
  },
  styles: { default: { document: { run: { font: "Calibri", size: 21 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children: [
      // ---------------------------------------------------------------- title
      new Paragraph({
        spacing: { after: 60 },
        children: [new TextRun({ text: "Open-Ended LLM Answer Aggregation",
          bold: true, size: 40, color: NAVY, font: "Calibri" })],
      }),
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({ text: "The niche-specialisation experiment — team catch-up",
          size: 26, color: GREY, font: "Calibri" })],
      }),
      new Paragraph({
        spacing: { after: 280 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 8 } },
        children: [new TextRun({ text: "Run 14 August 2026   ·   7 models   ·   996 questions   ·   6,939 responses",
          size: 19, color: GREY, font: "Calibri" })],
      }),

      CALLOUT("Bottom line", [
        "Specialisation across labs is real and measurable — that result holds.",
        "The method does not. Weighting is worth about 1.75 points over a majority vote, but the similarity kernel adds only about 0.5 on its own and nothing once weights are present: weighted exact-match voting (84.74%) scores higher than the full kernel method (84.54%).",
        "And run as designed — with weights estimated label-free, which is what makes it deployable — it scores 82.58%, below both plain majority voting (82.99%) and simply picking the best single model (84.23%).",
      ], RED),

      new Paragraph({ spacing: { after: 200 }, children: [] }),

      // ------------------------------------------------------------ 1. context
      H("1.  What this experiment was for", HeadingLevel.HEADING_1),
      P("The paper we are extending (Beyond Majority Voting, ICML 2026, arXiv:2510.01499) weights each model by its accuracy and takes a vote. That is provably optimal for multiple choice and breaks on open-ended answers, because it needs every wrong answer to be equally likely — and “Sydney” is a far likelier error than “photosynthesis” when the truth is “Canberra”."),
      P("Our method (KWA) replaces exact-match voting with a similarity kernel: pick the answer maximising the sum over models of βⱼ × similarity(answerⱼ, s). Set similarity to exact match and it reduces to the paper's method exactly."),
      PR([
        ["Earlier work on TriviaQA produced a negative result: KWA never beat the better of its two ingredients. The explanation offered was that all six models were unmatched in capability, so a single weight per model just tracked overall skill. ", {}],
        ["This experiment tests the regime that explanation predicts should work: models matched overall but each better in a niche.", { bold: true }],
      ]),

      // ------------------------------------------------------------- 2. setup
      H("2.  What was run", HeadingLevel.HEADING_1),
      P("Seven models, seven different labs — one per lab, because same-lab models share training data and their errors correlate, which is the thing that caps aggregation. Two providers, because Azure had zero quota for Anthropic, OpenAI, Mistral and Meta."),
      table(
        ["Model", "Lab", "Provider", "Accuracy"],
        [
          ["anthropic/claude-sonnet-5", "Anthropic", "OpenRouter", { t: "84.2%", bold: true }],
          ["grok-4.3", "xAI", "Azure", "83.9%"],
          ["openai/gpt-5.4", "OpenAI", "OpenRouter", "82.8%"],
          ["Kimi-K2.5", "Moonshot", "Azure", "82.5%"],
          ["MAI-Thinking-1", "Microsoft", "Azure", "80.8%"],
          ["DeepSeek-V4-Flash", "DeepSeek", "Azure", "80.3%"],
          ["Cohere-command-a-plus", "Cohere", "Azure", "76.9%"],
        ],
        [3200, 1800, 2160, 2200]
      ),
      new Paragraph({ spacing: { after: 140 }, children: [] }),
      P("996 questions across TriviaQA, GSM8K, MedQA and MMLU-Pro (split into six subjects). Graded by an independent LLM judge from a lab with no model in the pool. 970 questions have an answer from all seven models and form the paired analysis set."),

      // ------------------------------------------------------------ 3. results
      H("3.  Result 1 — the models are matched", HeadingLevel.HEADING_2),
      PR([["Spread from best to worst is 7.3 points, and the gap between first and second is ", {}],
          ["0.3 points", { bold: true }],
          [". The premise the experiment needs holds. This is not the earlier regime where one model dominated and weighting merely rediscovered it.", {}]]),

      H("4.  Result 2 — specialisation is real", HeadingLevel.HEADING_2),
      P("Different labs genuinely own different domains: Kimi on maths, Grok on medicine and psychology, Claude on physics and business, MAI on chemistry. After removing model and domain main effects, the leftover interaction has a standard deviation of 3.35 accuracy points, with a maximum of 10.56."),
      PR([["Permutation test: observed 3.35 against a null mean of 2.46 (95th percentile 2.90), ", {}],
          ["p < 0.001", { bold: true, color: GREEN }],
          [".", {}]]),
      CALLOUT("Read this before quoting the specialisation table", [
        "The null mean is 2.46, not zero. With about 110 questions per domain, sampling noise alone manufactures that much apparent specialisation. The genuine excess is roughly 0.9 points of standard deviation — real, but modest.",
        "Anyone eyeballing a model-by-domain accuracy table without this null will badly overread it. Most published tables of this kind do not report one.",
      ], NAVY),

      new Paragraph({ children: [new PageBreak()] }),

      H("5.  Result 3 — the aggregation comparison", HeadingLevel.HEADING_2),
      P("Answers embedded with all-mpnet-base-v2, cosine similarity, aggregators taken directly from kernel_agg.py. All weights are 5-fold cross-validated — weights for a question come only from folds that do not contain it. n = 970, paired bootstrap with 10,000 resamples."),
      table(
        ["Method", "Accuracy", "vs best single", "95% CI"],
        [
          ["majority vote (exact match)", "81.86", { t: "−2.37", color: RED }, "[−4.54, −0.21]"],
          ["medoid / cluster — kernel alone", "83.51", "−0.72", "[−2.78, +1.24]"],
          ["OW exact — weights alone", "83.51", "−0.72", "[−2.06, +0.62]"],
          ["KWA — supervised β (not the method)", "84.54", { t: "+0.31", color: GREY }, "[−1.55, +2.27]"],
          [{ t: "KWA — label-free β (the method)", bold: true, hi: true }, { t: "82.58", bold: true, hi: true },
           { t: "−1.65", bold: true, color: RED, hi: true }, { t: "[−3.71, +0.41]", hi: true }],
          ["best single model", "84.23", "—", ""],
          ["ceiling (any model correct)", "94.02", "+9.79", ""],
        ],
        [3260, 1500, 2100, 2500]
      ),
      new Paragraph({ spacing: { after: 160 }, children: [] }),
      P("KWA against its own two ingredients — the earlier refutation, retested:"),
      table(
        ["Comparison", "Difference", "95% CI"],
        [
          ["KWA − kernel alone", "+1.24", "[−0.52, +2.99]"],
          ["KWA − weights alone", "+1.24", "[−0.62, +3.09]"],
        ],
        [4360, 2200, 2800]
      ),

      H("What this supports", HeadingLevel.HEADING_2),
      BUL("Specialisation across labs is real: p < 0.001 against a correctly calibrated null. This is the one result that has survived every correction.", { bold: true }),
      BUL("Weighting helps: about +1.75 points over a real majority vote, when the weights are supervised."),

      H("What this does not support", HeadingLevel.HEADING_2),
      BUL("The label-free method (82.58) scores below plain majority voting (82.99) and 1.65 points below the best single model.", { bold: true }),
      BUL("The kernel adds nothing once weights are present: OW exact, with no kernel at all, beats the full method 84.74 to 84.54. Earlier work\'s finding that KWA never beats the better of its two ingredients does reproduce here."),
      BUL("Per-domain weights do not beat global weights. The specific hypothesis this experiment was built to test is unsupported, even though the specialisation is demonstrably there."),
      BUL("Anything computed from supervised weights overstates the deployable method. Read the β column before quoting any KWA number."),

      CALLOUT("The catch-22 — the real obstacle", [
        "The label-free estimator infers each model's weight from disagreement patterns, so it needs the models to differ enough for that structure to reveal who is sharp. On TriviaQA, where models were spread out, it scored +0.986 against true accuracy. On this pool — seven models within 7.3 points, clustered near 82% — it scores +0.473, p = 0.28, with four of seven pinned at the optimiser's floor.",
        "A matched pool is required for the specialisation question, and it starves the estimator. A spread pool feeds the estimator, but then the weights merely track overall capability, which earlier work already settled.",
        "The estimator works where it is not needed and fails where it is. That is a sharper problem than the aggregation result, because label-free skill estimation was our most practical standalone deliverable.",
      ], RED),

      new Paragraph({ children: [new PageBreak()] }),

      // -------------------------------------------------------- 6. caveats
      H("6.  What to trust, and what not to", HeadingLevel.HEADING_1),
      CALLOUT("Do not use analyze_domains.aggregation()", [
        "It compares answers by exact string identity of the last 120 characters and never imports the kernel. On this data 94.8% of questions have all seven answers as distinct strings, so its “majority vote” is a random pick among models and its “weighted vote” just selects the highest-weighted model.",
        "Its numbers measure model selection, not aggregation. The first draft of our results document drew the wrong conclusion from it. Use analyze_kernel.py.",
      ], RED),
      new Paragraph({ spacing: { after: 160 }, children: [] }),
      P("Cross-validation is load-bearing. Fitting per-domain weights and scoring them on the same questions inflates the result by 1.86 points — enough to flip the headline from “no effect” to “prediction confirmed”. Always report the cross-validated numbers.", { bold: true }),

      H("Limitations", HeadingLevel.HEADING_2),
      BUL("26 of 996 questions (2.6%) are excluded because at least one model never produced an answer. These are recorded as missing, not wrong — scoring them zero would fabricate a capability gap."),
      BUL("About 110 questions per domain. The interaction test is powered for effects around 3 points of standard deviation and no finer."),
      BUL("Domain labels come from the benchmark, not inferred. This measures whether specialisation is exploitable in principle, not what a deployed system achieves."),
      BUL("Kimi-K2.5 rather than K2.6 — one generation behind, because that is what got deployed."),

      // ------------------------------------------------------------ 7. bugs
      H("7.  Three bugs found during this run", HeadingLevel.HEADING_1),
      P("Recorded because each would have produced a confident, publishable-looking, wrong number — and because they share one shape."),
      table(
        ["Bug", "What it did", "Status"],
        [
          ["Empty responses written as rows", "resp=\"\" grades as incorrect, indistinguishable from a wrong answer. This is how an earlier run manufactured a fake 67-point capability spread.", "fixed"],
          ["Judge graded everything incorrect", "0 of 6,606 correct, while string matching said 65.8%. The judge was swapped to a reasoning model while max_tokens stayed at 5, so it returned no content — and empty was coerced to “incorrect”.", "fixed"],
          ["Permutation test could never reject", "It permuted rows then took a column mean, which is invariant to row order, so p was pinned at 1.0000 by construction.", "fixed"],
        ],
        [2700, 5060, 1600]
      ),
      new Paragraph({ spacing: { after: 160 }, children: [] }),
      PR([["The common thread: ", {}],
          ["an absent or unmeasured value being silently coerced into a confident one", { bold: true }],
          [". Worth treating as the default failure mode of this pipeline rather than three accidents.", {}]]),

      // -------------------------------------------------------- 8. next steps
      H("8.  Next steps", HeadingLevel.HEADING_1),
      PR([["1.  Establish the capability-spread precondition for the label-free estimator.", { bold: true }],
          [" Re-run it on a deliberately spread pool to confirm the mechanism, and state the precondition in any claim we make for it. This is now the highest-value next step, because it decides whether our most practical deliverable survives at all. Costs nothing new — the earlier TriviaQA data already exists.", {}]]),
      PR([["2.  Extend to roughly 2,500 questions.", { bold: true }],
          [" Every effect here is about a point against confidence intervals of about ±1.8. The run resumes from what already exists. Approximately $18 of OpenRouter credit.", {}]]),
      PR([["3.  Then per-question confidence weighting.", { bold: true }],
          [" A vote discards everything except each model's top answer, which is why 9.79 points sit above a method that already uses similarity and calibrated weights. At least one model answers 94.02% of questions correctly; the best single model manages 84.23%.", {}]]),

      H("Where the files are", HeadingLevel.HEADING_2),
      table(
        ["File", "What it is"],
        [
          ["RESULTS_v2.md", "Full writeup, with the correction notice"],
          ["HANDOFF.md", "Project state, method, traps"],
          ["analyze_kernel.py", "The correct aggregation comparison — use this"],
          ["analyze_domains.py", "Capability and specialisation only; its aggregation() is invalid"],
          ["data/v2.jsonl", "6,939 generations — valid"],
          ["data/v2_judged.jsonl", "6,938 judgements — valid"],
        ],
        [3000, 6360]
      ),
    ],
  }],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync("CATCHUP.docx", b);
  console.log("wrote CATCHUP.docx");
});
