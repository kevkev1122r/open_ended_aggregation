const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  LevelFormat, convertInchesToTwip, ImageRun } = require("docx");

const DIR = "/Users/kevinchou/Documents/open-ended-aggregation";
const ACCENT = "1A73E8", GREY = "5F6368", RED = "B3261E";

const P = (t, o = {}) => new Paragraph({ spacing: { after: o.after ?? 120, line: 276 },
  children: [new TextRun({ text: t, size: o.size ?? 21, color: o.color, bold: o.bold, italics: o.italics })] });
const Rich = (runs, o = {}) => new Paragraph({ spacing: { after: o.after ?? 120, line: 276 },
  children: runs.map(r => new TextRun({ size: 21, ...r })) });
const H1 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 140 } });
const H2 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 100 } });
const Num = t => new Paragraph({ numbering: { reference: "num", level: 0 }, spacing: { after: 80, line: 264 },
  children: [new TextRun({ text: t, size: 21 })] });
const Bul = t => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 70, line: 264 },
  children: [new TextRun({ text: t, size: 21 })] });
const Code = t => new Paragraph({ spacing: { after: 50, before: 50 }, indent: { left: convertInchesToTwip(0.25) },
  children: [new TextRun({ text: t, size: 18, font: "Menlo", color: "202124" })] });

const TOTAL = 9360;
function mkTable(header, rows, widths, boldRow) {
  const cell = (txt, { bold = false, shade = null, align = AlignmentType.LEFT, w }) =>
    new TableCell({ width: { size: w, type: WidthType.DXA },
      shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: "auto" } : undefined,
      margins: { top: 60, bottom: 60, left: 90, right: 90 },
      children: [new Paragraph({ alignment: align, spacing: { after: 0 },
        children: [new TextRun({ text: txt, size: 19, bold })] })] });
  return new Table({ columnWidths: widths, width: { size: TOTAL, type: WidthType.DXA },
    rows: [ new TableRow({ tableHeader: true, children: header.map((h, i) =>
        cell(h, { bold: true, shade: "E8F0FE", w: widths[i], align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) =>
        cell(String(c), { w: widths[i], shade: (boldRow === ri) ? "E6F4EA" : (ri % 2 ? "F8F9FA" : null),
          bold: boldRow === ri, align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) })) ] });
}
const img = (f, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
  children: [new ImageRun({ type: "png", data: fs.readFileSync(`${DIR}/results/figures/${f}`),
    transformation: { width: w, height: h } })] });
const Cap = t => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: t, size: 17, italics: true, color: GREY })] });

const doc = new Document({
  title: "Kernel-Weighted Aggregation: Real-Data Experiment Report",
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 460, hanging: 240 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 460, hanging: 240 } } } }] }] },
  styles: { default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "202124" } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: ACCENT } }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1080, bottom: 1080, left: 1260, right: 1260 } } },
    children: [
      new Paragraph({ spacing: { after: 60 }, children: [new TextRun({
        text: "Aggregating Open-Ended LLM Answers", bold: true, size: 36, color: "202124" })] }),
      new Paragraph({ spacing: { after: 40 }, children: [new TextRun({
        text: "A real-data experiment: 6 models, 3,000 TriviaQA questions, two prompt settings", size: 22, color: GREY })] }),
      new Paragraph({ spacing: { after: 240 }, children: [new TextRun({
        text: "Result: partially negative. The method beats majority voting but never beats the better of its own two components.",
        size: 19, italics: true, color: RED })],
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "DADCE0", space: 6 } } }),

      // ---------------- 1
      H1("1. What was being tested"),
      Rich([{ text: "Ai, Pan, Simchi-Levi, Tambe & Xu (ICML 2026), " },
            { text: "Beyond Majority Voting", italics: true },
            { text: " (arXiv:2510.01499), derive a Bayes-optimal way to combine several LLMs' answers on multiple-choice questions, and close by asking how to extend it to open-ended ones. Their method needs a fixed option set of size K; free-form generation has no K." }]),
      P("This project proposed replacing their exact-match test with a similarity kernel: score each candidate answer s by the weighted similarity of every model's answer to it."),
      Code("KWA:   pick s maximising   sum_j  beta_j * sim(answer_j, s)"),
      P("beta_j is how sharply model j concentrates near the truth, estimated with no labels by EM. Setting sim to exact match recovers their method exactly (verified numerically at 1.0000 for K = 2, 3, 4, 6, 10), so this is a strict generalisation rather than a competitor."),
      P("Prior work in this project was synthetic. This report covers the first test on real multi-model generations."),

      // ---------------- 2
      H1("2. Setup"),
      mkTable(["", ""], [
        ["Benchmark", "TriviaQA rc.nocontext, 3,000 questions sampled from validation"],
        ["Why this one", "genuinely free-form (no options offered), and ships official answer alias lists so grading is automatic string matching — no judge model, no judge cost, no judge noise"],
        ["Models", "llama-3.3-70b · gpt-4o-mini · phi-4 · gemma-3-12b · llama-3.1-8b · llama-3.2-3b"],
        ["Why these", "a deliberate skill spread (the theory says aggregation only helps when models differ), four families, three of them Llamas so same-family correlation is measurable"],
        ["Conditions", "short-answer (“just the answer”) and full-sentence (“answer the question”)"],
        ["Decoding", "temperature 0 — each model's characteristic answer, not sampling noise"],
        ["Scale / cost", "36,000 generations, $0.32 total, zero API errors"],
        ["Scored", "n = 2,757 questions with a complete response set from all 6 models"]],
        [1900, 7460]),
      P("", { after: 100 }),
      P("Two prompt settings were run as a controlled test of the proposed mechanism. The short-answer prompt pushes models onto the same string (2.6 distinct answers per question); the full-sentence prompt lets them phrase freely (5.3 distinct, with all six matching on only 0.3% of questions). Since the kernel's claimed advantage is pooling surface variants, its benefit should be larger under full-sentence answers. That prediction was made before the data was seen."),
      P("Measured single-model accuracy on the scored set:"),
      mkTable(["model", "short-answer", "full-sentence"], [
        ["meta-llama/llama-3.3-70b-instruct", "86.3%", "90.2%"],
        ["openai/gpt-4o-mini", "81.9%", "85.4%"],
        ["microsoft/phi-4", "65.7%", "76.9%"],
        ["google/gemma-3-12b-it", "71.6%", "76.3%"],
        ["meta-llama/llama-3.1-8b-instruct", "68.6%", "74.5%"],
        ["meta-llama/llama-3.2-3b-instruct", "56.8%", "55.8%"]],
        [5360, 2000, 2000]),

      // ---------------- 3
      H1("3. Headline results"),
      P("All methods are label-free and deployable. Significance is McNemar's exact paired test against majority voting on the same 2,757 questions."),
      H2("3.1  Short-answer prompt"),
      mkTable(["method", "accuracy", "vs MV", "p"], [
        ["Random (floor)", "62.21", "−20.82", "1.6e-114"],
        ["MV-exact", "83.03", "—", "—"],
        ["MV-cluster (Universal Self-Consistency)", "82.81", "−0.22", "0.561"],
        ["Medoid (kernel, no weights)", "82.88", "−0.15", "0.734"],
        ["KWA-EM (ours)", "86.25", "+3.23", "4.1e-11"]],
        [4360, 1700, 1600, 1700], 4),
      H2("3.2  Full-sentence prompt"),
      mkTable(["method", "accuracy", "vs MV", "p"], [
        ["Random (floor)", "75.70", "−3.19", "4.6e-05"],
        ["MV-exact", "78.89", "—", "—"],
        ["MV-cluster (Universal Self-Consistency)", "83.64", "+4.75", "2.2e-13"],
        ["Medoid (kernel, no weights)", "86.18", "+7.29", "8.1e-31"],
        ["KWA-EM (ours)", "85.42", "+6.53", "3.8e-22"]],
        [4360, 1700, 1600, 1700], 3),
      img("X1_real_results.png", 470, 244),
      Cap("Figure 1 — Accuracy by method under the short-answer prompt. Black outline marks the best method. Full-sentence results are in Table 3.2."),
      Rich([{ text: "The proposed method beats majority voting decisively in both conditions", bold: true },
            { text: " (+3.23 and +6.53, both p < 1e-10). But under the full-sentence prompt it is beaten by the medoid — the same similarity kernel with the weighting removed." }]),
      P("Two incidental findings worth recording. Universal Self-Consistency, the published standard for open-ended aggregation, fails to beat plain majority voting on short answers (−0.22, p = 0.56) and is beaten by the far simpler medoid on full sentences. And exact-match voting nearly collapses to chance on full sentences — 78.89 against a 75.70 random floor — because with 5.3 distinct answers per question there is rarely a majority to count."),

      // ---------------- 4
      H1("4. The ablation, which is the real finding"),
      P("The method has two separable ingredients: a similarity kernel, and label-free per-model weights. The medoid isolates the first (kernel with all weights equal to 1), so the difference between the three methods decomposes cleanly."),
      mkTable(["contribution", "short-answer", "p", "full-sentence", "p"], [
        ["similarity kernel alone   (MV → Medoid)", "−0.15", "0.73", "+7.29", "8e-31"],
        ["adding learned weights  (Medoid → KWA)", "+3.37", "1e-11", "−0.76", "0.13"],
        ["total   (MV → KWA)", "+3.23", "4e-11", "+6.53", "4e-22"]],
        [3960, 1350, 1350, 1350, 1350]),
      P("", { after: 100 }),
      img("X2_ingredients.png", 470, 235),
      Cap("Figure 2 — Each ingredient carries the method in one setting and contributes nothing in the other."),
      Rich([{ text: "The two halves work in exactly opposite regimes. ", bold: true },
            { text: "On short answers the kernel contributes nothing (−0.15, p = 0.73) and all of the gain comes from the weights. On full sentences the kernel contributes everything (+7.29) and the weights then give 0.76 points back. In neither setting does the combination beat the better single ingredient, so the central claim of this project — that weighted similarity beats both plain voting and plain similarity — is not supported." }]),
      Rich([{ text: "The stated mechanism prediction was also wrong. ", bold: true },
            { text: "The kernel's advantage did grow on full-sentence answers, as predicted, but the weighting collapsed there, which was not predicted and cancels the benefit." }]),

      // ---------------- 5
      H1("5. Why the weighting fails on full-sentence answers"),
      P("Two measured causes, both specific and at least partly fixable."),
      Rich([{ text: "Cause 1 — the estimator starves. ", bold: true },
            { text: "The EM recovers beta from inter-model agreement. Mean pairwise exact agreement is 0.564 on short answers but only 0.068 on full sentences: models almost never produce identical strings, so there is nothing to estimate from. Three of six betas pinned at the optimiser floor, including llama-3.3-70b — the strongest model, assigned essentially zero weight." }]),
      mkTable(["", "short-answer", "full-sentence"], [
        ["mean pairwise exact agreement", "0.564", "0.068"],
        ["Spearman(beta estimate, true accuracy)", "+0.986", "+0.395"]],
        [5360, 2000, 2000]),
      P("", { after: 100 }),
      Rich([{ text: "Cause 2 — boilerplate swamps the embedding. ", bold: true },
            { text: "Full-sentence responses average 24.4 words and share scaffolding (“The answer is X”, “X was the one who…”). The sentence embedding measures that shared format as much as the entity, compressing all pairwise similarities upward and leaving the kernel less than half its discriminating range." }]),
      mkTable(["", "short-answer", "full-sentence"], [
        ["mean answer length", "2.1 words", "24.4 words"],
        ["mean similarity between any two candidates", "0.474", "0.777"],
        ["similarity spread available to the kernel", "0.526", "0.223"]],
        [5360, 2000, 2000]),
      P("", { after: 100 }),
      P("Cause 2 suggests a concrete fix that has not yet been tested: compute similarity on an extracted answer span rather than the full response. That is a preprocessing change, not a change to the method."),
      Rich([{ text: "One genuine positive: ", bold: true },
            { text: "where agreement exists, label-free skill estimation works well. On short answers the estimator ranks the six models at Spearman +0.986 against their true accuracies, having never seen a correct answer. It also correctly identified the single best model in both conditions. That capability stands independently of the aggregation result." }]),

      // ---------------- 6
      H1("6. A structural ceiling that binds every method"),
      P("Beyond the method's own problems, the benchmark caps how much any aggregator can win. The assumption underneath this entire literature is that models err independently given the truth. They do not:"),
      mkTable(["P(all 6 models wrong)", "observed", "if independent", "ratio"], [
        ["short-answer", "7.04%", "0.033%", "215x"],
        ["full-sentence", "5.48%", "0.009%", "621x"]],
        [3360, 2000, 2000, 2000]),
      P("", { after: 100 }),
      P("Errors are correlated by two to three orders of magnitude — the models fail together on hard questions and succeed together on easy ones. Six models behave like far fewer independent voters. The consequence is arithmetic:"),
      Code("all 6 right   42.6%   -> every method wins, aggregation irrelevant"),
      Code("all 6 wrong    7.0%   -> no method can win, the answer is not in the pool"),
      Code("mixed         50.3%   -> the only battleground"),
      Code(""),
      Code("on the battleground the best single model is already right 86.7% of the time"),
      Code("TOTAL headroom for ANY aggregator:  6.67 points (short-answer), 4.32 points (full-sentence)"),
      P("And on the questions that are actually winnable, the correct answer is held by a mean of 1.88 models out of 6 — a minority. Winning them requires systematically overruling a 4–2 or 5–1 majority, which is precisely what vote-based methods are built not to do, and which no label-free weight estimate is confident enough to justify."),
      Rich([{ text: "This applies to the published methods too, not just to this one, ", bold: true },
            { text: "and it explains the paper's own MMLU result, where their strongest single model (91.02%) beat every method they proposed (90.37%)." }]),

      // ---------------- 7
      H1("7. Threats to validity"),
      Num("Grading is generous. An answer counts if any alias appears as a substring, so “not Paris, but London” scores a hit on London. Needed for the full-sentence setting, where answers are embedded in sentences, but not audited."),
      Num("List questions are graded wrongly. TriviaQA contains items such as “name the 9 countries bordering Germany”. The gold answer is a 9-item list and the matcher looks for one contiguous string, so these can never be scored correct. The output cap (16 tokens short-answer, 64 full-sentence) also truncates them mid-list. The affected fraction has not been measured; it adds noise to every method equally but the exact numbers should not be trusted until it is."),
      Num("One benchmark, one domain. TriviaQA factoid answers are close to binary right/wrong, which gives a similarity kernel little graded middle ground to exploit. A task with genuinely partial credit would be a fairer test of the kernel and has not been run."),
      Num("One embedding model (all-MiniLM-L6-v2). Absolute similarity values are encoder-specific."),
      Num("Six models from four families, all instruction-tuned and broadly contemporary. A more diverse ensemble would have less correlated errors and more headroom."),

      // ---------------- 8
      H1("8. Conclusions"),
      Rich([{ text: "Supported: ", bold: true, color: "0B8043" },
            { text: "similarity-aware selection substantially beats exact-match voting on open-ended answers — +7.3 points when answers are verbose, at p < 1e-30, on 2,757 real questions. That is a real and useful result, and it argues that the field's current default (cluster-then-vote) is leaving points on the table." }]),
      Rich([{ text: "Not supported: ", bold: true, color: RED },
            { text: "the specific contribution of this project. Weighted similarity never beat the better of its two ingredients. On full-sentence answers the unweighted medoid — three lines of code, no estimation — outperforms it." }]),
      Rich([{ text: "Independently useful: ", bold: true },
            { text: "label-free skill estimation works where inter-model agreement exists (Spearman +0.986) and correctly identified the best model without labels in both conditions. That is arguably a more practical deliverable than the aggregation itself." }]),
      H2("Next steps, in priority order"),
      Num("Test the extraction fix. Compute similarity on an extracted answer span rather than the whole response, and re-run the full-sentence setting. This directly tests whether the weighting failure is a preprocessing bug or a real limitation, and it is free."),
      Num("Beat the medoid or drop the weighting. The medoid is now the bar. If weights cannot beat it after the extraction fix, the honest paper is about similarity-aware selection, not about weighted similarity."),
      Num("Report the correlated-error ceiling. The finding that errors correlate 215–621x more than assumed, and that this caps all aggregation at 4–7 points, is a result about the field rather than about any one method, and is worth writing up on its own."),
      Num("Fix the grading harness before quoting exact numbers — audit list questions and substring false positives."),
      Num("Test on a task with graded correctness, where a similarity kernel has room to work."),

      new Paragraph({ spacing: { before: 240, after: 120 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "DADCE0", space: 6 } },
        children: [new TextRun({ text: "", size: 2 })] }),
      Rich([{ text: "Reproduction: ", color: GREY, size: 18 },
            { text: "generate.py", font: "Menlo", size: 18 }, { text: " (generation, resumable), ", color: GREY, size: 18 },
            { text: "analyze_real.py", font: "Menlo", size: 18 }, { text: " (aggregators and scoring), ", color: GREY, size: 18 },
            { text: "kernel_agg.py", font: "Menlo", size: 18 }, { text: " (library), ", color: GREY, size: 18 },
            { text: "data/gen_full.jsonl", font: "Menlo", size: 18 }, { text: " (all 36,000 raw responses), ", color: GREY, size: 18 },
            { text: "FINDINGS.md", font: "Menlo", size: 18 }, { text: " (synthetic and proxy work preceding this).", color: GREY, size: 18 }]),
    ] }] });

Packer.toBuffer(doc).then(b => { fs.writeFileSync(`${DIR}/REPORT.docx`, b); console.log("wrote REPORT.docx", b.length, "bytes"); });
