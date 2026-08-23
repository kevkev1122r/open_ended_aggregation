const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, LevelFormat
} = require('docx');
const fs = require('fs');

const INK = '16181D', SOFT = '3D4350', FAINT = '8A90A0';
const ACCENT = '0F6E70', ACC_BG = 'E6F2F2', NEG = '9B2C2C';
const RULE = 'E3E6EC', ROW = 'F7F8FA';
const MONO = 'Consolas', SANS = 'Calibri';
const W = 9360;

const t = (s, o = {}) => new TextRun({
  text: s, font: o.mono ? MONO : SANS, size: o.size || 21,
  bold: o.bold, italics: o.italic, color: o.color || INK
});
const b = s => t(s, { bold: true });
const r = s => t(s);
const mo = s => t(s, { mono: true, size: 19, color: ACCENT });
const P = (runs, o = {}) => new Paragraph({
  children: Array.isArray(runs) ? runs : [runs],
  spacing: { before: o.before || 0, after: o.after ?? 130, line: 280 },
  numbering: o.num, indent: o.indent, shading: o.shading, border: o.border
});
const H1 = s => new Paragraph({ children: [t(s, { size: 32, bold: true })],
  spacing: { before: 440, after: 60 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 6 } } });
const H2 = s => new Paragraph({ children: [t(s, { size: 24, bold: true })],
  spacing: { before: 320, after: 110 } });
const H3 = s => new Paragraph({ children: [t(s, { size: 21, bold: true, color: SOFT })],
  spacing: { before: 240, after: 90 } });
const L1 = runs => P(runs, { num: { reference: 'n1', level: 0 } });
const B1 = runs => P(runs, { num: { reference: 'b1', level: 0 } });
const B2 = runs => P(runs, { num: { reference: 'b1', level: 1 } });

function cell(content, w, o = {}) {
  const runs = Array.isArray(content) ? content : [t(String(content), { size: 18 })];
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs, spacing: { before: 0, after: 0, line: 240 } })]
  });
}
function table(headers, rows, widths) {
  const cw = widths.map(x => Math.round(W * x));
  cw[cw.length - 1] += W - cw.reduce((a, c) => a + c, 0);
  return new Table({
    columnWidths: cw, width: { size: W, type: WidthType.DXA },
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) =>
        cell([new TextRun({ text: h.toUpperCase(), font: MONO, size: 14, color: FAINT })], cw[i])) }),
      ...rows.map((rw, ri) => new TableRow({ children: rw.map((c, i) =>
        cell(c, cw[i], { fill: ri % 2 ? ROW : 'FFFFFF' })) }))
    ],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: RULE },
      insideVertical: { style: BorderStyle.NONE }
    }
  });
}
const note = runs => P(runs, {
  shading: { type: ShadingType.CLEAR, fill: ACC_BG }, indent: { left: 180, right: 180 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 8 } },
  before: 160, after: 200
});

const k = [];

k.push(new Paragraph({ children: [t('Paper outline — Introduction and Related Works', { size: 40, bold: true })], spacing: { after: 120 } }));
k.push(P(t('Open-ended aggregation · content-validation draft · 23 August 2026', { size: 19, color: FAINT, mono: true }), { after: 100 }));
k.push(P(t('Content only — writing style deliberately not polished, per the brief. Every number is cross-fitted and reproduces from cached generations.', { size: 20, color: SOFT, italic: true }), { after: 260 }));

// ================= INTRODUCTION
k.push(H1('Introduction'));

k.push(H2('1. Motivation — why should the reader care?'));
k.push(B1([r('Multi-model ensembles are now standard practice, but the aggregation step is still majority voting, which was designed for single-label tasks.')]));
k.push(B1([r('Most real LLM outputs are not single labels. They are '), b('sets of claims'), r(' — entity lists, or long-form answers containing many separable facts.')]));
k.push(B1([r('Voting on sets is actively harmful. On QAMPARI, strict majority scores '), b('23.11'), r(' while simply using the best single model scores '), b('27.40'), r('. Naive aggregation is worse than not aggregating at all.')]));
k.push(B1([r('So the practical stake is real: getting this wrong means paying 8× the inference cost for a loss.')]));

k.push(H2('2. Context and the gap'));
k.push(B1([b('Existing approach 1 — weighted voting.'), r(' Beyond Majority Voting derives provably optimal aggregation weights, but for binary and categorical predictions.')]));
k.push(B1([b('Why it is not enough.'), r(' It does not merely underperform on set-valued answers — it is undefined there. Its weight is log(p/(1−p)), which requires per-model accuracy above 0.5. Per-'), t('claim', { italic: true }), r(' precision on these tasks runs 5.8–28%, so every weight goes negative and the rule inverts: more agreement produces less support.')]));
k.push(B1([b('Existing approach 2 — atomic self-consistency.'), r(' Decompose a long-form answer into atomic units, cluster them, keep clusters above a threshold, recompose.')]));
k.push(B1([b('Why it is not enough.'), r(' It draws repeated samples from a '), t('single', { italic: true }), r(' model, so it has no notion of which model asserted what. There is no reliability weighting in it at all.')]));
k.push(B1([b('Existing approach 3 — response-level selection and fusion.'), r(' Rank whole responses and pick or blend the best.')]));
k.push(B1([b('Why it is not enough.'), r(' With open-ended answers no two responses ever coincide, so selection degenerates exactly to the best single model — we measure this and it reproduces best-single with a zero-width confidence interval.')]));
k.push(note([b('The gap. '), r('No principled weighting scheme exists for heterogeneous models producing set-valued answers — the case that covers most real long-form LLM output.')]));

k.push(H2('3. Insights and research questions'));
k.push(P(b('Research questions'), { after: 90 }));
k.push(L1([b('RQ1. '), r('Can reliability weighting improve multi-model aggregation of set-valued answers?')]));
k.push(L1([b('RQ2. '), r('If not, what information does help — and can we establish where the limit is rather than merely failing to beat it?')]));
k.push(L1([b('RQ3. '), r('Can we predict, before deploying an ensemble, how much aggregation will help?')]));
k.push(P(b('Insights'), { before: 160, after: 90 }));
k.push(B1([b('A provable ceiling.'), r(' Every reliability-weighting rule is a function of the support pattern — which agents asserted a claim. A direct lookup of P(correct | exactly this subset asserted) therefore upper-bounds all of them, including correlation-aware and omission-aware variants, because the pattern also determines who stayed silent. Measured, that ceiling is '), b('+0.25, not significant'), r('. RQ1 is answered "no," by construction rather than by a failed attempt.')]));
k.push(B1([b('The ceiling binds only one class of rule.'), r(' It constrains anything reading '), t('who', { italic: true }), r(' asserted. It says nothing about an agent’s own output structure — where it ranked a claim within its own list, and what it declined to mention.')]));
k.push(B1([b('The decision rule matters as much as the score.'), r(' Gold answer counts vary from 5 to 40 per question, so a single global threshold is wrong nearly everywhere. Choosing how many claims to keep, per question, is a separate and roughly independent axis of gain.')]));

k.push(H2('4. Method summary'));
k.push(P([b('Step 1 — score each claim.'), r(' An L2-regularised logistic regression predicting P(claim correct) from:')], { after: 90 }));
k.push(B2([b('support pattern'), r(' — which agents asserted it, and how many (what counting and weighting already use)')]));
k.push(B2([b('position'), r(' — where the claim sat in each asserting agent’s '), t('own', { italic: true }), r(' list, plus per-agent position bands')]));
k.push(B2([b('omission strength'), r(' — for each '), t('silent', { italic: true }), r(' agent, how much it said. Silence from an agent that listed 50 items is evidence; from one that listed 3 it is not.')]));
k.push(B2([b('stacked pattern probability'), r(' — the ceiling estimate fed in as one more feature')]));
k.push(P([b('Step 2 — choose how many to keep, per question.')], { before: 150, after: 80 }));
k.push(P(t('k*(q) = argmax_k   2 · (sum of the top-k probabilities) / (k + Ĝ_q)', { mono: true, size: 19 }),
  { shading: { type: ShadingType.CLEAR, fill: 'F4F6F8' }, indent: { left: 200 }, after: 110 }));
k.push(B1([r('This is plug-in expected F1. Ĝ is a ridge-predicted gold count fitted on training questions only.')]));
k.push(B1([r('Notably it does not need to be accurate: mean absolute error is 4.62 against a gold mean of 10.2, and substituting the true count gains only +0.55. But the cut must '), t('vary', { italic: true }), r(' — a constant k for every question gains nothing at all.')]));
k.push(B1([r('Everything — coefficients, the gold-count regression, thresholds — is cross-fitted by question. No labels are used at inference.')]));

k.push(H2('5. Contributions and results'));
k.push(B1([b('A ceiling result. '), r('Reliability weighting on set-valued aggregation is capped at +0.25 (not significant) on QAMPARI. The cap is established by construction, which converts a null result into a positive finding.')]));
k.push(B1([b('A method that clears it. '), r('+2.39 over counting (+8.1%) and +4.34 over the best single agent (+15.8%), both significant — using only a logistic regression and a closed-form keep-count, no boosting.')]));
k.push(B1([b('Breadth. '), r('Beats counting significantly in '), b('210 of 210'), r(' agent ensembles of size 3–6; +20.0% at n=3. Reliability weighting manages 105 of 210 and turns negative by n=6, so the published crossover is a property of pattern-based rules, not of aggregation.')]));
k.push(B1([b('Robustness. '), r('Survives leave-one-source-out, retaining 80–90% of the gain on question types never seen in training. The same test collapses the pattern-lookup arm from +0.25 to +0.02 (not significant), demonstrating that the protocol detects memorisation and that our features are not what it flags.')]));
k.push(B1([b('A predictive rule. '), r('The gain is forecastable from pool properties computable before running anything: leave-one-out R² = 0.817. Dominance (best ÷ mean of the rest) flips sign between the two comparisons, which explains why our two results tables look so different.')]));
k.push(B1([b('A dataset screen. '), r('The solo ratio, P(correct | 1 agent) ÷ P(correct | ≥2 agents), predicts which regime a benchmark is in — monotone across QAMPARI 0.06, TriviaQA 0.24, ASQA 0.47, tracking weighting’s value from null to +10.9%.')]));
k.push(B1([b('Reusable negative results. '), r('Error-correlation weighting (λ = 0 selected in 5 of 5 folds), question-domain weights (+0.00), pairwise co-assertion (+0.05), and alias merging (harmful at every threshold tested).')]));

// ================= RELATED WORKS
k.push(H1('Related Works'));
k.push(P(t('Structure follows Huang’s related-work guidance: divide into topics, tell each as a trajectory, state the relationship to our work explicitly, avoid laundry lists.', { size: 19, color: SOFT, italic: true }), { after: 200 }));

k.push(H2('Paper list'));
k.push(P([t('Confidence marks: ', { size: 19, color: SOFT }), mo('[read]'), t(' = read directly. ', { size: 19, color: SOFT }),
  mo('[search]'), t(' = from search summaries only; must be read before entering the bibliography.', { size: 19, color: SOFT })], { after: 140 }));

k.push(table(['#', 'Paper', 'Main finding'], [
  ['1', [b('Beyond Majority Voting'), t('  2510.01499 [read]', { mono: true, size: 16, color: ACCENT })], 'Optimal-weight and inverse-surprising-popularity aggregation provably beat majority voting; scope is binary and categorical predictions'],
  ['2', [b('Atomic Self-Consistency'), t('  2405.13131 [read]', { mono: true, size: 16, color: ACCENT })], 'Decompose long-form answers into atomic units, cluster, threshold, recompose with an LLM; m samples from one model'],
  ['3', [b('Universal Self-Consistency'), t('  2311.17311 [search]', { mono: true, size: 16, color: ACCENT })], 'An LLM selects the most consistent whole response, extending self-consistency to free-form output'],
  ['4', [b('More Agents Is All You Need'), t('  2402.05120 [read]', { mono: true, size: 16, color: ACCENT })], 'Performance scales with the number of sampled agents under plain majority voting'],
  ['5', [b('LLM-Blender'), t('  2306.02561 [read]', { mono: true, size: 16, color: ACCENT })], 'Pairwise ranking followed by generative fusion over complete responses'],
  ['6', [b('Finch-Zk'), t('  2508.14314 [read]', { mono: true, size: 16, color: ACCENT })], 'Fine-grained cross-model consistency filters hallucinated segments; no position, omission or budget signal'],
  ['7', [b('Atomic Calibration'), t('  2410.13246 [read]', { mono: true, size: 16, color: ACCENT })], 'Confidence estimation at atomic-claim level within a single model’s long-form generation'],
  ['8', [b('Cross-Contextual Consistency'), t('  2608.10315 [read]', { mono: true, size: 16, color: ACCENT })], 'Stability under perturbed prompts predicts correctness; single-model credibility scoring'],
  ['9', [b('Serial Position Effects'), t('  2406.15981 [read]', { mono: true, size: 16, color: ACCENT })], 'Primacy and recency bias in which of the supplied options an LLM selects — an input-side effect'],
  ['10', [b('Fast Dawid-Skene'), t('  1803.02781 [search]', { mono: true, size: 16, color: ACCENT })], 'Efficient expectation-maximisation for annotator-reliability label aggregation'],
  ['11', [b('Crowd-Certain'), t('  2310.16293 [search]', { mono: true, size: 16, color: ACCENT })], 'Label aggregation spanning crowdsourcing and ensemble classification'],
  ['12', [b('Soft Dawid-Skene for deep ensembles'), t('  2503.07119 [search]', { mono: true, size: 16, color: ACCENT })], 'Weighting ensemble members by estimated confusion matrices beats plain averaging'],
  ['13', [b('The Illusion of Progress'), t('  2508.08285 [search]', { mono: true, size: 16, color: ACCENT })], 'Response length alone is a powerful hallucination detector, often matching sophisticated methods'],
  ['14', [b('Verbosity ≠ Veracity'), t('  2411.07858 [search]', { mono: true, size: 16, color: ACCENT })], 'Models pad their answers when uncertain — verbosity compensation'],
  ['15', [b('QAMPARI'), t('  2205.12665 [search]', { mono: true, size: 16, color: ACCENT })], 'Open-domain QA benchmark whose answers are lists of many entities across paragraphs'],
  ['16', [b('ASQA'), t('  2204.06092 [search]', { mono: true, size: 16, color: ACCENT })], 'Ambiguous-question long-form benchmark; headline metric is the geometric mean of ROUGE-L and Disambig-F1'],
  ['17', [b('Power and Limitations of Aggregation'), t('  2602.21556 [read]', { mono: true, size: 16, color: ACCENT })], 'Theory of when aggregating multiple model outputs expands the reachable output set'],
  ['18', [b('Collective Intelligence for Biomedical QA'), t('  2508.01480 [read]', { mono: true, size: 16, color: ACCENT })], 'Multi-LLM list-question answering by taking the plain union of predicted items'],
  ['19', [b('Adaptive Coverage Policies'), t('  2510.04318 [search]', { mono: true, size: 16, color: ACCENT })], 'Conformal prediction with per-example adaptive set size and a target expected size'],
  ['20', [b('Prompt-Reverse Inconsistency'), t('  2504.01282 [search]', { mono: true, size: 16, color: ACCENT })], 'LLMs give conflicting answers when asked which items are correct versus incorrect'],
  ['21', [b('Self-consistency under prompt transformations'), t('  2305.14279 [search]', { mono: true, size: 16, color: ACCENT })], 'Low consistency across meaning-preserving prompt rewrites'],
  ['22', [b('MAD-Fact / VeriFact'), t('  2510.22967, 2505.09701 [search]', { mono: true, size: 16, color: ACCENT })], 'Multi-agent atomic-claim pipelines for factuality evaluation rather than answer construction'],
  ['23', [b('Reciprocal Rank Fusion'), t('  Cormack et al. 2009', { mono: true, size: 16, color: ACCENT })], 'Rank fusion by summing 1/(K + rank); a famously strong parameter-free retrieval baseline']
], [0.05, 0.34, 0.61]));

k.push(H2('Three axes'));
k.push(L1([b('Aggregating outputs from multiple models')]));
k.push(L1([b('Decomposing long-form answers into atomic claims')]));
k.push(L1([b('Estimating per-item reliability without labels')]));

k.push(H2('Axis 1 — Aggregating outputs from multiple models'));
k.push(H3('Trajectory'));
k.push(B1([r('The problem is old and the naive answer is majority voting, which is optimal only under strong and rarely-stated assumptions about independent, equally-reliable voters.')]));
k.push(B1([r('The field first relaxed the equal-reliability assumption: weight each model by how often it has been right, so historically accurate models carry more influence [4].')]));
k.push(B1([r('Recent work goes further and derives '), t('optimal', { italic: true }), r(' weights from higher-order information — not just how often a model is right, but the structure of agreement between models [1]. This is the state of the art for discrete answers.')]));
k.push(B1([r('A parallel branch avoids voting entirely: rank complete responses against each other and select or generatively fuse the best [3, 5]. Theoretical work has begun to characterise when aggregation can expand what a system can produce at all [17].')]));
k.push(B1([r('For set-valued answers specifically, practice has not moved past taking the union of every model’s items [18].')]));
k.push(H3('Relationship to our work'));
k.push(B1([r('Our work shares this line’s goal — deriving weights that outperform naive voting — and takes its central open question directly, since §7 of the optimal-weight paper asks how to derive weights for open-ended questions.')]));
k.push(B1([b('The key contrast is discrete versus set-valued.'), r(' These methods assume an answer space where two models can emit the same answer. With open-ended sets that never happens at the response level, so response selection provably degenerates to the best single model, and the optimal-weight rule requires per-model accuracy above 0.5, which per-claim precision never reaches.')]));
k.push(B1([r('Unlike this line, we find that once the unit of aggregation becomes the claim rather than the response, model identity carries far less information than expected — and we quantify exactly how much.')]));

k.push(H2('Axis 2 — Decomposing long-form answers into atomic claims'));
k.push(H3('Trajectory'));
k.push(B1([r('Self-consistency works by sampling repeatedly and taking the modal answer, which requires an extractable answer to compare. That restriction motivated a version where the model itself judges which of several free-form responses is most consistent [3].')]));
k.push(B1([r('The decisive step was decomposing a long answer into atomic units so that agreement can be measured '), t('within', { italic: true }), r(' an answer rather than between whole answers: cluster the units, keep those with enough support, and recompose [2].')]));
k.push(B1([r('That decomposition has since become general infrastructure — used for factuality evaluation of long-form text [22] and for calibration at claim level inside one model [7].')]));
k.push(B1([r('Most recently it has been applied across models, filtering claims by cross-model agreement [6].')]));
k.push(H3('Relationship to our work'));
k.push(B1([r('We adopt this decomposition wholesale as our task setup; it is what makes multi-model aggregation of long-form answers tractable at all.')]));
k.push(B1([b('The key contrast is one model versus many.'), r(' Prior work draws repeated samples from a single model, so "who said it" does not exist as a variable. Ours is the first setting where the atoms come from '), t('different', { italic: true }), r(' models and identity is available.')]));
k.push(B1([r('The result is counter-intuitive and is our main negative finding: having model identity available turns out to be worth almost nothing, and we prove the bound rather than reporting a failure to exploit it.')]));

k.push(H2('Axis 3 — Estimating per-item reliability without labels'));
k.push(H3('Trajectory'));
k.push(B1([r('Long before LLMs, crowdsourcing faced the same problem — many noisy annotators, no ground truth — and solved it by estimating each annotator’s reliability from patterns of disagreement [10, 11]. That machinery has recently been carried over to deep-model ensembles [12].')]));
k.push(B1([r('In parallel, LLM research has looked for label-free correctness signals in model '), t('behaviour', { italic: true }), r(' rather than in agreement: how long the response is [13, 14], how stable it is under paraphrase or perturbation [8, 20, 21], and how output position affects what a model selects [9].')]));
k.push(B1([r('Separately, conformal prediction supplies principled machinery for varying the size of a prediction set per example, though under coverage guarantees rather than an F1 objective [19].')]));
k.push(H3('Relationship to our work'));
k.push(B1([r('Our support-pattern and Naive-Bayes arms '), t('are', { italic: true }), r(' annotator-reliability models applied to LLM agents. We present them as baselines and as the ceiling, not as contributions, and credit that literature accordingly.')]));
k.push(B1([b('The key contrast is response-level versus claim-level, and asserters versus abstainers.'), r(' Existing behavioural signals score a whole response from one model. We use output structure per claim inside a multi-model set, and extend it to the models that stayed '), t('silent', { italic: true }), r(', where no prior signal exists — a silent agent has no response to score and no position to read, so how much it said elsewhere is the only available evidence.')]));
k.push(B1([r('On position specifically, prior work measures how position in a model’s '), b('input'), r(' biases what it selects [9]. We use position in a model’s '), b('own output'), r(' as a correctness signal, which is a different quantity and, as far as our search found, unclaimed.')]));

// ---- open questions
k.push(H1('Open decisions before writing'));
k.push(B1([b('Settled 23 Aug: '), r('boosting is removed from the method. Leave-one-source-out showed the boosted blend transfers worst of any arm, so the method is the plain rank scorer plus budget. Boosting is now reported as a rejected ablation.')]));
k.push(B1([r('Lead with the ceiling result or with the method? The ceiling is the more novel framing; the method is the more conventional one.')]));
k.push(B1([r('15 of 23 references are still marked [search] and need direct reads before citation.')]));
k.push(B1([r('The resources doc was not readable (401) — this outline answers the questions as stated in the assignment template and follows the linked related-work structure. Worth confirming nothing was missed.')]));

const doc = new Document({
  creator: 'Open-ended aggregation',
  title: 'Paper outline — Introduction and Related Works',
  styles: { default: { document: { run: { font: SANS, size: 21, color: INK } } } },
  numbering: {
    config: [
      { reference: 'b1', levels: [
        { level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 360, hanging: 200 } } } },
        { level: 1, format: LevelFormat.BULLET, text: '–', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 200 } } } }] },
      { reference: 'n1', levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 360, hanging: 220 } } } }] }
    ]
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: k
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('paper-outline.docx', buf);
  console.log('wrote paper-outline.docx', buf.length, 'bytes');
});
