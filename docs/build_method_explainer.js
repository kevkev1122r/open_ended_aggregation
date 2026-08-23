const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  LevelFormat, PageBreak, convertInchesToTwip
} = require('docx');
const fs = require('fs');

// ---- palette (matches the artifact) ---------------------------------------
const INK = '16181D', SOFT = '3D4350', MUTED = '6B7280', FAINT = '8A90A0';
const ACCENT = '0F6E70', ACC_BG = 'E6F2F2';
const POS = '1B6B3A', NEG = '9B2C2C', NS = '8A90A0';
const RULE = 'E3E6EC', ROW_BG = 'F7F8FA';
const MONO = 'Consolas', SANS = 'Calibri';

const W = 9360;                       // US Letter, 1in margins
const pct = a => a.map(x => Math.round(W * x));

// ---- helpers ---------------------------------------------------------------
const t = (text, o = {}) => new TextRun({
  text, font: o.mono ? MONO : SANS, size: o.size || 20,
  bold: o.bold, italics: o.italic, color: o.color || INK,
  allCaps: o.caps, characterSpacing: o.caps ? 10 : undefined
});

const P = (runs, o = {}) => new Paragraph({
  children: Array.isArray(runs) ? runs : [runs],
  spacing: { before: o.before ?? 0, after: o.after ?? 140, line: o.line || 276 },
  alignment: o.align, indent: o.indent,
  border: o.border, shading: o.shading, keepNext: o.keepNext
});

const body = (text, o = {}) => P(t(text, o), o);

const H = (text, lvl) => new Paragraph({
  text, heading: lvl,
  spacing: { before: lvl === HeadingLevel.HEADING_1 ? 400 : 300, after: 140 }
});

const sectionHead = (num, text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 420, after: 160 },
  children: [
    new TextRun({ text: num + '  ', font: MONO, size: 22, color: ACCENT, bold: false }),
    new TextRun({ text, font: SANS, size: 30, bold: true, color: INK })
  ]
});

const lede = text => P(t(text, { size: 21, color: SOFT }), { after: 200 });

const code = lines => lines.map((l, i) => new Paragraph({
  children: [new TextRun({ text: l || ' ', font: MONO, size: 18, color: INK })],
  spacing: { before: i === 0 ? 100 : 0, after: i === lines.length - 1 ? 180 : 0, line: 260 },
  shading: { type: ShadingType.CLEAR, fill: 'F4F6F8' },
  indent: { left: 200, right: 200 }
}));

const callout = (tag, runs, warn) => [
  new Paragraph({
    children: [new TextRun({ text: tag.toUpperCase(), font: MONO, size: 15,
      color: warn ? NEG : ACCENT, characterSpacing: 20 })],
    spacing: { before: 200, after: 60 },
    shading: { type: ShadingType.CLEAR, fill: warn ? 'F9EFEF' : ACC_BG },
    indent: { left: 200, right: 200 },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: warn ? NEG : ACCENT, space: 8 } }
  }),
  new Paragraph({
    children: runs,
    spacing: { after: 200, line: 276 },
    shading: { type: ShadingType.CLEAR, fill: warn ? 'F9EFEF' : ACC_BG },
    indent: { left: 200, right: 200 },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: warn ? NEG : ACCENT, space: 8 } }
  })
];

const bullets = items => items.map(runs => new Paragraph({
  children: Array.isArray(runs) ? runs : [t(runs)],
  numbering: { reference: 'dash', level: 0 },
  spacing: { after: 90, line: 276 }
}));

const numbered = items => items.map(runs => new Paragraph({
  children: Array.isArray(runs) ? runs : [t(runs)],
  numbering: { reference: 'nums', level: 0 },
  spacing: { after: 90, line: 276 }
}));

// cell content may be a string or an array of TextRuns
function cell(content, w, o = {}) {
  const runs = Array.isArray(content) ? content
    : [t(String(content), { size: 18, bold: o.bold, mono: o.mono, color: o.color })];
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({
      children: runs,
      alignment: o.right ? AlignmentType.RIGHT : AlignmentType.LEFT,
      spacing: { before: 0, after: 0, line: 250 }
    })]
  });
}

function table(headers, rows, widths, opts = {}) {
  if (widths.length !== headers.length)
    throw new Error(`table "${headers[0]}": ${headers.length} headers but ${widths.length} widths`);
  rows.forEach((r, i) => { if (r.length !== headers.length)
    throw new Error(`table "${headers[0]}" row ${i}: ${r.length} cells, expected ${headers.length}`); });
  const cw = pct(widths);
  cw[cw.length - 1] += W - cw.reduce((a, b) => a + b, 0);
  const numCols = headers.length;
  const rightFrom = opts.rightFrom ?? 1;

  const head = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => cell(
      [new TextRun({ text: String(h).toUpperCase(), font: MONO, size: 14,
        color: FAINT, characterSpacing: 14 })],
      cw[i], { fill: 'FFFFFF', right: i >= rightFrom }))
  });

  const trs = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => cell(c, cw[i], {
      right: i >= rightFrom,
      fill: (opts.hi || []).includes(ri) ? ACC_BG : (ri % 2 ? ROW_BG : 'FFFFFF'),
      bold: (opts.hi || []).includes(ri) && i === 0
    }))
  }));

  return new Table({
    columnWidths: cw,
    width: { size: W, type: WidthType.DXA },
    rows: [head, ...trs],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: RULE },
      insideVertical: { style: BorderStyle.NONE }
    }
  });
}

const capt = text => P(t(text, { size: 16, color: FAINT, italic: true }), { after: 200 });
const spacer = () => P(t(''), { after: 60 });

// shorthand run builders for mixed-format cells
const m = s => new TextRun({ text: s, font: MONO, size: 17, color: ACCENT });
const n = (s, c) => new TextRun({ text: s, font: MONO, size: 18, color: c || INK, bold: !!c });
const b = s => new TextRun({ text: s, font: SANS, size: 18, bold: true });
const r = s => new TextRun({ text: s, font: SANS, size: 18 });
const sig = s => n(s, POS), bad = s => n(s, NEG), nsr = s => n(s, NS);

// ============================================================ CONTENT
const kids = [];

// ---- title block
kids.push(new Paragraph({
  children: [new TextRun({ text: 'OPEN-ENDED AGGREGATION  ·  INTERNAL EXPLAINER',
    font: MONO, size: 16, color: ACCENT, characterSpacing: 30 })],
  spacing: { after: 200 }
}));
kids.push(new Paragraph({
  children: [new TextRun({ text: 'Reading how an agent answered, not just which agent answered',
    font: SANS, size: 46, bold: true, color: INK })],
  spacing: { after: 220, line: 460 }
}));
kids.push(P(t('Every reliability-weighting scheme we can write down is capped at +0.25 F1 on QAMPARI, and we can prove the cap. Conditioning instead on the structure of each agent’s own output — where it placed a claim in its own list, and what it stayed silent about — clears that cap and beats counting in 210 of 210 agent ensembles.',
  { size: 23, color: SOFT }), { after: 240, line: 330 }));
kids.push(new Paragraph({
  children: [new TextRun({
    text: 'Benchmarks: QAMPARI · ASQA · TriviaQA      Protocol: cross-fitted, bootstrap confidence intervals      Cost: cached generations, no API calls      Updated: 22 August 2026',
    font: MONO, size: 16, color: FAINT })],
  spacing: { after: 260 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 10 } }
}));

// ============ 1
kids.push(sectionHead('1', 'The result'));
kids.push(lede('QAMPARI, 777 questions, 8 agents, 5-fold cross-fitted over questions. All 110,801 candidate claims. The metric is set F1.'));
kids.push(table(
  ['Arm', 'F1', 'vs best single', 'vs multi-agent count'],
  [
    ['best single agent (Kimi-K2.5)', '27.40', '—', [bad('−1.95')]],
    [[r('Atomic Self-Consistency '), m('ASC')], '28.60', '+1.20', [bad('−0.75')]],
    [[r('multi-agent count '), m('count')], '29.35', [sig('+1.95 *')], '—'],
    [[r('support-pattern lookup '), m('pattern')], '29.60', [sig('+2.20 *')], [nsr('+0.25 n.s.')]],
    [[b('rank scorer + budget')], '31.74', [sig('+4.34 *')], [sig('+2.39 *')]],
    ['oracle selection (ceiling, not a method)', '59.34', '+31.94', '+29.99']
  ], [0.40, 0.14, 0.22, 0.24], { hi: [4] }));
kids.push(capt('* = 95% paired-bootstrap confidence interval over questions excludes zero. The method is +15.8% relative over the best single agent and +8.1% over counting. A gradient-boosted variant scores marginally higher in-distribution but transfers worse and was rejected — see section 8.'));
kids.push(P([r('Stable across three independent fold assignments: '), b('31.74 / 31.63 / 31.63'),
  r('. A 0.11 spread against confidence-interval half-widths near 1.0 — not seed luck.')]));

// ============ 2
kids.push(sectionHead('2', 'The task, precisely'));
kids.push(P([r('A QAMPARI question — '), new TextRun({ text: '“which manga did Ryoichi Ikegami illustrate?”', italics: true, font: SANS, size: 20 }),
  r(' — has many correct answers (5 to 40, mean 10.2). Each of 8 agents returns an '), b('ordered list'), r(' of entity names.')]));
kids.push(P([r('Pool every distinct normalised string across all agents. That is the '), b('candidate set'),
  r(' — about 143 claims per question. Our job is to choose a subset of it, scored by set F1 against gold.')]));
kids.push(...code([
  'candidate claim  →  who asserted it, and where in each of their lists',
  '',
  '"Crying Freeman"  →  grok@1, Kimi@3, MAI@7      (5 agents silent)',
  '"Sanctuary"       →  Kimi@41                    (7 agents silent)'
]));
kids.push(P([r('Only '), b('3.5%'), r(' of candidate claims are correct. The whole problem is separating that 3.5% from the rest.')]));

// ============ 3
kids.push(sectionHead('3', 'Three classes of rule'));
kids.push(lede('This is the single most useful idea in the project. Every method we have tested falls into one of three classes, defined by what information it is allowed to read. The classes explain all the results.'));
kids.push(table(
  ['Class', 'Reads', 'Members'],
  [
    ['Class A', 'The support count only — how many agents asserted the claim, nothing about which ones.',
      [r('multi-agent count, Atomic Self-Consistency, union, strict majority vote')]],
    ['Class B', 'Which agents asserted — the support pattern. Every reliability-weighting scheme lives here, including correlation-aware and omission-aware ones.',
      [r('marginal, solo, Naive Bayes variants, correlation-aware, support-pattern lookup')]],
    [[b('Class C (ours)')], 'The agents’ output structure — where the claim sat in each asserter’s own list, how much each silent agent said, and how many claims to keep on this question.',
      [r('rank scorer, omission strength, per-question budget')]]
  ], [0.14, 0.50, 0.36], { rightFrom: 99, hi: [2] }));
kids.push(P([b('The support pattern'), r(' is the object that separates Class A from Class B. For a claim it is '),
  new TextRun({ text: 'which', italics: true, font: SANS, size: 20 }), r(' agents asserted it, not merely how many:')]));
kids.push(...code([
  'support count   = 3                      ← Class A sees this',
  'support pattern = {grok, Kimi, MAI}      ← Class B sees this'
]));
kids.push(P([r('With 8 agents there are 2⁸−1 = 255 patterns but only 8 counts. The distinction matters: two-agent patterns range from '),
  b('1.9%'), r(' correct (MAI + gpt-5.4-nano) to '), b('22.5%'), r(' (MAI + grok) — a 12× spread at identical support count. Counting genuinely discards information.')]));

// ============ 4
kids.push(sectionHead('4', 'Arm glossary'));
kids.push(lede('Every arm named anywhere in our results, what it computes, and what it scored. Grouped by class. All numbers are QAMPARI, 8 agents, 5-fold cross-fitted.'));

kids.push(H('Class A — count only', HeadingLevel.HEADING_2));
kids.push(table(['Arm', 'What it does', 'F1', 'vs count'],
  [
    [[b('multi-agent count'), m('  count')], 'Keep a claim if at least θ agents assert it, θ cross-fitted. Exact string match, one vote per agent. Our proposed naive multi-agent method.', '29.35', '—'],
    [[b('Atomic Self-Consistency'), m('  ASC')], 'The published pipeline (arXiv:2405.13131). Edit-distance clustering at a tuned τ, strength = count of cluster members (not deduplicated per agent), θ tuned on a validation split for F1-at-recall-5.', '28.60', [bad('−0.75')]],
    [[b('union')], 'Keep every candidate claim. θ = 1.', '21.21', [bad('−8.14')]],
    [[b('strict majority vote'), m('  MV')], 'Keep only if more than half the agents assert it.', '23.11', [bad('−6.24')]]
  ], [0.22, 0.52, 0.12, 0.14], { rightFrom: 2 }));

kids.push(H('Class B — reads who asserted (all reliability weighting)', HeadingLevel.HEADING_2));
kids.push(table(['Arm', 'What it computes', 'F1', 'vs count'],
  [
    [[b('marginal reliability'), m('  marginal')], 'Weight w = P(correct | agent asserts); score = sum of weights over asserters. The scheme deployed before this work.', '28.11', [bad('−1.25 *')]],
    [[b('solo reliability'), m('  solo')], 'w = P(correct | agent asserts AND nobody else does). The right conditional, since the only decision weighting changes is whether to admit a solo claim.', '29.44', [nsr('+0.09 n.s.')]],
    [[b('Naive Bayes, assertions only'), m('  nb-assert')], 'Log-likelihood ratio over assertions. Silence contributes nothing.', '28.40', [bad('−0.95 *')]],
    [[b('Naive Bayes, assertions and silence'), m('  nb-full')], 'Same, but every silent agent contributes too. This is a Dawid–Skene annotator model in all but name.', '29.45', [nsr('+0.10 n.s.')]],
    [[b('correlation-aware weighting'), m('  corr')], 'Penalises agents that make the same mistakes together. λ = 0 was selected in all 5 folds.', '29.44', [nsr('+0.09 n.s.')]],
    [[b('support-pattern lookup'), m('  pattern')], 'Direct lookup of P(correct | exactly this subset asserted), shrunk toward the count-level rate. Upper-bounds every arm in this class.', '29.60', [nsr('+0.25 n.s.')]],
    [[b('Reciprocal Rank Fusion'), m('  RRF')], 'A standard information-retrieval fusion rule. An off-the-shelf baseline a reviewer will ask about.', '30.19', [sig('+0.84 *')]],
    [[b('Borda count, CombSUM')], 'Other classic information-retrieval fusion rules. Both score below plain counting.', '28.9', [bad('−0.4')]]
  ], [0.24, 0.50, 0.11, 0.15], { rightFrom: 2, hi: [5] }));
kids.push(capt('Reciprocal Rank Fusion uses position, so it is strictly a Class C member — listed here because it is a standard baseline rather than something we designed.'));

kids.push(H('Class C — reads output structure (ours)', HeadingLevel.HEADING_2));
kids.push(table(['Arm', 'What it adds', 'F1', 'vs count'],
  [
    [[b('logistic regression + rank'), m('  lr+rank')], 'Logistic regression on the support pattern plus the claim’s position in each asserting agent’s own list.', '30.68', [sig('+1.33 *')]],
    [[b('Naive Bayes, rank-conditioned'), m('  nb-rank')], 'Likelihoods conditioned on which position band the claim fell in, not merely on whether the agent asserted it.', '30.69', [sig('+1.34 *')]],
    [[b('+ omission strength')], 'Each silent agent’s evidence weighted by how much it said. Largest single feature block.', '31.01', [sig('+1.65 *')]],
    [[b('rank scorer'), m('  LEAN')], 'The assembled scorer: support pattern + position + per-agent position bands + omission strength + the support-pattern probability stacked in as a feature.', '30.98', [sig('+1.63 *')]],
    [[b('rank scorer + budget')], 'Adds the per-question keep-count. This is the method.', '31.74', [sig('+2.39 *')]],
    [[b('+ boosting  — REJECTED')], 'Blends the rank scorer with a gradient-boosted tree model over the same features. Marginally better in-distribution but transfers worst of any arm under shift, so it is not part of the method — see section 8.', '32.01', [sig('+2.66 *')]]
  ], [0.24, 0.50, 0.11, 0.15], { rightFrom: 2, hi: [4] }));
kids.push(...callout('Naming', [
  r('The code calls the scorer '), new TextRun({ text: 'LEAN', font: MONO, size: 19 }),
  r('. It is not an acronym — it is the '), new TextRun({ text: 'lean', italics: true, font: SANS, size: 20 }),
  r(' feature subset, chosen because the all-features model with 115 features scored 30.60, '),
  new TextRun({ text: 'below', italics: true, font: SANS, size: 20 }), r(' several of its own ablations. Fewer features, better result.')]));

// ============ 5
kids.push(sectionHead('5', 'Why the support-pattern lookup is a ceiling'));
kids.push(lede('This is the argument the paper turns on, and it is short.'));
kids.push(P([r('Every reliability-weighting rule is a '), b('function of the support pattern'),
  r('. A weighted sum, a Naive Bayes score, a correlation penalty — each takes “which agents asserted this claim” as input and returns a number.')]));
kids.push(P([r('The support-pattern lookup skips the function. It looks up P(correct | exactly this subset asserted) directly from training data. So it already knows everything any such rule could know, and none of them can beat it.')]));
kids.push(...callout('The part people miss', [
  r('This includes '), b('omission-aware'), r(' rules. Knowing exactly who asserted a claim also tells you exactly who stayed silent — the pattern determines both. So Naive Bayes with silence cannot escape it either.')]));
kids.push(P([r('Measured, the lookup is '), b('+0.25, not significant'), r('. That is not “our weighting scheme was bad.” It is: '),
  new TextRun({ text: 'no weighting scheme can do better than +0.25 here.', italics: true, font: SANS, size: 20 }),
  r(' Which is why the only way up was to leave Class B entirely.')]));
kids.push(H('The reason the ceiling is so low', HeadingLevel.HEADING_2));
kids.push(body('Counting is already an excellent filter on QAMPARI, because a solo claim is almost never right:'));
kids.push(table(['Agents asserting', '1', '2', '3', '4', '5', '6', '7', '8'],
  [
    ['share of claims', '87.1%', '6.2%', '3.1%', '1.5%', '0.9%', '0.6%', '0.4%', '0.2%'],
    ['P(correct)', [bad('1.2%')], '11.1%', '17.4%', '27.2%', '36.5%', '46.3%', '51.9%', [sig('56.5%')]]
  ], [0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]));
kids.push(P([r('A solo claim is '), b('16× less likely'), r(' to be right than a supported one. Discard all solos and you have captured nearly everything the support pattern has to offer. There is almost nothing left for weighting to decide.')]));

// ============ 6
kids.push(sectionHead('6', 'Our method, step by step'));
kids.push(lede('Two changes to counting: one to how you score a claim, one to how many you keep.'));
kids.push(H('Step 1 — score each claim', HeadingLevel.HEADING_2));
kids.push(body('An L2-regularised logistic regression predicting P(claim correct) from four feature groups:'));
kids.push(...code([
  'support pattern     which agents asserted, and how many',
  '                    ← what counting and weighting use',
  'position            where the claim sat in each asserter’s OWN list',
  'position bands      per-agent, log-spaced — each agent gets its own profile',
  'omission strength   for each SILENT agent, log(how many items it listed)',
  'stacked             the support-pattern probability, as one more feature'
]));
kids.push(H('Why position works', HeadingLevel.HEADING_3));
kids.push(body('Precision falls monotonically with position in the agent’s own list:'));
kids.push(table(['Position in list', '1', '2', '3', '5', '9', '11', '16', '21', '26'],
  [['P(correct)', [sig('29.2%')], '31.4%', '30.4%', '26.6%', '17.3%', '15.8%', '11.8%', '8.9%', [bad('6.3%')]]],
  [0.19, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]));
kids.push(body('Agents put their confident answers first and pad the tail. None of this is visible in the support pattern, so it is outside the ceiling.'));
kids.push(H('Why omission strength works', HeadingLevel.HEADING_3));
kids.push(P([r('Every prior rule treats silence as one thing. But our agents differ enormously in how much they say — median list lengths run from '),
  b('1'), r(' (gpt-5.4-nano) to '), b('13'), r(' (Kimi), with maxima of 1,067 and 2,431 respectively.')]));
kids.push(...callout('The intuition', [
  r('An agent that listed 3 items and did not mention X has barely spoken. An agent that listed 50 items and did not mention X has said a great deal. Weighting each silent agent’s evidence by how much it actually said was the '),
  b('largest single feature win, +1.65'), r('.')]));
kids.push(H('Step 2 — choose how many to keep, per question', HeadingLevel.HEADING_2));
kids.push(body('Sort the question’s candidates by predicted probability and keep the top k:'));
kids.push(...code([
  'k*(q) = argmax_k   2 · (sum of the top-k probabilities) / (k + Ĝ)',
  '',
  'plug-in expected F1:  the numerator estimates hits, k is what you kept,',
  'and Ĝ estimates how many gold answers exist for this question'
]));
kids.push(body('Gold counts vary 5–40 per question, so the optimal cut is genuinely question-dependent and no single global threshold can express it.'));
kids.push(H('How the gold count is estimated — and why it barely matters', HeadingLevel.HEADING_3));
kids.push(body('It is a ridge regression fitted on training questions only, using features available at inference: number of candidates, maximum support count, how many candidates have at least 2 supporters, and the sum of predicted probabilities.'));
kids.push(table(['Check', 'Result', 'Reading'],
  [
    ['accuracy of the estimate', 'mean absolute error 4.62', 'poor — the gold mean is 10.2'],
    ['substitute the true gold count', '+0.55', 'knowing k exactly is worth almost nothing'],
    ['one constant k for every question', [nsr('+0.06 n.s.')], 'a fixed budget is worth nothing at all']
  ], [0.32, 0.26, 0.42], { rightFrom: 99 }));
kids.push(...callout('The claim to make', [
  r('Not “we estimate k accurately” — we do not. It is: '),
  b('the keep-count must be question-dependent, and even a noisy estimate of how it should vary captures most of the value.')]));

// ============ 7
kids.push(sectionHead('7', 'Every ensemble, n = 3 to 6'));
kids.push(lede('All 210 subsets of the 8 agents, 2-fold cross-fitted. This is the strongest evidence we have.'));
kids.push(table(['n', 'ens', 'best single', 'ASC', 'count', 'weighted', 'pattern', 'ours', 'ours vs count'],
  [
    ['3', '56', '24.72', '21.38', '22.15', '24.40', '24.88', '26.59', [sig('+20.0%')]],
    ['4', '70', '26.00', '23.36', '23.69', '25.59', '26.26', '28.22', [sig('+19.1%')]],
    ['5', '56', '26.77', '25.68', '26.17', '26.56', '27.29', '29.30', [sig('+12.0%')]],
    ['6', '28', '27.18', '27.13', '27.74', '27.38', '28.11', '30.05', [sig('+8.3%')]]
  ], [0.06, 0.08, 0.14, 0.10, 0.10, 0.12, 0.11, 0.10, 0.19], { rightFrom: 0 }));
kids.push(H('How often each arm significantly beats counting', HeadingLevel.HEADING_2));
kids.push(table(['n', 'weighted', 'support-pattern lookup', 'rank', 'rank + budget'],
  [
    ['3', '39/56', '45/56', '56/56', [sig('56/56')]],
    ['4', '39/70', '44/70', '70/70', [sig('70/70')]],
    ['5', '20/56', '23/56', '50/56', [sig('56/56')]],
    ['6', '7/28', '7/28', '21/28', [sig('28/28')]],
    [[b('all')], '105/210', '119/210', '197/210', [sig('210/210')]]
  ], [0.12, 0.22, 0.28, 0.18, 0.20], { rightFrom: 0, hi: [4] }));
kids.push(...callout('The crossover, resolved', [
  r('Reliability weighting decays with ensemble size and goes '),
  new TextRun({ text: 'negative', italics: true, font: SANS, size: 20 }),
  r(' by n = 6 — that is the crossover we reported in August. Our method does not: it holds a significant margin at every size. '),
  b('The crossover is a property of Class B rules, not of aggregation.'),
  r(' Counting itself improves with n (22.15 → 27.74) because more agents sharpen the support count; what shrinks is the headroom left for weighting.')]));

// ============ 8
kids.push(sectionHead('8', 'Are we overfitting?'));
kids.push(lede('The right concern to raise. Two answers — the protocol answer, and a harder test.'));
kids.push(H('The protocol answer', HeadingLevel.HEADING_2));
kids.push(P([r('Everything is cross-fitted '), b('by question'), r(', 5-fold. The support-pattern lookup, the logistic coefficients, the gradient-boosted model, the gold-count regression and the keep threshold are all fitted on training folds and scored on a held-out fold. Nothing is tuned on what it is scored on.')]));
kids.push(P([r('This matters concretely: this project has already had a headline flip sign from '),
  b('+0.08 to −1.25'), r(' once thresholds were cross-fitted honestly. Anything tuned in-sample is not a result.')]));
kids.push(H('The harder test — leave-one-source-out', HeadingLevel.HEADING_2));
kids.push(body('Random folds draw train and test from the same question distribution, so a rule exploiting something peculiar to QAMPARI-as-a-whole would still look fine. QAMPARI ships five structurally different question sources; we train on four and test on the fifth.'));
kids.push(table(['Arm', 'random 5-fold', 'held-out source', 'shift', 'vs count'],
  [
    [[b('multi-agent count')], '29.35', '29.35', '0.00', '—'],
    [[b('support-pattern lookup')], '29.60', '29.37', '−0.23', [nsr('+0.02 n.s.')]],
    [[b('rank scorer')], '30.79', '30.73', '−0.05', [sig('+1.38 *')]],
    [[b('rank scorer + budget')], '31.74', '31.48', '−0.26', [sig('+2.13 *')]],
    [[b('rank scorer + boosting + budget')], '32.02', '31.44', '−0.58', [sig('+2.09 *')]]
  ], [0.34, 0.16, 0.17, 0.12, 0.21], { hi: [3] }));
kids.push(capt('Gains are positive on all five held-out sources: +1.28, +1.84, +3.11, +3.55, +2.64.'));
kids.push(P([b('80–90% of the gain survives a complete change of question type.'), r(' Two things worth noticing beyond that headline:')]));
kids.push(...callout('The test has teeth', [
  r('The support-pattern lookup is a 256-cell table — literally the memorisation the concern describes. Under random folds it is +0.25; under distribution shift it collapses to '),
  b('+0.02, not significant'), r('. So the protocol does detect pattern-memorisation, and our features are not what it flags.')]));
kids.push(...callout('Consequence for the paper', [
  r('The gradient-boosted model transfers worst. Out of distribution the rank scorer + budget (31.48) now '),
  b('beats'), r(' the boosted blend (31.44). The extra capacity buys +0.3 in-distribution and costs more than that out of it. '),
  b('Make the rank scorer + budget the headline method'), r(' — plain logistic plus the budget. Simpler to describe, one fewer moving part, more robust.')], true));

// ============ 9
kids.push(sectionHead('9', 'Predicting the gain'));
kids.push(lede('Requested at the 21 August meeting: given a pool of models, can we predict how much our method will help? Yes — leave-one-out R² = 0.817.'));
kids.push(body('Regressing the gain over the best single agent on properties computable before running the method, across all 210 ensembles:'));
kids.push(table(['Predictor', 'corr. with gain vs best single', 'corr. with gain vs counting'],
  [
    [[b('dominance — best ÷ mean of the rest')], [bad('−0.620')], [sig('+0.687')]],
    ['solo ratio', '−0.595', '+0.328'],
    ['mean F1 of the pool', '+0.345', '−0.460'],
    ['coefficient of variation (spread ÷ mean)', '−0.334', '+0.411'],
    ['ensemble size', '+0.314', '−0.407'],
    ['mean pairwise Jaccard of claim sets', '−0.068', '−0.416']
  ], [0.46, 0.27, 0.27], { hi: [0] }));
kids.push(...callout('Why our two results tables look so different', [
  b('Dominance flips sign between them.'),
  r(' One dominant model makes counting bad — so we beat counting easily (+0.687) — while simultaneously making the best single agent hard to beat (−0.620). Table 1 and Table 2 are not two views of one thing; they are driven by the same variable in opposite directions.')]));
kids.push(P([r('Parsimonious version for the paper — three interpretable quantities, size + dominance + (union recall − best agent’s recall), gives '),
  b('leave-one-out R² = 0.642'), r('. Dominance alone gives 0.368.')]));

// ============ 10
kids.push(sectionHead('10', 'The other two benchmarks'));
kids.push(lede('The method is QAMPARI-strong and does not transfer uniformly. Here is the honest picture.'));
kids.push(table(['Benchmark', 'Metric', 'counting', 'ours', 'gain', 'What holds'],
  [
    ['QAMPARI · 777q, 8 agents', 'set F1', '29.35', '32.01', [sig('+9.1% *')], 'everything'],
    ['ASQA · 400q, 6 agents', 'Disambiguation-ROUGE proxy', '24.77', '27.50', [sig('+11.1% *')], 'position scoring only — the budget does not transfer'],
    ['TriviaQA · 2757q, 6 agents', 'accuracy', '84.98', '86.69', [sig('+2.0% *')], 'position and budget undefined (single answer); self-consistency instead']
  ], [0.20, 0.16, 0.10, 0.09, 0.11, 0.34], { rightFrom: 2 }));
kids.push(H('Why the transfer is uneven — the solo ratio', HeadingLevel.HEADING_2));
kids.push(body('Once you keep everything with at least 2 supporters, the only decision weighting changes is whether to admit a solo claim. So its value is set entirely by how informative a solo claim is:'));
kids.push(...code([
  'solo ratio = P(correct | exactly 1 agent asserts)',
  '             ÷ P(correct | 2 or more agents assert)'
]));
kids.push(table(['Benchmark', 'P(correct | 1)', 'P(correct | ≥2)', 'solo ratio', 'weighting − counting'],
  [
    ['QAMPARI', '1.2%', '19.9%', '0.06', [nsr('+0.25 n.s.')]],
    ['TriviaQA', '20.1%', '82.3%', '0.24', [sig('+0.94 *')]],
    ['ASQA', '22.5%', '47.5%', '0.47', [sig('+2.69 *')]]
  ], [0.22, 0.17, 0.18, 0.14, 0.29]));
kids.push(capt('Monotone across three benchmarks with different answer formats, metrics and pools. Computable from cached generations before implementing anything — a cheap screen for candidate datasets.'));
kids.push(P([r('Low ratio → counting already filters well, weighting is redundant, and the wins come from position and budgets. High ratio → counting cannot filter at all and weighting is the whole game. '),
  b('Notably this is not pool heterogeneity'), r(': ASQA’s pool is far more homogeneous (coefficient of variation 0.125 versus QAMPARI’s 0.439) and is where weighting wins biggest — the opposite of what we predicted.')]));

// ============ 11
kids.push(sectionHead('11', 'What we do not claim'));
kids.push(lede('Negative results, each cross-fitted, each reusable.'));
kids.push(table(['Idea', 'Result', 'Why it matters'],
  [
    [[b('Gradient boosting over the same features')], [bad('−0.58 shift')], 'Gains +0.3 in-distribution and loses more than that under distribution shift, so the plain logistic wins out of distribution. Removed from the method, retained as an ablation.'],
    [[b('Verbosity as a signal')], [nsr('+0.01')], 'Not in the method. Adds +0.01 over position alone and was ablated out. List length survives only inside omission strength — see below. Prior art exists (response length as a hallucination signal) and we claim none of it.'],
    ['Error-correlation / diversity weighting', [bad('λ = 0')], 'Selected λ = 0 in 5 of 5 folds. It is a function of the support pattern, so the ceiling already caps it.'],
    ['Pairwise co-assertion (28 indicators)', [nsr('+0.05')], 'The same conclusion reached discriminatively, without assuming a functional form.'],
    ['Question-domain weights', [nsr('+0.00')], 'QAMPARI’s 5 question sources carry nothing. “Condition on the question” is not the same as topic.'],
    ['Within-question score normalisation', [bad('−0.9 to −2.6')], 'Destroys the cross-question information a global threshold needs.'],
    ['Learned per-question k', [bad('−3.71')], 'The optimal k is essentially unpredictable from question features; the plug-in rule is better.'],
    ['Alias merging before scoring', [bad('−0.02 to −1.30')], 'Bad merges outnumber good at every threshold (7 vs 40 at τ = 0.85). String-similar QAMPARI entities are usually different entities.'],
    ['Isotonic calibration', [nsr('−0.03')], 'The logistic was already calibrated: 3.55% predicted versus 3.55% actual.']
  ], [0.30, 0.16, 0.54], { rightFrom: 99 }));

kids.push(H('Why list length was dropped, in detail', HeadingLevel.HEADING_2));
kids.push(P([r('It is a fair question, because the raw signal looks strong: precision falls from '),
  b('30.2%'), r(' for items from a short list to '), b('3.4%'), r(' for items from a long one, a 26.7-point spread. Three things explain why it contributes +0.01 anyway.')]));
kids.push(table(['Precision by list length', 'short (<10)', 'medium (10–24)', 'long (25+)', 'spread'],
  [
    ['ignoring position', '30.2%', '22.8%', '3.4%', [bad('26.7 pts')]],
    ['within positions 1–3', '30.5%', '33.3%', '24.1%', '9.2 pts'],
    ['within positions 4–10', '29.7%', '23.5%', '15.4%', '14.3 pts'],
    ['within positions 11+', '—', '15.3%', '2.2%', '13.1 pts']
  ], [0.28, 0.18, 0.20, 0.16, 0.18]));
kids.push(capt('Controlling for position removes half to two-thirds of the apparent length effect.'));
kids.push(...numbered([
  [b('Most of it is position in disguise.'), r(' An item at position 40 can only exist in a list of at least 41 items, so long lists are largely lists with many high-position items. Conditioning on position removes half to two-thirds of the spread.')],
  [b('The remainder is already captured'), r(' by relative position (position ÷ list length), which is part of the position feature block. “Item 11 of 12” and “item 11 of 500” are already distinguished.')],
  [b('What is left sits where the decision is already made.'), r(' 98,000 of roughly 141,000 assertions are position-11-or-later inside long lists, at 2.2% precision — 70% of the mass, already excluded by position alone.')]
]));
kids.push(...callout('Why the same quantity earns its place on the omission side', [
  b('A silent agent has no position.'),
  r(' There is no rank to condition on, so how much it said is the only thing you know about it. That is why list length is redundant for agents that asserted (+0.01) and irreplaceable for agents that stayed quiet (+0.33) — and why omission strength is in the method while verbosity is not.')]));

kids.push(...callout('Prior art we must cite, not claim', [
  b('Dawid–Skene. '),
  r('Our Naive-Bayes-with-silence and support-pattern-lookup arms are, in substance, a classical annotator-reliability model applied to LLM agents. Cite the crowdsourcing literature and present them as baselines and ceilings, not contributions. This strengthens the argument: even the classical ceiling is +0.25 not significant, and output-structure signals beat it.')], true));
kids.push(...callout('One useful confirmation', [
  r('The Optimal Weight paper we extend (arXiv:2510.01499) targets '), b('binary and categorical'),
  r(' predictions, not set-valued generation. Its log-odds weight requires per-model accuracy above 0.5; our per-claim precisions run 5.8–28%, so every weight goes negative and the method inverts. That is the cleanest possible statement of the gap we fill.')]));

// ============ 12
kids.push(sectionHead('12', 'Open items'));
kids.push(...bullets([
  [b('The ASQA per-ensemble table is incomplete'), r(' — it has counting versus weighting only; our method is not in it. Roughly 30–45 minutes, free.')],
  [b('Paper-exact single-model Atomic Self-Consistency'), r(' needs new generation (777 × m calls on one model). The current arm substitutes agents for stochastic samples — unavoidable from cache, and it is the extension we propose, but it should be stated.')],
  [b('The ASQA metric'), r(' is a proxy: the square root of ROUGE-L times string exact-match. A faithful published metric needs the RoBERTa-SQuADv2 reader.')],
  [b('Roughly 15 novelty-check sources'), r(' were read only as search summaries and need direct reads before entering a bibliography.')],
  [b('The repository README is stale'), r(' — the front page still advertises the 18 August results, including QUEST, which was dropped.')]
]));

kids.push(P(t('All numbers reproduce from cached generations with no API calls. Code and results: github.com/kevkev1122r/open_ended_aggregation — analysis modules under open_ended_aggregation/analysis/, full per-ensemble listings in docs/ENSEMBLE_ANALYSIS.md, robustness in docs/ROBUSTNESS.md, literature in docs/NOVELTY_CHECK.md.',
  { size: 17, color: FAINT }), { before: 400 }));

// ============================================================ DOC
const doc = new Document({
  creator: 'Open-ended aggregation',
  title: 'Rank + budget — method, arms and evidence',
  styles: {
    default: {
      document: { run: { font: SANS, size: 20, color: INK }, paragraph: { spacing: { line: 276 } } },
      heading1: { run: { font: SANS, size: 30, bold: true, color: INK }, paragraph: { spacing: { before: 420, after: 160 } } },
      heading2: { run: { font: SANS, size: 24, bold: true, color: INK }, paragraph: { spacing: { before: 320, after: 120 } } },
      heading3: { run: { font: SANS, size: 21, bold: true, color: SOFT }, paragraph: { spacing: { before: 260, after: 100 } } }
    }
  },
  numbering: {
    config: [
      { reference: 'dash', levels: [{ level: 0, format: LevelFormat.BULLET, text: '–', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 200 } } } }] },
      { reference: 'nums', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 220 } } } }] }
    ]
  },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } }
    },
    children: kids
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('method-explainer.docx', buf);
  console.log('wrote method-explainer.docx', buf.length, 'bytes');
});
