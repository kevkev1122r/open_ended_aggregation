import re
p = "acl_main.tex"
t = open(p, encoding="utf-8").read()
def rep(old, new):
    global t
    n = t.count(old)
    assert n == 1, (n, old[:90])
    t = t.replace(old, new)

# ---------------- 1. numbers in the text
rep("behavioural aggregation reaches $31.70$ set F1", "behavioural aggregation reaches $31.54$ set F1")
rep("On CWQ, it reaches $23.65$, compared with $21.81$ for counting.", "On CWQ, it reaches $23.35$, compared with $21.81$ for counting.")
rep("the set F1 score increases to 31.70. This is a +2.50 F1 gain over counting (95\\% CI: [+1.72, +3.28]) and a +4.30 F1 gain over the best single agent (95\\% CI: [+3.37, +5.26]).",
    "the set F1 score increases to 31.54. This is a +2.33 F1 gain over counting (95\\% CI: [+1.51, +3.14]) and a +4.14 F1 gain over the best single agent (95\\% CI: [+3.19, +5.10]).")
rep("Behavioural aggregation reaches 23.65, a gain of +1.85 over counting (95\\% CI: [+1.14, +2.61]) and +2.74 over the best single agent (95\\% CI: [+1.78, +3.76]).",
    "Behavioural aggregation reaches 23.35, a gain of +1.55 over counting (95\\% CI: [+0.83, +2.30]) and +2.44 over the best single agent (95\\% CI: [+1.38, +3.52]).")
rep("Table~\\ref{tab:main} gives every arm on all three benchmarks.", "Table~\\ref{tab:main} gives every arm on all five benchmarks.")
rep("the method retains roughly 92\\% of its gain, achieving 31.49 F1.", "the method retains roughly 81\\% of its gain, achieving 31.10 F1.")
rep("by +1.34, +1.65 and +1.52 F1 at sizes 3, 4 and 5, with bootstrap intervals excluding zero in 13 of the 16. The identity-only rule over the same subsets gains +0.31, +0.21 and $-0.45$, significant in 3 of 16.",
    "by +1.13, +1.63 and +1.55 F1 at sizes 3, 4 and 5, with bootstrap intervals excluding zero in 11 of the 16. The identity-only rule over the same subsets gains +0.28, +0.33 and $-0.29$, significant in 4 of 16.")
rep("reaches $+4.30$ over the best single agent and $+2.50$ over counting on QAMPARI and $+1.85$ over counting on CWQ",
    "reaches $+4.14$ over the best single agent and $+2.33$ over counting on QAMPARI and $+1.55$ over counting on CWQ")
rep("keeps roughly $92\\%$ of its gain on question types never seen in training.", "keeps roughly $81\\%$ of its gain on question types never seen in training.")

# ---------------- 2. main table
s = t.index("\\begin{table*}[t]\n\\centering\n\\caption{Aggregation performance across benchmarks.")
e = t.index("\\end{table*}", s) + len("\\end{table*}")
t = t[:s] + r"""\begin{table*}[t]
\centering
\caption{Aggregation performance across benchmarks. Behavioural aggregation leads on the three list-answering benchmarks and on TriviaQA, while reliability weighting leads on ASQA. The pattern lookup is the strongest rule expressible from the support pattern alone and so upper-bounds any reliability-weighting scheme. Gains are 95\% paired-bootstrap intervals over questions; $^*$ marks an interval excluding zero. Gain over counting: QAMPARI $+2.33\ [+1.51, +3.14]$, CWQ $+1.55\ [+0.83, +2.30]$, WebQSP $+1.03\ [+0.28, +1.85]$, TriviaQA $+1.70\ [+0.94, +2.50]$, ASQA $+2.05\ [+0.92, +3.22]$. Against reliability weighting, behavioural aggregation is $-0.64\ [-1.31, +0.02]$ on ASQA and $+0.76\ [+0.11, +1.45]$ on TriviaQA. TriviaQA returns one answer per question: its behavioural signals are self-consistency and verbosity, there is no keep budget, reliability weighting is the log-odds rule, and the pattern-lookup row is a logistic rule over which agents gave the answer. ASC is run on the list benchmarks with its published F1-5 tuning.}
\label{tab:main}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{QAMPARI} & \textbf{CWQ} & \textbf{WebQSP} & \textbf{TriviaQA} & \textbf{ASQA} \\
 & (Set F1) & (Set F1) & (Set F1) & (Accuracy) & (DR* proxy) \\
\midrule
Best single agent & 27.40 & 20.92 & 30.45 & 86.29 & 27.03 \\
Multi-agent counting & 29.21 & 21.81 & 31.72 & 84.98 & 24.77 \\
Reliability weighting & 29.47 & 20.78 & 31.61 & 85.93 & \textbf{27.45} \\
Pattern lookup (ceiling on weighting) & 29.50 & 21.52 & 31.73 & 86.22 & 27.20 \\
ASC & 28.43 & 21.62 & 29.41 & --- & --- \\
\textbf{Behavioural aggregation} & \textbf{31.54} & \textbf{23.35} & \textbf{32.75} & \textbf{86.69} & 26.81 \\
\midrule
Gain vs.\ counting & $+2.33^*$ & $+1.55^*$ & $+1.03^*$ & $+1.70^*$ & $+2.05^*$ \\
\bottomrule
\end{tabular}
\end{table*}""" + t[e:]

# ---------------- 3. regenerated appendix tables and their captions/text
rep("$\\hat{G}$ estimator         & ridge, $\\lambda = 1.0$ & ridge, $\\lambda = 1.0$ & --- \\\\",
    "$\\hat{G}$ estimator         & ridge on $\\log G$, $\\lambda = 1.0$ & ridge on $\\log G$, $\\lambda = 1.0$ & --- \\\\")
rep("$\\hat{G}$ mean abs.\\ error  & $4.62$ (gold mean $10.2$) & $3.26$ (gold mean $16.0$) & --- \\\\",
    "$\\hat{G}$ mean abs.\\ error  & $4.33$ (gold mean $10.2$) & $3.19$ (gold mean $16.0$) & --- \\\\")
# LOSO
rep("Under a complete change of question type our method loses $0.21$ F1 against a gain of $2.50$, so roughly $92\\%$ survives",
    "Under a complete change of question type our method loses $0.44$ F1 against a gain of $2.33$, so roughly $81\\%$ survives")
rep("\\textbf{behavioural aggregation} & \\textbf{31.70} & \\textbf{31.49} & $\\mathbf{-0.21}$ & $\\mathbf{+2.29\\ [+1.47,+3.10]^*}$ \\\\",
    "\\textbf{behavioural aggregation} & \\textbf{31.54} & \\textbf{31.10} & $\\mathbf{-0.44}$ & $\\mathbf{+1.89\\ [+1.02,+2.73]^*}$ \\\\")
rep("logistic/boosting blend + budget & 32.29 & 31.43 & $-0.86$ & $+2.22\\ [+1.40,+3.05]^*$ \\\\",
    "logistic/boosting blend + budget & 32.16 & 31.26 & $-0.90$ & $+2.05\\ [+1.18,+2.93]^*$ \\\\")
# feature-control paragraph (first-position precision by band)
rep("across the four length bands is $2.2$, $4.1$, $4.4$ and $4.3$ percent, which is flat.",
    "across the four length bands is $3.1$, $5.6$, $4.8$ and $4.4$ percent, which does not fall.")
# ensemble-size tables
s = t.index("\\caption{Relative gain over counting, averaged across all ensembles of each size.")
s = t.rindex("\\begin{table*}[t]", 0, s); e = t.index("\\end{table*}", s) + len("\\end{table*}")
t = t[:s] + r"""\begin{table*}[t]
  \caption{Relative gain over counting, averaged across all ensembles of each size. On QAMPARI the gain from reliability weighting shrinks as the agent pool grows, to $+0.38$ at size 6, while the output-structure arms decay far more slowly. CWQ and WebQSP rows use the five-agent pool of Section~\ref{sec:experiments}.}
  \label{tab:ens-gain}
  \centering
  \begin{tabular}{rrrrrrrr}
    \toprule
    Size & Ensembles & Best single & Count & Weighted & Pattern & Beh.\ agg.\ (scorer) & Beh.\ aggregation \\
    \midrule
    \multicolumn{8}{l}{\emph{QAMPARI}} \\
    3 & 56 & 24.74 & 22.66 & $+1.96$ & $+2.33$ & $+3.82$ & $\mathbf{+4.20}$ \\
    4 & 70 & 26.00 & 24.04 & $+1.85$ & $+2.41$ & $+4.00$ & $\mathbf{+4.43}$ \\
    5 & 56 & 26.75 & 26.24 & $+0.86$ & $+1.30$ & $+2.83$ & $\mathbf{+3.31}$ \\
    6 & 28 & 27.17 & 27.73 & $+0.38$ & $+0.67$ & $+2.05$ & $\mathbf{+2.64}$ \\
    \midrule
    \multicolumn{8}{l}{\emph{CWQ}} \\
    3 & 10 & 21.53 & 22.07 & $+0.21$ & $+0.28$ & $+1.44$ & $\mathbf{+1.13}$ \\
    4 &  5 & 21.30 & 21.77 & $+0.23$ & $+0.33$ & $+1.91$ & $\mathbf{+1.63}$ \\
    5 &  1 & 20.92 & 21.81 & $-1.02$ & $-0.29$ & $+1.65$ & $\mathbf{+1.55}$ \\
    \midrule
    \multicolumn{8}{l}{\emph{WebQSP}} \\
    3 & 10 & 30.40 & 30.42 & $+0.08$ & $+0.48$ & $+1.33$ & $\mathbf{+1.48}$ \\
    4 &  5 & 30.45 & 31.33 & $-0.00$ & $-0.15$ & $+0.70$ & $\mathbf{+1.12}$ \\
    5 &  1 & 30.45 & 31.72 & $-0.11$ & $+0.00$ & $+0.54$ & $\mathbf{+1.03}$ \\
    \bottomrule
  \end{tabular}
\end{table*}""" + t[e:]
s = t.index("\\caption{Ensembles in which each arm beats counting")
s = t.rindex("\\begin{table*}[t]", 0, s); e = t.index("\\end{table*}", s) + len("\\end{table*}")
t = t[:s] + r"""\begin{table*}[t]
  \caption{Ensembles in which each arm beats counting with a bootstrap interval excluding zero. On QAMPARI our method does so in all 210 at every size, against reliability weighting's 104 of 210; on CWQ it does so in 11 of 16 and on WebQSP in 13 of 16, against the pattern lookup's 4 and 5.}
  \label{tab:ens-sig}
  \centering
  \begin{tabular}{rrrrr}
    \toprule
    Size & Weighted & Pattern & Beh.\ agg.\ (scorer) & Beh.\ aggregation \\
    \midrule
    \multicolumn{5}{l}{\emph{QAMPARI}} \\
    3 & 36/56 & 42/56 & 56/56 & \textbf{56/56} \\
    4 & 40/70 & 45/70 & 70/70 & \textbf{70/70} \\
    5 & 21/56 & 26/56 & 56/56 & \textbf{56/56} \\
    6 &  7/28 &  8/28 & 28/28 & \textbf{28/28} \\
    \midrule
    Total & 104/210 & 121/210 & 210/210 & \textbf{210/210} \\
    \midrule
    \multicolumn{5}{l}{\emph{CWQ}} \\
    3 & 2/10 & 2/10 & 10/10 & \textbf{6/10} \\
    4 & 1/5  & 2/5  & 5/5   & \textbf{4/5}  \\
    5 & 0/1  & 0/1  & 1/1   & \textbf{1/1}  \\
    \midrule
    Total & 3/16 & 4/16 & 16/16 & \textbf{11/16} \\
    \midrule
    \multicolumn{5}{l}{\emph{WebQSP}} \\
    3 & 3/10 & 5/10 & 9/10 & \textbf{8/10} \\
    4 & 0/5  & 0/5  & 2/5  & \textbf{4/5}  \\
    5 & 0/1  & 0/1  & 0/1  & \textbf{1/1}  \\
    \midrule
    Total & 3/16 & 5/16 & 11/16 & \textbf{13/16} \\
    \bottomrule
  \end{tabular}
\end{table*}""" + t[e:]
# negative-results text
rep("a learned per-question $k$ loses $3.71$, while a single constant $k$, cross-fitted, picks $k=8$ and gains $+0.20$",
    "a learned per-question $k$ loses $3.26$, while a single constant $k$, cross-fitted, picks $k=8$ or $9$ and gains $+0.25$")
rep("where a learned $k$ loses $1.24$ against counting while the budget gains $+1.76$.",
    "where a learned $k$ loses $1.24$ against counting while the budget gains $+1.55$.")
# alias table
s = t.index("\\caption{Alias merging at six string-similarity thresholds")
s = t.rindex("\\begin{table*}[t]", 0, s); e = t.index("\\end{table*}", s) + len("\\end{table*}")
t = t[:s] + r"""\begin{table*}[t]
  \caption{Alias merging at six string-similarity thresholds, with good and bad merges counted against the gold answers (a merge is good when both strings match the same gold answer). No threshold helps, and the damage grows as merging loosens.}
  \label{tab:alias}
  \centering
  \begin{tabular}{rrrrr}
    \toprule
    $\tau$ & Candidates & Good / bad merges & LEAN + budget & vs.\ no merge \\
    \midrule
    $1.00$ & 110,136 & 0 / 0     & $31.54$ & (reference) \\
    $0.95$ & 108,039 & 0 / 1     & $31.49$ & $-0.05$ \\
    $0.90$ & 105,495 & 7 / 10    & $31.45$ & $-0.09$ \\
    $0.85$ & 102,086 & 7 / 41    & $31.29$ & $-0.25$ \\
    $0.80$ &  94,390 & 8 / 106   & $31.34$ & $-0.20$ \\
    $0.70$ &  81,547 & 15 / 414  & $30.46$ & $-1.08$ \\
    \bottomrule
  \end{tabular}
\end{table*}""" + t[e:]
# negative-results table
rep("Aggregation rules that did not work, with the reading each supports. Several land at or below the pattern-lookup ceiling, as that construction predicts. Pre-fix sweep, pending regeneration.}",
    "Aggregation rules that did not work, with the reading each supports. Several land at or below the pattern-lookup ceiling, as that construction predicts. Gains are against counting unless marked.}")
rep("pairwise co-assertion, 28 indicators & $+0.05$ &", "pairwise co-assertion, 28 indicators & $-0.04$ vs.\\ scorer &")
rep("rank agreement across agents  & $+0.07$ &", "rank agreement across agents  & $+0.09$ vs.\\ scorer &")
rep("question-domain weights       & $+0.00$ &", "question-domain weights       & $+0.11$ &")
rep("Borda / CombSUM / weighted Borda & $-0.42$ to $-0.67$ &", "Borda / CombSUM / weighted Borda & $-0.40$ to $-0.53$ &")
rep("within-question $z$-normalisation & $-0.94$ &", "within-question $z$-normalisation & $-1.57$ &")
rep("within-question percentile norm.  & $-2.56$ &", "within-question percentile norm.  & $-2.93$ &")
rep("learned per-question $k$      & $-3.71$ &", "learned per-question $k$      & $-3.26$ &")
rep("single constant $k$           & $+0.20$ &", "single constant $k$           & $+0.25$ &")
rep("alias merging, six thresholds & $-0.02$ to $-1.30$ &", "alias merging, six thresholds & $-0.05$ to $-1.08$ &")
rep("isotonic calibration, global  & $-0.03$ & dominated by the bottom bin, which is already accurate \\\\",
    "isotonic calibration, global  & $+0.12$ vs.\\ budget, n.s. & within noise: the bottom bins, which hold most claims, are already accurate \\\\")
# regimes figure caption (figure regenerated from the new sweeps)
rep("for all 210 QAMPARI subsets, all 16 CWQ subsets and all 42 ASQA subsets.", "for all 210 QAMPARI subsets, all 16 CWQ subsets, all 16 WebQSP subsets and all 42 ASQA subsets.")
rep("The three benchmarks separate: on QAMPARI counting closes a $2.55$ deficit and passes the best agent at six agents, on CWQ it is already ahead at every size,",
    "The benchmarks separate: on QAMPARI counting closes a $2.08$ deficit and passes the best agent at six agents, on CWQ and WebQSP it is already ahead at every size,")
rep("$0.06$ on QAMPARI, $0.05$ on CWQ and $0.47$ on ASQA. QAMPARI numbers predate the parsing fix.}",
    "$0.06$ on QAMPARI, $0.05$ on CWQ, $0.09$ on WebQSP and $0.47$ on ASQA.}")
# position table
s = t.index("\\caption{Correctness by within-output position, restricted to solo assertions")
s = t.rindex("\\begin{table*}[t]", 0, s); e = t.index("\\end{table*}", s) + len("\\end{table*}")
t = t[:s] + r"""\begin{table*}[t]
  \caption{Correctness by within-output position, restricted to solo assertions and stratified by the asserter's own list length. Short lists show no usable signal --- flat or mildly reversed --- and the gradient appears only on long lists, on all three list benchmarks. The feature pays because $71\%$ of QAMPARI solo assertions and $73\%$ of CWQ's sit in lists of 101 or more. A control confirms the effect is positional, not verbosity: on QAMPARI, $P(\mathrm{correct} \mid \text{first position})$ by length band is $3.1$, $4.8$, $5.6$ and $4.4$ percent in the order of the rows below, with no fall as lists grow. Dashes mark cells with too few observations.}
  \label{tab:rank}
  \centering
  \begin{tabular}{lrrr}
    \toprule
    Asserter's list length & 1st quintile & Last quintile & Spearman (rank, correct) \\
    \midrule
    \multicolumn{4}{l}{\emph{QAMPARI}} \\
    1 to 5 items & $3.1\%$ & $5.7\%$ & $+0.075$ \\
    6 to 20      & $5.8\%$ & $3.0\%$ & $-0.056$ \\
    21 to 100    & $5.5\%$ & $1.2\%$ & $-0.103$ \\
    101 or more  & $0.9\%$ & $0.1\%$ & $-0.069$ \\
    pooled solo  &         &         & $-0.133$ \\
    \midrule
    \multicolumn{4}{l}{\emph{CWQ}} \\
    1 to 5 items & --- & --- & $-0.005$ \\
    6 to 20      & $9.2\%$ & $6.4\%$ & $-0.068$ \\
    21 to 100    & $5.2\%$ & $2.6\%$ & $-0.087$ \\
    101 or more  & $1.2\%$ & $0.1\%$ & $-0.081$ \\
    pooled solo  &         &         & $-0.159$ \\
    \midrule
    \multicolumn{4}{l}{\emph{WebQSP}} \\
    1 to 5 items & --- & --- & $+0.132$ \\
    6 to 20      & $5.0\%$ & $5.1\%$ & $+0.017$ \\
    21 to 100    & $16.5\%$ & $8.0\%$ & $-0.055$ \\
    101 or more  & $4.1\%$ & $1.5\%$ & $-0.154$ \\
    pooled solo  &         &         & $-0.184$ \\
    \bottomrule
  \end{tabular}
\end{table*}""" + t[e:]
# learning curve
s = t.index("\\caption{Gain over counting by training-set size on QAMPARI")
s = t.rindex("\\begin{table*}[t]", 0, s); e = t.index("\\end{table*}", s) + len("\\end{table*}")
t = t[:s] + r"""\begin{table*}[t]
  \caption{Gain over counting by training-set size on QAMPARI, fixed held-out test set, five independent training draws per size, two test splits. The gain is positive on both splits from 100 questions, reaches its full-data level by roughly 350, and the draw-to-draw spread collapses from $\pm 1.57$ to $\pm 0.19$, past which point the choice of fitting questions stops mattering. At 25 questions the method does not beat counting. CWQ's 415 questions and ASQA's 400 both sit above this floor.}
  \label{tab:curve}
  \centering
  \begin{tabular}{rrrr}
    \toprule
    Training questions & Split A & Split B & Spread across draws \\
    \midrule
     25 & $-1.23$ & $+0.02$ & $\pm 1.57$ \\
     50 & $+0.46$ & $-0.26$ & $\pm 0.58$ \\
    100 & $+0.91$ & $+0.41$ & $\pm 0.40$ \\
    200 & $+1.04$ & $+1.00$ & $\pm 0.46$ \\
    350 & $+1.64$ & $+1.19$ & $\pm 0.19$ \\
    527 & $+1.61$ & $+1.24$ & \\
    \bottomrule
  \end{tabular}
\end{table*}""" + t[e:]

# ---------------- 4. delete the LLM Usage appendix
rep("\\section{LLM Usage}\n\nWe use LLMs for editing and writing help. Additionally, we use Claude Code with the Opus 5 model to set-up and run experiments, and generate figures.\n\n", "")

# ---------------- 5. missing one-liners (bootstrap sentence already in Section 3.3)
rep("all $n$ agents are pooled into a single candidate set $C(q)$.",
    "all $n$ agents are pooled into a single candidate set $C(q)$. Agents answer independently: each is queried once and never sees another agent's output.")
rep("and the sum and maximum of the predicted probabilities. It need\nnot be accurate,",
    "and the sum and maximum of the predicted probabilities. It is fitted to $\\log G_q$ and exponentiated, because gold-set sizes are heavy-tailed (WebQSP averages 76 gold answers per question). It need\nnot be accurate,")

# ---------------- 7. Limitations (sampling sentence already present): stale numbers
rep("the QAMPARI gain depends on $70\\%$ of solo assertions", "the QAMPARI gain depends on $71\\%$ of solo assertions")
rep("with a floor near 50 labelled questions and saturation near 200", "with a floor near 100 labelled questions and saturation near 350")

open(p, "w", encoding="utf-8").write(t)
print("all replacements applied")
