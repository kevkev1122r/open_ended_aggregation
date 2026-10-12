import re, pathlib
p = pathlib.Path("acl_main.tex"); s = p.read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)

# ---- 10. notation in §2.1
rep(r"""We give the same question $q$ to each of $n$ language models, which we call agents $M$. Every agent returns one open-ended answer, either an ordered list of entities
or a paragraph of several statements. We split each answer into \emph{claims}, decomposed
parts of an answer. Each claim is then
\emph{normalised} by lowercasing it and stripping punctuation and articles, so that the
same claim written by two agents collapses to one entry, and the normalised claims from
all $n$ agents are pooled into a single candidate set $C(q)$. Agents answer independently: each is queried once and never sees another agent's output.""",
r"""We give the same question $q$ to each agent $m$ in a set $M$ of $n$ language models. Agents answer independently: each is queried once and never sees another agent's output. Every agent returns one open-ended answer, either an ordered list of entities
or a paragraph of several statements. We split each answer into \emph{claims} and
\emph{normalise} each claim by lowercasing it and stripping punctuation and articles, so that the
same claim written by two agents collapses to one entry. Pooling the normalised claims of
all $n$ agents gives the candidate set $C(q)$. For a candidate $c \in C(q)$, $S(c) \subseteq M$
is the set of agents that asserted it, $r_m(c)$ is the position of $c$ in agent $m$'s output
for $m \in S(c)$, and $L_m(q)$ is the number of distinct claims agent $m$ produced for $q$.
In Figure~\ref{fig:pipeline}, both claims have the same single asserter, so $|S(c)| = 1$
for each; the correct one has $r_m(c) = 2$ with $L_m(q) = 3$, the incorrect one
$r_m(c) = 26$ with $L_m(q) = 101$.""")

# ---- 13a. §2.4 agent-invariant scorer
rep(r"""It need
not be accurate, since the estimate enters only through a threshold that moves slowly in
$\hat{G}_q$.
""", r"""It need
not be accurate, since the estimate enters only through a threshold that moves slowly in
$\hat{G}_q$.

\subsection{A Scorer for Unseen Agents}
\label{sec:invariant}

Most of the scorer's 92 features are indexed by agent identity (the per-agent terms, the
per-agent rank bands, the support-pattern lookup), so it is undefined for agents it was not
trained on. For a new pool we replace every identity-indexed feature with one that means the
same in any pool, leaving 17: support as a fraction of the pool and of the question's largest
support; the minimum and mean of the asserters' log and relative positions; the asserters'
mean log output length; the silent agents' mean and maximum log output length; the
question's candidate count and mean support; and five terms that measure position and output
length against each agent's own median output length. The medians come from the new agents'
unlabelled outputs, so position 12 can read as deep for a terse agent and shallow for a
verbose one, which is what the identity features otherwise learn. The keep budget is
unchanged. The scorer is fitted on labelled questions answered by other agents; nothing about
the new agents requires a label.
""")

# ---- 14. budget ablation, after the CWQ paragraph in §4.1
rep(r"""Across both list-answering benchmarks the strongest rule expressible from agent identity fails to beat counting significantly, while the generation signals do.
""", r"""Across both list-answering benchmarks the strongest rule expressible from agent identity fails to beat counting significantly, while the generation signals do.

\begin{table}[t]
\centering
\small
\caption{Budget ablation: gain in set F1 over counting when each scorer is given the same per-question keep budget (Section~\ref{sec:slotB}). Same folds throughout; $^*$ marks a 95\% paired-bootstrap interval excluding zero.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Scorer & QAMPARI & CWQ & WebQSP \\
\midrule
Counting + budget & $-0.90$ & $+0.24$ & $-0.51$ \\
Weighting + budget & $-1.13^*$ & $-0.28$ & $-0.92^*$ \\
Pattern lookup + budget & $+1.39^*$ & $+0.59^*$ & $+0.31$ \\
Behavioural, no budget & $+1.47^*$ & $+1.65^*$ & $+0.54$ \\
\textbf{Behavioural + budget} & $\mathbf{+2.33^*}$ & $\mathbf{+1.55^*}$ & $\mathbf{+1.03^*}$ \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:ablation} separates the two stages. The budget alone does not produce the gain: counting with the budget does no better than counting, and reliability weighting with it is significantly worse on QAMPARI and WebQSP. The behavioural scorer carries most of the gain, beating counting by $+1.47$ on QAMPARI and $+1.65$ on CWQ before any budget is applied. The budget then adds $+0.86$ on QAMPARI and $+0.49$ on WebQSP, and nothing on CWQ, where the scorer alone is $0.10$ higher. It also lifts the pattern lookup on QAMPARI, by $+1.39$ over counting.
""")

# ---- 13b. results subsection, before the ASQA subsection
rep(r"""Appendix~\ref{app:negative} gives that ablation in full.
\subsection{ASQA Results: The Reliability Weighting Advantage}""",
r"""Appendix~\ref{app:negative} gives that ablation in full.

\subsection{Unseen Agents Without Their Labels}
\label{sec:transfer}

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Transfer to unseen agents on QAMPARI, set F1 averaged over splits. The scorer of Section~\ref{sec:invariant} and its budget are fitted on one group of agents and applied to a disjoint group on disjoint questions. \emph{Count} fits its threshold on the source group, as ours does; \emph{Count$^\dagger$} and \emph{Full$^\dagger$} (the 92-feature scorer) are fitted on the target group's own labels. Sig.: splits in which ours beats Count with a 95\% bootstrap interval excluding zero.}
\label{tab:transfer}
\begin{tabular}{lcccccc}
\toprule
Agents & Ours & Count & Gain & Sig. & Count$^\dagger$ & Full$^\dagger$ \\
\midrule
$4 \to 4$ & 27.63 & 22.72 & $+4.91$ & 70/70 & 23.94 & 28.25 \\
$3 \to 5$ & 28.14 & 23.58 & $+4.57$ & 46/56 & 26.19 & 29.33 \\
$5 \to 3$ & 26.29 & 19.83 & $+6.46$ & 56/56 & 22.57 & 26.66 \\
\bottomrule
\end{tabular}
\end{table}

We split the eight QAMPARI agents into two disjoint groups and the questions into two halves. The scorer of Section~\ref{sec:invariant} is fitted on one group's answers to one half and applied to the other group's answers to the other half, so source and target share no agent and no question, and no label of the target group is read. Table~\ref{tab:transfer} covers every split into groups of four (35 splits, both directions) and into groups of three and five. With four agents on each side the transferred scorer beats counting in all 70 splits, by $+4.91$ on average and significantly in every one. It also beats counting whose threshold is fitted on the target agents' own labels, in all 70 splits, and it is within 0.4 to 1.2 points of the full scorer trained on those labels. The three-to-five direction is the weakest: significant in 46 of 56 splits, with one significant loss.
\subsection{ASQA Results: The Reliability Weighting Advantage}""")

# ---- Limitations: cross-dataset sentence, appended after the sampling sentence
rep(r"""we do not measure sensitivity to temperature, to reasoning effort, or to repeated sampling.""",
r"""we do not measure sensitivity to temperature, to reasoning effort, or to repeated sampling. The scorer for unseen agents (Section~\ref{sec:transfer}) transfers across agent pools, not across datasets: fitted on QAMPARI it beats counting on CWQ but loses on WebQSP, whose gold sets are far larger.""")

p.write_text(s); print("ok")
