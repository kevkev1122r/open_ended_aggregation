"""NeurIPS source -> ACL (ARR review) source. Formatting only: every change is
listed here and nothing else in the file is touched."""
import re, sys

src = open("neurips_source.tex", encoding="utf-8").read()
out = src

# 1. preamble: NeurIPS class/style/workshop block -> ACL review template
start = out.index("\\documentclass{article}")
end = out.index("\\workshoptitle{FAST: Foundations of Agentic Systems Theory}") + len(
    "\\workshoptitle{FAST: Foundations of Agentic Systems Theory}")
acl_head = (
    "\\documentclass[11pt]{article}\n"
    "\n"
    "% ARR (ACL Rolling Review) submission. Converted from neurips_source.tex by\n"
    "% convert_to_acl.py: formatting changes only, no content changes.\n"
    "\\usepackage[review]{acl}   % review mode: anonymous, line numbers\n"
    "\\usepackage{times}\n"
    "\\usepackage{latexsym}")
out = out[:start] + acl_head + out[end:]

# 2. acl.sty loads hyperref itself
out = out.replace("\\usepackage{hyperref}       % hyperlinks\n", "")

# 3. template's typewriter font; keep the NeurIPS-only ack environment suppressed
out = out.replace(
    "\\usepackage{microtype}      % microtypography\n",
    "\\usepackage{microtype}      % microtypography\n"
    "\\usepackage{inconsolata}    % typewriter font (ACL template)\n")
out = out.replace(
    "\\usepackage{algpseudocode}  % pseudocode body\n",
    "\\usepackage{algpseudocode}  % pseudocode body\n"
    "\\usepackage{comment}        % ack is NeurIPS-only; suppressed as in the anonymized NeurIPS build\n"
    "\\excludecomment{ack}\n")

# 4. ACL is author-year: parenthetical citations need \citep (comment lines untouched)
lines = out.split("\n")
lines = [l if l.lstrip().startswith("%") else l.replace("\\cite{", "\\citep{") for l in lines]
out = "\n".join(lines)

# 5. bibliography style
out = out.replace("\\bibliographystyle{plainnat}", "\\bibliographystyle{acl_natbib}")
out = out.replace("{\\small\\bibliography{refs}}", "\\bibliography{refs}")

# 6. NeurIPS checklist is not part of an ARR paper (ARR checklist is on OpenReview)
out = out.replace("\\input{checklist.tex}",
                  "% \\input{checklist.tex}  % NeurIPS checklist: not used by ARR")

# 7. two-column layout: tables and figures span both columns
out = re.sub(r"\\begin\{table\}(\[[^\]]*\])?", r"\\begin{table*}[t]", out)
out = out.replace("\\end{table}", "\\end{table*}")
out = re.sub(r"\\begin\{figure\}(\[[^\]]*\])?", r"\\begin{figure*}[t]", out)
out = out.replace("\\end{figure}", "\\end{figure*}")

open("acl_main.tex", "w", encoding="utf-8").write(out)
print("wrote acl_main.tex")
