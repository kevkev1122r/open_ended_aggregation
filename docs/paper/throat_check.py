#!/usr/bin/env python3
"""
Detect LLM throat clearing in a LaTeX manuscript.

Throat clearing is text that delays the claim instead of making it. This script
finds the eight constructions that produced ~60 sites in iteration3.tex, plus a
caption check, and reports line numbers so each can be judged individually.

It is a *detector*, not a judge. Every hit needs the delete test (see
docs/THROAT_CLEARING.md): remove the span and ask whether any fact, number or
caveat is lost. If nothing is lost, it was throat clearing. Some hits are
load-bearing and must stay -- the script cannot tell which, you can.

Usage:
    python3 throat_check.py iteration3.tex
    python3 throat_check.py iteration3.tex --quiet     # counts only
"""
import argparse
import pathlib
import re
import sys

# ---------------------------------------------------------------- patterns ---
# Each entry: (name, regex, what it usually indicates)
PATTERNS = [
    ("caption-takeaway",
     r"The takeaway is that|The upshot is|What this shows is",
     "a caption that interprets instead of describing"),

    ("announces-the-writing",
     r"\b[Ww]e (state|present|report|note|observe|mention) (it|this|these|them|that)\b"
     r"|rather than leave (a reader|a reviewer|it) to"
     r"|does and does not establish"
     r"|\b[Ww]e report (it|this|these|them) as\b"
     r"|for completeness|collected here|repeated here"
     r"|A paper that (stopped|ended)",
     "a sentence about the paper rather than about the world"),

    ("enumerative-announcement",
     r"^\s*(Two|Three|Four|Five) (things|readings|consequences|statements|framing|"
     r"lines|points|questions|reasons|observations)\b"
     r"|\b(Two|Three|Four) (things|questions|consequences|readings) (follow|bear|occur|matter)\b"
     r"|is useful for (two|three) (other )?reasons",
     "counting the items before delivering them"),

    ("preemptive-objection",
     r"invites the objection|answers a specific objection|One might object"
     r"|A (reader|reviewer) (might|may|could) (object|ask|wonder)"
     r"|lest (a|the) reader",
     "staging an objection so it can be knocked down"),

    ("significance-announcement",
     r"\bThis matters\b|\b[Ii]t is worth (stating|noting|saying)\b|worth stating here"
     r"|\bThe (takeaway|measurement|operative property|content|point|key insight) is\b"
     r"|Importantly,|Notably,|Crucially,|It should be noted",
     "telling the reader something is important instead of showing it"),

    ("staged-contrast",
     r"\brather than a (theorem|caveat|target|law|measurement|counterexample|"
     r"contribution|truncation|failure|null|formality)\b"
     r"|\bnot (a|an|the) \w+ (but|so much as)\b"
     r"|\bis not X, it is\b",
     "an X-rather-than-Y frame where the contrast does no work"),

    ("defensive-qualification",
     r"\bwe (do not|make no|would not) claim\b|\b[Ww]e are not claiming\b"
     r"|\bwe do not (wish|want) to (suggest|imply)\b"
     r"|\bto be clear\b|\bthat said\b",
     "performing modesty instead of stating the limit"),

    ("aphoristic-closer",
     r"\bis what pays\b|\bgets nothing back\b|\bhad been discarding\b"
     r"|\btells you more than\b|\bis the whole (point|story|game)\b"
     r"|\bmeasured (precisely|badly)\b",
     "a flourish placed to land a paragraph"),
]

CAPTION_RE = re.compile(r"\\caption\{", re.M)
COMMENT_RE = re.compile(r"^\s*%")


def scan(path, quiet=False):
    text = path.read_text(encoding="utf-8")
    try:
        body = text[text.index(r"\begin{document}"):]
        offset = text[:text.index(r"\begin{document}")].count("\n")
    except ValueError:
        body, offset = text, 0

    lines = body.split("\n")
    total = 0
    print(f"\n{path.name}\n" + "=" * len(path.name))

    for name, pat, why in PATTERNS:
        rx = re.compile(pat)
        hits = []
        for i, line in enumerate(lines):
            if COMMENT_RE.match(line):
                continue
            for m in rx.finditer(line):
                a = max(0, m.start() - 55)
                hits.append((i + 1 + offset, line[a:m.end() + 55].strip()))
        total += len(hits)
        if hits or not quiet:
            print(f"\n{name}  ({len(hits)})")
            print(f"  -- {why}")
        if not quiet:
            for ln, ctx in hits:
                print(f"  L{ln}: ...{ctx}")

    # captions carrying an interpretive final sentence
    caps = CAPTION_RE.findall(body)
    takeaways = len(re.findall(r"\\caption\{[^}]*The takeaway is that", body, re.S))
    print(f"\ncaptions  ({len(caps)} total, {takeaways} with an explicit takeaway sentence)")
    print("  -- a caption should say what is plotted; the claim belongs in the body")

    print(f"\nTOTAL flagged spans: {total}")
    print("\nEvery hit needs the delete test. Keep the ones that carry a fact,")
    print("a number, a caveat, or a distinction the reader cannot recover.")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=pathlib.Path)
    ap.add_argument("--quiet", action="store_true", help="counts only, no contexts")
    a = ap.parse_args()
    worst = 0
    for p in a.paths:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            continue
        worst = max(worst, scan(p, a.quiet))
    return 0


if __name__ == "__main__":
    sys.exit(main())
