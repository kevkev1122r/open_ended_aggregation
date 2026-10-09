# Throat clearing: detect it, remove it, stop writing it

Written 28 Aug 2026 from an actual cleanup of `iteration3.tex`, which went from
80 flagged spans to 8. Every example below is a real before/after from that pass.

Run the detector first:

```bash
python3 docs/paper/throat_check.py docs/paper/iteration3.tex
```

---

## 1. What it is

**Throat clearing is text that delays the claim instead of making it.**

It is not the same as wordiness. Wordy prose says the thing slowly. Throat
clearing says something *about* the thing — how important it is, how many parts
it has, how honestly it is being presented — and then says the thing.

It reads as machine-written because a model rewarded for sounding thorough
produces the frame around a claim as reliably as the claim.

---

## 2. Three tests that settle it

The detector finds candidates. These decide.

### The delete test — the only one you always need

Delete the span. Does the reader lose a fact, a number, a caveat, or a
distinction they cannot recover from the surrounding text?

- **No** → it was throat clearing. It is already gone; leave it gone.
- **Yes** → keep it, however formulaic it sounds.

> `We state the limits of this argument rather than leave a reviewer to find them.`
> Deleted. The limits that follow are unchanged; nothing was lost.

> `...the paper's pipeline rather than a truncation of it.`
> Kept. It forecloses a specific objection about the ASC baseline. Same shape,
> different verdict.

### The world test

Does the sentence describe **the world** or **the paper**?

Papers should describe the world. A sentence whose subject is your own writing —
what you are about to say, why you are saying it, how fairly you are saying it —
is almost always removable.

> `A paper that stopped here would have established nothing empirical.` → cut.
> `Three statements organise the rest of the paper.` → cut.

### The announce/deliver test

Does the sentence **announce** content or **deliver** it?

> `Three questions follow: what is left in the identity channel...`
> → `We ask what is left in the identity channel...`

The list was already there. Counting it first added a sentence and no content.

---

## 3. The eight patterns

Counts are from `iteration3.tex` before cleanup.

### 3.1 The caption formula — 17 of 20 captions

Every caption ended `The takeaway is that ...`. By the fourth, a reader knows the
last sentence before reading it.

> − `... Right: where the claims actually are. The takeaway is that the spread is large where the claims are not, reaching 59 points at |S(c)|=4 ...`
> + `... Right: where the claims are. The spread is largest where the claims are fewest — 59 points at |S(c)|=4 ...`

**Rule:** a caption says what is plotted and gives the number. If the caption's
last sentence would work as a body sentence, move it to the body.

### 3.2 X rather than Y, as the default sentence shape — 43 hits

The most repeated construction in the document, including three `\paragraph`
headings within 250 lines: *The null is a bound, not an underpowered miss.* /
*The decision slot is not the problem.* / *Three points are an observation, not a law.*

**Rule:** keep the contrast only when the reader would otherwise draw the wrong
conclusion. Cut it when it exists to sound precise.

### 3.3 Enumerative announcements — 7 hits

`Two things follow.` `Two readings matter.` `Two consequences are worth stating.`
`Three lines of work bear on this problem.`

**Rule:** never count items before listing them. The list is the count.

### 3.4 Preemptive-objection framing — 3 hits

The same template three times: *X invites the objection that Y, so we Z.*

> − `A single 8-agent result invites the objection that the pool was favourable, so we evaluate all 210 subsets.`
> + `To test whether the 8-agent pool was favourable, we evaluate all 210 subsets.`

**Rule:** state the purpose. Don't stage a critic in order to defeat them.

### 3.5 Significance announcements — 22 hits

`This matters because...`, `It is worth stating that...`, `The measurement is
that...`, `Importantly,`. Telling the reader something is important is weaker
than showing it.

**Rule:** delete the announcement, keep the content. If the content isn't
important enough to survive alone, cut it too.

### 3.6 Aphoristic closers — 5 hits

`They are not remotely the same evidence.` `Those are three different ways of
reaching the bound from below.` `This is a small effect measured precisely, not a
large one measured badly.`

**Rule:** at most one per paper, in the conclusion. Everywhere else the preceding
sentence already landed it.

### 3.7 Defensive self-qualification — 3 hits

`We do not claim the independence assumption holds.`

**Rule:** assert the limitation as a fact about the world.

> + `The independence assumption does not hold here: agents share training data...`

Stronger, shorter, and it commits.

### 3.8 Meta-commentary about the writing — 16 hits

The purest form. Includes the identical `\paragraph{What this does and does not
establish.}` used **twice**, 500 lines apart.

**Rule:** if the subject of the sentence is your manuscript, delete or rewrite.

---

## 4. How to eliminate it

1. **Run the detector.** `throat_check.py` on the file.
2. **Delete test each hit.** Most fall in seconds.
3. **Do captions as a batch.** They drift into a house formula; fixing them
   together is faster and keeps them varied.
4. **Relocate before deleting.** Some flagged text carries provenance that lives
   nowhere else. When stripping footnotes out of a figure, check whether the
   caption needs to absorb them first — a "pre-parsing-fix; pending regeneration"
   note is not throat clearing, it is a disclosure.
5. **Re-run the detector after every rewrite.** See §6.
6. **Verify nothing else moved.** Diff the numbers:

```bash
python3 - <<'PY'
import re
a=set(re.findall(r'[\d]+\.[\d]+|\b\d+\b', open('before.tex').read()))
b=set(re.findall(r'[\d]+\.[\d]+|\b\d+\b', open('after.tex').read()))
print('removed:', sorted(a-b) or 'none', '| added:', sorted(b-a) or 'none')
PY
```

A throat-clearing pass should move **zero** numbers. If one moves, you cut
content, not framing.

---

## 5. How not to write it in the first place

- **Write the claim, not the frame around the claim.** If a sentence's job is to
  prepare the reader for the next sentence, the next sentence is the paragraph.
- **Start paragraphs on content.** Not on "This matters", not on a count.
- **Captions describe; bodies claim.**
- **Assert limitations, don't perform them.** "X does not hold here" beats "we do
  not claim X holds."
- **Earn contrasts.** Write X-rather-than-Y only when someone would otherwise
  believe Y.
- **One flourish per paper.** Spend it in the conclusion.
- **Never write a sentence whose subject is your paper.**

---

## 6. Two things that will surprise you

**Rewriting reintroduces it.** After the first pass cleared all eight patterns, a
second scan found seven new sites — every one of them *introduced by my own
rewrites*: `Three questions follow:`, `Three mechanisms are worth stating here.`,
`That also bounds where it applies.` The instinct that produces throat clearing is
the same instinct that produces rewrites. **Scan again after every editing pass**,
including passes whose purpose was something else.

**The second pass is much smaller than you expect.** Pass one removed ~2,500
characters across ~60 sites. Pass two removed 77 characters across 7. Once a
document has been through the filter twice, the remaining hits are almost all
load-bearing — at which point the detector's value flips from finding problems to
confirming there aren't any.

---

## 7. What survives, and should

The 8 spans still flagged in `iteration3.tex` are all deliberate keeps. Use them
as a calibration set:

| span | why it stays |
|---|---|
| `...the paper's pipeline rather than a truncation of it` | forecloses a real objection about the ASC baseline |
| `...an expectation rather than a measurement` | draws the line between measured and assumed |
| `...a baseline here rather than a contribution` | disclaims credit for Dawid-Skene |
| `our identity-channel claim is empirical rather than a theorem` | scopes the central claim |
| `we observe this on ASQA and read it as a statement about data volume` | names an interpretive choice the reader must be able to disagree with |
| `The model pays for the parameters and gets nothing back` | vivid, but carries the finding |
| the two closing clauses of §7 | the one permitted flourish |

Each is a false positive **for this document**. A detector that produced zero
hits would be tuned too loosely.
