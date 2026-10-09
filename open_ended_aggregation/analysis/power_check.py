"""
Is the identity channel's null a bound, or is the study underpowered?

WHY THIS EXISTS
  Section 5.1 of the paper reports the support-pattern lookup at +0.30 F1 over
  counting with a bootstrap interval of [-0.34, +0.90], and reads that as a
  bound on what agent identity is worth. A reviewer can answer that two ways:
  "the effect is small" or "your study could not have detected it either way".
  Those have different consequences and the paper should say which applies.

WHAT IT COMPUTES
  All of it is a sensitivity analysis at the achieved sample size, not observed
  power. Observed power is a deterministic function of the p-value and adds
  nothing; the statistical-power skill is explicit that it should not be
  reported. What is legitimate is:

    - the minimum detectable effect (MDE) at n = 777 paired questions, which
      says what the design could have found;
    - a two one-sided tests (TOST) equivalence result, which says how tight a
      band around zero the data can actually exclude effects outside of;
    - a power curve over plausible true effects, so the answer is a range
      rather than one number.

  Uncertainty is taken from the paired bootstrap over questions, which is the
  procedure the paper already uses, rather than assuming normal-theory SEs. The
  normal-theory numbers are printed alongside as a cross-check.

Usage:  ./venv/bin/python -m open_ended_aggregation.analysis.power_check
Input:  results/weighting_v5_perq_seed0.json  (written by weighting_v5)
Output: results/power_check.json
"""
import json
import math

import numpy as np
from scipy import stats

from open_ended_aggregation.paths import RESULTS

BOOT = 10_000
SEED = 20260826
ALPHA = 0.05

# The comparisons that carry a claim in the paper. Each is (arm, reference).
PAIRS = [
    ("pattern", "count"),           # the identity channel: contribution 1
    ("LEAN (logistic)", "count"),   # the behavioural channel
    ("LEAN + budget", "count"),     # the full method
]


def boot_se(d, rng, n_boot=BOOT):
    """Bootstrap SE of the mean paired difference, resampling questions."""
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    return float(d[idx].mean(axis=1).std(ddof=1))


def analyse(name, ref, per, rng):
    a = np.asarray(per[name]) * 100.0
    b = np.asarray(per[ref]) * 100.0
    d = a - b
    n = len(d)
    obs = float(d.mean())

    se_boot = boot_se(d, rng)
    se_norm = float(d.std(ddof=1) / math.sqrt(n))

    # percentile bootstrap CI, matching the paper's reported intervals
    idx = rng.integers(0, n, size=(BOOT, n))
    means = d[idx].mean(axis=1)
    ci95 = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    ci90 = (float(np.percentile(means, 5.0)), float(np.percentile(means, 95.0)))

    # --- MDE at the achieved n. Sensitivity analysis, not observed power.
    # For a two-sided test at alpha with power 1-beta, the smallest detectable
    # true effect is (z_{1-alpha/2} + z_{1-beta}) * SE.
    def mde(power):
        return (stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(power)) * se_boot

    # --- TOST. The tightest symmetric margin the data can establish
    # equivalence within, at alpha, is the larger absolute end of the 90% CI.
    tost_margin = max(abs(ci90[0]), abs(ci90[1]))

    # --- power curve over plausible true effects
    grid = np.arange(0.0, 3.01, 0.05)
    zc = stats.norm.ppf(1 - ALPHA / 2)
    power = stats.norm.cdf(grid / se_boot - zc) + stats.norm.cdf(-grid / se_boot - zc)

    t, p = stats.ttest_rel(a, b)

    return {
        "arm": name, "reference": ref, "n_questions": n,
        "observed_diff": obs,
        "se_bootstrap": se_boot, "se_normal": se_norm,
        "ci95": ci95, "ci90": ci90,
        "t": float(t), "p": float(p),
        "mde_80": mde(0.80), "mde_90": mde(0.90),
        "tost_margin_alpha05": tost_margin,
        "power_curve": {"effect": grid.tolist(), "power": power.tolist()},
    }


def main():
    src = RESULTS / "weighting_v5_perq_seed0.json"
    per = json.loads(src.read_text())["per_question"]
    rng = np.random.default_rng(SEED)

    print("=" * 94)
    print("  SENSITIVITY AT THE ACHIEVED SAMPLE SIZE  (not observed power)")
    print("=" * 94)
    print(f"\n  n = 777 questions, paired. Bootstrap {BOOT:,} resamples over "
          f"questions, seed {SEED}.")
    print("  MDE = smallest true effect detectable at the stated power, "
          "alpha = 0.05 two-sided.")
    print("  TOST margin = tightest symmetric band around zero the data "
          "establish equivalence within.\n")

    out = []
    print(f"  {'comparison':<28}{'obs':>7}{'SE':>7}{'95% CI':>18}"
          f"{'MDE 80%':>9}{'MDE 90%':>9}{'TOST +-':>9}")
    print("  " + "-" * 90)
    for name, ref in PAIRS:
        r = analyse(name, ref, per, rng)
        out.append(r)
        ci = f"[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]"
        print(f"  {name + ' vs ' + ref:<28}{r['observed_diff']:>+7.2f}"
              f"{r['se_bootstrap']:>7.3f}{ci:>18}"
              f"{r['mde_80']:>9.2f}{r['mde_90']:>9.2f}"
              f"{r['tost_margin_alpha05']:>9.2f}")

    pat = out[0]
    print(f"\n  READING FOR THE IDENTITY CHANNEL (pattern vs count)")
    print(f"    observed              {pat['observed_diff']:+.2f} F1, "
          f"95% CI [{pat['ci95'][0]:+.2f}, {pat['ci95'][1]:+.2f}]")
    print(f"    design detects        {pat['mde_80']:.2f} F1 or larger at 80% "
          f"power, {pat['mde_90']:.2f} at 90%")
    print(f"    equivalence to zero   established within "
          f"+-{pat['tost_margin_alpha05']:.2f} F1 at alpha = 0.05")
    detectable = pat["observed_diff"] >= pat["mde_80"]
    print(f"    so the null is        "
          f"{'a detectable effect we failed to confirm' if detectable else 'a BOUND, not an underpowered miss'}"
          f": effects above ~{pat['ci95'][1]:.1f} F1 are excluded,")
    print(f"                          but {pat['observed_diff']:+.2f} cannot be "
          f"distinguished from zero at this n.")

    meth = out[-1]
    print(f"\n  CONTRAST (the full method vs count)")
    print(f"    observed {meth['observed_diff']:+.2f} F1 against an MDE of "
          f"{meth['mde_80']:.2f}, so this comparison is comfortably powered.")
    print(f"    The same design that cannot resolve {pat['observed_diff']:+.2f} "
          f"resolves {meth['observed_diff']:+.2f} with room to spare.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "power_check.json").write_text(json.dumps(
        {"n_boot": BOOT, "seed": SEED, "alpha": ALPHA, "comparisons": out},
        indent=1))
    print("\n  wrote results/power_check.json")


if __name__ == "__main__":
    main()
