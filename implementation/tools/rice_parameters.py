#!/usr/bin/env python3
"""Golomb-Rice parameters for discrete Gaussians: the derivation behind the rule
of the profiles (codec version 3).

A Golomb-Rice codeword of an integer x with parameter k is k low bits of |x|,
|x| >> k in unary (that many ones and a terminating zero), and a sign bit if
x != 0.  For X ~ D_{Z, sigma} its expected length is

    L(sigma, k) = k + 1 + E[floor(|X| / 2^k)] + Pr[X != 0].

The implementations use the rule k(sigma) = max{k >= 0 : 1.2863 * 2^k <= sigma}.
This script

  1. finds, for large k, the width at which L(., k) = L(., k + 1), in units of
     2^k; it converges to 2.5726 = 2 * 1.2863, which gives the rule;
  2. compares, octave by octave, the rule's switch point 2.5726 * 2^k with the
     optimal one: between the two the rule picks the other parameter.  The loss
     L(k) - L(k + 1) is zero at the optimal switch point and largest at the
     rule's, where it is evaluated.  The rule is a large-width rule: at
     sigma = 2 it picks k = 0 although k = 1 is shorter;
  3. reports the largest excess of L(sigma, k(sigma)) over the entropy of
     D_{Z, sigma}, for sigma >= 8 and for sigma >= 2^15, evaluating on a grid
     and just below every switch point of the rule;
  4. for every stored profile, models each Gaussian component (a share
     coefficient, a partial-response coefficient) as D_{Z, sigma} with the
     component's variance, and reports whether the stored parameter minimizes
     the modelled L and how far, in lg sigma, the width is from the nearest
     optimal switch point.  This is a statement about the model; a share
     coefficient is a sum of independent discrete Gaussians, not exactly one.

L is computed by summation over the distribution for sigma <= 2^12 and with the
continuity-corrected normal tail above; the difference between the two at
sigma = 2^12 is reported.  Only the standard library is used.

Usage:  python3 tools/rice_parameters.py [--profiles DIR]
"""

import argparse
import glob
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULE_C = 1.2863
EXACT_LIMIT = 2 ** 12


def gaussian_pmf(sigma):
    """Pr[|X| = a] through about 14 sigma (normalized)."""
    top = int(14 * sigma) + 2
    rho = [math.exp(-a * a / (2 * sigma * sigma)) for a in range(top)]
    total = rho[0] + 2 * sum(rho[1:])
    return [rho[0] / total] + [2 * r / total for r in rho[1:]]


def expected_length_exact(pmf, k):
    return k + 1 + sum(p * ((a >> k) + (1 if a else 0)) for a, p in enumerate(pmf))


def tail(sigma, a):
    """Pr[|X| >= a] for integer a >= 1, continuity-corrected normal tail."""
    return math.erfc((a - 0.5) / (sigma * math.sqrt(2)))


def expected_length_tail(sigma, k):
    m = 1 << k
    total, j = 0.0, 1
    while True:
        t = tail(sigma, j * m)
        total += t
        if t < 1e-18:
            break
        j += 1
    return k + 1 + total + tail(sigma, 1)


def expected_length(sigma, k, pmf=None):
    if sigma <= EXACT_LIMIT:
        return expected_length_exact(pmf if pmf is not None else gaussian_pmf(sigma), k)
    return expected_length_tail(sigma, k)


def entropy(sigma, pmf=None):
    if sigma <= EXACT_LIMIT:
        pmf = pmf if pmf is not None else gaussian_pmf(sigma)
        # pmf is of |X|; split the mass of a >= 1 over +a and -a.
        h = -pmf[0] * math.log2(pmf[0])
        h -= sum(p * math.log2(p / 2) for p in pmf[1:] if p > 0)
        return h
    return 0.5 * math.log2(2 * math.pi * math.e * sigma * sigma)


def best_k(sigma, pmf=None):
    centre = max(0, int(math.log2(sigma)))
    candidates = range(max(0, centre - 3), centre + 4)
    return min(candidates, key=lambda k: (expected_length(sigma, k, pmf), k))


def rule_k(sigma):
    k = 0
    while RULE_C * 2 ** (k + 1) <= sigma:
        k += 1
    return k


def boundary(k):
    """The width in (2^k, 2^(k+2)) at which L(., k) = L(., k + 1), by bisection."""
    lo, hi = 2.0 ** k, 2.0 ** (k + 2)
    for _ in range(60):
        mid = (lo + hi) / 2
        if expected_length(mid, k + 1) < expected_length(mid, k):
            hi = mid
        else:
            lo = mid
    return hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profiles", default=os.path.join(REPO, "profiles"))
    args = ap.parse_args()
    failures = 0

    print("1. Where k and k + 1 give equal expected length, in units of 2^k:")
    for k in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 45):
        print(f"   k = {k:2d}: sigma = {boundary(k) / 2 ** k:.6f} * 2^k")
    print(f"   the rule switches at {2 * RULE_C:.4f} * 2^k")

    print("2. Octave by octave: the optimal switch point, the rule's, and the loss at the rule's:")
    largest_loss = {}
    for k in range(0, 17):
        exact, rule = boundary(k), 2 * RULE_C * 2 ** k
        # Just inside the mismatch interval at the rule's end, where the loss peaks.
        sigma = rule * (1 - 1e-12) if rule > exact else rule * (1 + 1e-12)
        pmf = gaussian_pmf(sigma) if sigma <= EXACT_LIMIT else None
        loss = abs(expected_length(sigma, k, pmf) - expected_length(sigma, k + 1, pmf))
        largest_loss[k] = loss
        print(f"   k = {k:2d} -> {k + 1:2d}: optimal {exact / 2 ** k:.6f} * 2^k, rule {rule / 2 ** k:.6f} * 2^k, "
              f"loss at the rule's switch {loss:.4f} bits")
    sigma0 = 8.0
    print(f"   for sigma >= {sigma0:g} (k >= 2) the rule loses at most "
          f"{math.ceil(1000 * max(v for k, v in largest_loss.items() if k >= 2)) / 1000:.3f} bits per coefficient")

    print("3. Excess of the rule's expected length over the entropy:")

    def excess(sigma):
        pmf = gaussian_pmf(sigma) if sigma <= EXACT_LIMIT else None
        return expected_length(sigma, rule_k(sigma), pmf) - entropy(sigma, pmf)

    points = [sigma0 * 1.02 ** i for i in range(int(math.log(2 ** 20 / sigma0) / math.log(1.02)) + 1)]
    points += [2 * RULE_C * 2 ** k * (1 - 1e-12) for k in range(2, 60)]
    points += [2.0 ** (45 + i / 100) for i in range(100)]
    worst = max((excess(x), x) for x in points)
    wide = max((excess(x), x) for x in points if x >= 2 ** 15)
    print(f"   sigma >= 8: largest excess {worst[0]:.4f} bits per coefficient (at sigma = {worst[1]:.4g})")
    print(f"   sigma >= 2^15 (the share and response widths of the author profiles): "
          f"largest excess {wide[0]:.4f} bits per coefficient")
    s = float(EXACT_LIMIT)
    print(f"   summation vs tail model at sigma = 2^12: "
          f"{abs(expected_length_exact(gaussian_pmf(s), 12) - expected_length_tail(s, 12)):.2e} bits")

    print("4. Stored profiles: does the stored parameter minimize the modelled length, and how far")
    print("   (in lg sigma) is each width from the nearest optimal switch point?")
    switch = {k: boundary(k) for k in range(0, 64)}
    closest = float("inf")
    for path in sorted(glob.glob(os.path.join(args.profiles, "*.json"))):
        with open(path) as f:
            p = json.load(f)
        g = p["gaussian"]
        dy = {key: int(g[key]["num"]) / 2 ** g[key]["frac_bits"] for key in ("sigma_s", "sigma_big_r", "sigma_r")}
        c_exp, beta_g, omega_c = p["sec"]["c_exp"], p["sec"]["beta_g"], p["challenge"]["omega_c"]
        share = [math.sqrt(j * j * dy["sigma_s"] ** 2 + c_exp * dy["sigma_big_r"] ** 2) for j in range(beta_g + 1)]
        response = math.sqrt(dy["sigma_r"] ** 2 + omega_c * share[-1] ** 2)
        widths = share + [response]
        stored = p["encoding"]["rice_share"] + [p["encoding"]["rice_response"]]
        modelled = [best_k(w) for w in widths]
        margins = [min(abs(math.log2(w) - math.log2(b)) for b in switch.values()) for w in widths]
        closest = min(closest, min(margins))
        ok = modelled == stored and min(widths) >= sigma0
        failures += not ok
        print(f"   {'ok  ' if ok else 'FAIL'} {p['id']}: widths "
              + ", ".join(f"2^{math.log2(w):.2f}" for w in widths)
              + f"; stored k {stored}, modelled optimum {modelled}; margins "
              + ", ".join(f"{m:.2f}" for m in margins))
    print(f"   the closest width is {closest:.3f} in lg sigma ({100 * (2 ** closest - 1):.1f}%) from a switch point")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
