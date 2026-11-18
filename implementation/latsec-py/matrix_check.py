"""matrix_check.py -- structural checks of an SEC matrix (Section 2.4 of the paper).

For a matrix fixture fixtures/author/G-*.json (G = [g | G_hat], one row per
party) this rechecks, exactly and exhaustively:

  * rank: G has full column rank k over Z;
  * k_p = ceil(2 (1 - eps) floor(gamma n)) - 1, the privacy threshold of [46];
  * expansion: |N(S)| >= ceil((1 - eps) c_exp |S|) for every set S of at most
    floor(gamma n) parties, N(S) the columns of G_hat that meet S;
  * peeling ([46, Algorithm 3]): on every set U of at most k_p parties, taking
    the smallest column that meets exactly one remaining row, and removing that
    row, empties U.  This certifies the unimodular minor M_U of Item 5;
  * the planted witnesses: lambda^T G_U = e_1^T on the fixture's planted sets.

With --spectral it also evaluates the float64 matrix constants (Table 3 of
the paper) by SVD over every set of at most k_p parties, as the SEC-geometry
notebooks do.  Those are floating-point values, not certified bounds.  This
needs numpy, and is slow in Python beyond n = 40; latsec-rs is the practical
tool for n = 60.

Usage:  python3 matrix_check.py ../fixtures/author/G-40-19-3.json [--spectral]
"""

import argparse
import json
import sys
import time
from fractions import Fraction
from math import ceil

import sec


def parse_ratio(text):
    return Fraction(text)


def privacy_threshold(n, gamma, eps):
    return ceil(2 * (1 - eps) * (gamma * n // 1)) - 1


def neighbor_masks(g):
    """One bitmask per party: the columns j of G_hat with G_hat[i][j] != 0."""
    return [sum(1 << j for j, x in enumerate(row[1:]) if x) for row in g]


def check_expansion(masks, c_exp, gamma, eps):
    """Returns (subsets checked, a violating set or None)."""
    n = len(masks)
    max_size = int(gamma * n)
    need = [ceil((1 - eps) * c_exp * s) for s in range(max_size + 1)]
    checked = 0

    def recurse(start, size, union, chosen):
        nonlocal checked
        for nxt in range(start, n):
            merged = union | masks[nxt]
            checked += 1
            if bin(merged).count("1") < need[size + 1]:
                return chosen + [nxt]
            if size + 1 < max_size:
                bad = recurse(nxt + 1, size + 1, merged, chosen + [nxt])
                if bad:
                    return bad
        return None

    violation = recurse(0, 0, 0, []) if max_size > 0 else None
    return max_size, checked, violation


def peel(masks, subset):
    """[46, Algorithm 3]: the peeling columns in order, or None if stuck."""
    rows = [masks[i] for i in subset]
    columns = []
    while rows:
        once = more = 0
        for mask in rows:
            more |= once & mask
            once |= mask
        unique = once & ~more
        if unique == 0:
            return None
        column = (unique & -unique).bit_length() - 1
        rows.pop(next(p for p, mask in enumerate(rows) if mask >> column & 1))
        columns.append(column)
    return columns


def check_peeling(masks, k_p):
    checked = 0

    def recurse(start, chosen):
        nonlocal checked
        for nxt in range(start, len(masks)):
            subset = chosen + [nxt]
            checked += 1
            if peel(masks, subset) is None:
                return subset
            if len(subset) < k_p:
                bad = recurse(nxt + 1, subset)
                if bad:
                    return bad
        return None

    stuck = recurse(0, [])
    return checked, stuck


def spectral_constants(g, k_p):
    """alpha_kp, beta_kp, delta_kp, gamma_kp, |G_hat|_2, |G|_2 and
    beta_Q = 1 + sqrt(1 + delta_kp^2) / gamma_kp, by float64 SVD."""
    import numpy as np
    from itertools import combinations
    G = np.array(g, dtype=float)
    g_hat = G[:, 1:]
    masks = neighbor_masks(g)
    alpha, beta, delta, gamma = float("inf"), 0.0, 0.0, float("inf")
    for size in range(1, k_p + 1):
        for subset in combinations(range(len(g)), size):
            columns = peel(masks, subset)
            if columns is None:
                raise ValueError(f"peeling is stuck on {subset}")
            minor = g_hat[np.ix_(subset, columns)]
            gamma = min(gamma, np.linalg.svd(minor, compute_uv=False).min())
            if size == k_p:
                s = np.linalg.svd(g_hat[list(subset)], compute_uv=False)
                alpha, beta = min(alpha, s.min()), max(beta, s.max())
                delta = max(delta, np.linalg.svd(G[list(subset)], compute_uv=False).max())
    norm_g_hat = np.linalg.svd(g_hat, compute_uv=False).max()
    norm_g = np.linalg.svd(G, compute_uv=False).max()
    return dict(alpha_kp=alpha, beta_kp=beta, delta_kp=delta, gamma_kp=gamma, norm_g_hat=norm_g_hat,
                norm_g=norm_g, beta_q=1 + (1 + delta * delta) ** 0.5 / gamma)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("matrix")
    ap.add_argument("--spectral", action="store_true")
    args = ap.parse_args()
    with open(args.matrix) as f:
        m = json.load(f)
    g, n, k, k_p, c_exp = m["g"], m["n"], m["k"], m["k_p"], m["c_exp"]
    gamma, eps = parse_ratio(m["gamma"]), parse_ratio(m["eps_exp"])
    ok = True
    print(f"matrix {m['name']} ({n} x {k}), c_exp = {c_exp}, gamma = {m['gamma']}, eps = {m['eps_exp']}")
    _, _, pivots = sec.row_hnf(g)
    print(f"  rank over Z: {len(pivots)} (k = {k})")
    ok &= len(pivots) == k
    kp_formula = privacy_threshold(n, gamma, eps)
    print(f"  k_p from the paper's formula: {kp_formula} (fixture: {k_p})")
    ok &= kp_formula == k_p
    masks = neighbor_masks(g)
    started = time.time()
    max_size, checked, violation = check_expansion(masks, c_exp, gamma, eps)
    print(f"  expansion: {checked} subsets of size <= {max_size} in {time.time() - started:.1f} s: "
          + ("certified" if violation is None else f"FAILS on {violation}"))
    ok &= violation is None
    started = time.time()
    checked, stuck = check_peeling(masks, k_p)
    print(f"  peeling ([46, Algorithm 3]): {checked} subsets of size <= {k_p} in {time.time() - started:.1f} s: "
          + ("all succeed" if stuck is None else f"STUCK on {stuck}"))
    ok &= stuck is None
    for support, witness in zip(m["planted_supports"], m["planted_witnesses"]):
        good = sec.is_reconstruction_vector(g, support, witness)
        print(f"  planted support of {len(support)} parties, witness weight {sum(1 for x in witness if x)}: "
              + ("reconstructs" if good else "DOES NOT reconstruct"))
        ok &= good
    if args.spectral:
        started = time.time()
        s = spectral_constants(g, k_p)
        print(f"  float64 constants in {time.time() - started:.1f} s (not certified):")
        print("    " + "  ".join(f"{key} = {value:.15g}" for key, value in s.items()))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
