#!/usr/bin/env python3
"""Generate reconstruction-vector reference fixtures with SageMath.

Run with SageMath's Python from the directory of the implementation:

    sage -python tools/gen_recon_fixtures.py

With ``--bulk`` the script instead writes a larger reference file (sets and final
vectors only) for ``latsec recon-compare``, without touching the fixtures:

    sage -python tools/gen_recon_fixtures.py --bulk G-60-39-3 --sizes 46-60 \
        --count 400 --out /tmp/bulk-60.json

For a fixed, seeded selection of party sets of each author matrix in
``fixtures/author`` this writes ``fixtures/sage/recon-<n>-<m>-<c>.json`` with, per set:

* whether the set is authorized,
* the echelonized (row-HNF) basis of the kernel lattice,
* Sage's LLL-reduced kernel basis,
* the vector after exact Babai and the final reconstruction vector.

The functions below follow the SageMath pipeline of the SEC paper [46], as given in
its reconstruction-vector notebooks (``hnf_witness``, ``kernel_basis``,
``babai_exact``, ``greedy_reduce_linf``, ``construct_lambda``), with
``passes = window = 2``.  The script is self-contained: it needs SageMath and the
fixtures, not those notebooks.  The Rust tests use these files to
check authorization, the canonical kernel basis, and bit-exact agreement of Babai and
the greedy stage on the same reduced basis.  They do not require the Rust LLL to
reproduce Sage's basis.
"""

import argparse
import json
import os
import random
import sys
from itertools import combinations
from math import comb

from sage.all import QQ, ZZ, Matrix, vector
from sage.version import version as sage_version

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSES = 2
WINDOW = 2
SEED = 20261106

# name -> list of (size, number of sampled sets); None means all sets of that size.
SELECTION = {
    "G-10-5-3": [(s, None) for s in range(1, 11)],
    "G-20-9-3": [(s, 6) for s in range(10, 21)],
    "G-40-19-3": [(s, 4) for s in range(26, 41)],
    "G-60-39-3": [(s, 3) for s in range(46, 61)],
}
# Extra sets below the signing threshold, where unauthorized sets are common.
EXTRA = {"G-40-19-3": [(20, 6), (22, 6)], "G-60-39-3": [(40, 4), (43, 4)]}


# Sets found with `latsec recon-stats` (seed 20261106, 100,000 sets per size) whose
# reconstruction vector exceeds the maxima observed in [46]: (4, 24) for n = 40
# and (8, 42) for n = 60.  They are listed explicitly so that the SageMath pipeline's
# result for them is recorded.
NOTABLE = {
    "G-40-19-3": [
        [0, 1, 2, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 18, 19, 22, 23, 24, 26, 31, 32, 33, 34, 35, 37, 38],
        [0, 2, 5, 6, 7, 8, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 39],
        [0, 1, 2, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 25, 26, 28, 29, 30, 31, 32, 33, 35, 36, 38],
        [0, 2, 3, 5, 6, 7, 8, 9, 11, 15, 16, 18, 19, 20, 21, 22, 24, 26, 27, 28, 30, 32, 33, 34, 35, 37, 38],
        [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 26, 28, 30, 31, 32, 33, 37, 38, 39]
    ],
    "G-60-39-3": [
        [0, 2, 4, 5, 6, 7, 8, 10, 11, 14, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 51, 52, 53, 54, 55, 57, 58, 59],
        [2, 3, 4, 5, 8, 12, 13, 14, 15, 16, 17, 18, 20, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    ],
}


def linf_norm(v):
    return max((abs(ZZ(x)) for x in v), default=ZZ(0))


def hnf_witness(g_s):
    if g_s.nrows() == 0:
        return None, False
    r, u = g_s.hermite_form(transformation=True)
    if r.nrows() == 0:
        return None, False
    e1 = vector(ZZ, [1] + [0] * (g_s.ncols() - 1))
    if r.row(0) != e1:
        return None, False
    return vector(ZZ, list(u.row(0))), True


def babai_exact(basis, target):
    r = basis.nrows()
    if r == 0:
        return vector(ZZ, list(target))
    b = [vector(QQ, list(basis.row(i))) for i in range(r)]
    bs = []
    for i in range(r):
        v = b[i]
        for j in range(i):
            d = bs[j].dot_product(bs[j])
            if d != 0:
                v = v - (b[i].dot_product(bs[j]) / d) * bs[j]
        bs.append(v)
    w = vector(QQ, list(target))
    for i in reversed(range(r)):
        d = bs[i].dot_product(bs[i])
        if d == 0:
            continue
        c = w.dot_product(bs[i]) / d
        ch = (c + QQ(1) / 2).floor()
        if ch != 0:
            w = w - QQ(ch) * b[i]
    return vector(ZZ, [ZZ(x) for x in w])


def greedy_reduce_linf(lam0, basis, passes=PASSES, window=WINDOW):
    lam = vector(ZZ, list(lam0))
    best = linf_norm(lam)
    if basis.nrows() == 0:
        return lam
    for _ in range(int(passes)):
        improved = False
        for i in range(basis.nrows()):
            bi = vector(ZZ, list(basis.row(i)))
            j = max(range(len(lam)), key=lambda t: abs(lam[t]))
            if bi[j] == 0:
                continue
            a0 = (QQ(-lam[j]) / QQ(bi[j]) + QQ(1) / 2).floor()
            lam_best, norm_best = lam, best
            for a in range(int(a0 - window), int(a0 + window) + 1):
                cand = lam + ZZ(a) * bi
                cn = linf_norm(cand)
                if cn < norm_best:
                    lam_best, norm_best = cand, cn
            if norm_best < best:
                lam, best, improved = lam_best, norm_best, True
        if not improved:
            break
    return lam


def case_for(g, subset):
    g_s = g.matrix_from_rows(subset)
    lam0, ok = hnf_witness(g_s)
    case = {"set": [int(i) for i in subset], "authorized": bool(ok)}
    kernel_hnf = g_s.transpose().right_kernel(ZZ).echelonized_basis_matrix()
    case["kernel_hnf"] = [[str(x) for x in row] for row in kernel_hnf.rows()]
    if not ok:
        return case
    reduced = kernel_hnf.LLL() if kernel_hnf.nrows() > 1 else kernel_hnf
    case["kernel_lll"] = [[int(x) for x in row] for row in reduced.rows()]
    if reduced.nrows() == 0:
        babai = final = lam0
    else:
        babai = babai_exact(reduced, lam0)
        final = greedy_reduce_linf(babai, reduced)
    assert final * g_s == vector(ZZ, [1] + [0] * (g.ncols() - 1))
    case["lambda_babai"] = [int(x) for x in babai]
    case["lambda"] = [int(x) for x in final]
    return case


def subsets_for(n, plan, rng):
    for size, count in plan:
        if count is None or count >= comb(n, size):
            yield from combinations(range(n), size)
        else:
            seen = set()
            while len(seen) < count:
                subset = tuple(sorted(rng.sample(range(n), size)))
                if subset not in seen:
                    seen.add(subset)
                    yield subset


def bulk(name, sizes, count, out_path, seed):
    """Write sets and SageMath's final vectors for `count` random sets per size."""
    with open(os.path.join(REPO, "fixtures", "author", name + ".json"), encoding="utf-8") as handle:
        author = json.load(handle)
    g = Matrix(ZZ, author["g"])
    low, high = (int(x) for x in sizes.split("-"))
    rng = random.Random(seed)
    cases = []
    for subset in subsets_for(author["n"], [(s, count) for s in range(low, high + 1)], rng):
        g_s = g.matrix_from_rows(subset)
        lam0, ok = hnf_witness(g_s)
        case = {"set": [int(i) for i in subset], "authorized": bool(ok)}
        if ok:
            basis = g_s.transpose().right_kernel(ZZ).echelonized_basis_matrix()
            if basis.nrows() > 1:
                basis = basis.LLL()
            final = lam0 if basis.nrows() == 0 else greedy_reduce_linf(babai_exact(basis, lam0), basis)
            case["lambda"] = [int(x) for x in final]
        cases.append(case)
    reference = {
        "format": "latsec-recon-reference-v1",
        "matrix": author["name"],
        "matrix_source_sha256": author["source"]["cell_sources_sha256"],
        "generator": "tools/gen_recon_fixtures.py --bulk",
        "sage_version": sage_version,
        "seed": seed,
        "passes": PASSES,
        "window": WINDOW,
        "cases": cases,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(reference, handle, separators=(",", ":"))
        handle.write("\n")
    print(f"wrote {out_path}: {len(cases)} sets, {sum(1 for c in cases if c['authorized'])} authorized")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bulk", metavar="MATRIX", help="fixture name, e.g. G-60-39-3")
    parser.add_argument("--sizes", default=None, help="size range A-B for --bulk")
    parser.add_argument("--count", type=int, default=400, help="sets per size for --bulk")
    parser.add_argument("--out", help="output file for --bulk")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.bulk:
        if not (args.sizes and args.out):
            parser.error("--bulk needs --sizes and --out")
        bulk(args.bulk, args.sizes, args.count, args.out, args.seed)
        return 0
    out_dir = os.path.join(REPO, "fixtures", "sage")
    os.makedirs(out_dir, exist_ok=True)
    for name, plan in SELECTION.items():
        with open(os.path.join(REPO, "fixtures", "author", name + ".json"), encoding="utf-8") as handle:
            author = json.load(handle)
        g = Matrix(ZZ, author["g"])
        rng = random.Random(SEED)
        cases = [case_for(g, s) for s in subsets_for(author["n"], plan + EXTRA.get(name, []), rng)]
        notable = [case_for(g, s) for s in NOTABLE.get(name, [])]
        fixture = {
            "format": "latsec-recon-reference-v1",
            "matrix": author["name"],
            "matrix_source_sha256": author["source"]["cell_sources_sha256"],
            "generator": "tools/gen_recon_fixtures.py",
            "sage_version": sage_version,
            "seed": SEED,
            "passes": PASSES,
            "window": WINDOW,
            "lll": "Sage Matrix.LLL() defaults (fplll wrapper, delta = 0.99, eta = 0.501)",
            "cases": cases,
            "notable": notable,
        }
        path = os.path.join(out_dir, name.replace("G-", "recon-") + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(fixture, handle, separators=(",", ":"))
            handle.write("\n")
        authorized = sum(1 for c in cases if c["authorized"])
        print(f"wrote {os.path.relpath(path, REPO)}: {len(cases)} sets, {authorized} authorized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
