"""alternatives.py -- other admissible reconstruction vectors for a signer set.

Under the predicate P of the paper (Figure 1) the initiator may put any
lambda_act with lambda_act^T G_act = e_1^T, |lambda_act|_inf <= beta_lambda and
wt(lambda_act) <= omega_lambda into the session message.  For a set whose
Algorithm 1 vector fails the bounds (Section 7.1), this script searches
lambda_0 + sum_j a_j b_j over the LLL-reduced kernel basis (b_j) with every
a_j in [-r, r], and reports how many admissible vectors it finds.

Usage:  python3 alternatives.py ../fixtures/author/G-40-19-3.json \\
            --set 0,1,2,5,10,...,38 --beta 5 --omega 25 [--radius 2]
With --rejected-notable, the set is the one of fixtures/sage that the
predicate rejects.
"""

import argparse
import itertools
import json
import os
import sys

import sec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kernel_basis(g_t):
    _, transform, pivots = sec.row_hnf(g_t)
    kernel = [list(r) for r in transform[len(pivots):]]
    kernel = kernel[:len(sec.hnf_in_place(kernel, len(g_t)))]
    sec.lll_reduce(kernel, 99, 100)
    return kernel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("matrix")
    ap.add_argument("--set", help="zero-based party indices, comma separated")
    ap.add_argument("--rejected-notable", action="store_true")
    ap.add_argument("--beta", type=int, required=True)
    ap.add_argument("--omega", type=int, required=True)
    ap.add_argument("--radius", type=int, default=2)
    args = ap.parse_args()
    with open(args.matrix) as f:
        m = json.load(f)
    g = m["g"]
    if args.rejected_notable:
        tag = os.path.basename(args.matrix)[2:-5]
        with open(os.path.join(REPO, "fixtures", "sage", f"recon-{tag}.json")) as f:
            notable = json.load(f)["notable"]
        act = next(c["set"] for c in notable
                   if sec.linf(c["lambda"]) > args.beta or sum(1 for x in c["lambda"] if x) > args.omega)
    else:
        act = [int(x) for x in args.set.split(",")]
    g_t = [g[i] for i in act]
    lam = sec.reconstruction_vector(g_t)
    if lam is None:
        print("the set is not authorized")
        return 1
    print(f"set of {len(act)} parties (zero-based): {act}")
    print(f"Algorithm 1: linf {sec.linf(lam)}, weight {sum(1 for x in lam if x)}")
    basis = kernel_basis(g_t)
    found = []
    for coeffs in itertools.product(range(-args.radius, args.radius + 1), repeat=len(basis)):
        v = [x + sum(c * b[i] for c, b in zip(coeffs, basis)) for i, x in enumerate(lam)]
        weight, linf = sum(1 for x in v if x), sec.linf(v)
        if linf <= args.beta and weight <= args.omega:
            assert sec.is_reconstruction_vector(g, act, v)
            found.append((weight, linf, v))
    found.sort()
    print(f"kernel dimension {len(basis)}, radius {args.radius}: "
          f"{(2 * args.radius + 1) ** len(basis)} candidates, {len(found)} admissible")
    for weight, linf, v in found[:3]:
        print(f"  weight {weight}, linf {linf}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
