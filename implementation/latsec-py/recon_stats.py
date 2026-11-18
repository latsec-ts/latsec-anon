"""recon_stats.py -- reconstruction-vector statistics per signer-set size.

For each size s, sets of s parties are tested with [46, Algorithm 1]
(sec.reconstruction_vector): how many are authorized, the distributions of
|lambda|_inf, wt(lambda) and |lambda|_1, and how many authorized sets a
predicate (beta_lambda, omega_lambda) rejects.  These are the quantities behind
Section 7.1 of the paper.

Sets are chosen as in `latsec recon-stats` (latsec-rs), so both produce the
same counts: all sets of size s when there are at most `samples` of them,
in lexicographic order; otherwise `samples` distinct sets drawn with
random_subset() from SeededRandom(seed, "recon-statistics/size-<s>").

Usage:  python3 recon_stats.py ../fixtures/author/G-40-19-3.json --sizes 26..27 \\
            --samples 1000 --seed demo [--beta 5 --omega 25] [--jobs 8] [--json FILE]
"""

import argparse
import json
import sys
import time
from collections import Counter
from itertools import combinations
from math import comb
from multiprocessing import Pool

import sec
from xof import SeededRandom, random_subset

_G = None


def _init(g):
    global _G
    _G = g


def _vector(subset):
    return subset, sec.reconstruction_vector([_G[i] for i in subset])


def sets_for_size(n, size, samples, seed):
    if comb(n, size) <= samples:
        return list(combinations(range(n), size)), True
    rng = SeededRandom(seed.encode(), f"recon-statistics/size-{size}")
    seen, out = set(), []
    while len(out) < samples:
        subset = tuple(random_subset(rng, n, size))
        if subset not in seen:
            seen.add(subset)
            out.append(subset)
    return out, False


def size_statistics(g, size, samples, seed, beta=None, omega=None, jobs=1):
    sets, exhaustive = sets_for_size(len(g), size, samples, seed)
    started = time.time()
    if jobs > 1:
        with Pool(jobs, initializer=_init, initargs=(g,)) as pool:
            results = pool.map(_vector, sets, chunksize=64)
    else:
        _init(g)
        results = [_vector(s) for s in sets]
    stats = {"size": size, "exhaustive": exhaustive, "tested": len(sets), "authorized": 0,
             "rejected_by_bounds": 0, "linf": Counter(), "weight": Counter(), "l1": Counter(),
             "rejected_sets": [], "seconds": time.time() - started}
    for subset, lam in results:
        if lam is None:
            continue
        stats["authorized"] += 1
        linf, weight, l1 = sec.linf(lam), sum(1 for x in lam if x), sum(abs(x) for x in lam)
        stats["linf"][linf] += 1
        stats["weight"][weight] += 1
        stats["l1"][l1] += 1
        if beta is not None and (linf > beta or weight > omega):
            stats["rejected_by_bounds"] += 1
            stats["rejected_sets"].append((list(subset), lam))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("matrix")
    ap.add_argument("--sizes", required=True, help="A..B")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", default="recon-stats")
    ap.add_argument("--beta", type=int)
    ap.add_argument("--omega", type=int)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--json")
    args = ap.parse_args()
    with open(args.matrix) as f:
        g = json.load(f)["g"]
    lo, hi = (int(x) for x in args.sizes.split(".."))
    report = []
    for size in range(lo, hi + 1):
        s = size_statistics(g, size, args.samples, args.seed, args.beta, args.omega, args.jobs)
        auth = max(s["authorized"], 1)
        print(f"size {size}: {s['tested']} sets{' (all)' if s['exhaustive'] else ''}, "
              f"{s['authorized']} authorized, max linf {max(s['linf'], default=0)}, "
              f"max weight {max(s['weight'], default=0)}, mean weight "
              f"{sum(k * c for k, c in s['weight'].items()) / auth:.2f}, "
              f"rejected by the bounds {s['rejected_by_bounds']} ({s['seconds']:.1f} s)")
        report.append({**s, **{key: dict(sorted(s[key].items())) for key in ("linf", "weight", "l1")}})
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
