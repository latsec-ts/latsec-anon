#!/usr/bin/env python3
"""Rerun the lattice estimates behind Table 4 with a given lattice-estimator checkout.

Run with SageMath's Python, passing the estimator checkout as an argument:

    sage -python tools/latsec_estimates.py --estimator /path/to/lattice-estimator

The instances are built exactly as in the parameter-selection notebook (build_params,
Sec_all): MLWE_{p, ell', q, d, sigma} -> LWE(n = ell' d, m = p d, Xs = Xe =
D(sigma)) with the full attack set under ADPS16 minus arora-gb and bkw, and
MSIS_{p, ell'+1, q, d, B_MSIS} -> SIS(n = p d, m = (ell + 1) d, norm 2).  Widths and
bounds come from latsec-py/params.py; a security level is lg of the cheapest finite rop.

Variants
--------
published     the six rows at their published primes
bq-minus-1    the same widths and bounds with the largest NTT prime below 2^(b_q - 1)
exact-floor   exact tail inversion, at the smallest b_q with B_MSIS < q, for the rows
              where that is below b_q - 1

The script executes the estimator code it is pointed at.  Use a checkout you trust,
and record its commit (printed below) with every number.
"""

import argparse
import datetime
import io
import json
import multiprocessing
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from math import floor

# The parameter engine lives in latsec-py/params.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "latsec-py"))
import params as lp  # noqa: E402

DENY_LIST = ("arora-gb", "bkw")
KINDS = ("MQ", "AK", "MSIS")


def build_tasks(variants, kinds=KINDS):
    """One task per (variant, row, instance kind)."""
    tasks = []
    for variant in variants:
        for row in lp.PUBLISHED:
            if variant == "published":
                inst = lp.estimator_instance(row)
            elif variant == "bq-minus-1":
                inst = lp.estimator_instance(row, b_q=row["b_q"] - 1)
            elif variant == "exact-floor":
                exact = lp.estimator_instance(row, theta=lp.theta_exact)
                b_q = floor(exact["lg_b_msis"]) + 1
                if b_q >= row["b_q"] - 1:
                    continue
                inst = lp.estimator_instance(row, b_q=b_q, theta=lp.theta_exact)
            else:
                raise ValueError(variant)
            for kind in kinds:
                tasks.append(dict(variant=variant, kind=kind, inst=inst))
    return tasks


def run_task(task):
    """Estimate one instance.  Runs in a worker; the estimator is already imported."""
    from estimator import LWE, ND, RC, SIS
    from sage.all import Integer, log, oo

    inst, kind = task["inst"], task["kind"]
    out = dict(variant=task["variant"], kind=kind, n=inst["n"], kappa=inst["kappa"],
               b_q=inst["b_q"], q=inst["q"], target=inst["targets"][kind])
    sink = io.StringIO()
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            if kind == "MSIS":
                params = SIS.Parameters(n=inst["sis_n"], q=Integer(inst["q"]),
                                        length_bound=inst["b_msis"], m=inst["sis_m"], norm=2)
                result = SIS.estimate(params, red_cost_model=RC.ADPS16)
            else:
                sigma = inst["sigma_mq"] if kind == "MQ" else inst["sigma_ak"]
                noise = ND.DiscreteGaussian(sigma)
                params = LWE.Parameters(n=inst["lwe_n"], q=Integer(inst["q"]),
                                        Xs=noise, Xe=noise, m=inst["lwe_m"])
                result = LWE.estimate(params, red_cost_model=RC.ADPS16, deny_list=DENY_LIST)
        attacks = {}
        for name, cost in result.items():
            rop = cost["rop"]
            if rop != oo:
                attacks[name] = float(log(rop, 2))
        out["attacks"] = attacks
        if attacks:
            out["sec"] = min(attacks.values())
        else:
            # estimate() catches per-attack failures and logs them instead of raising.
            log_lines = [line for line in sink.getvalue().splitlines() if line.strip()]
            reason = log_lines[-1].rsplit(" failed with ", 1)[-1] if log_lines else "no log"
            out["error"] = "no attack was costed: " + reason
    except Exception as exc:  # a direct failure, e.g. a rejected parameter set
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def print_table(results, variants):
    by_key = {(r["variant"], r["n"], r["kappa"], r["kind"]): r for r in results}
    blank = dict(skipped=True)
    for variant in variants:
        rows = sorted({(r["n"], r["kappa"], r["b_q"]) for r in results
                       if r["variant"] == variant})
        if not rows:
            continue
        print(f"\nvariant: {variant}")
        print("   n kappa  b_q |   Sec_MQ   Sec_AK (target)  Sec_MSIS (target) | retained")
        for n, kappa, b_q in rows:
            cells, retained = [], True
            for kind in KINDS:
                res = by_key.get((variant, n, kappa, kind), blank)
                if "skipped" in res:
                    cells.append("       -")
                    retained = None if retained else retained
                elif "error" in res:
                    cells.append("   error")
                    retained = False
                else:
                    cells.append(f"{res['sec']:8.1f}")
                    retained = retained and res["sec"] >= res["target"]
            target_ak = kappa + 1 + lp.ceil_lg(lp.GEOMETRY[n]["k"])
            verdict = {True: "yes", False: "NO", None: "n/a"}[retained]
            print(f"  {n:2d}  {kappa:3d}   {b_q:2d} | {cells[0]} {cells[1]} ({target_ak:3d})   "
                  f"{cells[2]} ({kappa:3d})    | {verdict}")
        for res in results:
            if res["variant"] == variant and "error" in res:
                print(f"    n={res['n']} kappa={res['kappa']} {res['kind']}: {res['error']}")
    published = {(r["n"], r["kappa"]): r["sec"] for r in lp.PUBLISHED}
    deviations = [abs(res["sec"] - published[(res["n"], res["kappa"])][res["kind"]])
                  for res in results if res["variant"] == "published" and "sec" in res]
    if deviations:
        print(f"\npublished variant: largest deviation from the values printed in the notebook = "
              f"{max(deviations):.2f} bits over {len(deviations)} estimates")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--estimator", required=True,
                        help="path to a lattice-estimator checkout (it will be executed)")
    parser.add_argument("--variant", action="append",
                        choices=("published", "bq-minus-1", "exact-floor"),
                        help="repeatable; default: all three")
    parser.add_argument("--kind", action="append", choices=KINDS,
                        help="repeatable; restrict to these instances (default: all)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--json", help="write the raw results to this file")
    args = parser.parse_args()
    variants = args.variant or ["published", "bq-minus-1", "exact-floor"]

    sys.path.insert(0, os.path.abspath(args.estimator))
    import estimator  # noqa: F401  (imported before forking so workers inherit it)
    from sage.version import version as sage_version

    meta = dict(estimator_commit=git_commit(args.estimator),
                sage=sage_version, python=sys.version.split()[0],
                date=datetime.date.today().isoformat(), variants=variants,
                kinds=list(args.kind or KINDS),
                cost_model="ADPS16", deny_list=list(DENY_LIST))
    print(json.dumps(meta, indent=2))

    tasks = build_tasks(variants, args.kind or KINDS)
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        results = pool.map(run_task, tasks, chunksize=1)
    print_table(results, variants)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(dict(meta=meta, results=results), handle, indent=2)
        print(f"\nraw results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
