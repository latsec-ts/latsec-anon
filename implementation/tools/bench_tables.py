#!/usr/bin/env python3
"""Print the tables of the extended-benchmark appendix from recorded measurements.

Reads the JSON records written by ``latsec measure`` (``results/<prefix>-author-
n{40,60}-k{128,192,256}-act<size>.json``) and prints LaTeX table rows, so that every
number in the appendix can be regenerated from the records.  Run from the
repository root:

    python3 tools/bench_tables.py [--prefix 20261116-measure]

Only the Python standard library is used.
"""

import argparse
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(prefix):
    """{(n, kappa, size): record} for the records with the given file prefix."""
    records = {}
    pattern = os.path.join(REPO, "results", f"{prefix}-author-n*-k*-act*.json")
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"-n(\d+)-k(\d+)-act(\d+)\.json$", path)
        with open(path, encoding="utf-8") as handle:
            records[tuple(int(x) for x in match.groups())] = json.load(handle)
    if not records:
        sys.exit(f"no records match {pattern}")
    return records


def ms(value):
    """A time in milliseconds with about three significant digits."""
    if value < 0.01:
        return "{<}0.01"
    if value < 1:
        return f"{value:.2f}"
    if value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def median(record, key):
    return record["timing_ms"][key]["median"]


def row(cells):
    return " & ".join(f"\\({c}\\)" for c in cells) + " \\\\"


# t_sig = k_c of the two SEC instances (Table 3 of the paper).
T_SIG = {40: 26, 60: 46}


def threshold_sets(records):
    """(n, kappa, t_sig) for the parameter sets with a record at |act| = t_sig.

    A set that was only measured at larger sizes is reported and left out, so
    that no other size is presented as the threshold."""
    present = []
    for n, kappa in sorted({(n, kappa) for n, kappa, _ in records}):
        if n not in T_SIG:
            sys.exit(f"no signing threshold known for n = {n}")
        if (n, kappa, T_SIG[n]) in records:
            present.append((n, kappa, T_SIG[n]))
        else:
            print(f"% MISSING: no record at |act| = t_sig = {T_SIG[n]} for n={n} kappa={kappa}")
    return present


def table_running_times(records):
    print("% Median running times in ms at |act| = t_sig")
    print("% n, kappa, Keygen, SS1, SS2, SS3, PV, Comb, Vfy, lambda, P")
    keys = ("keygen", "round1_per_signer", "round2_per_signer", "round3_per_signer",
            "partial_verify_per_call", "combine", "verify", "recon")
    for n, kappa, size in threshold_sets(records):
        record = records[(n, kappa, size)]
        cells = [n, kappa] + [ms(median(record, key)) for key in keys]
        cells.append(f"{1000 * median(record, 'predicate_per_call'):.1f}")      # microseconds
        print(row(cells))


def table_spread(records):
    print("% Quartiles relative to the median, in percent, at |act| = t_sig (largest over the phases)")
    for n, kappa, size in threshold_sets(records):
        record = records[(n, kappa, size)]
        worst = 0.0
        for key in ("round1_per_signer", "round3_per_signer", "partial_verify_per_call", "combine", "verify"):
            t = record["timing_ms"][key]
            worst = max(worst, (t["p75"] - t["median"]) / t["median"], (t["median"] - t["p25"]) / t["median"])
        print(f"%   n={n} kappa={kappa}: {100 * worst:.1f}%")


def table_scaling(records):
    print("% Scaling with |act|: n, kappa, |act|, sessions, wt(lambda), |lambda|_2, lambda ms, SS1, SS3, PV, Comb,"
          " Vfy, |sig| KiB")
    for key in sorted(records):
        n, kappa, size = key
        record = records[key]
        cells = [n, kappa, size, record["signed"],
                 f"{record['lambda']['weight']['mean']:.1f}", f"{record['lambda']['l2']['mean']:.1f}",
                 # With all n parties there is one signer set; its vector is computed once
                 # (during warm-up) and cached, so no computation time is recorded.
                 "\\text{--}" if size == n else ms(median(record, "recon")),
                 ms(median(record, "round1_per_signer")),
                 ms(median(record, "round3_per_signer")), ms(median(record, "partial_verify_per_call")),
                 ms(median(record, "combine")), ms(median(record, "verify")),
                 f"{record['bytes']['sig']['mean'] / 1024:.2f}"]
        print(row(cells))


def table_codec(records):
    print("% Encoding and decoding, median ms at |act| = t_sig:"
          " n, kappa, enc z_i, dec z_i, enc sig, dec sig, enc w_i, dec w_i")
    keys = ("encode_partial_response", "decode_partial_response", "encode_signature", "decode_signature",
            "encode_opening", "decode_opening")
    for n, kappa, size in threshold_sets(records):
        record = records[(n, kappa, size)]
        print(row([n, kappa] + [f"{median(record, key):.2f}" for key in keys]))


def summary(records):
    print("% Sets drawn and environment")
    drawn = sum(r["sets_drawn"] for r in records.values())
    signed = sum(r["signed"] for r in records.values())
    refused = sum(r["not_authorized"] for r in records.values())
    bounds = sum(r["rejected_by_bounds"] for r in records.values())
    aborted = sum(r["aborted"] for r in records.values())
    print(f"%   {len(records)} runs: {drawn} sets drawn, {signed} signed, {refused} not certified,"
          f" {bounds} outside the bounds, {aborted} aborted")
    environments = {json.dumps(r["environment"], sort_keys=True) for r in records.values()}
    for text in sorted(environments):
        print(f"%   {text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prefix", default="20261116-measure", help="file prefix of the records in results/")
    args = parser.parse_args()
    records = load(args.prefix)
    for table in (summary, table_running_times, table_spread, table_scaling, table_codec):
        table(records)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
