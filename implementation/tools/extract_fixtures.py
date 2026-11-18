#!/usr/bin/env python3
"""Extract the SEC matrices from the SEC-geometry notebooks into JSON fixtures.

The notebooks are read as data; no notebook code is executed.  For each selected
instance the script reads, from ``MatrixResult-<n>-<m>-<c>.ipynb``:

* the matrix ``G`` (the ``rows = [...]`` literal),
* the planted supports ``T`` and their witnesses ``lambdas``,
* the expander parameters ``gamma`` and ``eps_exp`` and the printed ``k_p``.

Each fixture records the notebook, the cell indices and a SHA-256 digest of the cell
sources it was read from.  Run from the directory of the implementation:

    python3 tools/extract_fixtures.py [--notebooks DIR]

The notebooks are looked for in ``../parameter-search/SEC-geometry``, where the
artifact has them, unless ``--notebooks`` names another directory.  The artifact
carries the notebooks of the two instances the paper uses, (n, k) = (40, 20) and
(60, 40).  The two small matrices used by the functional test profiles, (10, 6) and
(20, 10), are instances of the SEC paper [46] whose notebooks are not part of the
artifact: when the notebook of one of these two is not found, its existing fixture
is kept, with its recorded cell digest, and reported.  The notebooks of the two
instances of the paper are required: if one is missing, the script stops before
writing anything.

Only the Python standard library is used.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = os.path.join(os.path.dirname(REPO), "parameter-search", "SEC-geometry")
# How a notebook is named in a fixture: its path in the artifact.
ARTIFACT_DIR = "parameter-search/SEC-geometry"
OUT_DIR = os.path.join(REPO, "fixtures", "author")

# (n, m, c_exp) of the instances to extract.
INSTANCES = [(10, 5, 3), (20, 9, 3), (40, 19, 3), (60, 39, 3)]
# Instances of the SEC paper [46] whose notebooks the artifact does not carry; only
# for these may an existing fixture stand in for a missing notebook.
WITHOUT_NOTEBOOK = {(10, 5, 3), (20, 9, 3)}


def cell_source(cell):
    return "".join(cell["source"])


def find_cell(cells, predicate, what):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and predicate(cell_source(c))]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one cell with {what}, found {len(hits)}")
    return hits[0]


def literal_after(source, name):
    """Evaluate the Python literal assigned to ``name`` in ``source``."""
    match = re.search(rf"^{name}\s*=\s*", source, flags=re.MULTILINE)
    if match is None:
        raise SystemExit(f"no assignment to {name}")
    rest = source[match.end():]
    # Take the balanced bracket expression that starts the remainder.
    depth, end = 0, None
    for pos, char in enumerate(rest):
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
            if depth == 0:
                end = pos + 1
                break
    if end is None:
        raise SystemExit(f"unbalanced literal for {name}")
    return ast.literal_eval(rest[:end])


def fraction_after(source, name):
    """Return the right-hand side of ``name = a/b`` or ``name = 0.x`` as a string."""
    match = re.search(rf"^{name}\s*=\s*([0-9./]+)\s*$", source, flags=re.MULTILINE)
    if match is None:
        raise SystemExit(f"no assignment to {name}")
    return match.group(1)


def printed_output(cell):
    out = []
    for item in cell.get("outputs", []):
        out.append("".join(item.get("text", item.get("data", {}).get("text/plain", []))))
    return "".join(out).strip()


def notebook_name(n, m, c_exp):
    return f"MatrixResult-{n}-{m}-{c_exp}.ipynb"


def extract(n, m, c_exp, notebooks):
    rel_path = f"{ARTIFACT_DIR}/{notebook_name(n, m, c_exp)}"
    with open(os.path.join(notebooks, notebook_name(n, m, c_exp)), encoding="utf-8") as handle:
        cells = json.load(handle)["cells"]

    rows_cell = find_cell(cells, lambda s: s.lstrip().startswith("rows = ["), "the G literal")
    supports_cell = find_cell(cells, lambda s: s.lstrip().startswith("T = [["), "the supports")
    params_cell = find_cell(cells, lambda s: "eps_exp =" in s and "def k_p_bound" in s, "gamma")
    kp_cell = find_cell(cells, lambda s: s.lstrip().startswith("k_p = k_p_bound("), "k_p")

    rows = literal_after(cell_source(cells[rows_cell]), "rows")
    supports = literal_after(cell_source(cells[supports_cell]), "T")
    witnesses = literal_after(cell_source(cells[supports_cell]), "lambdas")
    gamma = fraction_after(cell_source(cells[params_cell]), "gamma")
    eps_exp = fraction_after(cell_source(cells[params_cell]), "eps_exp")
    k_p = int(printed_output(cells[kp_cell]))

    if len(rows) != n or any(len(r) != m + 1 for r in rows):
        raise SystemExit(f"{rel_path}: G is not {n} x {m + 1}")
    if any(x not in (-1, 0, 1) for r in rows for x in r):
        raise SystemExit(f"{rel_path}: G has an entry outside {{-1, 0, 1}}")
    if any(sum(1 for x in r[1:] if x) != c_exp for r in rows):
        raise SystemExit(f"{rel_path}: a row of G_hat does not have {c_exp} nonzeros")
    if len(supports) != len(witnesses) or any(len(t) != len(w) for t, w in zip(supports, witnesses)):
        raise SystemExit(f"{rel_path}: supports and witnesses do not match")

    used = [rows_cell, supports_cell, params_cell, kp_cell]
    digest = hashlib.sha256()
    for index in used:
        digest.update(cell_source(cells[index]).encode("utf-8"))
        digest.update(b"\0")

    return {
        "format": "latsec-author-matrix-v1",
        "name": f"author-{n}-{m}-{c_exp}",
        "source": {
            "notebook": rel_path,
            "cells": {"g": rows_cell, "supports": supports_cell, "expander": params_cell, "k_p": kp_cell},
            "cell_sources_sha256": digest.hexdigest(),
            "index_base": "supports are zero-based row indices, as in the notebook code",
        },
        "n": n,
        "m": m,
        "k": m + 1,
        "c_exp": c_exp,
        "gamma": gamma,
        "eps_exp": eps_exp,
        "k_p": k_p,
        "g": rows,
        "planted_supports": supports,
        "planted_witnesses": [list(w) for w in witnesses],
    }


def dump(fixture, path):
    """Write JSON with one matrix row per line."""
    text = json.dumps(fixture, indent=2)
    # Collapse the innermost integer lists onto single lines.
    text = re.sub(r"\[\s+((?:-?\d+,\s+)*-?\d+)\s+\]",
                  lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def keep_existing(n, m, c_exp, path):
    """The existing fixture of an instance whose notebook is not available, with
    its source block in the current form."""
    with open(path, encoding="utf-8") as handle:
        fixture = json.load(handle)
    old = fixture["source"]
    fixture["source"] = {
        "notebook": f"{notebook_name(n, m, c_exp)} of the SEC paper [46] (not part of this artifact)",
        "cells": old["cells"],
        "cell_sources_sha256": old["cell_sources_sha256"],
        "index_base": old["index_base"],
    }
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--notebooks", default=NOTEBOOKS, help="directory of the MatrixResult notebooks")
    args = parser.parse_args()
    # Decide everything first, so that a missing notebook stops the run before any
    # fixture is rewritten.
    results = []
    for instance in INSTANCES:
        n, m, c_exp = instance
        path = os.path.join(OUT_DIR, f"G-{n}-{m}-{c_exp}.json")
        notebook = os.path.join(args.notebooks, notebook_name(n, m, c_exp))
        if os.path.isfile(notebook):
            results.append((instance, path, extract(n, m, c_exp, args.notebooks), "wrote"))
        elif instance not in WITHOUT_NOTEBOOK:
            raise SystemExit(f"{notebook} not found: the notebooks of the instances of the paper are required "
                             "(see --notebooks); nothing was written")
        elif os.path.isfile(path):
            results.append((instance, path, keep_existing(n, m, c_exp, path), "kept (no notebook)"))
        else:
            raise SystemExit(f"neither {notebook} nor an existing fixture {path} was found; nothing was written")
    os.makedirs(OUT_DIR, exist_ok=True)
    for (n, m, c_exp), path, fixture, how in results:
        dump(fixture, path)
        print(f"{how} {os.path.relpath(path, REPO)}: {n} x {m + 1}, k_p = {fixture['k_p']}, "
              f"{len(fixture['planted_supports'])} planted supports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
