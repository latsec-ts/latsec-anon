#!/usr/bin/env python3
"""Copy the implementation into the artifact and check the copy for identifying text.

    python3 tools/export_artifact.py --out DIR [--redactions FILE] [--replace]
    python3 tools/export_artifact.py --check DIR [--redactions FILE]

The copy is an allowlist: the two implementations, the profiles, fixtures, recorded
results and tools, and the top-level README, Makefile and .gitignore.  Build output
and caches are left out.  Nothing else of the working repository is copied.

After copying (or with ``--check``, instead of copying) every file is scanned:

* for the regular expressions of the redaction list, a private file of names,
  affiliations and repository addresses that is not part of the artifact;
* for absolute home-directory paths and for references to files outside the copy.

Any hit is reported with its file and line, and the exit status is non-zero.  Files
that are not text are reported too, since they cannot be scanned.  If the directory
is inside a git repository, files that its .gitignore rules exclude are reported as
well: they would be left out of the published artifact.  ``--check`` needs
an existing directory, and a scan that finds no file at all, or cannot read a
directory, fails: a clean result always means that files were read.

``--out`` must name an empty or missing directory, or one that holds only a
placeholder ``readme.md``; with ``--replace`` an earlier export (a directory with
the same top-level entries) is removed first.

Only the Python standard library is used.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Top-level entries of the artifact's implementation directory.
INCLUDE = ["README.md", "Makefile", ".gitignore", "latsec-rs", "latsec-py", "profiles", "fixtures", "results", "tools"]

# Directory and file names never copied, wherever they occur.
SKIP_NAMES = {"target", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv"}
SKIP_SUFFIXES = (".pyc", ".pyo", ".prof", ".orig", ".rej", ".bak", ".swp")

# Results that are not part of the paper: a comparison that was withdrawn.
SKIP_PATHS = {
    "results/20261108-traccoon-ec24-thrc-act26-n40.log",
    "results/20261108-traccoon-ec24-thrc-act46-n60.log",
    "results/20261108-traccoon-notes.md",
}

# Patterns that are checked whatever the redaction list says.
BUILTIN = [
    (r"/home/[A-Za-z0-9_.-]+", "home-directory path"),
    (r"/Users/[A-Za-z0-9_.-]+", "home-directory path"),
    (r"(?<![\w./-])doc/[\w.-]+", "reference to the private doc/ directory"),
    # Written in pieces so that this file does not match its own pattern.
    ("AGENTS" + r"\.md|\.git" + "modules|git sub" + "module", "reference to a file that is not exported"),
]


def wanted(rel_path):
    parts = rel_path.split("/")
    if any(part in SKIP_NAMES for part in parts):
        return False
    return rel_path not in SKIP_PATHS and not rel_path.endswith(SKIP_SUFFIXES)


def files_to_copy():
    for entry in INCLUDE:
        top = os.path.join(REPO, entry)
        if os.path.isfile(top):
            yield entry
            continue
        for root, dirs, files in os.walk(top):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_NAMES)
            for name in sorted(files):
                rel = os.path.relpath(os.path.join(root, name), REPO).replace(os.sep, "/")
                if wanted(rel):
                    yield rel


def prepare(out, replace):
    if not os.path.exists(out):
        os.makedirs(out)
        return
    present = set(os.listdir(out))
    if present <= {"readme.md"}:
        for name in present:
            os.remove(os.path.join(out, name))
        return
    if replace and present <= set(INCLUDE):
        for name in present:
            path = os.path.join(out, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        return
    sys.exit(f"{out} is not empty; nothing was written"
             + (" (it holds entries that an export does not create)" if replace else " (see --replace)"))


def load_redactions(path):
    patterns = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append((re.compile(line, re.IGNORECASE), "redaction list"))
    return patterns


def walk_error(error):
    raise error


def scan(root, patterns):
    """(hits, number of files scanned), with one (path, line number, reason, text)
    per hit under the directory `root`.  A directory that cannot be read raises."""
    hits = []
    scanned = 0
    for folder, dirs, files in os.walk(root, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_NAMES and d != ".git")
        for name in sorted(files):
            path = os.path.join(folder, name)
            rel = os.path.relpath(path, root)
            scanned += 1
            for pattern, reason in patterns:
                if pattern.search(rel):
                    hits.append((rel, 0, reason + " (file name)", rel))
            try:
                with open(path, encoding="utf-8") as handle:
                    lines = handle.read().split("\n")
            except UnicodeDecodeError:
                hits.append((rel, 0, "not a text file; cannot be scanned", ""))
                continue
            for number, line in enumerate(lines, 1):
                for pattern, reason in patterns:
                    match = pattern.search(line)
                    if match:
                        start = max(0, match.start() - 40)
                        hits.append((rel, number, reason, line[start:match.end() + 40].strip()))
    return hits, scanned


def ignored_by_git(root):
    """Files under root that the enclosing git repository, if any, would not add.

    An exported file that a .gitignore of the artifact repository excludes would be
    missing from the published artifact without any error."""
    try:
        listing = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--ignored=matching", "--untracked-files=all", "--", "."],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []                       # not a git repository, or no git
    return sorted(line[3:] for line in listing.splitlines() if line.startswith("!! "))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", help="directory to export into")
    group.add_argument("--check", help="scan this directory without copying anything")
    parser.add_argument("--redactions", default=os.path.join(REPO, "doc", "artifact-redactions.txt"),
                        help="the private list of forbidden regular expressions")
    parser.add_argument("--replace", action="store_true", help="remove an earlier export in --out first")
    args = parser.parse_args()

    patterns = [(re.compile(text), reason) for text, reason in BUILTIN]
    if not os.path.exists(args.redactions):
        sys.exit(f"redaction list {args.redactions} not found; refusing to export without it")
    patterns += load_redactions(args.redactions)

    target = args.check or args.out
    if args.check and not os.path.isdir(args.check):
        sys.exit(f"{args.check} is not an existing directory; nothing was scanned")
    if args.out:
        prepare(args.out, args.replace)
        count = 0
        for rel in files_to_copy():
            destination = os.path.join(args.out, rel)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(os.path.join(REPO, rel), destination)
            if os.access(os.path.join(REPO, rel), os.X_OK):
                os.chmod(destination, 0o755)
            count += 1
        print(f"copied {count} files to {args.out}")

    hits, scanned = scan(target, patterns)
    for rel, number, reason, text in hits:
        print(f"LEAK {rel}:{number}: {reason}: {text}")
    print(f"scanned {scanned} files in {target} against {len(patterns)} patterns: {len(hits)} hits")
    if scanned == 0:
        print(f"no file was found in {target}; this is not a clean result")
        return 1
    ignored = ignored_by_git(target)
    for path in ignored:
        print(f"IGNORED {path}: excluded by a .gitignore of the repository; it would not be published")
    if ignored:
        print(f"{len(ignored)} files are ignored by git")
    return 1 if hits or ignored else 0


if __name__ == "__main__":
    sys.exit(main())
