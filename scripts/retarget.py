#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rewrite the absolute paths hardcoded in the research scripts.

The scripts in `cafe/` were written on the machine that produced the paper, so
they refer to the host code and the output directory by absolute path:

    E:/program/DSANet            the DSANet checkout (host architecture)
    E:/program/EDSS-main         a VadCLIP checkout (external anchor)
    E:/program/CAFE/out          where scores and dumps are written
    E:/program/CAFE/shim         the ipdb stub
    E:/program/python/paper      the analysis side

This script rewrites those prefixes to paths inside *this* repository:

    <root>/third_party/DSANet        third_party/VadCLIP
    <root>/results                   results            (out/)
    <root>/cafe/shim                 cafe/shim
    <root>                           (analysis side = repo root)

Both the forward-slash and backslash spellings are handled.  Nothing else in
the files is touched, so the diff is a pure path rename.

Usage
-----
    python scripts/retarget.py                       # dry run, from repo root
    python scripts/retarget.py --root D:/work/CAFE
    python scripts/retarget.py --apply               # actually write

Exit status is 0 when nothing is left to rewrite (or after --apply), 1 when a
dry run found files that still need rewriting.
"""
import argparse
import io
import os
import sys

# (old prefix, new relative path inside the repo); applied longest-first per
# file so that `CAFE/out` is rewritten before the bare `CAFE`.
MAPPING = [
    ("E:/program/python/paper/CAFE/out", "results"),
    ("E:/program/python/paper/CAFE", "."),
    ("E:/program/python/paper", "."),
    ("E:/program/CAFE/out", "results"),
    ("E:/program/CAFE/shim", "cafe/shim"),
    ("E:/program/CAFE", "cafe"),
    ("E:/program/DSANet", "third_party/DSANet"),
    ("E:/program/EDSS-main", "third_party/VadCLIP"),
]

SKIP = {"reproduce_from_scores.py"}

# A few scripts computed their output directory relative to their own location,
# assuming they lived next to a `CAFE/out` folder.  In this repository they live
# in `cafe/` and the artefacts live in `results/`, so those expressions are
# rewritten too.  Each entry is a literal, mechanical substitution -- no logic
# changes.
STRUCT = [
    # output directory of the analysis scripts
    ('os.path.join(os.path.dirname(os.path.abspath(__file__)), "CAFE", "out")',
     'os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '
     '"results")'),
    # table1 builder wrote next to its working directory (portable since v1.0,
    # kept for older copies of the script)
    ('open("CAFE/out/table1.json", "w")', 'open("results/table1_all.json", "w")'),
    # correlation script: table lives beside the other JSON, not below it
    ('HERE = r"E:\\program\\python\\paper\\CAFE"',
     'HERE = r"E:\\program\\python\\paper\\CAFE\\out"'),
    ('T1 = HERE + r"\\out\\table1.json"', 'T1 = HERE + r"\\table1.json"'),
    # scores and ground truth live in subdirectories of the results folder
    ('os.path.join(OUT, "scores_ucf_%s.npz" % tag)',
     'os.path.join(OUT, "scores", "scores_ucf_%s.npz" % tag)'),
    ('os.path.join(OUT, "scores_%s_%s.npz" % (ds, tag))',
     'os.path.join(OUT, "scores", "scores_%s_%s.npz" % (ds, tag))'),
    ('os.path.join(OUT, "gt_segment_%s.npy" % ds)',
     'os.path.join(OUT, "gt", "gt_segment_%s.npy" % ds)'),
    ('os.path.join(OUT, "gt_label_%s.npy" % ds)',
     'os.path.join(OUT, "gt", "gt_label_%s.npy" % ds)'),
    # per-frame score archives also live in the scores subdirectory
    ("E:/program/CAFE/out/scores_", "E:/program/CAFE/out/scores/scores_"),
    ('glob.glob("CAFE/out/dump_*.json")', 'glob.glob("results/dump_*.json")'),
]


def variants(prefix):
    """Both slash spellings of a prefix, longest first."""
    return [prefix, prefix.replace("/", "\\")]


def scan(root):
    targets = []
    for sub in ("cafe", os.path.join("cafe", "exploratory")):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".py") and name not in SKIP:
                targets.append(os.path.join(d, name))
    return targets


def rewrite(text, root, structural_only=False):
    n = 0
    for old, new in STRUCT:
        if old in text:
            n += text.count(old)
            text = text.replace(old, new)
    if structural_only:
        return text, n
    for old, rel in MAPPING:
        new = os.path.abspath(os.path.join(root, rel))
        new_fwd = new.replace("\\", "/")
        new_win = new.replace("/", "\\")
        for pat in variants(old):
            if pat in text:
                n += text.count(pat)
                rep = new_win if "\\" in pat else new_fwd
                text = text.replace(pat, rep)
    return text, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
        help="repository root (default: the parent of this script's directory)")
    ap.add_argument("--apply", action="store_true", help="write the files")
    ap.add_argument("--structural-only", action="store_true",
                    help="only relocate the working-directory idioms and leave "
                         "the host paths alone (used to make a fresh copy of this "
                         "repository runnable on the CPU path)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    files = scan(root)
    if not files:
        sys.exit("no scripts found under %s/cafe" % root)

    total_edits = changed = 0
    for path in files:
        with io.open(path, encoding="utf-8") as fh:
            text = fh.read()
        new, n = rewrite(text, root, structural_only=args.structural_only)
        if n == 0:
            continue
        changed += 1
        total_edits += n
        rel = os.path.relpath(path, root)
        if args.apply:
            with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(new)
            print("rewrote %-46s %3d paths" % (rel, n))
        else:
            print("would rewrite %-38s %3d paths" % (rel, n))

    print("\n%d file(s), %d path(s) %s"
          % (changed, total_edits, "rewritten" if args.apply else "to rewrite"))
    if not args.apply and changed:
        print("re-run with --apply to write")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
