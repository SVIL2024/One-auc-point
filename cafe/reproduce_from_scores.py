#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recompute the decomposition table from the stored per-frame scores.

This script is the CPU-only, dependency-free (numpy) counterpart of the GPU
path: `dsanet_dump.py` runs the network and dumps one `scores_*.npz` per
checkpoint; this script takes those files and recomputes every column of the
decomposition (theta, A, B, B|z, C) plus the per-video pair census, in seconds,
on any machine.  It is the fastest way to check that the numbers in the paper
follow from the released scores.

Usage
-----
    python cafe/reproduce_from_scores.py                     # all *.npz in results/scores
    python cafe/reproduce_from_scores.py --scores results/scores --table results/table1.json
    python cafe/reproduce_from_scores.py --scores x.npz -v   # one file, verbose

What the columns mean (Section 3-4 of the paper):
    theta  pooled AUC of the reported metric on this split, i.e. the number a
           benchmark would print
    A      cross-video block: AUC of positive frames against negative frames
           drawn from normal videos
    B      cross-video block: the pair-weighted within-video AUC, i.e. AUC of
           positive frames against background frames of the *same* video
    B|z    B after stratifying each video on its within-video motion (quintiles
           of the 5-frame smoothed |dx|), the estimate that survives the motion
           mediation check
    C      the residual block, defined by the identity
           theta = w*A + (1-w)*hard  and  hard = lam*B + (1-lam)*C
    w, lam the pair-census weights (data properties of the split, not of the
           checkpoint)

If `--table` is given, the recomputed rows are compared against the stored
JSON (a dict keyed by arm name) and the largest absolute deviation per column
is printed.  Deviations of order 1e-3 are expected and harmless: the stored
table was produced on the snippet grid (one score per 16 frames) while the
score files store the frame-level repeat, and the stratified estimate depends
on the quantile boundaries of the nuisance.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def auc(pos, neg):
    """ROC AUC of `pos` against `neg`, ties contributing one half.

    Average-rank Mann-Whitney, numerically identical to
    sklearn.metrics.roc_auc_score on binary labels, implemented here so the
    script needs nothing but numpy.
    """
    s = np.concatenate([pos, neg]).astype(np.float64)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    ranks[order] = ranks.copy()
    npos = y.sum()
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def pair_within(pairs, strat=None, nbins=5):
    """Pair-weighted within-video AUC, optionally stratified on `strat`."""
    num = den = 0.0
    for k, (s, g) in enumerate(pairs):
        if g.min() == g.max():
            continue
        m = g == 1
        p, b = s[m], s[~m]
        if not len(p) or not len(b):
            continue
        if strat is None:
            num += len(p) * len(b) * auc(p, b)
            den += len(p) * len(b)
        else:
            z = strat[k]
            e = np.quantile(z, np.linspace(0, 1, nbins + 1))
            e[-1] += 1e-9
            for j in range(nbins):
                sel = (z >= e[j]) & (z < e[j + 1])
                pj, bj = p[sel[m]], b[sel[~m]]
                if len(pj) and len(bj):
                    num += len(pj) * len(bj) * auc(pj, bj)
                    den += len(pj) * len(bj)
    return num / den if den else float("nan")


def decompose(pairs, zs=None):
    """Exact three-block decomposition of theta (Theorem 1)."""
    pos, negA, same = [], [], []
    for s, g in pairs:
        if g.sum() == 0:
            negA.append(s)
            continue
        m = g == 1
        pos.append(s[m])
        if (~m).any():
            same.append((s[m], s[~m]))
    P = np.concatenate(pos)
    A_ = np.concatenate(negA)
    Ball = np.concatenate([b for _, b in same])
    a = auc(P, A_)
    hard = auc(P, Ball)
    w = len(A_) / (len(A_) + len(Ball))
    wp = sum(len(p) * len(b) for p, b in same)
    tot = len(P) * len(Ball)
    B_ = pair_within(pairs)
    C_ = (hard * tot - B_ * wp) / (tot - wp)
    out = dict(w=float(w), lam=float(wp / tot), A=float(a), B=float(B_),
               B_cond=None, C=float(C_), hard=float(hard),
               theta=float(w * a + (1 - w) * hard))
    if zs is not None:
        out["B_cond"] = float(pair_within(pairs, strat=zs))
    return out


# --------------------------------------------------------------------------- #
def split_videos(x, lens):
    """Slice a concatenated frame-level array back into per-video arrays."""
    out, i = [], 0
    for n in lens:
        n = int(n)
        out.append(x[i:i + n])
        i += n
    return out


def load(path):
    d = np.load(path, allow_pickle=True)
    keys = list(d.files)
    skey = "s2" if "s2" in keys else next(
        (k for k in keys if k.startswith("s2")), None)
    if skey is None or "lens" not in keys or "gt" not in keys:
        raise KeyError("%s lacks a score/lens/gt array (found %s)" % (path, keys))
    n_expected = int(np.sum(d["lens"]))
    if len(d[skey]) != n_expected:
        raise ValueError("%s stores %d scores for %d snippets (frame-level "
                         "duplicate); re-dump at snippet resolution to use it "
                         "here" % (os.path.basename(path), len(d[skey]), n_expected))
    gt = d["gt"]
    if gt.dtype.kind in "OUS":          # XD files may store class codes ('A', 'B1', ...)
        gt = np.array([0 if str(v).strip().upper().startswith("A") else 1
                       for v in gt], dtype=np.int64)
    scores = split_videos(d[skey].astype(np.float64), d["lens"])
    labels = split_videos(gt.astype(np.int64), d["lens"])
    if "mlocal" in keys:
        mlocal = split_videos(d["mlocal"].astype(np.float64), d["lens"])
    else:
        mlocal = None
    return list(zip(scores, labels)), mlocal


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=os.path.join("results", "scores"),
                    help="a *.npz score file, or a directory containing them")
    ap.add_argument("--table", default=os.path.join("results", "table1.json"),
                    help="stored decomposition table to compare against (JSON)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if os.path.isdir(args.scores):
        files = sorted(glob.glob(os.path.join(args.scores, "*.npz")))
    else:
        files = [args.scores]
    if not files:
        sys.exit("no score files found in %s" % args.scores)

    stored = {}
    if args.table and os.path.exists(args.table):
        with open(args.table, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):          # [{arm: ..., theta: ...}, ...]
            stored = {row["arm"]: row for row in raw if "arm" in row}
        else:                              # {"ucf_base_s234": {...}, ...}
            stored = raw

    rows, worst = [], {}
    for path in files:
        tag = os.path.basename(path)[:-4].replace("scores_", "")
        try:
            pairs, zs = load(path)
        except (KeyError, ValueError) as exc:
            print("skipping %-24s %s" % (tag, exc))
            continue
        d = decompose(pairs, zs)
        rows.append((tag, d))
        ref = stored.get(tag) or stored.get(tag.replace("_", "-"))
        diffs = {}
        if ref:
            for k_ours, k_ref in (("theta", "theta"), ("A", "A"), ("B", "B"),
                                  ("B_cond", "Bz"), ("C", "C")):
                v = d.get(k_ours)
                r = ref.get(k_ref)
                if v is not None and r is not None:
                    diffs[k_ours] = abs(v - float(r))
            for k, v in diffs.items():
                worst[k] = max(worst.get(k, 0.0), v)
        if args.verbose:
            print("%-24s theta %.4f  A %.4f  B %.4f  B|z %.4f  C %.4f  "
                  "w %.4f lam %.5f  n_videos %d"
                  % (tag, d["theta"], d["A"], d["B"],
                     d["B_cond"] if d["B_cond"] is not None else float("nan"),
                     d["C"], d["w"], d["lam"], len(pairs)))
            if diffs:
                print("    max|diff| vs stored table: " +
                      ", ".join("%s %.1e" % (k, v) for k, v in diffs.items()))

    print("\n%-24s %7s %7s %7s %7s %7s" %
          ("checkpoint", "theta", "A", "B", "B|z", "C"))
    for tag, d in rows:
        print("%-24s %7.4f %7.4f %7.4f %7.4f %7.4f" %
              (tag, d["theta"], d["A"], d["B"],
               d["B_cond"] if d["B_cond"] is not None else float("nan"), d["C"]))
    if worst:
        print("\nlargest |recomputed - stored| over all files: " +
              ", ".join("%s %.2e" % (k, v) for k, v in sorted(worst.items())))
    else:
        print("\n(no stored table at %s; nothing to compare)" % args.table)


if __name__ == "__main__":
    main()
