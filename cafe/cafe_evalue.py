# -*- coding: utf-8 -*-
"""Inference for the within-video term: cluster bootstrap + E-value.

Reads the per-video scores saved by cafe_probe7.py and reports, for every arm,
    theta, A, B, B|z, C   (point estimates)
    B and B|z             (95% cluster bootstrap CI over videos)
    E-value               (how strong an unmeasured within-video confounder
                           would have to be, on the risk-ratio scale, to explain
                           the B|z signal away)

The bootstrap resamples *videos*, not frames: the video is the independent unit.
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan")


def pair_num_den(s, g, strat=None, nbins=5):
    """pair-weighted AUC as (numerator, denominator) so it can be re-aggregated"""
    if g.min() == g.max():
        return 0.0, 0.0
    m = g == 1
    p, b = s[m], s[~m]
    if not len(p) or not len(b):
        return 0.0, 0.0
    if strat is None:
        return len(p) * len(b) * auc(np.r_[np.ones(len(p)), np.zeros(len(b))],
                                     np.r_[p, b]), len(p) * len(b)
    z = strat
    e = np.quantile(z, np.linspace(0, 1, nbins + 1))
    e[-1] += 1e-9
    num = den = 0.0
    for j in range(nbins):
        sel = (z >= e[j]) & (z < e[j + 1])
        pj, bj = p[sel[m]], b[sel[~m]]
        if len(pj) and len(bj):
            num += len(pj) * len(bj) * auc(
                np.r_[np.ones(len(pj)), np.zeros(len(bj))], np.r_[pj, bj])
            den += len(pj) * len(bj)
    return num, den


def decompose(S, lens, gt, strat, idx=None):
    """theta/A/B/B|z/C over the videos in `idx` (default: all)."""
    n = len(S)
    idx = range(n) if idx is None else idx
    pos, negA, same_n, same_d, cond_n, cond_d = [], [], [], [], [], []
    for i in idx:
        s, g = S[i], gt[i]
        if g.sum() == 0:
            negA.append(s)
            continue
        m = g == 1
        pos.append(s[m])
        if (~m).any():
            same_n.append((s[m], s[~m]))
            a_, b_ = pair_num_den(s, g, strat[i])
            cond_n.append(a_)
            cond_d.append(b_)
    P, A = np.concatenate(pos), np.concatenate(negA)
    Bull = np.concatenate([b for _, b in same_n])
    a_t = auc(np.r_[np.ones(len(P)), np.zeros(len(A))], np.r_[P, A])
    hard = auc(np.r_[np.ones(len(P)), np.zeros(len(Bull))], np.r_[P, Bull])
    w = len(A) / (len(A) + len(Bull))
    num = den = 0.0
    for p, b in same_n:
        num += len(p) * len(b) * auc(np.r_[np.ones(len(p)), np.zeros(len(b))],
                                     np.r_[p, b])
        den += len(p) * len(b)
    B = num / den
    tot = len(P) * len(Bull)
    C = (hard * tot - B * den) / (tot - den)
    Bc = (sum(cond_n) / sum(cond_d)) if sum(cond_d) else float("nan")
    return dict(theta=w * a_t + (1 - w) * hard, A=a_t, B=B, B_cond=Bc, C=C)


def e_value(b, null=0.5):
    """E-value for the odds ratio implied by a within-video AUC vs chance."""
    if not np.isfinite(b) or b <= 0.5 or b >= 1:
        return 1.0
    rr = (b / (1 - b)) / (null / (1 - null))
    return float(rr + np.sqrt(rr * (rr - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    lens = d["lens"]
    gt_flat = d["gt"]
    keys = [k for k in d.files if k.endswith("|s")]
    off = np.r_[0, np.cumsum(lens)]
    gt = [gt_flat[off[i]:off[i + 1]] for i in range(len(lens))]
    res = []
    rng = np.random.RandomState(args.seed)
    for k in sorted(keys):
        S = [np.asarray(s, float) for s in d[k]]
        if "mlocal" in d.files:
            mf = d["mlocal"]
            strat = [mf[off[i]:off[i + 1]] for i in range(len(lens))]
        else:
            strat = [np.arange(len(s)) / max(len(s) - 1, 1) for s in S]
        pt = decompose(S, lens, gt, strat)
        boots = []
        for _ in range(args.boot):
            idx = rng.randint(0, len(S), len(S))
            boots.append(decompose(S, lens, gt, strat, idx))
        bb = np.array([[b["B"], b["B_cond"]] for b in boots])
        lo, hi = np.nanpercentile(bb, [2.5, 97.5], axis=0)
        row = dict(arm=k, **{f: pt[f] for f in ["theta", "A", "B", "B_cond",
                                                "C"]},
                   B_lo=lo[0], B_hi=hi[0], Bc_lo=lo[1], Bc_hi=hi[1],
                   e_value_B=e_value(pt["B"]),
                   e_value_Bc=e_value(pt["B_cond"]))
        res.append(row)
        print("[%s] theta %.4f | A %.4f B %.4f [%.4f,%.4f] | B|z %.4f "
              "[%.4f,%.4f] | C %.4f | E(B) %.2f E(B|z) %.2f"
              % (k, row["theta"], row["A"], row["B"], row["B_lo"],
                 row["B_hi"], row["B_cond"], row["Bc_lo"], row["Bc_hi"],
                 row["C"], row["e_value_B"], row["e_value_Bc"]), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
