# -*- coding: utf-8 -*-
"""CAFE-adjust: a label-free post-hoc estimator for the within-video term.

Given any scorer's frame scores and the (label-free) features, the within-video
comparison S_{vt'} vs S_{vt''} is confounded by within-video nuisance z (time
position, motion energy, feature norm).  This script

  1. reports the exact three-term decomposition of the *reported* AUC,
  2. fits, cross-fitted by video parity, a per-video regression of the score on
     z and reports the decomposition of the residual score s - E[s|z],
  3. bootstrap over videos for CIs and E-values.

Nothing here touches a label at fit time: z comes from features, the score from
the model.  Labels are used only to *evaluate*.

usage
  python cafe_adjust.py --npz out/scores_ucf_base_s789.npz \
      --csv E:/program/DSANet/list/ucf_CLIP_rgbtest.csv --repeat 16
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan")


def zcols(f):
    n = len(f)
    t = np.linspace(0, 1, n)
    dx = np.zeros_like(f)
    dx[1:] = f[1:] - f[:-1]
    mot = np.linalg.norm(dx, axis=1)
    ml = np.convolve(mot, np.ones(5) / 5.0, mode="same")
    return np.stack([t, mot, ml, np.linalg.norm(f, axis=1)], 1)


def pair_stats(s, g, strat=None, nbins=5):
    if g.min() == g.max():
        return 0.0, 0.0
    m = g == 1
    p, b = s[m], s[~m]
    if not len(p) or not len(b):
        return 0.0, 0.0
    if strat is None:
        return len(p) * len(b) * auc(np.r_[np.ones(len(p)), np.zeros(len(b))],
                                     np.r_[p, b]), len(p) * len(b)
    e = np.quantile(strat, np.linspace(0, 1, nbins + 1))
    e[-1] += 1e-9
    num = den = 0.0
    for j in range(nbins):
        sel = (strat >= e[j]) & (strat < e[j + 1])
        pj, bj = p[sel[m]], b[sel[~m]]
        if len(pj) and len(bj):
            num += len(pj) * len(bj) * auc(
                np.r_[np.ones(len(pj)), np.zeros(len(bj))], np.r_[pj, bj])
            den += len(pj) * len(bj)
    return num, den


def decompose(S, G, Z, idx=None):
    idx = range(len(S)) if idx is None else list(idx)
    pos, negA, Bn, Bd, Cn, Cd = [], [], [], [], [], []
    for i in idx:
        s, g, z = S[i], G[i], Z[i]
        if g.sum() == 0:
            negA.append(s)
            continue
        m = g == 1
        pos.append(s[m])
        if (~m).any():
            n_, d_ = pair_stats(s, g)
            Bn.append(n_)
            Bd.append(d_)
            n2, d2 = pair_stats(s, g, z[:, 2])
            Cn.append(n2)
            Cd.append(d2)
    P, A = np.concatenate(pos), np.concatenate(negA)
    Bull = np.concatenate([s[g == 0] for s, g in zip(S, G)
                           if g.min() != g.max()])
    a_t = auc(np.r_[np.ones(len(P)), np.zeros(len(A))], np.r_[P, A])
    hard = auc(np.r_[np.ones(len(P)), np.zeros(len(Bull))], np.r_[P, Bull])
    w = len(A) / (len(A) + len(Bull))
    B = (sum(Bn) / sum(Bd)) if sum(Bd) else float("nan")
    Bc = (sum(Cn) / sum(Cd)) if sum(Cd) else float("nan")
    tot = len(P) * len(Bull)
    C = (hard * tot - B * sum(Bd)) / (tot - sum(Bd))
    return dict(theta=w * a_t + (1 - w) * hard, A=a_t, hard=hard, B=B, Bz=Bc,
                C=C, w=w, vert=(1 - w) * sum(Bd) / tot)


def rdg(Z, s, alpha=1.0):
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Zs = (Z - mu) / sd
    c = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ s)
    return mu, sd, c


def rpred(m, Z):
    mu, sd, c = m
    return ((Z - mu) / sd) @ c


def adjust(S, Z, q=None, nbins=0):
    S2 = [None] * len(S)
    for parity in (0, 1):
        fit = [i for i in range(len(S)) if i % 2 == parity]
        app = [i for i in range(len(S)) if i % 2 != parity]
        Zf, sf = [], []
        for i in fit:
            s = S[i]
            m = (np.ones(len(s), bool) if q is None else
                 s <= np.quantile(s, q))
            Zf.append(Z[i][m])
            sf.append(s[m])
        mod = rdg(np.concatenate(Zf), np.concatenate(sf))
        for i in app:
            S2[i] = S[i] - rpred(mod, Z[i])
    return S2


def e_value(b, null=0.5):
    if not np.isfinite(b) or b <= null:
        return 1.0
    rr = (b / (1 - b)) / (null / (1 - null))
    return float(rr + np.sqrt(rr * (rr - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    d = np.load(args.npz)
    lens = d["lens"]
    gt_flat = d["gt"]
    off = np.r_[0, np.cumsum(lens)].astype(int)
    G = [gt_flat[off[i]:off[i + 1]].astype(int) for i in range(len(lens))]
    paths = pd.read_csv(args.csv)["path"].tolist()
    Z = []
    for p in paths[:len(lens)]:
        Z.append(zcols(np.load(p).astype(np.float32)))
    rng = np.random.RandomState(0)
    print("== %s ==" % (args.tag or args.npz))
    for key in [k for k in d.files if k.startswith("s")]:
        s0 = d[key]
        S = [s0[off[i]:off[i + 1]].astype(float) for i in range(len(lens))]
        variants = [("raw", S)]
        for nm, q in [("adj-all", None), ("adj-ctrl-med", 0.5),
                      ("adj-ctrl-q75", 0.75)]:
            variants.append((nm, adjust(S, Z, q)))
        for nm, Sv in variants:
            pt = decompose(Sv, G, Z)
            boot = []
            for _ in range(args.boot):
                idx = rng.randint(0, len(Sv), len(Sv))
                boot.append(decompose(Sv, G, Z, idx))
            bb = np.array([[b["B"], b["Bz"], b["theta"]] for b in boot])
            lo, hi = np.nanpercentile(bb, [2.5, 97.5], axis=0)
            print("%-12s %-13s theta %.4f [%.4f,%.4f] | A %.4f B %.4f "
                  "[%.4f,%.4f] | B|z %.4f [%.4f,%.4f] | C %.4f | E(B|z) %.2f"
                  % (key, nm, pt["theta"], lo[2], hi[2], pt["A"], pt["B"],
                     lo[0], hi[0], pt["Bz"], lo[1], hi[1], pt["C"],
                     e_value(pt["Bz"])), flush=True)


if __name__ == "__main__":
    main()
