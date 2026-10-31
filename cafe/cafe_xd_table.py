# -*- coding: utf-8 -*-
"""Exact three-block decomposition + motion stratification on XD-Violence.

Frame resolution: XD-Violence ground truth is at 30 fps while features and
scores are at snippet rate, so each snippet score is expanded to the 16 frames of
its snippet before pairing.  `theta` here reproduces the repo's AUC2, which is
the check that the expansion is the official one.

usage: python cafe_xd_table.py
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CSV = "E:/program/DSANet/list/xd_CLIP_rgbtest.csv"


def auc(s_pos, s_neg):
    n1, n0 = len(s_pos), len(s_neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.concatenate([s_pos, s_neg]))
    return (r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def motion(f, k=5):
    dx = np.zeros_like(f)
    dx[1:] = f[1:] - f[:-1]
    mot = np.linalg.norm(dx, axis=1)
    return np.convolve(mot, np.ones(k) / float(k), mode="same"), mot


def pair_auc(sp, zp, sn, zn, q=0):
    """AUC and pair count, optionally restricted to common motion bins."""
    if q <= 0:
        return auc(sp, sn) * len(sp) * len(sn), len(sp) * len(sn)
    tot = cnt = 0.0
    for b in range(q):
        mp, mn = zp == b, zn == b
        if mp.any() and mn.any():
            tot += auc(sp[mp], sn[mn]) * mp.sum() * mn.sum()
            cnt += mp.sum() * mn.sum()
    return tot, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="E:/program/CAFE/out/xdstack_*.npz")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()
    src = list(pd.read_csv(CSV)["path"])
    rows = []
    for fn in sorted(glob.glob(args.glob)):
        tag = os.path.basename(fn)[len("xdstack_"):-4]
        d = np.load(fn, allow_pickle=True)
        stack, lens, gt = d["stack"], d["lens"], d["gt"]
        per, corrs, off = [], [], 0
        for v, (path, n) in enumerate(zip(src, lens)):
            n = int(n)
            # the stack is already at frame rate (the XD loader expands each
            # snippet to its 16 frames), so only the motion has to be expanded
            f = np.nan_to_num(1.0 - stack[v][:, 0].astype(np.float64), nan=0.5)
            g = gt[off:off + len(f)].astype(int)
            off += len(f)
            mot, _ = motion(np.load(path).astype(np.float32))
            zf = np.repeat(mot, 16)[:len(f)]
            if len(mot) > 8 and mot.std() > 0 and len(zf) == len(f):
                corrs.append(np.corrcoef(rankdata(f), rankdata(zf))[0, 1])
            # per-video motion bins (deciles), computed on this video only
            if len(zf) > args.bins:
                e = np.quantile(zf, np.linspace(0, 1, args.bins + 1))
                e[-1] += 1e-9
                binz = np.clip(np.digitize(zf, e) - 1, 0, args.bins - 1)
            else:
                binz = np.zeros(len(zf), int)
            per.append((f, g, binz))
        anom = [(f[g == 1], bz[g == 1], f[g == 0], bz[g == 0], g)
                for f, g, bz in per if g.max() > 0]
        norm = [f[g == 0] for f, g, bz in per if g.max() == 0]
        sp_all = np.concatenate([a[0] for a in anom])
        A_auc = auc(sp_all, np.concatenate(norm))
        nA = len(sp_all) * sum(len(x) for x in norm)
        Bt = nb = Zt = nz = 0.0
        for sp, zp, sn, zn, _ in anom:
            if len(sn) == 0:
                continue
            t, c = pair_auc(sp, zp, sn, zn)
            Bt += t
            nb += c
            tz, cz = pair_auc(sp, zp, sn, zn, q=args.bins)
            Zt += tz
            nz += cz
        B_auc, Bz = Bt / nb, Zt / max(nz, 1)
        pooled = auc(sp_all, np.concatenate([a[2] for a in anom]))
        nP, nN = len(sp_all), sum(len(a[2]) for a in anom)
        nC = nP * nN - nb
        C_auc = (pooled * (nP * nN) - Bt) / nC
        theta = (nA * A_auc + Bt + nC * C_auc) / (nA + nb + nC)
        rows.append(dict(tag=tag, theta=theta, A=A_auc, B=B_auc, Bz=Bz, C=C_auc,
                         w_A=nA / float(nA + nb + nC),
                         w_B=nb / float(nA + nb + nC),
                         w_C=nC / float(nA + nb + nC),
                         roc1=float(d["roc1"]), ap1=float(d["ap1"]),
                         n_pos=int(nP), n_neg_normal=int(sum(len(x) for x in norm)),
                         corr=float(np.mean(corrs))))
        print({k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in rows[-1].items()}, flush=True)
        json.dump(rows, open("E:/program/CAFE/out/xd_table_raw.json", "w"), indent=1)


if __name__ == "__main__":
    main()
