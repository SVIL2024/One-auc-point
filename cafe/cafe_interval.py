# -*- coding: utf-8 -*-
"""How wide is the reported interval, really?  Compares a frame-level i.i.d.
bootstrap of the pooled AUC (what a DeLong-style interval assumes) with a
video-clustered bootstrap of the same statistic, on saved frame scores.

usage: python cafe_interval.py --npz E:/program/CAFE/out/scores/scores_ucf_base_s789.npz
"""
import argparse
import os

import numpy as np
from scipy.stats import rankdata


def auc(sp, sn):
    n1, n0 = len(sp), len(sn)
    r = rankdata(np.concatenate([sp, sn]))
    return (r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--npz", default=os.path.join(HERE, "results", "scores", "scores_ucf_base_s789.npz"))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    s, lens, gt = d["s2"], d["lens"], d["gt"]
    g = gt.reshape(-1, 16)[:, 0].astype(int) if gt.size != s.size else gt.astype(int)
    rng = np.random.default_rng(args.seed)
    pos, neg = s[g == 1], s[g == 0]
    # (a) i.i.d. frames: what a frame-level interval assumes
    ia = []
    for _ in range(args.reps):
        ia.append(auc(rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)))
    ia = np.percentile(ia, [2.5, 97.5])
    # (b) cluster bootstrap over videos
    per = []
    off = 0
    for n in lens:
        gg = g[off:off + n]
        ss = s[off:off + n]
        per.append((ss[gg == 1], ss[gg == 0]))
        off += n
    cb = []
    for _ in range(args.reps):
        idx = rng.integers(0, len(per), len(per))
        sp = np.concatenate([per[i][0] for i in idx if len(per[i][0])])
        sn = np.concatenate([per[i][1] for i in idx if len(per[i][1])])
        cb.append(auc(sp, sn))
    cb = np.percentile(cb, [2.5, 97.5])
    print("theta point estimate %.4f" % auc(pos, neg))
    print("i.i.d. frames      [%.4f, %.4f]  width %.4f" % (ia[0], ia[1], ia[1] - ia[0]))
    print("clustered (videos) [%.4f, %.4f]  width %.4f" % (cb[0], cb[1], cb[1] - cb[0]))
    print("ratio %.1fx" % ((cb[1] - cb[0]) / (ia[1] - ia[0])), flush=True)


if __name__ == "__main__":
    main()
