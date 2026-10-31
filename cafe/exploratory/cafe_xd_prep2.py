# -*- coding: utf-8 -*-
"""XD-Violence: convert captured per-FRAME score stacks into the scores_*.npz
format consumed by cafe_adjust.py, and print the pair census at the same
resolution the official evaluation uses (gt is frame-level, 16x the snippet
rate).

usage: python cafe_xd_prep2.py --tag xd_base_s789
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def census(lens, gt):
    n_vid = len(lens)
    pos = int(gt.sum())
    neg = len(gt) - pos
    same = a_cnt = c_cnt = 0
    off = 0
    posv, negv = [], []
    for n in lens:
        g = gt[off:off + n]
        posv.append(int(g.sum()))
        negv.append(int(n - g.sum()))
        off += n
    for p, q in zip(posv, negv):
        same += p * q
    for i in range(n_vid):
        for j in range(n_vid):
            if i != j:
                a_cnt += posv[i] * negv[j]
    nA = sum(posv[i] * negv[j] for i in range(n_vid) for j in range(n_vid)
             if i != j and negv[j] > 0 and posv[i] > 0 and not _is_anom_j(j))
    return same, a_cnt, pos, neg


def _is_anom_j(j):
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--csv", default="E:/program/DSANet/list/xd_CLIP_rgbtest.csv")
    args = ap.parse_args()
    d = np.load("E:/program/CAFE/out/xdstack_%s.npz" % args.tag,
                allow_pickle=True)
    stack = [np.asarray(a, dtype=np.float32) for a in d["stack"]]
    gt = np.asarray(d["gt"]).reshape(-1).astype(np.int64)
    roc1 = float(d["roc1"]) if "roc1" in d.files else float("nan")
    s = [1.0 - a[:, 0] for a in stack]          # column 0 == 'normal' prompt
    lens = np.asarray([len(x) for x in s], np.int64)
    flat = np.concatenate(s)
    if len(flat) != len(gt):
        raise SystemExit("length mismatch %d vs %d" % (len(flat), len(gt)))
    mine = roc_auc_score(gt, flat)
    print("[%s] pooled AUC %.4f | repo AUC2 %.4f | diff %.5f"
          % (args.tag, mine, roc1, mine - roc1), flush=True)
    # pair census at frame resolution, matching the official evaluation
    posv, negv = [], []
    off = 0
    for n in lens:
        g = gt[off:off + n]
        posv.append(int(g.sum()))
        negv.append(int(n - g.sum()))
        off += n
    nB = sum(p * q for p, q in zip(posv, negv))
    nP, nN = sum(posv), sum(negv)
    anom = [i for i in range(len(lens)) if posv[i] > 0]
    nA = sum(posv[i] * negv[j] for i in anom for j in range(len(lens))
             if j != i and negv[j] > 0 and posv[j] == 0)
    nC = nP * nN - nA - nB
    N = nP * nN
    w = nA / N
    lam = nB / N
    print("  videos %d frames %d pos %d | w_A %.6f w_B %.6f w_C %.6f"
          % (len(lens), len(gt), nP, w, lam, nC / N), flush=True)
    print("  same-video pairs %d of %d (%.4f%%) | 1 point = %.1f B points"
          % (nB, N, 100.0 * nB / N, 1.0 / lam if lam else float("nan")),
          flush=True)
    paths = pd.read_csv(args.csv)["path"].tolist()
    ml = []
    for p, n in zip(paths[:len(lens)], lens):
        f = np.load(p).astype(np.float32)
        dx = np.zeros_like(f)
        dx[1:] = f[1:] - f[:-1]
        mot = np.linalg.norm(dx, axis=1)
        smooth = np.convolve(mot, np.ones(5) / 5.0, mode="same")
        rep = 16 if len(flat) > sum(len(x) for x in [f]) else 1
        ml.append(np.repeat(smooth, int(round(n / len(f)))))
    np.savez_compressed(
        "E:/program/CAFE/out/scores/scores_%s.npz" % args.tag,
        **{"s2|s": flat}, lens=lens, gt=gt, mlocal=np.concatenate(ml))
    print("saved scores_%s.npz" % args.tag, flush=True)


if __name__ == "__main__":
    main()
