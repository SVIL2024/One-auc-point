# -*- coding: utf-8 -*-
"""Convert the captured XD score stacks into the scores_*.npz format that
cafe_adjust.py consumes, and check the pooled AUC against the repository's own
AUC2 before anything else is believed.

usage: python cafe_xd_prep.py --tag xd_base_s789
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def zcols(f):
    n = len(f)
    t = np.linspace(0, 1, n)
    dx = np.zeros_like(f)
    dx[1:] = f[1:] - f[:-1]
    mot = np.linalg.norm(dx, axis=1)
    ml = np.convolve(mot, np.ones(5) / 5.0, mode="same")
    return np.stack([t, mot, ml, np.linalg.norm(f, axis=1)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--csv", default="E:/program/DSANet/list/xd_CLIP_rgbtest.csv")
    args = ap.parse_args()
    d = np.load("E:/program/CAFE/out/xdstack_%s.npz" % args.tag,
                allow_pickle=True)
    stack = [np.asarray(a, dtype=np.float32) for a in d["stack"]]
    gt = d["gt"]
    roc1 = float(d["roc1"]) if "roc1" in d.files else float("nan")
    # XD score: one minus the probability of the 'normal' prompt
    s = [1.0 - a[:, 0] for a in stack]
    lens = np.asarray([len(x) for x in s], np.int64)
    flat = np.concatenate(s)
    gtf = gt.reshape(-1, 16)[:, 0].astype(np.int64)
    assert len(gtf) == len(flat), (len(gtf), len(flat))
    mine = roc_auc_score(gtf, np.repeat(flat, 16))
    print("[%s] pooled AUC (1-p_normal) %.4f | repo AUC2 %.4f | diff %.5f"
          % (args.tag, mine, roc1, mine - roc1), flush=True)
    paths = pd.read_csv(args.csv)["path"].tolist()
    zs = [zcols(np.load(p).astype(np.float32)) for p in paths[:len(lens)]]
    np.savez_compressed(
        "E:/program/CAFE/out/scores/scores_%s.npz" % args.tag,
        **{"s2|s": flat}, lens=lens, gt=gtf.astype(np.int64),
        mlocal=np.concatenate([z[:, 2] for z in zs]).astype(np.float32))
    print("saved scores_%s.npz (%d videos, %d frames, %d pos)"
          % (args.tag, len(lens), len(flat), int(gtf.sum())), flush=True)


if __name__ == "__main__":
    main()
