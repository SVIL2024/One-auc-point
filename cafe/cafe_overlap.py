# -*- coding: utf-8 -*-
"""Overlap (positivity) diagnostic for the motion-stratified estimator:
fraction of positive units that share a motion decile with at least one
control unit of the same video.  Data-only, no model involved.

usage: python cafe_overlap.py --dataset ucf --bins 10
"""
import argparse

import numpy as np
import pandas as pd

ROOT = "E:/program/DSANet/list"


def zml(f, k=5):
    dx = np.zeros_like(f)
    dx[1:] = f[1:] - f[:-1]
    mot = np.linalg.norm(dx, axis=1)
    return np.convolve(mot, np.ones(k) / float(k), mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucf", choices=["ucf", "xd"])
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()
    csv = "ucf_CLIP_rgbtest.csv" if args.dataset == "ucf" else \
        "xd_CLIP_rgbtest.csv"
    gtp = "gt_ucf.npy" if args.dataset == "ucf" else "gt.npy"
    df = pd.read_csv("%s/%s" % (ROOT, csv))
    src = list(df["path"])
    lens = [len(np.load(p)) for p in src]
    gt = np.load("%s/%s" % (ROOT, gtp)).reshape(-1, 16)[:, 0].astype(int)
    if args.dataset == "xd":
        # features live at snippet rate, gt at frame rate: expand the scores
        # by the ratio, keeping the comparison at frame resolution
        pass
    tot_pos = matched = 0
    per_video = []
    off = 0
    for p, n in zip(src, lens):
        g = gt[off:off + n] if args.dataset == "ucf" else None
        if args.dataset == "xd":
            step = 16
            g = np.repeat(gt[off // step:off // step + n], step)
        off += n if args.dataset == "ucf" else n * 16
        if g is None or g.min() == g.max():
            continue
        z = zml(np.load(p).astype(np.float32))
        if args.dataset == "xd":
            z = np.repeat(z, 16)
        edges = np.quantile(z, np.linspace(0, 1, args.bins + 1))
        edges[-1] += 1e-9
        b = np.clip(np.digitize(z, edges) - 1, 0, args.bins - 1)
        ok = 0
        np_pos = int(g.sum())
        for j in np.unique(b[g == 1]):
            if (b[g == 0] == j).any():
                ok += int(((g == 1) & (b == j)).sum())
        tot_pos += np_pos
        matched += ok
        per_video.append(ok / float(np_pos))
    print("%s: %d positive units, %.4f share a decile with a control unit "
          "(per-video mean %.4f, min %.4f)"
          % (args.dataset, tot_pos, matched / float(tot_pos),
             float(np.mean(per_video)), float(np.min(per_video))), flush=True)


if __name__ == "__main__":
    main()
