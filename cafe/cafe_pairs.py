# -*- coding: utf-8 -*-
"""Data-only pair census of the two benchmarks.

For a pooled frame-level metric every (anomalous frame, normal frame) pair is a
binary comparison.  Classify each pair by whether the two frames come from the
same video, and (when they do not) by whether the normal frame's video is itself
normal.  The three counts are exactly the weights of the decomposition; they are
fixed by the annotation, not by any model.

usage: python cafe_pairs.py            (run on the server, needs npy/csv only)
"""
import numpy as np
import pandas as pd

ROOT = "E:/program/DSANet/list"


def census(lens, gt):
    """returns w (normal-video share of negatives), lam (same-video share of
    anomalous-video negatives), and raw counts"""
    n = len(lens)
    off = np.r_[0, np.cumsum(lens)].astype(int)
    npos = int(gt.sum())
    negA = negB = 0
    for i in range(n):
        g = gt[off[i]:off[i + 1]]
        p, ng = int(g.sum()), int((1 - g).sum())
        if p == 0:
            negA += ng
        else:
            negB += ng
    n_neg = negA + negB
    w = negA / n_neg
    # same-video pairs among anomalous-video negatives
    same = 0
    for i in range(n):
        g = gt[off[i]:off[i + 1]]
        p, ng = int(g.sum()), int((1 - g).sum())
        if p and ng:
            same += p * ng
    anom_neg = negB
    lam = same / (npos * anom_neg)
    return dict(n_videos=n, n_frames=int(len(gt)), n_pos=npos,
                neg_normal_video=negA, neg_anom_bg=negB, w=w, lam=lam,
                pairs_same_video=same, pairs_cross=npos * n_neg - same,
                share_same=1.0 - (npos * n_neg - same) / (npos * n_neg),
                a_weight=w, b_weight=(1 - w) * lam, c_weight=(1 - w) * (1 - lam))


def main():
    out = {}
    # UCF-Crime
    df = pd.read_csv(ROOT + "/ucf_CLIP_rgbtest.csv")
    lens = [len(np.load(p)) for p in df["path"]]
    gt = np.load(ROOT + "/gt_ucf.npy").reshape(-1, 16)[:, 0].astype(int)
    assert sum(lens) == len(gt), (sum(lens), len(gt))
    out["ucf"] = census(lens, gt)
    # XD-Violence (gt is stored at 16x frame rate, exactly like UCF)
    df = pd.read_csv(ROOT + "/xd_CLIP_rgbtest.csv")
    lens = [len(np.load(p)) for p in df["path"]]
    gt = np.load(ROOT + "/gt.npy").reshape(-1, 16)[:, 0].astype(int)
    assert sum(lens) == len(gt), (sum(lens), len(gt))
    out["xd"] = census(lens, gt)
    for k, v in out.items():
        print("%s: %d videos %d frames %d pos | w=%.6f lam=%.6f | weights "
              "A %.4f B %.5f C %.4f | same-video pairs %d of %d (%.4f%%)"
              % (k, v["n_videos"], v["n_frames"], v["n_pos"], v["w"], v["lam"],
                 v["a_weight"], v["b_weight"], v["c_weight"],
                 v["pairs_same_video"], v["pairs_same_video"] + v["pairs_cross"],
                 100 * v["share_same"]))
        print("   exchange rate: 1 reported point = %.1f localization points"
              % (1.0 / v["b_weight"]))
    import json
    json.dump(out, open("E:/program/CAFE/out/pair_census.json", "w"), indent=1)


if __name__ == "__main__":
    main()
