# -*- coding: utf-8 -*-
"""Data-side constants for the pooled-AUC decomposition.

For the WS-VAD frame-level pooled AUC we have the exact identity

    theta = w * easy + (1 - w) * hard
    easy  = AUC(anomaly frames , normal frames in NORMAL videos)
    hard  = AUC(anomaly frames , normal frames in ANOMALOUS videos)
    w     = |A| / (|A| + |B|)

and `hard` itself splits, by *which video the normal frame comes from*, into

    hard = lam * tau_w + (1 - lam) * theta_out

    tau_w     = pair-weighted AUC over same-video (anomaly, normal) pairs
    theta_out = AUC over cross-video pairs among anomalous videos
    lam       = (#same-video pairs) / (#all pairs in the hard pool)
              = sum_v p_v n_v / (N_B * sum_v p_v)

lam is a pure property of the annotations. It is the exchange rate between
"localization ability" and "reported AUC": one point of tau_w moves the
reported number by (1 - w) * lam points only.

This script prints w, lam, the pair counts, and the per-video statistics that
the decomposition needs, for UCF-Crime and XD-Violence.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LIST = "E:/program/DSANet/list"


def gt_ucf():
    gt = np.load(os.path.join(LIST, "gt_ucf.npy"))
    return gt.reshape(-1, 16)[:, 0].astype(np.int64)


def gt_xd():
    gt = np.load(os.path.join(LIST, "gt.npy"))
    return np.asarray(gt).reshape(-1).astype(np.int64)


def load_feats(csv):
    df = pd.read_csv(csv)
    return [np.load(p) for p in df["path"]], list(df["label"])


def analyse(name, csv, gt):
    feats, labels = load_feats(csv)
    total = sum(len(f) for f in feats)
    if len(gt) != total:
        print("[%s] WARN gt %d != frames %d" % (name, len(gt), total))
        n = min(len(gt), total)
        keep, acc = [], 0
        for f in feats:
            if acc + len(f) <= n:
                keep.append(f)
                acc += len(f)
            else:
                break
        feats, gt = keep, gt[:acc]
        total = acc
    off = 0
    nA = nB = P = 0
    pv, nv = [], []
    for f, lab in zip(feats, labels):
        g = gt[off:off + len(f)]
        off += len(f)
        if g.sum() == 0:
            nA += int((g == 0).sum())
        else:
            p = int((g == 1).sum())
            q = int((g == 0).sum())
            P += p
            nB += q
            pv.append(p)
            nv.append(q)
    pv, nv = np.array(pv), np.array(nv)
    w = nA / (nA + nB)
    same = float((pv * nv).sum())
    allp = float(nB * pv.sum())
    lam = same / allp
    d = dict(dataset=name, n_videos=len(feats), frames=total, pos=P,
             neg_normal_video=nA, neg_anomalous_video=nB, w=w,
             lam=lam, same_video_pairs=same, hard_pairs=allp,
             n_anom_videos=len(pv),
             median_pos_per_video=float(np.median(pv)),
             median_neg_per_video=float(np.median(nv)),
             frac_videos_both=float((((pv > 0) & (nv > 0)).sum()) / max(len(pv), 1)))
    d["rate_tau_to_theta"] = (1 - w) * lam
    d["rate_theta_to_tau"] = 1.0 / ((1 - w) * lam)
    print(json.dumps(d, indent=1))
    return d


def main():
    out = []
    out.append(analyse("UCF-Crime", os.path.join(LIST, "ucf_CLIP_rgbtest.csv"),
                       gt_ucf()))
    p = os.path.join(LIST, "xd_CLIP_rgbtest.csv")
    if os.path.exists(p):
        try:
            out.append(analyse("XD-Violence", p, gt_xd()))
        except Exception as e:  # noqa
            print("[XD] failed:", repr(e))
    with open("E:/program/CAFE/out/atlas.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for d in out:
        print("%-12s w=%.4f lam=%.6f  dtheta/dtau=%.2e  1 theta pt = %.0f tau pt"
              % (d["dataset"], d["w"], d["lam"], d["rate_tau_to_theta"],
                 d["rate_theta_to_tau"]))


if __name__ == "__main__":
    main()
