# -*- coding: utf-8 -*-
"""Stage-1 probe for the causal within-video framework (CAFE).

Question answered here (no deep model, no training loop, minutes on one GPU):

  1. What are the dataset constants of the pooled-AUC identity?
         pooled AUC = w * AUC(pos vs negatives from NORMAL videos)
                    + (1-w) * AUC(pos vs BACKGROUND of anomalous videos)
     w is a pure data property (frame counts), so the identity is exact.

  2. Does a *within-transformed* snippet head (features centred per video,
     i.e. the fixed-effect transformation that removes video identity) buy
     within-video localisation over the same head on raw features?

  3. What does a video-constant score achieve?  (the degenerate "confounded"
     estimator: one score per video, within-video AUC = 0.5 by construction)

Usage:
    python cafe_probe.py --dataset ucf
    python cafe_probe.py --dataset xd
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = "E:/dataset"
LIST = "E:/program/DSANet/list"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def uniform_extract(feat, t_max):
    new = np.zeros((t_max, feat.shape[1]), dtype=np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    for i in range(t_max):
        if r[i] != r[i + 1]:
            new[i] = feat[r[i]:r[i + 1]].mean(0)
        else:
            new[i] = feat[r[i]]
    return new


def load_train(csv, t_max=200, limit=None):
    df = pd.read_csv(csv)
    if limit:
        df = df.iloc[:limit]
    feats, labels, names = [], [], []
    for _, row in df.iterrows():
        f = np.load(row["path"]).astype(np.float32)
        feats.append(uniform_extract(f, t_max) if len(f) > t_max else np.pad(
            f, ((0, t_max - len(f)), (0, 0))))
        labels.append(0 if str(row["label"]).strip() == "Normal" else 1)
        names.append(row["path"])
    return feats, np.asarray(labels, np.float32), names


def load_test(csv, gt_path, repeat):
    df = pd.read_csv(csv)
    feats, names = [], []
    for _, row in df.iterrows():
        f = np.load(row["path"]).astype(np.float32)
        feats.append(f)
        names.append(row["path"])
    gt = np.load(gt_path)
    if repeat == 16:
        gt_frame = gt.reshape(-1, 16)[:, 0].astype(np.int64)
    else:
        gt_frame = np.asarray(gt).astype(np.int64).reshape(-1)
    total = sum(len(f) for f in feats)
    if len(gt_frame) != total:
        print("[warn] gt length %d != frame count %d; truncating" % (
            len(gt_frame), total))
        n = min(len(gt_frame), total)
        gt_frame = gt_frame[:n]
        # drop trailing videos that do not fit
        keep, acc = [], 0
        for f in feats:
            if acc + len(f) <= n:
                keep.append(f)
                acc += len(f)
            else:
                break
        feats = keep
    return feats, gt_frame


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auc(y, p):
    y = np.asarray(y)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def split_pairs(feats, scores, gt_frame):
    off, out = 0, []
    for f, s in zip(feats, scores):
        g = gt_frame[off:off + len(f)]
        off += len(f)
        out.append((np.asarray(s), g))
    return out


def report(pairs, tag):
    pos, negA, negB, within = [], [], [], []
    for s, g in pairs:
        if g.sum() == 0:
            negA.append(s)
            continue
        m = g == 1
        pos.append(s[m])
        if (~m).any():
            negB.append(s[~m])
            if g.min() != g.max():
                within.append(roc_auc_score(g, s))
    P = np.concatenate(pos)
    A = np.concatenate(negA) if negA else np.array([])
    B = np.concatenate(negB) if negB else np.array([])
    w = len(A) / max(len(A) + len(B), 1)
    pooled = auc(np.r_[np.ones(len(P) + 0), np.zeros(len(A) + len(B))],
                 np.r_[P, A, B])
    a_easy = auc(np.r_[np.ones(len(P)), np.zeros(len(A))], np.r_[P, A])
    a_hard = auc(np.r_[np.ones(len(P)), np.zeros(len(B))], np.r_[P, B])
    wth = np.asarray(within)
    d = dict(tag=tag, pooled_auc=pooled, w=w, auc_easy=a_easy,
             auc_hard=a_hard, identity_check=w * a_easy + (1 - w) * a_hard,
             within_mean=float(np.nanmean(wth)),
             within_median=float(np.nanmedian(wth)),
             within_below_half=float(np.nanmean(wth < 0.5)),
             n_videos=len(pairs), n_pos=int(len(P)),
             n_negA=int(len(A)), n_negB=int(len(B)))
    print("[%s] pooled %.4f | easy %.4f hard %.4f | w %.4f | identity %.4f"
          " | within %.4f (med %.4f, <0.5 %.3f)"
          % (tag, d["pooled_auc"], a_easy, a_hard, w, d["identity_check"],
             d["within_mean"], d["within_median"], d["within_below_half"]),
          flush=True)
    return d


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class MILHead(nn.Module):
    """Linear snippet scorer trained with top-k MIL on video labels."""

    def __init__(self, dim=512, hidden=0):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, 1))
        else:
            self.net = nn.Linear(dim, 1)

    def forward(self, x, mask):
        s = self.net(x).squeeze(-1)
        return s.masked_fill(mask, -1e4)


def train_mil(feats, labels, centred=False, center_mode="mean", k=3,
              epochs=60, lr=1e-3, seed=0, hidden=0, device="cuda",
              verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.tensor(np.stack(feats))              # (N, T, D)
    Y = torch.tensor(labels)
    if centred:
        if center_mode == "mean":
            c = X.mean(dim=1, keepdim=True)
        elif center_mode == "median":
            c = X.median(dim=1, keepdim=True).values
        elif center_mode == "trimmed":
            kk = max(1, int(0.1 * X.shape[1]))
            topk = X.topk(kk, dim=1).values
            c = (X.sum(dim=1, keepdim=True) - topk.sum(dim=1, keepdim=True)) / (
                X.shape[1] - kk)
        else:
            raise ValueError(center_mode)
        X = X - c
        X = X / (X.norm(dim=-1, keepdim=True).mean() + 1e-6)
    X, Y = X.to(device), Y.to(device)
    n, t, d = X.shape
    model = MILHead(d, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = torch.arange(n)
    for ep in range(epochs):
        perm = idx[torch.randperm(n)]
        tot = 0.0
        for i in range(0, n, 32):
            b = perm[i:i + 32]
            x = X[b]
            s = model.net(x).squeeze(-1)
            top = s.topk(min(k, t), dim=1).values.mean(1)
            loss = F.binary_cross_entropy_with_logits(top, Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print("  ep %02d loss %.4f" % (ep, tot / n), flush=True)
    model.eval()
    return model


@torch.no_grad()
def score(model, feats, centred=False, center_mode="mean", device="cuda"):
    out = []
    for f in feats:
        x = torch.tensor(f, device=device).unsqueeze(0)
        if centred:
            if center_mode == "mean":
                c = x.mean(dim=1, keepdim=True)
            elif center_mode == "median":
                c = x.median(dim=1, keepdim=True).values
            elif center_mode == "trimmed":
                kk = max(1, int(0.1 * x.shape[1]))
                c = (x.sum(dim=1, keepdim=True)
                     - x.topk(kk, dim=1).values.sum(dim=1, keepdim=True)) / (
                    x.shape[1] - kk)
            x = x - c
            x = x / (x.norm(dim=-1, keepdim=True).mean() + 1e-6)
        s = model.net(x).squeeze(-1).squeeze(0).cpu().numpy()
        out.append(s)
    return out


class VideoHead(nn.Module):
    def __init__(self, dim=512, hidden=0):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1)) if hidden
                    else nn.Linear(dim, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_video(feats, labels, epochs=120, lr=1e-3, seed=0, hidden=0,
                device="cuda"):
    torch.manual_seed(seed)
    X = torch.tensor(np.stack([f.mean(0) for f in feats])).to(device)
    Y = torch.tensor(labels).to(device)
    model = VideoHead(X.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        s = model(X)
        loss = F.binary_cross_entropy_with_logits(s, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def score_video(model, feats, device="cuda"):
    out = []
    for f in feats:
        m = torch.tensor(f.mean(0), device=device).unsqueeze(0)
        out.append(float(model(m).item()))
    return out


def zscores(list_of_arrays):
    allv = np.concatenate(list_of_arrays)
    mu, sd = allv.mean(), allv.std() + 1e-9
    return [(a - mu) / sd for a in list_of_arrays]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucf", choices=["ucf", "xd"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    t0 = time.time()

    if args.dataset == "ucf":
        train_csv = os.path.join(LIST, "ucf_CLIP_rgb.csv")
        test_csv = os.path.join(LIST, "ucf_CLIP_rgbtest.csv")
        gt_path = os.path.join(LIST, "gt_ucf.npy")
        repeat = 16
    else:
        train_csv = os.path.join(LIST, "xd_CLIP_rgb.csv")
        test_csv = os.path.join(LIST, "xd_CLIP_rgbtest.csv")
        gt_path = os.path.join(LIST, "gt.npy")
        repeat = 1

    tr_feats, tr_labels, _ = load_train(train_csv, limit=args.limit or None)
    te_feats, gt_frame = load_test(test_csv, gt_path, repeat)
    print("train videos %d (pos %d) | test videos %d frames %d gt_pos %d"
          % (len(tr_feats), int(tr_labels.sum()), len(te_feats), len(gt_frame),
             int(gt_frame.sum())), flush=True)

    results = []

    # (0) degenerate video-constant score ------------------------------------
    vid = train_video(tr_feats, tr_labels, seed=args.seed, device=device)
    s_vid = score_video(vid, te_feats, device=device)
    const = [np.full(len(f), s, dtype=np.float32) for f, s in zip(te_feats, s_vid)]
    results.append(report(split_pairs(te_feats, const, gt_frame), "video-const"))

    # (1) raw snippet head ---------------------------------------------------
    m_raw = train_mil(tr_feats, tr_labels, centred=False, k=args.k,
                      epochs=args.epochs, seed=args.seed, hidden=args.hidden,
                      device=device)
    s_raw = score(m_raw, te_feats, device=device)
    results.append(report(split_pairs(te_feats, s_raw, gt_frame), "raw-MIL"))

    # (2) within-transformed snippet head ------------------------------------
    for mode in ("mean", "trimmed"):
        m_c = train_mil(tr_feats, tr_labels, centred=True, center_mode=mode,
                        k=args.k, epochs=args.epochs, seed=args.seed,
                        hidden=args.hidden, device=device)
        s_c = score(m_c, te_feats, centred=True, center_mode=mode, device=device)
        results.append(report(split_pairs(te_feats, s_c, gt_frame),
                              "within-%s" % mode))
        # (3) fusion with the video head: within effect + between effect
        z_c = zscores(s_c)
        z_v = zscores(s_vid)
        s_f = [c + v for c, v in zip(z_c, z_v)]
        results.append(report(split_pairs(te_feats, s_f, gt_frame),
                              "within-%s+video" % mode))

    print("elapsed %.1f s" % (time.time() - t0), flush=True)
    if args.out:
        df = pd.DataFrame(results)
        df.to_csv(args.out, index=False)
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
