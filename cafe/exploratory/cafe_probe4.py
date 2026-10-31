# -*- coding: utf-8 -*-
"""Stage-2 probe: the within-video contrast as a *causal* estimand.

The reported metric decomposes exactly (see out/atlas.json) into three
pair-sets with data-fixed weights on UCF test:

    theta = 0.6317 * A  +  0.0039 * B  +  0.3644 * C
    A = AUC(pos , normal-video frames)          (cross-video, "scene")
    B = AUC(pos , same-video background)        (within-video, V held fixed)
    C = AUC(pos , other anomalous videos' bg)   (cross-video)

Only B is a causal contrast with the video identity V fixed.  This probe asks
two questions:

Q1 (confounding): how much of B is explained by within-video nuisance z
   (time position, motion magnitude, feature norm) rather than by the event?
   -> report the within-video AUC of z alone, and of the score after removing
      z fitted on *control* frames with cross-fitting.

Q2 (adjustment): does a cross-fitted control-fitted residualisation of the
   score raise B, and what does it do to theta (prediction from the law:
   (1-w)*lam = 0.0039 per unit of B, i.e. nothing measurable).

Usage: python cafe_probe4.py --dataset ucf --epochs 40 --limit 4000
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

LIST = "E:/program/DSANet/list"
T = 200


def uniform_extract(feat, t_max=T):
    new = np.zeros((t_max, feat.shape[1]), dtype=np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    for i in range(t_max):
        if r[i] != r[i + 1]:
            new[i] = feat[r[i]:r[i + 1]].mean(0)
        else:
            new[i] = feat[r[i]]
    return new


def build_train(csv, cache):
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"].astype(np.float32), d["y"].astype(np.float32)
    df = pd.read_csv(csv)
    X = np.zeros((len(df), T, 512), dtype=np.float16)
    y = np.zeros(len(df), dtype=np.float32)
    for i, (p, lab) in enumerate(zip(df["path"], df["label"])):
        f = np.load(p).astype(np.float32)
        X[i] = uniform_extract(f) if len(f) > T else np.pad(
            f, ((0, T - len(f)), (0, 0)))
        y[i] = 0.0 if str(lab).strip() == "Normal" else 1.0
    np.savez(cache, X=X, y=y)
    return X.astype(np.float32), y


def load_test(csv, gt_path, repeat=16):
    df = pd.read_csv(csv)
    feats = [np.load(p).astype(np.float32) for p in df["path"]]
    gt = np.load(gt_path)
    gt_frame = (gt.reshape(-1, 16)[:, 0] if repeat == 16
                else np.asarray(gt).reshape(-1)).astype(np.int64)
    total = sum(len(f) for f in feats)
    if len(gt_frame) != total:
        n = min(len(gt_frame), total)
        keep, acc = [], 0
        for f in feats:
            if acc + len(f) <= n:
                keep.append(f)
                acc += len(f)
            else:
                break
        feats, gt_frame = keep, gt_frame[:acc]
    return feats, gt_frame


def auc(y, p):
    y = np.asarray(y)
    return (float(roc_auc_score(y, p)) if y.size and y.min() != y.max()
            else float("nan"))


def decompose(pairs, tag, store=None, extra=None):
    """Exact 3-term decomposition over disjoint pair sets."""
    pos, negA, negB_same, negB_cross = [], [], [], []
    within, within_pairs_num, within_pairs_den = [], 0.0, 0.0
    for s, g in pairs:
        if g.sum() == 0:
            negA.append(s)
            continue
        m = g == 1
        p = s[m]
        b = s[~m]
        if len(b) == 0:
            continue
        pos.append(p)
        negB_cross.append(b)                      # cross pairs handled below
        negB_same.append((p, b))
        if g.min() != g.max():
            within.append(roc_auc_score(g, s))
        within_pairs_num += float(len(p) * len(b))
    P = np.concatenate(pos)
    A = np.concatenate(negA)
    B_all = np.concatenate(negB_cross)
    # A term
    a_term = auc(np.r_[np.ones(len(P)), np.zeros(len(A))], np.r_[P, A])
    # B term (same-video pairs) : pair-count-weighted average of per-video AUCs
    num = 0.0
    for p, b in negB_same:
        if len(b):
            num += float(len(p) * len(b)) * auc(np.r_[np.ones(len(p)),
                                                     np.zeros(len(b))],
                                                np.r_[p, b])
    lam = within_pairs_num / (len(B_all) * len(P))
    b_term = num / within_pairs_num if within_pairs_num else float("nan")
    c_term = num  # placeholder
    # C term: pos vs background of OTHER anomalous videos, exact pair count
    # total B-pool pairs -> same (num pairs) + cross pairs; AUC over the union
    hard = auc(np.r_[np.ones(len(P)), np.zeros(len(B_all))], np.r_[P, B_all])
    w = len(A) / (len(A) + len(B_all))
    theta = w * a_term + (1 - w) * hard
    # cross term by subtraction is only valid on pair counts:
    tot_pairs = within_pairs_num + (len(B_all) - 0.0) * len(P) - within_pairs_num
    c_pairs = tot_pairs - within_pairs_num
    c_term = (hard * tot_pairs - num) / c_pairs if c_pairs > 1e-9 else float("nan")
    d = dict(tag=tag, theta=theta, w=w, A=a_term, B=b_term, C=c_term, hard=hard,
             lam=lam, tau_mean=float(np.nanmean(within)),
             tau_median=float(np.nanmedian(within)),
             n_videos=len(within))
    if store is not None:
        store.append(d)
    print("[%s] theta %.4f = %.4f*A(%.4f) + %.5f*B(%.4f) + %.4f*C(%.4f)"
          " | tau_mean %.4f"
          % (tag, theta, w, a_term, (1 - w) * lam, b_term, (1 - w) * (1 - lam),
             c_term, d["tau_mean"]), flush=True)
    if extra:
        print("      extra:", extra, flush=True)
    return d


def zfeat(x):
    """within-video nuisance covariates z (T, d)."""
    n = len(x)
    t = np.linspace(0, 1, n)[:, None]
    dx = np.zeros_like(x)
    dx[1:] = x[1:] - x[:-1]
    motion = np.linalg.norm(dx, axis=1, keepdims=True)
    motion_local = np.zeros_like(motion)
    for i in range(n):
        lo, hi = max(0, i - 4), min(n, i + 5)
        motion_local[i] = motion[lo:hi].mean()
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.concatenate([t, motion, motion_local, norm], 1)


def ridge_fit(Z, s, alpha=1.0):
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Zs = (Z - mu) / sd
    A_ = Zs.T @ Zs + alpha * np.eye(Zs.shape[1])
    b_ = Zs.T @ s
    coef = np.linalg.solve(A_, b_)
    return (mu, sd, coef)


def ridge_pred(model, Z):
    mu, sd, coef = model
    return ((Z - mu) / sd) @ coef


def train_raw(X, Y, k=3, epochs=40, lr=1e-3, seed=0, device="cuda"):
    torch.manual_seed(seed)
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    net = nn.Linear(X.shape[2], 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n, t, _ = X.shape
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            b = perm[i:i + 32]
            s = net(X[b]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                s.topk(min(k, t), dim=1).values.mean(1), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net


@torch.no_grad()
def score(net, feats, device="cuda"):
    out = []
    for f in feats:
        x = torch.tensor(f, device=device).unsqueeze(0)
        out.append(net(x).squeeze(0).cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucf", choices=["ucf", "xd"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tr = os.path.join(LIST, "ucf_CLIP_rgb.csv")
    te = os.path.join(LIST, "ucf_CLIP_rgbtest.csv")
    gt = os.path.join(LIST, "gt_ucf.npy")
    cache = "E:/program/CAFE/out/cache_ucf_train.npz"
    Xtr, ytr = build_train(tr, cache)
    if args.limit and args.limit < len(Xtr):
        rng = np.random.RandomState(0)
        pos = np.where(ytr == 1)[0]
        neg = np.where(ytr == 0)[0]
        sel = np.r_[rng.choice(pos, min(len(pos), args.limit // 2), False),
                    rng.choice(neg, args.limit - min(len(pos), args.limit // 2),
                               False)]
        Xtr, ytr = Xtr[sel], ytr[sel]
    feats, gt_frame = load_test(te, gt)
    print("device %s | train %d | test %d videos %d frames"
          % (device, len(Xtr), len(feats), len(gt_frame)), flush=True)
    net = train_raw(Xtr, ytr, epochs=args.epochs, seed=args.seed, device=device)
    S = score(net, feats, device)

    # per-video arrays
    off = 0
    pairs, Zs, Gs = [], [], []
    for f, s in zip(feats, S):
        g = gt_frame[off:off + len(f)]
        off += len(f)
        pairs.append((np.asarray(s, float), g))
        Zs.append(zfeat(f))
        Gs.append(g)

    store = []
    decompose(pairs, "raw", store)

    # Q2a: how well does z alone rank inside a video?
    for name, col in [("time", [0]), ("motion", [1]), ("motion_local", [2]),
                      ("fnorm", [3])]:
        zs_pairs = [(Zs[i][:, col].sum(1), Gs[i]) for i in range(len(Zs))]
        d = decompose(zs_pairs, "z-" + name, None)
        d["pooled_only"] = None

    # Q2b: global ridge on all frames, cross-fitted by video parity
    def adjust(fit_mask_fn, tag, store, extra_note=""):
        S2 = [None] * len(S)
        for parity in (0, 1):
            fit_idx = [i for i in range(len(S)) if i % 2 == parity]
            app_idx = [i for i in range(len(S)) if i % 2 != parity]
            Zf = np.concatenate([Zs[i][fit_mask_fn(i, S[i], Gs[i])]
                                 for i in fit_idx])
            sf = np.concatenate([S[i][fit_mask_fn(i, S[i], Gs[i])]
                                 for i in fit_idx])
            model = ridge_fit(Zf, sf)
            for i in app_idx:
                S2[i] = S[i] - ridge_pred(model, Zs[i])
        p2 = [(S2[i], Gs[i]) for i in range(len(S))]
        decompose(p2, tag, store, extra_note)
        return S2

    adjust(lambda i, s, g: np.ones(len(s), bool), "adj-all", store)
    adjust(lambda i, s, g: s <= np.median(s), "adj-control-low", store,
           "fit on frames below the video's median score")
    adjust(lambda i, s, g: s <= np.quantile(s, 0.75), "adj-control-q75", store)

    # Q2c: reference - per-video centring of the score
    p3 = [(S[i] - S[i].mean(), Gs[i]) for i in range(len(S))]
    decompose(p3, "score-centred", store)

    print("elapsed %.1f s" % (time.time() - t0), flush=True)
    if args.out:
        pd.DataFrame(store).to_csv(args.out, index=False)
        json.dump(store, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
