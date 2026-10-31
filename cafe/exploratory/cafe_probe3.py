# -*- coding: utf-8 -*-
"""Stage-1c probe: which weak-label loss identifies the WITHIN-video functional?

Stage-1 established (see out/probe_ucf.csv, out/probe2_ucf.csv) that

  pooled AUC = w*easy + (1-w)*hard,   w = 0.6317 (UCF test, data property)
  w*easy + (1-w)*hard splits again as  hard = lam*tau_w + (1-lam)*theta_out
  with lam = 0.0105 (out/atlas.json)  =>  1 reported AUC point ~= 258 tau points.

and that a per-video-constant scorer (a plain video classifier) already reaches
pooled 0.82 with tau = 0.50 exactly, 1.1 points below a real MIL detector.

Stage-1b also showed that a *fixed-effect* style loss (hinge on max_k - mean,
no video-level anchor) is hopeless: tau drops to 0.31, the score is unoriented.
That is the point: from video labels alone the within-video ordering is
identified only up to orientation -- you cannot pin the level.

Here we test the loss that *is* identified by video labels: put the video label
on a shift-invariant statistic of the within-video score profile, i.e. on

    kappa_v = mean_{top k} s_v - mean_t s_v        (Mundlak / contrast MIL)

Because kappa is invariant to any per-video offset, the network cannot buy the
video label with a video-level cue, so what it learns has to be usable *inside*
a video.  Arms:

  video    : per-video constant  (baseline: tau = 0.5 by construction)
  raw      : ordinary top-k MIL on s            (the field's estimator)
  kappa    : BCE(kappa, y)
  mix      : BCE(top-k, y) + BCE(kappa, y)
  all      : kappa + video head g(mean x) fused at test time

Usage: python cafe_probe3.py --dataset ucf --epochs 40 --limit 4000
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
        print("cache hit", cache, d["X"].shape, flush=True)
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


def load_test(csv, gt_path, repeat):
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


def split_pairs(feats, scores, gt_frame):
    off, out = 0, []
    for f, s in zip(feats, scores):
        g = gt_frame[off:off + len(f)]
        off += len(f)
        out.append((np.asarray(s, dtype=np.float64), g))
    return out


def report(pairs, tag, store):
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
    pooled = auc(np.r_[np.ones(len(P)), np.zeros(len(A) + len(B))],
                 np.r_[P, A, B])
    a_easy = auc(np.r_[np.ones(len(P)), np.zeros(len(A))], np.r_[P, A])
    a_hard = auc(np.r_[np.ones(len(P)), np.zeros(len(B))], np.r_[P, B])
    wth = np.asarray(within)
    d = dict(tag=tag, pooled_auc=pooled, w=w, auc_easy=a_easy, auc_hard=a_hard,
             identity=w * a_easy + (1 - w) * a_hard,
             within_mean=float(np.nanmean(wth)),
             within_median=float(np.nanmedian(wth)),
             within_frac_below_half=float((wth < 0.5).mean()))
    print("[%-14s] pooled %.4f | easy %.4f hard %.4f | within %.4f (med %.4f)"
          " frac<.5 %.2f | ident %.4f"
          % (tag, pooled, a_easy, a_hard, d["within_mean"],
             d["within_median"], d["within_frac_below_half"], d["identity"]),
          flush=True)
    store.append(d)
    return d


def zs(list_of_arrays):
    arrs = [np.asarray(a, dtype=np.float64) for a in list_of_arrays]
    allv = np.concatenate([a.reshape(-1) for a in arrs])
    mu, sd = allv.mean(), allv.std() + 1e-9
    return [(a - mu) / sd for a in arrs]


class Net(nn.Module):
    def __init__(self, dim=512, hidden=0):
        super().__init__()
        self.f = (nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                nn.Linear(hidden, 1)) if hidden
                  else nn.Linear(dim, 1))
        self.g = nn.Linear(dim, 1)

    def snippet(self, x):
        return self.f(x).squeeze(-1)


def kappa(s, k):
    kk = min(k, s.shape[1])
    return s.topk(kk, dim=1).values.mean(1) - s.mean(1)


def train(arm, X, Y, k=3, epochs=40, lr=1e-3, seed=0, hidden=0, mu=1.0,
          device="cuda", log=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    n, t, d = X.shape
    net = Net(d, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    if log:
        print("  %s: %d videos T=%d hidden=%d" % (arm, n, t, hidden), flush=True)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 32):
            b = perm[i:i + 32]
            x, y = X[b], Y[b]
            s = net.snippet(x)
            if arm == "video":
                loss = F.binary_cross_entropy_with_logits(
                    net.g(x.mean(1)).squeeze(-1), y)
            elif arm == "raw":
                loss = F.binary_cross_entropy_with_logits(
                    s.topk(min(k, t), dim=1).values.mean(1), y)
            elif arm == "kappa":
                loss = F.binary_cross_entropy_with_logits(kappa(s, k), y)
            elif arm == "mix":
                loss = (F.binary_cross_entropy_with_logits(
                    s.topk(min(k, t), dim=1).values.mean(1), y)
                    + F.binary_cross_entropy_with_logits(kappa(s, k), y))
            elif arm == "all":
                loss = (F.binary_cross_entropy_with_logits(kappa(s, k), y)
                        + mu * F.binary_cross_entropy_with_logits(
                            net.g(x.mean(1)).squeeze(-1), y))
            else:
                raise ValueError(arm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if log and (ep % 10 == 0 or ep == epochs - 1):
            print("  %s ep %02d loss %.4f" % (arm, ep, tot / n), flush=True)
    net.eval()
    return net


@torch.no_grad()
def score(net, feats, device="cuda"):
    sf, sg = [], []
    for f in feats:
        x = torch.tensor(f, device=device).unsqueeze(0)
        sf.append(net.snippet(x).squeeze(0).cpu().numpy())
        sg.append(float(net.g(x.mean(1)).squeeze().item()))
    return sf, sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucf", choices=["ucf", "xd"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--arms", default="video,raw,kappa,mix,all")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if args.dataset == "ucf":
        tr = os.path.join(LIST, "ucf_CLIP_rgb.csv")
        te = os.path.join(LIST, "ucf_CLIP_rgbtest.csv")
        gt = os.path.join(LIST, "gt_ucf.npy")
        rep = 16
    else:
        tr = os.path.join(LIST, "xd_CLIP_rgb.csv")
        te = os.path.join(LIST, "xd_CLIP_rgbtest.csv")
        gt = os.path.join(LIST, "gt.npy")
        rep = 1

    cache = "E:/program/CAFE/out/cache_%s_train.npz" % args.dataset
    Xtr, ytr = build_train(tr, cache)
    if args.limit and args.limit < len(Xtr):
        rng = np.random.RandomState(0)
        pos = np.where(ytr == 1)[0]
        neg = np.where(ytr == 0)[0]
        npos = min(len(pos), args.limit // 2)
        nneg = min(len(neg), args.limit - npos)
        sel = np.r_[rng.choice(pos, npos, replace=False),
                    rng.choice(neg, nneg, replace=False)]
        Xtr, ytr = Xtr[sel], ytr[sel]
    te_feats, gt_frame = load_test(te, gt, rep)
    print("device %s | train %d (pos %d) | test %d frames %d pos %d"
          % (device, len(Xtr), int(ytr.sum()), len(te_feats), len(gt_frame),
             int(gt_frame.sum())), flush=True)

    store = []
    for arm in args.arms.split(","):
        net = train(arm, Xtr, ytr, k=args.k, epochs=args.epochs, seed=args.seed,
                    hidden=args.hidden, device=device)
        sf, sg = score(net, te_feats, device)
        report(split_pairs(te_feats, sf, gt_frame), arm + "-s", store)
        if arm in ("video",):
            const = [np.full(len(f), v) for f, v in zip(te_feats, sg)]
            report(split_pairs(te_feats, const, gt_frame), "video-const", store)
            continue
        if arm in ("raw", "mix", "kappa"):
            m = [np.full(len(f), s.max()) for f, s in zip(te_feats, sf)]
            report(split_pairs(te_feats, m, gt_frame), arm + "-max", store)
        zf, zg = zs(sf), zs(sg)
        report(split_pairs(te_feats, [a + b for a, b in zip(zf, zg)],
                           gt_frame), arm + "+g", store)

    print("elapsed %.1f s" % (time.time() - t0), flush=True)
    if args.out:
        pd.DataFrame(store).to_csv(args.out, index=False)
        with open(args.out + ".json", "w") as fh:
            json.dump(store, fh, indent=1)


if __name__ == "__main__":
    main()
