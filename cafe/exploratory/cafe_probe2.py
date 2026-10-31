# -*- coding: utf-8 -*-
"""Stage-1b probe: does a *causally motivated* estimator beat plain MIL?

Key identity verified on this server (out/probe_ucf.csv):

  pooled AUC = w * AUC(pos vs negatives from NORMAL videos)     [between-video]
             + (1-w) * AUC(pos vs BACKGROUND of anomalous vids)  [within-video]
  w = 0.6317 on UCF-Crime test, a pure data property.

  First probe: top-k MIL (linear head) -> within 0.656; a *feature-centred* head
  (x - mean_t x) collapses to 0.314, i.e. the anomaly signal in frozen CLIP
  features lives mostly in the video mean, so the fixed-effect transform has to
  act on a learned score, not on the raw features.

Arms
  video    : g(mean_t x) only, constant per video (degenerate: within = 0.5)
  raw      : top-k MIL on f(x)                        (the field's estimator)
  fus      : f + g, both trained with video-level BCE (Mundlak-style sum)
  contrast : f trained with the WITHIN-VIDEO contrast max_k - mean, g with the
             video label (between part left to g alone)
  cmil     : contrast + top-k MIL on f

Usage: python cafe_probe2.py --dataset ucf --epochs 40 --limit 4000 --log out/p2.log
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
        if i % 2000 == 0:
            print("  loaded %d/%d" % (i, len(df)), flush=True)
    np.savez(cache, X=X, y=y)
    print("wrote cache", cache, flush=True)
    return X.astype(np.float32), y


def load_test(csv, gt_path, repeat):
    df = pd.read_csv(csv)
    feats = [np.load(p).astype(np.float32) for p in df["path"]]
    gt = np.load(gt_path)
    gt_frame = (gt.reshape(-1, 16)[:, 0] if repeat == 16
                else np.asarray(gt).reshape(-1)).astype(np.int64)
    total = sum(len(f) for f in feats)
    if len(gt_frame) != total:
        print("[warn] gt %d != frames %d" % (len(gt_frame), total), flush=True)
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
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


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
             delta_pooled_from_within=(1 - w) * (np.nanmean(wth) - 0.5))
    print("[%-16s] pooled %.4f | easy %.4f hard %.4f | within %.4f (med %.4f)"
          " | identity %.4f" % (tag, pooled, a_easy, a_hard, d["within_mean"],
                                d["within_median"], d["identity"]), flush=True)
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


def train(arm, X, Y, k=3, margin=1.0, lam=1.0, epochs=40, lr=1e-3, seed=0,
          hidden=0, device="cuda", tag=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    n, t, d = X.shape
    net = Net(d, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    print("  %s: %d videos, T=%d" % (arm, n, t), flush=True)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 32):
            b = perm[i:i + 32]
            x = X[b]
            y = Y[b]
            s = net.snippet(x)
            gv = net.g(x.mean(1)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(gv, y)
            if arm in ("raw", "cmil"):
                m = s.topk(min(k, t), dim=1).values.mean(1)
                loss = loss + F.binary_cross_entropy_with_logits(m, y)
            if arm in ("contrast", "cmil"):
                m = s.topk(min(k, t), dim=1).values.mean(1)
                stat = m - s.mean(1)
                cv = y * F.relu(margin - stat) + (1 - y) * F.relu(margin + stat)
                loss = loss + lam * cv.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if ep % 5 == 0 or ep == epochs - 1:
            print("  %s ep %02d loss %.4f" % (tag or arm, ep, tot / n),
                  flush=True)
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
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="video,raw,fus,contrast,cmil")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
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
    print("train %d (pos %d) | test %d frames %d pos %d"
          % (len(Xtr), int(ytr.sum()), len(te_feats), len(gt_frame),
             int(gt_frame.sum())), flush=True)

    store = []
    sf_cache, sg_cache = {}, {}
    for arm in args.arms.split(","):
        if arm == "video":
            net = train("video", Xtr, ytr, epochs=args.epochs, seed=args.seed,
                        device=device)
            sf, sg = score(net, te_feats, device)
            const = [np.full(len(f), v) for f, v in zip(te_feats, sg)]
            report(split_pairs(te_feats, const, gt_frame), "video", store)
            sf_cache[arm], sg_cache[arm] = sf, sg
            continue
        net = train(arm, Xtr, ytr, k=args.k, margin=args.margin, lam=args.lam,
                    epochs=args.epochs, seed=args.seed, hidden=args.hidden,
                    device=device)
        sf, sg = score(net, te_feats, device)
        sf_cache[arm], sg_cache[arm] = sf, sg
        report(split_pairs(te_feats, sf, gt_frame), arm + "-f", store)
        if arm in ("raw", "cmil"):
            m = [np.full(len(f), s.max()) for f, s in zip(te_feats, sf)]
            report(split_pairs(te_feats, m, gt_frame), arm + "-max", store)
        if "video" in sf_cache:
            zf, zg = zs(sf), zs(sg_cache["video"])
            report(split_pairs(te_feats, [a + b for a, b in zip(zf, zg)],
                               gt_frame), arm + "+vid", store)

    print("elapsed %.1f s" % (time.time() - t0), flush=True)
    if args.out:
        pd.DataFrame(store).to_csv(args.out, index=False)
        with open(args.out + ".json", "w") as fh:
            json.dump(store, fh, indent=1)
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
