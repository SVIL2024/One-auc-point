# -*- coding: utf-8 -*-
"""Stage-3: does an event signal exist *beyond* the within-video nuisance z?

Stage-2 found that the within-video term B is largely explained by motion
energy: a training-free 5-frame motion statistic reaches within-video AUC
0.7186 while the trained MIL head reaches 0.6690.  So the natural question is
identifiability, not capacity:

  Is there any frame-level evidence for the event that is *not* an increasing
  function of local motion energy?

We answer it with an orthogonality (deconfounding) constraint inside the video:

  L = BCE(topk(s), y)  +  gamma * mean_v corr^2(s_v , z_v)

with z_v the within-video nuisance block [t, ||dx||, 5-frame mean ||dx||,
||x||].  gamma = 0 is the field's estimator; gamma > 0 forces the score to be
linear-in-z-orthogonal inside every video, i.e. it forbids the model from
buying within-video discrimination with motion.  We then report

  B        within-video AUC (unconditional, what everyone reports)
  B|z      within-video AUC with pairs matched on z strata (conditional)
  theta    the benchmark number

If B|z at gamma=0 is ~0.5, the field's localization term is motion; if gamma>0
lifts B|z above the gamma=0 value, an event signal beyond motion exists and is
reachable by training.

Usage: python cafe_probe6.py --gammas 0,0.1,1,10 --epochs 40 --limit 4000
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
    raise SystemExit("cache missing: " + cache)


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


def pair_within(pairs, strat=None, nbins=5):
    num, den = 0.0, 0.0
    for k, (s, g) in enumerate(pairs):
        if g.min() == g.max():
            continue
        m = g == 1
        p, b = s[m], s[~m]
        if len(p) == 0 or len(b) == 0:
            continue
        if strat is None:
            num += len(p) * len(b) * roc_auc_score(np.r_[np.ones(len(p)),
                                                         np.zeros(len(b))],
                                                   np.r_[p, b])
            den += len(p) * len(b)
        else:
            z = strat[k]
            e = np.quantile(z, np.linspace(0, 1, nbins + 1))
            e[-1] += 1e-9
            for j in range(nbins):
                sel = (z >= e[j]) & (z < e[j + 1])
                pj, bj = p[sel[m]], b[sel[~m]]
                if len(pj) and len(bj):
                    num += len(pj) * len(bj) * roc_auc_score(
                        np.r_[np.ones(len(pj)), np.zeros(len(bj))],
                        np.r_[pj, bj])
                    den += len(pj) * len(bj)
    return num / den if den else float("nan")


def decompose(pairs, tag, store, strat=None):
    pos, negA, same = [], [], []
    for s, g in pairs:
        if g.sum() == 0:
            negA.append(s)
            continue
        m = g == 1
        pos.append(s[m])
        if (~m).any():
            same.append((s[m], s[~m]))
    P = np.concatenate(pos)
    A = np.concatenate(negA)
    Ball = np.concatenate([b for _, b in same])
    a_term = auc(np.r_[np.ones(len(P)), np.zeros(len(A))], np.r_[P, A])
    hard = auc(np.r_[np.ones(len(P)), np.zeros(len(Ball))], np.r_[P, Ball])
    w = len(A) / (len(A) + len(Ball))
    wp = sum(len(p) * len(b) for p, b in same)
    tot = len(P) * len(Ball)
    B = pair_within(pairs)
    C = (hard * tot - B * wp) / (tot - wp)
    th = w * a_term + (1 - w) * hard
    Bc = pair_within(pairs, strat=strat) if strat is not None else float("nan")
    d = dict(tag=tag, theta=th, A=a_term, B=B, C=C, Bcond=Bc, w=w,
             lam=wp / tot)
    store.append(d)
    print("[%-12s] theta %.4f | A %.4f B %.4f B|z %.4f C %.4f"
          % (tag, th, a_term, B, Bc, C), flush=True)
    return d


def zblocks(feats):
    out = []
    for x in feats:
        n = len(x)
        t = np.linspace(0, 1, n)
        dx = np.zeros_like(x)
        dx[1:] = x[1:] - x[:-1]
        mo = np.linalg.norm(dx, axis=1)
        ml = np.convolve(mo, np.ones(5) / 5.0, mode="same")
        fn = np.linalg.norm(x, axis=1)
        out.append(np.stack([t, mo, ml, fn], 1).astype(np.float32))
    return out


def standardize_z(Z):
    """per-video standardisation of the nuisance block (within-video centre)"""
    out = []
    for z in Z:
        mu, sd = z.mean(0, keepdims=True), z.std(0, keepdims=True) + 1e-8
        out.append((z - mu) / sd)
    return out


def train_orth(X, Y, Ztr, gamma=1.0, k=3, epochs=40, lr=1e-3, seed=0,
               device="cuda"):
    """MIL + within-video orthogonality of the score w.r.t. z."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X).to(device)
    Yt = torch.tensor(Y).to(device)
    Zt = torch.tensor(Ztr).to(device)          # (N,T,4) standardised per video
    n, t, _ = Xt.shape
    net = nn.Linear(Xt.shape[2], 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            b = perm[i:i + 32]
            s = net(Xt[b]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                s.topk(min(k, t), dim=1).values.mean(1), Yt[b])
            if gamma > 0:
                sc = s - s.mean(1, keepdim=True)
                zc = Zt[b] - Zt[b].mean(1, keepdim=True)
                num = (sc.unsqueeze(-1) * zc).mean(1)          # (B,4)
                den = sc.std(1, keepdim=True) * zc.std(1) + 1e-6
                r = num / (den * 1.0)
                loss = loss + gamma * (r ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net


@torch.no_grad()
def score(net, feats, device="cuda"):
    return [net(torch.tensor(f, device=device).unsqueeze(0)).squeeze(0)
            .squeeze(-1).cpu().numpy() for f in feats]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", default="0,0.3,1,3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    cache = "E:/program/CAFE/out/cache_ucf_train.npz"
    Xtr, ytr = build_train(os.path.join(LIST, "ucf_CLIP_rgb.csv"), cache)
    if args.limit and args.limit < len(Xtr):
        rng = np.random.RandomState(0)
        pos = np.where(ytr == 1)[0]
        neg = np.where(ytr == 0)[0]
        sel = np.r_[rng.choice(pos, min(len(pos), args.limit // 2), False),
                    rng.choice(neg, args.limit - min(len(pos), args.limit // 2),
                               False)]
        Xtr, ytr = Xtr[sel], ytr[sel]
    Ztr = standardize_z([zblocks([x])[0] for x in Xtr])
    feats, gt_frame = load_test(os.path.join(LIST, "ucf_CLIP_rgbtest.csv"),
                               os.path.join(LIST, "gt_ucf.npy"))
    Zte = standardize_z(zblocks(feats))
    mlocal = [z[:, 2] for z in Zte]
    print("device %s | train %d | test %d | gammas %s"
          % (device, len(Xtr), len(feats), args.gammas), flush=True)
    store = []
    for g in [float(x) for x in args.gammas.split(",")]:
        net = train_orth(Xtr, ytr, Ztr, gamma=g, epochs=args.epochs,
                         seed=args.seed, device=device)
        S = score(net, feats, device)
        pairs = [(np.asarray(S[i], float), gt_frame[sum(len(f) for f in
                                                        feats[:i]):
                                                    sum(len(f) for f in
                                                        feats[:i + 1])])
                 for i in range(len(feats))]
        d = decompose(pairs, "gamma=%.2g" % g, store, strat=mlocal)
        # correlation of the score with motion inside a video
        cs = [float(np.corrcoef(S[i] - S[i].mean(), mlocal[i])[0, 1])
              for i in range(len(S)) if mlocal[i].std() > 1e-9]
        d["corr_motion"] = float(np.mean(cs))
        print("             corr(score, mlocal) %.3f" % d["corr_motion"],
              flush=True)
    print("elapsed %.1f s" % (time.time() - t0), flush=True)
    if args.out:
        pd.DataFrame(store).to_csv(args.out, index=False)
        json.dump(store, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
