# -*- coding: utf-8 -*-
"""CAFE main probe: nuisance-orthogonal within-video training, multi-seed.

L = BCE(topk(S_v), y_v) + gamma * mean_v corr^2(S_v, Z_v)

Reports the exact decomposition (theta, A, B, C) plus the motion-stratified
within-video term B|z, and saves the per-frame scores so that the inference
script (cafe_evalue.py: cluster bootstrap + E-value) can run without retraining.

Usage:
  python cafe_probe7.py --gammas 0,0.3,1 --seeds 0,1,2 --z mlocal
  python cafe_probe7.py --gammas 1 --seeds 0 --z t,motion,mlocal,fnorm --placebo
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
ZKEYS = ["t", "motion", "mlocal", "fnorm"]


def uniform_extract(feat, t_max=T):
    new = np.zeros((t_max, feat.shape[1]), dtype=np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    for i in range(t_max):
        if r[i] != r[i + 1]:
            new[i] = feat[r[i]:r[i + 1]].mean(0)
        else:
            new[i] = feat[r[i]]
    return new


def load_sets(ds, limit=4000, seed=0):
    if ds == "ucf":
        tr_csv = os.path.join(LIST, "ucf_CLIP_rgb.csv")
        te_csv = os.path.join(LIST, "ucf_CLIP_rgbtest.csv")
        gt_path = os.path.join(LIST, "gt_ucf.npy")
    else:
        tr_csv = os.path.join(LIST, "xd_CLIP_rgb.csv")
        te_csv = os.path.join(LIST, "xd_CLIP_rgbtest.csv")
        gt_path = os.path.join(LIST, "gt.npy")
    cache = "E:/program/CAFE/out/cache_%s_train.npz" % ds
    if os.path.exists(cache):
        d = np.load(cache)
        X, Y = d["X"].astype(np.float32), d["y"].astype(np.float32)
    else:
        df = pd.read_csv(tr_csv)
        X = np.zeros((len(df), T, 512), dtype=np.float16)
        Y = np.zeros(len(df), dtype=np.float32)
        for i, (p, lab) in enumerate(zip(df["path"], df["label"])):
            f = np.load(p).astype(np.float32)
            X[i] = uniform_extract(f) if len(f) > T else np.pad(
                f, ((0, T - len(f)), (0, 0)))
            Y[i] = 0.0 if str(lab).strip() == "Normal" else 1.0
        np.savez(cache, X=X, y=Y)
        X, Y = X.astype(np.float32), Y.astype(np.float32)
    if limit and limit < len(X):
        rng = np.random.RandomState(seed)
        pos = np.where(Y == 1)[0]
        neg = np.where(Y == 0)[0]
        k = min(len(pos), limit // 2)
        sel = np.r_[rng.choice(pos, k, False), rng.choice(neg, limit - k, False)]
        X, Y = X[sel], Y[sel]
    df = pd.read_csv(te_csv)
    feats = [np.load(p).astype(np.float32) for p in df["path"]]
    gt = np.load(gt_path).reshape(-1).astype(np.int64)
    gt = gt.reshape(-1, 16)[:, 0] if len(gt) == 16 * sum(
        len(f) for f in feats) else gt
    return X, Y, feats, gt


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


def standardize(Z):
    out = []
    for z in Z:
        mu, sd = z.mean(0, keepdims=True), z.std(0, keepdims=True) + 1e-8
        out.append((z - mu) / sd)
    return out


def pair_within(pairs, strat=None, nbins=5):
    num = den = 0.0
    for k, (s, g) in enumerate(pairs):
        if g.min() == g.max():
            continue
        m = g == 1
        p, b = s[m], s[~m]
        if not len(p) or not len(b):
            continue
        if strat is None:
            num += len(p) * len(b) * roc_auc_score(
                np.r_[np.ones(len(p)), np.zeros(len(b))], np.r_[p, b])
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


def decompose(pairs, strat=None):
    pos, negA, same = [], [], []
    for s, g in pairs:
        if g.sum() == 0:
            negA.append(s)
            continue
        m = g == 1
        pos.append(s[m])
        if (~m).any():
            same.append((s[m], s[~m]))
    P, A = np.concatenate(pos), np.concatenate(negA)
    Ball = np.concatenate([b for _, b in same])
    a = roc_auc_score(np.r_[np.ones(len(P)), np.zeros(len(A))], np.r_[P, A])
    hard = roc_auc_score(np.r_[np.ones(len(P)), np.zeros(len(Ball))],
                         np.r_[P, Ball])
    w = len(A) / (len(A) + len(Ball))
    wp = sum(len(p) * len(b) for p, b in same)
    tot = len(P) * len(Ball)
    B = pair_within(pairs)
    return dict(theta=float(w * a + (1 - w) * hard), A=float(a), B=float(B),
                B_cond=float(pair_within(pairs, strat=strat))
                if strat is not None else None,
                C=float((hard * tot - B * wp) / (tot - wp)),
                w=float(w), lam=float(wp / tot))


def train(X, Y, Z, gamma, k=3, epochs=40, lr=1e-3, seed=0, device="cuda"):
    torch.manual_seed(seed)
    Xt = torch.tensor(X).to(device)
    Yt = torch.tensor(Y).to(device)
    Zt = torch.tensor(Z).to(device) if Z is not None else None
    n, t, _ = Xt.shape
    net = nn.Linear(Xt.shape[2], 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            b = perm[i:i + 32]
            s = net(Xt[b]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                s.topk(min(k, t), dim=1).values.mean(1), Yt[b])
            if gamma > 0 and Zt is not None:
                sc = s - s.mean(1, keepdim=True)
                zc = Zt[b] - Zt[b].mean(1, keepdim=True)
                num = (sc.unsqueeze(-1) * zc).mean(1)
                den = sc.std(1, keepdim=True) * zc.std(1) + 1e-6
                loss = loss + gamma * (((num / den) ** 2).mean())
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
    ap.add_argument("--dataset", default="ucf")
    ap.add_argument("--gammas", default="0,0.3,1")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--z", default="mlocal")
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    X, Y, feats, gt = load_sets(args.dataset, args.limit)
    Ztr_full = standardize(zblocks([np.pad(x, ((0, 0), (0, 0)))
                                    for x in X.astype(np.float32)]))
    Zte_full = standardize(zblocks(feats))
    cols = [ZKEYS.index(k.strip()) for k in args.z.split(",") if k.strip()]
    Ztr = [np.ascontiguousarray(z[:, cols]) for z in Ztr_full]
    Zte = [np.ascontiguousarray(z[:, cols]) for z in Zte_full]
    if args.placebo:                       # break the within-video link
        rng = np.random.RandomState(12345)
        perm = rng.permutation(len(Ztr))
        Ztr = [Ztr[p] for p in perm]
    mlocal = [z[:, ZKEYS.index("mlocal")] for z in Zte_full]
    lens = [len(f) for f in feats]
    print("device %s | train %d | test %d videos | z=%s placebo=%s"
          % (device, len(X), len(feats), args.z, args.placebo), flush=True)
    store, scores = [], {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        for g in [float(x) for x in args.gammas.split(",")]:
            net = train(X, Y, Ztr, g, epochs=args.epochs, seed=seed,
                        device=device)
            S = score(net, feats, device)
            off = 0
            pairs = []
            for i in range(len(S)):
                pairs.append((np.asarray(S[i], float), gt[off:off + lens[i]]))
                off += lens[i]
            d = decompose(pairs, strat=mlocal)
            cs = [float(np.corrcoef(S[i] - S[i].mean(), mlocal[i])[0, 1])
                  for i in range(len(S)) if mlocal[i].std() > 1e-9]
            d.update(gamma=g, seed=seed, dataset=args.dataset, z=args.z,
                     placebo=bool(args.placebo),
                     corr_motion=float(np.mean(cs)),
                     within_mean=float(np.nanmean(
                         [roc_auc_score(p[1], p[0]) for p in pairs
                          if p[1].min() != p[1].max()])))
            store.append(d)
            key = "%s_g%g_s%d%s" % (args.dataset, g, seed,
                                    "_placebo" if args.placebo else "")
            scores[key] = dict(scores=[np.asarray(s, np.float32).tolist()
                                       for s in S],
                               lens=lens, gt=gt.tolist())
            print("[%s] gamma %.2f seed %d | theta %.4f A %.4f B %.4f "
                  "B|z %.4f C %.4f | corr %.3f within_mean %.4f"
                  % (args.dataset, g, seed, d["theta"], d["A"], d["B"],
                     d["B_cond"], d["C"], d["corr_motion"],
                     d["within_mean"]), flush=True)
    print("elapsed %.1f s" % (time.time() - t0), flush=True)
    if args.out:
        base = args.out.rstrip(".json")
        pd.DataFrame(store).to_csv(base + ".csv", index=False)
        json.dump(store, open(base + ".json", "w"), indent=1)
        np.savez_compressed(base + "_scores.npz",
                            **{k + "|s": np.asarray(v["scores"], np.float32)
                               for k, v in scores.items()},
                            lens=np.asarray(lens), gt=gt,
                            mlocal=np.concatenate(mlocal).astype(np.float32))
        print("saved", base, flush=True)


if __name__ == "__main__":
    main()
