# -*- coding: utf-8 -*-
"""Stage-2b probe: the within-video term as a *causally adjusted* estimand.

Reported metric (exact, UCF-Crime test, out/atlas.json):

    theta = w*A + (1-w)*[lam*B + (1-lam)*C]
          = 0.6317*A + 0.00388*B + 0.3645*C

Only B (same-video pairs) holds the video identity V fixed, so B is the only
term that can measure localization.  B is also confounded *inside* the video by
nuisance z (motion energy, time position, feature norm): an anomaly entails
motion, but motion also raises a motion-sensitive score on its own.

This probe estimates B three ways and reports the gap:
  B        : unconditional within-video AUC (what everyone implicitly reports)
  B|z      : motion-stratified within-video AUC (pairs matched on z-decile)
  B_res    : cross-fitted, control-fitted residualisation of the score on z

and it asks whether a *training-free* z statistic (local motion energy) already
matches the trained model:
  B(z)     : within-video AUC of z alone.

Usage: python cafe_probe5.py --dataset ucf --epochs 40 --limit 4000
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


def pair_weighted_within(pairs, strat=None, nbins=5):
    """Pair-count weighted within-video AUC, optionally stratified on z.

    strat: list of 1-D arrays aligned with each pair's frames, or None.
    Returns (auc, n_pairs, n_videos).
    """
    num, den, nv = 0.0, 0.0, 0
    for k, (s, g) in enumerate(pairs):
        if g.min() == g.max():
            continue
        m = g == 1
        p, b = s[m], s[~m]
        if len(p) == 0 or len(b) == 0:
            continue
        nv += 1
        if strat is None:
            num += len(p) * len(b) * roc_auc_score(np.r_[np.ones(len(p)),
                                                         np.zeros(len(b))],
                                                   np.r_[p, b])
            den += len(p) * len(b)
        else:
            z = strat[k]
            edges = np.quantile(z, np.linspace(0, 1, nbins + 1))
            edges[-1] += 1e-9
            for j in range(nbins):
                sel = (z >= edges[j]) & (z < edges[j + 1])
                pj, bj = p[sel[m]], b[sel[~m]]
                if len(pj) and len(bj):
                    num += len(pj) * len(bj) * roc_auc_score(
                        np.r_[np.ones(len(pj)), np.zeros(len(bj))],
                        np.r_[pj, bj])
                    den += len(pj) * len(bj)
    return (num / den if den else float("nan")), den, nv


def decompose(pairs, tag, store=None, note=""):
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
    B_all = np.concatenate([b for _, b in same])
    a_term = auc(np.r_[np.ones(len(P)), np.zeros(len(A))], np.r_[P, A])
    hard = auc(np.r_[np.ones(len(P)), np.zeros(len(B_all))], np.r_[P, B_all])
    w = len(A) / (len(A) + len(B_all))
    wp = 0.0
    for p, b in same:
        wp += len(p) * len(b)
    tot = len(P) * len(B_all)
    lam = wp / tot
    b_term = pair_weighted_within([(s, g) for s, g in pairs])[0]
    c_term = (hard * tot - b_term * wp) / (tot - wp)
    theta = w * a_term + (1 - w) * hard
    d = dict(tag=tag, theta=theta, w=w, A=a_term, B=b_term, C=c_term,
             hard=hard, lam=lam, note=note)
    if store is not None:
        store.append(d)
    print("[%-16s] theta %.4f = %.4f*A(%.4f) + %.5f*B(%.4f) + %.4f*C(%.4f)"
          % (tag, theta, w, a_term, (1 - w) * lam, b_term, (1 - w) * (1 - lam),
             c_term), flush=True)
    return d


def zfeat(x):
    n = len(x)
    t = np.linspace(0, 1, n)
    dx = np.zeros_like(x)
    dx[1:] = x[1:] - x[:-1]
    motion = np.linalg.norm(dx, axis=1)
    mlocal = np.convolve(motion, np.ones(5) / 5.0, mode="same")
    fnorm = np.linalg.norm(x, axis=1)
    return t, motion, mlocal, fnorm


def ridge_fit(Z, s, alpha=1.0):
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Zs = (Z - mu) / sd
    coef = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ s)
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
        out.append(net(x).squeeze(0).squeeze(-1).cpu().numpy())
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
    feats, gt_frame = load_test(os.path.join(LIST, "ucf_CLIP_rgbtest.csv"),
                               os.path.join(LIST, "gt_ucf.npy"))
    print("device %s | train %d | test %d videos %d frames"
          % (device, len(Xtr), len(feats), len(gt_frame)), flush=True)
    net = train_raw(Xtr, ytr, epochs=args.epochs, seed=args.seed, device=device)
    S = score(net, feats, device)
    off, pairs, Gs = 0, [], []
    Zt, Zm, Zml, Zf = [], [], [], []
    for f, s in zip(feats, S):
        g = gt_frame[off:off + len(f)]
        off += len(f)
        pairs.append((np.asarray(s, float), g))
        Gs.append(g)
        t, m, ml, fn = zfeat(f)
        Zt.append(t)
        Zm.append(m)
        Zml.append(ml)
        Zf.append(fn)
    store = []
    base = decompose(pairs, "score", store)
    # z-only baselines
    for nm, Z in [("z-time", Zt), ("z-motion", Zm), ("z-mlocal", Zml),
                  ("z-fnorm", Zf)]:
        decompose([(Z[i], Gs[i]) for i in range(len(Z))], nm, store)
    # conditional B: stratify by motion decile
    for nm, Z in [("motion", Zm), ("mlocal", Zml)]:
        b_adj, npair, nv = pair_weighted_within(pairs, strat=Z, nbins=5)
        b_adj10 = pair_weighted_within(pairs, strat=Z, nbins=10)[0]
        print("  B|%-8s (5 deciles) %.4f (n_pairs %.3g, videos %d) | 10 dec %.4f"
              % (nm, b_adj, npair, nv, b_adj10), flush=True)
        store.append(dict(tag="B|" + nm, B=b_adj, B10=b_adj10, n_pairs=npair,
                          n_videos=nv))
        # the same stratification applied to the z statistic itself
        b_self = pair_weighted_within([(Z[i], Gs[i]) for i in range(len(Z))],
                                      strat=Z, nbins=5)[0]
        print("  B|%-8s for z itself %.4f (if ~0.50, z carries nothing left)"
              % (nm, b_self), flush=True)
        store.append(dict(tag="B|%s(z)" % nm, B=b_self))
    # within-video correlation between score and motion
    cs = []
    for i in range(len(S)):
        if Gs[i].min() == Gs[i].max():
            continue
        a = S[i] - S[i].mean()
        b = Zm[i] - Zm[i].mean()
        if a.std() > 1e-9 and b.std() > 1e-9:
            cs.append(float(np.corrcoef(a, b)[0, 1]))
    print("  within-video corr(score, motion): mean %.3f median %.3f"
          % (np.mean(cs), np.median(cs)), flush=True)
    store.append(dict(tag="corr_score_motion", corr_mean=float(np.mean(cs)),
                      corr_median=float(np.median(cs))))
    # cross-fitted residualisation, fitted on control frames only
    Zall = [np.stack([Zt[i], Zm[i], Zml[i], Zf[i]], 1)
            for i in range(len(Zt))]
    for tag, q in [("adj-all", None), ("adj-ctrl-med", 0.5),
                   ("adj-ctrl-q75", 0.75)]:
        S2 = [None] * len(S)
        for parity in (0, 1):
            fit = [i for i in range(len(S)) if i % 2 == parity]
            app = [i for i in range(len(S)) if i % 2 != parity]
            Zf_, sf_ = [], []
            for i in fit:
                m = (np.ones(len(S[i]), bool) if q is None
                     else S[i] <= np.quantile(S[i], q))
                Zf_.append(Zall[i][m])
                sf_.append(S[i][m])
            model = ridge_fit(np.concatenate(Zf_), np.concatenate(sf_))
            for i in app:
                S2[i] = S[i] - ridge_pred(model, Zall[i])
        d = decompose([(S2[i], Gs[i]) for i in range(len(S))], tag, store,
                      "residual score")
        d["B|motion"] = pair_weighted_within(
            [(S2[i], Gs[i]) for i in range(len(S))], strat=Zm, nbins=5)[0]
        print("  %s: B|motion %.4f" % (tag, d["B|motion"]), flush=True)
    print("elapsed %.1f s" % (time.time() - t0), flush=True)
    if args.out:
        pd.DataFrame(store).to_csv(args.out, index=False)
        json.dump(store, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
