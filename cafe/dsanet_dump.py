# -*- coding: utf-8 -*-
"""Dump per-frame scores from *existing* DSANet checkpoints and decompose them.

No training.  For each checkpoint we reproduce the published number and then
report the exact three-way decomposition of it:

    theta = w*A + (1-w)*lam*B + (1-w)*(1-lam)*C

plus the within-video nuisance analysis (B|z, corr(score, motion)).

Usage (on the server, from E:\\program\\CAFE):
    python dsanet_dump.py --sets ucf --arms base s234 s555 s789
    python dsanet_dump.py --sets xd  --arms base s234
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

SRC = "E:/program/DSANet/src"
SHIM = "E:/program/CAFE/shim"      # ipdb stub: the released DSANet imports ipdb
sys.path.insert(0, SHIM)
sys.path.insert(0, SRC)
os.chdir(SRC)

from model import DSANet                                     # noqa: E402
from utils.dataset import UCFDataset                          # noqa: E402
from utils.tools import get_batch_mask, get_prompt_text       # noqa: E402
import ucf_option                                             # noqa: E402

LABEL_MAP = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest',
                  'Arson': 'Arson', 'Assault': 'Assault',
                  'Burglary': 'Burglary', 'Explosion': 'Explosion',
                  'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents',
                  'Robbery': 'Robbery', 'Shooting': 'Shooting',
                  'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing',
                  'Vandalism': 'Vandalism'})

# arm name -> the CLI fragment the arm was trained with (structure only: the
# scalar beta/gamma values live in the checkpoint, so they are irrelevant here)
ARM_FLAGS = {
    "base": "",
    "cons0": "--cons-weight 0",
    "attr": "--desc-variant attr",
    "res": "--residual-pool",
    "motion4.63k3": "--motion-feat --motion-stride 3",
    "motion6": "--motion-feat",
    "motion6shuf5678": "--motion-feat --motion-shuffle "
                       "--motion-shuffle-seed 5678",
    "motion6shuf9999": "--motion-feat --motion-shuffle "
                       "--motion-shuffle-seed 9999",
    "core8g0.1": "--core-proto --core-k 8 --core-gamma 0.1",
    "core8g1": "--core-proto --core-k 8 --core-gamma 1",
    "core14g1": "--core-proto --core-k 14 --core-gamma 1",
    "core14txt": "--core-proto --core-k 14 --core-text-init",
    "core14txtg1": "--core-proto --core-k 14 --core-gamma 1 "
                   "--core-text-init",
}


def refine(logits1, logits2, temp=5.0):
    eps = 1e-12
    p_ab = torch.sigmoid(logits1 / temp)
    p_nor = 1.0 - p_ab
    pa = F.softmax(logits2 / temp, dim=1)[:, 1:]
    pa = pa / (pa.sum(1, keepdim=True) + eps)
    return torch.cat([p_nor, p_ab * pa], 1)


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


def decompose(pairs, zs=None):
    pos, negA, same, wsum = [], [], [], []
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
    a = roc_auc_score(np.r_[np.ones(len(P)), np.zeros(len(A))], np.r_[P, A])
    hard = roc_auc_score(np.r_[np.ones(len(P)), np.zeros(len(Ball))],
                         np.r_[P, Ball])
    w = len(A) / (len(A) + len(Ball))
    wp = sum(len(p) * len(b) for p, b in same)
    tot = len(P) * len(Ball)
    B = pair_within(pairs)
    C = (hard * tot - B * wp) / (tot - wp)
    out = dict(w=float(w), lam=float(wp / tot), A=float(a), B=float(B),
               B_cond=None, C=float(C), hard=float(hard),
               theta=float(w * a + (1 - w) * hard))
    if zs is not None:
        out["B_cond"] = float(pair_within(pairs, strat=zs))
    return out


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


@torch.no_grad()
def run(model, loader, maxlen, prompt_text, DNP_use, device):
    model.eval()
    probs1, probs2, lens = [], [], []
    for item in loader:
        visual, length = item[0].squeeze(0), int(item[2])
        len_cur = length
        if len_cur < maxlen:
            visual = visual.unsqueeze(0)
        visual = visual.to(device)
        lengths = torch.zeros(int(length / maxlen) + 1)
        for j in range(len(lengths)):
            if j == 0 and length < maxlen:
                lengths[j] = length
            elif length > maxlen:
                lengths[j] = maxlen
                length -= maxlen
            else:
                lengths[j] = length
        lengths = lengths.to(torch.int)
        pm = get_batch_mask(lengths, maxlen).to(device)
        outs = model(visual, pm, prompt_text, lengths, DNP_use)
        l1, l2 = outs[1], outs[2]
        l1 = l1.reshape(-1, l1.shape[-1])[:len_cur]
        l2 = l2.reshape(-1, l2.shape[-1])[:len_cur]
        p2 = 1 - refine(l1, l2)[:, 0]
        probs1.append(torch.sigmoid(l1.squeeze(-1)).float().cpu().numpy())
        probs2.append(p2.float().cpu().numpy())
        lens.append(len_cur)
    return probs1, probs2, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="ucf")
    ap.add_argument("--arms", default="base 234,555,789")
    ap.add_argument("--root", default="E:/program/DSANet")
    ap.add_argument("--out", default="E:/program/CAFE/out/dsanet_dump.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = [a.strip() for a in args.arms.replace(",", " ").split()]
    # "--arms 'base 234 555 | motion6 789'" -> groups, one model each
    groups, cur = [], []
    for a in args.arms.replace(",", " ").replace("|", " | ").split():
        if a == "|":
            groups.append(cur)
            cur = []
        else:
            cur.append(a)
    if cur:
        groups.append(cur)
    groups = [(g[0], [int(x) for x in g[1:]]) for g in groups if g]
    results = {}
    for ds in args.sets.split(","):
        if ds == "ucf":
            test_list = os.path.join(args.root, "list", "ucf_CLIP_rgbtest.csv")
            gt_path = os.path.join(args.root, "list", "gt_ucf.npy")
            visual_length = 256
        else:
            test_list = os.path.join(args.root, "list", "xd_CLIP_rgbtest.csv")
            gt_path = os.path.join(args.root, "list", "gt.npy")
            visual_length = 256
        oargs = ucf_option.parser.parse_args([])
        prompt_text = get_prompt_text(LABEL_MAP)
        gt = np.load(gt_path)
        gt_frame = (gt.reshape(-1, 16)[:, 0].astype(np.int64) if ds == "ucf"
                    else np.asarray(gt).reshape(-1).astype(np.int64))
        dset = UCFDataset(visual_length, test_list, True, LABEL_MAP)
        loader = DataLoader(dset, batch_size=1, shuffle=False)
        feats = [np.load(p).astype(np.float32)
                 for p in pd.read_csv(test_list)["path"]]
        zs = zblocks(feats)
        mlocal = [z[:, 2] for z in zs]   # 5-frame |dx|: nuisance we stratify on
        for arm, seeds in groups:
            for seed in seeds:
                ck = os.path.join(args.root, "model", "%s_%s_s%d.pth"
                                  % (ds, arm, seed))
                if not os.path.exists(ck):
                    print("MISS", ck, flush=True)
                    continue
                t0 = time.time()
                try:
                    oargs = ucf_option.parser.parse_args(
                        ARM_FLAGS.get(arm, "").split())
                    oargs.visual_length = visual_length
                    oargs.test_list = test_list
                    oargs.model_path = ""
                    model = DSANet(oargs.classes_num, oargs.embed_dim,
                                   oargs.visual_length, oargs.visual_width,
                                   oargs.visual_head, oargs.visual_layers,
                                   oargs.attn_window, oargs.prompt_prefix,
                                   oargs.prompt_postfix, oargs, device)
                    sd = torch.load(ck, map_location="cpu",
                                    weights_only=False)
                    miss, unexp = model.load_state_dict(sd, strict=False)
                    if len(miss):
                        print("  WARN missing %d keys e.g. %s"
                              % (len(miss), miss[:3]), flush=True)
                    model.to(device)
                    p1, p2, lens = run(model, loader, oargs.visual_length,
                                       prompt_text, oargs.DNP_use, device)
                except Exception as e:  # noqa: BLE001 - one arm must not
                    print("FAIL %s s%d: %s: %s"
                          % (arm, seed, type(e).__name__, e), flush=True)
                    del model
                    torch.cuda.empty_cache()
                    continue
                off = 0
                pairs1, pairs2 = [], []
                for i in range(len(p1)):
                    g = gt_frame[off:off + lens[i]]
                    off += lens[i]
                    pairs1.append((p1[i], g))
                    pairs2.append((p2[i], g))
                tag = "%s_%s_s%d" % (ds, arm, seed)
                r = dict(tag=tag, ckpt=ck, n_frames=int(off),
                         d1=decompose(pairs1, mlocal),
                         d2=decompose(pairs2, mlocal))
                r["within_mean1"] = float(np.nanmean(
                    [roc_auc_score(g, s) for s, g in pairs1
                     if g.min() != g.max()]))
                r["within_mean2"] = float(np.nanmean(
                    [roc_auc_score(g, s) for s, g in pairs2
                     if g.min() != g.max()]))
                cs = []
                for i in range(len(p2)):
                    a = p2[i] - p2[i].mean()
                    b = zs[i][:, 2] - zs[i][:, 2].mean()
                    if a.std() > 1e-9 and b.std() > 1e-9:
                        cs.append(float(np.corrcoef(a, b)[0, 1]))
                r["corr_motion2"] = float(np.mean(cs))
                r["secs"] = round(time.time() - t0, 1)
                results[tag] = r
                # keep the raw scores: post-hoc adjustment, bootstrap and
                # E-values are computed from them, not from the aggregates
                np.savez_compressed(
                    "E:/program/CAFE/out/scores/scores_%s.npz" % tag,
                    s2=np.concatenate([np.asarray(s, np.float32)
                                       for s in p2]),
                    s1=np.concatenate([np.asarray(s, np.float32)
                                       for s in p1]),
                    lens=np.asarray(lens, np.int64), gt=gt_frame,
                    mlocal=np.concatenate([z[:, 2] for z in zs]
                                          ).astype(np.float32))
                print("[%s] theta1 %.4f (A %.4f B %.4f B|z %.4f C %.4f)"
                      % (tag, r["d1"]["theta"], r["d1"]["A"], r["d1"]["B"],
                         r["d1"]["B_cond"] or float("nan"), r["d1"]["C"]),
                      flush=True)
                print("[%s] theta2 %.4f (A %.4f B %.4f B|z %.4f C %.4f)"
                      " | within_mean2 %.4f corr %.3f | %.0fs"
                      % (tag, r["d2"]["theta"], r["d2"]["A"], r["d2"]["B"],
                         r["d2"]["B_cond"] or float("nan"), r["d2"]["C"],
                         r["within_mean2"], r["corr_motion2"], r["secs"]),
                      flush=True)
                del model
                torch.cuda.empty_cache()
    json.dump(results, open(args.out, "w"), indent=1)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
