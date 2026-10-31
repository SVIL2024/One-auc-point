"""Measure the released VadCLIP checkpoints with the CAFE estimator suite.

Runs on the experiment server from the EDSS-main root:

    set PYTHONPATH=E:\\program\\EDSS-main\\_win_shim
    cd /d E:\\program\\EDSS-main
    python src\\cafe_vadclip.py --ds ucf --model-path model\\vadclip_model_ucf.pth

Produces, exactly like cafe_map11.py does for the DSANet arms:
  E:\\program\\CAFE\\out\\scores_<ds>_vadclip.npz   s2 / s1 / lens / gt / mlocal
  E:\\program\\CAFE\\out\\vadclip.json              theta, A, B, B|z, C, roc2, mAP

Nothing here imports DSANet: the decomposition helpers are copied verbatim
from dsanet_dump.py so the numbers stay comparable with Table 3.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score)
from torch.utils.data import DataLoader

from model import CLIPVAD
from utils.dataset import UCFDataset, XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.runner import load_checkpoint
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option
import xd_option

LABEL_MAP = dict({
    'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest',
    'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary',
    'Explosion': 'Explosion', 'Fighting': 'Fighting',
    'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery',
    'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting',
    'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})
XD_LABEL_MAP = dict({'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
                     'F': 'F'})


# ---------------------------------------------------------------- estimator
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


# ---------------------------------------------------------------- inference
@torch.no_grad()
def infer(model, loader, maxlen, prompt_text, device):
    model.eval()
    probs1, probs2, lens, stack = [], [], [], []
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

        _, l1, l2 = model(visual, pm, prompt_text, lengths)
        l1 = l1.reshape(-1, l1.shape[-1])[:len_cur]
        l2 = l2.reshape(-1, l2.shape[-1])[:len_cur]

        p2 = 1 - l2.softmax(dim=-1)[:, 0]
        p1 = torch.sigmoid(l1.squeeze(-1))
        probs1.append(p1.float().cpu().numpy())
        probs2.append(p2.float().cpu().numpy())
        lens.append(len_cur)
        stack.append(np.repeat(l2.softmax(dim=-1).cpu().numpy(), 16, 0))
    return probs1, probs2, lens, stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="ucf", choices=["ucf", "xd"])
    ap.add_argument("--root", default=r"E:\program\EDSS-main")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--out", default=r"E:\program\CAFE\out")
    ap.add_argument("--temp", type=float, default=5.0,
                    help="kept for symmetry with the DSANet path; VadCLIP "
                         "scores its own head, no refine() is applied")
    ap.add_argument("--skip-map", action="store_true",
                    help="skip the repository's detection mAP (it hard-codes "
                         "fourteen classes and cannot run on XD-Violence)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = args.ds
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    if ds == "ucf":
        oargs = ucf_option.parser.parse_args([])
        test_list = os.path.join(args.root, "list", "ucf_CLIP_rgbtest.csv")
        gt = np.load(os.path.join(args.root, "list", "gt_ucf.npy"))
        gt_frame = gt.reshape(-1, 16)[:, 0].astype(np.int64)
        seg_p = os.path.join(args.root, "list", "gt_segment_ucf.npy")
        lab_p = os.path.join(args.root, "list", "gt_label_ucf.npy")
        if not os.path.exists(seg_p):
            seg_p = os.path.join(args.root, "list", "gt_segment.npy")
            lab_p = os.path.join(args.root, "list", "gt_label.npy")
        label_map = LABEL_MAP
        dset_cls = UCFDataset
    else:
        oargs = xd_option.parser.parse_args([])
        test_list = os.path.join(args.root, "list", "xd_CLIP_rgbtest.csv")
        gt = np.load(os.path.join(args.root, "list", "gt.npy"))
        # gt.npy is frame-level (2,331,296 = 16 x 145,706 snippets); the scores
        # are one value per snippet, so take the first frame of each block
        gt_frame = np.asarray(gt).reshape(-1, 16)[:, 0].astype(np.int64)
        seg_p = os.path.join(args.root, "list", "gt_segment.npy")
        lab_p = os.path.join(args.root, "list", "gt_label.npy")
        label_map = XD_LABEL_MAP
        dset_cls = XDDataset

    gtsegments = np.load(seg_p, allow_pickle=True)
    gtlabels = np.load(lab_p, allow_pickle=True)
    if not args.model_path:
        args.model_path = os.path.join(
            args.root, "model", "vadclip_model_%s.pth" % ds)

    prompt_text = get_prompt_text(label_map)
    dset = dset_cls(oargs.visual_length, test_list, True, label_map)
    loader = DataLoader(dset, batch_size=1, shuffle=False)

    t0 = time.time()
    model = CLIPVAD(oargs.classes_num, oargs.embed_dim, oargs.visual_length,
                    oargs.visual_width, oargs.visual_head, oargs.visual_layers,
                    oargs.attn_window, oargs.prompt_prefix, oargs.prompt_postfix,
                    device)
    sd = torch.load(args.model_path, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print("ckpt %s  missing=%d unexpected=%d"
          % (os.path.basename(args.model_path), len(miss), len(unexp)),
          flush=True)
    if len(miss):
        print("  missing e.g. %s" % (miss[:5],), flush=True)
    if len(unexp):
        print("  unexpected e.g. %s" % (unexp[:5],), flush=True)
    model.to(device)

    probs1, probs2, lens, stack = infer(model, loader, oargs.visual_length,
                                        prompt_text, device)
    tag = "%s_vadclip" % ds

    if args.skip_map:
        # the repository's getDetectionMAP hard-codes fourteen classes, so it
        # cannot be run on XD-Violence (six classes); record the rest anyway
        dmap, ious = np.array([], float), np.array([], float)
        m50 = favg = float("nan")
    else:
        dmap, ious = dmAP(stack, gtsegments, gtlabels, excludeNormal=False)
        dmap = np.asarray(dmap, float)
        ious = np.asarray(ious, float)
        m50 = float(dmap[int(np.argmin(np.abs(ious - 0.5)))])
        favg = float(dmap.mean())

    off, pairs = 0, []
    for i, n in enumerate(lens):
        pairs.append((probs2[i], gt_frame[off:off + n]))
        off += n
    assert off == len(gt_frame), "gt length %d != %d scored snippets" % (
        len(gt_frame), off)
    feats = [np.load(p).astype(np.float32)
             for p in pd.read_csv(test_list)["path"]]
    assert sum(len(f) for f in feats) == off, "feature/snippet count mismatch"
    zs = zblocks(feats)
    mlocal = [z[:, 2] for z in zs]
    dec = decompose(pairs, mlocal)

    flat_gt = np.concatenate([gt_frame[sum(lens[:i]):sum(lens[:i]) + n]
                              for i, n in enumerate(lens)])
    roc2 = float(roc_auc_score(flat_gt, np.concatenate(probs2)))
    ap2 = float(average_precision_score(flat_gt, np.concatenate(probs2)))
    roc1 = float(roc_auc_score(flat_gt, np.concatenate(probs1)))
    ap1 = float(average_precision_score(flat_gt, np.concatenate(probs1)))

    np.savez_compressed(
        os.path.join(args.out, "scores_%s.npz" % tag),
        s2=np.concatenate(probs2), s1=np.concatenate(probs1),
        lens=np.asarray(lens, np.int64), gt=gt_frame,
        mlocal=np.concatenate([z[:, 2] for z in zs]).astype(np.float32))

    rec = dict(tag=tag, ds=ds, model=os.path.abspath(args.model_path),
               mAP=dict(("iou%.1f" % i, float(v)) for i, v in zip(ious, dmap)),
               mAP50=m50, mAPavg=favg,
               theta=dec["theta"], A=dec["A"], B=dec["B"],
               Bz=dec["B_cond"], C=dec["C"], w=dec["w"], lam=dec["lam"],
               roc2=roc2, ap2=ap2, roc1=roc1, ap1=ap1,
               secs=round(time.time() - t0, 1))
    out_json = os.path.join(args.out, "vadclip.json")
    res = {}
    if os.path.exists(out_json):
        try:
            res = json.load(open(out_json))
        except Exception:
            res = {}
    res[tag] = rec
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1)

    print("AUC1 %.4f  AP1 %.4f | AUC2 %.4f  AP2 %.4f"
          % (roc1, ap1, roc2, ap2), flush=True)
    for i in range(len(ious)):
        print("mAP@{0:.1f} = {1:.2f}%%".format(ious[i], dmap[i]), flush=True)
    print("average MAP: %.2f" % favg, flush=True)
    print("theta %.4f  A %.4f  B %.4f  B|z %.4f  C %.4f  w %.6f"
          % (dec["theta"], dec["A"], dec["B"], dec["B_cond"], dec["C"],
             dec["w"]), flush=True)
    print("secs %.1f" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
