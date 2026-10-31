# -*- coding: utf-8 -*-
"""Official TLoC recomputation + frame-level AUC/AP, one arm per process.

Two jobs:

1. Recompute the detection-mAP column of Table 7 with the *reference*
   implementation (``utils.ucf_detectionMAP.getDetectionMAP``, Zhong et al.'s
   ``getLocMAP``, IoU 0.1-0.5, ``excludeNormal=False``) and, in the same
   process, with an independent re-implementation written from the algorithm
   description (different code path: vectorised run extraction, own NMS, own
   greedy matcher).  The two must agree to <1e-9; that is what licenses the
   claim that the column is the published protocol rather than our variant.

2. Compute the frame-level numbers that the papers themselves report
   (``roc_auc_score`` and ``average_precision_score`` over all frames of the
   test split), so the released anchor can be checked against its own paper.

    python cafe_official12.py --arm base --seed 234
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, "E:/program/CAFE")
import dsanet_dump as D                                         # noqa: E402

from model import DSANet                                        # noqa: E402
from utils.dataset import UCFDataset                            # noqa: E402
from utils.tools import get_batch_mask, get_prompt_text          # noqa: E402
from utils.ucf_detectionMAP import getDetectionMAP as dmAP       # noqa: E402
import ucf_option                                                # noqa: E402

ROOT = "E:/program/DSANet"
OUT = "E:/program/CAFE/out"
RES = os.path.join(OUT, "official12.json")

CLASSES = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
           'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
           'Shoplifting', 'Stealing', 'Vandalism']


# ---------------------------------------------------------------- independent
def _cls_idx(name):
    for i, c in enumerate(CLASSES):
        if name == c:
            return i
    return -1


def _nms_keep(boxes, thr=0.6):
    """Greedy NMS over [start, end, score]; independent of the repo version."""
    if len(boxes) == 0:
        return []
    b = np.asarray(boxes, float)
    order = list(np.argsort(-b[:, 2]))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            inter = max(0.0, min(b[i, 1], b[j, 1]) - max(b[i, 0], b[j, 0]))
            union = (b[i, 1] - b[i, 0]) + (b[j, 1] - b[j, 0]) - inter
            if inter / max(union, 1e-12) <= thr:
                rest.append(j)
        order = rest
    return keep


def _runs(mask):
    """(start, end) of maximal True runs."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(idx[0::2], idx[1::2]))


def loc_ap_reimpl(preds, segs, labs, iou_th):
    """Independent per-class AP for one IoU threshold; returns mean over 14."""
    aps = []
    for c in range(14):
        # proposals, NMS applied inside each video (as in the reference code)
        props = []
        for i, p in enumerate(preds):
            tmp = np.asarray(p, float)[:, c]
            k = max(1, tmp.size // 16)
            c_s = float(np.mean(np.sort(tmp)[::-1][:k]))
            if c_s <= 0.0:
                tmp = np.zeros_like(tmp)
            lo, hi = float(tmp.min()), float(tmp.max())
            thr = hi - (hi - lo) * 0.6
            vid = [[s, e, float(tmp[s:e].max()) + 0.7 * c_s]
                   for s, e in _runs(tmp > thr) if e - s >= 2]
            for j in _nms_keep(vid, 0.6):
                props.append([i, vid[j][0], vid[j][1], vid[j][2]])
        if not props:
            aps.append(0.0)
            continue
        props.sort(key=lambda r: -r[3])

        gt = [[i, int(segs[i][j][0]), int(segs[i][j][1])]
              for i in range(len(segs)) for j in range(len(segs[i]))
              if _cls_idx(labs[i][j]) == c]
        gtpos = len(gt)
        tp = []
        for pr in props:
            flag, best_iou, best_j = 0.0, 0.0, -1
            a0, a1 = int(pr[1]), int(pr[2])
            for j, g in enumerate(gt):
                if g[0] != pr[0]:
                    continue
                b0, b1 = g[1], g[2]
                inter = max(0, min(a1, b1) - max(a0, b0))
                union = (a1 - a0) + (b1 - b0) - inter
                iou = inter / float(union) if union > 0 else 0.0
                if iou >= iou_th and iou > best_iou:
                    best_iou, best_j, flag = iou, j, 1.0
            if flag > 0:
                del gt[best_j]
            tp.append(flag)
        tp = np.asarray(tp, float)
        if tp.sum() == 0 or gtpos == 0:
            aps.append(0.0)
        else:
            prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
            aps.append(float(np.sum(prec * tp) / gtpos))
    return 100.0 * float(np.mean(aps))


# ---------------------------------------------------------------------- model
@torch.no_grad()
def infer(model, loader, maxlen, prompt_text, DNP_use, device):
    model.eval()
    stack, s2list, lens = [], [], []
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
        probs = D.refine(l1, l2, 5.0).detach().cpu().numpy()
        stack.append(np.repeat(probs, 16, 0))
        s2list.append((1.0 - probs[:, 0]).astype(np.float32))
        lens.append(len_cur)
    return stack, s2list, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=234)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_list = os.path.join(ROOT, "list", "ucf_CLIP_rgbtest.csv")
    gt = np.load(os.path.join(ROOT, "list", "gt_ucf.npy"))
    gt_frame = gt.reshape(-1, 16)[:, 0].astype(np.int64)
    seg_p = os.path.join(ROOT, "list", "gt_segment_ucf.npy")
    lab_p = os.path.join(ROOT, "list", "gt_label_ucf.npy")
    gtsegments = np.load(seg_p, allow_pickle=True)
    gtlabels = np.load(lab_p, allow_pickle=True)

    oargs = ucf_option.parser.parse_args(D.ARM_FLAGS.get(args.arm, "").split())
    oargs.visual_length = 256
    oargs.test_list = test_list
    oargs.model_path = ""
    prompt_text = get_prompt_text(D.LABEL_MAP)
    dset = UCFDataset(oargs.visual_length, test_list, True, D.LABEL_MAP)
    loader = DataLoader(dset, batch_size=1, shuffle=False)

    tag = "ucf_%s_s%d" % (args.arm, args.seed)
    ck = os.path.join(ROOT, "model", "%s.pth" % tag)
    t0 = time.time()
    model = DSANet(oargs.classes_num, oargs.embed_dim, oargs.visual_length,
                   oargs.visual_width, oargs.visual_head, oargs.visual_layers,
                   oargs.attn_window, oargs.prompt_prefix, oargs.prompt_postfix,
                   oargs, device)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    miss, _ = model.load_state_dict(sd, strict=False)
    if len(miss):
        print("  WARN missing %d keys e.g. %s" % (len(miss), miss[:3]))
    model.to(device)
    stack, s2list, lens = infer(model, loader, oargs.visual_length,
                                prompt_text, oargs.DNP_use, device)

    dmap, ious = dmAP(stack, gtsegments, gtlabels, excludeNormal=False)
    dmap = np.asarray(dmap, float)
    ious = np.asarray(ious, float)
    i50 = int(np.argmin(np.abs(ious - 0.5)))
    reimpl = [loc_ap_reimpl(stack, gtsegments, gtlabels, th) for th in ious]
    diff = float(np.max(np.abs(np.asarray(reimpl) - dmap)))

    y = gt_frame
    s = np.concatenate(s2list)
    auc_f = float(roc_auc_score(y, s))
    ap_f = float(average_precision_score(y, s))

    rec = dict(tag=tag, arm=args.arm, seed=args.seed,
               mAP=dict(("iou%.1f" % t, float(v)) for t, v in zip(ious, dmap)),
               mAP50=float(dmap[i50]), mAPavg=float(dmap.mean()),
               mAP50_reimpl=float(reimpl[i50]), reimpl_maxdiff=diff,
               auc_frame=auc_f, ap_frame=ap_f, secs=round(time.time() - t0, 1))
    res = {}
    if os.path.exists(RES):
        res = json.load(io.open(RES, encoding="utf-8"))
    res[tag] = rec
    json.dump(res, io.open(RES, "w", encoding="utf-8"), indent=1)
    print("[%s] official mAP@0.5 %.2f | reimpl %.2f | maxdiff %.1e | "
          "frame AUC %.4f AP %.4f | %ds"
          % (tag, dmap[i50], reimpl[i50], diff, auc_f, ap_f,
             rec["secs"]), flush=True)


if __name__ == "__main__":
    main()
