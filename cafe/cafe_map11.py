# -*- coding: utf-8 -*-
"""Criterion validity: per-checkpoint TLoC mAP@0.5 alongside theta, B, B|z.

For each DSANet arm we reproduce the repo's own detection-mAP protocol
(``utils.ucf_detectionMAP.getDetectionMAP``, IoU 0.1-0.5, excludeNormal=False)
from the same per-frame probabilities the AUC is computed from, and we also
keep the raw scores.  This makes the B-vs-mAP rank correlation possible on all
eleven checkpoints of Table 3, not just on the four that had score files.

Run one arm per process (CUDA on this box throws the occasional illegal memory
access, and a single bad arm must not take the batch down):

    python cafe_map11.py --arm base --seed 234
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, "E:/program/CAFE")
import dsanet_dump as D                                        # noqa: E402

from model import DSANet                                   # noqa: E402
from utils.dataset import UCFDataset                            # noqa: E402
from utils.tools import get_batch_mask, get_prompt_text          # noqa: E402
from utils.ucf_detectionMAP import getDetectionMAP as dmAP       # noqa: E402
import ucf_option                                                # noqa: E402

ROOT = "E:/program/DSANet"
OUT = "E:/program/CAFE/out"
RES = os.path.join(OUT, "map11.json")


@torch.no_grad()
def infer(model, loader, maxlen, prompt_text, DNP_use, device):
    """Per-video class-probability matrices, exactly as ucf_test.test()."""
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
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = "ucf"

    test_list = os.path.join(args.root, "list", "ucf_CLIP_rgbtest.csv")
    gt = np.load(os.path.join(args.root, "list", "gt_ucf.npy"))
    gt_frame = gt.reshape(-1, 16)[:, 0].astype(np.int64)
    seg_p = os.path.join(args.root, "list", "gt_segment_ucf.npy")
    lab_p = os.path.join(args.root, "list", "gt_label_ucf.npy")
    if not os.path.exists(seg_p):
        seg_p = os.path.join(args.root, "list", "gt_segment.npy")
        lab_p = os.path.join(args.root, "list", "gt_label.npy")
    gtsegments = np.load(seg_p, allow_pickle=True)
    gtlabels = np.load(lab_p, allow_pickle=True)

    oargs = ucf_option.parser.parse_args(D.ARM_FLAGS.get(args.arm, "").split())
    oargs.visual_length = 256
    oargs.test_list = test_list
    oargs.model_path = ""
    prompt_text = get_prompt_text(D.LABEL_MAP)
    dset = UCFDataset(oargs.visual_length, test_list, True, D.LABEL_MAP)
    loader = DataLoader(dset, batch_size=1, shuffle=False)

    ck = os.path.join(args.root, "model", "%s_%s_s%d.pth"
                      % (ds, args.arm, args.seed))
    tag = "%s_%s_s%d" % (ds, args.arm, args.seed)
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
    m50 = float(dmap[2] if False else dmap[int(np.argmin(np.abs(ious - 0.5)))])
    favg = float(dmap.mean())

    # the same decomposition as Table 3, from the same scores
    off, pairs = 0, []
    for i, n in enumerate(lens):
        pairs.append((s2list[i], gt_frame[off:off + n]))
        off += n
    feats = [np.load(p).astype(np.float32)
             for p in D.pd.read_csv(test_list)["path"]]
    zs = D.zblocks(feats)
    mlocal = [z[:, 2] for z in zs]
    dec = D.decompose(pairs, mlocal)
    roc2 = float(roc_auc_score(np.repeat(gt_frame, 1),
                               np.concatenate(s2list)))

    np.savez_compressed(os.path.join(OUT, "scores_%s.npz" % tag),
                        s2=np.concatenate(s2list),
                        lens=np.asarray(lens, np.int64), gt=gt_frame,
                        mlocal=np.concatenate([z[:, 2] for z in zs]
                                              ).astype(np.float32))
    rec = dict(tag=tag, arm=args.arm, seed=args.seed,
               mAP=dict(("iou%.1f" % i, float(v)) for i, v in zip(ious, dmap)),
               mAP50=m50, mAPavg=favg,
               theta=dec["theta"], A=dec["A"], B=dec["B"],
               Bz=dec["B_cond"], C=dec["C"], roc2=roc2,
               secs=round(time.time() - t0, 1))
    res = {}
    if os.path.exists(RES):
        res = json.load(io.open(RES, encoding="utf-8"))
    res[tag] = rec
    json.dump(res, io.open(RES, "w", encoding="utf-8"), indent=1)
    print("[%s] theta %.4f B %.4f B|z %.4f | mAP@0.5 %.2f%% avg %.2f%% | %ds"
          % (tag, dec["theta"], dec["B"], dec["B_cond"], m50,
             favg, rec["secs"]), flush=True)


if __name__ == "__main__":
    main()
