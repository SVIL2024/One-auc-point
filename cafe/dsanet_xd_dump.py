# -*- coding: utf-8 -*-
"""Dump XD-Violence frame scores from DSANet checkpoints.

Reuses the repository's own XD test path (utils.dataset.XDDataset with the XD
prompt set, xd_test.test) so that the scores are the published ones; dmAP is
stubbed out to capture the per-video score stack instead of recomputing mAP.

usage: python dsanet_xd_dump.py --arms "base 789 | motion3 789"
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "E:/program/CAFE/shim")     # stub for `import ipdb`
os.chdir("E:/program/DSANet/src")
sys.path.insert(0, "E:/program/DSANet/src")

import xd_option                                              # noqa: E402
import xd_test                                                # noqa: E402
from model import DSANet                                      # noqa: E402
from utils.dataset import XDDataset                           # noqa: E402
from utils.tools import get_prompt_text                       # noqa: E402

LABEL_MAP = {'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot',
             'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'}
ARM_FLAGS = {
    "base": "",
    "motion3": "--motion-feat --motion-stride 3",
    "motion6": "--motion-feat",
    "res": "--residual-pool",
    "attr": "--desc-variant attr",
}
CAP = {}


def fake_dmap(stack, gtsegments, gtlabels, excludeNormal=False):
    CAP["stack"] = stack
    return [0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4, 0.5]


xd_test.dmAP = fake_dmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base 789")
    ap.add_argument("--root", default="E:/program/DSANet")
    ap.add_argument("--out", default="E:/program/CAFE/out/xd_scores.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    prompt_text = get_prompt_text(LABEL_MAP)
    res = {}
    for arm, seeds in groups:
        for seed in seeds:
            ck = os.path.join(args.root, "model", "xd_%s_s%d.pth" % (arm, seed))
            if not os.path.exists(ck):
                print("MISS", ck, flush=True)
                continue
            oargs = xd_option.parser.parse_args(ARM_FLAGS.get(arm, "").split())
            oargs.test_list = os.path.join(args.root, "list",
                                           "xd_CLIP_rgbtest.csv")
            for k in ("gt_path", "gt_segment_path", "gt_label_path"):
                if not os.path.isabs(getattr(oargs, k)):
                    setattr(oargs, k, os.path.join(args.root, getattr(oargs, k)))
            gtsegments = np.load(oargs.gt_segment_path, allow_pickle=True)
            gtlabels = np.load(oargs.gt_label_path, allow_pickle=True)
            gt = np.load(oargs.gt_path).reshape(-1).astype(int)
            dset = XDDataset(oargs.visual_length, oargs.test_list, True,
                             LABEL_MAP)
            loader = DataLoader(dset, batch_size=1, shuffle=False)
            print("args:", {k: v for k, v in vars(oargs).items()
                            if "gt" in k or "list" in k}, flush=True)
            model = DSANet(oargs.classes_num, oargs.embed_dim,
                           oargs.visual_length, oargs.visual_width,
                           oargs.visual_head, oargs.visual_layers,
                           oargs.attn_window, oargs.prompt_prefix,
                           oargs.prompt_postfix, oargs, device)
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            miss, _ = model.load_state_dict(sd, strict=False)
            if len(miss):
                print("  WARN missing %s" % (miss[:3],), flush=True)
            model.to(device)
            CAP.clear()
            roc1, ap1 = xd_test.test(model, loader, oargs.visual_length,
                                     prompt_text, gt, gtsegments, gtlabels,
                                     oargs.DNP_use, oargs, device)[:2]
            stack = CAP["stack"]
            tag = "xd_%s_s%d" % (arm, seed)
            arrs = []
            for e in stack:
                a = np.asarray(e, dtype=object)
                arrs.append(np.asarray(e))
            shapes = [a.shape for a in arrs[:3]]
            print("[%s] roc1 %.4f ap1 %.4f | stack %d shapes %s"
                  % (tag, roc1, ap1, len(arrs), shapes), flush=True)
            np.savez_compressed("E:/program/CAFE/out/xdstack_%s.npz" % tag,
                                stack=np.array(arrs, dtype=object),
                                lens=np.asarray([len(a) for a in arrs]),
                                gt=gt, roc1=roc1, ap1=ap1)
            res[tag] = dict(roc1=float(roc1), ap1=float(ap1),
                            shapes=[list(s) for s in shapes])
            del model
            torch.cuda.empty_cache()
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
