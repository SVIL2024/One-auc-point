# -*- coding: utf-8 -*-
"""Collect every arm's metrics.json into one CSV (UCF + XD)."""
import glob
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root = "E:/program/DSANet/model"
rows = []
for p in sorted(glob.glob(os.path.join(root, "*_metrics.json"))):
    name = os.path.basename(p)[:-len("_metrics.json")]
    try:
        d = json.load(open(p))
    except Exception as e:
        print("skip", p, e)
        continue
    best = d.get("best") or {}
    fs = best.get("full_set") or {}
    comp = best.get("components") or {}
    xd = best.get("xd_map") or {}
    rows.append(dict(
        arm=name,
        auc=fs.get("pooled_auc"),
        ap=fs.get("pooled_ap"),
        r1=fs.get("recall_at_fpr_0.01"),
        easy=comp.get("auc_pos_vs_normal_video"),
        hard=comp.get("auc_pos_vs_anom_bg"),
        w=comp.get("weight_normal_video_negs"),
        dpred=comp.get("design_prediction"),
        dres=comp.get("design_residual"),
        tau=best.get("within_video_auc_mean"),
        tau_med=best.get("within_video_auc_median"),
        tau_lt=best.get("within_video_auc_frac_below_0p5"),
        xd_map=xd.get("average_map") or xd.get("avg_map") or xd.get("map"),
        epoch=best.get("epoch"),
        n_hist=len(d.get("history") or []),
    ))
hdr = ["arm", "auc", "ap", "r1", "easy", "hard", "w", "dpred", "dres", "tau",
       "tau_med", "tau_lt", "xd_map", "epoch", "n_hist"]
out = ["\t".join(hdr)]
for r in rows:
    out.append("\t".join("" if r[h] is None else
                         ("%.6f" % r[h] if isinstance(r[h], float) else str(r[h]))
                         for h in hdr))
txt = "\n".join(out)
open("E:/program/CAFE/out/arms_tsv.txt", "w", encoding="utf-8").write(txt)
print(txt)
print("n=", len(rows))
