# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, os
L = "E:/program/DSANet/list"
df = pd.read_csv(os.path.join(L, "xd_CLIP_rgbtest.csv"))
print("rows", len(df), "cols", list(df.columns))
print(df.head(3).to_string())
lens = []
for p in df["path"][:5]:
    a = np.load(p)
    print(os.path.basename(p), a.shape, a.dtype)
gt = np.load(os.path.join(L, "gt.npy"))
print("gt.npy", gt.shape, gt.dtype, gt[:20], "sum", int(gt.sum()))
n = 0
for p in df["path"]:
    n += len(np.load(p, mmap_mode="r"))
print("total frames in csv", n)
print("gt len", len(gt), "ratio", len(gt) / max(n, 1))
g2 = np.load(os.path.join(L, "gt_segment_ucf.npy")) if os.path.exists(
    os.path.join(L, "gt_segment_ucf.npy")) else None
for f in ["gt_segment.npy", "gt_label.npy", "gt_label_ucf.npy"]:
    p = os.path.join(L, f)
    if os.path.exists(p):
        a = np.load(p)
        print(f, a.shape, a.dtype, a[:10])
