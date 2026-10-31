import numpy as np, os, sys
root = r"E:\program\DSANet"
sys.path.insert(0, os.path.join(root, "src"))
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
z = np.load(r"E:\program\CAFE\out\scores_ucf_base_s789.npz")
lens = z["lens"]; s = z["s2"]; gt = z["gt"]
seg = np.load(os.path.join(root, "list", "gt_segment_ucf.npy"), allow_pickle=True)
lab = np.load(os.path.join(root, "list", "gt_label_ucf.npy"), allow_pickle=True)
print("n videos", len(lens), "total snippets", len(s), "segments", len(seg))
for i in (0, 1, 2, 5):
    print("vid", i, "len_snip", lens[i], "rep16", lens[i]*16, "segs", seg[i], "labels", lab[i])
print("gt sum", gt.sum(), "gt len", len(gt))
