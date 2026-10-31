import numpy as np, os, sys, json
root = r"E:\program\DSANet"
sys.path.insert(0, os.path.join(root, "src"))
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
seg0 = np.load(os.path.join(root, "list", "gt_segment_ucf.npy"), allow_pickle=True)
lab = np.load(os.path.join(root, "list", "gt_label_ucf.npy"), allow_pickle=True)
outs = {}
for tag, path in [("base_s789", r"E:\program\CAFE\out\scores_ucf_base_s789.npz"),
                  ("vadclip", r"E:\program\CAFE\out\scores_ucf_vadclip.npz")]:
    if not os.path.exists(path):
        print("MISS", path); continue
    z = np.load(path); s, lens = z["s2"], z["lens"]
    stack, off = [], 0
    for n in lens:
        p = s[off:off+int(n)]; off += int(n)
        stack.append(np.repeat(p.reshape(-1,1) if p.ndim==1 else p, 16, 0))
    pad = 14 - stack[0].shape[1]
    if pad > 0:
        stack = [np.concatenate([x, np.zeros((len(x), pad), x.dtype)], 1) for x in stack]
    res = {}
    for sc in [1.0, 30.0/16.0, 2.0, 16.0/30.0, 1.875*2, 0.5]:
        segs = [np.array([[float(a)*sc, float(b)*sc] for a, b in S], dtype=float) if len(S)
                else np.zeros((0,2)) for S in seg0]
        try:
            dm, iou = dmAP(stack, segs, lab, excludeNormal=False)
            res["sc%.3f" % sc] = [round(float(v),2) for v in dm]
        except Exception as e:
            res["sc%.3f" % sc] = "ERR %s" % str(e)[:60]
    outs[tag] = res
    print(tag, json.dumps(res))
