# -*- coding: utf-8 -*-
import torch, os, glob, json
p = "E:/program/DSANet/model/ucf_base_s789.pth"
if not os.path.exists(p):
    cands = sorted(glob.glob("E:/program/DSANet/model/*.pth"))
    print("no ucf_base_s789.pth; candidates:", [os.path.basename(c) for c in cands])
    p = cands[0] if cands else None
ck = torch.load(p, map_location="cpu", weights_only=False)
print("file", p, "type", type(ck))
if isinstance(ck, dict):
    ks = list(ck.keys())
    print("top keys", ks[:20], "n", len(ks))
    for k in ks:
        v = ck[k]
        if isinstance(v, dict):
            print(" dict", k, "n", len(v), list(v.keys())[:8])
        elif isinstance(v, (int, float, str, bool)):
            print(" scalar", k, v)
        else:
            print(" other", k, type(v), getattr(v, "shape", ""))
