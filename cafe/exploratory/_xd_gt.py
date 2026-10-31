import numpy as np
for p in [r"E:\program\EDSS-main\list\gt.npy",
          r"E:\program\DSANet\list\gt.npy"]:
    try:
        g = np.load(p)
        print(p, "shape", g.shape, "dtype", g.dtype, "size", g.size,
              "sum", int(np.asarray(g).sum()), flush=True)
        a = np.asarray(g).reshape(-1)
        print("   first 40:", a[:40].tolist(), flush=True)
        print("   unique:", np.unique(a)[:10].tolist(), flush=True)
    except Exception as e:
        print(p, "ERR", e, flush=True)
for p in [r"E:\program\EDSS-main\list\xd_CLIP_rgbtest.csv",
          r"E:\program\DSANet\list\xd_CLIP_rgbtest.csv"]:
    try:
        import pandas as pd
        d = pd.read_csv(p)
        print(p, "rows", len(d), "cols", list(d.columns), flush=True)
    except Exception as e:
        print(p, "ERR", e, flush=True)
