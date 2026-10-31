# -*- coding: utf-8 -*-
"""Inspect the structure of a CAFE dump json / score npz on the server."""
import io, json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = r"E:\program\CAFE\out"


def describe(obj, name, depth=0, maxdepth=3):
    pad = "  " * depth
    if isinstance(obj, dict):
        print("%s%s: dict(%d keys) -> %s" % (pad, name, len(obj), list(obj.keys())[:12]))
        if depth < maxdepth:
            for k in list(obj.keys())[:4]:
                describe(obj[k], k, depth + 1, maxdepth)
    elif isinstance(obj, list):
        print("%s%s: list(%d)" % (pad, name, len(obj)))
        if obj and depth < maxdepth:
            describe(obj[0], name + "[0]", depth + 1, maxdepth)
    else:
        v = repr(obj)
        print("%s%s: %s %s" % (pad, name, type(obj).__name__, v[:80]))


files = sorted(os.listdir(OUT))
js = [f for f in files if f.startswith("dump_") and f.endswith(".json")]
npzs = [f for f in files if f.endswith(".npz")]
print("== json dumps ==")
print(js)
print("== npz ==")
print(npzs)

for fn in ("dump_base.json", "dump_789.json"):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        continue
    print("\n#### %s (%.1f MB)" % (fn, os.path.getsize(p) / 1e6))
    d = json.load(io.open(p, encoding="utf-8"))
    describe(d, "root", 0, 3)

print("\n#### npz keys")
import numpy as np
for fn in npzs[:6]:
    p = os.path.join(OUT, fn)
    z = np.load(p, allow_pickle=True)
    print(fn, "%.1f MB" % (os.path.getsize(p) / 1e6))
    for k in z.files:
        a = z[k]
        print("   ", k, a.dtype, a.shape)
