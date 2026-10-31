"""Criterion validity, with the released VadCLIP checkpoint added as a twelfth
point and with a second, independently implemented localisation metric.

Reads
  CAFE/out/table1.json   theta / B / B|z for the eleven self-trained UCF arms
  CAFE/out/map11.json    detection mAP from the repository's own protocol
  CAFE/out/vadclip.json  same quantities for model/vadclip_model_ucf.pth
  CAFE/out/tloc.json     TLoC-style AP@0.5 from cafe_tloc.py

Prints Spearman rho with a Fisher-z interval, plus partial correlations, for
both localisation criteria.  Plain numpy.
"""
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load():
    T = json.load(open(os.path.join(OUT, "table1.json")))
    M = json.load(open(os.path.join(OUT, "map11.json")))
    V = json.load(open(os.path.join(OUT, "vadclip.json")))["ucf_vadclip"]
    L = json.load(open(os.path.join(OUT, "tloc.json")))
    rows = []
    for d in T:
        key = d["arm"] if d["arm"].startswith("ucf_") else "ucf_" + d["arm"]
        if key not in M:
            continue
        m = M[key]
        rows.append(dict(
            tag=key[4:], theta=d["theta"], B=d["B"], Bz=d["Bz"],
            mAP50=float(m["mAP50"]), mAPavg=float(m["mAPavg"]),
            AP50=float(L.get(key, {}).get("AP50", np.nan))))
    rows.append(dict(
        tag="vadclip(official)", theta=V["theta"], B=V["B"], Bz=V["Bz"],
        mAP50=float(V["mAP50"]), mAPavg=float(V["mAPavg"]),
        AP50=float(L.get("ucf_vadclip", {}).get("AP50", np.nan))))
    return rows


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    # average ties
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def fisher(rho, n):
    if not np.isfinite(rho) or n < 4:
        return float("nan"), float("nan")
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def pval(rho, n):
    # two-sided t test with df = n-2, via a normal approximation on Fisher z
    if not np.isfinite(rho) or n < 4:
        return float("nan")
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999)) * np.sqrt(n - 3)
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2.0)))


def partial(x, y, z):
    """rank-residual partial correlation"""
    rx, ry, rz = rank(x), rank(y), rank(z)
    A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    rows = load()
    n = len(rows)
    print("%-20s %7s %7s %7s %7s %7s" % ("checkpoint", "theta", "B", "B|z",
                                         "mAP50", "AP50"))
    for r in rows:
        print("%-20s %7.4f %7.4f %7.4f %7.2f %7.2f"
              % (r["tag"], r["theta"], r["B"], r["Bz"], r["mAP50"],
                 100 * r["AP50"] if np.isfinite(r["AP50"]) else float("nan")))
    col = lambda k: np.array([r[k] for r in rows], float)
    print("\nn = %d points (eleven self-trained arms + released VadCLIP)" % n)
    for crit, label in ((col("mAP50"), "repo mAP@0.5"),
                        (col("mAPavg"), "repo avg mAP"),
                        (col("AP50"), "TLoC-style AP@0.5")):
        print("\n-- %s" % label)
        for name in ("theta", "B", "Bz"):
            x = col(name)
            ok = np.isfinite(crit)
            rho = spearman(x[ok], crit[ok])
            lo, hi = fisher(rho, int(ok.sum()))
            print("   rho(%s, .) = %+.4f  p=%.4f  CI[%+.3f, %+.3f]"
                  % (name, rho, pval(rho, int(ok.sum())), lo, hi))
        ok = np.isfinite(crit)
        print("   partial rho(crit, B | theta) = %+.4f"
              % partial(crit[ok], col("B")[ok], col("theta")[ok]))
        print("   partial rho(crit, theta | B) = %+.4f"
              % partial(crit[ok], col("theta")[ok], col("B")[ok]))
    ok = np.isfinite(col("AP50")) & np.isfinite(col("mAP50"))
    print("\nrho(repo mAP@0.5, TLoC AP@0.5) = %+.4f (n=%d)"
          % (spearman(col("mAP50")[ok], col("AP50")[ok]), int(ok.sum())))


if __name__ == "__main__":
    main()
