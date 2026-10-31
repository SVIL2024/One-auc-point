# -*- coding: utf-8 -*-
"""Rank correlations among the eleven UCF-Crime checkpoints, with the
interval a reviewer will ask for.

Spearman's rho is reported with (a) the two-sided p-value, and (b) a 95%
interval obtained by Fisher's z transform, which is the only honest way to
say what eleven points can and cannot exclude.  The paper used to quote
rho=0.22, p=0.52 as "no association"; p>0.05 is not an interval.
"""
import io
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
T1 = os.path.join(RES, "table1.json")
MAP11 = os.path.join(RES, "map11.json")


def fisher_ci(rho, n, alpha=0.05):
    if not np.isfinite(rho) or abs(rho) >= 1.0 or n <= 3:
        return float("nan"), float("nan")
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3.0)
    half = stats.norm.ppf(1 - alpha / 2) * se
    return float(np.tanh(z - half)), float(np.tanh(z + half))


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y given z (rank residuals)."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)

    def resid(a, b):
        b = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(b, a, rcond=None)
        return a - b @ beta

    ex, ey = resid(rx, rz), resid(ry, rz)
    r = float(np.corrcoef(ex, ey)[0, 1])
    n = len(x)
    if abs(r) >= 1.0:
        return r, 0.0, float("nan"), float("nan")
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r))
    p = float(2 * (1 - stats.t.cdf(abs(t), n - 3)))
    lo, hi = fisher_ci(r, n)
    return r, p, lo, hi


def report(name, x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rho, p = stats.spearmanr(x, y)
    lo, hi = fisher_ci(rho, len(x))
    print("%-28s n=%2d  rho=%+.4f  p=%.4f  95%% CI [%+.3f, %+.3f]"
          % (name, len(x), rho, p, lo, hi))
    return rho, p, lo, hi


def main():
    rows = json.load(io.open(T1, encoding="utf-8"))
    if isinstance(rows, dict):
        rows = list(rows.values())
    rows = [r for r in rows if r.get("theta") is not None]
    # motion9.72v was excluded from Table 3: its structural flag reconstruction
    # does not reproduce the archived theta, so its scores are not trustworthy.
    rows = [r for r in rows if "9.72v" not in r.get("arm", "")]
    rows.sort(key=lambda r: -r["theta"])
    for r in rows:
        print("%-22s theta %.4f  B %.4f  B|z %.4f  A %.4f  C %.4f"
              % (r["arm"], r["theta"], r["B"], r["Bz"], r["A"], r["C"]))
    th = [r["theta"] for r in rows]
    bb = [r["B"] for r in rows]
    bz = [r["Bz"] for r in rows]
    aa = [r["A"] for r in rows]
    cc = [r["C"] for r in rows]
    print()
    report("Spearman(theta, B)", th, bb)
    report("Spearman(theta, B|z)", th, bz)
    report("Spearman(theta, A)", th, aa)
    report("Spearman(B, B|z)", bb, bz)

    # criterion validity, if the mAP table exists
    try:
        m = json.load(io.open(MAP11, encoding="utf-8"))
    except Exception:
        print("\nmap11.json not available yet")
        return
    by = {k: v for k, v in m.items() if v.get("mAP50") is not None}
    if len(by) < 4:
        print("\nmap11.json has only %d arms so far" % len(by))
        return
    keys = sorted(by)
    print("\n--- criterion validity (%d checkpoints) ---" % len(keys))
    for k in keys:
        v = by[k]
        print("%-26s theta %.4f  B %.4f  B|z %.4f  mAP@0.5 %6.2f%%  avg %6.2f%%"
              % (k, v["theta"], v["B"], v["Bz"], v["mAP50"], v["mAPavg"]))
    th = [by[k]["theta"] for k in keys]
    bb = [by[k]["B"] for k in keys]
    bz = [by[k]["Bz"] for k in keys]
    mp = [by[k]["mAP50"] for k in keys]
    ma = [by[k]["mAPavg"] for k in keys]
    print()
    report("Spearman(theta, mAP@0.5)", th, mp)
    report("Spearman(B, mAP@0.5)", bb, mp)
    report("Spearman(B|z, mAP@0.5)", bz, mp)
    report("Spearman(B, avg mAP)", bb, ma)
    report("Spearman(B|z, avg mAP)", bz, ma)
    print()
    for nm, xx in (("B", bb), ("B|z", bz), ("theta", th)):
        r, p, lo, hi = partial_spearman(xx, mp, th if nm != "theta" else bb)
        adj = "given theta" if nm != "theta" else "given B"
        print("partial rho(mAP, %-5s | %-5s) = %+.4f  p=%.4f  95%% CI [%+.3f, %+.3f]"
              % (nm, adj, r, p, lo, hi))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
