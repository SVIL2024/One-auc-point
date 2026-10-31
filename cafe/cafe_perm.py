# -*- coding: utf-8 -*-
"""Checkpoint-level permutation test for the rank correlations of Section 6.7.

The question a reviewer asks about ``rho_s(B, mAP) = 0.15`` is not "is it
significant under a t-approximation" but "what does the exact permutation
distribution say, given twelve checkpoints".  This script computes it:

  * two-sided permutation p for every reported Spearman correlation, by
    shuffling the criterion (mAP / average mAP / frame-level AP) across
    checkpoints -- 100,000 draws, fixed seed;
  * the same for the two partial correlations, by shuffling the rank-residual
    of the criterion after regressing it on theta (and on B);
  * the whole thing twice: twelve rows (eleven self-trained arms plus the
    released anchor) and eleven rows (anchor dropped), so the reader can see
    that the anchor is not carrying the result.

    python cafe_perm.py
"""
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
NPERM = 100000
SEED = 20261031


def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x)
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def perm_p(x, y, nperm=NPERM, seed=SEED):
    rng = np.random.default_rng(seed)
    rx = ranks(x)
    ry = ranks(y)
    obs = float(np.corrcoef(rx, ry)[0, 1])
    yv = ry.copy()
    hits = 0
    for _ in range(nperm):
        rng.shuffle(yv)
        r = float(np.corrcoef(rx, yv)[0, 1])
        if abs(r) >= abs(obs) - 1e-15:
            hits += 1
    return obs, (hits + 1.0) / (nperm + 1.0)


def resid(v, z):
    rz = ranks(z)
    rv = ranks(v)
    A = np.vstack([np.ones_like(rz), rz]).T
    coef, *_ = np.linalg.lstsq(A, rv, rcond=None)
    return rv - A @ coef


def partial_perm(x, y, z, nperm=NPERM, seed=SEED):
    rx = resid(x, z)
    ry = resid(y, z)
    obs = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    yv = ry.copy()
    hits = 0
    for _ in range(nperm):
        rng.shuffle(yv)
        if abs(float(np.corrcoef(rx, yv)[0, 1])) >= abs(obs) - 1e-15:
            hits += 1
    return obs, (hits + 1.0) / (nperm + 1.0)


def load():
    m = json.load(io.open(os.path.join(OUT, "map11.json"), encoding="utf-8"))
    v = json.load(io.open(os.path.join(OUT, "vadclip.json"), encoding="utf-8"))
    o = json.load(io.open(os.path.join(OUT, "official12.json"), encoding="utf-8"))
    rows = []
    for tag, d in m.items():
        rows.append(dict(tag=tag, theta=d["theta"], B=d["B"], Bz=d["Bz"],
                         mAP50=d["mAP50"], mAPavg=d["mAPavg"],
                         ap_frame=o[tag]["ap_frame"], external=False))
    for tag, d in v.items():
        if d.get("ds") != "ucf":
            continue
        rows.append(dict(tag=tag, theta=d["theta"], B=d["B"], Bz=d["Bz"],
                         mAP50=d["mAP50"], mAPavg=d["mAPavg"],
                         ap_frame=d["ap1"], external=True))
    return rows


def fisher_ci(rho, n, z=1.959963985):
    rho = max(min(rho, 0.999999), -0.999999)
    se = 1.0 / np.sqrt(n - 3.0)
    hi = np.tanh(np.arctanh(rho) + z * se)
    lo = np.tanh(np.arctanh(rho) - z * se)
    return lo, hi


def report(rows, name):
    crit = ["mAP50", "mAPavg", "ap_frame"]
    pred = ["theta", "B", "Bz"]
    out = {"n": len(rows), "tag": name, "simple": {}, "partial": {}}
    print("\n== %s (n=%d) ==" % (name, len(rows)))
    for c in crit:
        y = [r[c] for r in rows]
        for p in pred:
            x = [r[p] for r in rows]
            rho, pv = perm_p(x, y)
            lo, hi = fisher_ci(rho, len(rows))
            out["simple"]["%s~%s" % (c, p)] = dict(rho=rho, p_perm=pv,
                                                   ci=[lo, hi])
            print("  rho(%s, %s) = %+.4f [%+.2f,%+.2f]  p_perm = %.4f"
                  % (c, p, rho, lo, hi, pv))
    # partial: mAP vs B given theta ; mAP vs theta given B
    for c in crit:
        y = [r[c] for r in rows]
        for a, b in (("B", "theta"), ("theta", "B"), ("Bz", "theta")):
            x = [r[a] for r in rows]
            z = [r[b] for r in rows]
            rho, pv = partial_perm(x, y, z)
            out["partial"]["%s~%s|%s" % (c, a, b)] = dict(rho=rho, p_perm=pv)
            print("  partial rho(%s, %s | %s) = %+.4f   p_perm = %.4f"
                  % (c, a, b, rho, pv))
    # cross-check: the two criteria against each other
    rho, pv = perm_p([r["mAP50"] for r in rows], [r["ap_frame"] for r in rows])
    out["simple"]["ap_frame~mAP50"] = dict(rho=rho, p_perm=pv)
    print("  rho(ap_frame, mAP50) = %+.4f   p_perm = %.4f" % (rho, pv))
    return out


def main():
    rows = load()
    print("rows: %d" % len(rows))
    res = {"nperm": NPERM, "seed": SEED}
    res["twelve"] = report(rows, "twelve")
    res["eleven"] = report([r for r in rows if not r["external"]], "eleven")
    json.dump(res, io.open(os.path.join(OUT, "perm.json"), "w",
                           encoding="utf-8"), indent=1)
    print("\nwrote perm.json")


if __name__ == "__main__":
    main()
