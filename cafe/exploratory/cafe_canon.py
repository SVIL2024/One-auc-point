# -*- coding: utf-8 -*-
"""Canonical decomposition for every exported checkpoint + partial-identification
numbers for Proposition 5 (delta-contamination bound).

B is the pair-weighted within-video contrast; B|z stratifies within each video on
five equally populated quintiles of the 5-frame moving-average motion statistic
(the `mlocal` column of every npz). Both are recomputed here from one code path so
that every table in the paper agrees to four decimals.

usage:  python cafe_canon.py --glob "out/scores_ucf_*.npz"
        python cafe_canon.py --sim        # sharpness / coverage check
"""
import argparse
import glob
import os

import numpy as np


def auc_pairs(sp, sn):
    from scipy.stats import rankdata
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    r = rankdata(np.concatenate([sp, sn]))
    rp = r[:len(sp)].sum()
    return (rp - len(sp) * (len(sp) + 1) / 2.0) / (len(sp) * len(sn))


def split(a, lens):
    out, o = [], 0
    for L in lens:
        out.append(a[o:o + L])
        o += L
    return out


def canonical(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    S, G, Z = split(d["s2"], d["lens"]), split(d["gt"], d["lens"]), split(d["mlocal"], d["lens"])

    pos = np.concatenate([s[g == 1] for s, g in zip(S, G)])
    neg = np.concatenate([s[g == 0] for s, g in zip(S, G)])
    theta = auc_pairs(pos, neg)

    neg_norm = np.concatenate([s[g == 0] for s, g in zip(S, G) if (g == 1).sum() == 0])
    A = auc_pairs(pos, neg_norm)
    neg_anom = np.concatenate([s[g == 0] for s, g in zip(S, G) if (g == 1).sum() > 0])
    C = auc_pairs(pos, neg_anom)

    num = den = 0.0
    bnum = bden = 0.0
    for s, g, z in zip(S, G, Z):
        sp, sn = s[g == 1], s[g == 0]
        if len(sp) == 0 or len(sn) == 0:
            continue
        num += len(sp) * len(sn) * auc_pairs(sp, sn)
        den += len(sp) * len(sn)
        zz = z.astype(float)
        sd = zz.std()
        if sd > 1e-12:
            zz = (zz - zz.mean()) / sd
        qs = np.quantile(zz, [0.2, 0.4, 0.6, 0.8])
        b = np.searchsorted(qs, zz, side="right")
        for k in range(5):
            mp, mn = (g == 1) & (b == k), (g == 0) & (b == k)
            if mp.sum() and mn.sum():
                bnum += mp.sum() * mn.sum() * auc_pairs(s[mp], s[mn])
                bden += mp.sum() * mn.sum()
    B = num / den
    Bz = bnum / bden
    return dict(theta=theta, A=A, B=B, Bz=Bz, C=C, n=len(S),
                n_pos=int((np.concatenate(G) == 1).sum()))


def bound_table(rows):
    """delta-contamination: tau in [(B-d)/(1-d), min(1, B/(1-d))]."""
    print("\n%-22s %7s %7s %7s %7s %7s %8s %8s" % (
        "tag", "theta", "B", "B|z", "D", "d_mot", "delta*", "tau_LB(.2)"))
    for r in rows:
        D = r["B"] - r["Bz"]
        d_mot = D / (1.0 - r["Bz"])
        dstar = 2.0 * r["Bz"] - 1.0
        lb = (r["B"] - 0.2) / (1.0 - 0.2)
        print("%-22s %7.4f %7.4f %7.4f %7.4f %7.3f %8.3f %8.4f" % (
            r["tag"], r["theta"], r["B"], r["Bz"], D, d_mot, dstar, lb))
    if len(rows) > 2:
        from scipy.stats import spearmanr
        th = [r["theta"] for r in rows]
        bz = [r["Bz"] for r in rows]
        bb = [r["B"] for r in rows]
        print("\nSpearman(theta,B|z) = %.4f (p=%.4f) | Spearman(theta,B) = %.4f (p=%.4f) | "
              "Spearman(B,B|z) = %.4f (p=%.4f)" % (
                  spearmanr(th, bz)[0], spearmanr(th, bz)[1],
                  spearmanr(th, bb)[0], spearmanr(th, bb)[1],
                  spearmanr(bb, bz)[0], spearmanr(bb, bz)[1]))
        print("span theta %.4f  B %.4f  B|z %.4f  A %.4f  C %.4f" % (
            max(th) - min(th), max(bb) - min(bb), max(bz) - min(bz),
            max(r["A"] for r in rows) - min(r["A"] for r in rows),
            max(r["C"] for r in rows) - min(r["C"] for r in rows)))
        print("D range %.4f-%.4f median %.4f" % (
            min(r["B"] - r["Bz"] for r in rows), max(r["B"] - r["Bz"] for r in rows),
            float(np.median([r["B"] - r["Bz"] for r in rows]))))


def simulate():
    """Is the contamination bound right, and is it sharp?"""
    rng = np.random.default_rng(0)
    m, n, reps = 400, 4000, 200
    rows = []
    for delta in (0.0, 0.1, 0.2, 0.3, 0.5):
        emp = []
        for _ in range(reps):
            sp = rng.normal(1.0, 1.0, m)
            sn = rng.normal(0.0, 1.0, n)
            tau0 = ((sp[:, None] > sn[None, :]).mean())  # confounder-free contrast
            k = int(round(delta * m))
            if k:
                sp = sp.copy()
                sp[:k] = sn.max() + 1.0        # worst case: contaminated pairs always win
            emp.append((sp[:, None] > sn[None, :]).mean())
        emp = float(np.mean(emp))
        tau_hat = ((rng.normal(1.0, 1.0, 100000)[:, None] >
                    rng.normal(0.0, 1.0, 100000)[None, :]).mean())
        lo = (emp - delta) / (1 - delta)
        hi = min(1.0, emp / (1 - delta))
        rows.append((delta, emp, lo, hi))
    print("\n%-6s %8s %8s %8s %8s" % ("delta", "E[B]obs", "tau_LB", "tau_UB", "tau_true~0.76"))
    for delta, emp, lo, hi in rows:
        print("%-6.2f %8.4f %8.4f %8.4f" % (delta, emp, lo, hi))
    # closed forms for the upper endpoint under the worst-case construction
    print("\nworst case predicts B = (1-d)*tau + d; check against data rows above.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/scores_ucf_*.npz")
    ap.add_argument("--sim", action="store_true")
    a = ap.parse_args()
    if a.sim:
        simulate()
        return
    rows = []
    for p in sorted(glob.glob(a.glob)):
        tag = os.path.basename(p).replace("scores_", "").replace(".npz", "")
        try:
            r = canonical(p)
            r["tag"] = tag
            rows.append(r)
        except Exception as e:  # noqa: BLE001
            print("FAIL %s %s" % (tag, e))
    bound_table(rows)


if __name__ == "__main__":
    main()
