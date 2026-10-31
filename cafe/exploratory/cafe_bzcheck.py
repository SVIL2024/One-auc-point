# -*- coding: utf-8 -*-
"""Which B|z definition produced which number?

Computes B and B|z under several variants (bin count, z column) so we can pin
the canonical definition and make every table in the paper agree.

usage: python cafe_bzcheck.py --npz out/scores_ucf_base_s789.npz [--tag base_s789]
"""
import argparse
import numpy as np


def auc_pairs(sp, sn):
    """Mann-Whitney with average ranks (ties handled)."""
    from scipy.stats import rankdata
    allv = np.concatenate([sp, sn])
    r = rankdata(allv)
    rp = r[:len(sp)].sum()
    return (rp - len(sp) * (len(sp) + 1) / 2.0) / (len(sp) * len(sn))


def split(s, lens):
    out, o = [], 0
    for L in lens:
        out.append(s[o:o + L])
        o += L
    return out


def compute(S, G, Z, nbins=None):
    """theta (pooled AUC), A (vs normal-video negatives), B (within-video),
    B|z (within-video, stratified on within-video deciles of Z), C."""
    n_pos = sum(int((g == 1).sum()) for g in G)
    n_neg = sum(int((g == 0).sum()) for g in G)
    n_norm_v = sum(1 for g in G if (g == 1).sum() == 0)
    negA = sum(int((g == 0).sum()) for g in G if (g == 1).sum() == 0)
    negB = n_neg - negA
    # pooled
    allS = np.concatenate(S)
    allG = np.concatenate(G)
    theta = auc_pairs(allS[allG == 1], allS[allG == 0])
    A = auc_pairs(allS[allG == 1], np.concatenate([s[g == 0] for s, g in zip(S, G) if (g == 1).sum() == 0]))
    # within-video B, pair weighted
    num = den = 0.0
    nbz_num = nbz_den = 0.0
    pos_other = []
    for s, g, z in zip(S, G, Z):
        sp, sn = s[g == 1], s[g == 0]
        if len(sn) == 0:
            continue
        num += len(sp) * len(sn) * auc_pairs(sp, sn)
        den += len(sp) * len(sn)
        if nbins:
            zz = z.copy()
            sd = zz.std()
            if sd > 1e-12:
                zz = (zz - zz.mean()) / sd
            qs = np.quantile(zz, np.linspace(0, 1, nbins + 1)[1:-1])
            b = np.searchsorted(qs, zz, side="right")
            for k in range(nbins):
                mp = (g == 1) & (b == k)
                mn = (g == 0) & (b == k)
                if mp.sum() and mn.sum():
                    nbz_num += mp.sum() * mn.sum() * auc_pairs(s[mp], s[mn])
                    nbz_den += mp.sum() * mn.sum()
        pos_other.append(sp)
    B = num / den
    Bz = nbz_num / nbz_den if nbz_den else float("nan")
    # C: anomalous-video positives vs negatives of *other* anomalous videos
    pos_all = np.concatenate([p for p in pos_other if len(p)]) if pos_other else np.array([])
    negB_all = np.concatenate([s[g == 0] for s, g in zip(S, G) if (g == 1).sum() > 0])
    C = auc_pairs(pos_all, negB_all)
    w = np.array([negA, negB, 1.0 * 0], float)
    return dict(theta=theta, A=A, B=B, Bz=Bz, C=C, n_pos=n_pos, n_neg=n_neg,
                negA=negA, negB=negB, n_norm_v=n_norm_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    s2, lens, gt = d["s2"], d["lens"], d["gt"]
    mlocal = d["mlocal"] if "mlocal" in d.files else np.zeros_like(s2)
    S = split(s2, lens)
    G = split(gt, lens)
    Z = split(mlocal, lens)
    print("== %s ==" % (a.tag or a.npz))
    for nb in (None, 5, 10, 20):
        r = compute(S, G, Z, nb)
        print("nbins=%-4s theta %.4f A %.4f B %.4f B|z %s C %.4f" % (
            nb, r["theta"], r["A"], r["B"],
            ("%.4f" % r["Bz"]) if nb else "   -   ", r["C"]))


if __name__ == "__main__":
    main()
