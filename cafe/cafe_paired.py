"""cafe_paired.py -- paired cluster bootstrap for the eleven UCF-Crime checkpoints.

Reads the per-frame score dumps written by cafe_map11.py (scores_ucf_<tag>.npz,
keys s2/lens/gt/mlocal) and, locally, on CPU:

  * recomputes B and B|z exactly as dsanet_dump.pair_within does (weighted
    within-video Mann-Whitney, w_v = n+_v * n-_v; B|z stratified into five
    within-video motion quintiles),
  * bootstraps *videos* as clusters (same resample applied to every arm, so the
    differences are paired),
  * reports the minimum resolvable difference between two checkpoints, the
    interval of the largest between-arm gap, and the flip rate of the arm
    ordering (how often the top two swap in a resample).

Writes out/paired11.json.
"""

import json
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TAGS = ["base_s234", "base_s555", "base_s789", "motion6_s555", "motion6_s789",
        "motion6shuf5678_s234", "res_s234", "attr_s234", "cons0_s234",
        "core8g1_s234", "core14txt_s234"]

# the eleven self-trained arms are the family the paper ranks; the released
# VadCLIP weights can be added as a twelfth row with --tags ... ,vadclip
if "--tags" in sys.argv:
    TAGS = sys.argv[sys.argv.index("--tags") + 1].split(",")
OUTNAME = "paired%d.json" % len(TAGS)
if "--out" in sys.argv:
    OUTNAME = sys.argv[sys.argv.index("--out") + 1]
B = 2000
SEED = 20261030


def auc_pairs(p, b):
    """AUC with average ranks for ties, plus the pair count."""
    n1, n0 = len(p), len(b)
    if n1 == 0 or n0 == 0:
        return np.nan, 0
    x = np.concatenate([p, b])
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r1 = ranks[:n1].sum()
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0), n1 * n0


def per_video(s, g, z=None, nbins=5):
    """Return (num_v, den_v) contributions to B (z=None) or B|z."""
    num = den = 0.0
    if g.min() == g.max():
        return 0.0, 0
    m = g == 1
    p, b = s[m], s[~m]
    if z is None:
        a, w = auc_pairs(p, b)
        return (a * w if w else 0.0), w
    edges = np.quantile(z, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    for j in range(nbins):
        sel = (z >= edges[j]) & (z < edges[j + 1])
        a, w = auc_pairs(p[sel[m]], b[sel[~m]])
        if w:
            num += a * w
            den += w
    return num, den


def load(tag):
    z = np.load(os.path.join(OUT, "scores", "scores_ucf_%s.npz" % tag))
    s, g, ml, lens = z["s2"], z["gt"], z["mlocal"], z["lens"]
    rows = []
    off = 0
    for n in lens:
        sl = slice(off, off + int(n))
        num_b, den_b = per_video(s[sl], g[sl])
        num_z, den_z = per_video(s[sl], g[sl], z=ml[sl])
        rows.append((num_b, den_b, num_z, den_z))
        off += int(n)
    return np.asarray(rows, dtype=np.float64)


def main():
    rng = np.random.default_rng(SEED)
    data = {t: load(t) for t in TAGS}
    point = {}
    for t, d in data.items():
        point[t] = dict(B=float(d[:, 0].sum() / d[:, 1].sum()),
                        Bz=float(d[:, 2].sum() / d[:, 3].sum()),
                        n=int(len(d)))
    nvid = point[TAGS[0]]["n"]
    for t in TAGS:
        assert point[t]["n"] == nvid

    boot = {t: np.empty(B) for t in TAGS}
    bootz = {t: np.empty(B) for t in TAGS}
    for r in range(B):
        idx = rng.integers(0, nvid, nvid)
        for t, d in data.items():
            sub = d[idx]
            boot[t][r] = sub[:, 0].sum() / sub[:, 1].sum()
            bootz[t][r] = sub[:, 2].sum() / sub[:, 3].sum()

    order = sorted(TAGS, key=lambda t: -point[t]["B"])
    flo, fhi = order[0], order[-1]
    dgap = boot[flo] - boot[fhi]
    lo, hi = np.percentile(dgap, [2.5, 97.5])

    # smallest between-arm gap, i.e. the closest adjacent pair by B
    gaps = []
    for a, b_ in zip(order[:-1], order[1:]):
        d = boot[a] - boot[b_]
        gaps.append((b_ + " - " + a, point[a]["B"] - point[b_]["B"],
                     float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
                     float(d.std(ddof=1))))
    closest = min(gaps, key=lambda g: abs(g[1]))

    # paired sd of a single arm (cluster) and the resolvable difference
    sd_single = {t: float(boot[t].std(ddof=1)) for t in TAGS}
    mdd = 2.0 * np.sqrt(2.0) * np.median([sd_single[t] for t in TAGS]) * 1.96 / 2.0

    # ordering flip rates
    rank_by_boot = np.argsort(np.argsort(-np.stack([boot[t] for t in TAGS]), axis=0), axis=0)
    top2 = [TAGS.index(order[0]), TAGS.index(order[1])]
    flip_top2 = float((rank_by_boot[top2[0]] != 0).mean())
    order_motion = sorted(TAGS, key=lambda t: -point[t]["Bz"])
    rbp = np.argsort(np.argsort(-np.stack([bootz[t] for t in TAGS]), axis=0), axis=0)
    flip_top2_z = float((rbp[TAGS.index(order_motion[0])] != 0).mean())

    print("tag              B       B|z    sd(B)   sd(B|z)")
    for t in order:
        print("%-16s %.4f  %.4f  %.4f  %.4f" % (t, point[t]["B"], point[t]["Bz"],
                                                sd_single[t], float(bootz[t].std(ddof=1))))
    print("\nlargest gap  %s - %s : dB=%.4f  CI[%.4f, %.4f]  cross-zero=%s"
          % (flo, fhi, point[flo]["B"] - point[fhi]["B"], lo, hi, bool(lo < 0 < hi)))
    print("closest pair %s : dB=%.4f CI[%.4f, %.4f] sd=%.4f"
          % (closest[0], closest[1], closest[2], closest[3], closest[4]))
    print("median paired sd of one arm  %.4f" % np.median(list(sd_single.values())))
    print("resolvable difference (2*1.96*sd/sqrt2, paired)  %.4f" % mdd)
    print("flip rate, top-2 by B   %.3f | by B|z  %.3f" % (flip_top2, flip_top2_z))

    json.dump(dict(tags=TAGS, point=point, sd_B=sd_single,
                   sd_Bz={t: float(bootz[t].std(ddof=1)) for t in TAGS},
                   order=order, order_motion=order_motion,
                   largest_gap=dict(pair=[flo, fhi], dB=point[flo]["B"] - point[fhi]["B"],
                                    ci=[lo, hi]),
                   closest=dict(pair=closest[0], dB=closest[1], ci=[closest[2], closest[3]],
                                sd=closest[4]),
                   median_paired_sd=float(np.median(list(sd_single.values()))),
                   resolvable=mdd, flip_top2_B=flip_top2, flip_top2_Bz=flip_top2_z,
                   n_videos=nvid, n_boot=B),
              open(os.path.join(OUT, OUTNAME), "w"), indent=1)
    print("\nwrote", os.path.join(OUT, OUTNAME))


if __name__ == "__main__":
    main()
