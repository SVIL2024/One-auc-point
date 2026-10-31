#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cafe_report.py -- one-command second-column report for a VAD scorer.

Computes, from frame-level scores only:
  * the pair census (the three comparison blocks and their weights)
  * theta        : the reported frame-level micro-AUC
  * B            : the pair-weighted within-video contrast (localization)
  * B|z          : the same, matched on within-video motion quintiles
  * video-clustered bootstrap intervals for all three
  * the E-value of B and of B|z
  * the partial-identification numbers of the paper (delta*, delta_mot, tau_LB)
  * a verdict for an author who claims an AUC/AP improvement

Usage
-----
  python cafe_report.py --scores out/scores_ucf_base_s789.npz
  python cafe_report.py --scores out/xdstack_xd_base_s789.npz --tag XD-base-s789
  python cafe_report.py --scores s.npz --claim-delta-auc 0.8 --sigma-theta 0.0048
  python cafe_report.py --scores s.npz --json report.json --boot 2000

Accepted score files (auto-detected, .npz)
------------------------------------------
  schema "flat" : s2 | s  (concatenated frame scores), gt (0/1, same length),
                  lens (frames per video), optional mlocal (motion statistic).
                  If mlocal is absent but `feat_dir` is given, the 5-frame
                  moving average of ||delta x|| is computed from the features.
  schema "xd"   : stack (one (n_i, C) array per video, class 0 = normal),
                  lens, gt (frame-level ground truth).

Only numpy + scipy are required; no torch, no model, no training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


# --------------------------------------------------------------------------
# small statistics helpers
# --------------------------------------------------------------------------
def mann_whitney(sp, sn):
    """P(s_p > s_n) + 0.5 P(tie), average ranks, O((p+n) log(p+n))."""
    from scipy.stats import rankdata

    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    r = rankdata(np.concatenate([sp, sn]))
    return (r[: len(sp)].sum() - len(sp) * (len(sp) + 1) / 2.0) / (len(sp) * len(sn))


def auc_pairs(sp, sn):
    return mann_whitney(sp, sn)


def e_value(rate):
    """E-value for a rate above 1/2 (VanderWeele & Ding, 2017)."""
    if not np.isfinite(rate) or rate >= 1.0:
        return float("inf")
    if rate <= 0.5:
        return 1.0
    rr = (rate / (1.0 - rate)) / 1.0
    return rr + np.sqrt(rr * (rr - 1.0))


def quantile_bins(z, q):
    zz = np.asarray(z, dtype=np.float64)
    sd = zz.std()
    if sd > 1e-12:
        zz = (zz - zz.mean()) / sd
    edges = np.quantile(zz, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, zz, side="right")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def split_by_len(a, lens):
    out, o = [], 0
    for L in lens:
        out.append(a[o : o + L])
        o += L
    if o != len(a):
        raise SystemExit("length mismatch: %d frames vs %d from lens" % (len(a), o))
    return out


def load_scores(path, feat_dir=None):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    lens = np.asarray(d["lens"], dtype=int)

    if "s2" in keys or "s" in keys:
        s = np.asarray(d["s2"] if "s2" in keys else d["s"], dtype=np.float64)
        gt = np.asarray(d["gt"]).astype(int).ravel()
        schema = "flat"
        mlocal = np.asarray(d["mlocal"], dtype=np.float64) if "mlocal" in keys else None
        if mlocal is not None and len(mlocal) != len(s):
            mlocal = None
    elif "stack" in keys:
        stack = d["stack"]
        s_parts, gt_parts = [], []
        o = 0
        for v, arr in enumerate(stack):
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] > 1:
                # column 0 is the normal class: score = -logit(normal) is a monotone
                # transform of 1 - p(normal), so it ranks identically to the
                # softmax version used for the reported metric
                s_parts.append(-arr[:, 0])
            else:
                s_parts.append(arr.ravel())
            gt_parts.append(np.asarray(d["gt"])[o : o + len(s_parts[-1])])
            o += len(s_parts[-1])
        s = np.concatenate(s_parts)
        gt = np.concatenate(gt_parts).astype(int)
        schema = "xd"
        mlocal = None
        lens = np.array([len(x) for x in s_parts], dtype=int)
    else:
        raise SystemExit("unrecognised score file: keys = %s" % sorted(keys))

    if mlocal is None and feat_dir:
        mlocal = _motion_from_features(feat_dir, lens)

    return dict(s=s, gt=gt.astype(int), lens=lens, mlocal=mlocal, schema=schema,
                path=path, keys=sorted(keys))


def _motion_from_features(feat_dir, lens):
    """5-frame moving average of ||x_t - x_{t-1}|| on frozen features."""
    files = sorted(os.listdir(feat_dir))
    files = [f for f in files if f.endswith(".npy")]
    if len(files) != len(lens):
        files = files[: len(lens)]
    out, o = [], 0
    for f, L in zip(files, lens):
        x = np.load(os.path.join(feat_dir, f)).astype(np.float32)
        if len(x) != L:
            x = x[:L] if len(x) > L else np.pad(x, ((0, L - len(x)), (0, 0)), mode="edge")
        dx = np.linalg.norm(np.diff(x, axis=0), axis=1)
        dx = np.concatenate([[0.0], dx])
        k = 5
        ker = np.ones(k) / k
        out.append(np.convolve(dx, ker, mode="same"))
        o += L
    return np.concatenate(out)


# --------------------------------------------------------------------------
# per-video sufficient statistics (so bootstrap is cheap)
# --------------------------------------------------------------------------
def video_stats(S, G, Z=None, q=5):
    """Return per-video (n_pos, n_neg, U_sum) plus B|z bin sums and censused
    pair counts for the three blocks."""
    stats = []
    for v, (s, g) in enumerate(zip(S, G)):
        sp, sn = s[g == 1], s[g == 0]
        npv, nnv = len(sp), len(sn)
        u = mann_whitney(sp, sn) if (npv and nnv) else 0.0
        rec = dict(np=npv, nn=nnv, u=u, npairs=npv * nnv,
                   bins=None, npos_all=npv, nneg_all=nnv, is_normal_video=(npv == 0))
        if Z is not None and npv and nnv:
            b = quantile_bins(Z[v], q)
            rows = []
            for k in range(q):
                mp, mn = (g == 1) & (b == k), (g == 0) & (b == k)
                if mp.sum() and mn.sum():
                    rows.append((int(mp.sum()), int(mn.sum()),
                                 mann_whitney(s[mp], s[mn])))
            rec["bins"] = rows
        stats.append(rec)
    return stats


def aggregate(stats, weights=None):
    """Pooled B and B|z from sufficient statistics."""
    if weights is None:
        weights = np.ones(len(stats), dtype=float)
    w = np.asarray(weights, dtype=float)
    npairs = np.array([r["npairs"] for r in stats], dtype=float)
    u = np.array([r["u"] for r in stats], dtype=float)
    den = float(np.sum(w * npairs))
    B = float(np.sum(w * npairs * u)) / den if den else float("nan")

    num = bden = 0.0
    for r, wv in zip(stats, w):
        if not r["bins"]:
            continue
        for (a, b, uu) in r["bins"]:
            num += wv * a * b * uu
            bden += wv * a * b
    Bz = num / bden if bden else float("nan")
    return dict(B=B, Bz=Bz)


def pooled_theta(S, G, w=None):
    sp = np.concatenate([s[g == 1] for s, g in zip(S, G)])
    sn = np.concatenate([s[g == 0] for s, g in zip(S, G)])
    return mann_whitney(sp, sn), len(sp), len(sn)


def census(S, G):
    pos = np.concatenate([s[g == 1] for s, g in zip(S, G)])
    neg_norm = np.concatenate([s[g == 0] for s, g in zip(S, G) if (g == 1).sum() == 0])
    neg_anom = np.concatenate([s[g == 0] for s, g in zip(S, G) if (g == 1).sum() > 0])
    n_pos, negA, negB = len(pos), len(neg_norm), len(neg_anom)
    w = negA / (negA + negB)
    within = sum(int((g == 1).sum()) * int((g == 0).sum()) for g in G)
    total = n_pos * (negA + negB)
    # lambda is the *conditional* share of within-video pairs among the pairs that
    # keep the video in the anomalous set, so that the three weights sum to one:
    #   theta = w * A + (1-w) * lambda * B + (1-w) * (1-lambda) * C
    lam = within / max(n_pos * negB, 1)
    omega_B = (1.0 - w) * lam
    return dict(n_pos=n_pos, neg_normal_video=negA, neg_anom_bg=negB, w=w, lam=lam,
                omega_A=w, omega_B=omega_B, omega_C=(1.0 - w) * (1.0 - lam),
                within_pairs=within, total_pairs=total,
                within_share=within / max(total, 1),
                exchange=(1.0 / omega_B) if omega_B > 0 else float("inf"))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="second-column report for a VAD scorer")
    ap.add_argument("--scores", required=True, help=".npz with scores, gt, lens")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--feat-dir", default=None, help="folder of per-video features (for B|z)")
    ap.add_argument("--boot", type=int, default=2000, help="video-cluster bootstrap replicates")
    ap.add_argument("--q", type=int, default=5, help="within-video motion bins")
    ap.add_argument("--claim-points", "--claim-delta-auc", dest="claim_delta_auc",
                    type=float, default=None,
                    help="improvement of the headline metric the author wants to claim, "
                         "in points (0.8 = +0.008 AUC)")
    ap.add_argument("--sigma-theta", type=float, default=None,
                    help="seed sd of the reported metric (e.g. 0.0048 on UCF-Crime)")
    ap.add_argument("--json", default=None, help="write the numbers to this json file")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    tag = a.tag or os.path.basename(a.scores).replace(".npz", "")
    d = load_scores(a.scores, a.feat_dir)
    S = split_by_len(d["s"], d["lens"])
    G = split_by_len(d["gt"], d["lens"])
    Z = split_by_len(d["mlocal"], d["lens"]) if d["mlocal"] is not None else None

    theta, n_pos, n_neg = pooled_theta(S, G)
    cen = census(S, G)
    stats = video_stats(S, G, Z, q=a.q)
    agg = aggregate(stats)

    # ---- clustered bootstrap over videos --------------------------------
    rng = np.random.default_rng(a.seed)
    V = len(S)
    npairs_v = np.array([r["npairs"] for r in stats], dtype=float)
    u_v = np.array([r["u"] for r in stats], dtype=float)
    bin_v, bin_a, bin_b, bin_u = [], [], [], []
    for v, r in enumerate(stats):
        for (ba, bb, bu) in (r["bins"] or []):
            bin_v.append(v)
            bin_a.append(ba)
            bin_b.append(bb)
            bin_u.append(bu)
    bin_v = np.array(bin_v, dtype=int)
    bin_ab = np.array(bin_a, dtype=float) * np.array(bin_b, dtype=float)
    bin_abu = bin_ab * np.array(bin_u, dtype=float)
    has_bins = len(bin_v) > 0

    b_r = np.empty(a.boot)
    bz_r = np.full(a.boot, np.nan)
    th_r = np.full(a.boot, np.nan)
    do_theta = len(d["s"]) <= 400000
    pos_by_v = [S[v][G[v] == 1] for v in range(V)]
    neg_by_v = [S[v][G[v] == 0] for v in range(V)]
    for r in range(a.boot):
        idx = rng.integers(0, V, V)
        cnt = np.bincount(idx, minlength=V).astype(float)
        num = float(np.sum(cnt * npairs_v * u_v))
        den = float(np.sum(cnt * npairs_v))
        b_r[r] = num / den if den else np.nan
        if has_bins:
            c = cnt[bin_v]
            bz_r[r] = float(np.sum(c * bin_abu)) / float(np.sum(c * bin_ab))
        if do_theta:
            sp = np.concatenate([pos_by_v[i] for i in idx if len(pos_by_v[i])])
            sn = np.concatenate([neg_by_v[i] for i in idx if len(neg_by_v[i])])
            th_r[r] = mann_whitney(sp, sn)

    def ci(x):
        x = x[np.isfinite(x)]
        if len(x) < 10:
            return (float("nan"), float("nan"))
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    th_ci, b_ci, bz_ci = ci(th_r), ci(b_r), ci(bz_r)

    # ---- derived quantities ---------------------------------------------
    B, Bz = agg["B"], agg["Bz"]
    D = B - Bz
    delta_mot = D / (1 - Bz) if np.isfinite(Bz) and Bz < 1 else float("nan")
    delta_star = 2 * B - 1
    tau_lb = max(0.0, (B - 0.2) / 0.8)
    tau_ub = min(1.0, B / 0.8)
    EB, EBz = e_value(B), e_value(Bz)
    within_contrib = cen["omega_B"] * B

    out = dict(tag=tag, schema=d["schema"], n_videos=V, n_frames=int(len(d["s"])),
               n_pos=cen["n_pos"], n_neg=cen["neg_normal_video"] + cen["neg_anom_bg"],
               theta=theta, theta_ci=th_ci,
               A=auc_pairs(np.concatenate([s[g == 1] for s, g in zip(S, G)]),
                           np.concatenate([s[g == 0] for s, g in zip(S, G)
                                           if (g == 1).sum() == 0]))
               if cen["neg_normal_video"] else float("nan"),
               C=auc_pairs(np.concatenate([s[g == 1] for s, g in zip(S, G)]),
                           np.concatenate([s[g == 0] for s, g in zip(S, G)
                                           if (g == 1).sum() > 0])),
               B=B, B_ci=b_ci, Bz=Bz, Bz_ci=bz_ci, D=D, delta_mot=delta_mot,
               delta_star=delta_star, tau_lb_02=tau_lb, tau_ub_02=tau_ub,
               E_B=EB, E_Bz=EBz, within_contribution=within_contrib,
               census=cen, q=a.q)

    # ---- report ----------------------------------------------------------
    W = 74
    print("=" * W)
    print("CAFE report: %s   (%s schema, %d videos, %d frames, %d positive)"
          % (tag, d["schema"], V, len(d["s"]), cen["n_pos"]))
    print("=" * W)
    print("census      : w=%.4f  omega_B=%.5f  (within-video pairs %.4f%% of all pairs)"
          % (cen["w"], cen["omega_B"], 100 * cen["within_share"]))
    print("              exchange rate: 1 reported point = %.1f within-video points"
          % cen["exchange"])
    if np.isfinite(th_ci[0]):
        print("reported    : theta = %.4f  [%.4f, %.4f]   (video-clustered 95%% CI)"
              % (theta, th_ci[0], th_ci[1]))
    else:
        print("reported    : theta = %.4f   (CI skipped: split too large to bootstrap)"
              % theta)
    print("localization: B     = %.4f  [%.4f, %.4f]" % (B, b_ci[0], b_ci[1]))
    if np.isfinite(Bz):
        print("              B|z   = %.4f  [%.4f, %.4f]   (motion-matched, q=%d)"
              % (Bz, bz_ci[0], bz_ci[1], a.q))
        print("              D = B - B|z = %.4f ; motion share delta_mot = %.3f"
              % (D, delta_mot))
    print("evidence    : E(B) = %.2f%s" % (EB, ("   E(B|z) = %.2f" % EBz) if np.isfinite(Bz) else ""))
    print("bounds      : delta* = %.3f  (above-chance certified for nuisance share < delta*)"
          % delta_star)
    print("              tau in [%.4f, %.4f] at 20%% contamination" % (tau_lb, tau_ub))
    print("attribution : the within-video block contributes %.5f of theta (%.2f%%)"
          % (within_contrib, 100 * within_contrib / theta if theta else 0.0))

    if a.claim_delta_auc is not None:
        claim = a.claim_delta_auc / 100.0  # given in points of the reported metric
        need_b = claim / max(cen["omega_B"], 1e-12)
        print("-" * W)
        print("claim check : you want to report +%.4f of the headline metric (%.1f points)."
              % (claim, a.claim_delta_auc))
        print("              That requires dB = claim / omega_B = %.4f of B"
              % need_b)
        if a.sigma_theta:
            print("              The metric's own seed sd is %.4f, so a change of that"
                  % a.sigma_theta)
            print("              size is %.1f sigma of seed noise."
                  % (claim / a.sigma_theta))
            print("              Resolvable only if sd(dtheta) <= %.1e."
                  % (need_b * cen["omega_B"]))
        if B + need_b > 1.0:
            print("              NOTE: dB is larger than the remaining headroom (1 - B = %.4f)."
                  % (1.0 - B))
        print("              Report the second column, or do not claim localization.")
        out["claim"] = dict(delta_headline=claim, required_delta_B=need_b,
                            sigma_theta=a.sigma_theta)

    print("=" * W)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)
        print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
