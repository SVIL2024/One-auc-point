"""A self-contained TLoC-style localisation AP for the criterion-validity check
(section 6.7 of the paper).

Why not reuse `utils/ucf_detectionMAP.getDetectionMAP`: that routine thresholds
`max - (max-min)*thr` on per-class softmax outputs and then NMS-merges sixty
class proposals per video.  Its absolute scale is not the published one -- the
released VadCLIP checkpoint scores 88.08 AUC under this codebase but only 2.95
mAP@0.5, while the TLoC literature reports tens of points -- so it cannot be
quoted as "the detection mAP that UCF-Crime papers publish".

The protocol implemented here is the one the detection literature uses, spelled
out so it can be checked by hand:

  1. per video, sweep nine relative thresholds `max - c*(max-min)`, c in
     0.1..0.9, and take the maximal runs above each threshold as proposals;
  2. a proposal's confidence is the *mean* raw score inside the run (the max
     would be attained by the whole-video run at the lowest threshold and would
     rank every video's entire timeline first);
  3. rank all proposals of the test set by that confidence, match them greedily
     to ground-truth segments at IoU >= 0.5, one ground-truth segment per
     proposal;
  4. interpolate the precision-recall curve and integrate.

Ground truth: the UCF-Crime annotation file stores segments in frames at the
same 16-frame-per-snippet granularity as the scores (`gt_segment_ucf.npy`), and
the 150 normal test videos carry the placeholder label 'A'; they are treated as
having no segment, so every proposal raised inside them is a false positive.

Plain numpy, no checkpoint dependency: it reads the `scores_*.npz` dumps.
"""
import argparse
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
GRID = np.arange(0.1, 1.0, 0.1)


def load_scores(tag, ds="ucf"):
    z = np.load(os.path.join(OUT, "scores", "scores_%s_%s.npz" % (ds, tag)))
    s, lens = z["s2"].astype(np.float64), z["lens"]
    per, off = [], 0
    for n in lens:
        n = int(n)
        per.append(s[off:off + n])
        off += n
    return per


def load_segments(ds="ucf"):
    seg = np.load(os.path.join(OUT, "gt", "gt_segment_%s.npy" % ds), allow_pickle=True)
    lab = np.load(os.path.join(OUT, "gt", "gt_label_%s.npy" % ds), allow_pickle=True)
    out = []
    for v, l in zip(seg, lab):
        if len(l) and str(l[0]) == "A":          # normal video placeholder
            out.append([])
        else:
            out.append([(float(a) / 16.0, float(b) / 16.0) for a, b in v])
    return out


def proposals(s):
    """maximal runs above nine relative thresholds; confidence = mean raw score"""
    lo, hi = float(np.min(s)), float(np.max(s))
    if hi - lo < 1e-12:
        return []
    out = []
    for c in GRID:
        thr = hi - c * (hi - lo)
        m = s > thr
        if not m.any():
            continue
        idx = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
        for a, b in zip(idx[0::2], idx[1::2]):
            out.append((int(a), int(b), float(s[a:b].mean())))
    return out


def iou(a, b, A, B):
    inter = max(0.0, min(b, B) - max(a, A))
    union = max(b, B) - min(a, A)
    return inter / union if union > 0 else 0.0


def average_precision(per_video, segs, iou_thr=0.5):
    hits, scores, npos = [], [], 0
    for s, g in zip(per_video, segs):
        npos += len(g)
        cand = proposals(s)
        cand.sort(key=lambda c: -c[2])
        free = list(g)
        for a, b, sc in cand:
            best, bj = 0.0, -1
            for j, (A, B) in enumerate(free):
                v = iou(a, b, A, B)
                if v > best:
                    best, bj = v, j
            if best >= iou_thr:
                hits.append(1)
                free.pop(bj)
            else:
                hits.append(0)
            scores.append(sc)
    if npos == 0 or not hits:
        return float("nan")
    hits = np.asarray(hits, float)
    scores = np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable")
    hits = hits[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(1.0 - hits)
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / npos
    ap, prev = 0.0, 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev)
        prev = r
    return float(ap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="ucf")
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", default=os.path.join(OUT, "tloc.json"))
    args = ap.parse_args()
    tags = [t for t in args.tags.split(",") if t] or [
        t[len("scores_%s_" % args.ds):-4]
        for t in sorted(os.listdir(os.path.join(OUT, "scores")))
        if t.startswith("scores_%s_" % args.ds) and t.endswith(".npz")]
    segs = load_segments(args.ds)
    res = {}
    if os.path.exists(args.out):
        try:
            res = json.load(open(args.out))
        except Exception:
            res = {}
    for tag in tags:
        per = load_scores(tag, args.ds)
        a50 = average_precision(per, segs, 0.5)
        a30 = average_precision(per, segs, 0.3)
        res["%s_%s" % (args.ds, tag)] = dict(AP50=a50, AP30=a30)
        print("%-28s AP@0.5 %6.2f%%   AP@0.3 %6.2f%%"
              % (tag, 100 * a50, 100 * a30), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
