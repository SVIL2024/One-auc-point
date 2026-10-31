# CAFE: what the reported number measures

Code, scores, ground truth and paper for

> **What Does One AUC Point Buy? Localization Ability in Weakly-Supervised
> Video Anomaly Detection** *(manuscript under review, 2026)*

A weakly supervised video anomaly detector is trained on video-level labels and
evaluated with a single number.  This repository contains the measurement study
of that number: an exact decomposition of the pooled AUC into three blocks, the
estimators that identify each block, the causal check that decides whether the
within-video block is a confound or a mediator, and a criterion-validity test
that asks whether the second block predicts any independent localization metric.

Everything here runs from the published scores alone on a CPU in seconds, and
from the stored checkpoints on a single 24 GB GPU in about an hour.

---

## Headline result

With `theta` the pooled AUC that a benchmark prints, the identity

```
theta = w * A + (1 - w) * hard ,        hard = lam * B + (1 - lam) * C
```

holds exactly (max absolute error `1e-16` over sixteen checkpoints), where

| block | meaning | value on UCF-Crime |
|---|---|---|
| `A` | positive frames vs. negative frames of **normal videos** | 0.967–0.979 |
| `B` | positive frames vs. background of the **same video**, pair-weighted | 0.708–0.758 |
| `C` | the residual block | 0.681–0.717 |
| `w` | share of pairs that are cross-video (a property of the split, not of the model) | 0.6317 |

`w = 0.6317` and `lam = 0.0039` mean the headline number is **99.6% A + 36.5% C +
0.4% B**: a detector can score well by ranking *videos* while barely ordering
frames *within* a video, and a per-video constant scorer already reaches
`theta = 0.8234` — 99.7% of a model trained on the same frames.

Reproduced Table 3 (eleven UCF-Crime arms we trained, plus the released VadCLIP
weights as an external anchor):

| checkpoint | theta | A | B | B\|z | C | det. mAP@0.5 | det. avg mAP | TLoC AP@0.5 |
|---|---|---|---|---|---|---|---|---|
| base_s234              | 0.8800 | 0.9751 | 0.7270 | 0.6729 | 0.7167 | 5.68 | 10.10 | 4.46 |
| base_s555              | 0.8705 | 0.9752 | 0.7236 | 0.6688 | 0.6907 | 4.12 | 11.30 | 2.53 |
| base_s789              | 0.8772 | 0.9773 | 0.7558 | 0.7169 | 0.7050 | 5.18 | 12.84 | 3.94 |
| motion6_s555           | 0.8717 | 0.9728 | 0.7180 | 0.6604 | 0.6981 | 2.88 | 10.15 | 2.86 |
| motion6_s789           | 0.8778 | 0.9728 | 0.7576 | 0.7230 | 0.7144 | 4.61 | 12.46 | 2.41 |
| motion6shuf5678_s234   | 0.8616 | 0.9670 | 0.7171 | 0.6786 | 0.6806 | 2.44 | 7.17 | 1.85 |
| attr_s234              | 0.8724 | 0.9788 | 0.7080 | 0.6545 | 0.6899 | 5.82 | 13.03 | 2.79 |
| cons0_s234             | 0.8759 | 0.9774 | 0.7235 | 0.6744 | 0.7016 | 5.46 | 11.16 | 2.51 |
| core8g1_s234           | 0.8684 | 0.9759 | 0.7124 | 0.6621 | 0.6837 | 4.05 | 10.64 | 2.45 |
| core14txt_s234         | 0.8717 | 0.9725 | 0.7169 | 0.6728 | 0.6989 | 5.03 | 11.88 | 4.24 |
| res_s234               | 0.8682 | 0.9767 | 0.7467 | 0.7093 | 0.6814 | 4.63 | 9.83 | 1.89 |
| **VadCLIP (released)** | **0.8560** | **0.9737** | **0.7157** | **0.6713** | **0.6536** | **2.95** | **7.36** | **1.18** |

`B|z` is `B` after stratifying each video on its within-video motion; the
stratified value stays above 0.65, which is why within-video motion is treated
as a *mediator* rather than a confound (Section 5 of the paper).

Two results that are deliberately unflattering:

* **The second block does not predict localization.** Spearman `rho(B, det.
  mAP@0.5) = 0.15 [-0.47, 0.66]`, `p = 0.65`; `rho(B, TLoC AP@0.5) = 0.11`,
  `p = 0.74`; partial `rho(mAP, B | theta) = -0.32`, `p = 0.31`.  The reported
  number `theta` tracks detection mAP (`rho = 0.71`, `p = 0.012`) and mAP does
  not track within-video order.  The paper reports this and withdraws the
  replacement-metric claim it would otherwise have made.
* **Rankings inside the family are not resolvable.** The largest between-arm gap
  is `dB = 0.0495` with a paired bootstrap interval `[-0.005, +0.095]` that
  contains zero, and the top-2 order flips in 51% of resamples.

---

## Quickstart (CPU, no GPU, no dataset)

Recompute the whole decomposition table from the released scores:

```bash
pip install numpy
python cafe/reproduce_from_scores.py
```

```
checkpoint                 theta       A       B     B|z       C
ucf_base_s234             0.8800  0.9751  0.7270  0.6726  0.7167
...
largest |recomputed - stored| over all files: A 1.1e-16, B 0.0e+00,
    B_cond 8.4e-04, C 1.1e-16, theta 1.1e-16
```

The `B|z` deviation of `8e-4` is expected and documented in the script: the
stored table was produced on the snippet grid while the score files store the
frame-level repeat, so the quantile boundaries of the stratifier move slightly.
`A`, `B`, `C` and `theta` are recovered to machine precision.

The comparison table, the correlation and permutation tests, the paired
bootstrap and the two localization metrics can all be recomputed from
`results/*.json` with the same scripts used for the paper — see
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Full reproduction (one GPU, ~2 hours)

| step | what it does | script | time |
|---|---|---|---|
| 0 | download the third-party host code and weights | [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) | — |
| 1 | ingest UCF-Crime: scores + decomposition for every checkpoint | `cafe/dsanet_dump.py` | ~11 × 55 s |
| 2 | ingest XD-Violence | `cafe/dsanet_xd_dump.py` | ~15 min |
| 3 | external anchor (released VadCLIP weights) | `cafe/cafe_vadclip.py` | ~55 s |
| 4 | analysis: table, localization metrics, correlations, bootstrap | `cafe/cafe_*.py` | < 2 min CPU |

Point every script at your own checkout with

```bash
python scripts/retarget.py --root /path/to/this/repo --apply
```

which rewrites the `E:/program/...` paths hardcoded in the research scripts.

## Layout

```
cafe/                 the measurement code (19 modules, one job each)
cafe/exploratory/     the probes that produced the paper's negative results
cafe/shim/ipdb.py     stub for `import ipdb` in the released host code
results/              every number, as JSON, plus the raw scores and ground truth
  scores/             one .npz per checkpoint: frame scores, lengths, labels, motion
  logs/               training logs of the eleven arms
  gt/                 UCF-Crime segment ground truth used by the localization metrics
paper/                the manuscript (PDF, docx, Markdown source, LaTeX, figure)
paper/tools/          the build chain: Markdown -> LaTeX -> PDF, no hand-edited .tex
scripts/              retarget.py and the Windows batch runners
docs/                 reproduction, data, third-party and file-map notes
```

## Scope and honest limitations

* The ranking claim is confined to **one architecture family** (DSANet host,
  CLIP features, one protocol) plus **one external anchor** (released VadCLIP
  weights).  Third-party RTFM/VadCLIP checkpoints beyond that anchor could not be
  added: the experiment machine has no outbound network, and the RTFM weights
  are not on Hugging Face or its mirrors ([`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)).
* `n = 12` checkpoints is a small sample.  Every correlation is reported with a
  Fisher-z interval and a permutation `p`, and none of the "no association"
  statements is phrased as independence.
* Both localization metrics are **our** implementations of the published
  protocols, not the literature's leaderboard numbers; the detection mAP follows
  the reference implementation shared by UCF-Crime papers and reproduced
  bit-for-bit by an independent reimplementation (`max|diff| = 0.0` over
  5 IoU × 11 checkpoints), but its absolute scale is a property of the metric,
  not of the detector.
* Nothing here claims a new detector.  The contribution is a measurement: three
  blocks, their estimators, one causal verdict, and one criterion-validity check
  reported against interest.

## Citation

```bibtex
@article{cafe2026,
  title  = {What Does One {AUC} Point Buy? Localization Ability in
            Weakly-Supervised Video Anomaly Detection},
  author = {Anonymous},
  year   = {2026},
  note   = {Manuscript under review. Code: this repository}
}
```

See [`CITATION.cff`](CITATION.cff).  Replace `Anonymous` before publishing.

## License

Code: MIT ([`LICENSE`](LICENSE)).  Scores, ground truth and paper: CC BY 4.0.
Third-party code and weights keep their own licenses — see
[`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
