# Reproducing every number in the paper

Three levels, cheapest first.  Level 1 needs nothing but `numpy` and the files in
this repository.

---

## Level 1 — the decomposition table, from the released scores (CPU, ~10 s)

```bash
python cafe/reproduce_from_scores.py
```

Reads `results/scores/*.npz` (one file per checkpoint: frame-level scores,
per-video lengths, frame labels, within-video motion) and recomputes `theta`,
`A`, `B`, `B|z`, `C` for each one.  Expected output ends with

```
checkpoint                 theta       A       B     B|z       C
ucf_base_s234             0.8800  0.9751  0.7270  0.6726  0.7167
...
ucf_vadclip               0.8560  0.9737  0.7157  0.6710  0.6536
xd_base_s789              0.9449  0.9962  0.8281  0.7998  0.8258
xd_vadclip                0.7060  0.7753  0.6145  0.5942  0.5452

largest |recomputed - stored| over all files:
    A 1.11e-16, B 0.00e+00, B_cond 8.40e-04, C 1.11e-16, theta 1.11e-16
```

`theta`, `A`, `B`, `C` are recovered to machine precision.  `B|z` differs by
`8e-4` because the stored table was produced on the snippet grid while the score
files store the frame-level repeat, which moves the quintile boundaries of the
stratifier.  That is the only expected deviation, and it does not move any
reported value in its fourth decimal.

### What the columns are

| column | definition |
|---|---|
| `theta` | pooled AUC over all (positive frame, negative frame) pairs — the number a benchmark prints |
| `A` | AUC of positive frames against negative frames of *normal videos* |
| `B` | pair-weighted AUC of positive frames against background frames of the *same* video |
| `B|z` | `B` recomputed within quintiles of the within-video motion of each video |
| `C` | residual block: `hard = (wp/tot) * B + (1 - wp/tot) * C` |
| `w` | share of pairs that are cross-video (`0.6317` on UCF-Crime, a property of the split) |

The identity `theta = w*A + (1-w)*hard` with `hard` as above is Theorem 1 and
holds exactly; `reproduce_from_scores.py` asserts it numerically to `1e-12`.

---

## Level 2 — the analyses that produce the paper's tables and claims (CPU, ~1 min)

All of these read from `results/` and write back into `results/`.  Run them from
the repository root.

| command | writes | reproduces |
|---|---|---|
| `python cafe/cafe_table1.py` | `results/table1_all.json` | the 20-row checkpoint census of §6.1 (see the docstring: it includes the voided arm and four superseded XD rows) |
| `python cafe/cafe_rho_ci.py` | prints | the correlation table of §6.2/§6.6 with Fisher-z intervals and partial correlations |
| `python cafe/cafe_rho_ext.py` | prints | §6.7: `rho(theta, mAP)`, `rho(B, mAP)`, partials, and the agreement between the two localization metrics |
| `python cafe/cafe_perm.py` | `results/perm.json` | permutation `p`-values (10^5 permutations, seed 20261031) for every correlation in §6.7 |
| `python cafe/cafe_paired.py` | `results/paired11.json` | §6.6 lapse audit: `dB = 0.0495`, interval `[-0.005, +0.095]`, flip rate 0.51 |
| `python cafe/cafe_tloc.py --ds ucf` | `results/tloc.json` | the TLoC-style `AP@0.5` column of Table 7 (`AP@0.3` alongside) |
| `python cafe/cafe_interval.py` | prints | the video-cluster bootstrap intervals of §5.9 |

`cafe_perm.py` and `cafe_paired.py` need only `numpy`; the others need `scipy`
(`cafe_table1.py`, `cafe_rho_ci.py`, `cafe_tloc.py`).  `cafe_tloc.py` and
`cafe_official12.py` also use `sklearn.metrics` for the frame-level AUC/AP.

Expected headline values, so you can check a run without reading the paper:

```
rho(theta, mAP@0.5)   = +0.71   [+0.24, +0.91]  p_perm = 0.012   (n = 12)
rho(B,     mAP@0.5)   = +0.15   [-0.47, +0.66]  p_perm = 0.65
rho(B|z,   mAP@0.5)   = +0.08   [-0.52, +0.62]  p_perm = 0.82
partial rho(mAP, B | theta) = -0.32            p_perm = 0.31
partial rho(mAP, theta | B) = +0.74            p_perm = 0.007
rho(repo mAP@0.5, TLoC AP@0.5) = +0.57
```

---

## Level 3 — re-extract the scores from the checkpoints (one 24 GB GPU, ~2 h)

Needed only if you want to go from raw videos + weights to `results/scores`.

### 3a. Put the host code and data in place

```bash
git clone <DSANet> third_party/DSANet        # see docs/THIRD_PARTY.md
# weights: third_party/DSANet/model/ucf_*.pth
# features: third_party/DSANet/features/... (CLIP features, see docs/DATA.md)
```

The eleven UCF arms differ only in training flags (`motion`, `attr`, `cons`,
`core`, `res`), which are the `ARM_FLAGS` dictionary in `cafe/dsanet_dump.py`;
three of them are seeds of the same configuration.

### 3b. Point the scripts at your checkout

The research scripts still carry the absolute paths of the machine that produced
the paper (`E:/program/...`).  One command rewrites them to your clone:

```bash
python scripts/retarget.py                # dry run: shows the diff
python scripts/retarget.py --apply
```

After that, `E:/program/DSANet` is `<clone>/third_party/DSANet`,
`E:/program/CAFE/out` is `<clone>/results`, and `E:/program/EDSS-main` is
`<clone>/third_party/VadCLIP`.  The working-directory idioms are already
portable in this release, so the CPU path (levels 1 and 2) never needs this step.

For a **second** machine, `_remote.py`-style scripting was used in the original
work; it is not part of this release (it contains credentials).

### 3c. Run the ingest

```bat
scripts\run_ucf_scores.bat      :: 11 arms, ~1 min each -> results/map11.json + scores
scripts\run_ucf_official.bat    :: 11 arms, ~1 min each -> results/official12.json
scripts\run_vadclip.bat         :: the external anchor -> results/vadclip.json
scripts\run_xd_scores.bat       :: 4 arms, ~15 min total -> results/xd_scores.json
```

Each batch runs **one process per arm** on purpose: this GPU raises a sticky
`CUDA error: an illegal memory access` every few dozen forward passes, and a
fresh process keeps one bad arm from killing the batch.  The runners retry a
failed arm once.

The XD localization column is not produced by the batch: the reference
implementation hardcodes UCF-Crime's 14 classes and XD-Violence has 6, so `dmAP`
raises `IndexError: index 6 is out of bounds for axis 1 with size 6`.  Use
`cafe/cafe_tloc.py --ds xd` instead, and note that the XD anchor
(`cafe_vadclip.py --ds xd --skip-map`) is reported separately in §6.9 because
its score convention differs.

### 3d. The official detection mAP

`cafe/cafe_official12.py` runs the reference implementation and an independent
reimplementation **in the same process** and reports the largest absolute
deviation between them.  It is `0.0` on all 5 IoU thresholds for all 11 arms
(55 cells), which is the evidence behind the sentence "this column is the
published protocol, not our variant".

---

## Environment used for the paper

| | |
|---|---|
| GPU | one NVIDIA RTX A5000, 24 GB |
| torch | 2.5.1+cu121, Python 3.12.4 |
| CPU path tested with | Python 3.8 + numpy 1.24 (and 3.12 + numpy 2.x) |
| paper build | pandoc + MiKTeX (`pdflatex`), see `paper/tools/README` notes in `docs/FILE_MAP.md` |

## Troubleshooting

* **`ModuleNotFoundError: ipdb`** — the released DSANet imports `ipdb` in
  `adapter_modules.py`.  `cafe/shim/ipdb.py` is a stub; put it on `PYTHONPATH`
  (the batch runners do this) or `pip install ipdb`.
* **`CUDA error: an illegal memory access`** — rerun that arm; the runners
  already retry once.
* **`ValueError: Input contains NaN`** — seen once on `motion6shuf5678_s234`;
  rerunning the arm fixed it.  If it persists, the feature file for that arm is
  corrupt.
* **All `mAP` numbers ~2-6%** — expected: that column is a *detection* metric
  (one segment per video, IoU-matched), not the frame-level AUC.  Its absolute
  scale is a property of the metric; see §6.7.
