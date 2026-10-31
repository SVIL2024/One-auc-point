# Data in this repository

Everything needed to reproduce the paper's numbers is here.  Nothing that is
large, licensed by someone else, or private is.

## What is included

### `results/scores/*.npz` — per-frame scores (14 files, ~9 MB)

One file per checkpoint.  Arrays are *frame-level*: each snippet's score is
repeated 16 times, so that frame labels line up with frame scores.

| key | shape (UCF) | dtype | meaning |
|---|---|---|---|
| `s2` | `(69368,)` | float32 | anomaly score per frame, from the two-branch head (`1 - softmax(logits2)[:,0]`) |
| `s1` | `(69368,)` | float32 | one-class branch score (`sigmoid(logits1)`), present only in the VadCLIP files |
| `lens` | `(290,)` | int64 | frames per video, in test-list order |
| `gt` | `(69368,)` | int64 | frame label, 1 = annotated anomaly |
| `mlocal` | `(69368,)` | float32 | within-video motion nuisance (5-frame smoothed |dx|), used to stratify `B|z` |

Shapes are for UCF-Crime (290 test videos, 69,368 frames).  XD-Violence files
use 800 videos and 145,706 snippets; `results/scores/scores_xd_base_s789.npz`
stores them without the 16x repeat (`s2` length 2,331,296, `lens` in frames) --
`cafe/reproduce_from_scores.py` handles both layouts and says so.

The score files were produced by `cafe/cafe_map11.py` (our arms),
`cafe/dsanet_xd_dump.py` (XD arms) and `cafe/cafe_vadclip.py` (the released
VadCLIP weights).  All three write the same format.

### `results/gt/*.npy` — anomaly segments (2 files, 14 kB)

| file | shape | meaning |
|---|---|---|
| `gt_segment_ucf.npy` | `(290,)` object array of `(n,2)` float arrays | `[start, end]` per annotated segment, in units of **16 frames** (divide by 16 for snippet indices) |
| `gt_label_ucf.npy` | `(290,)` object array of str arrays | class label per segment; normal videos carry the placeholder `'A'` and no segments |

Only 140 of the 290 test videos have annotations; the other 150 are normal and
carry `'A'`.  There are 156 annotated segments in total.  `cafe_tloc.py` uses
exactly this convention and scores a proposal from a normal video as a false
positive.

### `results/*.json` — every number in the paper

| file | contents |
|---|---|
| `table1.json` | 12 UCF rows: `theta, A, B, Bz, C, within, corr` (11 arms + the voided arm) |
| `table1_all.json` | the 20-row census the paper's §6.1 is drawn from (includes superseded XD rows — read the docstring of `cafe_table1.py`) |
| `map11.json` | detection mAP at 5 IoU thresholds, per arm, plus the same decomposition |
| `official12.json` | reference-protocol mAP, independent reimplementation, frame-level AUC and AP |
| `tloc.json` | TLoC-style `AP@0.5` / `AP@0.3` per arm |
| `vadclip.json` | the external anchor (released VadCLIP weights), both datasets |
| `xd_table_raw.json` | the four XD arms: `theta, A, B, Bz, C, w_A, w_B, w_C, roc1, ap1` |
| `xd_scores.json` | XD ingest summary |
| `perm.json` | permutation `p`-values, n = 12 and n = 11 |
| `paired11.json`, `paired12.json` | paired video-level bootstrap of the lapse audit |
| `pair_census.json` | pair counts and weights (`w`, `lam`) |
| `dsanet_dump.json`, `dump_*.json` | raw dumps: per-checkpoint `d1`/`d2` blocks with `theta, A, B, B_cond, C, w, lam` |
| `atlas.json`, `probe*.csv.json`, `rep_ucf_base.json` | the training-free scorer atlas and probe outputs of §6.1/§6.3 |

### `results/logs/*.log` — training logs of the eleven arms

Per-epoch mAP at five IoU thresholds and loss terms, as printed by the host
code.  Two arms (`base_s234`, `res_s234`) have no log because their runs
predate log capture; their weights and scores are complete.

## What is NOT included

| excluded | why | how to get it |
|---|---|---|
| model checkpoints (`*.pth`, ~50 MB each) | size, and the DSANet host code is a third party | retrain with `third_party/DSANet`; the flags per arm are `ARM_FLAGS` in `cafe/dsanet_dump.py` |
| CLIP feature files (UCF ~1.6 GB) | size, and they are derived from a licensed dataset | `third_party/DSANet/features` — see the DSANet/VadCLIP repos and `docs/THIRD_PARTY.md` |
| UCF-Crime / XD-Violence videos | licensed by their authors | [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/), [XD-Violence](https://roc-ng.github.io/XD-Violence/) |
| `cache_ucf_train.npz` (3.3 GB) | training-frame feature cache | rebuilt by `cafe/exploratory/cafe_probe7.py` |
| released VadCLIP weights (650 MB each) | third-party license | see `docs/THIRD_PARTY.md` |

## Provenance and checksums

* The eleven UCF arms are DSANet (AAAI 2026) with the flag variations listed in
  `cafe/dsanet_dump.py`; `base_s234`, `base_s555`, `base_s789` are three seeds of
  the same configuration.
* `ucf_motion9.72v_s234` is **voided**: its archive flags do not rebuild the
  archived theta, so it is excluded from every eleven-arm statistic and kept only
  in `table1.json` for the version census.
* `results/scores/scores_ucf_*` files are byte-for-byte what the server wrote;
  the only post-processing applied anywhere in this repository is the one
  renaming of a misnamed array (`s2|s` -> `s2`) in
  `scores_xd_base_s789.npz`.
* UCF-Crime pair census on the test split: `w = 0.631653` (cross-video share),
  `lam = 0.003880` (share of the same-video pairs after the hard block).
  Both are properties of the split, not of any model, and every script that
  needs them reads them from the data rather than hardcoding them (except
  `cafe_table1.py`, which prints the analytic contributions with
  `W = dict(A=0.631653, B=0.003880, C=0.364467)`).
