# Third-party components

This repository contains **our** measurement code, our scores, our analyses and
our manuscript.  It does not redistribute any third-party code or weights.  The
components you need in order to run the GPU path are listed here with what we
know about their licensing; check the original repositories before publishing
anything derived from them.

## 1. DSANet — the host architecture

* Paper: *DSANet*, AAAI 2026, arXiv **2511.10334**.
* Our eleven UCF-Crime arms are DSANet with flag variations
  (`motion`, `attr`, `cons`, `core`, `res`); the flags are visible in
  `ARM_FLAGS` in `cafe/dsanet_dump.py`.
* License: not declared in the release we used.  We do not ship its source; clone
  it into `third_party/DSANet` from its own repository and follow its terms.
* We patched nothing in the released model code except adding an `ipdb` stub on
  `PYTHONPATH` (`cafe/shim/ipdb.py`), because `src/utils/adapter_modules.py`
  imports `ipdb` unconditionally.
* The checkpoint names in `results/` and `results/logs/` follow DSANet's own
  naming (`base`, `motion6`, `attr`, `cons0`, `core8g1`, `core14txt`, `res`).

## 2. VadCLIP — the external anchor

* Paper: *VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video
  Anomaly Detection*, AAAI 2024.
* We used the **released** weights `vadclip_model_ucf.pth` (649,186,055 B) and
  `vadclip_model_xd.pth` (636,572,191 B) through a VadCLIP checkout
  (`third_party/VadCLIP`, an internal fork of the authors' code that keeps
  `src/model.py`'s `CLIPVAD`, the official test path and the option defaults).
* These weights are what makes the external-anchor result in §6.7/§6.9 possible:
  someone else's training run, someone else's score convention.  They are **not**
  redistributed here.
* License: research use, as declared by the authors.  Cite the VadCLIP paper when
  using the anchor numbers.

## 3. The reference detection mAP (`getLocMAP`)

The detection-mAP column follows the implementation that UCF-Crime papers share
(the `ucf_detectionMAP.py` distributed with the DSANet/VadCLIP codebases, itself
derived from Zhong et al.'s Sultani-CVPR-lineage evaluation):

```
threshold = max - (max - min) * 0.6        # one threshold, thr_set = [0.6]
segment confidence = max(score in segment) + 0.7 * class_score
per-video NMS at 0.6, per-class AP = sum(precision * tp) / n_gt_segments
return 100 * mean over the 14 UCF-Crime classes
```

Two facts about it that the paper relies on:

* it hardcodes **14 classes**, which is why XD-Violence (6 classes) raises
  `IndexError: index 6 is out of bounds for axis 1 with size 6` and why the XD
  localization numbers use `cafe/cafe_tloc.py` instead;
* `smooth()` in the released file is the identity (the Savitzky-Golay call is
  commented out), so the "smoothed" predictions are the raw ones.

`cafe/cafe_official12.py` reimplements this metric independently and reports the
maximum absolute deviation from the reference: `0.0` over 5 IoU × 11 arms.

## 4. CLIP features

Both DSANet and VadCLIP consume CLIP ViT-B/16 features (512-d per snippet) that
are extracted from the videos and cached.  The extractor belongs to the upstream
repositories.  We do not ship the features; regenerate them with the upstream
scripts, or use the feature dumps those projects publish.

## 5. Datasets

| dataset | use here | terms |
|---|---|---|
| **UCF-Crime** | 290 test videos, 14 anomaly classes, 69,368 frames | research use; download from the authors' page |
| **XD-Violence** | 800 test videos, 6 classes, 145,706 snippets | research use; download from the authors' page |

The ground truth we ship is only the *segment list* needed to compute detection
metrics (`results/gt/*.npy`), not any video content.

## 6. RTFM — why it is not here

The natural third anchor would be the released RTFM weights (Tian et al.,
CVPR 2021).  We could not add it:

* the official weights are distributed via Google Drive, not a package index;
* `huggingface.co` is unreachable from the experiment machine and its mirror
  `hf-mirror.com` has **no** RTFM checkpoint (`api/models?search=RTFM` returns
  three unrelated repositories, none of them video anomaly detection);
* there is no RTFM code or weight in the working tree we worked from.

Training an RTFM-style top-k MIL head ourselves was considered and rejected:
a head we train is not an external anchor, so it cannot address the criticism it
would be meant to answer (§7.6 states this explicitly).

## 7. Build tools (paper only)

`pandoc` for Markdown → LaTeX, MiKTeX/`pdflatex` for the PDF, `bibtex` for the
references, `matplotlib` for Figure 1.  None of them are redistributed; the
scripts in `paper/tools/` are ours.

## Citation obligation

If you use the decomposition, cite our manuscript (`CITATION.cff`).  If you use
the anchor numbers or the DSANet arms, cite DSANet and VadCLIP as well.
