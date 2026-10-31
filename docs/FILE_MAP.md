# File map

What every file is for.  Scripts are grouped by the job they did for the paper,
not by the order they were written in.

## `cafe/` — the measurement (19 scripts)

The pipeline: ingest scores → decompose → identify/bound → cross-check against
independent criteria.

| script | job | writes |
|---|---|---|
| `reproduce_from_scores.py` | recompute the whole decomposition from the released scores, CPU, numpy only | prints; compares against `results/table1.json` |
| `dsanet_dump.py` | run DSANet checkpoints and dump per-frame scores + the decomposition; holds `ARM_FLAGS` (the eleven arms) | `results/dump_*.json`, `results/scores/scores_ucf_<tag>.npz` |
| `dsanet_xd_dump.py` | same for XD-Violence through the repo's own XD test path | `results/xd_scores.json`, `results/xdstack_*.npz` |
| `cafe_vadclip.py` | measure the released VadCLIP weights with our estimators; uses VadCLIP's own head, not DSANet's `refine` | `results/vadclip.json`, `results/scores/scores_*_vadclip.npz` |
| `cafe_map11.py` | detection mAP@0.5 per arm with the repo protocol, next to `theta/B/B|z` | `results/map11.json`, score npz |
| `cafe_official12.py` | reference mAP + independent reimplementation + frame-level AUC/AP in one process | `results/official12.json` |
| `cafe_tloc.py` | our own TLoC-style localisation AP (9 relative thresholds, greedy IoU matching) | `results/tloc.json` |
| `cafe_table1.py` | checkpoint census from the dumps | `results/table1_all.json` |
| `cafe_rho_ci.py` | rank correlations among the eleven arms, Fisher-z and partial | prints |
| `cafe_rho_ext.py` | the same with the anchor as a twelfth point and the second localisation metric | prints |
| `cafe_perm.py` | checkpoint-level permutation test (10^5 permutations) for every §6.7 correlation | `results/perm.json` |
| `cafe_paired.py` | paired video-level bootstrap of the lapse audit (gap, interval, flip rate) | `results/paired<N>.json` |
| `cafe_evalue.py` | cluster bootstrap + E-value for `B` (how much unmeasured confounding would explain it away) | `results/evalue.json` |
| `cafe_interval.py` | frame-level i.i.d. bootstrap vs video-cluster bootstrap (the 12x width gap) | prints |
| `cafe_adjust.py` | label-free post-hoc estimate of the within-video term from features only | prints |
| `cafe_pairs.py` | data-only pair census: `w`, `lam`, the three comparison blocks | `results/pair_census.json` |
| `cafe_overlap.py` | positivity check for the motion-stratified estimator | prints |
| `cafe_xd_table.py` | XD decomposition at snippet resolution with the corrected ground truth | `results/xd_table_raw.json` |
| `reproduce_from_scores.py` | (see above) | |

## `cafe/exploratory/` — the probes behind the negative results (19 scripts)

Kept because the paper reports that these interventions **failed**; the code is
the evidence.  Each header states the question it asked and what it found.

| script | question |
|---|---|
| `cafe_probe.py` | what are the three blocks, before any model? (pair census on real scores) |
| `cafe_probe2.py` | does a causally motivated estimator beat plain MIL? |
| `cafe_probe3.py` | which weak-label loss identifies the within-video functional? |
| `cafe_probe4.py` | is the within-video contrast a causal estimand? |
| `cafe_probe5.py` | is it a causally *adjusted* estimand (motion as confound)? |
| `cafe_probe6.py` | does an event signal survive beyond the nuisance `z`? |
| `cafe_probe7.py` | nuisance-orthogonal training (`gamma` penalty), multi-seed |
| `cafe_report.py` | one-command report for any scorer: census + three blocks |
| `cafe_atlas.py` | data-side constants: `w`, `lam`, exchange rate |
| `cafe_canon.py` | canonical decomposition for every exported checkpoint + Prop. 5 bounds |
| `cafe_bzcheck.py` | which `B|z` definition produced which number (bin count, nuisance column) |
| `cafe_arms.py` | collect every arm's `metrics.json` into one CSV |
| `cafe_xd_prep.py`, `cafe_xd_prep2.py` | convert captured XD score stacks to the `scores_*.npz` format |
| `xd_probe.py` | inspect XD test-list rows and score shapes |
| `ck_probe.py` | inspect a checkpoint's structure before loading it into the host model |
| `_inspect_dump.py` | print the structure of a dump json / score npz |
| `_mapdiag.py`, `_mapdiag2.py` | diagnose the reference mAP: what it returns, why the scale is small |
| `_xd_gt.py` | how XD ground truth is stored (frame-level `gt.npy` vs snippet scores) |

## `scripts/` — portable entry points

| file | job |
|---|---|
| `retarget.py` | rewrite the `E:/program/...` paths to your clone (`--apply`), or just the working-directory idioms (`--structural-only`) |
| `run_ucf_scores.bat` | eleven arms, one process each, retry once |
| `run_ucf_official.bat` | the same for the reference mAP column |
| `run_xd_scores.bat` | four XD arms |
| `run_vadclip.bat` | the external anchor (UCF, and XD if its weights are present) |

## `results/` — every number

| path | contents |
|---|---|
| `*.json` | the tables and test outputs; see `docs/DATA.md` for a file-by-file list |
| `scores/*.npz` | per-frame scores, per-video lengths, frame labels, motion |
| `gt/*.npy` | UCF-Crime segment ground truth (16-frame units, `'A'` = normal video) |
| `logs/*.log` | training logs of the eleven arms |

## `paper/` — the manuscript

| file | contents |
|---|---|
| `CAFE.pdf` | the compiled 47-page manuscript |
| `CAFE.docx` | Word version (8 real tables, Figure 1, no literal LaTeX) |
| `CAFE_manuscript.md` | the single source of truth (Markdown + LaTeX math) |
| `CAFE.tex` | generated LaTeX (do not hand-edit: the `authoryear` option is injected by the build) |
| `references.bib` | 20 references, checked against the publishers |
| `fig1.png`, `fig1.pdf` | Figure 1 (census, exchange rate, reordering) |
| `tools/` | `_rebuild_cafe.py` (the one-command build) and the stage scripts it supersedes |

## `docs/`

| file | contents |
|---|---|
| `REPRODUCE.md` | three reproduction levels, expected outputs, troubleshooting |
| `DATA.md` | what is in `results/`, what is excluded and where to get it |
| `THIRD_PARTY.md` | DSANet, VadCLIP, the reference mAP, datasets, and why RTFM is absent |
| `FILE_MAP.md` | this file |
| `handoff_notes_zh.md` | the Chinese build notes kept during the work: the full number registry, the review rounds, and every trap that cost time |

## Conventions worth knowing

* **Scores are frame-level** (snippet score repeated 16x) so that labels line up;
  the decompositions are computed on the repeated arrays, which leaves `A`, `B`,
  `C` unchanged and moves `B|z` in the fourth decimal only.
* **`dmAP` returns a percentage** (0-100), not a fraction.  Multiplying it by 100
  again was the first bug in `cafe_map11.py`.
* **`motion9.72v_s234` is voided** — its archived flags do not rebuild the
  archived `theta`.  It appears in `table1.json` for the version census and in no
  eleven-arm statistic.
* **One process per arm** — the GPU raises a sticky illegal-memory-access error
  every few dozen forward passes.
