# -*- coding: utf-8 -*-
"""Convert prose author-year citations in CAFE.tex to natbib \\citet/\\citep."""
import io, sys, re

P = r"E:\paper\drafts\2026\CAFE\elsarticle\CAFE.tex"
s = io.open(P, encoding="utf-8").read()

REPL = [
    # --- intro / audits ---
    ("Song and Lee\n(arXiv:2608.21854) observe",
     r"\citet{song2026lookinside} observe"),
    ("Abdulaziz and Bondarev\n(arXiv:2608.11985) show",
     r"\citet{abdulaziz2026auditing} show"),
    ("Song and Lee report the same fraction",
     r"\citet{song2026lookinside} report the same fraction"),
    ("the empirical observation of Song and Lee: a",
     r"the empirical observation of \citet{song2026lookinside}: a"),
    # --- companion study ---
    ("A companion study\n  (DebiasVAD) tried thirteen",
     "A companion study\n  \\citep{debiasvad2026} tried thirteen"),
    ("a companion study (DebiasVAD) found",
     r"a companion study \citep{debiasvad2026} found"),
    # --- partial identification ---
    ("see Manski, 1990, for the original worst-case",
     r"see \citep{manski1990bounds} for the original worst-case"),
    # --- e-value ---
    ("the E-value of VanderWeele and Ding\napplies:",
     "the E-value of \\citet{vanderweele2017sensitivity}\napplies:"),
    # --- experiments host ---
    ("We use DSANet (AAAI-26), a current",
     r"We use DSANet \citep{yin2026dsanet}, a current"),
    # --- relation to audits ---
    ("Song and Lee (arXiv:2608.21854) observe that within-video pairs are 0.071--0.388\\%",
     r"\citet{song2026lookinside} observe that within-video pairs are 0.071--0.388\%"),
    ("Abdulaziz and\nBondarev (arXiv:2608.11985) show",
     "\\citet{abdulaziz2026auditing} show"),
    ("Liu et al.\n(arXiv:2505.19022) identify three limitations",
     "\\citet{liu2025rethinking} identify three limitations"),
    # --- related work: WSVAD ---
    ("(Sultani\net al., CVPR 2018)",
     r"\citep{sultani2018realworld}"),
    ("(Zhong et al., RTFM, ICCV 2021)",
     r"\citep{tian2021rtfm}"),
    ("VadCLIP (Wu et al., CVPR 2024) replaced",
     r"VadCLIP \citep{wu2024vadclip} replaced"),
    ("DSANet (AAAI 2026) adds",
     r"DSANet \citep{yin2026dsanet} adds"),
    ("anomaly detection\n(Kim et al., AAAI 2022) established",
     "anomaly detection\n\\citep{kim2022towards} established"),
    # --- causal VAD ---
    ("Lin et al.~(AAAI 2022) cast",
     r"\citet{lin2022causal} cast"),
    ("with counterfactuals (AAAI 2023)",
     r"with counterfactuals \citep{sun2023event}"),
    ("causal\nrepresentation consistency (CIR, 2023)",
     "causal\nrepresentation consistency \\citep{cir2023causality}"),
    ("learning (CRCL, 2025)",
     r"learning \citep{crcl2025causal}"),
    # --- MIL / identifiability ---
    ("is folklore in the MIL literature\n(Ilse et al., ICML 2018; the noisy-OR/ max identifiability discussions) and is",
     "is folklore in the MIL literature\n\\citep{ilse2018attention}, as are the noisy-OR/max identifiability discussions, and is"),
    ("the formal core of UMIL's (2023) analysis of instance-level bias",
     r"the formal core of the UMIL analysis of instance-level bias \citep{umil2023unbiased}"),
    # --- sensitivity analysis ---
    ("The E-value (VanderWeele and Ding, Annals of Internal\nMedicine 2017) and Rosenbaum-style bounds are",
     "The E-value \\citep{vanderweele2017sensitivity} and Rosenbaum-style bounds\n\\citep{rosenbaum1987sensitivity} are"),
    # --- AUC algebra ---
    ("behind multiclass AUC, Hand and Till 2001).",
     r"behind multiclass AUC, \citep{hand2001simple})."),
    # --- dataset citations ---
    ("Table 1 lists the census for the two standard benchmarks.",
     r"Table 1 lists the census for the two standard benchmarks \citep{sultani2018realworld,wu2020xdviolence}."),
]

bad = []
for old, new in REPL:
    n = s.count(old)
    if n != 1:
        bad.append((n, old[:70]))
        continue
    s = s.replace(old, new)

if bad:
    for n, o in bad:
        print("MISS(%d): %s" % (n, o))
    sys.exit(1)

# drop \nocite{*} now that every entry is cited
before = s.count(r"\nocite{*}")
s = s.replace("\\nocite{*}\n", "").replace("\\nocite{*}", "")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("ok: %d replacements, nocite removed=%d" % (len(REPL), before))
