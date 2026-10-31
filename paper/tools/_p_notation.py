# -*- coding: utf-8 -*-
"""Stage 5: notation audit.  Every symbol the paper uses is either defined on
first use or listed in the Notation paragraph; overloaded symbols are renamed so
that one letter keeps one meaning.

Reads  paper_CAFE_v5.md  ->  writes  paper_CAFE_v6.md
"""
import io, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v5.md"
DST = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v6.md"

txt = io.open(SRC, encoding="utf-8").read()
applied = []


def rep(old, new, count=1, why=""):
    global txt
    n = txt.count(old)
    assert n == count, "expected %d occurrence(s) of %r, found %d (%s)" % (
        count, old[:60], n, why)
    txt = txt.replace(old, new)
    applied.append(why)


# ---------------------------------------------------------------- 1. notation
NOTATION = r"""**Notation.** We use one letter for one thing. $s$ is the scorer, $s_i=s(x_i)$
the score of unit $i$, and $S_i$ the corresponding random variable; $y_i\in\{0,1\}$
is the ground-truth label of unit $i$ and $y_v$ the label vector of video $v$;
$a_v$ and $b_v$ count the anomalous and the background units of an anomalous video
$v$. $\theta$ is the reported statistic of Eq.~\eqref{eq:theta}; $A$, $B$, $C$ are
the block statistics of Theorem~1 and $\mathcal{A}$, $\mathcal{B}$, $\mathcal{C}$
the corresponding pair sets. $B$ is also our name for the localization estimand
(they coincide by construction), $B|z$ is its motion-stratified version
(Section~\ref{sec:strat}), $D=B-B|z$, and a prefix $\Delta$ always means the
difference between two scorers on one split ($\Delta\theta=\theta-\theta'$ and so
on). Intervals are 95\% video-clustered bootstrap confidence intervals (CI), $E$
is the E-value of Section~\ref{sec:evalue}, $\rho_s$ is Spearman's rank
correlation across a family of checkpoints together with its two-sided $p$-value,
and $\delta$ is the share of contaminated pair mass of Section~\ref{sec:bound}.

"""

anchor = "## 3. Setup and notation {#sec:setup}\n\n"
rep(anchor, anchor + NOTATION, 1, "Notation paragraph added at the head of Section 3")

# ------------------------------------------------------- 2. label of a unit
rep("""**Data.** A test split is a collection of $V$ videos; video $v$ contributes
$s_v$ scored units (snippets) with binary ground-truth labels
$y_v\\in\\{0,1\\}^{s_v}$.""",
    """**Data.** A test split is a collection of $V$ videos; video $v$ contributes
$n_v$ scored units (snippets), each unit carrying a binary label
$y_i\\in\\{0,1\\}$, and $y_v\\in\\{0,1\\}^{n_v}$ collects the labels of one video.""",
    1, "s_v (video length) renamed n_v; y_i defined")

# the equation number used by the Notation paragraph
rep("""\\begin{equation}
\\theta=\\frac{1}{N}\\sum_{(i,j)\\in\\mathcal{P}}\\mathbb{1}[s_i>s_j],
\\end{equation}""",
    """\\begin{equation}
\\theta=\\frac{1}{N}\\sum_{(i,j)\\in\\mathcal{P}}\\mathbb{1}[s_i>s_j],
\\label{eq:theta}
\\end{equation}""",
    1, "eq:theta label added")

# ------------------------------------------------- 3. A_i -> y_i in Section 5
rep("""$b_v$ background units. For a unit $i$ let $A_i\\in\\{0,1\\}$ be its (annotated)
status and $S_i=s(x_i)$ its score.""",
    """$b_v$ background units. For a unit $i$ let $y_i\\in\\{0,1\\}$ be its (annotated)
status and $S_i=s(x_i)$ its score.""",
    1, "A_i -> y_i (status of a unit, Section 5.1)")

rep("""\\tau_v=P(S_i>S_j\\mid A_i=1,A_j=0,\\ i,j\\in v),""",
    """\\tau_v=P(S_i>S_j\\mid y_i=1,y_j=0,\\ i,j\\in v),""",
    1, "A_i -> y_i inside the tau_v equation")

rep("""A_i \\perp \\mathrm{rank}(S_i)\\mid Z_i,v.""",
    """y_i \\perp \\mathrm{rank}(S_i)\\mid Z_i,v.""",
    1, "A_i -> y_i inside assumption A2")

# --------------------------------------------- 4. q (bins) -> Q (contamination)
rep("standardised inside the video) define $q$-quantile bins",
    "standardised inside the video) define $Q$-quantile bins",
    1, "quantile count renamed Q (q is the contamination share)")
rep("average with pair weights. With $q=5$ this is a matched comparison: the",
    "average with pair weights. With $Q=5$ this is a matched comparison: the",
    1, "Q=5 at first use")
rep("short videos; $q=10$ lowers $B|z$ by 0.7 point on the base checkpoint and does",
    "short videos; $Q=10$ lowers $B|z$ by 0.7 point on the base checkpoint and does",
    1, "Q=10")
rep("with 2,000 bootstrap replicates and $q=5$, the report is:",
    "with 2,000 bootstrap replicates and $Q=5$, the report is:",
    1, "Q=5 in the one-command report")

# --------------------------------------------------- 5. top-k, k and margin m
rep("""auxiliary head to make the top-k-minus-mean contrast $T_v=\\mathrm{top}_k(s)-\\mathrm{mean}(s)$
positive on anomalous videos --- fails for a structural reason visible in one
line: $T_v\\ge 0$ always, so the constraint it imposes on normal videos
($y=0\\Rightarrow T\\le m$) is unsatisfiable,""",
    """auxiliary head to make the top-$k$-minus-mean contrast
$T_v=\\mathrm{top}_k(s)-\\mathrm{mean}(s)$ --- where $\\mathrm{top}_k$ is the mean of
the $k$ largest scores of a video and $\\mathrm{mean}$ the mean of all of them ---
positive on anomalous videos fails for a structural reason visible in one line:
$T_v\\ge 0$ always, so the constraint it imposes on normal videos
($y_v=0\\Rightarrow T_v\\le m$, for a margin $m$) is unsatisfiable,""",
    1, "k and the margin m defined; T_v notation")

# ------------------------------------------------- 6. the trained probe's B
rep("it accounts for $(0.6552-0.5)/(B_{\\mathrm{model}}-0.5)=86\\%$ of",
    "it accounts for $(0.6552-0.5)/(0.6803-0.5)=86\\%$ of",
    1, "undefined B_model replaced by the measured value")

# --------------------------------------------------- 7. N_pos and N_neg
rep("""The weights are computable before a single model is trained. With $n_v$ units in
video $v$, $\\omega_B\\approx\\sum_v a_v b_v/(N_{\\mathrm{pos}}N_{\\mathrm{neg}})$, so a split measures localization""",
    """The weights are computable before a single model is trained. With $n_v$ units in
video $v$ and $N_{\\mathrm{pos}}$, $N_{\\mathrm{neg}}$ the numbers of positive and of
negative units in the split,
$\\omega_B\\approx\\sum_v a_v b_v/(N_{\\mathrm{pos}}N_{\\mathrm{neg}})$, so a split measures localization""",
    1, "N_pos and N_neg defined")

# ------------------------------------------------ 8. mangled exchange-rate line
rep("2. **Report the exchange rate next to every $\\Delta A$UC.**",
    "2. **Report the exchange rate next to every gain in $\\theta$.**",
    1, "mangled '$\\Delta A$UC' repaired")

# ------------------------------------------- 9. V_an defined in Appendix A
rep("""\\noindent and $\\omega_Z=n_Z/(n_A+n_B+n_C)$ for $Z\\in\\{A,B,C\\}$.""",
    """\\noindent and $\\omega_Z=n_Z/(n_A+n_B+n_C)$ for $Z\\in\\{A,B,C\\}$, where
$\\mathcal{V}_{\\mathrm{an}}$ is the set of anomalous videos of the split.""",
    1, "V_an defined in Appendix A")

# ------------------------------------------------ 10. one symbol for the loss
rep("Let the training loss $L$ depend on the frame scores",
    "Let the training loss $\\mathcal{L}$ depend on the frame scores",
    1, "loss written \\mathcal{L}, as in the display that follows")

# ------------------------------------- 11. corr and B|z typeset consistently
txt = txt.replace("$B\\!\\mid\\!z$", "$B|z$")
txt = txt.replace("B\\!\\mid\\! z", "B|z")
applied.append("tight $B\\!\\mid\\!z$ normalised to $B|z$ in tables")
txt = txt.replace("$corr$", "$\\mathrm{corr}$")
applied.append("$corr$ -> $\\mathrm{corr}$ in captions and headers")

# --------------------------- 12. name the four nuisance coordinates explicitly
rep("""*Post-hoc residualisation on real checkpoints.* Cross-fitted subtraction of a
ridge fit of the score on $[t,\\ \\lVert\\Delta x\\rVert,\\ \\overline{\\lVert\\Delta x\\rVert},\\ \\lVert x\\rVert]$, folds assigned by
video parity,""",
    """*Post-hoc residualisation on real checkpoints.* Cross-fitted subtraction of a
ridge fit of the score on four nuisance coordinates --- the normalised time
position $t/n_v$, the instantaneous motion $\\lVert\\Delta x_t\\rVert$, its 5-frame
moving average $\\overline{\\lVert\\Delta x_t\\rVert}$ and the feature norm
$\\lVert x_t\\rVert$ --- with folds assigned by video parity,""",
    1, "the four residualisation covariates named")

# ------------------- 13. tildes that pandoc would ship as a printed tilde glyph
rep("the block statistics of Theorem~1 and",
    "the block statistics of Theorem 1 and",
    1, "Theorem~1 -> Theorem 1 (pandoc escaped the tie)")
rep("$\\hat{B}$ averages over ~10^6 pairs but only 140 anomalous videos",
    "$\\hat{B}$ averages over about $10^6$ pairs but only 140 anomalous videos",
    1, "approximate '~10^6' written out")
rep("A $B$ difference below ~0.03 between two systems on",
    "A $B$ difference below about 0.03 between two systems on",
    1, "approximate '~0.03' written out")
rep("localization claim below ~0.05 on UCF-Crime is not established",
    "localization claim below about 0.05 on UCF-Crime is not established",
    1, "approximate '~0.05' written out")
rep("5. Motion accounts for ~5 points of the ~25-point within-video excess, and",
    "5. Motion accounts for roughly 5 of the roughly 25 points of within-video excess, and",
    1, "approximate '~5' / '~25-point' written out")

# ------------------------------------- 14. last wording / notation nits
rep("training objective of a MIL model, where the top-k/max pool makes the loss",
    "training objective of a MIL model, where the top-$k$/max pool makes the loss",
    1, "top-$k$ typeset as math in assumption A1")
rep("Fitting $s\\sim f(z)$ on control units ($s$ below the video median) of one half of",
    "Fitting the score on $z$ by ridge regression over the control units (those scored\nbelow the video median) of one half of",
    1, "residualisation fit described in words, no overloading of \\sim")
rep("of scenes --- is measured by A and C.",
    "of scenes --- is measured by $A$ and $C$.",
    1, "block statistics A and C typeset as math")
rep("$\\gamma\\,\\mathrm{mean}_v\\mathrm{corr}^2(S_v,Z_v)$ to the loss, which shrinks the within-video",
    "the penalty $\\gamma\\,\\mathrm{mean}_v\\mathrm{corr}^2(S_v,Z_v)$, with weight $\\gamma$, to the\nloss, which shrinks the within-video",
    1, "penalty weight gamma introduced explicitly")

rep("would need $\\sigma(\\Delta\\theta)\\le 1.9\\cdot10^{-5}$, i.e. seed-averaging 247 times finer than the",
    "would need $\\sigma(\\Delta\\theta)\\le 1.9\\cdot10^{-5}$, i.e. seed-averaging 253 times finer than the",
    1, "253 = 0.0049 / 1.94e-5")

rep("unregularised level (0.689 vs 0.680 baseline for $B$, 0.648 vs 0.643 for",
    "unregularised level (0.689 against a baseline of 0.680 for $B$, 0.648 against 0.643 for",
    1, "unbalanced parenthesis around the placebo comparison")

# ------------------- 16. remaining indented-math lines -> prose or numbered equation
rep("""$B|z$ is a diagnostic of how much of the localization signal is motion-mediated,

  $D = B - B|z$  (3.5--5.8 points, median 4.9, on eleven checkpoints).""",
    """$B|z$ is a diagnostic of how much of the localization signal is motion-mediated.
The size of that adjustment is the gap $D=B-B|z$, which is 3.5--5.8 points
(median 4.9) across the eleven UCF-Crime checkpoints.""",
    1, "D = B - B|z line turned into prose")

rep("""  $\\tau_{\\mathrm{LB}}(\\delta)=\\frac{B-\\delta}{1-\\delta}\\le\\tau\\le\\min\\{1,\\tfrac{B}{1-\\delta}\\}=\\tau_{\\mathrm{UB}}(\\delta)$,  $0\\le\\delta<1$.""",
    """\\begin{equation}
\\tau_{\\mathrm{LB}}(\\delta)=\\frac{B-\\delta}{1-\\delta}\\;\\le\\;\\tau\\;\\le\\;\\min\\Big\\{1,\\frac{B}{1-\\delta}\\Big\\}=\\tau_{\\mathrm{UB}}(\\delta),
\\qquad 0\\le\\delta<1.
\\label{eq:tau-bound}
\\end{equation}""",
    1, "partial-identification bound promoted to a numbered equation")

rep("""For every reported system, publish the four numbers

  $(\\theta,B,B|z)$ with a video-clustered 95% interval on the latter two, and $E$.""",
    """For every reported system, publish the four numbers $\\theta$, $B$, $B|z$ and $E$,
the last three with a video-clustered 95\\% interval.""",
    1, "protocol list turned into prose")
# sample sd (ddof=1) of the three base runs: theta 0.0049 (was 0.0048), B 0.0177,
# B|z 0.0267 (was 0.0245); every derived figure is updated consistently.
rep("""Three base checkpoints that differ only in seed give

  $\\theta\\in\\{0.8800,0.8705,0.8772\\}$, sd 0.0048;
  $B\\in\\{0.7270,0.7236,0.7558\\}$, sd 0.0177;
  $B|z\\in\\{0.6729,0.6688,0.7169\\}$, sd 0.0245.

Two consequences, both quantitative.""",
    """Three base checkpoints that differ only in seed --- rows 1, 3 and 8 of
Table~\\ref{tab:decomp-ucf} --- give $\\theta=0.876\\pm0.005$, $B=0.736\\pm0.018$ and
$B|z=0.686\\pm0.027$ (mean $\\pm$ one standard deviation over the three runs).
Localization is therefore three to five times as noisy as the headline number,
and that noise is invisible in it.

Two consequences, both quantitative.""",
    1, "seed-noise paragraph rewritten as prose with mean +- sd, sd values recomputed")

rep("by Theorem 3 the standard error of a $\\Delta B$ imputed from a single-seed $\\Delta\\theta$\nis $0.0048/0.00388=1.24$",
    "by Theorem 3 the standard error of a $\\Delta B$ imputed from a single-seed $\\Delta\\theta$\nis $0.0049/0.00388=1.26$",
    1, "SE(Delta B) recomputed with sd(theta)=0.0049")

rep("is 0.0023 of $\\theta$ --- two-thirds of one seed\nstandard deviation",
    "is 0.0023 of $\\theta$ --- half of one seed\nstandard deviation",
    1, "0.0023/sd = 0.47, not two-thirds")

rep("i.e. seed-averaging 247\ntimes finer than the observed run-to-run noise",
    "i.e. seed-averaging 253\ntimes finer than the observed run-to-run noise",
    1, "253 = 0.0049 / 1.94e-5")

txt = txt.replace("$0.0048$", "$0.0049$")
txt = txt.replace("is 0.0048,", "is 0.0049,")
txt = txt.replace("noise is 0.0048.", "noise is 0.0049.")
txt = txt.replace("noise of 0.0048;", "noise of 0.0049;")
txt = txt.replace("$\\sigma(\\theta)=0.0048$", "$\\sigma(\\theta)=0.0049$")
applied.append("sd(theta) 0.0048 -> 0.0049 wherever it is quoted")

txt = txt.replace("$\\Delta B=1.24$", "$\\Delta B=1.26$")
txt = txt.replace("\\mathrm{SE}(\\Delta\\hat{B})=1.24", "\\mathrm{SE}(\\Delta\\hat{B})=1.26")
applied.append("1.24 -> 1.26 (SE(Delta B) with the corrected sd)")

rep("""*Between-seed noise.* $\\mathrm{sd}(B|z)=0.0245$ across three seeds of the same
configuration, i.e. a third of the full 0.07 spread of Table~\\ref{tab:decomp-ucf}.""",
    """*Between-seed noise.* $\\mathrm{sd}(B|z)=0.0267$ across three seeds of the same
configuration, i.e. two fifths of the full 0.07 spread of Table~\\ref{tab:decomp-ucf}.""",
    1, "sd(B|z) recomputed 0.0245 -> 0.0267, share 'a third' -> 'two fifths'")

io.open(DST, "w", encoding="utf-8").write(txt)

print("wrote", DST, len(txt), "bytes")
print("applied %d edits:" % len(applied))
for a in applied:
    print("  -", a)

# ---------------------------------------------------------------- self-checks
bad = []
for tok in ["$s_v$", "$A_i$", "$q=5$", "$q=10$", "$corr$", "B\\!\\mid\\!z",
            "$\\\\Delta A$UC", "B_{\\\\mathrm{model}}"]:
    if tok in txt:
        bad.append(tok)
print("leftover undefined/clashing tokens:", bad if bad else "none")
