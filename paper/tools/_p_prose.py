# -*- coding: utf-8 -*-
"""Stage 4: remove the code listings from the paper (Section 5.10 tail, Appendix A,
Appendix B) and replace them with prose and numbered equations.
Writes paper_CAFE_v5.md.
"""
import io, re, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v4.md"
DST = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v5.md"

SEC_510 = r"""**A one-command implementation.** The protocol costs one evaluation pass over a
frozen score file --- no model, no retraining, no GPU. For the base checkpoint,
with 2,000 bootstrap replicates and $q=5$, the report is: $w=0.6317$ and
$\omega_B=0.00388$, i.e.\ an exchange rate of 257.7; $\theta=0.8772$ with a
video-clustered interval $[0.8299,0.9154]$; $B=0.7558$ $[0.6791,0.8360]$;
$B\mid z=0.7165$ $[0.6454,0.7872]$; $D=0.0393$ with motion share
$\delta_{\mathrm{mot}}=0.139$; $E(B)=5.64$ and $E(B\mid z)=4.49$;
$\delta^{*}=0.512$, with $\tau\in[0.6948,0.9448]$ at 20\% contamination; and a
within-video contribution of 0.00294 to $\theta$ (0.33\%). The same evaluation
answers the reverse question, which is the one a localization claim implicitly
asks: it converts a headline improvement $\Delta\theta$ into the movement it
demands of $\Delta B=\Delta\theta/\omega_B$ and compares that with the remaining
headroom $1-B$ and with the split's own seed noise. On UCF-Crime a claim of +0.8
points of AUC demands $\Delta B=2.06$, larger than the 0.24 of headroom that
exists, so no scorer can support that claim on this split; the arithmetic follows
from $\omega_B=0.00388$ and from nothing else.
"""

APP_A = r"""## Appendix A: computing the census and the estimators {#sec:appA}

Everything in Section~\ref{sec:contrast} is arithmetic on the labelled scores of
one split. We state the recipe explicitly so that the census and the second
column can be reproduced from a score file.

*Census.* For an anomalous video $v$ let $a_v$ and $b_v$ denote its numbers of
anomalous and of normal units, $M$ the total number of anomalous units of the
split, $K$ the total number of units in normal videos, and $B_{\mathrm{tot}}$ the
total number of normal units inside anomalous videos. Then

\begin{equation}
n_A=MK,\qquad
n_B=\sum_{v\in\mathcal{V}_{\mathrm{an}}} a_v b_v,\qquad
n_C=\sum_{v\in\mathcal{V}_{\mathrm{an}}} a_v\,(B_{\mathrm{tot}}-b_v),
\end{equation}

\noindent and $\omega_Z=n_Z/(n_A+n_B+n_C)$ for $Z\in\{A,B,C\}$.

*Estimators.* With $U_v$ the Mann--Whitney count of video $v$ (one sort of its
scores, average ranks for ties),

\begin{equation}
\hat{B}_v=\frac{U_v}{a_v b_v},\qquad
\hat{B}=\frac{\sum_v a_v b_v\,\hat{B}_v}{\sum_v a_v b_v},
\end{equation}

\noindent the pair-weighted version reported throughout. The stratified variant
$\hat{B}\mid z$ replaces $U_v$ by the sum of the per-bin Mann--Whitney counts
inside video $v$, with the bins cut at the within-video quintiles of $z$. The
interval is a bootstrap over videos: resample the $V$ videos with replacement,
recompute $\hat{B}$ and $\hat{B}\mid z$ on the resampled set, and take the
2.5--97.5\% quantiles; the unit of replication is the video, not the pair. The
E-value is computed from $\hat{B}$ as in Section~\ref{sec:evalue}, and the
partial-identification endpoints as in Section~\ref{sec:bound}.

Two implementation notes that matter in practice. First, ties must be broken the
way the reported metric breaks them, i.e.\ by average ranks; ranked scores make
the whole estimate one sort per video, so a 290-video split takes about a second.
Second, the motion bins must be cut *inside each video*: bins computed on the
pooled split would compare a high-motion video's control units with a low-motion
video's anomalous units, which is exactly the confounded comparison that the
stratification is meant to remove.
"""

APP_B = r"""## Appendix B: artefacts {#sec:appB}

We release the reference implementation of the report of
Section~\ref{sec:protocol}, together with the per-frame scores of the eight
checkpoints used here --- the score vector, the per-video lengths, the
frame-level ground truth, and the motion statistic --- which is sufficient to
reproduce every number in Section~\ref{sec:exp}. The archived per-run metric
files used for the verification of Section~\ref{sec:host} are included as well.
Evaluation of one checkpoint takes about 16\,s on a single RTX A5000 (24\,GB);
the controlled probes of Section~\ref{sec:mediation} train in 30--70\,s per arm
on 4,000 cached clips.
"""


def replace_between(text, start_marker, end_marker, new):
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    return text[:i] + new + text[j:]


def main():
    t = io.open(SRC, encoding="utf-8").read()

    # 5.10: from the "one-command" paragraph up to "## 6. Experiments"
    t = replace_between(t, "**A one-command implementation.**",
                        "## 6. Experiments", SEC_510 + "\n")
    # Appendix A
    t = replace_between(t, "## Appendix A: computing", "## Appendix B: artefacts", APP_A + "\n")
    # Appendix B
    t = replace_between(t, "## Appendix B: artefacts", "## References", APP_B + "\n")

    t = t.replace('**No new method.** This paper does not propose a detector. We consider that a\nfeature rather than a gap: the measurement problem had to be settled before any\nmethod could be evaluated, and our own attempts to build the method first\n(Section~\\ref{sec:failures}) produced four negative results that this framework explains.', '**Measurement, not a new detector.** We do not propose a detector, and the\ntheorems explain why a detector-shaped contribution cannot settle the question\nthis paper asks: a scorer that is constant inside every video keeps 99.7\\% of\nthe headline number while localizing at chance, so a comparison of two detectors\non that number carries no information about the capability the field claims to\nmeasure. What we propose instead is an instrument: an estimand, three estimators\nwith their intervals and sensitivity analysis, a partial-identification bound,\nand a reporting protocol. Each of these is a method in the ordinary sense --- a\nprocedure that turns labelled scores into a claim that can be checked --- and each\nis released as reference code. The four negative results of\nSection~\\ref{sec:failures} are then not a gap in the paper but its evidence: they\nare what happens when method work is built on the unmeasured quantity.')
    t = t.replace("The second display is Theorem 1.",
                  "The second display is Eq.~\\eqref{eq:decomp}.")
    # two placeholders that live in the running text
    t = t.replace("A released script (\x00) computes all of it from a",
                  "A released script computes all of it from a")
    t = t.replace("we ran the published test path (\x00 with",
                  "we ran the published test path with")

    assert "```" not in t, "code fence left"
    assert "\x00" not in t, "NUL placeholder left"
    # one spelling for the stratified estimator everywhere it is prose
    t = re.sub(r"(\\hat\{B\}|B)\\mid z", r"\1|z", t)
    io.open(DST, "w", encoding="utf-8").write(t)
    print("wrote", DST, len(t), "bytes;", t.count("\\begin{equation}"), "equations;",
          t.count("\\begin{table}"), "tables")


if __name__ == "__main__":
    main()
