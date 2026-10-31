# -*- coding: utf-8 -*-
"""Stage 1: replace the markdown pipe tables of the CAFE paper by booktabs
table floats (caption above, centred), merge the two motion tables, and fix the
hard-coded table numbers in the running text.

The numbering becomes document order:
  1 census            (§3)
  2 partial ident.    (§5.3)
  3 decomposition UCF (§6.2)
  4 degeneracy        (§6.3)
  5 motion mediation  (§6.5, gamma scan + placebo merged)
  6 post-hoc resid.   (§6.5, base s789 with CIs)
  7 post-hoc resid.   (§6.5, res s234)
  8 XD decomposition  (§6.8)
"""
import io, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v1.md"
DST = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v2.md"

T1 = r"""\begin{table}[t]
\centering
\caption{Pair census of the two benchmarks, computed from the ground-truth
arrays alone.  The weights are properties of the label layout, not of any
model.}
\label{tab:census}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
split & videos & frames & positives & $\omega_A$ & $\omega_B$ & $\omega_C$ & $n_B/N$ & $1/\omega_B$ \\
\midrule
UCF-Crime test & 290 & 69,368 & 5,260 & 0.6317 & 0.00388 & 0.3645 & 0.388\% & 257.7 \\
XD-Violence test & 800 & 2,331,296 & 537,882 & 0.6985 & 0.00071 & 0.3008 & 0.071\% & 1,407.6 \\
\bottomrule
\end{tabular}
\end{table}"""

T2 = r"""\begin{table}[t]
\centering
\caption{Partial identification of the within-video contrast for every
checkpoint.  $\delta_{\mathrm{mot}} = (B-B\!\mid\! z)/(1-B\!\mid\! z)$ is the
motion-mediated share implied by the stratified estimator;
$\delta^{*} = 2B-1$ is the contamination share at which the localization claim
dies; $\tau_{\mathrm{LB}}(0.2)$ is the lower endpoint of the identified interval
under 20\% contamination.}
\label{tab:partial}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
checkpoint & $B$ & $B\!\mid\!z$ & $\delta_{\mathrm{mot}}$ & $\delta^{*}$ & $\tau_{\mathrm{LB}}(0.2)$ \\
\midrule
UCF base s234          & 0.7270 & 0.6729 & 0.165 & 0.454 & 0.6587 \\
UCF motion6 s789       & 0.7576 & 0.7230 & 0.125 & 0.515 & 0.6970 \\
UCF base s789          & 0.7558 & 0.7169 & 0.138 & 0.512 & 0.6948 \\
UCF cons0 s234         & 0.7235 & 0.6744 & 0.151 & 0.447 & 0.6544 \\
UCF attr s234          & 0.7080 & 0.6545 & 0.155 & 0.416 & 0.6351 \\
UCF core14txt s234     & 0.7169 & 0.6728 & 0.135 & 0.434 & 0.6462 \\
UCF motion6 s555       & 0.7180 & 0.6604 & 0.170 & 0.436 & 0.6476 \\
UCF base s555          & 0.7236 & 0.6688 & 0.165 & 0.447 & 0.6546 \\
UCF core8g1 s234       & 0.7124 & 0.6621 & 0.149 & 0.425 & 0.6406 \\
UCF res s234           & 0.7467 & 0.7093 & 0.128 & 0.493 & 0.6833 \\
UCF motion6-shuf s234  & 0.7171 & 0.6786 & 0.120 & 0.434 & 0.6463 \\
\midrule
XD base s234           & 0.8466 & 0.8223 & 0.137 & 0.693 & 0.8083 \\
XD base s555           & 0.8419 & 0.8160 & 0.141 & 0.684 & 0.8024 \\
XD base s789           & 0.8281 & 0.8000 & 0.140 & 0.656 & 0.7851 \\
XD motion3 s789        & 0.8046 & 0.7774 & 0.122 & 0.609 & 0.7557 \\
\bottomrule
\end{tabular}
\end{table}"""

T3 = r"""\begin{table}[t]
\centering
\caption{Exact three-block decomposition of frame-level micro-AUC on
UCF-Crime, for the eleven checkpoints that survived our re-evaluation.  $A$,
$B$, $C$ are the block statistics, $B\!\mid\!z$ the motion-stratified variant,
$corr$ the mean within-video correlation between score and motion energy.
Ranks are within the eleven rows.}
\label{tab:decomp-ucf}
\begin{tabular}{@{}rlrrrrrrrr@{}}
\toprule
\# & checkpoint & $\theta$ & $A$ & $B$ & $B\!\mid\!z$ & $C$ & $corr$ & rank $\theta$ & rank $B\!\mid\!z$ \\
\midrule
1  & base s234        & 0.8800 & 0.9751 & 0.7270 & 0.6729 & 0.7167 & 0.150 & 1  & 6  \\
2  & motion6 s789     & 0.8778 & 0.9728 & 0.7576 & 0.7230 & 0.7144 & 0.233 & 2  & 1  \\
3  & base s789        & 0.8772 & 0.9773 & 0.7558 & 0.7169 & 0.7050 & 0.141 & 3  & 2  \\
4  & cons0 s234       & 0.8759 & 0.9774 & 0.7235 & 0.6744 & 0.7016 & 0.145 & 4  & 5  \\
5  & attr s234        & 0.8724 & 0.9788 & 0.7080 & 0.6545 & 0.6899 & 0.172 & 5  & 11 \\
6  & core14txt s234   & 0.8717 & 0.9725 & 0.7169 & 0.6728 & 0.6989 & 0.151 & 6  & 7  \\
7  & motion6 s555     & 0.8717 & 0.9728 & 0.7180 & 0.6604 & 0.6981 & 0.288 & 7  & 10 \\
8  & base s555        & 0.8705 & 0.9752 & 0.7236 & 0.6688 & 0.6907 & 0.139 & 8  & 8  \\
9  & core8g1 s234     & 0.8684 & 0.9759 & 0.7124 & 0.6621 & 0.6837 & 0.126 & 9  & 9  \\
10 & res s234         & 0.8682 & 0.9767 & 0.7467 & 0.7093 & 0.6814 & 0.328 & 10 & 3  \\
11 & motion6-shuf s234 & 0.8616 & 0.9670 & 0.7171 & 0.6786 & 0.6806 & 0.121 & 11 & 4  \\
\midrule
   & spread           & 0.0183 & 0.0118 & 0.0495 & 0.0685 & 0.0383 & --    & -- & -- \\
\bottomrule
\end{tabular}
\end{table}"""

T4 = r"""\begin{table}[t]
\centering
\caption{Constant-within-video scorers against frame-level scorers on the same
cached CLIP features and training protocol (UCF-Crime, linear heads, top-$k$ MIL
for the frame models).}
\label{tab:degeneracy}
\small
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{6.0cm}rrrr@{}}
\toprule
scorer & $\theta$ & $A$ & $B$ & within-video AUC \\
\midrule
video-level classifier (one score per video)   & 0.8234 & 0.9766 & 0.5000 & 0.5000 \\
per-video maximum of frame scores             & 0.7959 & --     & 0.5000 & 0.5000 \\
frame-level MIL (same features, protocol)     & 0.8257 & 0.9571 & 0.6571 & 0.6571 \\
DSANet base s789 (published)                  & 0.8772 & 0.9773 & 0.7558 & 0.7046 \\
\bottomrule
\end{tabular}
\end{table}"""

T5 = r"""\begin{table}[t]
\centering
\caption{The three interventions on the nuisance.  Top: an orthogonality
penalty against the true motion statistic, three seeds per row, reported as
their mean.  Bottom: the same penalty at $\gamma=1$ against real versus
shuffled motion.  The penalty does decorrelate the score from motion, and it
lowers the localization estimate rather than raising it.}
\label{tab:interventions}
\begin{tabular}{@{}lrrrr@{}}
\toprule
condition & $\theta$ & $B$ & $B\!\mid\!z$ & $corr$ \\
\midrule
\multicolumn{5}{@{}l}{\emph{training-time orthogonalisation (mean of three seeds)}} \\
$\gamma=0$ (baseline) & 0.8352 & 0.6805 & 0.6429 & 0.163 \\
$\gamma=0.1$          & 0.8331 & 0.6772 & 0.6417 & 0.154 \\
$\gamma=0.3$          & 0.8300 & 0.6744 & 0.6426 & 0.144 \\
$\gamma=1.0$          & 0.7980 & 0.6236 & 0.6043 & 0.089 \\
\midrule
\multicolumn{5}{@{}l}{\emph{placebo: $\gamma=1$ against shuffled motion}} \\
baseline ($\gamma=0$)   & 0.8352 & 0.6805 & 0.6429 & 0.163 \\
$\gamma=1$, real motion & 0.7980 & 0.6236 & 0.6043 & 0.096 \\
$\gamma=1$, shuffled     & 0.8362 & 0.6891 & 0.6480 & 0.169 \\
\bottomrule
\end{tabular}
\end{table}"""

T6 = r"""\begin{table}[t]
\centering
\small
\caption{Post-hoc residualisation on two checkpoints. The score is regressed on
four nuisance coordinates, the fit is removed with video-parity cross-fitting,
and the report is recomputed. Every variant lowers both localization measures on
both checkpoints. Video-cluster bootstrap intervals (500 replicates) for the raw
rows are $B\in[0.684,0.836]$ and $B|z\in[0.647,0.791]$ on base s789, and
$[0.681,0.819]$, $[0.656,0.774]$ on res s234.}
\label{tab:resid}
\begin{tabular}{@{}lrrrr@{}}
\toprule
variant & $\theta$ & $B$ & $B\!\mid\!z$ & $E$ \\
\midrule
\multicolumn{5}{@{}l}{\emph{base s789 (score--motion correlation 0.141)}} \\
raw                          & 0.8772 & 0.7558 & 0.7166 & 4.50 \\
residualised, all frames     & 0.8600 & 0.7356 & 0.7099 & 4.33 \\
residualised, control frames & 0.8668 & 0.7392 & 0.7026 & 4.16 \\
\midrule
\multicolumn{5}{@{}l}{\emph{res s234 (score--motion correlation 0.328)}} \\
raw                          & 0.8682 & 0.7467 & 0.7089 & 4.31 \\
residualised, all frames     & 0.8493 & 0.7028 & 0.7001 & -- \\
residualised, control frames & 0.8532 & 0.7031 & 0.6915 & -- \\
\bottomrule
\end{tabular}
\end{table}"""

T8 = r"""\begin{table}[t]
\centering
\small
\caption{The decomposition on XD-Violence (800 videos, 2,331,296 scored
frames, 537,882 positive frames). The pooled number agrees with the
repository's own frame-level AUC to six decimals.}
\label{tab:decomp-xd}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
checkpoint & $\theta$ & $A$ & $B$ & $B\!\mid\!z$ & $C$ & $corr$ & rank $\theta$ & rank $B\!\mid\!z$ \\
\midrule
base s234    & 0.9536 & 0.9966 & 0.8466 & 0.8223 & 0.8540 & 0.192 & 1 & 1 \\
base s555    & 0.9437 & 0.9969 & 0.8419 & 0.8160 & 0.8204 & 0.203 & 2 & 2 \\
base s789    & 0.9449 & 0.9962 & 0.8281 & 0.8000 & 0.8258 & 0.200 & 3 & 3 \\
motion3 s789 & 0.9368 & 0.9920 & 0.8046 & 0.7774 & 0.8090 & 0.201 & 4 & 4 \\
\bottomrule
\end{tabular}
\end{table}"""


# (start-marker line, end-marker line, replacement tex)
REPL = [
    ("| split | videos | frames | positives |",
     "*Table 1. Pair census of the two benchmarks, computed from the ground-truth\narrays alone.*",
     T1),
    ("| checkpoint | B | B\\|z |",
     "a 20% contamination.*",
     T2),
    ("| # | checkpoint |",
     "motion energy. Ranks are within the eleven rows.*",
     T3),
    ("| scorer | θ |",
     None,
     T4),
    ("| γ | θ |", None, T5),          # covers the next table (placebo) as well
    ("| variant | θ | B [95% CI] |", None, T6),
    ("| checkpoint | θ | A | B | B\\|z | C | corr |", None, T8),
]

if __name__ == "__main__":
    text = io.open(SRC, encoding="utf-8").read()
    L = text.split("\n")

    def find(pred, start=0):
        for k in range(start, len(L)):
            if pred(L[k]):
                return k
        return -1

    for marker, endmark, tex in REPL:
        i = find(lambda l: l.startswith(marker))
        assert i >= 0, marker
        # table block
        j = i
        while j < len(L) and L[j].startswith("|"):
            j += 1
        # for the merged motion table, swallow the intervening caption/paragraph
        # up to the second table, and the placebo table itself.
        if marker in ("| γ | θ |", "| variant | θ | B [95% CI] |"):
            k = j
            head = "| condition |" if marker == "| γ | θ |" else "| variant | θ | B |"
            while not L[k].startswith(head):
                k += 1
            while k < len(L) and L[k].startswith("|"):
                k += 1
            j = k
        new = [tex]
        L[i:j] = new
        text = "\n".join(L)

    # delete the italic caption paragraphs that are now inside \caption
    caps = [
        "*Table 1. Pair census of the two benchmarks, computed from the ground-truth\narrays alone.*",
        "*Table 2. Exact three-block decomposition of frame-level micro-AUC on UCF-Crime",
        "*Table 3. Constant-within-video scorers against frame-level scorers, on the same",
        "*Table 4. Partial identification of the within-video contrast per checkpoint",
        "*Table 5. The decomposition on XD-Violence (800 videos, 2,331,296 scored",
    ]
    for c in caps:
        if c in text:
            i = text.index(c)
            # remove through the closing '*' of the caption paragraph
            j = text.index("\n\n", i)
            text = text[:i] + text[j + 2:]

    # fix hard-coded table numbers in the running text
    pairs = [
        ("Table 4 collects the numbers.", "Table~\\ref{tab:partial} collects the numbers."),
        ("`B` and `B|z` from Table 2 and Table 5;",
         "`B` and `B|z` from Tables~\\ref{tab:decomp-ucf} and~\\ref{tab:decomp-xd};"),
        ("Four facts about Table 2 carry the paper.",
         "Four facts about Table~\\ref{tab:decomp-ucf} carry the paper."),
        ("(Table 2, column B|z)", "(Table~\\ref{tab:decomp-ucf}, column $B\\!\\mid\\!z$)"),
        ("(mean 0.18 across eleven; Table 2)",
         "(mean 0.18 across eleven; Table~\\ref{tab:decomp-ucf})"),
        ("a third of the full 0.07 spread of Table 2.",
         "a third of the full 0.07 spread of Table~\\ref{tab:decomp-ucf}."),
        ("Proposition 5 and Tables 2–5 give", "Proposition 5 and Tables~\\ref{tab:decomp-ucf}--\\ref{tab:decomp-xd} give"),
        ("(res s234 in Table 2 does", "(res s234 in Table~\\ref{tab:decomp-ucf} does"),
        ("reproduce Table 1 and the second column",
         "reproduce Table~\\ref{tab:census} and the second column"),
        ("Table 1 lists the census", "Table~\\ref{tab:census} lists the census"),
    ]
    for a, b in pairs:
        if a in text:
            text = text.replace(a, b)
        else:
            print("WARN not found:", a[:50])

    io.open(DST, "w", encoding="utf-8").write(text)
    print("wrote", DST, len(text), "bytes;",
          text.count("\\begin{table}"), "table floats;",
          len([l for l in text.split("\n") if l.startswith("|")]), "pipe rows left")
