# What Does One AUC Point Buy?

## Localization Ability in Weakly-Supervised Video Anomaly Detection

*(methodology paper)*

---

### Abstract





Frame-level AUC and AP are the currency of weakly-supervised video anomaly
detection: leaderboards decide which model "localizes anomalies better". We
show that this currency cannot buy what it is spent on. Written as a
probability over frame pairs, the reported AUC decomposes exactly into three
comparison sets --- pairs across videos, pairs inside one anomalous video, and
pairs spanning two anomalous videos --- with weights fixed by the test split
alone. On UCF-Crime fewer than 0.4% of pairs compare frames of the same video
(0.07% on XD-Violence), so **one point of within-video localization buys
0.0039 points of reported AUC --- equivalently, one reported point costs 258
(UCF-Crime) or 1,408 (XD-Violence) points of localization**. Verified on
sixteen checkpoints, three consequences follow: a scorer constant inside each
video attains 93.9% of a published state-of-the-art AUC while localizing at
chance; the entire attainable localization range moves the reported number by
0.0019, four tenths of the seed noise; and the multiple-instance objective
leaves within-video ordering a free parameter, so weakly-supervised
post-processing cannot be certified to improve it. We release an estimator
suite and a three-column protocol ($\theta$, $B$, $B|z$) with video-clustered
intervals, and report a criterion-validity check against interest: detection
mAP tracks the pooled AUC ($\rho_s=0.71$), not the localization column
($0.15$). We therefore offer $B$ as a diagnostic, not a replacement headline
number, and flag its external validation as open.

---

## 1. Introduction

Weakly-supervised video anomaly detection trains on videos labelled only
*normal* or *anomalous* and is evaluated by how well the produced frame scores
separate anomalous frames from normal ones. The community reports a single
number --- frame-level micro-AUC (or AP) pooled over the whole test split --- and
publishes gains in the third decimal place as evidence of progress (UCF-Crime
has moved from 75.4% to 89.4% over five years; XD-Violence AP from 73.6% to
86.9%).

Two recent audits have begun to question this practice. Song and Lee
(arXiv:2608.21854) observe that within-video frame pairs are a vanishing
fraction of all pairs and that a video-level classifier --- which produces a
constant score inside each video --- attains 81--97% of the reported micro-AUC
while its within-video AUC is exactly 50%. Abdulaziz and Bondarev
(arXiv:2608.11985) show that the pooled statistic does not predict frame-level
localization and that the resolution of the benchmark is too coarse for the
claimed margins. Both are *empirical diagnoses*: they measure that something is
wrong.

This paper does something different: it treats the reported statistic as an
**estimand**, derives what it is an estimate *of*, and then asks what can and
cannot be learned about localization from it. Our contributions are four.

**(1) An exact decomposition of the reported metric, with an exchange rate.**
Pooled AUC is a probability over ordered frame pairs, and partitioning those
pairs by whether the two frames come from the same video writes it as the
weighted average $\theta=\omega_A A+\omega_B B+\omega_C C$ of three block
statistics, with weights that are pair-count ratios --- properties of the test
split, identical for every method ever evaluated on it ($\omega_B=0.0039$ on
UCF-Crime, 0.00071 on XD-Violence). The identity is exact --- machine precision
on all sixteen checkpoints we examined --- and differentiating it gives the
exchange rate: **one point of within-video localization buys 0.0039 points of
reported AUC on UCF-Crime, so one reported point costs 258 (UCF-Crime) or 1,408
(XD-Violence) points of localization**. Since the full attainable localization
range is worth 0.0019 of $\theta$ while the seed-to-seed noise is 0.0049, **the
reported metric cannot resolve localization ability at the precision this
literature reports** (Section~\ref{sec:results}).

**(2) Degeneracy is the optimum of the estimand, and weak labels leave
localization unidentified.** A scorer constant within each video attains
$B\equiv 0.5$ and yet 93.9% of the published AUC of a state-of-the-art
detector; the identity explains why, since such a scorer still wins all of A
and most of C, so the leaderboard's ordering is dominated by scene-level
separability --- a *different* capability from localization. On top of this, a
multiple-instance learning (MIL) objective enters the loss only through a
symmetric function of each anomalous video's frame scores, so the within-video
ordering --- hence $B$ --- is a free parameter of the training problem. This is
a statement about the objective, not about optimization, and it predicts,
correctly, that every post-processing method that uses only the weak labels
fails to move $B$ --- confirmed on thirteen companion variants, a targeted
counter-example, and our own attempts (Section~\ref{sec:thm4}).

**(3) A causal reading of the within-video contrast, with a
partial-identification fallback.** The comparisons that define $B$ hold the
video fixed --- scene, camera, illumination are the video's own control --- so
the estimand is a within-unit contrast whose identification requires that,
inside an anomalous video, frame status be ignorable given observed covariates.
The obvious covariate is motion energy, and adjustment alone cannot resolve
whether it is confounder or mediator, since both hypotheses predict a lower
estimate; stratifying, orthogonalizing, and residualizing all *decrease* $B$
while a shuffled-motion placebo does not --- a pattern consistent with
mediation, on which basis the honest estimand is the unadjusted contrast
(Section~\ref{sec:contrast}). Because ignorability cannot be verified without
re-annotation, we also replace point identification by a bound: under an
unknown nuisance share $\delta$ of the pair mass, the reported $B$ certifies
above-chance localization up to $\delta^{*}=2B-1$.

**(4) Estimators, a reporting protocol, and a criterion-validity check
reported against interest.** We give the pair-weighted estimator of $B$, its
motion-stratified variant $B|z$, video-clustered bootstrap intervals (the unit
of replication is the video, not the pair), and an E-value for unmeasured
confounding; a released script computes the full report
$(\theta,B,B|z,\mathrm{CI},E)$ from a score file in seconds. Re-ranking the
eleven UCF-Crime checkpoints by $B|z$ moves five of them by three or more
places, though the family's own spread is not resolvable as a ranking
(Section~\ref{sec:audit}). Finally, detection mAP --- recomputed for twelve
checkpoints under two independent implementations --- tracks $\theta$
($\rho_s=0.71$) and not $B$ ($\rho_s=0.15$): the most obvious validation of our
own proposal fails, and the paper says so. We therefore offer $B$ as a
diagnostic, not a replacement headline number, with its external validation
flagged as open (Section~\ref{sec:criterion}). Figure 1 summarises the three
facts that drive everything below: how the scored frame pairs distribute over
the three blocks, how little a gain in localization is worth in the reported
number, and what happens to a leaderboard when the stratified column is taken
seriously.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig1.pdf}
\caption{What one reported AUC point is made of. (a) The scored frame pairs fall
into three blocks: $\mathcal{A}$ (a frame from an anomalous video against a
frame from a normal video; 63.2\% of pairs on UCF-Crime, 69.9\% on XD-Violence),
$\mathcal{B}$ (both frames from the same anomalous video; 0.388\% and 0.071\%),
and $\mathcal{C}$ (two different anomalous videos). Only $\mathcal{B}$ can
measure localization. (b) Because the reported statistic is a weighted average of
the block statistics, a gain $\Delta B$ in localization is worth only
$\omega_B\,\Delta B$ of reported AUC: $+0.01$ AUC would require $\Delta B=2.6$ on
UCF-Crime, and the seed-to-seed noise of the reported metric ($0.0049$) is
matched only at $\Delta B=1.26$. (c) Sorting the eleven UCF-Crime checkpoints by
the stratified column $B|z$ instead of $\theta$ moves five of them by at least three
places: res s234 climbs from 10 to 3, and the AUC winner
base s234 falls from 1 to 6.}

\label{fig:overview}
\end{figure}

The rest of the paper is organized as follows. Section~\ref{sec:related} reviews related work and Section~\ref{sec:setup}
fixes notation. Section~\ref{sec:results} states the results: the decomposition, the exchange rate, the
degeneracy, the resolution bound, and the non-identifiability theorem. Section~\ref{sec:contrast} turns
the within-video contrast into an estimand --- causal identification, the
confounder-versus-mediator test, the partial-identification bound --- and gives
the estimators with their inference and the reporting protocol. Section~\ref{sec:exp} reports
experiments on UCF-Crime and XD-Violence, Section~\ref{sec:discussion} discusses what the result implies
and where it stops, and Section~\ref{sec:conclusion} concludes.

## 2. Related work {#sec:related}

**Weakly-supervised VAD.** The deep era began with learned temporal regularity
(Hasan et al., CVPR 2016); MIL over snippet scores with a sparsity term
(Sultani et al., CVPR 2018) and its top-k variant (Tian et al., RTFM, ICCV 2021)
remain the backbone of the field. Successive refinements sharpen the objective
or the features: multiple synthetic outliers as self-training supervision
(Li et al., MSL, AAAI 2022), self-supervised sparse representation
(Wu et al., S3R, ECCV 2022), magnitude-contrastive glance-and-focus learning
(Chen et al., MGFN, AAAI 2023), uncertainty-regulated dual memory units
(Zhou et al., UR-DMU, AAAI 2023), and a BatchNorm-based weakly supervised
detector (Zhou et al., BN-WVAD, 2023). VadCLIP (Wu et al., AAAI 2024) replaced
the feature extractor with frozen CLIP and added a language branch; DSANet (AAAI 2026) adds
disentangled semantic alignment and is our host. All of these report
frame-level micro-AUC or AP as the headline number; none decomposes it.

**Metric critique.** The point-adjust critique in time-series anomaly detection
(Kim et al., AAAI 2022) established that a single convention can inflate scores
by a large factor. For VAD, the two 2026 audits discussed in Section~\ref{sec:audits} document the
consequences of pooling. Our contribution is orthogonal in kind: not another
audit but a decomposition with weights, an exchange rate, a degeneracy
construction, an identification analysis, and estimators.

**Causal VAD.** Lin et al. (AAAI 2022) cast *unsupervised* VAD as a causal
inference problem with pseudo-labels as treatment and matching plus a
counterfactual ensemble. Later work uses causal vocabulary for regularisation:
event-relevant factor disentanglement with counterfactuals (AAAI 2023), causal
representation consistency (CIR, 2023), scene-debiasing with causal normality
learning (CRCL, 2025), prototype interventions (Ning et al., Pattern Recognition 2026), and causal
feature-guided augmentation (Li et al., CVIU 2026). We do not propose a causal *model*; we
use the causal toolkit for what it is best at here --- defining the estimand that
the metric is a weighted average of, and separating a mediator from a confounder
with a placebo intervention.

**Weak supervision and identifiability.** That MIL objectives identify only
symmetric functions of the instance scores is folklore in the MIL literature
(Ilse et al., ICML 2018; the noisy-OR/ max identifiability discussions) and is
the formal core of UMIL's (2023) analysis of instance-level bias. Theorem 4 turns
it into a statement about the *metric*: what the objective does not identify, the
report cannot measure.

**Sensitivity analysis.** The E-value (VanderWeele and Ding, Annals of Internal
Medicine 2017) and Rosenbaum-style bounds are standard in epidemiology and rare
in computer vision. We use the E-value because the contrast here is a pair-level
risk ratio by construction, and because a single number ("an unmeasured
within-video factor would need a 4.5-fold association with both status and
score") is more useful to a reader than an unqualified claim of robustness.

**Decomposition of AUC.** Splitting a pooled AUC into within-group and
between-group components is classical in the discrimination literature (the
within/between variance decomposition behind multiclass AUC, Hand and Till 2001).
What is new here is not the algebra but its use as a *measurement instrument*:
computing the weights from the split, deriving the exchange rate from them,
bounding what any method can achieve, and defining an estimator for the component
the field believes it is reporting.

## 3. Setup and notation {#sec:setup}

**Notation.** We use one letter for one thing. $s$ is the scorer, $s_i=s(x_i)$
the score of unit $i$, and $S_i$ the corresponding random variable; $y_i\in\{0,1\}$
is the ground-truth label of unit $i$ and $y_v$ the label vector of video $v$;
$a_v$ and $b_v$ count the anomalous and the background units of an anomalous video
$v$. $\theta$ is the reported statistic of Eq.~\eqref{eq:theta}; $A$, $B$, $C$ are
the block statistics of Theorem 1 and $\mathcal{A}$, $\mathcal{B}$, $\mathcal{C}$
the corresponding pair sets. $B$ is also our name for the localization estimand
(they coincide by construction), $B|z$ is its motion-stratified version
(Section~\ref{sec:strat}), $D=B-B|z$, and a prefix $\Delta$ always means the
difference between two scorers on one split ($\Delta\theta=\theta-\theta'$ and so
on). Intervals are 95\% video-clustered bootstrap confidence intervals (CI), $E$
is the E-value of Section~\ref{sec:evalue}, $\rho_s$ is Spearman's rank
correlation across a family of checkpoints together with its two-sided $p$-value,
and $\delta$ is the share of contaminated pair mass of Section~\ref{sec:bound};
$\pi_{\mathrm{mot}}=(B-B|z)/(1-B|z)$ is the motion-mediated share of the
within-video signal (a mediation proportion, Section~\ref{sec:bound}).

**Data.** A test split is a collection of $V$ videos; video $v$ contributes
$n_v$ scored units (snippets), each unit carrying a binary label
$y_i\in\{0,1\}$, and $y_v\in\{0,1\}^{n_v}$ collects the labels of one video. Let $\mathcal{P}$ be the set of *ordered* frame pairs
$(i, j)$ with $y_i=1,\ y_j=0$, and $N=|\mathcal{P}|$. Let $\mathcal{A}\subseteq\mathcal{P}$ be the pairs whose
negative member lies in a *normal* video, $\mathcal{B}\subseteq\mathcal{P}$ the pairs whose negative member
lies in the *same* anomalous video as the positive member, and $\mathcal{C}=\mathcal{P}\setminus(\mathcal{A}\cup\mathcal{B})$
the pairs whose negative member lies in a *different* anomalous video. For a
scoring function $s$, the reported statistic is

\begin{equation}
\theta=\frac{1}{N}\sum_{(i,j)\in\mathcal{P}}\Big(\mathbb{1}[s_i>s_j]+\tfrac{1}{2}\,\mathbb{1}[s_i=s_j]\Big),
\label{eq:theta}
\end{equation}

the Mann--Whitney form of frame-level micro-AUC (Hanley and McNeil, 1982; ties contribute one half,
equivalently the average-rank convention; AP is
handled in Section~\ref{sec:ap}).

Write $n_A=|\mathcal{A}|$, $n_B=|\mathcal{B}|$, $n_C=|\mathcal{C}|$, and let $A, B, C$ be the same
statistic restricted to each block. Then $\omega_Z=n_Z/N$ for $Z\in\{A,B,C\}$.

**Why these three blocks.** $\mathcal{A}$ compares an anomalous frame with a frame from a
video that is entirely normal: it can be won by recognizing the *scene* (a
highway vs. a mall), since normal videos and anomalous videos are different
recordings of different places. $\mathcal{B}$ is the only block in which both frames come
from the same recording, so scene, camera, illumination, and encoding are held
fixed by construction: it is the only block that can measure *localization*.
$\mathcal{C}$ again crosses videos, but both sides are anomalous recordings, so it
measures relative severity rather than normality.

**The pair census is data-only.** $\omega$ depends on the label layout of the split,
not on any model. Table~\ref{tab:census} lists the census for the two standard benchmarks.

\begin{table}[t]
\centering
\caption{Pair census of the two benchmarks, computed from the ground-truth
arrays alone.  The weights are properties of the label layout, not of any
model.}
\label{tab:census}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
split & videos & frames & positives & $\omega_A$ & $\omega_B$ & $\omega_C$ & $n_B/N$ & $1/\omega_B$ \\
\midrule
UCF-Crime test & 290 & 69,368 & 5,260 & 0.6317 & 0.00388 & 0.3645 & 0.388\% & 257.7 \\
XD-Violence test & 800 & 2,331,296 & 537,882 & 0.6985 & 0.00071 & 0.3008 & 0.071\% & 1,407.6 \\
\bottomrule
\end{tabular}
\end{table}

Two facts are worth stating before any theory. First, $\omega_B$ is about two
orders of magnitude smaller than the other two weights. Second, $\omega_B$ shrinks as
the benchmark grows (0.0039 $\to$ 0.0007) because the number of same-video pairs
grows only linearly in the video length, while the cross-video pairs grow
quadratically: **the larger the benchmark, the less the reported number is about
localization.** Song and Lee report the same fraction in the range 0.071--0.388%
across five benchmarks; our census reproduces both endpoints of that range,
UCF-Crime at 0.388% and XD-Violence at 0.071%.

**The estimand.** We define the *localization ability* of a scorer as
$\tau(s)=B$, the probability that, inside an anomalous video, a randomly chosen
anomalous frame is scored above a randomly chosen normal frame of that same
video --- a within-unit contrast whose control group is supplied by the video
itself. If a scorer is to be credited with localization, $\tau$ is the quantity that
must move; $\theta$ is not.

## 4. What the reported number is a measure of {#sec:results}

Throughout, $s$ is any real-valued scorer of the scored units, ties contribute
one half as in Eq.~\eqref{eq:theta}, and $\omega_Z$ are the split-level pair weights of Section~\ref{sec:setup}.

### 4.1 Theorem 1 (exact decomposition) {#sec:thm1}

For every scorer $s$,

\begin{equation}
\theta=\omega_A A+\omega_B B+\omega_C C,\qquad \omega_A+\omega_B+\omega_C=1 .
\label{eq:decomp}
\end{equation}

*Proof.* $\mathcal{P}=\mathcal{A}\sqcup\mathcal{B}\sqcup\mathcal{C}$ is a partition, and $\theta$ is an average over $\mathcal{P}$.
Grouping the sum by block gives the result, with $A, B, C$ the block averages.
The weights are pair-count ratios and do not involve $s$. $\blacksquare$

The identity is not an approximation: we verified it to 0.00000 on eleven UCF-Crime and four XD-Violence
checkpoints (maximum absolute deviation $<10^{-12}$) and on every
probe below. It is also *not* the same statement as "AUC is dominated by
cross-video pairs" --- it says exactly how much, and it separates the two
different cross-video mechanisms (A: scene/normality; C: relative severity).
The decomposition is an accounting identity, not a causal attribution; the
danger of reading an aggregate association as a component effect is the
classical one (Simpson, 1951).

**Corollary 1 (exchange rate).** $\partial\theta/\partial B=\omega_B$. On UCF-Crime one point of $B$
buys 0.0039 points of $\theta$; equivalently **one reported point costs 257.7 points of
localization** on UCF-Crime and 1,407.6 on XD-Violence. Since
$(\omega_A+\omega_C)/\omega_B=256.7$ (UCF) and $1,407.6$ (XD), a method that raises
*scene-level* discrimination by one point gains what a method that raises
localization by 257 (UCF) or 1,408 (XD) points gains.

### 4.2 Theorem 2 (degeneracy: the constant-within-video scorer) {#sec:thm2}

Let $s$ be constant inside every video, $s_i=f(v)$ for all $i\in v$. Then
$B(s)=1/2$ exactly, and

\begin{equation}
\theta=\omega_A A_f+\omega_C C_f+\omega_B/2,
\end{equation}

with $A_f,C_f$ the block statistics of the video-level score $f$. Consequently,
*among scorers that are constant inside each video*,
$\sup\theta=\omega_A+\omega_C\,C^{*}+\omega_B/2$, where $C^{*}$ is the largest
pair-weighted concordance that any ranking of the anomalous videos attains on
$\mathcal{C}$. $C^{*}<1$ strictly: for every ordered pair of distinct anomalous
videos, both directions lie in $\mathcal{C}$, and at most one of them can win.
(Over all scorers the supremum is of course 1, attained by a perfect localizer.)

*Proof.* For any pair inside the same video the two scores are equal, so the
indicator is a fair coin: $B = 1/2$. The second display is Eq.~\eqref{eq:decomp}.
For the supremum: $A_f\le 1$, attained by ranking every anomalous video above
every normal one; subject to that, $\theta$ is maximized by maximizing the
weighted concordance $C_f$ over rankings of the anomalous videos, a finite
linear-ordering problem whose maximum $C^{*}$ is strictly below 1 by the
two-direction argument above. $\blacksquare$

Theorem 2 is the mechanism behind the empirical observation of Song and Lee: a
scorer with *no* localization ability whatsoever is not far below the state of
the art, because the only block it forfeits entirely is $\mathcal{B}$ --- a
video-level score can still win all of $\mathcal{A}$ and most of $\mathcal{C}$
(the classifier below attains $C_f\approx0.56$). Measured: a
linear video-level classifier trained on cached CLIP features reaches
$\theta=0.8234$ with $B=0.5000$ exactly, against $\theta=0.8257$ for a frame-level
MIL model trained identically ($B=0.657$) and 0.8772 for the published
DSANet checkpoint. The video-constant scorer thus keeps **99.7% of the
identically-trained frame model's $\theta$ and 93.9% of the published DSANet
checkpoint's $\theta$** --- and 92.1% of the best number reported in the
literature (89.4%) --- while its localization term is exactly chance.

Note what Theorem 2 does *not* say: it does not say the constant scorer has a
high $B$, nor that $\theta$ is a poor measure of *anomaly detection in general*. It
says that $\theta$ is an average over three comparison sets of which one --- the only
one that can see localization --- carries 0.388% of the weight.

### 4.3 Theorem 3 (the inversion is uninformative and unusable) {#sec:thm3}

Suppose a paper reports $\theta$ and $\theta'$ for two scorers and claims the difference is
due to localization. By Theorem 1,

\begin{equation}
\Delta B=(\Delta\theta-\omega_A\Delta A-\omega_C\Delta C)/\omega_B.
\end{equation}

Two things follow.

**(a) Non-identification.** Without separately measuring $\Delta A$ and $\Delta C$, a
difference $\Delta\theta$ is compatible with *any* $\Delta B\in[-1,1]$: choose $\Delta A$ accordingly
($\Delta A=(\Delta\theta-\omega_B\Delta B-\omega_C\Delta C)/\omega_A$, which is feasible for every $\Delta B$ because
$\omega_A=163\,\omega_B$). A reported AUC difference therefore carries no information
about which block improved until A and C are reported.

**(b) Error amplification.** If $\Delta\theta$ is measured with standard error $\sigma$, the
implied $\Delta\hat{B}$ has standard error $\sigma/\omega_B=258\sigma$ (UCF) and $1{,}408\sigma$ (XD). With the
seed-to-seed $\sigma(\theta)=0.0049$ that we measure on our checkpoints (Section~\ref{sec:resolution}),
a single-seed comparison gives $\mathrm{SE}(\Delta\hat{B})=1.26$: the 95% interval implied for
$\Delta B$ is about ten times the width of $B$'s entire range (0.5), so the
reconstructed quantity is not even bounded by the parameter space. To certify a
one-point localization gain ($\Delta B=0.01$) with a two-standard-error margin one
would need $\sigma(\Delta\theta)\le 1.9\cdot10^{-5}$ --- a standard error 253 times
smaller than the observed seed noise, which at $\sigma\propto k^{-1/2}$ would take
roughly $253^2\approx64{,}000$ seed replicates.

*Proof.* Algebra of Theorem 1; feasibility of $\Delta A$ uses $\omega_A>0$; the
amplification is $1/\omega_B$. $\blacksquare$

Theorem 3 is why we do not merely *report* B alongside $\theta$: we argue that B cannot
be *recovered* from reported $\theta$ differences, so the community's practice of
reading localization progress off the AUC column is not merely noisy but
structurally empty. The only remedy is to measure the blocks.

### 4.4 Theorem 4 (weak labels do not identify the within-video ordering) {#sec:thm4}

Let the training loss $\mathcal{L}$ depend on the frame scores of anomalous video $v$ only
through a symmetric function of the multiset $\{s_i: i\in v\}$ --- as every MIL
objective does (top-k mean, maximum, attention pool, expectation over a
categorical head). For any video-wise permutation $\pi$ that keeps the labels of
the frames fixed, $\mathcal{L}(s\circ\pi)=\mathcal{L}(s)$. Hence the minimizer set of the training
problem is closed under such permutations, and $B$ is not a functional of the
labels: **for the same minimal loss, every ordering of the frames inside a video
is admissible**, including ones with $B=1$ and ones with $B=0$. The scope of the
statement is set by its hypothesis: it covers objectives all of whose terms are
functions of the multiset of scores, and it excludes any term that couples frames
at different time indices --- in particular the temporal-smoothness penalty
$\sum_t\lVert s_{t+1}-s_t\rVert^{2}$ used by Sultani et al., which is *not*
permutation-invariant and therefore does constrain the ordering.

Two things delimit the claim, and we prefer to state both here than to have a
reader find them.

*Objectives with time coupling are outside the theorem.* A temporal-smoothness
penalty on the scores, or any other term that depends on where a frame sits in
the sequence, breaks the permutation invariance and pins part of the ordering.
The original MIL objective of Sultani et al.\ contains such a term, so the
theorem does not cover it; what covers it is the empirical record of item (ii)
below, which tests objectives with and without such regularisers and finds the
ordering unmoved either way.

*The premise concerns the value of the loss, not its gradient.* The
top-$k$-minus-mean contrast $T_v$ of remark (ii) is symmetric *as a functional*
--- it is a function of the multiset of scores --- so the theorem does apply to
it: every permutation of the frames attains the same loss and the minimizer set
remains permutation-closed. What is asymmetric is where the gradient flows, and
that does select an ordering during optimization. We count that, and every
regulariser with the same property, as part of a method's inductive bias rather
than as identification: nothing in the weak labels makes the ordering it selects
correct, and the constraint is itself an assumption about anomalies (that they
are a sparse, top-scoring sub-segment) that the annotation does not verify. The
empirical claim --- that published objectives do not pin the ordering --- is
unaffected.

*Proof.* Symmetry of the pooling function gives equality of the loss term, and
the same argument applies term by term to every other term that is a function of
the multiset of scores; by the hypothesis the objective contains no term of any
other kind. Since $B$ is a function of the pairwise ordering, the claim
follows. $\blacksquare$

Three remarks.

(i) Theorem 4 is a statement about the *objective*, not about optimization: SGD
plus architectural inductive bias (temporal attention, positional encodings)
does impose *some* ordering, but nothing in the objective rewards the ordering
being correct.

(ii) It explains a body of negative results. A companion study
(DebiasVAD) tried thirteen post-processing schemes on top of weak-label training
(halo suppression, IPW reweighting, score-invariant reordering, oracle
recalibration with ground-truth distances, \dots{}); all thirteen failed, and an oracle
that reweights by the *true* distance to the annotated segment was 0.0045 AP
*worse* than no correction. An independent attempt of ours --- training an
auxiliary head to make the top-$k$-minus-mean contrast
$T_v=\mathrm{top}_k(s)-\mathrm{mean}(s)$ --- where $\mathrm{top}_k$ is the mean of
the $k$ largest scores of a video and $\mathrm{mean}$ the mean of all of them ---
positive on anomalous videos fails for a structural reason visible in one line:
$T_v\ge 0$ always, so the constraint it imposes on normal videos
($y_v=0\Rightarrow T_v\le m$, for a margin $m$) is unsatisfiable, and the arm ends with $B=0.3124$ and 81% of
the videos ranked *backwards*.

(iii) The corollary is a design constraint, not a counsel of despair: it says
that localization must be obtained from something *other* than the weak labels ---
architecture, auxiliary modalities, or a training signal that does not collapse
the frames into a symmetric statistic. It also says that any claimed
localization improvement must be *measured* on B, since the labels could not have
produced it.

### 4.5 What about AP? {#sec:ap}

Average precision is not additive over blocks, because precision is computed
along the global ranking and interacts with the pool of negatives as a whole. We
therefore do not claim an exact law for AP, but the qualitative statement is
stronger rather than weaker: AP is an average over the same pool of
positive-negative pairs but weighted by the rank positions of positives, and the
$n_B/N$ share is the same 0.388% (UCF) / 0.071% (XD), so the arguments of
Sections~\ref{sec:thm1}--\ref{sec:thm4} about which pairs the statistic can see transfer qualitatively. The
pair-weighted view of AP has a precedent in precision at a recall target
(Tatbul et al., NeurIPS 2018), where the same pool-dependence is why a single
scalar is not additive. What
does not transfer is the exact law: AP's weights depend on rank positions and
therefore on the whole negative pool, so an AP version of Theorem 1 would need a
price-weighted decomposition that we do not attempt. Empirically the two
statistics agree in ordering on the checkpoints we have: the four XD-Violence
checkpoints of Section~\ref{sec:xd} have frame AP 0.8113--0.8658 and micro-AUC 0.9368--0.9536, and
the two orderings coincide exactly. We keep the theory to AUC and say so
wherever AP is used.

## 5. Identifying, bounding, and estimating the within-video contrast {#sec:contrast}

Theorems 1--4 say what to measure. They do not say whether the measured $B$ is
the quantity we want, because $B$ is an *observational* contrast: inside an
anomalous video the frames labelled anomalous differ from the frames labelled
background in many ways, and the scorer may be responding to any of them. This
section makes the target explicit and then tests the threat that dominates the
discussion in practice --- motion.

### 5.1 The contrast as a causal estimand {#sec:causal}

Fix a *frozen* scorer $s$ and an anomalous video $v$ with $a_v$ anomalous and
$b_v$ background units. For a unit $i$ let $y_i\in\{0,1\}$ be its (annotated)
status and $S_i=s(x_i)$ its score. The within-video contrast is

\begin{equation}
\tau_v=P(S_i>S_j\mid y_i=1,y_j=0,\ i,j\in v),
\end{equation}

and the localization term is the pair-count weighted average $B=\sum_v w_v\tau_v$,
$w_v\propto a_v b_v$. Two features distinguish $\tau_v$ from the usual detection
estimand. First, the video is its own control: scene, camera, illumination and
encoding are held fixed by construction, which is exactly why $\mathcal{B}$ is the
informative block. Second, the unit of replication is the *video*: $\tau_v$ is
estimated inside one video, so uncertainty must be assessed by resampling videos
(Section~\ref{sec:inference}), not pairs. Identification of $\tau_v$ requires
four assumptions:

- **A1 (no interference / well-defined unit score).** $S_i$ depends on unit $i$
  only. This holds for a post-hoc scorer applied unit-wise, and it is what a
  deployed detector does *within a window*, but it does **not** hold for the
  training objective of a MIL model, where the top-$k$/max pool makes the loss
  depend on the joint configuration of scores (Section~\ref{sec:thm4}). We therefore estimate $B$
  for frozen checkpoints at inference time and we flag that the score of a frame
  produced by window attention is itself a joint quantity.
- **A2 (within-video ignorability).** Given a covariate $Z$ that varies inside
  the video, the status of a unit is independent of its score's rank:
\begin{equation}
y_i \perp \mathrm{rank}(S_i)\mid Z_i,v.
\end{equation}
- **A3 (positivity).** $a_v,b_v>0$ --- true by construction of $\mathcal{B}$.
- **A4 (no annotation error).** The labels equal the true status.

Three of these are known to be strained in this literature and we prefer to name
them than to hide them. A1 is violated by design at training time (Section~\ref{sec:thm4}). A3
holds for UCF-Crime and XD-Violence but *not* for the 150 UCF-Crime test videos
that are entirely normal (51.7% of the split): those videos contribute to
$\mathcal{A}$ but cannot
contribute to $\mathcal{B}$, so $B$ is estimated on a biased subset of the split --- the
anomalous ones. A4 is strained by the coarse, segment-level annotation of both
benchmarks: a companion study (DebiasVAD) found that 95--97% of the AP loss of a
trained detector comes from "halo" frames --- normal-looking frames inside
anomalous videos that the model scores high --- and that these are partly
*genuinely* anomalous frames outside the annotated segment; reordering frames by
their distance to the annotated segment made AP *worse* ($-$ 0.029). If A4 fails,
$B$ is biased *downward* (true anomalies are counted as controls), so our
estimates are conservative. A2 is the one assumption that no observable feature of
the data can check, so we do not ask the reader to grant it: Section~\ref{sec:bound} replaces point
identification by an interval whose width is a nuisance share the reader chooses.

### 5.2 Motion: confounder, mediator, or neither? {#sec:motion}

The canonical nuisance inside a video is motion energy. The argument for
treating it as a confounder is direct: anomalies involve motion, and a scorer
sensitive to motion energy will rate any high-motion frame highly, so part of
$B$ could be pure motion response rather than anomaly evidence. If motion is a
confounder --- a common cause of the label and the score --- the observed
contrast is inflated, and adjusting for it should move the estimate *down*
toward the direct effect. The alternative is that motion is on the causal path
--- the anomaly *causes* motion, which *causes* the score --- in which case
adjusting also moves the estimate down, by removing signal. This is the
confounder-versus-mediator fork of causal adjustment (Pearl, 2001). The sign of the
adjustment response therefore cannot by itself separate the two hypotheses; what
the design below tests is whether the response is specific to real motion
structure, and even that test is one-sided, because the placebo of item (3) is a
weak control: a nuisance permuted across videos is nearly uncorrelated with the
score by construction.

Stratification alone cannot separate the two, because the adjustment is the same
operation. We therefore run three interventions, one of which is a placebo:

1. **Stratify** (observational): compare anomalous and background units only
   within the same motion quintile inside the same video ($B|z$, Section~\ref{sec:strat}).
2. **Intervene on training** (experimental): add to the loss a penalty on the
   squared within-video score--motion correlation, with weight $\gamma$:
   $\gamma\,\mathrm{mean}_v\mathrm{corr}^2(S_v,Z_v)$
   (Section~\ref{sec:mediation}).
3. **Placebo**: the same penalty applied against *shuffled* motion --- $Z$ is
   permuted across videos so that the penalty acts on the wrong video's motion
   profile. If the effect of (2) is due to decorrelating against real
   anomaly-related structure, (3) must be inert; if it is a generic
   regularisation effect, (3) will look like (2).

The result (Section~\ref{sec:mediation}) is a consistent pattern: (1) lowers $B$ by 3.5--5.8 points; (2)
lowers it monotonically in $\gamma$; and (3) is inert --- it recovers the
unregularised level (0.689 against a baseline of 0.680 for $B$, 0.648 against 0.643 for
$B|z$). Both hypotheses predict a drop under (1) and (2), so the sign alone does
not decide between them; what the pattern establishes is that the response is
specific to real motion structure rather than to regularisation as such. We read
motion as a **mediator** --- the annotated events are largely motion-defined, so
the path anomaly $\to$ motion $\to$ score is the natural one --- and on that
reading adjusting for it is over-adjustment: the honest target is the
*unadjusted* $B$, and the adjusted $B|z$ is a diagnostic of how much of the
localization signal is motion-mediated. The size of that adjustment is the gap
$D=B-B|z$, which is 3.5--5.8 points (median 4.9) across the eleven UCF-Crime
checkpoints.

The contrast with a confounder is worth stressing, because the instinct in this
literature is to "de-bias" against motion: had motion been a confounder, $D$
would be the *spurious* part and $B|z$ the estimand of interest. We flag two
limits of the verdict. First, the placebo is a weak control for the reason given
above, so its inertness rules out less than the design suggests. Second, the
mediator reading predicts that the gap $D$ should grow with the score--motion
correlation; across the eleven checkpoints it does not (Spearman
$\rho\approx-0.08$). The mediator reading remains the more natural one, but it is
a modelling assumption we make and state, not a causal verdict the design
delivers.

Two further diagnostics agree. A zero-training motion statistic --- the 5-frame
moving average of $\lVert\Delta x_t\rVert$ on frozen features --- attains $B=0.6552$ and
$B|z=0.5465$, i.e. it accounts for $(0.6552-0.5)/(0.6803-0.5)=86\%$ of
the within-video signal of a trained frame-level probe and for *nothing* after
stratifying on itself. And the remaining coupling in trained models is small but
non-zero: the within-video correlation between score and motion is 0.12--0.33
across the eleven UCF-Crime checkpoints (mean 0.18). A scorer is not a motion detector, but
in the only block that measures localization it is close to one.

### 5.3 When A2 fails: a partial-identification bound {#sec:bound}

A2 is the load-bearing assumption of the whole construction, and it is not
verifiable without re-annotating the benchmark: nothing in the data tells us
whether the background frames of a violent video are unremarkable *apart from*
the violence. Rather than assert it, we replace point identification by an
interval and let the reader choose the violation. Model the violation as
contamination of the pair mass: within each video, an unknown fraction $\delta$ of
positive--background pairs is ranked by a video-level nuisance (temporal
position, exposure, camera drift, annotation spill-over) rather than by the
anomaly, and for those pairs the comparison may go either way; the remaining
$1-\delta$ are ranked as the causal contrast $\tau$ would rank them. (Bounds of exactly
this kind are the standard substitute for unverifiable point identification in the
partial-identification literature; see Manski, 1990, for the original worst-case
construction in a treatment-effect setting.)

**Proposition 5 (partial identification of $\tau$ under A2 violation).** Under A1, A3,
A4 and $\delta$-contamination as above, with $B$ the reported within-video contrast,

\begin{equation}
\tau_{\mathrm{LB}}(\delta)=\max\Big\{0,\frac{B-\delta}{1-\delta}\Big\}\;\le\;\tau\;\le\;\min\Big\{1,\frac{B}{1-\delta}\Big\}=\tau_{\mathrm{UB}}(\delta),
\qquad 0\le\delta<1.
\label{eq:tau-bound}
\end{equation}

*Proof.* Conditioning on which pairs are contaminated, $B=(1-\delta)\tau+\delta q$ with
$q\in[0,1]$ the (unknown) rate at which the contaminated pairs happen to be
ranked correctly; solving for $\tau$ over the range of $q$ gives the two endpoints.
Both endpoints are then intersected with the unit interval, the range of $\tau$:
the upper one because $\tau\le 1$, the lower one because a nuisance share above $B$
leaves no information and the bound correctly degrades to the trivial $0\le\tau$.
$\blacksquare$

The bound is stated under A1, and A1 is the one assumption the host does not
satisfy: a frame score produced by attention across the window is a function of
the whole video rather than of its own frame. Interference of that kind can move
a given pair either way, so the interval is not a certificate for the
attention-based checkpoints we actually measure; it is the bound a unit-wise
scorer would earn, and for our host it should be read as the tolerance
$\delta^{*}>1/2$ of a *fixed* window-level aggregation rather than as evidence
about localization. Three further limitations of the reading appear in
Section~\ref{sec:limits}.

Three consequences make the bound useful rather than decorative.

*Identification power, per checkpoint.* $\tau_{\mathrm{LB}}(\delta)>0.5$ --- the claim "this model
ranks anomalous frames above its own video's background better than chance" ---
holds exactly while $\delta<2B-1=:\delta^{*}$. So every reported $B$ carries its own
tolerance: a nuisance share below $\delta^{*}$ cannot explain the result away, one above
it can. On the eleven UCF-Crime checkpoints $\delta^{*}\in[0.416,0.515]$ (median 0.447) and
on the four XD-Violence checkpoints $\delta^{*}\in[0.609,0.693]$, i.e. the localization
claim survives unless roughly 40% (UCF) to 60% (XD) of the within-video pair mass
is nuisance-driven.

*What the measured nuisance does and does not buy.* The only nuisance we can
quantify is motion, and the stratified estimator attaches to it the mediation
proportion $\pi_{\mathrm{mot}}=(B-B|z)/(1-B|z)\in[0.120,0.170]$ (median 0.149) on UCF and
$[0.122,0.141]$ on XD. This is *not* a share of contaminated pair mass and must
not be compared with $\delta^{*}$: it is derived from the mixture model by
setting $q=1$ --- contaminated pairs ranked correctly --- the endpoint most
favourable to the claim, so it measures how much of the *signal* is
motion-carried, not how much of the pair mass a nuisance governs. A nuisance
that threatens the localization claim lives at the opposite end of the mixture
($q$ small), and no measured candidate populates that end. The bound, not the
measured share, is the tool for unnamed nuisances.

*Monotonicity.* $\tau_{\mathrm{LB}}$ is increasing in the reported statistic --- it is
affine in $B$ with slope $1/(1-\delta)>1$ --- so the bound cannot invert the ordering
of checkpoints. What it does to differences is the opposite of compressing them:
pairwise gaps are *stretched* by $1/(1-\delta)$, while the admissible range shrinks
to $[(1/2-\delta)/(1-\delta),1]$, which at $\delta=0.2$ is $[0.375,1]$ and still places the
estimates at 0.635--0.697 (UCF) and 0.756--0.808 (XD), well above chance.

Table~\ref{tab:partial} (Appendix~\ref{sec:appC}) collects the numbers per
checkpoint. The bound is deliberately conservative in one direction only:
contamination can push $B$ *up* as well as down, which is why a high $B$ under
an unknown $\delta$ must not be read as a high $\tau$ unless $\delta<\delta^{*}$.

We check the algebra and the sharpness of the bound by simulation (Appendix A):
with $m=400$ positives, $n=4{,}000$ negatives and a known $\tau\approx0.760$, pushing
the $\delta m$ contaminated positives above every negative gives observed
$B=0.785/0.809/0.832/0.880$ for $\delta=0.1/0.2/0.3/0.5$, and the
interval $[\tau_{\mathrm{LB}},\tau_{\mathrm{UB}}]$ contains the truth in every run, with $\tau_{\mathrm{LB}}$ converging to
it from below as $\delta$ grows (0.7611 / 0.7606 / 0.7606 / 0.7603). The upper endpoint
is vacuous once the worst case saturates $q=1$, which is expected: a bound that
allows the nuisance to be maximally favourable carries its information in the
lower endpoint.

### 5.4 What we therefore report {#sec:report}

$B$ (unadjusted) as the localization estimand, $B|z$ as the motion-mediated
share, both with video-clustered intervals, and the E-value of $B$ as a bound on
how much unmeasured confounding (time position, camera motion, annotation
error) the claim could survive. We do *not* propose an adjusted estimator as an
improvement: our attempts to build one (post-hoc residualisation, training-time
orthogonalisation) both reduce $B$ on real checkpoints, which is what the
mediator diagnosis predicts (Section~\ref{sec:mediation}).

### 5.5 The pair-weighted estimator {#sec:pairw}

For a frozen scorer and per-video label vectors $y_v$,

\begin{equation}
\begin{gathered}\hat{B}=\frac{\sum_v a_v b_v \hat{B}_v}{\sum_v a_v b_v},\\ \hat{B}_v=\frac{1}{a_v b_v}\sum_{i:\,y_i=1}\sum_{j:\,y_j=0}\mathbb{1}[s_i>s_j].\end{gathered}
\end{equation}

$\hat{B}_v$ is the Mann--Whitney statistic (Mann and Whitney, 1947) of video $v$ and is computed from ranks in
$O(n_v\log n_v)$; the whole estimate costs one sort per video. $\hat{B}$ is the
pair-count weighted mean, *not* the unweighted mean of the $\hat{B}_v$ --- the two
differ systematically, because videos with few positive frames (short events)
have both noisy and differently weighted $\hat{B}_v$. On UCF-Crime the difference is
not academic: the unweighted mean over videos is 0.7046 for the base checkpoint
while the pair-weighted $\hat{B}$ is 0.7558 (0.05 apart), and in a motion baseline the
two move in *opposite* directions relative to the model (unweighted 0.7186 vs
pair-weighted 0.6552). We report the pair-weighted version because it is the
component of $\theta$ (Theorem 1); the unweighted version answers a different question
("does the average video localise?") and should be labelled as such.

### 5.6 The stratified estimator {#sec:strat}

Motion is observed per unit. For a nuisance $z$ (here: the 5-frame moving
average of $\lVert\Delta x_t\rVert$, standardised inside the video) define $Q$-quantile bins
inside each video, keep only pairs whose two members fall in the same bin, and
average with pair weights. With $Q=5$ this is a matched comparison: the
anomalous and background units being compared have the same motion energy to
within a quintile. (Five rather than ten bins is a bias--variance compromise: each
bin then keeps enough control units for the within-bin contrast to be estimable in
short videos; $Q=10$ lowers $B|z$ by 0.7 point on the base checkpoint and does
not change any conclusion.) Two diagnostics accompany it: the *overlap* (the fraction of
positive units that share a bin with at least one control unit; 0.943 on
UCF-Crime, per-video mean 0.941) and the same statistic evaluated on $z$ itself,
which must fall towards 0.5 if the stratification has taken the nuisance out (it
does: 0.6552 $\to$ 0.5465). A residualised variant --- the natural "de-biased"
estimator, ridge-fitted on control units and subtracted out-of-fold --- lowers
$B$ from 0.7558 to 0.7392 on the base checkpoint, exactly as the mediator
diagnosis predicts; we report it as a sensitivity check only, and it is the
reason we do not sell it as a method (Appendix~\ref{sec:appC}).

### 5.8 Inference: the video is the unit {#sec:inference}

$\hat{B}$ averages over about $10^6$ pairs but only 140 anomalous videos, and the frame-level
pairs inside a video are strongly dependent. Any interval that treats pairs as
independent is off by orders of magnitude. We use a **cluster bootstrap** (Efron and Tibshirani, 1993): draw
$V$ videos with replacement, recompute $\hat{B}$ and $B|z$ on the resampled set
(videos with no positive or no negative units contribute nothing), and take
percentile intervals over 2,000 replicates. For the base UCF-Crime checkpoint
this gives $B=0.7558\ [0.6844,0.8364]$ and $B|z=0.7166\ [0.6467,0.7907]$ ---
a $\pm$ 0.08 interval, i.e. **the between-video variation dominates everything the
leaderboard reports**. Reporting $B$ without such an interval would repeat the
original sin at a smaller scale.

### 5.9 An E-value for the contrast {#sec:evalue}

Treat "unit $i$ outscores unit $j$" as the outcome and the status of $i$ as the
exposure. Then $P(\mathrm{win}\mid\mathrm{anomaly})=\tau$ and $P(\mathrm{win}\mid\mathrm{background})=1-\tau$, so the
pair-level risk ratio is $RR=\tau/(1-\tau)$ and the E-value of VanderWeele and Ding
applies:

\begin{equation}
E=RR+\sqrt{RR\,(RR-1)}.
\end{equation}

$E$ is the minimum strength of association (on the risk-ratio scale) that an
unmeasured confounder would need with *both* the status and the score to
explain away the contrast. For the base checkpoint $B=0.7558$ gives $E = 5.6$
and $B|z=0.7166$ gives $E = 4.5$: to be an artefact, an unmeasured
within-video factor would have to raise the odds of being anomalous and of
scoring high by more than 4.5-fold each. That is a substantive statement about
the evidence, and it is cheap to compute. Two things must be said against it, and
the first is arithmetic. An E-value computed at a point estimate is the optimistic
end of a bound: computed at the lower end of the video-clustered interval the same
checkpoint gives $E\approx3.8$ for $B$ ($0.684$) and $\approx3.1$ for $B|z$
($0.647$), and those are the values a reader should quote. The second is
structural: because $P(\mathrm{win}\mid\mathrm{background})=1-\tau$, the risk ratio is a
deterministic function of $\tau$, so $E$ is a monotone relabelling of $B$ --- it
supplies a scale that a reader can argue about, not information $B$ does not
already contain, and it inherits every construct-validity question $B$ has
(Campbell and Fiske, 1959).
Caveat: the E-value also treats pairs as independent; with clustering it is
optimistic, so we report it alongside, not instead of, the bootstrap interval.

### 5.10 Protocol {#sec:protocol}

For every reported system, publish the four numbers $\theta$, $B$, $B|z$ and $E$,
the last three with a video-clustered 95\% interval.

The cost is one extra evaluation pass on frozen scores (seconds) and one
paragraph of text. The rest of the protocol follows from the theorems:

1. Compute the pair census once per split and state $\omega$ (it is a property of the
   data, not of the method).
2. Do not claim localization from $\theta$. By Theorem 3 a $\Delta\theta$ is compatible with any
   $\Delta B$; localization claims must cite $B$ or $B|z$. Cite them as a
   diagnostic column, not as a replacement headline number: by
   Section~\ref{sec:criterion} we could not certify $B$ against detection mAP,
   and we will not ask others to report a number whose external criterion we
   cannot name.
3. Require the interval. A $B$ difference on UCF-Crime is not resolvable at the
   resolution the field currently quotes: the between-seed sd is 0.018 on $B$ and
   0.027 on $B|z$, the cluster sd of a single checkpoint is 0.037--0.053, and the
   largest gap inside our eleven-arm family ($\Delta B=0.0495$) has an interval
   that includes zero (Section~\ref{sec:audit}). Same-seed comparisons, whose
   differences are paired and therefore partly correlated, resolve differences of
   roughly 0.02. Numbers below that are decoration.
4. Report $D = B - B|z$ as the motion-mediated share, and $B|z$ itself when the
   claim is about *non-motion* evidence.
5. Treat $\mathcal{A}$-only gains as scene-level: by Theorem 2 they can be obtained with a
   scorer that is constant inside each video.
6. Quote the bound, not the point estimate alone: state $\delta^{*}=2B-1$, the nuisance
   share below which the localization claim survives (Section~\ref{sec:bound}).

**A one-command implementation.** The protocol costs one evaluation pass over a
frozen score file --- no model, no retraining, no GPU. For the base checkpoint,
with 2,000 bootstrap replicates and $Q=5$, the report is: $\omega_A=0.6317$ and
$\omega_B=0.00388$, i.e.\ an exchange rate of 257.7; $\theta=0.8772$ with a
video-clustered interval $[0.8237,0.9152]$; $B=0.7558$ $[0.6844,0.8364]$;
$B|z=0.7166$ $[0.6467,0.7907]$; $D=0.0392$ with mediation share
$\pi_{\mathrm{mot}}=0.138$; $E(B)=5.64$ and $E(B|z)=4.49$;
$\delta^{*}=0.512$, with $\tau\in[0.6948,0.9448]$ at 20\% contamination; and a
within-video contribution of 0.00294 to $\theta$ (0.33\%). The same evaluation
answers the reverse question, which is the one a localization claim implicitly
asks: it converts a headline improvement $\Delta\theta$ into the movement it
demands of $\Delta B=\Delta\theta/\omega_B$ and compares that with the remaining
headroom $1-B$ and with the split's own seed noise. On UCF-Crime a claim of +0.8
points of AUC demands $\Delta B=2.06$, larger than the 0.24 of headroom that
exists, so no scorer can support that claim on this split; the arithmetic follows
from $\omega_B=0.00388$ and from nothing else.

## 6. Experiments {#sec:exp}

### 6.1 Host model, checkpoints, and verification {#sec:host}

We use DSANet (AAAI-26), a current state-of-the-art weakly-supervised detector
that operates on frozen CLIP ViT-B/16 features with a disentangled
semantic-alignment head. Of the fourteen UCF-Crime checkpoints stored on the
training machine, eleven are used here; the other three are excluded on
reconstruction grounds (two could not be rebuilt from their stored
configuration, and one rebuilds to a different pooled AUC). All four
XD-Violence checkpoints are used.
One of the excluded checkpoints (a motion-injection variant) rebuilds to
$\theta$ = 0.8494 against an archived 0.8779; we mention it because the
discrepancy --- six times the between-seed noise --- is exactly the kind of
silent reconstruction error that a third-decimal headline number invites.

*Verification.* For every retained checkpoint we recomputed $\theta$ from the stored
frame scores and compared it with the number archived by the training run:
agreement was $<10^{-4}$ in all cases (0.8772/0.8772, 0.8800/0.8800,
0.8717/0.8717, \dots{}). The decomposition of Theorem 1 reproduced $\theta$ to
$<10^{-12}$. For XD-Violence we ran the published test path (with
the XD prompt set) and reproduced the repository's printed pooled AUC exactly
(0.9449 base-s789, 0.9437 base-s555), which validates the frame-to-snippet
alignment that the pair census depends on.

*Controlled probes.* Because the published checkpoints are few and their
architecture is fixed, we also train a deliberately minimal family --- a linear
frame head with top-k MIL on the *same cached features* (4,000 training clips,
40--60 epochs, 3 seeds) --- which lets us vary one design decision at a time and
gives the arms of Section~\ref{sec:degen} and Section~\ref{sec:mediation}. These probes are not meant to be competitive;
they are meant to be interpretable.

### 6.2 The decomposition on published checkpoints {#sec:decomp}

\begin{table}[t]
\centering
\caption{Exact three-block decomposition of frame-level micro-AUC on
UCF-Crime, for the eleven checkpoints that survived our re-evaluation.  $A$,
$B$, $C$ are the block statistics, $B|z$ the motion-stratified variant,
$\mathrm{corr}$ the mean within-video correlation between score and motion energy.
Ranks are within the eleven rows.}
\label{tab:decomp-ucf}
\footnotesize
\begin{tabular}{@{}rlrrrrrrrr@{}}
\toprule
\# & checkpoint & $\theta$ & $A$ & $B$ & $B|z$ & $C$ & $\mathrm{corr}$ & rank $\theta$ & rank $B|z$ \\
\midrule
1  & base s234        & 0.8800 & 0.9751 & 0.7270 & 0.6729 & 0.7167 & 0.150 & 1  & 6  \\
2  & motion6 s789     & 0.8778 & 0.9728 & 0.7576 & 0.7230 & 0.7144 & 0.233 & 2  & 1  \\
3  & base s789        & 0.8772 & 0.9773 & 0.7558 & 0.7166 & 0.7050 & 0.141 & 3  & 2  \\
4  & cons0 s234       & 0.8759 & 0.9774 & 0.7235 & 0.6744 & 0.7016 & 0.145 & 4  & 5  \\
5  & attr s234        & 0.8724 & 0.9788 & 0.7080 & 0.6545 & 0.6899 & 0.172 & 5  & 11 \\
6  & core14txt s234   & 0.8717 & 0.9725 & 0.7169 & 0.6728 & 0.6989 & 0.151 & 6  & 7  \\
7  & motion6 s555     & 0.8717 & 0.9728 & 0.7180 & 0.6604 & 0.6981 & 0.288 & 7  & 10 \\
8  & base s555        & 0.8705 & 0.9752 & 0.7236 & 0.6688 & 0.6907 & 0.139 & 8  & 8  \\
9  & core8g1 s234     & 0.8684 & 0.9759 & 0.7124 & 0.6621 & 0.6837 & 0.126 & 9  & 9  \\
10 & res s234         & 0.8682 & 0.9767 & 0.7467 & 0.7093 & 0.6814 & 0.328 & 10 & 3  \\
11 & motion6-shuf s234 & 0.8616 & 0.9670 & 0.7171 & 0.6786 & 0.6806 & 0.121 & 11 & 4  \\
\midrule
   & spread           & 0.0184 & 0.0118 & 0.0496 & 0.0685 & 0.0361 & --    & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

Four facts about Table~\ref{tab:decomp-ucf} carry the paper.

**(a) The $\omega_B$ share is an invariant of the split, not of the model.** The
contribution of B to $\theta$ is $\omega_B B$: for every row it lies between 0.0027 and
0.0029, i.e. **0.31--0.33% of the reported number**. The remaining 99.68% is
scene-level (A, C). A corollary worth stating in words: the architecture that
wins the UCF-Crime leaderboard wins it on the 99.7% of the metric that a
video-level classifier could also produce.

**(b) $\theta$ does not rank localization.** $\rho_s(\theta,B|z)=0.218$, 95\%
CI $[-0.44,0.72]$; $\rho_s(\theta,B)=0.418$, CI $[-0.24,0.81]$; and even
$\rho_s(\theta,A)=0.191$, CI $[-0.46,0.71]$, so within this architecture family
the $\theta$ ordering is not explained by the scene-level term either. One
sensitivity belongs here, because a rank correlation over eleven points is
easily moved by a single arm: the placebo of (d) is a constructed outlier, and
recomputing without it gives $\rho_s(\theta,B|z)=0.239$ and
$\rho_s(\theta,B)=0.294$ at $n=10$ --- the first rises, the second falls, and
neither reaches significance. The
intervals are wide --- eleven points cannot exclude a moderate positive
association, so we read these correlations as *inconclusive rather than as
evidence of independence*, and we rest the claim on the decomposition instead:
the localization term is 0.31--0.33% of $\theta$ by construction, which is a
statement about the metric and not about eleven draws. Meanwhile the two
localization measures agree with each other:
$\rho_s(B,B|z)=0.800$ (CI $[0.39,0.95]$, $p=0.003$). The leaderboard and the
localization ranking are not the same ranking.

**(c) The localization spread is large --- and invisible.** $B|z$ spans 6.9 points
(0.6545 to 0.7230), four times the $\theta$ spread (1.8 points). At the exchange rate
that entire spread is worth $0.00388\times0.0685=0.00027$ AUC --- one eighteenth of
the seed-to-seed noise (Section~\ref{sec:resolution}). The community's practice of comparing $\theta$ to the
third decimal therefore cannot see a difference that is, in localization terms,
an order of magnitude larger than anything it does see.

**(d) A placebo arm is not last.** The eleventh row is a deliberately corrupted
model: the motion channels were computed and then shuffled during training, so
the feature is noise. It has the *lowest* $\theta$ (0.8616) --- as intended, it is
visibly broken --- but it ranks **4th of 11** by $B|z$ (0.6786), ahead of six
honest arms; by the unadjusted $B$ the same arm ranks 8th, so it is the
stratified column that lifts it. Whatever $B|z$ is measuring, it is not "model quality" in the sense
the leaderboard means; and any audit that promoted it would be promoting a
corrupted run. This is a caution about single-number localization statistics
(including ours) that we take seriously in Section~\ref{sec:infer-practice} and Section~\ref{sec:limits}.

### 6.3 Degeneracy in practice {#sec:degen}

\begin{table}[t]
\centering
\caption{Constant-within-video scorers against frame-level scorers on the same
cached CLIP features and training protocol (UCF-Crime, linear heads, top-$k$ MIL
for the frame models). The last column is the *unweighted* mean of the per-video
within-video AUC, which coincides with the pair-weighted $B$ for the first three
rows and differs for DSANet, whose videos carry unequal numbers of pairs.}
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
\end{table}

The constant scorer keeps **99.7% of the identically-trained frame-level model's
$\theta$** (0.8234 vs 0.8257) and **93.9% of the published DSANet checkpoint**
(92.1% of the best number reported in the literature), while its localization
term is *exactly* chance (B = 0.5000, all within-video pairs tied). Note also
that the constant scorer has a *higher* A than the frame model (0.9766 vs
0.9571).

Theorem 2 predicts the shape of this table, and it also says how to read the
0.0023 that separates the two scorers --- which is *not* the price of
localization. Term by term,
$\omega_A\Delta A=0.6317\times(-0.0195)=-0.0123$,
$\omega_B\Delta B=0.00388\times(+0.1571)=+0.0006$ and
$\omega_C\Delta C=0.3645\times(+0.0385)=+0.0140$: the localization term
contributes six tenths of a thousandth, and the headline difference is the near
cancellation of a twelve-point loss in the scene term against a fourteen-point
gain in the severity term (the implied $C$ is 0.561 for the video-level scorer
and 0.600 for the frame model). The frame model is a better localizer by 15.7
points of $B$ and is still separated from a scorer that cannot localize at all
by 0.0023 of $\theta$ --- half of one seed standard deviation, and far below the
third-decimal reporting granularity the community uses to claim progress. What
the 0.0023 measures is a cancellation, not the value of localization.

### 6.4 Resolution: what the metric can and cannot represent {#sec:resolution}

Three base checkpoints that differ only in seed --- rows 1, 3 and 8 of
Table~\ref{tab:decomp-ucf} --- give $\theta=0.876\pm0.005$, $B=0.736\pm0.018$ and
$B|z=0.686\pm0.027$ (mean $\pm$ one standard deviation over the three runs).
Localization is therefore three to six times as noisy as the headline number,
and that noise is invisible in it.

Two consequences, both quantitative.

First, by Theorem 3 the standard error of a $\Delta B$ imputed from a single-seed $\Delta\theta$
is $0.0049/0.00388=1.26$ --- 2.5 times the full parameter range of B. Nothing
about localization can be read off an AUC comparison at the precision this
literature reports.

Second, the *attainable* range: since B $\in$ [0.5, 1], a localization improvement
that went from chance to perfect would move $\theta$ by $\omega_B/2=0.0019$, which is 0.4
of the seed standard deviation. The entire observed localization spread across
eleven checkpoints (Section~\ref{sec:decomp}) is worth 0.06 seed-$\sigma$. **The reported metric is
strictly less sensitive to localization than the training run is to its random
seed.** This is not a claim about current methods; it is a property of the split
and the statistic.

The same computation gives the planning number for anyone who wants to detect a
localization difference with the standard metric: to certify $\Delta B=0.01$ with a
two-standard-error margin you need $\sigma(\Delta\theta)\le 1.9\cdot10^{-5}$ --- a
standard error 253 times smaller than the observed run-to-run noise, which at
$\sigma\propto k^{-1/2}$ would take roughly $64{,}000$ seed replicates and is
unattainable in this
literature (most papers report one run). Conclusion: measure B.

### 6.5 Motion behaves as a mediator, not a confounder {#sec:mediation}

We run the three interventions of Section~\ref{sec:motion}.

*Observational stratification* (Table~\ref{tab:decomp-ucf}, column $B|z$): $D = B - B|z$ is 3.45--5.77
points across the eleven checkpoints, median 4.92. Two normalisations of that gap
circulate and they are different numbers, so we fix both here. Relative to the
remaining headroom, $\pi_{\mathrm{mot}}=D/(1-B|z)$, the median is 0.149
(Section~\ref{sec:bound}); relative to the distance from chance,
$D/(B-1/2)$, the median is 0.220. It is the second that the sentence
"roughly **5 of the 25 points that separate the model's within-video AUC from
chance are motion-mediated**" expresses --- 4.92 points of a 25-point span ---
and only the first appears in the tables. A large share either way, but a
minority.

*Intervention on training.* Adding $\gamma\,\mathrm{mean}_v\mathrm{corr}^2(S_v,Z_v)$ where $Z$ is the
5-frame motion energy, on the linear probe:

\begin{table}[t]
\centering
\caption{The three interventions on the nuisance.  Top: an orthogonality
penalty against the true motion statistic, three seeds per row, reported as
their mean.  Bottom: the same penalty at $\gamma=1$ against real versus
shuffled motion.  The penalty does decorrelate the score from motion, and it
lowers the localization estimate rather than raising it.  The $\gamma=1$
real-motion run is one run and appears in both blocks, so its four numbers are
stated identically twice.}
\label{tab:interventions}
\begin{tabular}{@{}lrrrr@{}}
\toprule
condition & $\theta$ & $B$ & $B|z$ & $\mathrm{corr}$ \\
\midrule
\multicolumn{5}{@{}l}{\emph{training-time orthogonalisation (mean of three seeds)}} \\
$\gamma=0$ (baseline) & 0.8352 & 0.6805 & 0.6429 & 0.163 \\
$\gamma=0.1$          & 0.8331 & 0.6772 & 0.6417 & 0.154 \\
$\gamma=0.3$          & 0.8300 & 0.6744 & 0.6426 & 0.144 \\
$\gamma=1.0$          & 0.7980 & 0.6236 & 0.6043 & 0.096 \\
\midrule
\multicolumn{5}{@{}l}{\emph{placebo: $\gamma=1$ against shuffled motion}} \\
baseline ($\gamma=0$)   & 0.8352 & 0.6805 & 0.6429 & 0.163 \\
$\gamma=1$, real motion & 0.7980 & 0.6236 & 0.6043 & 0.096 \\
$\gamma=1$, shuffled     & 0.8362 & 0.6891 & 0.6480 & 0.169 \\
\bottomrule
\end{tabular}
\end{table}

The placebo is inert --- it is statistically indistinguishable from the baseline
(0.6891 vs 0.6805 for B; 0.6480 vs 0.6429 for $B|z$) and it does *not* reduce the
correlation with real motion (0.169). The degradation under the real penalty is
therefore specific to real motion structure rather than to regularisation as
such --- though we note again that a nuisance permuted across videos is nearly
uncorrelated with the score by construction, so the placebo is a weak control
and its inertness is close to guaranteed. On the mediator reading --- the
annotated events are largely motion-defined, so motion lies on the path from
anomaly to score --- adjusting for it is over-adjustment; on the confounder
reading, $B|z$ would be the honest estimand. We report both and return to the
limits of this verdict in Section~\ref{sec:motion}.

*Post-hoc residualisation on real checkpoints.* Cross-fitted subtraction of a
ridge fit of the score on four nuisance coordinates --- the normalised time
position $t/n_v$, the instantaneous motion $\lVert\Delta x_t\rVert$, its 5-frame
moving average $\overline{\lVert\Delta x_t\rVert}$ and the feature norm
$\lVert x_t\rVert$ --- with folds assigned by video parity, on base s789 (intervals: 500 video-cluster bootstrap replicates):

\begin{table}[t]
\centering
\small
\caption{Post-hoc residualisation on two checkpoints. The score is regressed on
four nuisance coordinates, the fit is removed with video-parity cross-fitting,
and the report is recomputed. Every variant lowers both localization measures on
both checkpoints. Video-cluster bootstrap intervals (500 replicates) for the raw
rows are $B\in[0.684,0.836]$ and $B|z\in[0.647,0.791]$ on base s789, and
$[0.681,0.819]$, $[0.656,0.774]$ on res s234. The $E$ column is the E-value of
$B|z$, not of $B$.}
\label{tab:resid}
\begin{tabular}{@{}lrrrr@{}}
\toprule
variant & $\theta$ & $B$ & $B|z$ & $E$ \\
\midrule
\multicolumn{5}{@{}l}{\emph{base s789 (score--motion correlation 0.141)}} \\
raw                          & 0.8772 & 0.7558 & 0.7166 & 4.50 \\
residualised, all frames     & 0.8600 & 0.7356 & 0.7099 & 4.33 \\
residualised, control frames & 0.8668 & 0.7392 & 0.7026 & 4.16 \\
\midrule
\multicolumn{5}{@{}l}{\emph{res s234 (score--motion correlation 0.328)}} \\
raw                          & 0.8682 & 0.7467 & 0.7093 & 4.31 \\
residualised, all frames     & 0.8493 & 0.7028 & 0.7001 & -- \\
residualised, control frames & 0.8532 & 0.7031 & 0.6915 & -- \\
\bottomrule
\end{tabular}
\end{table}

Every variant lowers both localization measures on both checkpoints --- by 1.7--4.4
points for B and 0.7--1.8 for $B|z$ --- which the mediator diagnosis predicts
(the confounder reading predicts the same sign, Section~\ref{sec:motion}). They also cost 1.0--1.9 points
of $\theta$, so a paper selling them as a de-biasing method would be selling a method
that lowers the very quantity it claims to improve.

*Three independent lines of evidence for the same conclusion.* (i) A
zero-training motion statistic (5-frame moving average of $\lVert\Delta x_t\rVert$) reaches
$B=0.6552$ and $B|z=0.5465$: 86% of a trained probe's excess over chance is
reproduced by motion, and all of it is absorbed by stratifying on motion. That
number is the sharpest limit on the construct and we do not soften it: in the
only block that measures localization, a training-free motion statistic
reproduces most of what a trained frame-level probe reports. It is why the column
is described throughout as *within-video ranking concordance* --- what $B$
measures by construction --- rather than as localization ability, and why the
86% is a statement about the construct rather than a defect of the estimator. (ii)
Trained checkpoints retain a within-video score--motion correlation of
0.12--0.33 (mean 0.18 across eleven; Table~\ref{tab:decomp-ucf}) that cannot be removed without
removing the signal. (iii) The one arm with the *highest* score--motion
correlation (res s234, 0.328) is also the third best by $B|z$ (0.7093), so the
correlation is not monotonically a symptom of "detecting motion instead of
anomalies".

### 6.6 The audit: what happens when the leaderboard is re-sorted {#sec:audit}

Ranking the eleven checkpoints by $B|z$ instead of $\theta$ (Section~\ref{sec:decomp}, last two columns)
changes the conclusion in four ways: (i) the $\theta$-winner (base s234, 0.8800) falls to
6th; (ii) a run ranked 10th by $\theta$ (res s234, 0.8682) rises to 3rd; (iii) the
checkpoint with the best *scene* term (attr s234, A = 0.9788, rank 5 by $\theta$) is
last by localization (11th of 11); and (iv) the placebo arm ranks 4th.

The rank correlation is 0.22 with a 95\% interval that includes zero
([-0.44, 0.72], p = 0.52). Eleven checkpoints cannot demonstrate independence,
and we do not claim they do: the audit shows that the ordering by
$\theta$ is not an ordering by the localization term *of the same metric*
(0.31--0.33% of it), while the rank evidence alone is merely inconclusive --- on
this family of models, which is the only family we can compare *architectures*
within at fixed features and protocol.

The audit is also where we owe the reader the sharpest limit of the paper, so we
state it here rather than in a footnote. A paired cluster bootstrap over videos
(the protocol of Section~\ref{sec:resolution}, 2,000 resamples, the *same*
resample applied to every arm so that the differences are paired) gives a
cluster standard deviation of 0.037--0.053 on $B$ and 0.031--0.048 on $B|z$ for a
single checkpoint. The largest between-arm gap that the family contains ---
motion6 s789 against attr s234, $B=0.7576$ vs $0.7080$ --- is $\Delta B=0.0495$
with a 95\% interval of $[-0.005,+0.095]$: it includes zero. The top two arms by
$B$ swap order in 51.2\% of resamples (51.3\% by $B|z$), i.e. the ordering of the
best checkpoints is a coin toss at this sample size, and arms that differ only in
seed partially share their variance (a paired $\Delta B$ between neighbouring
arms has a standard deviation near 0.011) without becoming resolvable.

So the audit should be read as a statement about *sets*, not about ranks: the
$\theta$ ordering does not carry the localization ordering, but neither does the
$B$ ordering carry a certification that one arm localizes better than another by
four decimals. What the eleven-arm family supports is (i) the decomposition and
its weights, (ii) the *magnitude* of the gap between the two rankings, and (iii)
the demonstration that a re-sort is possible at all; it does not support a new
leaderboard of its own, and we do not offer one. This is exactly why the
recommendation of Section~\ref{sec:protocol} is a third column with an interval,
not a new headline. Adding the released VadCLIP weights as a twelfth row changes
none of this: the gap, its interval and the flip rate are unchanged
($\Delta B=0.0495$ over $[-0.005,+0.095]$, 51.4\% flips), and the released
weights sit tenth of twelve by $B$ --- inside the band, below its top.

We are careful about what this does and does not show. It does not show that the
published ordering is wrong: A and C are legitimate quantities (a detector that
separates anomalous from normal *scenes* does something useful, and C measures
severity ranking). It shows that the ordering is not an ordering *by
localization*, while the papers read it as one. The remedy is not to replace the
metric but to report the missing column.

### 6.7 Criterion validity: do the extra columns predict detection mAP? {#sec:criterion}

A re-sorting argument is only as good as its localization columns, and a reviewer is
entitled to ask what $B$ predicts that anybody outside this paper cares about.
The natural outside criterion is the detection metric that the same papers
report, so we compute two of them, on every checkpoint and from the same
per-frame scores that produce $\theta$ and $B$, rather than quoting numbers from
training logs. The first is TLoC-style detection mAP at IoU 0.5 under the
evaluation routine the UCF-Crime detection literature shares (segment mAP,
$\mathrm{excludeNormal}=\mathrm{False}$, IoU $0.1$--$0.5$, fourteen classes,
per-class proposals from a relative score threshold, $0.7\times$ class score
added to the proposal score, NMS at 0.6). We run that reference code as
published and, in the same process, an independent re-derivation of the same
algorithm written for this paper (vectorised run extraction, its own
non-maximum suppression, its own greedy matcher). The two agree exactly: the
largest absolute deviation over all $5\times11$ cells is $0.0$. This column is
therefore the published protocol and not our variant of it. The second
implementation is still reported, because it does not inherit the reference
code's per-class score gating and is short enough to check by hand: nine
relative thresholds per video, maximal runs above each threshold as proposals, a
proposal's confidence taken as the mean score inside the run, greedy matching at
IoU 0.5 and an interpolated precision-recall integral.


The detection numbers are small in absolute terms, and the reason is a
definition rather than a defect. The papers on this benchmark publish
frame-level AUC and average precision, not detection mAP: for the released
checkpoint we reproduce the frame-level AUC its own paper reports on the same
test split ($0.8808$ against $88.02$), so the scores are the model's and the
pipeline that turns them into numbers is calibrated at that level. Under the
detection protocol the same checkpoint scores $2.95$ mAP@0.5, and our eleven
arms span $2.44$--$5.82$; that metric credits one segment per video above
everything else, and weakly supervised models trained on frame-level labels are
not optimised for it. We therefore use the detection columns as *rank* criteria
inside one fixed protocol, never as absolute numbers comparable to the
literature, and we rest part of the reading on the agreement of the two
implementations: $\rho_s=0.57$ between them across twelve checkpoints. We also
correlate against the frame-level average precision itself, since that is the
number these papers do report: it lies between $0.294$ and $0.374$ across the
eleven self-trained checkpoints and is $0.339$ for the released anchor.

What the recomputation does not buy is a second measurement. The two detection
implementations share our scores and our labels, and $\rho_s=0.57$ is weak
agreement between two pieces of code, not two experiments. Nor is the ground
truth better than it is: the file the protocol reads stores segments at 16-frame
granularity and marks the 150 normal test videos with a placeholder class label,
so any proposal inside them is counted as a false positive by construction. What
survives all of that is a rank comparison inside one fixed protocol, which is
what the next paragraph reports. Nothing here establishes that detection mAP
*is* the right criterion for $B$, and no external criterion in this paper does.


\begin{table}[t]
\centering
\caption{Criterion validity: the reported number, the two localization columns,
and two independently implemented detection metrics, for the eleven
self-trained checkpoints and the released VadCLIP weights (last row, marked
$\dagger$). All three metric columns are percentages produced under one fixed
protocol each and are used as rank criteria only. The mAP@0.5 and avg mAP
columns are the benchmark's reference protocol, whose values an independent
re-derivation reproduces exactly; the last column is our own shorter
implementation with a different proposal rule.}
\label{tab:criterion}
\footnotesize
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
checkpoint & $\theta$ & $B$ & $B|z$ & mAP@0.5 & avg mAP & AP@0.5 \\
\midrule
base s234        & 0.8800 & 0.7270 & 0.6729 & 5.68 & 10.10 & 4.46 \\
motion6 s789     & 0.8778 & 0.7576 & 0.7230 & 4.61 & 12.46 & 2.41 \\
base s789        & 0.8772 & 0.7558 & 0.7166 & 5.18 & 12.84 & 3.94 \\
cons0 s234       & 0.8759 & 0.7235 & 0.6744 & 5.46 & 11.16 & 2.51 \\
attr s234        & 0.8724 & 0.7080 & 0.6545 & 5.82 & 13.03 & 2.79 \\
core14txt s234   & 0.8717 & 0.7169 & 0.6728 & 5.03 & 11.88 & 4.24 \\
motion6 s555     & 0.8717 & 0.7180 & 0.6604 & 2.88 & 10.15 & 2.86 \\
base s555        & 0.8705 & 0.7236 & 0.6688 & 4.12 & 11.30 & 2.53 \\
core8g1 s234     & 0.8684 & 0.7124 & 0.6621 & 4.05 & 10.64 & 2.45 \\
res s234         & 0.8682 & 0.7467 & 0.7093 & 4.63 & 9.83 & 1.89 \\
motion6-shuf s234 & 0.8616 & 0.7171 & 0.6786 & 2.44 & 7.17 & 1.85 \\
VadCLIP$^\dagger$ & 0.8560 & 0.7157 & 0.6713 & 2.95 & 7.36 & 1.18 \\
\bottomrule
\end{tabular}
\end{table}

Three correlations settle the question for each metric, all on twelve points, all
reported with the Fisher-$z$ interval that twelve points can support, and all
with a two-sided permutation $p$ over $10^5$ sampled orderings of the criterion
(Table~\ref{tab:criterion}). For the detection protocol,
$\rho_s(\theta,\text{mAP@0.5})=0.71$, 95\% CI $[0.24,0.91]$, $p=0.012$;
$\rho_s(B,\text{mAP@0.5})=0.15$, CI $[-0.47,0.66]$, $p=0.65$;
$\rho_s(B|z,\text{mAP@0.5})=0.08$, CI $[-0.52,0.62]$, $p=0.82$. The independent
implementation returns the same pattern with different numbers:
$0.69\ [0.20,0.91]$ for $\theta$ ($p=0.011$), $0.11\ [-0.49,0.64]$ for $B$
($p=0.74$) and $-0.20\ [-0.70,0.42]$ for $B|z$ ($p=0.54$); the average-mAP column
agrees again ($0.63\ [0.09,0.88]$ for $\theta$, $0.10$ for $B$).

The frame-level average precision behaves differently, and we report the
difference rather than the half of it that suits the argument. On all twelve
points it tracks neither column ($\rho_s(\theta,\cdot)=0.45$, $p=0.14$;
$\rho_s(B,\cdot)=0.33$, CI $[-0.30,0.76]$, $p=0.29$), but that is the released
anchor's doing: its average precision is unremarkable ($0.339$) while its
$\theta$ is the lowest in the table. Drop the anchor and the eleven self-trained
points give $\rho_s(\theta,\cdot)=0.74$, CI $[0.24,0.93]$, $p=0.013$, and
$\rho_s(B,\cdot)=0.49$, CI $[-0.15,0.84]$, $p=0.13$ --- an interval that admits
zero, a moderate association and a strong one, so this criterion neither
certifies $B$ nor excludes it.

Two cautions on the partial correlations, which we computed and do not rest on.
Partialling $\theta$ out conditions on a variable that contains $B$
($\theta=\omega_A A+\omega_B B+\omega_C C$), which mechanically biases the
partial association downward; and at $n=12$ no partial association here is
distinguishable from zero ($-0.32$, permutation $p=0.31$, under the detection
protocol; $+0.14$, $p=0.66$, for the frame-level criterion). The clean
partialling --- on $A$ and $C$, the other two blocks --- is the analysis a
follow-up should run.


**The honest reading is that mAP, as we can measure it, does not supply the
missing column either.**
It tracks $\theta$ --- which is what the decomposition predicts, since 99.7% of
the pairs driving any frame-level score are cross-video pairs and segment AP
over $16\times$ repeated frame scores is dominated by them --- and it shows no
association with $B$ beyond what $\theta$ already carries, with intervals wide
enough that a moderate association is not excluded.
We did not expect this when we designed the check; it is a negative result for
the most obvious way to validate $B$, and we report it as one. Three things
follow, and they constrain the claims of this paper.

1. $B$ is not offered as a replacement headline number. We cannot certify it
   against the field's own localization metric, so promoting it to the
   leaderboard would be a stronger claim than these data support.
2. Neither field metric, as we can measure it, orders the axis the theorems say
   is missing.
   Frame-level AUC sees 0.39% of within-video pairs; detection mAP, which one
   would expect to see them, tracks $\theta$ instead; and the frame-level
   average precision, the number the papers publish, leaves the question open at
   this sample size rather than answering it. Certifying or refuting
   within-video ordering against an outside criterion remains open.
3. The remedy remains reporting, not replacing: three columns instead of two,
   with the third one explicitly labelled as a diagnostic whose external
   validation is an open problem (Section~\ref{sec:limits}).

**The twelfth row is an external anchor.** The released VadCLIP weights, scored
by this paper's suite rather than by its own authors, reproduce the published
frame-level AUC under the repository's own $16\times$ convention ($0.8808$
against the $0.8802$ the paper reports) and land inside the band our eleven
arms span on the snippet grid used throughout ($B\in[0.708,0.758]$,
$B|z\in[0.655,0.723]$). The two AUC conventions, the unit-of-analysis
difference, and the mAP comparison are detailed in Appendix~\ref{sec:appD}.

One numerical caution belongs here rather than in a footnote. Twelve
checkpoints, a mAP range of 1.2--5.8% and a between-seed mAP spread comparable
to the between-architecture spread mean that these rank correlations are
fragile: the intervals above are wide by construction, only the $\theta$
association survives a two-sided test, and nothing in this table should be read
to the second decimal.

### 6.8 Inference in practice {#sec:infer-practice}

*Clustered intervals.* The bootstrap of Section~\ref{sec:inference} (2,000 replicates, videos as
clusters) gives $B=0.7558\ [0.6844,0.8364]$ and $B|z=0.7166\ [0.6467,0.7907]$
for the base checkpoint: a $\pm$ 0.07 window for a quantity estimated from 140
anomalous videos.

The same exercise is instructive for the *reported* number. On identical scores,
a frame-level i.i.d. bootstrap --- the assumption behind a DeLong-style interval (DeLong et al., 1988) ---
gives $\theta=0.8772\ [0.8734,0.8808]$, a width of 0.0075, while resampling
*videos* gives $[0.8237,0.9152]$, a width of 0.0916: **12.3 times wider**. The
precision the field attributes to its headline number is therefore partly an
artefact of treating 69,368 frames as 69,368 independent observations when they
are 290. This is a milder version of the error this paper is about, and it has
the same remedy: state the cluster. ($B|z$'s clustered interval, 0.144, is in
fact *wider* than $\theta$'s 0.092: conditioning on motion keeps only same-bin
pairs, and the discarded cross-bin pairs are the honest price of the stratified
estimator.)

*Between-seed noise.* $\mathrm{sd}(B|z)=0.0267$ across three seeds of the same
configuration, i.e. two fifths of the full 0.07 spread of Table~\ref{tab:decomp-ucf}. Any single-seed
localization claim below about 0.05 on UCF-Crime is not established, and papers that
report one run should say so.

*Overlap.* 94.3% of positive units share a motion quintile with at least one
control unit in the same video (per-video mean 0.941), so the stratified
estimator is well supported; the remaining 5.7% come from videos whose
anomalous segment is uniformly more dynamic than its background --- exactly the
cases where the motion question is unresolved.

*E-values.* $E(B)=5.6$ and $E(B|z)=4.5$ for the base checkpoint: an
unmeasured within-video confounder would need to be associated with both the
status and the score by a risk ratio of 4.5 to explain the localization contrast
away. Given that we have measured and ruled out the dominant candidate (motion),
this is a defensible statement of robustness --- and the number to report instead
of the words "we controlled for scene bias".

### 6.9 XD-Violence {#sec:xd}

XD-Violence is the harder test of the framework: it has an order of magnitude
more frames, denser annotation (537,882 positive frames of 2,331,296), six
violence categories, and the field reports it with AP/mAP over IoU thresholds rather than
micro-AUC. Four DSANet checkpoints were available (three seeds of the base
configuration and one motion-injection variant).

\begin{table}[t]
\centering
\small
\caption{The decomposition on XD-Violence (800 videos, 2,331,296 scored
frames, 537,882 positive frames). The pooled number agrees with the
repository's own frame-level AUC to six decimals.}
\label{tab:decomp-xd}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
checkpoint & $\theta$ & $A$ & $B$ & $B|z$ & $C$ & $\mathrm{corr}$ & rank $\theta$ & rank $B|z$ \\
\midrule
base s234    & 0.9536 & 0.9966 & 0.8466 & 0.8223 & 0.8540 & 0.192 & 1 & 1 \\
base s789    & 0.9449 & 0.9962 & 0.8281 & 0.8000 & 0.8258 & 0.200 & 2 & 3 \\
base s555    & 0.9437 & 0.9969 & 0.8419 & 0.8160 & 0.8204 & 0.203 & 3 & 2 \\
motion3 s789 & 0.9368 & 0.9920 & 0.8046 & 0.7774 & 0.8090 & 0.201 & 4 & 4 \\
\bottomrule
\end{tabular}
\end{table}

The weights are $\omega=(0.6985,0.00071,0.3008)$: **0.07% of the metric's pairs
are within-video**, and the localization share of $\theta$ is $\omega_B B=0.0006$, i.e.
*less* than on UCF-Crime even though XD-Violence annotates three times as large a
fraction of its frames. The reason is arithmetic rather than conceptual: with
800 videos and 1.25M frames in normal videos, the number of cross-video pairs grows
much faster than the within-video ones. A benchmark does not become
localization-sensitive by annotating more frames; it becomes so by having fewer
or shorter videos per split. This is a design lesson that costs nothing to
apply and that we make concrete in Section~\ref{sec:discussion}.

Two differences from UCF-Crime are worth recording honestly. First, $B$ itself is
much higher ($0.80$--$0.85$ vs $0.71$--$0.76$): XD-Violence is dense --- violence
occupies
large portions of most anomalous videos --- so a within-video contrast has more
signal to find, and the localization picture on XD-Violence is genuinely better
than on UCF-Crime. Second, on these four checkpoints the $\theta$ and $B|z$ orderings
largely agree ($\rho_s=0.8$, p = 0.2, n = 4), unlike the eleven-checkpoint
UCF-Crime family of Section~\ref{sec:decomp}. With n = 4 we draw no conclusion from either; we note
that the two settings differ in what they vary (design vs seed), and that our
central claims --- the $\omega$ weights, the exchange rate, the degeneracy, the
irreversibility --- are properties of the split and the statistic, not of the
ordering, so they transfer unchanged.

The motion-mediated share is smaller on XD-Violence ($D = B - B|z$ between 2.4
and 2.9 points, vs 3.5--5.8 on UCF-Crime) while the score--motion correlation is
similar (0.19--0.20). Motion matters, again, but is not the whole story: after
stratifying on motion the contrast is still 28--32 points above chance, so on
XD-Violence most of the within-video separation is *not* motion-mediated. On
UCF-Crime the corresponding figure is 15--22 points, of which motion accounts for
roughly a quarter.

The seed spread is comparable to UCF-Crime ($\mathrm{sd}(\theta)=0.0055$,
$\mathrm{sd}(B)=0.0095$, $\mathrm{sd}(B|z)=0.0114$ over three seeds), so the resolution
argument of Section~\ref{sec:resolution} holds with the same arithmetic: the localization limits are
$\omega_B/2=0.00036$ for motion-independent perfect localization, and one reported
XD point is worth 1,408 within-video points.

### 6.10 Summary of the empirical claims {#sec:summary}

The empirical content of the paper reduces to eight numbers, each checkable
from the released scores: the within-video pair share ($\omega_B=0.00388$ UCF /
0.00071 XD); the 99.7%/93.9% degeneracy figures; the 6.9-point localization
spread worth 0.00027 AUC against seed noise of 0.0049; the inconclusive
$\rho_s(\theta,B|z)=0.22$; the mAP criterion failure ($\rho_s=0.71$ with
$\theta$ against $0.15$ with $B$); the 5-of-25-point motion share; the
nuisance-share tolerance $2B-1=0.42$--$0.52$ (UCF) / $0.61$--$0.69$ (XD); and
the unresolvability of the eleven-arm spread as a ranking
(Section~\ref{sec:audit}). Section~\ref{sec:criterion} is the one we expected
to come out differently.

## 7. Discussion {#sec:discussion}

### 7.1 The answer in one sentence, and what follows from it {#sec:answer}

Theorems 1--4, Proposition 5 and Tables~\ref{tab:decomp-ucf}--\ref{tab:decomp-xd} give a single sentence answer: *frame-level micro-AUC
is a scene-level severity ranking with a small localization component attached*,
and the size of that component is computable in advance from the data. On
UCF-Crime it is 0.39% of the metric's pairs and 0.31--0.33% of its value; on
XD-Violence 0.07% and 0.06%. Everything else --- the 150 entirely normal test
videos (51.7% of the split), the ranking of anomalous videos by severity, the separation
of scenes --- is measured by the cross-video blocks $A$ and $C$.

This reframes several familiar results. A model that "improves AUC by 1 point"
has, at most, moved the localization term by 1/258 of a point (UCF) or 1/1408
(XD), and in practice has moved it by nothing measurable (Section~\ref{sec:resolution}). A model that
loses 1 point of AUC may have *gained* localization (res s234 in Table~\ref{tab:decomp-ucf} does
exactly this: rank 10 by $\theta$, rank 3 by $B|z$). And the widely observed fact that
video-level classifiers score highly on these benchmarks is not a bug in those
classifiers --- it is the metric doing what its weights say.

None of this makes the metric useless. For *triage* --- deciding which video to
send to a human --- a scene-level severity ranking is close to the right quantity,
and A is a defensible measure of it. The error is one of naming: the field
reports a triage metric and reads it as a localization metric.

One claim is withdrawn here rather than in a footnote. It would be convenient if
the localization column were also a *better* number to optimize, and
Section~\ref{sec:criterion} shows that the obvious route to that claim fails:
detection mAP, the field's own localization metric, tracks $\theta$ and not $B$.
So the decomposition buys a *diagnosis* --- what the reported number is made of
--- and not a replacement. The difference matters for how this work should be
used: to read a leaderboard correctly, not to win one.

### 7.2 Designing a split that can measure localization {#sec:design}

The weights are computable before a single model is trained. With $n_v$ units in
video $v$ and $N_{\mathrm{pos}}$, $N_{\mathrm{neg}}$ the numbers of positive and of
negative units in the split,
$\omega_B\approx\sum_v a_v b_v/(N_{\mathrm{pos}}N_{\mathrm{neg}})$, so a split measures localization
when its anomalous videos are *short* and *few* relative to the number of
background frames. XD-Violence annotates 23% of its frames and still gets
$\omega_B=0.07\%$, because it has 800 videos and 1.25M frames in normal videos; annotation
density is not the lever, video count is.

Three practical consequences.

1. **Report the census.** One line per benchmark: $\omega_A,\omega_B,\omega_C$ and the
   exchange rate. It takes seconds and it tells every reader how much of the
   reported number can possibly be about localization.
2. **Report the exchange rate next to every gain in $\theta$.** On UCF-Crime a
   claimed gain of 0.9 points of AUC cannot be a localization claim: converted
   at the exchange rate it demands $\Delta B=0.009/0.00388=2.3$ --- more than
   four times the full range of $B$ (0.5) and nearly ten times the remaining
   headroom $1-B\approx0.24$ --- while the whole attainable localization range
   is worth only 0.0019 of $\theta$, a fifth of the claimed gain. A paper
   making such a claim should say which block moved.
3. **If localization is the claim, report $B$.** No new statistic needs to be
   invented: $B$ is simply micro-AUC with the pair population reweighted to the
   within-video pairs, and it is the estimand of Theorem 1's second term. A
   benchmark that wanted to be localization-sensitive could reweight once and be
   done --- at the cost of a wider interval (Section~\ref{sec:infer-practice}), which is the honest price.

### 7.3 What our own failures say about method research {#sec:failures}

Every intervention we tried in order to *raise* the localization estimate failed:
motion orthogonalisation at training time ($\gamma$ scan, monotone degradation),
cross-fitted residualisation at test time ($-$ 1.7 to $-$ 4.4 points of B on the two
checkpoints we tried it on), and --- in a
companion study --- thirteen post-hoc debiasing variants including an oracle that
uses ground-truth distances ($-$ 0.0045 AP). Theorem 4 explains the pattern: the
weak label constrains a symmetric function of the scores inside an anomalous
video, so the *ordering* of most frames within a video is unconstrained by the
objective, and no amount of post-processing of a trained scorer's outputs can
create information the objective never supplied.

This suggests where method work should go, and where it cannot go. It cannot go
into loss reweighting, score calibration, or threshold tuning: those operate on
a quantity the label does not identify. It can go into *new evidence* --- object
tracks, audio, interactions between people, multi-scale temporal semantics --- that
creates within-video variation the label *can* constrain, because such evidence
makes the frame-level ordering identifiable through the top-k pool. That is a
research programme, not a loss term, and it is the one direction our negative
results leave open.

### 7.4 Relation to the 2026 audit literature {#sec:audits}

Two concurrent audits reach compatible empirical conclusions.
Song and Lee (arXiv:2608.21854) observe that within-video pairs are 0.071--0.388%
of all anomalous--normal pairs and that video-level classifiers can score
81.40--97.18% micro-AUROC with within-video AUC exactly 50%. Abdulaziz and
Bondarev (arXiv:2608.11985) show that pooled AUC does not reliably predict
within-video localization and that normal clips alone can separate videos by
acquisition attributes. Our Sections~\ref{sec:setup} and~\ref{sec:results} supply the algebra those observations are
instances of: the pair-count weights (Theorem 1) give the 0.071--0.388% directly
from the data, the constant-scorer construction (Theorem 2) gives the exact 50%,
and the video-level classifier built on it attains 0.93 of the state of the art on UCF-Crime,
and the inverse problem (Theorem 3) quantifies why a pooled difference cannot be
read as a localization difference. What we add beyond the audits is the exchange
rate, the identification analysis (what $B$ *is*, and why motion is a mediator
rather than a confounder), the estimators with clustered inference, and the
reporting protocol.

### 7.5 Concurrent work that proposes replacement metrics {#sec:concurrent}

A third line of work replaces the metric rather than decomposing it. Liu et al.
(arXiv:2505.19022) identify three limitations --- single-annotation bias, no reward
for early detection, and no way to measure scene overfitting --- and respond with
probabilistic AUC/AP over multi-round annotations, a latency-aware AP, and two
hard-normal benchmarks evaluated on ten systems. The difference in kind is worth
stating plainly: their remedies require *new annotation rounds and new data*, and
they change what is being measured; ours requires *neither*, because the identity
in Theorem 1 is a property of the metric the field already reports. The two are
complementary and can be combined --- a latency-aware or probabilistic AP would be
decomposed by the same pair-count argument, and the resulting $\omega_B$ would be
computed once from the annotation rounds. One caution from
Section~\ref{sec:criterion} applies to any such proposal, including ours: a
metric that is not itself an annotation-round construct must be validated
against an outside criterion, and when we tried that for $B$ the outside
criterion turned out to be a re-measurement of $\theta$. We therefore read this
line of work as evidence that the field accepts metric-level contributions, not
as competition for the same slot: they ask what we *should* report, we ask what
the reported number *already contains*.

### 7.6 Limitations and threats to validity {#sec:limits}

**Scope of the evidence: one architecture family, plus one anchor.** The
decomposition is exact and the weights are data properties, so Theorems 1--3
transfer to any split. The *empirical* claims --- the ranking disagreement of
Section~\ref{sec:decomp}, the size of the motion share, the seed noise --- are
measured on DSANet checkpoints on UCF-Crime and XD-Violence, plus the released
VadCLIP weights as a sixteenth checkpoint used as an external anchor rather
than as a member of the family (Section~\ref{sec:criterion}). The machine used
for these experiments has no outbound network access, so RTFM and other public
releases could not be fetched; the family (VadCLIP features, DSANet host, one
protocol) remains a single-model-family study --- eleven arms in which the
design is fixed while the architecture varies, which is exactly what the
decomposition needs, but it is not the published leaderboard, and one anchor
does not turn it into one. On XD-Violence the four checkpoints we have agree in
their $\theta$ and $B|z$ orderings, so the "leaderboard does not rank
localization" finding should be read as established for the UCF-Crime
eleven-checkpoint family (which spans design variants and includes a corrupted
control) and *not* as a universal law; the metric-level claims do not depend on
it. Even inside the family the spread is not resolvable *as a ranking*: the
largest gap has an interval that includes zero and the top-two ordering flips
in half the resamples (Section~\ref{sec:audit}). Re-scoring the remaining
published checkpoints would be a straightforward experiment for a group that
can reach them; it is not one we can run here, and we prefer to say so than to
generalise from eleven arms.

**Strained assumptions: annotation, unit scores, and the choice of nuisance.**
A4 fails to some degree in both benchmarks: annotations are segment-level and
coarse, and a companion study found that re-ordering frames by distance to the
annotated segment *reduces* AP, i.e. that part of what looks like "halo" error
is genuine anomaly outside the label. Under A4 failure, true anomalies are
counted as controls, so $B$ is biased *downward* and our estimates are
conservative; we do not attempt to correct for it, since doing so requires a
model of the annotation process that no benchmark currently supplies. A1 fails
for any model whose frame score is produced by attention across the video,
which is all transformer-based detectors including our host: $B$ is then a
statement about the scorer as a black box evaluated frame-wise, which is what
deployment does, but a claim about the *representation* would need an
intervention on the model's internal aggregation that we do not attempt.
Finally, $B|z$ conditions on motion only; camera motion, lighting changes, and
crowd density are plausible within-video nuisances we do not measure. Two
partial answers: the bound of Section~\ref{sec:bound} does not require the
nuisance to be named at all --- it charges an arbitrary share $\delta$ of the
pair mass and reports the interval --- and the E-value gives the confounding a
claim could survive. Both are bounds, and the E-value treats pairs as
independent and is therefore optimistic relative to the clustered interval.

**No external criterion for $B$.** Section~\ref{sec:criterion} is the limitation
we would most like to have answered and could not. Detection mAP at IoU 0.5,
the field's own localization metric, tracks $\theta$ ($\rho_s=0.71$,
permutation $p=0.012$) and not $B$ ($\rho_s=0.15$, $p=0.65$) under either of two
implementations, so $B$ has no established external criterion on these
benchmarks. The frame-level average precision, the number these papers actually
publish, is the one criterion that shows a positive association with $B$
($\rho_s=0.49$ over the eleven self-trained checkpoints), but with a $95\%$
interval of $[-0.15,0.84]$ and $n=11$ that association is not separable from
zero, and it weakens to $0.33$ once the released anchor enters. Two things limit
how far even the negative reading goes: the detection protocol is faithful ---
an independent re-derivation matches it exactly --- but not comparable to
published magnitudes, since the released weights score $2.95$ mAP@0.5 under it
while the papers report frame-level numbers, which we do reproduce ($0.8808$
against $88.02$); so all we have is a rank criterion, and a rank criterion over
twelve points supports only the negative statement that $B$ does not order
detection mAP. We therefore present $B$ as a *diagnostic of a construct the
current evaluation system does not report*, not as a validated replacement, and
we flag the missing criterion as the first thing a follow-up should build: a
split with enough within-video pairs to make segment-level localization a
dominant term, and re-annotated segments good enough to serve as a criterion.
Our own attempts (three interventions, four negative results,
Section~\ref{sec:failures}) suggest this is harder than it looks.

**An instrument, not a detector --- and not yet an AP theory.** We do not
propose a detector, and the theorems explain why a detector-shaped contribution
cannot settle the question this paper asks: a scorer that is constant inside
every video keeps 99.7% of the headline number while localizing at chance, so a
comparison of two detectors on that number carries no information about the
capability the field claims to measure. What we propose instead is an
instrument: an estimand, three estimators with their intervals and sensitivity
analysis, a partial-identification bound, and a reporting protocol --- each a
method in the ordinary sense, a procedure that turns labelled scores into a
claim that can be checked, and each released as reference code. The four
negative results of Section~\ref{sec:failures} are then not a gap in the paper
but its evidence: they are what happens when method work is built on the
unmeasured quantity. The exchange-rate theorem also does not apply to AP,
XD-Violence's headline statistic; empirically AP and AUC almost always agree in
*ordering* on every checkpoint we have (Section~\ref{sec:ap}), so the
qualitative conclusions carry over, but an AP version of Theorem 1 would need a
price-weighted decomposition that we leave open.

### 7.7 Deployment, ethics, and what a diagnostic can and cannot license {#sec:ethics}

Video anomaly detection is a surveillance technology, and the benchmarks we audit
are built from a corpus of real surveillance footage. Three things follow from
that, none of them comfortable.

*The cost of a false alarm is not in the metric.* Neither $\theta$, nor $B$, nor
$B|z$ contains a base rate. At the prevalence of anomalous footage in a deployed
queue --- which is far below the roughly 50:50 balance of a benchmark test split
--- a detector with high $\theta$ can still be wrong on most of the segments it
flags, and no column this paper proposes changes that. A system that selects
which video a human reviews should be evaluated with the operating point and the
review budget named, not with an AUC.

*The label defines the anomaly.* "Anomalous" in UCF-Crime and XD-Violence means
the categories those annotators chose, labelled at segment granularity by people
watching footage from particular places. This paper's own negative result is a
statement about that boundary: part of what a trained detector flags inside an
anomalous video lies outside the annotated segment, and we cannot tell from the
labels whether it is error or genuine anomaly. A claim that a scorer "localizes
anomalies well" is therefore always a claim about agreement with one annotation
process, and the population it was drawn from is not the population a deployed
system would face.

*An audit metric is a target once it is adopted.* $B$ is a pair-ranking
statistic, so it is directly optimisable: a pairwise ranking loss with pseudo-labels
--- the strategy the companion study tried and that failed under weak labels
(Section~\ref{sec:failures}) --- would improve $B$ without improving detection,
and the protocol of Section~\ref{sec:protocol} would then be measuring the
optimiser rather than the detector. This is the Goodhart problem (Strathern, 1997), and it applies
to our proposal as much as to AUC. The partial defence is structural: $B$ is an
offline diagnostic that needs unit-level ground truth, so it belongs to the
benchmark side and cannot be monitored in deployment; and it should be reported
next to an *extent* criterion (a TLoC-style detection measure, or average IoU
against the annotated segments) so that no single column can be improved alone.
We do not claim that defence is sufficient, only that naming it is the minimum a
new metric owes.

What a diagnostic of this kind can license is narrow, and we state it in the same
spirit as the rest of the paper: it can tell a reader how much of a reported gain
could possibly be about localization, and it can tell an author that a claimed
gain cannot be one. It cannot certify a detector, and it should not be cited as
if it did.

## 8. Conclusion {#sec:conclusion}

Frame-level micro-AUC, the standard headline number of weakly-supervised video
anomaly detection, is an exact weighted average of three pairwise contrasts, and
the weights are properties of the test split rather than of the method. On
UCF-Crime 0.39% of the pairs behind the number are within-video --- the only ones
that concern localization --- and on XD-Violence 0.07%. The exchange rate follows
mechanically: one reported point buys at most 0.0039 points of within-video AUC on
UCF-Crime and 0.0007 on XD-Violence. Because the localization term is bounded,
the entire attainable range of localization ability moves the reported number by
0.0019, well inside the seed-to-seed noise of 0.0049; no experiment reported at
the field's current precision can distinguish a detector that localizes from one
that cannot, and a scorer that is constant inside each video --- which localizes
nothing by construction --- retains 93.9% of the state of the art's score.

The weak label does not identify the within-video ordering, which is why
post-hoc corrections, including ours, fail; the motion nuisance that dominates
the discussion behaves as a mediator rather than a confounder --- adjusting for
it removes signal --- though we state plainly that our design separates the two
readings only partially. We therefore propose to *measure* the localization component
directly: an unadjusted within-video contrast, a motion-stratified version as a
mediator diagnostic, video-clustered intervals, and an E-value, reported
alongside the number the field already computes. The cost is one
evaluation pass; the benefit is that a paper can say what it actually
demonstrated.

## Appendix A: computing the census and the estimators {#sec:appA}

Everything in Section~\ref{sec:contrast} is arithmetic on the labelled scores of
one split. We state the recipe explicitly so that the census and the localization
columns can be reproduced from a score file.

*Census.* For an anomalous video $v$ let $a_v$ and $b_v$ denote its numbers of
anomalous and of normal units, $M$ the total number of anomalous units of the
split, $K$ the total number of units in normal videos, and $B_{\mathrm{tot}}$ the
total number of normal units inside anomalous videos. Then

\begin{equation}
n_A=MK,\qquad
n_B=\sum_{v\in\mathcal{V}_{\mathrm{an}}} a_v b_v,\qquad
n_C=\sum_{v\in\mathcal{V}_{\mathrm{an}}} a_v\,(B_{\mathrm{tot}}-b_v),
\end{equation}

\noindent and $\omega_Z=n_Z/(n_A+n_B+n_C)$ for $Z\in\{A,B,C\}$, where
$\mathcal{V}_{\mathrm{an}}$ is the set of anomalous videos of the split.

*Estimators.* With $U_v$ the Mann--Whitney count of video $v$ (one sort of its
scores, average ranks for ties),

\begin{equation}
\hat{B}_v=\frac{U_v}{a_v b_v},\qquad
\hat{B}=\frac{\sum_v a_v b_v\,\hat{B}_v}{\sum_v a_v b_v},
\end{equation}

\noindent the pair-weighted version reported throughout. The stratified variant
$\hat{B}|z$ replaces $U_v$ by the sum of the per-bin Mann--Whitney counts
inside video $v$, with the bins cut at the within-video quintiles of $z$. The
interval is a bootstrap over videos: resample the $V$ videos with replacement,
recompute $\hat{B}$ and $\hat{B}|z$ on the resampled set, and take the
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

## Appendix B: artefacts {#sec:appB}

We release the reference implementation of the report of
Section~\ref{sec:protocol}, together with the per-frame scores of the sixteen
checkpoints used here --- the score vector, the per-video lengths, the
frame-level ground truth, and the motion statistic --- which is sufficient to
reproduce every number in Section~\ref{sec:exp}. The archived per-run metric
files used for the verification of Section~\ref{sec:host} are included as well.
Evaluation of one checkpoint takes about 16 s on a single RTX A5000 (24 GB)
(the released VadCLIP weights take about 55 s, most of it feature extraction);
the controlled probes of Section~\ref{sec:mediation} train in 30--70 s per arm
on 4,000 cached clips.

## Appendix C: a residualised sensitivity check, and per-checkpoint bounds {#sec:appC}

### A residualised variant, as a sensitivity check only {#sec:resid}

Fitting the score on $z$ by ridge regression over the control units (those scored
below the video median) of one half of
the videos and subtracting the fitted value on the other half, with fold
assignment by video parity, gives $B^{\mathrm{res}}$. We report it because it is the
natural "de-biased" estimator and because it *fails*: on the base checkpoint it
lowers $B$ from 0.7558 to 0.7392 and $B|z$ from 0.7166 to 0.7026 (Section~\ref{sec:mediation}). Under
the mediator diagnosis that is the expected outcome, and it is the reason we do
not sell it as a method.

### Per-checkpoint partial identification {#sec:appc-bounds}

\begin{table}[t]
\centering
\caption{Partial identification of the within-video contrast for every
checkpoint.  $\pi_{\mathrm{mot}} = (B-B|z)/(1-B|z)$ is the
motion-mediated share of the headroom above the stratified estimator (a
mediation proportion, not a contamination share; Section~\ref{sec:bound});
$\delta^{*} = 2B-1$ is the contamination share at which the localization claim
dies; $\tau_{\mathrm{LB}}(0.2)$ is the lower endpoint of the identified interval
under 20\% contamination.}
\label{tab:partial}
\footnotesize
\begin{tabular}{@{}lrrrrr@{}}
\toprule
checkpoint & $B$ & $B|z$ & $\pi_{\mathrm{mot}}$ & $\delta^{*}$ & $\tau_{\mathrm{LB}}(0.2)$ \\
\midrule
UCF base s234          & 0.7270 & 0.6729 & 0.165 & 0.454 & 0.6587 \\
UCF motion6 s789       & 0.7576 & 0.7230 & 0.125 & 0.515 & 0.6970 \\
UCF base s789          & 0.7558 & 0.7166 & 0.138 & 0.512 & 0.6948 \\
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
\end{table}

## Appendix D: the released VadCLIP weights as an external anchor {#sec:appD}

The twelfth row of Table~\ref{tab:criterion} is the checkpoint released with
VadCLIP, scored by this paper's suite rather than by its own authors: weights
load without a missing or unexpected key. Two AUC conventions appear here, and
we keep them separate. The repository's own evaluation path scores frames with
the per-snippet score repeated $16\times$; under that convention the released
weights reproduce the published frame-level AUC ($0.8808$ against the $0.8802$
the paper reports). This paper's census and its eleven-arm comparison are
computed on the snippet grid --- one score and one label per sixteen frames ---
where the same weights give $\theta=0.8560$, and it is this snippet-grid value
that enters Table~\ref{tab:criterion} and the band comparison; the $0.025$
difference between the two conventions is a unit-of-analysis difference, not a
reproduction failure. On the snippet grid $\theta$ is $0.0056$
below the weakest of our arms, which is within the $0.0049$ between-seed
standard deviation of $\theta$, and its
$(\theta,B,B|z)=(0.8560,0.7157,0.6713)$ land inside the band spanned by our
eleven arms ($B\in[0.708,0.758]$, $B|z\in[0.655,0.723]$). The scores are the model's own head, not our
refinement, and the pair census comes out at the same $\omega=0.6317$, so the
comparison is between two independent score pipelines rather than two settings
of one. The band therefore brackets the published model
rather than being a self-selected set that happens to exclude it; its mAP@0.5
under that detection protocol ($2.95$) sits inside the family's span
$[2.44,5.82]$, while its AP@0.5 ($1.18$) falls below the family's minimum
($1.85$). That is the sense in which the localization columns
are comparable across implementations --- not that it predicts them.

## References

*Benchmarks and detectors.*
Hasan, Choi, Neumann, Roy-Chowdhury, Davis. Learning Temporal Regularity in
Video Sequences. CVPR 2016, 733--742. --- Sultani, Chen, Shah. Real-world
Anomaly Detection in Surveillance Videos. CVPR 2018. --- Wu, Liu, Shi, Sun,
Shao, Wu, Yang. Not only Look, But Also Listen: Learning Multimodal Violence
Detection Under Weak Supervision (XD-Violence). ECCV 2020, LNCS 12375:322--339.
--- Tian, Pang, Chen, Singh, Verjans, Carneiro. Weakly-supervised Video Anomaly
Detection with Robust Temporal Feature Magnitude Learning (RTFM). ICCV 2021,
4955--4966. --- Li, Liu, Jiao. Self-Training Multi-Sequence Learning with
Transformer for Weakly Supervised Video Anomaly Detection (MSL). AAAI 2022,
36(2):1395--1403. --- Wu, Hsieh, Chen, Fuh, Liu. Self-Supervised Sparse
Representation for Video Anomaly Detection (S3R). ECCV 2022, 729--745. ---
Chen, Liu, Zhang, Fok, Qi, Wu. MGFN: Magnitude-Contrastive Glance-and-Focus
Network for Weakly-Supervised Video Anomaly Detection. AAAI 2023, 37(1). ---
Zhou, Yu, Yang. Dual Memory Units with Uncertainty Regulation for Weakly
Supervised Video Anomaly Detection (UR-DMU). AAAI 2023, 37(3):3769--3777. ---
Zhou, Qu, Xu, Shen, Song, Shen. BatchNorm-based Weakly Supervised Video Anomaly
Detection (BN-WVAD). arXiv:2311.15367, 2023. --- Wu, Zhou, Pang, Zhou, Yan,
Wang, Zhang. VadCLIP: Adapting Vision-Language Models for Weakly Supervised
Video Anomaly Detection. AAAI 2024, 38(6):6074--6082. --- Yin, Wang, et al.
Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via
Disentangled Semantic Alignment (DSANet). AAAI-26, 40(14):12027--12035;
arXiv:2511.10334.

*Metric critique in this field and its neighbours.*
Song, Lee. Look Inside Each Video: Rethinking How Video Anomaly Detection Is
Evaluated. arXiv:2608.21854, 2026. --- Abdulaziz, Bondarev. Auditing Frame-Level
AUC in Weakly-Supervised Video Anomaly Detection: Granularity, Resolution, and
Scene Bias. arXiv:2608.11985, 2026. --- Liu, Wu, Li, Yang, Wang. Rethinking
Metrics and Benchmarks of Video Anomaly Detection. arXiv:2505.19022, 2025
(probabilistic AUC/AP, latency-aware AP, hard-normal benchmarks). --- Kim, Choi,
Choi, Lee. Towards a Rigorous
Evaluation of Time-Series Anomaly Detection. AAAI 2022. --- Tatbul, Lee, Zdonik,
Alam, Gottschlich. Precision and Recall for Time Series. NeurIPS 2018,
31:1924--1934. --- Hand, Till. A Simple
Generalisation of the Area Under the ROC Curve for Multiple Class
Classification Problems. Machine Learning 45:171--186, 2001.

*Causal and weak-supervision accounts of VAD.*
Lin, et al. A Causal Inference Look at Unsupervised Video Anomaly Detection.
AAAI 2022. --- Sun, et al. Learning Event-Relevant Factors for Video Anomaly
Detection. AAAI 2023. --- Learning Causality-inspired Representation Consistency
for Video Anomaly Detection (CIR). arXiv:2308.01537, 2023. --- Causal
Representation Consistency Learning for Anomaly Detection in Surveillance Videos
(CRCL). arXiv:2503.18808, 2025. --- Ning, Gao, Liu, Yuan, Zhang, He, Ning.
Causally Invariant Video Anomaly Detection via Counterfactual Reasoning and
Prototype Intervention (CIVC). Pattern Recognition 2026, 113809. --- Li, Shi,
Sun, Wu. Robust Video Anomaly Detection via Causal Feature-Guided Data
Augmentation. CVIU 2026, 265:104671. --- Unbiased Multiple Instance Learning for
Weakly Supervised Video Anomaly Detection (UMIL). arXiv:2303.12369, 2023. ---
Ilse, Tomczak, Welling. Attention-based Deep Multiple Instance Learning. ICML
2018.

*Statistical methodology classics.*
Mann, Whitney. On a Test of Whether one of Two Random Variables is Stochastically
Larger than the Other. Annals of Mathematical Statistics 18(1):50--60, 1947. ---
Simpson. The Interpretation of Interaction in Contingency Tables. JRSS-B
13(2):238--241, 1951. --- Campbell, Fiske. Convergent and Discriminant Validation
by the Multitrait-Multimethod Matrix. Psychological Bulletin 56(2):81--105,
1959. --- Hanley, McNeil. The Meaning and Use of the Area under a Receiver
Operating Characteristic (ROC) Curve. Radiology 143(1):29--36, 1982. --- DeLong,
DeLong, Clarke-Pearson. Comparing the Areas under Two or More Correlated
Receiver Operating Characteristic Curves: A Nonparametric Approach. Biometrics
44(3):837--845, 1988. --- Efron, Tibshirani. An Introduction to the Bootstrap.
Chapman & Hall/CRC, 1993. --- Strathern. 'Improving Ratings': Audit in the
British University System. European Review 5(3):305--321, 1997. --- Pearl.
Direct and Indirect Effects. UAI 2001, 411--420.

*Sensitivity analysis and partial identification.*
VanderWeele, Ding. Sensitivity Analysis in Observational Research: Introducing
the E-Value. Annals of Internal Medicine 167(4):268--274, 2017. --- Rosenbaum.
Sensitivity Analysis for Certain Permutation Inferences in Matched Observational
Studies. Biometrika 74(1):13--26, 1987. --- Manski. Nonparametric Bounds on Treatment
Effects. American Economic Review 80(2):319--323, 1990.

*Companion negative result.*
DebiasVAD (unpublished companion study): thirteen post-hoc debiasing variants,
all negative, including an oracle variant that uses ground-truth anomaly
distances ($-$ 0.0045 AP); cited here only as evidence for Theorem 4 and Section~\ref{sec:failures}.
