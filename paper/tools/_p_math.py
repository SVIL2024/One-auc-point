# -*- coding: utf-8 -*-
"""Stage 2: turn the markdown of the CAFE paper into typographically sane LaTeX
math.

- every inline code span that denotes a symbol becomes real math;
- spans that are plain numbers become plain text;
- spans that are file names, keys or commands are removed (rewritten prose);
- lines that consist of a single formula become numbered equations.

Writes paper_CAFE_v3.md.
"""
import io, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v2.md"
DST = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v3.md"

# ---------------------------------------------------------------- span mapping
# explicit overrides for spans that a rule-based mapper gets wrong
EXPLICIT = {
    # blocks and pools (calligraphic sets, standard in ML)
    "𝔄": r"\mathcal{A}", "𝔅": r"\mathcal{B}", "ℭ": r"\mathcal{C}", "𝔓": r"\mathcal{P}",
    "𝔄 ⊆ 𝔓": r"\mathcal{A}\subseteq\mathcal{P}",
    "𝔅 ⊆ 𝔓": r"\mathcal{B}\subseteq\mathcal{P}",
    "ℭ = 𝔓 ∖ (𝔄 ∪ 𝔅)": r"\mathcal{C}=\mathcal{P}\setminus(\mathcal{A}\cup\mathcal{B})",
    "𝔓 = 𝔄 ⊔ 𝔅 ⊔ ℭ": r"\mathcal{P}=\mathcal{A}\sqcup\mathcal{B}\sqcup\mathcal{C}",
    "N = |𝔓|": r"N=|\mathcal{P}|",
    "n_A = |𝔄|": r"n_A=|\mathcal{A}|",
    "n_B = |𝔅|": r"n_B=|\mathcal{B}|",
    "n_C = |ℭ|": r"n_C=|\mathcal{C}|",
    "ω_Z = n_Z/N": r"\omega_Z=n_Z/N",
    "Z ∈ {A,B,C}": r"Z\in\{A,B,C\}",
    # statistics
    "B": r"B", "A": r"A", "C": r"C", "z": r"z", "Z": r"Z", "s": r"s",
    "v": r"v", "i": r"i", "j": r"j", "q": r"q", "f": r"f", "E": r"E",
    "V": r"V", "D": r"D", "L": r"L", "π": r"\pi", "γ": r"\gamma",
    "σ": r"\sigma", "θ": r"\theta", "τ": r"\tau", "ω": r"\omega",
    "B|z": r"B|z", "B\\|z": r"B|z",
    "B̂": r"\hat{B}", "Â_v": r"\hat{A}_v", "ΔB̂": r"\Delta\hat{B}",
    "B̂ = Σ_v a_v b_v · Â_v / Σ_v a_v b_v":
        r"\hat{B}=\frac{\sum_v a_v b_v \hat{A}_v}{\sum_v a_v b_v}",
    "Â_v = (1/(a_v b_v)) Σ_{i: g_i=1} Σ_{j: g_j=0} 1[s_i > s_j]":
        r"\hat{A}_v=\frac{1}{a_v b_v}\sum_{i:\,y_i=1}\sum_{j:\,y_j=0}\mathbb{1}[s_i>s_j]",
    "s_v": r"s_v", "τ_v": r"\tau_v", "a_v": r"a_v", "b_v": r"b_v",
    "w_v ∝ a_v b_v": r"w_v\propto a_v b_v", "a_v, b_v > 0": r"a_v,b_v>0",
    "n_v": r"n_v", "O(n_v log n_v)": r"O(n_v\log n_v)",
    # labels: y for the ground truth, A_i kept for the causal node
    "g_v": r"y_v", "g_i = 1, g_j = 0": r"y_i=1,\ y_j=0",
    "g_v ∈ {0,1}^{s_v}": r"y_v\in\{0,1\}^{s_v}",
    "A_i ∈ {0,1}": r"A_i\in\{0,1\}",
    "S_i = s(x_i)": r"S_i=s(x_i)", "S_i": r"S_i",
    "A_i ⟂ rank(S_i) | Z_i, v": r"A_i \perp \mathrm{rank}(S_i)\mid Z_i,v",
    # weights and identities
    "ω_A": r"\omega_A", "ω_B": r"\omega_B", "ω_C": r"\omega_C", "ω_Z": r"\omega_Z",
    "ω_B = 0.00388": r"\omega_B=0.00388",
    "ω_B/2": r"\omega_B/2", "ω_B/2 = 0.0019": r"\omega_B/2=0.0019",
    "ω_B·B": r"\omega_B B", "ω_B·B = 0.0006": r"\omega_B B=0.0006",
    "ω_A > 0": r"\omega_A>0", "ω_A = 163·ω_B": r"\omega_A=163\,\omega_B",
    "1/ω_B": r"1/\omega_B", "n_B/N": r"n_B/N",
    "θ = ω_A·A + ω_B·B + ω_C·C": r"\theta=\omega_A A+\omega_B B+\omega_C C",
    "θ(s) = ω_A·A(s) + ω_B·B(s) + ω_C·C(s)":
        r"\theta=\omega_A A+\omega_B B+\omega_C C",
    "ω_A + ω_B + ω_C = 1": r"\omega_A+\omega_B+\omega_C=1",
    "ω_A = 0.6317, ω_B = 0.0039, ω_C = 0.3645":
        r"\omega_A=0.6317,\ \omega_B=0.0039,\ \omega_C=0.3645",
    "ω = (0.6985, 0.000710, 0.3008)": r"\omega=(0.6985,0.00071,0.3008)",
    "ω_A, ω_B, ω_C": r"\omega_A,\omega_B,\omega_C",
    "ω_B ≈ Σ_v a_v b_v / (N_pos·N_neg)":
        r"\omega_B\approx\sum_v a_v b_v/(N_{\mathrm{pos}}N_{\mathrm{neg}})",
    "ω_B = 0.07%": r"\omega_B=0.07\%",
    "(ω_A + ω_C)/ω_B = 256.7": r"(\omega_A+\omega_C)/\omega_B=256.7",
    "θ(s) = (1/N) Σ_{(i,j) ∈ 𝔓} 1[s_i > s_j]":
        r"\theta=\frac{1}{N}\sum_{(i,j)\in\mathcal{P}}\mathbb{1}[s_i>s_j]",
    "θ(s) = ω_A·A_f + ω_C·C_f + ω_B/2":
        r"\theta=\omega_A A_f+\omega_C C_f+\omega_B/2",
    "A_f, C_f": r"A_f,C_f", "A_f, C_f ≤ 1": r"A_f,C_f\le 1",
    "A_f = C_f = 1": r"A_f=C_f=1",
    "s_i = f(v)": r"s_i=f(v)", "i ∈ v": r"i\in v", "B(s) = 1/2": r"B(s)=1/2",
    "B ≡ 0.5": r"B\equiv 0.5", "B = 0.5": r"B=0.5", "B = 1": r"B=1",
    "sup θ = 1 − ω_B/2": r"\sup\theta=1-\omega_B/2",
    "∂θ/∂B = ω_B": r"\partial\theta/\partial B=\omega_B",
    "τ(s) = B": r"\tau(s)=B",
    "B = Σ_v w_v τ_v": r"B=\sum_v w_v\tau_v",
    "τ_v = P(S_i > S_j | A_i = 1, A_j = 0, i,j ∈ v)":
        r"\tau_v=P(S_i>S_j\mid A_i=1,A_j=0,\ i,j\in v)",
    "P(win | anomaly) = τ": r"P(\mathrm{win}\mid\mathrm{anomaly})=\tau",
    "P(win | background) = 1 − τ": r"P(\mathrm{win}\mid\mathrm{background})=1-\tau",
    "RR = τ/(1−τ)": r"RR=\tau/(1-\tau)",
    "E = RR + sqrt(RR·(RR − 1))": r"E=RR+\sqrt{RR\,(RR-1)}",
    # exchange rate and error propagation
    "Δθ": r"\Delta\theta", "ΔB": r"\Delta B", "ΔA": r"\Delta A", "ΔC": r"\Delta C",
    "θ'": r"\theta'",
    "ΔB = (Δθ − ω_A·ΔA − ω_C·ΔC)/ω_B":
        r"\Delta B=(\Delta\theta-\omega_A\Delta A-\omega_C\Delta C)/\omega_B",
    "ΔA = (Δθ − ω_BΔB − ω_CΔC)/ω_A":
        r"\Delta A=(\Delta\theta-\omega_B\Delta B-\omega_C\Delta C)/\omega_A",
    "ΔB ∈ [−1, 1]": r"\Delta B\in[-1,1]",
    "ΔB = Δθ/ω_B": r"\Delta B=\Delta\theta/\omega_B",
    "σ/ω_B = 258σ": r"\sigma/\omega_B=258\sigma",
    "1,408σ": r"1{,}408\sigma",
    "SE(ΔB̂) = 1.24": r"\mathrm{SE}(\Delta\hat{B})=1.24",
    "σ(Δθ) ≤ 1.9·10^{-5}": r"\sigma(\Delta\theta)\le 1.9\cdot10^{-5}",
    "σ(θ) = 0.0048": r"\sigma(\theta)=0.0048",
    "0.0048/0.00388 = 1.24": r"0.0048/0.00388=1.24",
    "2B − 1": r"2B-1", "1 − B": r"1-B", "1 − δ": r"1-\delta",
    "< 10^{-12}": r"<10^{-12}", "< 10^{-4}": r"<10^{-4}",
    # partial identification
    "δ*": r"\delta^{*}", "δ* = 2B − 1": r"\delta^{*}=2B-1",
    "δ* = 0.416–0.515": r"\delta^{*}\in[0.416,0.515]",
    "δ* = 0.609–0.693": r"\delta^{*}\in[0.609,0.693]",
    "τ_LB": r"\tau_{\mathrm{LB}}", "τ_LB(0.2)": r"\tau_{\mathrm{LB}}(0.2)",
    "δ_mot = D/(1−B|z) = 0.120–0.170": r"\delta_{\mathrm{mot}}\in[0.120,0.170]",
    "δ_mot = (B − B|z)/(1 − B|z)": r"\delta_{\mathrm{mot}}=(B-B\mid z)/(1-B\mid z)",
    "τ_LB(δ) = (B − δ) / (1 − δ)  ≤  τ  ≤  min{1, B / (1 − δ)} = τ_UB(δ)":
        r"\tau_{\mathrm{LB}}(\delta)=\frac{B-\delta}{1-\delta}\le\tau\le"
        r"\min\{1,\tfrac{B}{1-\delta}\}=\tau_{\mathrm{UB}}(\delta)",
    "0 ≤ δ < 1": r"0\le\delta<1", "δ = 0.2": r"\delta=0.2", "δ < δ*": r"\delta<\delta^{*}",
    "B = (1−δ)·τ + δ·q": r"B=(1-\delta)\tau+\delta q", "q ∈ [0,1]": r"q\in[0,1]",
    "τ_LB(δ) > 0.5": r"\tau_{\mathrm{LB}}(\delta)>0.5",
    "δ < 2B − 1 =: δ*": r"\delta<2B-1=:\delta^{*}",
    "[(B−δ)/(1−δ), min(1, B/(1−δ))]":
        r"[(B-\delta)/(1-\delta),\ \min(1,B/(1-\delta))]",
    "[τ_LB, τ_UB]": r"[\tau_{\mathrm{LB}},\tau_{\mathrm{UB}}]",
    "0.122–0.141": r"[0.122,0.141]",
    "δ·m": r"\delta m", "m = 400": r"m=400", "n = 4,000": r"n=4{,}000",
    "τ ≈ 0.760": r"\tau\approx0.760", "q = 1": r"q=1", "q = 5": r"q=5", "q = 10": r"q=10",
    "B = 0.785 / 0.809 / 0.832 / 0.880": r"B=0.785/0.809/0.832/0.880",
    "δ = 0.1 / 0.2 / 0.3 / 0.5": r"\delta=0.1/0.2/0.3/0.5",
    # unidentifiability
    "{s_i : i ∈ v}": r"\{s_i: i\in v\}",
    "L(s ∘ π) = L(s)": r"\mathcal{L}(s\circ\pi)=\mathcal{L}(s)",
    "κ_v = top_k(s) − mean(s)": r"T_v=\mathrm{top}_k(s)-\mathrm{mean}(s)",
    "κ_v ≥ 0": r"T_v\ge 0", "y = 0 ⇒ κ ≤ m": r"y=0\Rightarrow T\le m",
    "B = 0": r"B=0", "B = 0.3124": r"B=0.3124",
    # nuisance coordinates
    "‖Δx_t‖": r"\lVert\Delta x_t\rVert",
    "[t, ‖Δx‖, moving-average ‖Δx‖, ‖x‖]":
        r"[t,\ \lVert\Delta x\rVert,\ \overline{\lVert\Delta x\rVert},\ \lVert x\rVert]",
    "γ·mean_v corr²(S_v, Z_v)": r"\gamma\,\mathrm{mean}_v\mathrm{corr}^2(S_v,Z_v)",
    "s ~ f(z)": r"s\sim f(z)", "B_res": r"B^{\mathrm{res}}",
    "A → motion → S": r"A\to\mathrm{motion}\to S",
    "corr": r"\mathrm{corr}",
    # numbers with symbols
    "B = 0.6552": r"B=0.6552", "B|z = 0.5465": r"B\mid z=0.5465",
    "B = 0.7558": r"B=0.7558", "B|z = 0.7166": r"B\mid z=0.7166",
    "B = 0.66": r"B=0.66", "B = 0.536": r"B=0.536",
    "B = 0.3124": r"B=0.3124",
    "ΔB = 0.01": r"\Delta B=0.01", "ΔB = 2.06": r"\Delta B=2.06",
    "θ = 0.8234": r"\theta=0.8234", "B = 0.5000": r"B=0.5000",
    "θ = 0.8257": r"\theta=0.8257", "B = 0.657": r"B=0.657",
    "θ = 0.8772 [0.8734, 0.8808]": r"\theta=0.8772\ [0.8734,0.8808]",
    "[0.8237, 0.9152]": r"[0.8237,0.9152]",
    "0.5·ω_B = 0.0019": r"0.5\,\omega_B=0.0019",
    "0.00388 × 0.0685 = 0.00027": r"0.00388\times0.0685=0.00027",
    "E(B) = 5.6": r"E(B)=5.6", "E(B|z) = 4.5": r"E(B\mid z)=4.5",
    "sd(B|z) = 0.0245": r"\mathrm{sd}(B\mid z)=0.0245",
    "sd(θ) = 0.0055": r"\mathrm{sd}(\theta)=0.0055",
    "sd(B) = 0.0095": r"\mathrm{sd}(B)=0.0095",
    "sd(B|z) = 0.0114": r"\mathrm{sd}(B\mid z)=0.0114",
    "ω_B/2 = 0.00036": r"\omega_B/2=0.00036",
    "(θ, B, B|z, CI, E)": r"(\theta,B,B\mid z,\mathrm{CI},E)",
    "(θ, B, B|z)": r"(\theta,B,B\mid z)",
    "B|z = 0.7165 [0.6454, 0.7872]": r"B\mid z=0.7165\ [0.6454,0.7872]",
    "B = 0.7558 [0.6844, 0.8364]": r"B=0.7558\ [0.6844,0.8364]",
    "B|z = 0.7166 [0.6467, 0.7907]": r"B\mid z=0.7166\ [0.6467,0.7907]",
    "(0.6552 − 0.5)/(B_model − 0.5) = 86%":
        r"(0.6552-0.5)/(B_{\mathrm{model}}-0.5)=86\%",
    "Spearman(θ, B|z) = 0.22 (p = 0.52)": r"\rho_s(\theta,B\mid z)=0.22\ (p=0.52)",
    "Spearman(θ, B|z) = 0.218 (p = 0.52)": r"\rho_s(\theta,B\mid z)=0.218\ (p=0.52)",
    "Spearman(θ, B) = 0.418 (p = 0.20)": r"\rho_s(\theta,B)=0.418\ (p=0.20)",
    "Spearman(θ, A) = 0.191 (p = 0.57)": r"\rho_s(\theta,A)=0.191\ (p=0.57)",
    "Spearman(B, B|z) = 0.800 (p = 0.003)": r"\rho_s(B,B\mid z)=0.800\ (p=0.003)",
    "Spearman = 0.8": r"\rho_s=0.8",
    "θ ∈ {0.8800, 0.8705, 0.8772}": r"\theta\in\{0.8800,0.8705,0.8772\}",
    "B ∈ {0.7270, 0.7236, 0.7558}": r"B\in\{0.7270,0.7236,0.7558\}",
    "B|z ∈ {0.6729, 0.6688, 0.7169}": r"B\mid z\in\{0.6729,0.6688,0.7169\}",
    "γ = 1": r"\gamma=1",
    "1,408.6": "1,407.6",
}

# spans that are file names / keys / commands: dropped from the prose
DROP = {
    "cafe_report.py", ".npz", "XDDataset", "mann_whitney", "rankdata",
    "--claim-points", "s2", "s1", "lens", "gt", "mlocal", "*_metrics.json",
}

NUM = re.compile(r"^[\d][\d,.\u2013\u2014%\-\u2212\s/]*$")


def texify(body):
    out = body
    # multi-character sequences first
    seq = [
        ("B̂", r"\hat{B}"), ("Â", r"\hat{A}"), ("Δθ", r"\Delta\theta"),
        ("Δx", r"\Delta x"), ("ΔA", r"\Delta A"), ("ΔB", r"\Delta B"),
        ("ΔC", r"\Delta C"), ("δ_mot", r"\delta_{\mathrm{mot}}"),
        ("δ*", r"\delta^{*}"), ("τ_LB", r"\tau_{\mathrm{LB}}"),
        ("τ_UB", r"\tau_{\mathrm{UB}}"), ("ω_A", r"\omega_A"),
        ("ω_B", r"\omega_B"), ("ω_C", r"\omega_C"), ("ω_Z", r"\omega_Z"),
        ("B|z", r"B|z"), ("‖", r"\lVert "), ("∖", r"\setminus "),
        ("⊆", r"\subseteq "), ("⊔", r"\sqcup "), ("⇒", r"\Rightarrow "),
        ("≡", r"\equiv "), ("⟂", r"\perp "), ("≈", r"\approx "),
        ("≤", r"\le "), ("≥", r"\ge "), ("−", "-"), ("·", r"\cdot "),
        ("×", r"\times "), ("∘", r"\circ "), ("→", r"\to "), ("∈", r"\in "),
        ("Σ", r"\sum "), ("²", "^{2}"), ("θ", r"\theta "), ("τ", r"\tau "),
        ("ω", r"\omega "), ("δ", r"\delta "), ("σ", r"\sigma "),
        ("γ", r"\gamma "), ("μ", r"\mu "), ("…", r"\dots "),
    ]
    for a, b in seq:
        out = out.replace(a, b)
    # collapse "x _{" -> "x_{", "x  " -> "x "
    out = re.sub(r"\s+_", "_", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def convert(span):
    if span in EXPLICIT:
        return "$%s$" % EXPLICIT[span]
    if span in DROP:
        return "\x00"          # caller rewrites such sentences by hand
    if NUM.match(span):
        return span            # plain number, no math
    return "$%s$" % texify(span)


def main():
    text = io.open(SRC, encoding="utf-8").read()

    seen = set()

    def repl(m):
        span = m.group(1)
        seen.add(span)
        return convert(span)

    text = re.sub(r"`([^`\n]{1,90})`", repl, text)

    unknown = [s for s in seen if s not in EXPLICIT and s not in DROP and not NUM.match(s)]
    for u in sorted(unknown):
        print("UNMAPPED:", repr(u), "->", convert(u))

    # numbered equations: a paragraph line that is exactly one formula
    lines = text.split("\n")
    out = []
    eqn = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^\s*\$([^$]+)\$([,.;]?)\s*$", ln)
        if m:
            # collect consecutive formula lines into one display block
            group = []
            while i < len(lines):
                mm = re.match(r"^\s*\$([^$]+)\$([,.;]?)\s*$", lines[i])
                if not mm:
                    break
                group.append(mm.group(1) + mm.group(2))
                i += 1
            eqn += 1
            body = group[0] if len(group) == 1 else r"\begin{gathered}" + r"\\ ".join(group) + r"\end{gathered}"
            out.append(r"\begin{equation}")
            out.append(body)
            out.append(r"\end{equation}")
            continue
        out.append(ln)
        i += 1
    text = "\n".join(out)
    # the decomposition is the paper's central display: number it
    old = ("  $\\theta=\\omega_A A+\\omega_B B+\\omega_C C$,   "
           "$\\omega_A+\\omega_B+\\omega_C=1$.")
    new = ("\\begin{equation}\n"
           "\\theta=\\omega_A A+\\omega_B B+\\omega_C C,\\qquad "
           "\\omega_A+\\omega_B+\\omega_C=1 .\n"
           "\\label{eq:decomp}\n"
           "\\end{equation}")
    if old in text:
        text = text.replace(old, new)
        eqn += 1
    else:
        print("WARN: decomposition display not found")
    io.open(DST, "w", encoding="utf-8").write(text)
    print("wrote", DST, len(text), "bytes;", eqn, "numbered equations;",
          text.count("$"), "dollar signs")


if __name__ == "__main__":
    main()
