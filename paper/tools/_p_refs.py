# -*- coding: utf-8 -*-
"""Stage 3: replace every "§n.m" cross-reference by Section~\ref{...}, give each
heading a stable label, and turn the stray unicode math that sits in the running
text into real math.  Writes paper_CAFE_v4.md.
"""
import io, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v3.md"
DST = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v4.md"

LABELS = {
    "2": "sec:related", "3": "sec:setup", "4": "sec:results",
    "4.1": "sec:thm1", "4.2": "sec:thm2", "4.3": "sec:thm3", "4.4": "sec:thm4",
    "4.5": "sec:ap", "5": "sec:contrast", "5.1": "sec:causal", "5.2": "sec:motion",
    "5.3": "sec:bound", "5.4": "sec:report", "5.5": "sec:pairw",
    "5.6": "sec:strat", "5.7": "sec:resid", "5.8": "sec:inference",
    "5.9": "sec:evalue", "5.10": "sec:protocol", "6": "sec:exp",
    "6.1": "sec:host", "6.2": "sec:decomp", "6.3": "sec:degen",
    "6.4": "sec:resolution", "6.5": "sec:mediation", "6.6": "sec:audit",
    "6.7": "sec:infer-practice", "6.8": "sec:xd", "6.9": "sec:summary",
    "7": "sec:discussion", "7.1": "sec:answer", "7.2": "sec:design",
    "7.3": "sec:failures", "7.4": "sec:audits", "7.5": "sec:concurrent",
    "7.6": "sec:limits", "8": "sec:conclusion",
    "A": "sec:appA", "B": "sec:appB",
}

# ---------------------------------------------------------------- refs
def fix_refs(text):
    # ranges such as §4.1–§4.4
    def rng(m):
        a, b = m.group(1), m.group(2)
        return "Sections~\\ref{%s}--\\ref{%s}" % (LABELS[a], LABELS[b])
    text = re.sub(r"§\s?(\d+(?:\.\d+)?)–§\s?(\d+(?:\.\d+)?)", rng, text)
    text = re.sub(r"§\s?(\d+) and §\s?(\d+)",
                  lambda m: "Sections~\\ref{%s} and~\\ref{%s}" % (LABELS[m.group(1)], LABELS[m.group(2)]),
                  text)
    text = re.sub(r"§\s?(\d+\.\d+)\s?/\s?§\s?(\d+\.\d+)",
                  lambda m: "Sections~\\ref{%s} and~\\ref{%s}" % (LABELS[m.group(1)], LABELS[m.group(2)]),
                  text)
    text = re.sub(r"§\s?(\d+(?:\.\d+)?)",
                  lambda m: "Section~\\ref{%s}" % LABELS[m.group(1)], text)
    return text


# ---------------------------------------------------------------- labels
def add_labels(text):
    lines = text.split("\n")
    out = []
    for l in lines:
        m = re.match(r"^(##|###) (.+)$", l)
        if m and "{" not in l:
            title = m.group(2)
            num = re.match(r"^(\d+(?:\.\d+)?)\.?\s", title)
            key = None
            if num:
                key = LABELS.get(num.group(1))
            elif title.startswith("Appendix A"):
                key = "sec:appA"
            elif title.startswith("Appendix B"):
                key = "sec:appB"
            if key:
                l = "%s %s {#%s}" % (m.group(1), title, key)
        out.append(l)
    return "\n".join(out)


# ---------------------------------------------------------------- bare symbols
GREEK = {"θ": r"\theta", "τ": r"\tau", "ω": r"\omega", "δ": r"\delta",
         "σ": r"\sigma", "γ": r"\gamma", "μ": r"\mu", "π": r"\pi",
         "Δ": r"\Delta", "Σ": r"\sum"}
OPS = {"≤": r"\le", "≥": r"\ge", "≈": r"\approx", "−": "-", "·": r"\cdot",
       "×": r"\times", "→": r"\to", "∈": r"\in", "‖": r"\lVert",
       "≡": r"\equiv", "⟂": r"\perp", "⇒": r"\Rightarrow", "∖": r"\setminus",
       "⊆": r"\subseteq", "⊔": r"\sqcup", "²": "^{2}", "…": r"\dots"}


def normalize_plain(seg):
    """turn stray unicode math in a plain-text segment into $...$."""
    # protect already-written macros: nothing to do, we only touch non-$ text
    pat = [
        (r"ω_([ABCZ])", lambda m: r"$\omega_%s$" % m.group(1)),
        (r"δ\*", lambda m: r"$\delta^{*}$"),
        (r"δ_mot", lambda m: r"$\delta_{\mathrm{mot}}$"),
        (r"τ_LB", lambda m: r"$\tau_{\mathrm{LB}}$"),
        (r"τ_UB", lambda m: r"$\tau_{\mathrm{UB}}$"),
        (r"Δθ", lambda m: r"$\Delta\theta$"),
        (r"Δ([ABC])", lambda m: r"$\Delta %s$" % m.group(1)),
        (r"B\|z", lambda m: r"$B\mid z$"),
        (r"B̂", lambda m: r"$\hat{B}$"),
        (r"Â", lambda m: r"$\hat{A}$"),
        (r"corr²", lambda m: r"$\mathrm{corr}^2$"),
        (r"10\^\{(-?\d+)\}", lambda m: r"$10^{%s}$" % m.group(1)),
        (r"([𝔄𝔅ℭ𝔓]+)", lambda m: "$" + "".join(
            {r"\mathcal{A}": r"\mathcal{A}", }.get("", "") or
            {c: d for c, d in [("𝔄", r"\mathcal{A}"), ("𝔅", r"\mathcal{B}"),
                               ("ℭ", r"\mathcal{C}"), ("𝔓", r"\mathcal{P}")]}[ch]
            for ch in m.group(1)) + "$"),
        (r"—", lambda m: "---"),
        (r"–", lambda m: "--"),
        (r"…", lambda m: r"\dots{}"),
        (r"±", lambda m: r"$\pm$"),
        (r"∎", lambda m: r"$\blacksquare$"),
        (r"([θτωδσγμπΣ])", lambda m: "$%s$" % GREEK[m.group(1)]),
        (r"(?<![$.])([≤≥≈−·×→∈‖≡⟂⇒∖⊆₂²])(?![$])", lambda m: "$%s$" % OPS.get(m.group(1), m.group(1))),
    ]
    for p, f in pat:
        seg = re.sub(p, f, seg)
    # a math span glued to a following digit is read as currency by pandoc
    seg = re.sub(r"(\$[^$]+\$)(?=\d)", r"\1 ", seg)
    return seg


def normalize(text):
    out = []
    i = 0
    # do not touch math, equations, tables, code
    for m in re.finditer(r"(\$\$.*?\$\$|\$[^$]+\$|\\begin\{[a-z]+\*?\}.*?\\end\{[a-z]+\*?\})",
                         text, re.S):
        out.append(normalize_plain(text[i:m.start()]))
        out.append(m.group(0))
        i = m.end()
    out.append(normalize_plain(text[i:]))
    return "".join(out)


def main():
    t = io.open(SRC, encoding="utf-8").read()
    t = add_labels(t)
    t = fix_refs(t)
    t = normalize(t)
    io.open(DST, "w", encoding="utf-8").write(t)
    left = sorted({c for c in t if ord(c) > 127})
    print("wrote", DST, len(t), "bytes")
    print("remaining non-ascii:", [(c, hex(ord(c))) for c in left])
    print("refs:", t.count("\\ref{"), "| §:", t.count("§"))


if __name__ == "__main__":
    main()
