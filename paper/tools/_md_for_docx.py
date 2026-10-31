# -*- coding: utf-8 -*-
"""Turn the LaTeX-flavoured paper markdown into pandoc-friendly markdown for .docx.

The master file (paper_CAFE_v6.md) keeps its tables and equations in raw LaTeX
because that is what the PDF route needs.  Pandoc drops raw LaTeX when the target
is docx, so this script rewrites the eight tables as pipe tables, the twelve
equations as numbered plain-text paragraphs, and resolves every \\ref into a
literal number (sections from their headings, tables from their order).

Usage:  python _md_for_docx.py [paper_CAFE_v6.md] [-o out.md]
"""
import io
import re
import sys

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ",
    "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Delta": "Δ", "Gamma": "Γ", "Lambda": "Λ", "Sigma": "Σ", "Omega": "Ω",
}
SYM = {
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠",
    "approx": "≈", "sim": "~", "propto": "∝", "in": "∈", "to": "→",
    "rightarrow": "→", "Rightarrow": "⇒", "Leftrightarrow": "⇔",
    "times": "×", "cdot": "·", "pm": "±", "mp": "∓", "infty": "∞",
    "dots": "…", "ldots": "…", "cdots": "⋯", "prime": "′",
    "circ": "∘", "mid": " | ", "perp": "⊥", "subset": "⊂", "cup": "∪",
    "cap": "∩", "emptyset": "∅", "forall": "∀", "exists": "∃",
    "lVert": "‖", "rVert": "‖", "ll": "≪", "gg": "≫",
    "setminus": "∖", "sqcup": "⊔", "blacksquare": "■", "sup": "sup",
    "inf": "inf", "argmax": "argmax", "argmin": "argmin",
    "overline": "", "lognorm": "", "partial": "∂", "equiv": "≡", "propto": "∝",
}


def _grab(s, i):
    """s[i] == '{' -> return (content, index_after_closing_brace)."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return s[i + 1:], len(s)


def convert_frac(s):
    out = []
    i = 0
    while True:
        k = s.find("\\frac", i)
        if k < 0:
            out.append(s[i:])
            break
        out.append(s[i:k])
        j = k + 5
        while j < len(s) and s[j] == " ":
            j += 1
        if j < len(s) and s[j] == "{":
            a, j = _grab(s, j)
            while j < len(s) and s[j] == " ":
                j += 1
            if j < len(s) and s[j] == "{":
                b, j = _grab(s, j)
                out.append("(%s)/(%s)" % (a, b))
            else:
                out.append("(%s)/" % a)
        i = j
    return "".join(out)


def clean_math(m):
    """Render one math snippet as readable plain text."""
    s = m
    s = s.replace("\\mathcal{A}", "A").replace("\\mathcal{B}", "B")
    s = s.replace("\\mathcal{C}", "C").replace("\\mathcal{L}", "L")
    s = s.replace("\\mathrm", "").replace("\\text", "").replace("\\mathbf", "")
    s = s.replace("\\left", "").replace("\\right", "")
    # fractions: \frac{a}{b} -> (a)/(b) with brace matching (handles nesting)
    s = s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    s = convert_frac(s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
    s = re.sub(r"\\mathbb\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\\\", "; ")
    s = re.sub(r"\\(?:Big|big|Bigg|bigg)(?=[()\[\]{}\\])", "", s)
    s = s.replace("\\lVert", "‖").replace("\\rVert", "‖").replace("\\|", "‖")
    s = s.replace("\\;", "  ").replace("\\,", " ").replace("\\!", "")
    s = re.sub(r"\\q?quad\b", "   ", s)
    s = re.sub(r"\\hat\{([^{}]*)\}", lambda m: m.group(1) + "\u0302", s)
    s = re.sub(r"\\hat\s*([A-Za-z])", lambda m: m.group(1) + "\u0302", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", lambda m: m.group(1) + "\u0304", s)
    s = re.sub(r"\\tilde\{([^{}]*)\}", lambda m: m.group(1) + "\u0303", s)
    s = re.sub(r"\\sum(?:_\{?([A-Za-z])\}?)?", r"Σ_\1", s)
    s = s.replace("\\prod", "∏").replace("\\int", "∫")
    s = s.replace("\\max", "max").replace("\\min", "min")
    s = s.replace("\\log", "log").replace("\\exp", "exp").replace("\\Pr", "Pr")
    s = s.replace("\\dagger", "†").replace("\\ddagger", "‡")
    s = s.replace("^†", "†")
    for k, v in sorted(GREEK.items(), key=lambda kv: -len(kv[0])):
        s = s.replace("\\" + k, v)
    for k, v in sorted(SYM.items(), key=lambda kv: -len(kv[0])):
        s = s.replace("\\" + k, v)
    s = s.replace("\\_", "_").replace("\\%", "%").replace("\\{", "{").replace("\\}", "}")
    s = s.replace("\\#", "#").replace("\\&", "&").replace("\\$", "$")
    s = re.sub(r"\\(?:mathrm|text|mathcal)\{([^{}]*)\}", r"\1", s)
    s = s.replace("^{", "^(").replace("_{", "_(")
    s = s.replace("}", ")").replace("{", "(")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def demath(text):
    """Replace every $...$ by its plain rendering; keep the rest of the text."""
    out = []
    i = 0
    while True:
        a = text.find("$", i)
        if a < 0:
            out.append(text[i:])
            break
        b = text.find("$", a + 1)
        if b < 0:
            out.append(text[i:])
            break
        out.append(text[i:a])
        out.append(clean_math(text[a + 1:b]))
        i = b + 1
    s = "".join(out)
    s = s.replace("\\%", "%").replace("\\_", "_")
    s = s.replace("\\#", "#").replace("\\&", "&").replace("\\$", "$")
    s = s.replace("---", "—").replace("--", "–")
    s = re.sub(r"\\dots\b", "…", s)
    s = s.replace("\\dagger", "†").replace("\\ddagger", "‡")
    s = re.sub(r"\\S\b", "§", s)
    s = s.replace("\\emph", "").replace("\\textbf", "")
    s = re.sub(r"\\(?:citep|citet|cite)\{([^}]*)\}", r"[\1]", s)
    s = s.replace("~", " ")
    return s


def split_cells(row):
    """Split a tabular row on & at brace depth zero."""
    cells, depth, cur = [], 0, []
    for ch in row:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur))
    return [c.strip() for c in cells]


def parse_tabular(body):
    rows = []
    for raw in re.split(r"\\\\", body):
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b", "", line).strip()
        if not line:
            continue
        line = re.sub(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{([^{}]*)\}",
                      lambda m: m.group(2) + " | " * (int(m.group(1)) - 1), line)
        cells = split_cells(line)
        if len(cells) > 1:
            rows.append(cells)
    return rows


def build_labels(md):
    sec = {}
    for m in re.finditer(r"(?m)^(#{1,3})\s+(\d+(?:\.\d+)*)\.?\s+.*?\{#((?:sec|app):[^}]+)\}", md):
        sec[m.group(3)] = m.group(2)
    tabs = {}
    for i, m in enumerate(re.finditer(r"\\begin\{table\}", md), 1):
        tail = md[m.end():m.end() + 900]
        lm = re.search(r"\\label\{(tab:[^}]+)\}", tail)
        if lm:
            tabs[lm.group(1)] = str(i)
    eqs = {}
    for i, m in enumerate(re.finditer(r"\\begin\{equation\}", md), 1):
        tail = md[m.end():m.end() + 900]
        lm = re.search(r"\\label\{(eq:[^}]+)\}", tail)
        if lm:
            eqs[lm.group(1)] = str(i)
    return sec, tabs, eqs


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "paper_CAFE_v6.md"
    out = "paper_CAFE_v6_docx.md"
    if "-o" in sys.argv:
        out = sys.argv[sys.argv.index("-o") + 1]
    md = io.open(src, encoding="utf-8").read()
    sec, tabs, eqs = build_labels(md)

    # ---- tables -------------------------------------------------------------
    def repl_table(m):
        block = m.group(0)
        cap = re.search(r"\\caption\{(.+?)\}\s*\n", block, re.S)
        lab = re.search(r"\\label\{(tab:[^}]+)\}", block)
        num = tabs.get(lab.group(1), "?") if lab else "?"
        tab = re.search(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}(.+?)\\end\{tabular\}",
                        block, re.S)
        if not tab:
            return demath(cap.group(1)) + "\n" if cap else ""
        rows = parse_tabular(tab.group(1))
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        def cell(c, width_hint=None):
            # a literal pipe would split the pipe-table column, so escape it
            return demath(c).replace("|", "\\|")

        lines = ["**Table %s.** %s" % (num, demath(cap.group(1)).replace("\n", " ").strip()
                                       if cap else ""), ""]
        head = [cell(c) for c in rows[0]]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join(["@@SEP@@"] * width) + "|")
        for r in rows[1:]:
            lines.append("| " + " | ".join(cell(c) for c in r) + " |")
        return "\n".join(lines) + "\n"

    md = re.sub(r"\\begin\{table\}.*?\\end\{table\}", repl_table, md, flags=re.S)

    # ---- figures ------------------------------------------------------------
    def repl_fig(m):
        block = m.group(0)
        cap = re.search(r"\\caption\{(.+?)\}\s*\n", block, re.S)
        cap_text = demath(cap.group(1)).replace("\n", " ").strip() if cap else ""
        return "\n![Figure 1. %s](fig1.png)\n\n**Figure 1.** %s\n" % (cap_text[:40], cap_text)

    md = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", repl_fig, md, flags=re.S)

    # ---- equations ----------------------------------------------------------
    counter = {"n": 0}

    def repl_eq(m):
        block = m.group(0)
        counter["n"] += 1
        lab = re.search(r"\\label\{(eq:[^}]+)\}", block)
        num = eqs.get(lab.group(1), str(counter["n"])) if lab else str(counter["n"])
        body = re.sub(r"\\label\{[^}]*\}", "", block)
        body = re.sub(r"\\begin\{equation\}|\\end\{equation\}", "", body)
        body = re.sub(r"\\(?:begin|end)\{(?:gathered|aligned|split|array)\}"
                      r"(?:\{[^}]*\})?", " ", body)
        text = clean_math(body)
        return "\n**(%s)**  %s\n" % (num, text)

    md = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", repl_eq, md, flags=re.S)

    # ---- refs ---------------------------------------------------------------
    def label_of(key):
        if key.startswith("tab:"):
            return tabs.get(key, "?")
        if key.startswith("eq:"):
            return eqs.get(key, "?")
        return sec.get(key, "?")

    def repl_ref(m):
        return label_of(m.group(1))

    # "Section~\ref{x}" -> "Section 6.7"; "Table~\ref{x}" -> "Table 4"
    md = re.sub(r"\\eqref\{(eq:[^}]+)\}", lambda m: "(" + label_of(m.group(1)) + ")", md)
    md = re.sub(r"(Table|Section|Appendix)\s*~?\s*\\ref\{([^}]+)\}",
                lambda m: m.group(1) + " " + label_of(m.group(2)), md)
    md = re.sub(r"\\ref\{([^}]+)\}", repl_ref, md)

    # ---- leftovers ----------------------------------------------------------
    s = re.sub(r"\\label\{[^}]*\}", "", md)
    md = s
    md = re.sub(r"\\noindent\b\s*", "", md)
    md = md.replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ")
    md = md.replace("i.e.\\ ", "i.e. ").replace("\\@", "")
    md = re.sub(r"(?m)^\\(?:small|footnotesize|centering|noindent)\s*$", "", md)
    md = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", md)
    md = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b", "", md)
    md = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", md, flags=re.S)
    md = demath(md)
    md = md.replace("@@SEP@@", "---")
    md = re.sub(r"\n{3,}", "\n\n", md)
    io.open(out, "w", encoding="utf-8").write(md)
    print("wrote", out, len(md), "chars")
    print("residual backslashes:",
      len([m for m in re.finditer(r"\\", md) if md[m.start():m.start() + 2] != "\\|"]))


if __name__ == "__main__":
    main()
