# -*- coding: utf-8 -*-
"""Assemble the elsarticle manuscript for CAFE.

input : CAFE/latex/body_raw.tex   (pandoc output of paper_CAFE_v1.md)
output: E:\\paper\\drafts\\2026\\CAFE\\elsarticle\\CAFE.tex
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = r"E:\program\python\paper\CAFE\latex\body_raw.tex"
OUTDIR = r"E:\paper\drafts\2026\CAFE\elsarticle"
DST = os.path.join(OUTDIR, "CAFE.tex")

SINGLE = {
    "\u2014": "---", "\u2013": "--", "\u00a7": r"\S{}",
    "\u00b7": r"\ensuremath{\cdot}", "\u2026": r"\ldots{}",
    "\u220e": r"\ensuremath{\blacksquare}",
    "\u03b8": r"\ensuremath{\theta}", "\u03c4": r"\ensuremath{\tau}",
    "\u03b4": r"\ensuremath{\delta}", "\u0394": r"\ensuremath{\Delta}",
    "\u03bb": r"\ensuremath{\lambda}", "\u03c9": r"\ensuremath{\omega}",
    "\u03b3": r"\ensuremath{\gamma}", "\u03c3": r"\ensuremath{\sigma}",
    "\u03a3": r"\ensuremath{\Sigma}", "\u03ba": r"\ensuremath{\kappa}",
    "\u03c0": r"\ensuremath{\pi}", "\u03b1": r"\ensuremath{\alpha}",
    "\u03b2": r"\ensuremath{\beta}", "\u03bc": r"\ensuremath{\mu}",
    "\u03c1": r"\ensuremath{\rho}", "\u03b5": r"\ensuremath{\varepsilon}",
    "\u2212": r"\ensuremath{-}", "\u00d7": r"\ensuremath{\times}",
    "\u00b2": r"\ensuremath{^2}", "\u00b1": r"\ensuremath{\pm}",
    "\u2264": r"\ensuremath{\le}", "\u2265": r"\ensuremath{\ge}",
    "\u2261": r"\ensuremath{\equiv}", "\u2248": r"\ensuremath{\approx}",
    "\u2208": r"\ensuremath{\in}", "\u2286": r"\ensuremath{\subseteq}",
    "\u222a": r"\ensuremath{\cup}", "\u2294": r"\ensuremath{\sqcup}",
    "\u2216": r"\ensuremath{\setminus}", "\u2218": r"\ensuremath{\circ}",
    "\u21d2": r"\ensuremath{\Rightarrow}", "\u2192": r"\ensuremath{\to}",
    "\u221d": r"\ensuremath{\propto}", "\u27c2": r"\ensuremath{\perp}",
    "\u2202": r"\ensuremath{\partial}", "\u2016": r"\ensuremath{\|}",
    "\U0001d504": r"\ensuremath{\mathfrak{A}}",
    "\U0001d505": r"\ensuremath{\mathfrak{B}}",
    "\u212d": r"\ensuremath{\mathfrak{C}}",
    "\U0001d513": r"\ensuremath{\mathfrak{P}}",
    "\u00c2": r"\ensuremath{\hat{A}}",
}
HAT = "\u0302"
BASE = {"A": "A", "B": "B", "C": "C", "S": "S", "m": "m", "q": "q", "p": "p",
        "\u03c4": r"\tau", "\u03b8": r"\theta", "\u0394": r"\Delta",
        "\u03bb": r"\lambda", "\u03c9": r"\omega", "\u03b4": r"\delta"}


CMDGLUE = re.compile(
    r"\\(" + "|".join([
        "mathfrak", "blacksquare", "varepsilon", "Rightarrow", "subseteq",
        "setminus", "partial", "approx", "equiv", "Delta", "Sigma", "omega",
        "theta", "delta", "sigma", "lambda", "times", "kappa", "alpha",
        "gamma", "propto", "perp", "cdot", "beta", "perp", "rho", "tau",
        "mu", "pi", "le", "ge", "in", "to", "pm", "hat",
    ]) + r")(?=[A-Za-z])")


def fix_text(chunk):
    def hat(m):
        b = BASE.get(m.group(1))
        return m.group(0) if b is None else r"\ensuremath{\hat{%s}}" % b
    chunk = re.sub("(.)" + HAT, hat, chunk)
    for k, v in SINGLE.items():
        # a trailing space keeps \Delta from gluing onto a following letter and
        # is inert inside math mode
        chunk = chunk.replace(k, v + " " if v.startswith("\\ensuremath") else v)
    chunk = chunk.replace("|", r"\textbar{}")   # literal pipe in text mode
    return chunk


def fix_outside_verbatim(body):
    out, pos = [], 0
    for m in re.finditer(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", body, re.S):
        out.append(fix_text(body[pos:m.start()]))
        # code listings must stay ASCII for pdflatex
        out.append(m.group(0).encode("ascii", "replace").decode("ascii")
                   .replace("?", ""))
        pos = m.end()
    out.append(fix_text(body[pos:]))
    return "".join(out)


def strip_number(title):
    return re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", title).strip()


def demote(body):
    body = body.replace("\\subsubsection{", "\\SUBSECTIONMARK{")
    body = body.replace("\\subsection{", "\\section{")
    body = body.replace("\\SUBSECTIONMARK{", "\\subsection{")
    return body


def main():
    tex = io.open(SRC, encoding="utf-8").read()

    m_title = re.search(r"\\section\{([^}]*)\}", tex)
    title = m_title.group(1)
    m_sub = re.search(r"\\subsection\{([^}]*)\}", tex)
    subtitle = m_sub.group(1)

    # abstract = body of the "Abstract" subsection
    m_abs = re.search(
        r"\\(?:sub)*section\{Abstract\}(.*?)(?=\\subsection\{1\. Introduction\})",
        tex, re.S)
    abstract = m_abs.group(1).strip()
    abstract = re.sub(r"\\begin\{center\}.*?\\end\{center\}", "", abstract,
                      flags=re.S).strip()

    body = tex[m_abs.end():]                       # starts at "1. Introduction"
    body = body.split("\\subsection{References}")[0]

    body = demote(body)

    # strip leading numbers from section titles so LaTeX numbers them
    def ren(m):
        return "\\section{%s}" % strip_number(m.group(1))
    body = re.sub(r"\\section\{([^}]*)\}", ren, body)
    body = re.sub(r"\\subsection\{([^}]*)\}",
                  lambda m: "\\subsection{%s}" % strip_number(m.group(1)), body)

    # headings whose title pandoc wrapped in \texorpdfstring{} keep the number
    def _strip_pair(m):
        return "\\texorpdfstring{%s}{%s}" % (strip_number(m.group(1)),
                                             strip_number(m.group(2)))
    body = re.sub(r"\\texorpdfstring\{([^{}]*)\}\{([^{}]*)\}", _strip_pair, body)

    # appendix
    body = body.replace(
        "\\section{Appendix A: computing the census and the estimators}",
        "\\appendix\n\\section{Computing the census and the estimators}")
    body = body.replace(
        "\\section{Appendix B: artefacts}", "\\section{Artefacts}")

    body = fix_outside_verbatim(body)

    sys.path.insert(0, r"E:\program\python\paper")
    import _texmathify
    body = _texmathify.apply(body)

    head = r"""%% =====================================================================
%%  What Does One AUC Point Buy?
%%  Decomposing, Identifying, and Reporting Localization Ability in
%%  Weakly-Supervised Video Anomaly Detection
%%
%%  Manuscript prepared for submission to
%%  Computer Vision and Image Understanding (Elsevier).
%%  Build:  pdflatex CAFE && bibtex CAFE && pdflatex CAFE && pdflatex CAFE
%% =====================================================================
\documentclass[authoryear,preprint,12pt]{elsarticle}

\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{longtable,booktabs,array}\usepackage{calc}
\usepackage{url}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

%% keep the wide tables and the code listing inside the text block
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\AtBeginEnvironment{verbatim}{\scriptsize}
\setlength{\tabcolsep}{4pt}
\setlength{\emergencystretch}{2em}

\journal{Computer Vision and Image Understanding}

\begin{document}

\begin{frontmatter}

\title{@@TITLE@@}

\author{@@AUTHOR@@}
\address{@@ADDRESS@@}

\begin{abstract}
@@ABSTRACT@@
\end{abstract}

\begin{keyword}
weakly-supervised video anomaly detection \sep evaluation metrics \sep
frame-level AUC \sep partial identification \sep sensitivity analysis
\end{keyword}

\end{frontmatter}

@@BODY@@

\nocite{*}% all cited works are named in prose; the list is generated from references.bib
\bibliographystyle{elsarticle-harv}
\bibliography{references}

\end{document}
"""

    head = (head.replace("@@TITLE@@", title + r" \\ \large " + subtitle)
                .replace("@@AUTHOR@@", "Anonymous Authors")
                .replace("@@ADDRESS@@", "Manuscript under review")
                .replace("@@ABSTRACT@@",
                         _texmathify.rewrite_texttt(fix_text(abstract)))
                .replace("@@BODY@@", body))

    os.makedirs(OUTDIR, exist_ok=True)
    io.open(DST, "w", encoding="utf-8", newline="\n").write(head)
    print("wrote", DST, len(head), "chars")
    left = sorted({c for c in head if ord(c) > 127})
    if left:
        print("UNMAPPED:", [(c, hex(ord(c))) for c in left])


if __name__ == "__main__":
    main()





