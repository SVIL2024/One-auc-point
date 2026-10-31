# -*- coding: utf-8 -*-
"""Rebuild CAFE.tex from the v6 markdown.

markdown (single source of truth)
  -> pandoc (--no-highlight --wrap=preserve) -> elsarticle/body_raw.tex
  -> build/_build_elsarticle.py (preamble, section demotion, abstract)
  -> build/_citetify.py (\\cite -> \\citep/\\citet against references.bib)
  -> pdflatex x3 + bibtex
"""
import io
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MD = r"E:\paper\drafts\2026\CAFE\paper_CAFE_v6.md"
ELS = r"E:\paper\drafts\2026\CAFE\elsarticle"
BUILD = os.path.join(ELS, "build")
RAW = os.path.join(ELS, "body_raw.tex")
TEX = os.path.join(ELS, "CAFE.tex")
MKDIR = r"D:\Program Files\CTEX\MiKTeX\miktex\bin"


def run(cmd, cwd=None, shell=False):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, shell=shell, capture_output=True,
                       text=True, errors="replace")
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit("FAILED: %s" % cmd)
    return r


def main():
    if os.path.exists(TEX):
        shutil.copy2(TEX, TEX + ".prerebuild")
    run(["pandoc", "-f", "markdown", "-t", "latex", "--no-highlight",
         "--wrap=preserve", "-o", RAW, MD])
    print("body_raw.tex bytes:", os.path.getsize(RAW))
    run([sys.executable, os.path.join(BUILD, "_build_elsarticle.py")])
    run([sys.executable, os.path.join(BUILD, "_citetify.py")])
    print("CAFE.tex bytes:", os.path.getsize(TEX))
    tex = io.open(TEX, encoding="utf-8").read()
    print("documentclass authoryear:", "documentclass[authoryear" in tex)
    for cmd in ("pdflatex -interaction=nonstopmode CAFE.tex",):
        for i in range(2):
            run(cmd, cwd=ELS, shell=True)
    run("bibtex CAFE", cwd=ELS, shell=True)
    for i in range(2):
        run("pdflatex -interaction=nonstopmode CAFE.tex", cwd=ELS, shell=True)
    log = io.open(os.path.join(ELS, "CAFE.log"), encoding="utf-8",
                  errors="replace").read()
    pages = [l for l in log.split("\n") if "Output written" in l]
    print(pages)
    print("undefined refs:", log.count("undefined"))
    print("overfull:", log.count("Overfull"))


if __name__ == "__main__":
    main()
