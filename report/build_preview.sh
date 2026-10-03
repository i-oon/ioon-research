#!/bin/bash
# Build the same source and fonts as Overleaf (Compiler: XeLaTeX).
set -e
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null
xelatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null
pdfinfo report.pdf | grep Pages
