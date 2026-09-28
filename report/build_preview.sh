#!/bin/bash
# Build report/proposal_preview.pdf. Uses a temporary copy with the newtx font line swapped for amssymb,
# because newtx is not installed on this machine. Run from anywhere: bash report/build_preview.sh
cd "$(dirname "$0")"
sed 's/\\usepackage{newtxtext,newtxmath}/\\usepackage{amssymb}/; s/\\renewcommand{\\ttdefault}{lmtt}//' report.tex > _preview_build.tex
pdflatex -interaction=nonstopmode _preview_build.tex >/dev/null
pdflatex -interaction=nonstopmode _preview_build.tex | grep -E -A3 "^!|undefined"
[ -f _preview_build.pdf ] || { echo BUILD FAILED; exit 1; }
mv _preview_build.pdf report.pdf
rm -f _preview_build.*
pdfinfo report.pdf | grep Pages
