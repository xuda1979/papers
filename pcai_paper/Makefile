# Minimal Makefile to build the LaTeX paper
# Requires: latexmk, pdflatex
.PHONY: all clean pdf

all: pdf

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error pcai_paper.tex

clean:
	latexmk -C
