PAPER=paper/black_entangle.tex
PDF=$(PAPER:.tex=.pdf)

.PHONY: all clean run

all: $(PDF)

$(PDF): $(PAPER) bib/references.bib
	latexmk -pdf -halt-on-error -interaction=nonstopmode $(PAPER)

run:
	python3 code/simulate_tpchshr.py --spin 0.6 --bmin 3.0 --bmax 8.0 --samples 40

clean:
	latexmk -C
	rm -f paper/*.pdf
	rm -rf results
