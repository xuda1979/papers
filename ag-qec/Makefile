PY=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PYTHON=$(VENV)/bin/python

.PHONY: env data figs pdf clean

env:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) simulation.py --config configs/depolarizing.yaml --outdir results/dep

figs: data

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error ag-qec.tex

clean:
	latexmk -C
	rm -rf results
