PY=python

.PHONY: all matrices verify simulate plot pdf clean

all: matrices verify simulate plot pdf

matrices:
	$(PY) scripts/generate_css.py --seed 20250825 --n 255 --rx 111 --rz 111 --col_wx 3 --col_wz 3

verify:
	$(PY) scripts/verify_css.py

simulate:
	$(PY) scripts/run_experiment.py \
		--seed 20250825 --n 255 --rx 111 --rz 111 \
		--pX 0.01 --pZ 0.03 --pY 0.00 --eta 0.60 \
		--trials 50000 --iters 60 --damp 0.5 --osd True --osd_topk 8 \
		--out data/logical_error_rate.csv

plot:
	$(PY) scripts/plot_logical_error_rate.py --in data/logical_error_rate.csv --out figs/logical_error_rate.png

pdf:
	@which latexmk >/dev/null 2>&1 && latexmk -pdf -silent ag-qec.tex || echo "Install LaTeX/latexmk to build PDF."

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk ag-qec.pdf
