PY=python3
SEED?=1337
N?=255
RX?=110
RZ?=112
TRIALS?=2000
ETA_LIST?=0,0.3,0.6
PX?=0.02
PZ?=0.04
PY_ERR?=0.0
ITERS?=100
DAMP?=0.5
OSD?=2
DATA_DIR?=data
FIG_DIR?=figs

export PYTHONHASHSEED=0

.PHONY: all fast codes simulate figs baseline clean

all: codes simulate figs

fast: RX=60
fast: RZ=62
fast: TRIALS=200
fast: ETA_LIST=0
fast: ITERS=50
fast: DAMP=0.3
fast: OSD=1
fast: codes simulate figs

codes:
	@mkdir -p $(DATA_DIR)
	$(PY) code/generate_css.py --n $(N) --rX $(RX) --rZ $(RZ) --seed $(SEED) --out $(DATA_DIR)/css_n$(N)_seed$(SEED).npz --stats $(DATA_DIR)/stats_seed$(SEED).json

simulate:
	@mkdir -p $(DATA_DIR)
	$(PY) code/experiment.py \
	--css $(DATA_DIR)/css_n$(N)_seed$(SEED).npz \
	--results $(DATA_DIR)/results.csv \
	--seed $(SEED) \
	--trials $(TRIALS) \
	--eta-list $(ETA_LIST) \
	--pX $(PX) --pZ $(PZ) --pY $(PY_ERR) \
	--iters $(ITERS) --damping $(DAMP) --osd $(OSD)

baseline:
	@mkdir -p $(DATA_DIR)
	$(PY) code/experiment.py --baseline-lp --results $(DATA_DIR)/baseline_lp.csv --trials 500 --seed $(SEED)

figs:
	@mkdir -p $(FIG_DIR)
	$(PY) plots/make_plots.py --results $(DATA_DIR)/results.csv --outdir $(FIG_DIR)

clean:
	rm -rf $(DATA_DIR) $(FIG_DIR)

