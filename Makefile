PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
CONDA_ENV ?= foodflow
CONDA_RUN ?= conda run --no-capture-output -n $(CONDA_ENV)
STREAMLIT ?= $(CONDA_RUN) streamlit
STREAMLIT_FLAGS ?=
SMOKE_RAW ?= data/sample/raw
SMOKE_PROCESSED ?= data/sample/processed
SMOKE_RESULTS ?= outputs/smoke/results
SMOKE_FIGURES ?= outputs/smoke/figures
SMOKE_REPORT ?= outputs/smoke/report.md
SEQ_SEARCH_RESULTS ?= outputs/experiments/seq_weight_search_smoke.csv

.PHONY: setup conda-setup conda-test conda-smoke seq-tune-smoke download mock preprocess preprocess-full eval simulate audit figures report notebooklm-pack demo demo-full demo-check test smoke clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

conda-setup:
	conda env update -f environment.yml --prune

conda-test:
	$(CONDA_RUN) python -m pytest -q

seq-tune-smoke:
	$(CONDA_RUN) python scripts/search_seq_weights.py --processed-dir data/processed --output $(SEQ_SEARCH_RESULTS) --user-limit 20 --candidate-limit 60 --trials 1 --seed 2026

conda-smoke:
	$(CONDA_RUN) python -m foodflow.cli mock-data --raw-dir $(SMOKE_RAW) --seed 42
	$(CONDA_RUN) python -m foodflow.cli preprocess --raw-dir $(SMOKE_RAW) --processed-dir $(SMOKE_PROCESSED) --sample-orders 50000 --seed 42
	$(CONDA_RUN) python -m foodflow.cli eval-offline --processed-dir $(SMOKE_PROCESSED) --output $(SMOKE_RESULTS)/offline_metrics.csv --top-k 10 20 --seed 42
	$(CONDA_RUN) python -m foodflow.cli simulate --processed-dir $(SMOKE_PROCESSED) --output $(SMOKE_RESULTS)/simulation_metrics.csv --seed 42
	$(CONDA_RUN) python -m foodflow.cli audit-data --raw-dir $(SMOKE_RAW) --processed-dir $(SMOKE_PROCESSED) --output $(SMOKE_RESULTS)/data_audit.json --markdown outputs/smoke/DATA_AUDIT.md
	$(CONDA_RUN) python -m foodflow.cli figures --results-dir $(SMOKE_RESULTS) --figures-dir $(SMOKE_FIGURES)
	$(CONDA_RUN) python -m foodflow.cli report --results-dir $(SMOKE_RESULTS) --figures-dir $(SMOKE_FIGURES) --output $(SMOKE_REPORT) --data-note $(SMOKE_PROCESSED)/data_note.json --data-audit $(SMOKE_RESULTS)/data_audit.json
	$(CONDA_RUN) python -m pytest -q

download:
	$(PYTHON) -m foodflow.cli download --skip-graph

mock:
	$(PYTHON) -m foodflow.cli mock-data --raw-dir data/raw --seed 42

preprocess:
	$(PYTHON) -m foodflow.cli preprocess --raw-dir data/raw --processed-dir data/processed --sample-orders 50000 --seed 42

preprocess-full:
	$(PYTHON) -m foodflow.cli preprocess --raw-dir data/raw --processed-dir data/processed --sample-orders 0 --seed 42

eval:
	$(PYTHON) -m foodflow.cli eval-offline --processed-dir data/processed --output outputs/results/offline_metrics.csv --top-k 10 20 --seed 42

simulate:
	$(PYTHON) -m foodflow.cli simulate --processed-dir data/processed --output outputs/results/simulation_metrics.csv --seed 42

audit:
	$(PYTHON) -m foodflow.cli audit-data --raw-dir data/raw --processed-dir data/processed --output outputs/results/data_audit.json --markdown docs/DATA_AUDIT.md

figures:
	$(PYTHON) -m foodflow.cli figures --results-dir outputs/results --figures-dir outputs/figures

report:
	$(PYTHON) -m foodflow.cli report --results-dir outputs/results --figures-dir outputs/figures --output report/实验报告.md --data-audit outputs/results/data_audit.json

notebooklm-pack:
	$(PYTHON) scripts/prepare_notebooklm_pack.py

demo-check:
	@printf "\nChecking common Streamlit ports...\n"
	@if command -v lsof >/dev/null 2>&1; then \
		lsof -nP -iTCP:8501 -sTCP:LISTEN || true; \
		lsof -nP -iTCP:8502 -sTCP:LISTEN || true; \
	else \
		printf "lsof is not installed; skip port listing.\n"; \
	fi
	@printf "If the browser still shows an old page, stop the old Streamlit terminal or rerun demo on another port.\n\n"

demo:
	@$(MAKE) --no-print-directory demo-check
	@printf "\nFoodFlow demo is a long-running Streamlit web server, not a batch command.\n"
	@printf "Keep this terminal open, then visit the Streamlit Local URL below. Default is http://localhost:8501 .\n"
	@printf "Press Ctrl+C here to stop.\n\n"
	$(STREAMLIT) run app.py --server.headless true $(STREAMLIT_FLAGS)

demo-full:
	@$(MAKE) --no-print-directory demo-check
	@printf "\nFoodFlow demo-full uses all processed train orders and may take longer on first load.\n"
	@printf "Keep this terminal open, then visit the Streamlit Local URL below. Default is http://localhost:8501 .\n"
	@printf "Press Ctrl+C here to stop.\n\n"
	FOODFLOW_DEMO_MAX_ORDERS=0 $(STREAMLIT) run app.py --server.headless true $(STREAMLIT_FLAGS)

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m foodflow.cli mock-data --raw-dir $(SMOKE_RAW) --seed 42
	$(PYTHON) -m foodflow.cli preprocess --raw-dir $(SMOKE_RAW) --processed-dir $(SMOKE_PROCESSED) --sample-orders 50000 --seed 42
	$(PYTHON) -m foodflow.cli eval-offline --processed-dir $(SMOKE_PROCESSED) --output $(SMOKE_RESULTS)/offline_metrics.csv --top-k 10 20 --seed 42
	$(PYTHON) -m foodflow.cli simulate --processed-dir $(SMOKE_PROCESSED) --output $(SMOKE_RESULTS)/simulation_metrics.csv --seed 42
	$(PYTHON) -m foodflow.cli audit-data --raw-dir $(SMOKE_RAW) --processed-dir $(SMOKE_PROCESSED) --output $(SMOKE_RESULTS)/data_audit.json --markdown outputs/smoke/DATA_AUDIT.md
	$(PYTHON) -m foodflow.cli figures --results-dir $(SMOKE_RESULTS) --figures-dir $(SMOKE_FIGURES)
	$(PYTHON) -m foodflow.cli report --results-dir $(SMOKE_RESULTS) --figures-dir $(SMOKE_FIGURES) --output $(SMOKE_REPORT) --data-note $(SMOKE_PROCESSED)/data_note.json --data-audit $(SMOKE_RESULTS)/data_audit.json
	$(PYTHON) -m pytest -q

clean:
	rm -rf data/sample outputs/smoke data/processed/*.csv data/processed/data_note.json outputs/results/*.csv outputs/results/*.json outputs/figures/*.png report/实验报告.md docs/DATA_AUDIT.md
