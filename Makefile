PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup download mock preprocess eval simulate figures report demo test smoke clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

download:
	$(PYTHON) -m foodflow.cli download --skip-graph

mock:
	$(PYTHON) -m foodflow.cli mock-data --raw-dir data/raw --seed 42

preprocess:
	$(PYTHON) -m foodflow.cli preprocess --raw-dir data/raw --processed-dir data/processed --sample-orders 50000 --seed 42

eval:
	$(PYTHON) -m foodflow.cli eval-offline --processed-dir data/processed --output outputs/results/offline_metrics.csv --top-k 10 20 --seed 42

simulate:
	$(PYTHON) -m foodflow.cli simulate --processed-dir data/processed --output outputs/results/simulation_metrics.csv --seed 42

figures:
	$(PYTHON) -m foodflow.cli figures --results-dir outputs/results --figures-dir outputs/figures

report:
	$(PYTHON) -m foodflow.cli report --results-dir outputs/results --figures-dir outputs/figures --output report/实验报告.md

demo:
	.venv/bin/streamlit run app.py

test:
	$(PYTHON) -m pytest -q

smoke: mock preprocess eval simulate figures report test

clean:
	rm -rf data/raw/*.txt data/processed/*.csv outputs/results/*.csv outputs/figures/*.png report/实验报告.md
