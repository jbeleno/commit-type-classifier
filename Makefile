.PHONY: install data train-baseline train-cnn train-distilbert train-codebert train-ensemble train-all eval-all test app cli clean

UV ?= uv

install:
	$(UV) sync --extra dev

data:
	$(UV) run python -m src.data.download --sample 1000000
	$(UV) run python -m src.data.preprocess
	$(UV) run python -m src.data.split

train-baseline:
	$(UV) run python -m src.models.baseline_tfidf all

train-cnn:
	$(UV) run python -m src.models.cnn_text all

train-distilbert:
	$(UV) run python -m src.models.distilbert_model all --max-train 5000

train-codebert:
	$(UV) run python -m src.models.codebert_model all --max-train 5000

train-ensemble:
	$(UV) run python -m src.models.ensemble all

train-all: train-baseline train-cnn train-distilbert train-codebert train-ensemble

eval-all:
	$(UV) run python -m src.evaluate_all

test:
	$(UV) run pytest tests/ -q

app:
	$(UV) run streamlit run app/streamlit_app.py

cli:
	$(UV) run python -m app.cli --help

clean:
	rm -rf data/raw data/processed data/splits models_saved db .venv
