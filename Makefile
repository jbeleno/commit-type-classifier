.PHONY: install data balance train-baseline train-cnn train-distilbert train-codebert train-ensemble train-all eval-all test app cli diagrams clean

UV ?= uv

install:
	$(UV) sync --extra dev

data:
	$(UV) run python -m src.data.download --sample 1000000
	$(UV) run python -m src.data.preprocess
	$(UV) run python -m src.data.split

balance:
	$(UV) run python -m src.data.balance --target 1600

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

diagrams:
	plantuml -tpng -o ../png docs/diagrams/puml/*.puml

docs-pdf:
	mkdir -p docs/exports
	pandoc docs/documentation.md \
	  -o docs/exports/documentation.pdf \
	  --toc --number-sections \
	  --pdf-engine=typst \
	  --resource-path=docs:docs/diagrams/png

slides:
	mkdir -p docs/exports
	$(UV) run python scripts/build_slides.py

docs-docx:
	mkdir -p docs/exports
	pandoc docs/documentation.md \
	  -o docs/exports/documentation.docx \
	  --toc --number-sections \
	  --resource-path=docs:docs/diagrams/png

clean:
	rm -rf data/raw data/processed data/splits models_saved db .venv
