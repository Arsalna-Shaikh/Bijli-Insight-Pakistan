# Makefile: Single-command CLI runner for reproducibility

.PHONY: data test pipeline app

# 1. Rebuild entire 3-tier warehouse from scratch
data:
	python -m backend.pipeline.warehouse_pipeline

# 2. Run all unit tests and contract assertions
test:
	python -m pytest tests/ -v

# 3. Execute one cycle of the hourly pipeline
pipeline:
	python -m backend.pipeline.flow

# 4. Launch the Streamlit dashboard locally
app:
	streamlit run frontend/app.py