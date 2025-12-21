.PHONY: setup prepare-dataset train train-several zip-results clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup           - Install uv, create virtual environment, and install dependencies"
	@echo "  prepare-dataset - Download and prepare the dataset"
	@echo "  train           - Train the model"
	@echo "  train-several   - Train all CNN backbones sequentially"
	@echo "  zip-results     - Compress the results folder into a zip archive"
	@echo "  clean           - Remove virtual environment and cache files"

# Setup: Install uv, create venv, and install dependencies
setup:
	pip install uv
	uv venv
	uv pip install -r requirements.txt

# Prepare dataset
prepare-dataset:
	uv run python -m src.download_dataset

# Train model
train:
	uv run python -m src.train

# Train all CNN backbones sequentially
train-several:
	uv run python -m src.train_several_models

# Zip results folder
zip-results:
	uv run python src/zip_results.py

# Clean up environment
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
