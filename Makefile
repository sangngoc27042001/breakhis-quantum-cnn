.PHONY: setup prepare-dataset train clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup           - Install uv, create virtual environment, and install dependencies"
	@echo "  prepare-dataset - Download and prepare the dataset"
	@echo "  train           - Train the model"
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

# Clean up environment
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
