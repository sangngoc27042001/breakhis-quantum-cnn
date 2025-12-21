.PHONY: setup setup-lightning-gpu prepare-dataset train train-several zip-results clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup               - Install uv, create virtual environment, and install dependencies"
	@echo "  setup-lightning-gpu - Build and install PennyLane Lightning-GPU from source (run after setup)"
	@echo "  prepare-dataset     - Download and prepare the dataset"
	@echo "  train               - Train the model"
	@echo "  train-several       - Train all CNN backbones sequentially"
	@echo "  zip-results         - Compress the results folder into a zip archive"
	@echo "  clean               - Remove virtual environment and cache files"

# Setup: Install uv, create venv, and install base dependencies
setup:
	pip install uv
	uv venv
	uv pip install -r requirements.txt
	@echo ""
	@echo "Base dependencies installed. Now run 'make setup-lightning-gpu' to build Lightning-GPU from source."

# Build and install Lightning-GPU from source
setup-lightning-gpu:
	@echo "Installing PennyLane from master branch..."
	uv pip install git+https://github.com/PennyLaneAI/pennylane.git@master
	@echo ""
	@echo "Cloning PennyLane Lightning repository..."
	rm -rf pennylane-lightning
	git clone https://github.com/PennyLaneAI/pennylane-lightning.git
	@echo ""
	@echo "Setting up CUQUANTUM_SDK environment variable..."
	$(eval CUQUANTUM_SDK := $(shell uv run python -c "import site; print(f'{site.getsitepackages()[0]}/cuquantum')"))
	@echo "CUQUANTUM_SDK=$(CUQUANTUM_SDK)"
	@echo ""
	@echo "Installing build dependencies..."
	cd pennylane-lightning && uv pip install -r requirements.txt
	@echo ""
	@echo "Step 1: Building Lightning-Qubit (base package, without compilation)..."
	cd pennylane-lightning && \
		PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py && \
		SKIP_COMPILATION=True uv pip install . -vv
	@echo ""
	@echo "Step 2: Building Lightning-GPU from source..."
	cd pennylane-lightning && \
		export CUQUANTUM_SDK=$(CUQUANTUM_SDK) && \
		PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py && \
		uv pip install . -vv
	@echo ""
	@echo "Lightning-GPU installation complete!"
	@echo "Verifying installation..."
	uv run python -c "import pennylane as qml; print('PennyLane version:', qml.__version__); dev = qml.device('lightning.gpu', wires=2); print('Lightning-GPU device created successfully!')"

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
	rm -rf pennylane-lightning
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
