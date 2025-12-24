# Master Thesis - Model Training Results

## Quick Start

### Setup and Training
```bash
# Install dependencies
make setup

# Download and prepare dataset
make prepare-dataset

# Train models
make train                    # Train single model
make train-several           # Train multiple models concurrently
make training-status         # Check training progress
```

### Evaluation Commands

#### Backbone Timing Benchmark
Evaluate training and inference time for each backbone model (parameters, memory, speed):
```bash
make evaluate-backbone-timing
# Or directly:
uv run python -m src.evaluate.evaluate_backbone_timing
```

#### Compare Model Performance
Compare all trained models across train/val/test accuracy and generalization metrics:
```bash
make compare-models
# Or directly:
uv run python -m src.evaluate.compare_models
```

#### Evaluate Specific Model
Load and evaluate a specific trained model on the test set with architecture summary:
```bash
make evaluate-model MODEL_DIR=results_quantum_cnn/cnn_quantum_mobilenetv3_small_100_dense-rotation_two_design_depth-5_qubits-8_20251222_222519
# Or directly:
uv run python -m src.evaluate.evaluate_model --model_dir results_quantum_cnn/cnn_quantum_mobilenetv3_small_100_dense-rotation_two_design_depth-5_qubits-8_20251222_222519
```

#### Paired Hypothesis Testing
Advanced paired statistical testing for quantum CNN hyperparameters using sample-level predictions (McNemar's test, Cochran's Q test, per-class and difficulty-stratified analysis):
```bash
make paired-hypothesis-test
# Or directly:
uv run python -m src.evaluate.paired_hypothesis_testing
```

#### McNemar Test: Quantum vs Classical
Statistical comparison of quantum-enhanced CNN vs classical CNN using McNemar's test for binary classification (cancer vs benign):
```bash
make mcnemar-test
# Or directly:
uv run python -m src.evaluate.mcnemar_test_quantum_vs_classical
```

### Utilities
```bash
make zip-results             # Compress results folder
make git-reset-pull          # Hard reset and pull latest changes
make clean                   # Remove virtual environment and cache files
```

## Makefile Reference

```makefile
.PHONY: setup prepare-dataset train train-several training-status evaluate-backbone-timing compare-models evaluate-model paired-hypothesis-test mcnemar-test zip-results clean git-reset-pull help

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

# Train combinations concurrently (safe for multiple terminals)
train-several:
	uv run python -m src.train_backbone_models

# Check training status
training-status:
	uv run python -m src.training_status

# Evaluate backbone timing (parameters, inference, training time)
evaluate-backbone-timing:
	uv run python -m src.evaluate.evaluate_backbone_timing

# Compare all trained models
compare-models:
	uv run python -m src.evaluate.compare_models

# Evaluate specific model (requires MODEL_DIR variable)
evaluate-model:
	uv run python -m src.evaluate.evaluate_model --model_dir $(MODEL_DIR)

# Paired hypothesis testing for quantum hyperparameters
paired-hypothesis-test:
	uv run python -m src.evaluate.paired_hypothesis_testing

# McNemar test: quantum vs classical
mcnemar-test:
	uv run python -m src.evaluate.mcnemar_test_quantum_vs_classical

# Zip results folder
zip-results:
	uv run python src/zip_results.py

# Git reset and pull
git-reset-pull:
	git reset --hard
	git pull

# Clean up environment
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

## Training Times

| Model | Parameters (M) | Trainable Params (M) | Inference Time - Single (ms) | Inference Time - Batch (ms) | Inference Time - Per Sample (ms) | Training Epoch Time (sec) | Training Epoch Time (min) | Memory (MB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mobilenetv3_small_100 | 1.528 | 1.528 | 4.48 | 18.61 | 0.073 | 25.11 | 0.42 | 27.94 |
| regnetx_002 | 2.319 | 2.319 | 4.30 | 32.89 | 0.128 | 25.81 | 0.43 | 32.10 |
| regnety_002 | 2.798 | 2.798 | 6.44 | 35.52 | 0.139 | 26.51 | 0.44 | 33.94 |
| ghostnet_100 | 3.914 | 3.914 | 8.49 | 57.10 | 0.223 | 30.59 | 0.51 | 40.56 |
| mnasnet_100 | 3.115 | 3.115 | 4.33 | 60.20 | 0.235 | 30.61 | 0.51 | 28.69 |
| efficientnet_lite0 | 3.384 | 3.384 | 4.39 | 72.29 | 0.282 | 37.08 | 0.62 | 43.22 |
| mobilevit_xs | 1.937 | 1.937 | 7.29 | 146.74 | 0.573 | 55.98 | 0.93 | 40.63 |