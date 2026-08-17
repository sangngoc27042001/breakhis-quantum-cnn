# Quantum-Enhanced CNN for Highly Accurate Breast Cancer Classification: A Proof-of-Concept Study

**Replication package for the Master's Thesis of Sang Vo, University of Oulu, 2025–2026 academic year.**

- **Degree programme:** Information Processing Science, University of Oulu
- **Author:** Sang Vo
- **Supervisor:** Assoc. Prof. Arif Ali Khan · **Co-supervisor:** Mr. Boshuai Ye
- **Industry collaborators (TECHNIA):** Mr. Ghassan Sultan, Mr. Rupayan Mukherjee, Mr. Johannes Storvik
- **Thesis PDF:** [`Finalised_Thesis_SangVo.pdf`](Finalised_Thesis_SangVo.pdf)

This project was carried out **in collaboration with [TECHNIA](https://www.technia.com/)** — with coordination and
industry guidance from Mr. Ghassan Sultan, Mr. Rupayan Mukherjee, and Mr. Johannes Storvik — and was **supported by a
grant from the TechFinland100 Foundation** under its programme
[Grants for Master's theses on AI topics](https://techfinland100.fi/grants-for-masters-theses-on-ai-topics/).

## About the study

The thesis investigates whether appending a variational quantum layer to a classical CNN yields real, measurable gains
in breast cancer histopathology classification. Experiments use the **BreakHis** dataset (7,909 microscopy images,
8 tumour subtypes) and proceed in four stages:

1. **Classical backbone selection** — seven lightweight ImageNet-pretrained backbones (<7M parameters).
2. **Quantum configuration search** — grid over circuit template, qubit count, and circuit depth.
3. **Statistical analysis** — paired significance testing over sample-level predictions.
4. **Quantum vs. classical comparison** — McNemar's test on the binary cancer/benign decision.

Headline results: the best classical model (`regnetx_002`) reaches **90.80%** test accuracy, while the best hybrid model
(`mobilenetv3_small_100` + SimplifiedTwoDesign, 8 qubits, 3 layers) reaches **92.72%** — a **6.42 pp** gain over its own
classical counterpart. The quantum layer improved accuracy on all seven backbones (p < 0.001), and circuit *template*
mattered considerably more than qubit count (58.3% vs. 16.7% of comparisons statistically significant). Full numbers,
methodology, and threats to validity are in the thesis PDF.

## Repository layout

```
src/
├── config.py                       # All hyperparameters, paths, quantum circuit settings
├── download_dataset.py             # BreakHis download + split + balancing
├── data_preparation.py             # Preprocessing (224×224, augmentation-based balancing)
├── breakhis_data_loader.py         # PyTorch data pipeline
├── train.py                        # Train a single model
├── train_backbone_models.py        # Stage 1/2 sweeps, safe to run from several terminals
├── train_several_models.py         # Multi-config training driver
├── training_status.py              # Sweep progress reporting
├── model_implementations/
│   ├── small_models.py             # Classical backbones (timm)
│   ├── cnn_classical.py            # Classical baseline head
│   └── cnn_quantum.py              # Hybrid CNN + quantum dense layer
├── utils/
│   ├── quantum_dense_layer/        # Variational quantum dense layer (PennyLane)
│   └── quantum_ring_rotation/      # Ring-rotation encoding layer
└── evaluate/                       # Benchmarking and statistical analysis scripts

results/                            # Stage 1: classical backbone runs
results_quantum_config/             # Stage 2: quantum hyperparameter grid
results_quantum_cnn/                # Stage 3/4: best config across all backbones
mcnemar_test_results.json           # Quantum vs. classical significance results
performance_comparison.csv          # Aggregated model comparison table
```

Each run directory contains `config.json`, `training_history.json`, per-epoch metrics, sample-level predictions
(`epoch20_detail_predictions.csv`, used by the paired statistical tests), model weights, and TensorBoard logs.

## Requirements

Python 3.10+, PyTorch, `timm`, and PennyLane (see [`requirements.txt`](requirements.txt)). A CUDA GPU is strongly
recommended — the reported runs were produced on an NVIDIA V100. The quantum layer runs on PennyLane simulators; no
quantum hardware is required.

## Quick start

```bash
# 1. Install uv, create the virtual environment, install dependencies
make setup

# 2. Download and prepare the BreakHis dataset (once)
make prepare-dataset

# 3. Train
make train                    # single model, as configured in src/config.py
make train-several            # sweep of combinations, concurrency-safe
make training-status          # sweep progress
```

Model choice and all quantum settings (`QUANTUM_CNN_CONFIG_*`: backbone, qubit count, encoding, circuit template,
depth) are set in [`src/config.py`](src/config.py).

## Reproducing the analysis

### Backbone timing benchmark (Stage 1)
Parameters, memory, and training/inference speed for each backbone:
```bash
make evaluate-backbone-timing
# uv run python -m src.evaluate.evaluate_backbone_timing
```

### Model comparison
Train/val/test accuracy and generalization metrics across all trained models:
```bash
make compare-models
# uv run python -m src.evaluate.compare_models
```

### Evaluate a specific run
```bash
make evaluate-model MODEL_DIR=results_quantum_cnn/cnn_quantum_mobilenetv3_small_100_dense-rotation_two_design_depth-3_qubits-8_20251230_192507
# uv run python -m src.evaluate.evaluate_model --model_dir <run_dir>
```

### Paired hypothesis testing (Stage 3)
Paired tests over sample-level predictions for the quantum hyperparameter grid — McNemar's test, Cochran's Q, per-class
and difficulty-stratified analysis:
```bash
make paired-hypothesis-test
# uv run python -m src.evaluate.paired_hypothesis_testing
```

### McNemar test, quantum vs. classical (Stage 4)
Binary cancer/benign comparison of each hybrid model against its classical counterpart; writes
`mcnemar_test_results.json`:
```bash
make mcnemar-test
# uv run python -m src.evaluate.mcnemar_test_quantum_vs_classical
```

### Utilities
```bash
make zip-results             # Compress the results folders
make git-reset-pull          # Hard reset and pull latest changes
make clean                   # Remove virtual environment and caches
```

## Backbone benchmark (Stage 1 output)

| Model | Parameters (M) | Trainable Params (M) | Inference - Single (ms) | Inference - Batch (ms) | Inference - Per Sample (ms) | Training Epoch (sec) | Training Epoch (min) | Memory (MB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mobilenetv3_small_100 | 1.528 | 1.528 | 4.48 | 18.61 | 0.073 | 25.11 | 0.42 | 27.94 |
| regnetx_002 | 2.319 | 2.319 | 4.30 | 32.89 | 0.128 | 25.81 | 0.43 | 32.10 |
| regnety_002 | 2.798 | 2.798 | 6.44 | 35.52 | 0.139 | 26.51 | 0.44 | 33.94 |
| ghostnet_100 | 3.914 | 3.914 | 8.49 | 57.10 | 0.223 | 30.59 | 0.51 | 40.56 |
| mnasnet_100 | 3.115 | 3.115 | 4.33 | 60.20 | 0.235 | 30.61 | 0.51 | 28.69 |
| efficientnet_lite0 | 3.384 | 3.384 | 4.39 | 72.29 | 0.282 | 37.08 | 0.62 | 43.22 |
| mobilevit_xs | 1.937 | 1.937 | 7.29 | 146.74 | 0.573 | 55.98 | 0.93 | 40.63 |

## Dataset

**BreakHis** — 7,909 histopathological images from 82 patients, 8 classes (benign: adenosis, fibroadenoma, phyllodes
tumour, tubular adenoma; malignant: ductal, lobular, mucinous, papillary carcinoma). `make prepare-dataset` downloads
the dataset, splits it into train/val/test, resizes to 224×224, and balances the training set via augmentation.

If you use the dataset, please cite:

```
F. A. Spanhol, L. S. Oliveira, C. Petitjean and L. Heutte,
"A Dataset for Breast Cancer Histopathological Image Classification,"
IEEE Transactions on Biomedical Engineering, vol. 63, no. 7, pp. 1455-1462, July 2016.
```

## Citing this work

```
S. Vo, "Quantum-Enhanced CNN for Highly Accurate Breast Cancer Classification:
A Proof-of-Concept Study," Master's Thesis, University of Oulu,
Information Processing Science, 2026.
```

## Acknowledgements

Supervision by Assoc. Prof. Arif Ali Khan and co-supervision by Mr. Boshuai Ye (University of Oulu); industry
collaboration and guidance from Mr. Ghassan Sultan, Mr. Rupayan Mukherjee, and Mr. Johannes Storvik at TECHNIA;
and financial support from the
[TechFinland100 Foundation](https://techfinland100.fi/grants-for-masters-theses-on-ai-topics/) grant for Master's
theses on AI topics.

## License

Released for research and educational purposes. Dataset use is governed by the original BreakHis license.
