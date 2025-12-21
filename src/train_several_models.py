"""
Script to train multiple CNN backbone models sequentially.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from src import config
from src.train import main


def clear_gpu_memory():
    """Clear GPU cache and force garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("GPU memory cleared")


def load_progress():
    """Load training progress from JSON file."""
    progress_file = Path("results/training_progress.json")
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"combinations_done": 0}


def save_progress(combinations_done):
    """Save training progress to JSON file."""
    progress_file = Path("results/training_progress.json")
    progress_file.parent.mkdir(exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump({"combinations_done": combinations_done}, f, indent=2)
    print(f"Progress saved: {combinations_done} combinations completed")


# if __name__ == "__main__":
#     models_to_train = [
#         "mobilenetv3_small_100",
#         "mnasnet_100",
#         "regnetx_002",
#         "regnety_002",
#         "ghostnet_100",
#         "efficientnet_lite0",
#         "mobilevit_xs",
#     ]

#     for idx, model_name in enumerate(models_to_train, 1):
#         print("\n" + "=" * 80)
#         print(f"Training model {idx}/{len(models_to_train)}: {model_name}")
#         print("=" * 80)

#         # Set the model in config
#         config.DEFAULT_MODEL = model_name

#         # Train the model
#         main()

#         # Clear GPU memory after training
#         clear_gpu_memory()

#         print(f"\nCompleted {model_name} ({idx}/{len(models_to_train)})")


if __name__ == "__main__":
    """
    Train all combinations of quantum CNN configurations.
    Total: 18 combinations (2 qubits × 3 templates × 3 depths)
    """
    # Base configuration
    base_model = "cnn_quantum"
    backbone = "regnetx_002"
    encoding_method = "rotation"

    # Parameter combinations
    qubits_options = [8, 12]
    template_options = ["strong", "two_design", "basic"]
    depth_options = [3, 7]

    # Generate all combinations
    models_to_train = []
    for n_qubits in qubits_options:
        for template in template_options:
            for depth in depth_options:
                models_to_train.append({
                    "n_qubits": n_qubits,
                    "template": template,
                    "depth": depth
                })

    print(f"Total configurations to train: {len(models_to_train)}")
    print("=" * 80)

    # Load progress
    progress = load_progress()
    combinations_done = progress.get("combinations_done", 0)
    print(f"Previously completed: {combinations_done} combinations")
    print("=" * 80)

    # Train each configuration
    for idx, model_config in enumerate(models_to_train, 1):
        # Skip already completed combinations
        if idx <= combinations_done:
            print(f"\nSkipping configuration {idx}/{len(models_to_train)} (already completed)")
            continue
        print("\n" + "=" * 80)
        print(f"Training configuration {idx}/{len(models_to_train)}")
        print(f"Model: {base_model}")
        print(f"Backbone: {backbone}")
        print(f"Qubits: {model_config['n_qubits']}")
        print(f"Template: {model_config['template']}")
        print(f"Depth: {model_config['depth']}")
        print(f"Encoding: {encoding_method}")
        print("=" * 80)

        # Update config
        config.DEFAULT_MODEL = base_model
        config.QUANTUM_CNN_CONFIG_BACKBONE = backbone
        config.QUANTUM_CNN_CONFIG_NO_QUBITS = model_config['n_qubits']
        config.QUANTUM_CNN_CONFIG_DENSE_ENCODING_METHOD = encoding_method
        config.QUANTUM_CNN_CONFIG_DENSE_TEMPLATE = model_config['template']
        config.QUANTUM_CNN_CONFIG_DENSE_DEPTH = model_config['depth']
        config.QUANTUM_CNN_CONFIG_COMBINED_NAME = (
            f"cnn_quantum_"
            f"{backbone}_"
            f"dense-{encoding_method}_"
            f"{model_config['template']}_"
            f"depth-{model_config['depth']}_"
            f"qubits-{model_config['n_qubits']}"
        )

        try:
            # Train the model
            main()

            # Clear GPU memory after training
            clear_gpu_memory()

            print(f"\nCompleted configuration {idx}/{len(models_to_train)}")

            # Save progress after successful completion
            save_progress(idx)

        except Exception as e:
            print(f"\nError in configuration {idx}/{len(models_to_train)}: {str(e)}")
            print("Continuing to next configuration...")
            clear_gpu_memory()
            continue

    print("\n" + "=" * 80)
    print("All training configurations completed!")
    print("=" * 80)
