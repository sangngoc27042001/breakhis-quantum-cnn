"""
Script to train multiple CNN backbone models sequentially.
"""
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


if __name__ == "__main__":
    models_to_train = [
        "mobilenetv3_small_100",
        "mnasnet_100",
        "regnetx_002",
        "regnety_002",
        "ghostnet_100",
        "efficientnet_lite0",
        "mobilevit_xs",
    ]

    for idx, model_name in enumerate(models_to_train, 1):
        print("\n" + "=" * 80)
        print(f"Training model {idx}/{len(models_to_train)}: {model_name}")
        print("=" * 80)

        # Set the model in config
        config.DEFAULT_MODEL = model_name

        # Train the model
        main()

        # Clear GPU memory after training
        clear_gpu_memory()

        print(f"\nCompleted {model_name} ({idx}/{len(models_to_train)})")
