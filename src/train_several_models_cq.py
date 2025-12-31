from .train_several_models import clear_gpu_memory
from src import config
from src.train import main

if __name__ == "__main__":
    backbones_to_explore = [
        "shufflenetv2_x1_0",
        "lcnet_100",
        "squeezenet1_1"
    ]

    # Configuration for quantum models
    quantum_config = {
        "n_qubits": 8,
        "encoding_method": "rotation",
        "template": "two_design",
        "depth": 3
    }

    total_models = len(backbones_to_explore) * 2  # classical + hybrid for each
    current_idx = 0

    for backbone in backbones_to_explore:
        # Train classical version
        current_idx += 1
        print("\n" + "=" * 80)
        print(f"Training model {current_idx}/{total_models}: {backbone} (CLASSICAL)")
        print("=" * 80)

        config.DEFAULT_MODEL = backbone
        main()
        clear_gpu_memory()

        print(f"\nCompleted {backbone} classical ({current_idx}/{total_models})")

        # Train hybrid CNN-quantum version
        current_idx += 1
        print("\n" + "=" * 80)
        print(f"Training model {current_idx}/{total_models}: {backbone} (HYBRID CNN-QUANTUM)")
        print(f"Qubits: {quantum_config['n_qubits']}")
        print(f"Encoding: {quantum_config['encoding_method']}")
        print(f"Template: {quantum_config['template']}")
        print(f"Depth: {quantum_config['depth']}")
        print("=" * 80)

        config.DEFAULT_MODEL = "cnn_quantum"
        config.QUANTUM_CNN_CONFIG_BACKBONE = backbone
        config.QUANTUM_CNN_CONFIG_NO_QUBITS = quantum_config['n_qubits']
        config.QUANTUM_CNN_CONFIG_DENSE_ENCODING_METHOD = quantum_config['encoding_method']
        config.QUANTUM_CNN_CONFIG_DENSE_TEMPLATE = quantum_config['template']
        config.QUANTUM_CNN_CONFIG_DENSE_DEPTH = quantum_config['depth']
        config.QUANTUM_CNN_CONFIG_COMBINED_NAME = (
            f"cnn_quantum_"
            f"{backbone}_"
            f"dense-{quantum_config['encoding_method']}_"
            f"{quantum_config['template']}_"
            f"depth-{quantum_config['depth']}_"
            f"qubits-{quantum_config['n_qubits']}"
        )

        main()
        clear_gpu_memory()

        print(f"\nCompleted {backbone} hybrid ({current_idx}/{total_models})")

    print("\n" + "=" * 80)
    print("All models trained successfully!")
    print(f"Total: {total_models} models ({len(backbones_to_explore)} classical + {len(backbones_to_explore)} hybrid)")
    print("=" * 80)
