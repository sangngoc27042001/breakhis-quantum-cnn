"""
Evaluate quantum configurations from results_quantum_config folder.
Analyzes all quantum model runs to find the best configurations for validation and test accuracy.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class QuantumConfig:
    """Data class to store quantum configuration and results."""
    folder_name: str
    qubits: int
    template: str
    depth: int
    best_val_acc: float
    val_acc_epoch50: float
    test_acc_epoch50: float
    config_path: str


def load_quantum_results(results_dir: str = "results_quantum_config") -> List[QuantumConfig]:
    """
    Load all quantum configuration results from the results directory.

    Args:
        results_dir: Path to the quantum results directory

    Returns:
        List of QuantumConfig objects containing all configurations
    """
    results_path = Path(results_dir)
    configs = []

    for folder in results_path.iterdir():
        if not folder.is_dir() or folder.name.startswith('.'):
            continue

        config_file = folder / "config.json"
        metrics_file = folder / "epoch20_metrics.json"

        if not config_file.exists() or not metrics_file.exists():
            continue

        try:
            # Load config
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Load epoch 50 metrics
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Extract quantum parameters
            qubits = config.get('quantum_cnn_n_qubits')
            template = config.get('quantum_cnn_dense_template')
            depth = config.get('quantum_cnn_dense_depth')
            best_val_acc = config.get('best_val_acc')

            if None in [qubits, template, depth, best_val_acc]:
                continue

            quantum_config = QuantumConfig(
                folder_name=folder.name,
                qubits=qubits,
                template=template,
                depth=depth,
                best_val_acc=best_val_acc,
                val_acc_epoch50=metrics.get('val_acc', 0.0),
                test_acc_epoch50=metrics.get('test_acc', 0.0),
                config_path=str(folder)
            )

            configs.append(quantum_config)

        except Exception as e:
            print(f"Error loading {folder.name}: {e}")
            continue

    return configs


def find_best_configs(configs: List[QuantumConfig]) -> Tuple[QuantumConfig, QuantumConfig]:
    """
    Find the configurations with best validation and test accuracy.

    Args:
        configs: List of QuantumConfig objects

    Returns:
        Tuple of (best_val_acc_config, best_test_acc_config)
    """
    if not configs:
        raise ValueError("No configurations found")

    best_val = max(configs, key=lambda c: c.best_val_acc)
    best_test = max(configs, key=lambda c: c.test_acc_epoch50)

    return best_val, best_test


def load_normal_model_results(results_dir: str = "results/regnetx_002_20251221_095741") -> Dict:
    """
    Load normal (non-quantum) model results for comparison.

    Args:
        results_dir: Path to the normal model results directory

    Returns:
        Dictionary containing normal model metrics
    """
    results_path = Path(results_dir)
    config_file = results_path / "config.json"
    metrics_file = results_path / "epoch50_metrics.json"

    if not config_file.exists() or not metrics_file.exists():
        raise FileNotFoundError(f"Normal model results not found in {results_dir}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return {
        'model_name': config.get('model_name'),
        'best_val_acc': config.get('best_val_acc'),
        'val_acc_epoch50': metrics.get('val_acc'),
        'test_acc_epoch50': metrics.get('test_acc'),
        'total_params': config.get('total_params')
    }


def print_config_details(config: QuantumConfig, title: str):
    """Print detailed information about a quantum configuration."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"Qubits:              {config.qubits}")
    print(f"Template:            {config.template}")
    print(f"Depth:               {config.depth}")
    print(f"Best Val Acc:        {config.best_val_acc:.2f}%")
    print(f"Val Acc (epoch 50):  {config.val_acc_epoch50:.2f}%")
    print(f"Test Acc (epoch 50): {config.test_acc_epoch50:.2f}%")
    print(f"Folder:              {config.folder_name}")
    print(f"Path:                {config.config_path}")


def print_comparison(normal_model: Dict, quantum_config: QuantumConfig, metric: str):
    """Print comparison between normal and quantum model."""
    if metric == "val_acc":
        normal_val = normal_model['val_acc_epoch50']
        quantum_val = quantum_config.val_acc_epoch50
        diff = quantum_val - normal_val

        print(f"\nValidation Accuracy Comparison:")
        print(f"  Normal Model:   {normal_val:.2f}%")
        print(f"  Quantum Model:  {quantum_val:.2f}%")
        print(f"  Difference:     {diff:+.2f}%")

    elif metric == "test_acc":
        normal_test = normal_model['test_acc_epoch50']
        quantum_test = quantum_config.test_acc_epoch50
        diff = quantum_test - normal_test

        print(f"\nTest Accuracy Comparison:")
        print(f"  Normal Model:   {normal_test:.2f}%")
        print(f"  Quantum Model:  {quantum_test:.2f}%")
        print(f"  Difference:     {diff:+.2f}%")


def analyze_quantum_configs(
    results_dir: str = "results_quantum_config",
    normal_model_dir: str = "results/regnetx_002_20251221_095741"
):
    """
    Main analysis function to evaluate all quantum configurations.

    Args:
        results_dir: Path to quantum results directory
        normal_model_dir: Path to normal model results directory
    """
    print("=" * 80)
    print("QUANTUM CONFIGURATION ANALYSIS")
    print("=" * 80)

    # Load all quantum configurations
    print(f"\nLoading quantum configurations from: {results_dir}")
    configs = load_quantum_results(results_dir)
    print(f"Found {len(configs)} configurations")

    if not configs:
        print("No valid configurations found!")
        return

    # Find best configurations
    best_val_config, best_test_config = find_best_configs(configs)

    # Print best configurations
    print_config_details(best_val_config, "BEST CONFIGURATION BY VALIDATION ACCURACY")
    print_config_details(best_test_config, "BEST CONFIGURATION BY TEST ACCURACY")

    # Load normal model for comparison
    print(f"\n{'=' * 80}")
    print("COMPARISON WITH NORMAL MODEL")
    print(f"{'=' * 80}")

    try:
        normal_model = load_normal_model_results(normal_model_dir)
        print(f"\nNormal Model: {normal_model['model_name']}")
        print(f"Best Val Acc:        {normal_model['best_val_acc']:.2f}%")
        print(f"Val Acc (epoch 50):  {normal_model['val_acc_epoch50']:.2f}%")
        print(f"Test Acc (epoch 50): {normal_model['test_acc_epoch50']:.2f}%")
        print(f"Total Params:        {normal_model['total_params']:,}")

        # Compare best val_acc config with normal model
        print(f"\n{'=' * 80}")
        print("BEST VAL_ACC CONFIG vs NORMAL MODEL")
        print(f"{'=' * 80}")
        print_comparison(normal_model, best_val_config, "val_acc")
        print_comparison(normal_model, best_val_config, "test_acc")

        # Compare best test_acc config with normal model
        print(f"\n{'=' * 80}")
        print("BEST TEST_ACC CONFIG vs NORMAL MODEL")
        print(f"{'=' * 80}")
        print_comparison(normal_model, best_test_config, "val_acc")
        print_comparison(normal_model, best_test_config, "test_acc")

    except FileNotFoundError as e:
        print(f"\nWarning: Could not load normal model results: {e}")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")

    # Group by template
    templates = {}
    for config in configs:
        if config.template not in templates:
            templates[config.template] = []
        templates[config.template].append(config)

    print("\nAverage Test Accuracy by Template:")
    for template, template_configs in sorted(templates.items()):
        avg_test_acc = sum(c.test_acc_epoch50 for c in template_configs) / len(template_configs)
        print(f"  {template:15s}: {avg_test_acc:.2f}% (n={len(template_configs)})")

    # Group by qubits
    qubits_groups = {}
    for config in configs:
        if config.qubits not in qubits_groups:
            qubits_groups[config.qubits] = []
        qubits_groups[config.qubits].append(config)

    print("\nAverage Test Accuracy by Number of Qubits:")
    for qubits, qubit_configs in sorted(qubits_groups.items()):
        avg_test_acc = sum(c.test_acc_epoch50 for c in qubit_configs) / len(qubit_configs)
        print(f"  {qubits} qubits: {avg_test_acc:.2f}% (n={len(qubit_configs)})")

    # Group by depth
    depth_groups = {}
    for config in configs:
        if config.depth not in depth_groups:
            depth_groups[config.depth] = []
        depth_groups[config.depth].append(config)

    print("\nAverage Test Accuracy by Depth:")
    for depth, depth_configs in sorted(depth_groups.items()):
        avg_test_acc = sum(c.test_acc_epoch50 for c in depth_configs) / len(depth_configs)
        print(f"  depth={depth:2d}: {avg_test_acc:.2f}% (n={len(depth_configs)})")

    print("\n" + "=" * 80)


def analyze_early_training_speed(
    results_dir: str = "results_quantum_config",
    epoch: int = 20
):
    """
    Analyze which quantum configurations learn fastest in early training.

    Args:
        results_dir: Path to quantum results directory
        epoch: Which epoch to analyze (default: 20)
    """
    print("=" * 80)
    print(f"EARLY TRAINING SPEED ANALYSIS (Epoch {epoch})")
    print("=" * 80)

    results_path = Path(results_dir)
    configs = []

    for folder in results_path.iterdir():
        if not folder.is_dir() or folder.name.startswith('.'):
            continue

        config_file = folder / "config.json"
        metrics_file = folder / f"epoch{epoch}_metrics.json"

        if not config_file.exists() or not metrics_file.exists():
            continue

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            qubits = config.get('quantum_cnn_n_qubits')
            template = config.get('quantum_cnn_dense_template')
            depth = config.get('quantum_cnn_dense_depth')

            if None in [qubits, template, depth]:
                continue

            configs.append({
                'folder': folder.name,
                'qubits': qubits,
                'template': template,
                'depth': depth,
                'train_acc': metrics.get('train_acc', 0.0),
                'val_acc': metrics.get('val_acc', 0.0),
                'test_acc': metrics.get('test_acc', 0.0),
                'train_loss': metrics.get('train_loss', 0.0)
            })

        except Exception as e:
            print(f"Error loading {folder.name}: {e}")
            continue

    if not configs:
        print("No configurations found!")
        return

    print(f"\nFound {len(configs)} configurations")

    # Sort by different metrics
    print("\n" + "=" * 80)
    print(f"TOP 5 CONFIGS BY VALIDATION ACCURACY AT EPOCH {epoch}")
    print("=" * 80)
    sorted_by_val = sorted(configs, key=lambda x: x['val_acc'], reverse=True)[:5]
    for i, cfg in enumerate(sorted_by_val, 1):
        print(f"\n{i}. Val Acc: {cfg['val_acc']:.2f}% | Test Acc: {cfg['test_acc']:.2f}%")
        print(f"   Qubits: {cfg['qubits']}, Template: {cfg['template']}, Depth: {cfg['depth']}")
        print(f"   Folder: {cfg['folder']}")

    print("\n" + "=" * 80)
    print(f"TOP 5 CONFIGS BY TEST ACCURACY AT EPOCH {epoch}")
    print("=" * 80)
    sorted_by_test = sorted(configs, key=lambda x: x['test_acc'], reverse=True)[:5]
    for i, cfg in enumerate(sorted_by_test, 1):
        print(f"\n{i}. Test Acc: {cfg['test_acc']:.2f}% | Val Acc: {cfg['val_acc']:.2f}%")
        print(f"   Qubits: {cfg['qubits']}, Template: {cfg['template']}, Depth: {cfg['depth']}")
        print(f"   Folder: {cfg['folder']}")

    print("\n" + "=" * 80)
    print(f"TOP 5 CONFIGS BY TRAINING ACCURACY AT EPOCH {epoch}")
    print("=" * 80)
    sorted_by_train = sorted(configs, key=lambda x: x['train_acc'], reverse=True)[:5]
    for i, cfg in enumerate(sorted_by_train, 1):
        print(f"\n{i}. Train Acc: {cfg['train_acc']:.2f}% | Val Acc: {cfg['val_acc']:.2f}% | Test Acc: {cfg['test_acc']:.2f}%")
        print(f"   Qubits: {cfg['qubits']}, Template: {cfg['template']}, Depth: {cfg['depth']}")
        print(f"   Folder: {cfg['folder']}")

    # Aggregate statistics
    print("\n" + "=" * 80)
    print(f"AVERAGE METRICS AT EPOCH {epoch} BY CONFIGURATION")
    print("=" * 80)

    # By template
    print("\nBy Template:")
    templates = {}
    for cfg in configs:
        if cfg['template'] not in templates:
            templates[cfg['template']] = []
        templates[cfg['template']].append(cfg)

    for template, template_configs in sorted(templates.items()):
        avg_train = sum(c['train_acc'] for c in template_configs) / len(template_configs)
        avg_val = sum(c['val_acc'] for c in template_configs) / len(template_configs)
        avg_test = sum(c['test_acc'] for c in template_configs) / len(template_configs)
        print(f"  {template:15s}: Train={avg_train:.2f}%, Val={avg_val:.2f}%, Test={avg_test:.2f}% (n={len(template_configs)})")

    # By qubits
    print("\nBy Number of Qubits:")
    qubits_groups = {}
    for cfg in configs:
        if cfg['qubits'] not in qubits_groups:
            qubits_groups[cfg['qubits']] = []
        qubits_groups[cfg['qubits']].append(cfg)

    for qubits, qubit_configs in sorted(qubits_groups.items()):
        avg_train = sum(c['train_acc'] for c in qubit_configs) / len(qubit_configs)
        avg_val = sum(c['val_acc'] for c in qubit_configs) / len(qubit_configs)
        avg_test = sum(c['test_acc'] for c in qubit_configs) / len(qubit_configs)
        print(f"  {qubits} qubits: Train={avg_train:.2f}%, Val={avg_val:.2f}%, Test={avg_test:.2f}% (n={len(qubit_configs)})")

    # By depth
    print("\nBy Depth:")
    depth_groups = {}
    for cfg in configs:
        if cfg['depth'] not in depth_groups:
            depth_groups[cfg['depth']] = []
        depth_groups[cfg['depth']].append(cfg)

    for depth, depth_configs in sorted(depth_groups.items()):
        avg_train = sum(c['train_acc'] for c in depth_configs) / len(depth_configs)
        avg_val = sum(c['val_acc'] for c in depth_configs) / len(depth_configs)
        avg_test = sum(c['test_acc'] for c in depth_configs) / len(depth_configs)
        print(f"  depth={depth:2d}: Train={avg_train:.2f}%, Val={avg_val:.2f}%, Test={avg_test:.2f}% (n={len(depth_configs)})")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--early':
        epoch = 20
        if len(sys.argv) > 2:
            try:
                epoch = int(sys.argv[2])
            except ValueError:
                print(f"Invalid epoch number: {sys.argv[2]}, using default 20")
        analyze_early_training_speed(epoch=epoch)
    else:
        analyze_quantum_configs()


if __name__ == "__main__":
    main()
