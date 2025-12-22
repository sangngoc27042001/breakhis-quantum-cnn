"""
Compare model performance across different metrics and datasets.
Provides detailed accuracy analysis for train/val/test sets.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class ModelPerformance:
    """Performance metrics for a single model."""
    model_name: str
    folder_name: str

    # Training metrics
    final_train_loss: float
    final_train_acc: float

    # Validation metrics
    final_val_loss: float
    final_val_acc: float
    best_val_acc: float
    best_val_acc_epoch: int

    # Test metrics (if available)
    test_acc: float = 0.0
    test_loss: float = 0.0

    # Model info
    total_params: int = 0
    trainable_params: int = 0
    epochs: int = 20

    # Learning curve analysis
    train_acc_std: float = 0.0  # Standard deviation of training accuracy
    val_acc_std: float = 0.0    # Standard deviation of validation accuracy
    overfitting_gap: float = 0.0  # final_train_acc - final_val_acc

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


def load_model_performance(result_folder: Path) -> ModelPerformance:
    """
    Load performance metrics from a result folder.

    Args:
        result_folder: Path to the result folder

    Returns:
        ModelPerformance object
    """
    # Load training history
    history_file = result_folder / "training_history.json"
    config_file = result_folder / "config.json"

    if not history_file.exists():
        raise FileNotFoundError(f"training_history.json not found in {result_folder}")

    with open(history_file, 'r') as f:
        history = json.load(f)

    # Load config
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    # Extract model name
    folder_name = result_folder.name
    model_name = folder_name.split('_202')[0]

    # Extract metrics
    train_acc = history.get('train_acc', [])
    val_acc = history.get('val_acc', [])
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])

    # Best validation accuracy
    best_val_acc = max(val_acc) if val_acc else 0
    best_val_acc_epoch = val_acc.index(best_val_acc) + 1 if val_acc else 0

    # Final epoch metrics
    final_train_loss = train_loss[-1] if train_loss else 0
    final_train_acc = train_acc[-1] if train_acc else 0
    final_val_loss = val_loss[-1] if val_loss else 0
    final_val_acc = val_acc[-1] if val_acc else 0

    # Calculate standard deviations
    import statistics
    train_acc_std = statistics.stdev(train_acc) if len(train_acc) > 1 else 0
    val_acc_std = statistics.stdev(val_acc) if len(val_acc) > 1 else 0

    # Overfitting gap
    overfitting_gap = final_train_acc - final_val_acc

    # Try to load test metrics if available
    test_acc = 0.0
    test_loss = 0.0

    # Check for epoch metrics file
    epoch_metrics_file = result_folder / f"epoch{len(train_acc)}_metrics.json"
    if epoch_metrics_file.exists():
        with open(epoch_metrics_file, 'r') as f:
            metrics = json.load(f)
            test_acc = metrics.get('test_acc', 0.0)
            test_loss = metrics.get('test_loss', 0.0)

    return ModelPerformance(
        model_name=model_name,
        folder_name=folder_name,
        final_train_loss=final_train_loss,
        final_train_acc=final_train_acc,
        final_val_loss=final_val_loss,
        final_val_acc=final_val_acc,
        best_val_acc=best_val_acc,
        best_val_acc_epoch=best_val_acc_epoch,
        test_acc=test_acc,
        test_loss=test_loss,
        total_params=config.get('total_params', 0),
        trainable_params=config.get('trainable_params', 0),
        epochs=len(train_acc),
        train_acc_std=train_acc_std,
        val_acc_std=val_acc_std,
        overfitting_gap=overfitting_gap
    )


def compare_all_models(results_dir: str = "results") -> List[ModelPerformance]:
    """
    Compare all models in the results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        List of ModelPerformance objects
    """
    results_path = Path(results_dir)
    all_performance = []

    for folder in results_path.iterdir():
        if not folder.is_dir() or folder.name.startswith('.'):
            continue

        try:
            perf = load_model_performance(folder)
            all_performance.append(perf)
        except Exception as e:
            print(f"Warning: Could not analyze {folder.name}: {e}")

    return all_performance


def print_performance_comparison(perf_list: List[ModelPerformance]):
    """Print detailed performance comparison."""
    if not perf_list:
        print("No models found!")
        return

    print("\n" + "=" * 120)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 120)

    # Table 1: Overall Performance
    print("\nOVERALL PERFORMANCE:")
    print("-" * 120)
    print(f"{'Model':<30} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Best Val':<12} {'Epoch':<8}")
    print("-" * 120)

    sorted_by_val = sorted(perf_list, key=lambda x: x.best_val_acc, reverse=True)
    for perf in sorted_by_val:
        test_str = f"{perf.test_acc:.2f}%" if perf.test_acc > 0 else "N/A"
        print(f"{perf.model_name:<30} {perf.final_train_acc:<12.2f} {perf.final_val_acc:<12.2f} "
              f"{test_str:<12} {perf.best_val_acc:<12.2f} {perf.best_val_acc_epoch:<8}")

    # Table 2: Train/Val/Test Breakdown
    print("\n" + "=" * 120)
    print("ACCURACY BREAKDOWN BY DATASET:")
    print("-" * 120)
    print(f"{'Model':<30} {'Train Set':<25} {'Validation Set':<25} {'Test Set':<25}")
    print(f"{'':30} {'Acc':<12} {'Loss':<12} {'Acc':<12} {'Loss':<12} {'Acc':<12} {'Loss':<12}")
    print("-" * 120)

    for perf in sorted_by_val:
        test_acc_str = f"{perf.test_acc:.2f}" if perf.test_acc > 0 else "N/A"
        test_loss_str = f"{perf.test_loss:.4f}" if perf.test_loss > 0 else "N/A"
        print(f"{perf.model_name:<30} {perf.final_train_acc:<12.2f} {perf.final_train_loss:<12.4f} "
              f"{perf.final_val_acc:<12.2f} {perf.final_val_loss:<12.4f} "
              f"{test_acc_str:<12} {test_loss_str:<12}")

    # Table 3: Generalization Analysis
    print("\n" + "=" * 120)
    print("GENERALIZATION ANALYSIS:")
    print("-" * 120)
    print(f"{'Model':<30} {'Overfitting Gap':<18} {'Train Std':<15} {'Val Std':<15} {'Params':<15}")
    print("-" * 120)

    sorted_by_gap = sorted(perf_list, key=lambda x: x.overfitting_gap)
    for perf in sorted_by_gap:
        print(f"{perf.model_name:<30} {perf.overfitting_gap:<18.2f} "
              f"{perf.train_acc_std:<15.2f} {perf.val_acc_std:<15.2f} {perf.total_params:<15,}")

    # Statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS:")
    print("-" * 120)

    avg_train = sum(p.final_train_acc for p in perf_list) / len(perf_list)
    avg_val = sum(p.final_val_acc for p in perf_list) / len(perf_list)
    avg_best_val = sum(p.best_val_acc for p in perf_list) / len(perf_list)

    test_models = [p for p in perf_list if p.test_acc > 0]
    avg_test = sum(p.test_acc for p in test_models) / len(test_models) if test_models else 0

    print(f"Average Training Accuracy:    {avg_train:.2f}%")
    print(f"Average Validation Accuracy:  {avg_val:.2f}%")
    print(f"Average Best Val Accuracy:    {avg_best_val:.2f}%")
    if avg_test > 0:
        print(f"Average Test Accuracy:        {avg_test:.2f}%")

    best_overall = max(perf_list, key=lambda x: x.best_val_acc)
    print(f"\nBest Overall Model:           {best_overall.model_name}")
    print(f"  Best Val Accuracy:          {best_overall.best_val_acc:.2f}%")
    print(f"  Test Accuracy:              {best_overall.test_acc:.2f}%" if best_overall.test_acc > 0 else "  Test Accuracy:              N/A")

    best_generalization = min(perf_list, key=lambda x: x.overfitting_gap)
    print(f"\nBest Generalization:          {best_generalization.model_name}")
    print(f"  Overfitting Gap:            {best_generalization.overfitting_gap:.2f}%")

    print("=" * 120)


def create_performance_table(perf_list: List[ModelPerformance]) -> pd.DataFrame:
    """Create a pandas DataFrame for detailed analysis."""
    data = [p.to_dict() for p in perf_list]
    df = pd.DataFrame(data)

    # Sort by best validation accuracy
    df = df.sort_values('best_val_acc', ascending=False)

    return df


def save_comparison_results(perf_list: List[ModelPerformance], output_file: str = "performance_comparison.csv"):
    """Save comparison results to CSV."""
    df = create_performance_table(perf_list)
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    return df


def analyze_per_set_accuracy(perf_list: List[ModelPerformance]):
    """Analyze accuracy for each dataset (train/val/test)."""
    print("\n" + "=" * 100)
    print("ACCURACY ANALYSIS PER DATASET")
    print("=" * 100)

    # Training set
    print("\nTRAINING SET:")
    print("-" * 100)
    sorted_by_train = sorted(perf_list, key=lambda x: x.final_train_acc, reverse=True)
    print(f"{'Rank':<6} {'Model':<35} {'Accuracy':<15} {'Loss':<15}")
    print("-" * 100)
    for i, perf in enumerate(sorted_by_train, 1):
        print(f"{i:<6} {perf.model_name:<35} {perf.final_train_acc:<15.2f} {perf.final_train_loss:<15.4f}")

    # Validation set
    print("\nVALIDATION SET:")
    print("-" * 100)
    sorted_by_val = sorted(perf_list, key=lambda x: x.final_val_acc, reverse=True)
    print(f"{'Rank':<6} {'Model':<35} {'Accuracy':<15} {'Loss':<15}")
    print("-" * 100)
    for i, perf in enumerate(sorted_by_val, 1):
        print(f"{i:<6} {perf.model_name:<35} {perf.final_val_acc:<15.2f} {perf.final_val_loss:<15.4f}")

    # Test set
    test_models = [p for p in perf_list if p.test_acc > 0]
    if test_models:
        print("\nTEST SET:")
        print("-" * 100)
        sorted_by_test = sorted(test_models, key=lambda x: x.test_acc, reverse=True)
        print(f"{'Rank':<6} {'Model':<35} {'Accuracy':<15} {'Loss':<15}")
        print("-" * 100)
        for i, perf in enumerate(sorted_by_test, 1):
            print(f"{i:<6} {perf.model_name:<35} {perf.test_acc:<15.2f} {perf.test_loss:<15.4f}")
    else:
        print("\nTEST SET: No test results available")

    print("=" * 100)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare model performance across datasets')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Path to results directory')
    parser.add_argument('--output', type=str, default='performance_comparison.csv',
                        help='Output CSV file path')
    parser.add_argument('--per-set', action='store_true',
                        help='Show per-dataset accuracy analysis')

    args = parser.parse_args()

    print(f"Analyzing models from: {args.results_dir}")

    # Load all model performance
    perf_list = compare_all_models(args.results_dir)

    if not perf_list:
        print("No models found!")
        return

    print(f"Found {len(perf_list)} trained models")

    # Print comparison
    print_performance_comparison(perf_list)

    # Per-set analysis if requested
    if args.per_set:
        analyze_per_set_accuracy(perf_list)

    # Save results
    save_comparison_results(perf_list, args.output)


if __name__ == "__main__":
    main()
