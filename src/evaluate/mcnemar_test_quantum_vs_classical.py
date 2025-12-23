#!/usr/bin/env python3
"""
McNemar Test: Quantum-Enhanced CNN vs Classical CNN
Binary classification: Cancer vs Benign
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2
from sklearn.metrics import accuracy_score
import json


# Class mapping
CLASS_NAMES = {
    0: 'Adenosis',
    1: 'Ductal_Carcinoma',
    2: 'Fibroadenoma',
    3: 'Lobular_Carcinoma',
    4: 'Mucinous_Carcinoma',
    5: 'Papillary_Carcinoma',
    6: 'Phyllodes_Tumor',
    7: 'Tubular_Adenoma'
}

BENIGN_CLASSES = [0, 2, 6, 7]  # Adenosis, Fibroadenoma, Phyllodes_Tumor, Tubular_Adenoma
CANCER_CLASSES = [1, 3, 4, 5]  # Ductal, Lobular, Mucinous, Papillary Carcinoma


def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Perform McNemar's test for paired nominal data.

    Returns: dict with chi2 statistic, p-value, and contingency table
    """
    both_correct = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    both_wrong = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
    model1_correct_model2_wrong = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    model1_wrong_model2_correct = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))

    b = model1_correct_model2_wrong
    c = model1_wrong_model2_correct

    # McNemar's test statistic with continuity correction
    if b + c == 0:
        chi2_stat = 0
        p_value = 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return {
        'chi2': chi2_stat,
        'p_value': p_value,
        'both_correct': int(both_correct),
        'both_wrong': int(both_wrong),
        'classical_correct_quantum_wrong': int(b),
        'classical_wrong_quantum_correct': int(c),
        'n_samples': len(y_true)
    }


def load_predictions(results_dir, model_pattern):
    """Load test set predictions from results directory."""
    result_dirs = [d for d in Path(results_dir).iterdir()
                   if d.is_dir() and model_pattern in d.name]

    if not result_dirs:
        raise ValueError(f"No directory found for pattern '{model_pattern}' in {results_dir}")

    result_dir = result_dirs[0]
    pred_file = result_dir / "epoch20_detail_predictions.csv"

    df = pd.read_csv(pred_file)
    df_test = df[df['split'] == 'test'].copy()

    return df_test, result_dir.name


def calculate_binary_metrics(df_test):
    """Calculate accuracy and cancer recall (binary: cancer vs benign)."""
    y_true = df_test['groundtruth'].values
    y_pred = df_test['predict'].values

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100

    # Convert to binary: 1 = cancer, 0 = benign
    y_true_binary = np.isin(y_true, CANCER_CLASSES).astype(int)
    y_pred_binary = np.isin(y_pred, CANCER_CLASSES).astype(int)

    # Cancer recall: Among actual cancer samples, how many predicted as cancer?
    cancer_mask = y_true_binary == 1
    n_cancer_samples = np.sum(cancer_mask)

    if n_cancer_samples > 0:
        cancer_recall = np.sum(y_pred_binary[cancer_mask] == 1) / n_cancer_samples * 100
    else:
        cancer_recall = 0.0

    return {
        'accuracy': accuracy,
        'cancer_recall': cancer_recall,
        'n_cancer_samples': int(n_cancer_samples)
    }


def analyze_model_pair(classical_pattern, quantum_pattern, results_dir_classical, results_dir_quantum):
    """Analyze one pair of classical vs quantum models."""

    # Load data
    df_classical, classical_name = load_predictions(results_dir_classical, classical_pattern)
    df_quantum, quantum_name = load_predictions(results_dir_quantum, quantum_pattern)

    # Merge on same samples
    df_merged = df_classical.merge(
        df_quantum,
        on=['file_path', 'groundtruth'],
        suffixes=('_classical', '_quantum')
    )

    y_true = df_merged['groundtruth'].values
    y_pred_classical = df_merged['predict_classical'].values
    y_pred_quantum = df_merged['predict_quantum'].values

    # 1. Calculate metrics for each model
    classical_metrics = calculate_binary_metrics(df_classical)
    quantum_metrics = calculate_binary_metrics(df_quantum)

    # 2. McNemar test for overall accuracy (all samples)
    mcnemar_accuracy = mcnemar_test(y_true, y_pred_classical, y_pred_quantum)

    # 3. McNemar test for cancer recall (only cancer samples, binary prediction)
    # Convert to binary
    y_true_binary = np.isin(y_true, CANCER_CLASSES).astype(int)
    y_pred_classical_binary = np.isin(y_pred_classical, CANCER_CLASSES).astype(int)
    y_pred_quantum_binary = np.isin(y_pred_quantum, CANCER_CLASSES).astype(int)

    # Filter only cancer samples
    cancer_mask = y_true_binary == 1
    y_true_cancer = y_true_binary[cancer_mask]
    y_pred_classical_cancer = y_pred_classical_binary[cancer_mask]
    y_pred_quantum_cancer = y_pred_quantum_binary[cancer_mask]

    mcnemar_cancer_recall = mcnemar_test(y_true_cancer, y_pred_classical_cancer, y_pred_quantum_cancer)

    return {
        'model': classical_pattern,
        'classical_name': classical_name,
        'quantum_name': quantum_name,
        'n_test_samples': len(df_merged),
        'classical_metrics': classical_metrics,
        'quantum_metrics': quantum_metrics,
        'mcnemar_accuracy': mcnemar_accuracy,
        'mcnemar_cancer_recall': mcnemar_cancer_recall
    }


def print_results(results):
    """Print results in a clean, organized format."""

    print("\n" + "=" * 100)
    print("QUANTUM-ENHANCED CNN vs CLASSICAL CNN - McNemar Test Analysis")
    print("Binary Classification: CANCER vs BENIGN")
    print("=" * 100)

    for i, result in enumerate(results, 1):
        print(f"\n{'=' * 100}")
        print(f"PAIR {i}/7: {result['model'].upper()}")
        print(f"{'=' * 100}")

        print(f"\nClassical: {result['classical_name']}")
        print(f"Quantum:   {result['quantum_name']}")
        print(f"Test samples: {result['n_test_samples']}")

        # Metrics comparison
        print(f"\n--- METRICS COMPARISON ---")

        classical = result['classical_metrics']
        quantum = result['quantum_metrics']

        print(f"\nOverall Accuracy (8-class):")
        print(f"  Classical: {classical['accuracy']:6.2f}%")
        print(f"  Quantum:   {quantum['accuracy']:6.2f}%")
        print(f"  Δ:         {quantum['accuracy'] - classical['accuracy']:+6.2f} pp")

        print(f"\nCancer Recall (binary - among {classical['n_cancer_samples']} cancer samples):")
        print(f"  Classical: {classical['cancer_recall']:6.2f}%")
        print(f"  Quantum:   {quantum['cancer_recall']:6.2f}%")
        print(f"  Δ:         {quantum['cancer_recall'] - classical['cancer_recall']:+6.2f} pp")

        # McNemar test results
        print(f"\n--- McNEMAR TEST RESULTS ---")

        # Accuracy test
        mc_acc = result['mcnemar_accuracy']
        sig_acc = "✓✓ HIGHLY SIG" if mc_acc['p_value'] < 0.01 else "✓ SIGNIFICANT" if mc_acc['p_value'] < 0.05 else "NOT SIGNIFICANT"

        print(f"\nOverall Accuracy (all {mc_acc['n_samples']} test samples, 8-class):")
        print(f"  Chi-square: {mc_acc['chi2']:8.4f}")
        print(f"  P-value:    {mc_acc['p_value']:8.6f}  [{sig_acc}]")
        print(f"  Contingency:")
        print(f"    Both correct:           {mc_acc['both_correct']:5d}")
        print(f"    Both wrong:             {mc_acc['both_wrong']:5d}")
        print(f"    Classical ✓, Quantum ✗:  {mc_acc['classical_correct_quantum_wrong']:5d}")
        print(f"    Classical ✗, Quantum ✓:  {mc_acc['classical_wrong_quantum_correct']:5d}")

        # Cancer recall test
        mc_cancer = result['mcnemar_cancer_recall']
        sig_cancer = "✓✓ HIGHLY SIG" if mc_cancer['p_value'] < 0.01 else "✓ SIGNIFICANT" if mc_cancer['p_value'] < 0.05 else "NOT SIGNIFICANT"

        print(f"\nCancer Recall (only {mc_cancer['n_samples']} cancer samples, binary):")
        print(f"  Chi-square: {mc_cancer['chi2']:8.4f}")
        print(f"  P-value:    {mc_cancer['p_value']:8.6f}  [{sig_cancer}]")
        print(f"  Contingency:")
        print(f"    Both correct (predicted as cancer):     {mc_cancer['both_correct']:5d}")
        print(f"    Both wrong (predicted as benign):       {mc_cancer['both_wrong']:5d}")
        print(f"    Classical ✓, Quantum ✗:                  {mc_cancer['classical_correct_quantum_wrong']:5d}")
        print(f"    Classical ✗, Quantum ✓:                  {mc_cancer['classical_wrong_quantum_correct']:5d}")

    # Summary
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")

    # Count significant improvements
    sig_accuracy = sum(1 for r in results if r['mcnemar_accuracy']['p_value'] < 0.05)
    highly_sig_accuracy = sum(1 for r in results if r['mcnemar_accuracy']['p_value'] < 0.01)

    sig_cancer = sum(1 for r in results if r['mcnemar_cancer_recall']['p_value'] < 0.05)
    highly_sig_cancer = sum(1 for r in results if r['mcnemar_cancer_recall']['p_value'] < 0.01)

    print(f"\nOverall Accuracy Improvements (8-class):")
    print(f"  Significant (p < 0.05):        {sig_accuracy}/7")
    print(f"  Highly significant (p < 0.01): {highly_sig_accuracy}/7")

    print(f"\nCancer Recall Improvements (binary):")
    print(f"  Significant (p < 0.05):        {sig_cancer}/7")
    print(f"  Highly significant (p < 0.01): {highly_sig_cancer}/7")

    # Best improvements
    print(f"\nLargest Accuracy Improvements:")
    acc_improvements = [(r['model'],
                        r['quantum_metrics']['accuracy'] - r['classical_metrics']['accuracy'],
                        r['mcnemar_accuracy']['p_value'])
                       for r in results]
    acc_improvements.sort(key=lambda x: x[1], reverse=True)

    for model, improvement, p_value in acc_improvements[:3]:
        sig = "✓✓" if p_value < 0.01 else "✓" if p_value < 0.05 else ""
        print(f"  {model:25s}: {improvement:+6.2f} pp  (p={p_value:.6f}) {sig}")

    print(f"\nLargest Cancer Recall Improvements:")
    cancer_improvements = [(r['model'],
                           r['quantum_metrics']['cancer_recall'] - r['classical_metrics']['cancer_recall'],
                           r['mcnemar_cancer_recall']['p_value'])
                          for r in results]
    cancer_improvements.sort(key=lambda x: x[1], reverse=True)

    for model, improvement, p_value in cancer_improvements[:3]:
        sig = "✓✓" if p_value < 0.01 else "✓" if p_value < 0.05 else ""
        print(f"  {model:25s}: {improvement:+6.2f} pp  (p={p_value:.6f}) {sig}")


def main():
    results_dir_classical = "results"
    results_dir_quantum = "results_quantum_cnn"

    model_pairs = [
        ('efficientnet_lite0', 'cnn_quantum_efficientnet_lite0'),
        ('ghostnet_100', 'cnn_quantum_ghostnet_100'),
        ('mnasnet_100', 'cnn_quantum_mnasnet_100'),
        ('mobilenetv3_small_100', 'cnn_quantum_mobilenetv3_small_100'),
        ('mobilevit_xs', 'cnn_quantum_mobilevit_xs'),
        ('regnetx_002', 'cnn_quantum_regnetx_002'),
        ('regnety_002', 'cnn_quantum_regnety_002'),
    ]

    results = []

    for classical_pattern, quantum_pattern in model_pairs:
        try:
            result = analyze_model_pair(
                classical_pattern, quantum_pattern,
                results_dir_classical, results_dir_quantum
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR analyzing {classical_pattern}: {e}")
            continue

    # Print results
    print_results(results)

    # Save to JSON
    output_file = "mcnemar_test_results.json"

    # Convert to JSON-serializable format
    json_results = []
    for r in results:
        json_result = {
            'model': r['model'],
            'classical_name': r['classical_name'],
            'quantum_name': r['quantum_name'],
            'n_test_samples': r['n_test_samples'],
            'classical_accuracy': r['classical_metrics']['accuracy'],
            'quantum_accuracy': r['quantum_metrics']['accuracy'],
            'accuracy_improvement': r['quantum_metrics']['accuracy'] - r['classical_metrics']['accuracy'],
            'classical_cancer_recall': r['classical_metrics']['cancer_recall'],
            'quantum_cancer_recall': r['quantum_metrics']['cancer_recall'],
            'cancer_recall_improvement': r['quantum_metrics']['cancer_recall'] - r['classical_metrics']['cancer_recall'],
            'n_cancer_samples': r['classical_metrics']['n_cancer_samples'],
            'mcnemar_accuracy': r['mcnemar_accuracy'],
            'mcnemar_cancer_recall': r['mcnemar_cancer_recall']
        }
        json_results.append(json_result)

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n{'=' * 100}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
