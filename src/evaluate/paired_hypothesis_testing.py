"""
Advanced Paired Hypothesis Testing for Quantum CNN Hyperparameters

Uses sample-level predictions to perform more powerful statistical tests:
1. McNemar's Test - Paired comparison of classification accuracy
2. Cochran's Q Test - Multiple paired comparisons
3. Per-class analysis - Class-specific effects
4. Difficulty-stratified analysis - Easy vs hard samples
5. Agreement analysis - Where do models disagree?

This approach is MUCH more powerful than comparing mean accuracies because
it accounts for the fact that all models are tested on the SAME samples.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import chi2
from statsmodels.stats.contingency_tables import mcnemar
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def load_all_predictions(results_dir: str = "results_quantum_config") -> pd.DataFrame:
    """
    Load detailed predictions from all quantum configurations.

    Returns:
        DataFrame with columns: file_path, groundtruth, config_name, template, depth, qubits, prediction, correct
    """
    results_path = Path(results_dir)
    all_data = []

    for folder in results_path.iterdir():
        if not folder.is_dir() or folder.name.startswith('.'):
            continue

        config_file = folder / "config.json"
        predictions_file = folder / "epoch20_detail_predictions.csv"

        if not config_file.exists() or not predictions_file.exists():
            continue

        try:
            # Load config
            with open(config_file, 'r') as f:
                config = json.load(f)

            qubits = config.get('quantum_cnn_n_qubits')
            template = config.get('quantum_cnn_dense_template')
            depth = config.get('quantum_cnn_dense_depth')

            if None in [qubits, template, depth]:
                continue

            # Load predictions
            pred_df = pd.read_csv(predictions_file)

            # Filter for test split only
            pred_df = pred_df[pred_df['split'] == 'test'].copy()

            # Add configuration info
            pred_df['config_name'] = folder.name
            pred_df['template'] = template
            pred_df['depth'] = depth
            pred_df['qubits'] = qubits
            pred_df['correct'] = (pred_df['groundtruth'] == pred_df['predict']).astype(int)

            all_data.append(pred_df)

        except Exception as e:
            print(f"Error loading {folder.name}: {e}")
            continue

    if not all_data:
        raise ValueError("No prediction data found!")

    # Combine all predictions
    combined_df = pd.concat(all_data, ignore_index=True)

    return combined_df


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def step1_data_overview(df: pd.DataFrame):
    """Step 1: Overview of the sample-level data."""
    print_section_header("STEP 1: SAMPLE-LEVEL DATA OVERVIEW")

    print(f"\nTotal prediction records: {len(df)}")
    print(f"Unique samples: {df['file_path'].nunique()}")
    print(f"Unique configurations: {df['config_name'].nunique()}")

    # Check if all configs tested on same samples
    samples_per_config = df.groupby('config_name')['file_path'].apply(set)
    reference_samples = samples_per_config.iloc[0]

    all_same = all(samples == reference_samples for samples in samples_per_config)

    if all_same:
        print(f"✓ All configurations tested on SAME {len(reference_samples)} samples (paired data!)")
    else:
        print("⚠ WARNING: Configurations tested on DIFFERENT samples")
        for config, samples in samples_per_config.items():
            print(f"  {config}: {len(samples)} samples")

    print(f"\nConfigurations:")
    config_summary = df.groupby(['template', 'depth', 'qubits']).agg({
        'config_name': 'first',
        'correct': ['count', 'sum', 'mean']
    }).round(4)
    print(config_summary)

    # Class distribution
    print(f"\nClass distribution in test set:")
    class_dist = df[df['config_name'] == df['config_name'].unique()[0]].groupby('groundtruth').size()
    print(class_dist)

    return all_same


def mcnemar_test_pairwise(df: pd.DataFrame, config1: str, config2: str):
    """
    Perform McNemar's test between two configurations.

    McNemar's test is appropriate for paired binary outcomes (correct/incorrect).
    It tests if two configurations have significantly different success rates.
    """
    # Get predictions for both configs on same samples
    pred1 = df[df['config_name'] == config1].set_index('file_path')['correct']
    pred2 = df[df['config_name'] == config2].set_index('file_path')['correct']

    # Ensure same samples
    common_samples = pred1.index.intersection(pred2.index)
    pred1 = pred1.loc[common_samples]
    pred2 = pred2.loc[common_samples]

    # Create contingency table
    # Table: [[both_correct, config1_correct_config2_wrong],
    #         [config1_wrong_config2_correct, both_wrong]]
    both_correct = ((pred1 == 1) & (pred2 == 1)).sum()
    config1_only = ((pred1 == 1) & (pred2 == 0)).sum()
    config2_only = ((pred1 == 0) & (pred2 == 1)).sum()
    both_wrong = ((pred1 == 0) & (pred2 == 0)).sum()

    table = np.array([[both_correct, config1_only],
                      [config2_only, both_wrong]])

    # McNemar's test (exact for small sample sizes)
    try:
        result = mcnemar(table, exact=True)
        p_value = result.pvalue
    except:
        # Fall back to chi-square approximation if exact fails
        result = mcnemar(table, exact=False, correction=True)
        p_value = result.pvalue

    # Calculate effect size (odds ratio)
    if config2_only > 0:
        odds_ratio = config1_only / config2_only
    else:
        odds_ratio = float('inf') if config1_only > 0 else 1.0

    return {
        'both_correct': both_correct,
        'config1_only': config1_only,
        'config2_only': config2_only,
        'both_wrong': both_wrong,
        'statistic': result.statistic,
        'p_value': p_value,
        'odds_ratio': odds_ratio,
        'n_samples': len(common_samples)
    }


def step2_mcnemar_tests(df: pd.DataFrame):
    """Step 2: Pairwise McNemar's tests."""
    print_section_header("STEP 2: MCNEMAR'S PAIRED TESTS")

    print("\nMcNemar's Test compares paired binary outcomes (correct/incorrect)")
    print("Much more powerful than comparing mean accuracies!")
    print("H0: The two configurations have equal success rates")
    print("H1: The configurations have different success rates\n")

    configs = df['config_name'].unique()

    # Test by hyperparameter groups
    results = []

    # A. Compare by Qubits (holding template and depth constant)
    print("\n" + "-" * 80)
    print("A. EFFECT OF QUBITS (within same template and depth)")
    print("-" * 80)

    template_depth_groups = df.groupby(['template', 'depth'])

    for (template, depth), group in template_depth_groups:
        configs_in_group = group['config_name'].unique()
        if len(configs_in_group) < 2:
            continue

        # Find configs with different qubits
        config_8 = group[group['qubits'] == 8]['config_name'].unique()
        config_16 = group[group['qubits'] == 16]['config_name'].unique()

        if len(config_8) > 0 and len(config_16) > 0:
            result = mcnemar_test_pairwise(df, config_8[0], config_16[0])

            print(f"\nTemplate={template}, Depth={depth}:")
            print(f"  8 qubits vs 16 qubits")
            print(f"  Both correct: {result['both_correct']}")
            print(f"  Only 8-qubits correct: {result['config1_only']}")
            print(f"  Only 16-qubits correct: {result['config2_only']}")
            print(f"  Both wrong: {result['both_wrong']}")
            print(f"  McNemar statistic: {result['statistic']:.4f}")
            print(f"  p-value: {result['p_value']:.6f}")

            if result['p_value'] < 0.05:
                print(f"  ✓ SIGNIFICANT difference!")
            else:
                print(f"  ✗ Not significant")

            results.append({
                'comparison': 'qubits',
                'template': template,
                'depth': depth,
                'p_value': result['p_value'],
                'significant': result['p_value'] < 0.05
            })

    # B. Compare by Template (holding qubits and depth constant)
    print("\n" + "-" * 80)
    print("B. EFFECT OF TEMPLATE (within same qubits and depth)")
    print("-" * 80)

    qubit_depth_groups = df.groupby(['qubits', 'depth'])

    for (qubits, depth), group in qubit_depth_groups:
        templates_in_group = group['template'].unique()

        if len(templates_in_group) < 2:
            continue

        # Pairwise comparisons between templates
        for temp1, temp2 in combinations(sorted(templates_in_group), 2):
            config1 = group[group['template'] == temp1]['config_name'].iloc[0]
            config2 = group[group['template'] == temp2]['config_name'].iloc[0]

            result = mcnemar_test_pairwise(df, config1, config2)

            print(f"\nQubits={qubits}, Depth={depth}:")
            print(f"  {temp1} vs {temp2}")
            print(f"  Both correct: {result['both_correct']}")
            print(f"  Only {temp1} correct: {result['config1_only']}")
            print(f"  Only {temp2} correct: {result['config2_only']}")
            print(f"  Both wrong: {result['both_wrong']}")
            print(f"  McNemar statistic: {result['statistic']:.4f}")
            print(f"  p-value: {result['p_value']:.6f}")

            if result['p_value'] < 0.05:
                print(f"  ✓ SIGNIFICANT difference!")
            else:
                print(f"  ✗ Not significant")

            results.append({
                'comparison': 'template',
                'qubits': qubits,
                'depth': depth,
                'template1': temp1,
                'template2': temp2,
                'p_value': result['p_value'],
                'significant': result['p_value'] < 0.05
            })

    # C. Compare by Depth (holding template and qubits constant)
    print("\n" + "-" * 80)
    print("C. EFFECT OF DEPTH (within same template and qubits)")
    print("-" * 80)

    template_qubit_groups = df.groupby(['template', 'qubits'])

    for (template, qubits), group in template_qubit_groups:
        configs_in_group = group['config_name'].unique()
        if len(configs_in_group) < 2:
            continue

        # Find configs with different depths
        config_3 = group[group['depth'] == 3]['config_name'].unique()
        config_5 = group[group['depth'] == 5]['config_name'].unique()

        if len(config_3) > 0 and len(config_5) > 0:
            result = mcnemar_test_pairwise(df, config_3[0], config_5[0])

            print(f"\nTemplate={template}, Qubits={qubits}:")
            print(f"  Depth=3 vs Depth=5")
            print(f"  Both correct: {result['both_correct']}")
            print(f"  Only depth-3 correct: {result['config1_only']}")
            print(f"  Only depth-5 correct: {result['config2_only']}")
            print(f"  Both wrong: {result['both_wrong']}")
            print(f"  McNemar statistic: {result['statistic']:.4f}")
            print(f"  p-value: {result['p_value']:.6f}")

            if result['p_value'] < 0.05:
                print(f"  ✓ SIGNIFICANT difference!")
            else:
                print(f"  ✗ Not significant")

            results.append({
                'comparison': 'depth',
                'template': template,
                'qubits': qubits,
                'p_value': result['p_value'],
                'significant': result['p_value'] < 0.05
            })

    return pd.DataFrame(results)


def cochran_q_test(df: pd.DataFrame, group_col: str):
    """
    Perform Cochran's Q test - extension of McNemar's for 3+ paired groups.
    Tests if multiple configurations have different success rates.
    """
    # Pivot to get matrix: rows=samples, cols=configurations, values=correct/incorrect
    pivot = df.pivot_table(
        index='file_path',
        columns=group_col,
        values='correct',
        aggfunc='first'
    )

    # Remove any rows with missing data
    pivot = pivot.dropna()

    n = len(pivot)  # number of samples
    k = len(pivot.columns)  # number of configurations

    # Calculate Cochran's Q statistic
    # Q = (k-1) * [k * sum(col_sums^2) - (grand_sum)^2] / [k * grand_sum - sum(row_sums^2)]

    col_sums = pivot.sum(axis=0)  # successes per configuration
    row_sums = pivot.sum(axis=1)  # successes per sample
    grand_sum = pivot.sum().sum()  # total successes

    numerator = k * (col_sums**2).sum() - grand_sum**2
    denominator = k * grand_sum - (row_sums**2).sum()

    if denominator == 0:
        return None

    Q = (k - 1) * numerator / denominator

    # Q follows chi-square distribution with k-1 degrees of freedom
    df_q = k - 1
    p_value = 1 - chi2.cdf(Q, df_q)

    return {
        'statistic': Q,
        'df': df_q,
        'p_value': p_value,
        'n_samples': n,
        'n_groups': k,
        'group_means': col_sums / n
    }


def step3_cochran_q_tests(df: pd.DataFrame):
    """Step 3: Cochran's Q tests for multiple paired comparisons."""
    print_section_header("STEP 3: COCHRAN'S Q TEST (Multiple Paired Comparisons)")

    print("\nCochran's Q Test: Extension of McNemar's for 3+ configurations")
    print("Tests if multiple configurations have different success rates")
    print("H0: All configurations have equal success rates")
    print("H1: At least one configuration differs\n")

    # Test by each hyperparameter
    for param in ['template', 'qubits', 'depth']:
        print("\n" + "-" * 80)
        print(f"Testing effect of: {param.upper()}")
        print("-" * 80)

        result = cochran_q_test(df, param)

        if result is None:
            print("Could not compute Cochran's Q (insufficient variation)")
            continue

        print(f"\nNumber of configurations: {result['n_groups']}")
        print(f"Number of samples: {result['n_samples']}")
        print(f"\nMean success rate by {param}:")
        for group, mean_acc in result['group_means'].items():
            print(f"  {group}: {mean_acc:.4f} ({mean_acc*100:.2f}%)")

        print(f"\nCochran's Q statistic: {result['statistic']:.4f}")
        print(f"Degrees of freedom: {result['df']}")
        print(f"p-value: {result['p_value']:.6f}")

        if result['p_value'] < 0.05:
            print(f"✓ SIGNIFICANT: At least one {param} value differs significantly!")
        else:
            print(f"✗ NOT SIGNIFICANT: No significant difference across {param} values")


def step4_per_class_analysis(df: pd.DataFrame):
    """Step 4: Analyze if hyperparameters affect specific classes differently."""
    print_section_header("STEP 4: PER-CLASS ANALYSIS")

    print("\nDo hyperparameters affect specific cancer subtypes differently?")
    print("This can reveal class-specific effects masked in overall accuracy\n")

    # Get class labels (assume same across all configs)
    sample_config = df['config_name'].unique()[0]
    class_map = df[df['config_name'] == sample_config][['file_path', 'groundtruth']].drop_duplicates()
    class_map = dict(zip(class_map['file_path'], class_map['groundtruth']))

    # Add class label to all rows
    df['class'] = df['file_path'].map(class_map)

    # Analyze by class
    classes = sorted(df['class'].unique())
    print(f"Classes found: {classes}\n")

    for param in ['template', 'qubits', 'depth']:
        print("\n" + "-" * 80)
        print(f"Effect of {param.upper()} by class")
        print("-" * 80)

        class_accuracies = df.groupby(['class', param])['correct'].mean().unstack()

        print(f"\nAccuracy by class and {param}:")
        print(class_accuracies.round(4))

        # Statistical test for each class
        print(f"\nCochran's Q test per class:")
        for cls in classes:
            class_df = df[df['class'] == cls]
            result = cochran_q_test(class_df, param)

            if result is not None:
                sig_marker = "✓" if result['p_value'] < 0.05 else "✗"
                print(f"  Class {cls}: Q={result['statistic']:.4f}, p={result['p_value']:.6f} {sig_marker}")


def step5_difficulty_analysis(df: pd.DataFrame):
    """Step 5: Analyze if hyperparameters help on difficult vs easy samples."""
    print_section_header("STEP 5: DIFFICULTY-STRATIFIED ANALYSIS")

    print("\nDo hyperparameters help more on difficult samples?")
    print("Define 'difficult' as samples that most models get wrong\n")

    # Calculate difficulty score for each sample (fraction of models that got it wrong)
    difficulty_scores = df.groupby('file_path')['correct'].apply(lambda x: 1 - x.mean())

    # Classify samples by difficulty
    df['difficulty'] = df['file_path'].map(difficulty_scores)

    # Create difficulty bins
    df['difficulty_category'] = pd.cut(
        df['difficulty'],
        bins=[0, 0.25, 0.75, 1.0],
        labels=['easy', 'medium', 'hard'],
        include_lowest=True
    )

    print(f"Sample difficulty distribution:")
    print(df.groupby('difficulty_category')['file_path'].nunique())

    # Analyze hyperparameter effects by difficulty
    for param in ['template', 'qubits', 'depth']:
        print("\n" + "-" * 80)
        print(f"Effect of {param.upper()} by sample difficulty")
        print("-" * 80)

        diff_accuracies = df.groupby(['difficulty_category', param])['correct'].mean().unstack()

        print(f"\nAccuracy by difficulty and {param}:")
        print(diff_accuracies.round(4))

        # Test if effect varies by difficulty
        for difficulty in ['easy', 'medium', 'hard']:
            diff_df = df[df['difficulty_category'] == difficulty]
            if len(diff_df) == 0:
                continue

            result = cochran_q_test(diff_df, param)

            if result is not None:
                sig_marker = "✓ SIG" if result['p_value'] < 0.05 else "✗ n.s."
                print(f"  {difficulty.capitalize()} samples: Q={result['statistic']:.4f}, p={result['p_value']:.6f} {sig_marker}")


def step6_disagreement_analysis(df: pd.DataFrame, output_dir: str = "results/hypothesis_testing"):
    """Step 6: Analyze samples where configurations disagree."""
    print_section_header("STEP 6: DISAGREEMENT ANALYSIS")

    print("\nIdentify samples where configurations disagree")
    print("These are the most informative samples!\n")

    # Calculate agreement for each sample
    agreement_scores = df.groupby('file_path')['correct'].std()

    # Find samples with disagreement
    disagreement_samples = agreement_scores[agreement_scores > 0].index

    print(f"Total samples: {df['file_path'].nunique()}")
    print(f"Samples with disagreement: {len(disagreement_samples)}")
    print(f"Samples with perfect agreement: {df['file_path'].nunique() - len(disagreement_samples)}")

    if len(disagreement_samples) > 0:
        # Analyze disagreement samples
        disagreement_df = df[df['file_path'].isin(disagreement_samples)]

        print("\n--- Configurations' accuracy on disagreement samples ---")
        disagree_acc = disagreement_df.groupby(['template', 'depth', 'qubits'])['correct'].mean()
        print(disagree_acc.sort_values(ascending=False))

        # Save disagreement samples for inspection
        disagreement_detail = disagreement_df.pivot_table(
            index='file_path',
            columns=['template', 'depth', 'qubits'],
            values='correct',
            aggfunc='first'
        )

        output_path = Path(output_dir) / "disagreement_samples.csv"
        disagreement_detail.to_csv(output_path)
        print(f"\nSaved disagreement details to: {output_path}")


def step7_summary(mcnemar_results: pd.DataFrame):
    """Step 7: Comprehensive summary."""
    print_section_header("STEP 7: SUMMARY OF PAIRED ANALYSIS")

    print("\n--- McNemar's Test Results Summary ---")

    if len(mcnemar_results) == 0:
        print("No pairwise comparisons were possible")
        return

    sig_count = mcnemar_results['significant'].sum()
    total_count = len(mcnemar_results)

    print(f"\nTotal pairwise comparisons: {total_count}")
    print(f"Significant differences (p < 0.05): {sig_count}")
    print(f"Percentage significant: {sig_count/total_count*100:.1f}%")

    if sig_count > 0:
        print("\n✓ SIGNIFICANT COMPARISONS FOUND:")
        sig_results = mcnemar_results[mcnemar_results['significant']]
        for idx, row in sig_results.iterrows():
            print(f"  - {row['comparison']}: p={row['p_value']:.6f}")
            print(f"    Details: {row.to_dict()}")
    else:
        print("\n✗ NO SIGNIFICANT DIFFERENCES FOUND in any pairwise comparison")

    # By comparison type
    print("\n--- Breakdown by comparison type ---")
    for comp_type in ['qubits', 'template', 'depth']:
        subset = mcnemar_results[mcnemar_results['comparison'] == comp_type]
        if len(subset) > 0:
            sig_in_type = subset['significant'].sum()
            print(f"{comp_type.upper()}: {sig_in_type}/{len(subset)} significant comparisons")


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 80)
    print("ADVANCED PAIRED HYPOTHESIS TESTING")
    print("Using Sample-Level Predictions")
    print("=" * 80)

    # Load data
    print("\nLoading sample-level predictions...")
    df = load_all_predictions()
    print(f"Loaded {len(df)} prediction records")

    # Analysis steps
    is_paired = step1_data_overview(df)

    if not is_paired:
        print("\n⚠ WARNING: Data is not fully paired. McNemar's test requires paired data.")
        print("Proceeding with caution...")

    mcnemar_results = step2_mcnemar_tests(df)

    step3_cochran_q_tests(df)

    step4_per_class_analysis(df)

    step5_difficulty_analysis(df)

    step6_disagreement_analysis(df)

    step7_summary(mcnemar_results)

    # Save results
    output_dir = Path("results/hypothesis_testing")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(mcnemar_results) > 0:
        mcnemar_results.to_csv(output_dir / "mcnemar_results.csv", index=False)
        print(f"\nMcNemar results saved to: {output_dir / 'mcnemar_results.csv'}")

    print("\n" + "=" * 80)
    print("PAIRED ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
