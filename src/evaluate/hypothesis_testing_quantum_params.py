"""
Comprehensive Hypothesis Testing for Quantum CNN Hyperparameters

This script performs rigorous statistical hypothesis testing to determine which
hyperparameters (Template, Depth, Qubits) significantly affect test set accuracy.

Statistical Methods:
1. Descriptive Statistics
2. One-way ANOVA for each factor
3. Normality tests (Shapiro-Wilk)
4. Homogeneity of variance tests (Levene's test)
5. Post-hoc pairwise comparisons (Tukey HSD)
6. Multi-way ANOVA for interaction effects
7. Effect size calculations (eta-squared, Cohen's d)
8. Visualization of distributions and effects

Dataset: processed_breakhis_balanced_50_25_25/test
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway, shapiro, levene, normaltest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')


def load_quantum_results_to_dataframe(results_dir: str = "results_quantum_config") -> pd.DataFrame:
    """
    Load all quantum configuration results into a pandas DataFrame.

    Args:
        results_dir: Path to the quantum results directory

    Returns:
        DataFrame with columns: qubits, template, depth, test_acc, val_acc, train_acc, etc.
    """
    results_path = Path(results_dir)
    data = []

    for folder in results_path.iterdir():
        if not folder.is_dir() or folder.name.startswith('.'):
            continue

        config_file = folder / "config.json"
        metrics_file = folder / "epoch20_metrics.json"

        if not config_file.exists() or not metrics_file.exists():
            print(f"Skipping {folder.name}: missing files")
            continue

        try:
            # Load config
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Load metrics
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Extract data
            qubits = config.get('quantum_cnn_n_qubits')
            template = config.get('quantum_cnn_dense_template')
            depth = config.get('quantum_cnn_dense_depth')

            if None in [qubits, template, depth]:
                print(f"Skipping {folder.name}: missing quantum parameters")
                continue

            data.append({
                'folder': folder.name,
                'qubits': qubits,
                'template': template,
                'depth': depth,
                'test_acc': metrics.get('test_acc', np.nan),
                'val_acc': metrics.get('val_acc', np.nan),
                'train_acc': metrics.get('train_acc', np.nan),
                'test_loss': metrics.get('test_loss', np.nan),
                'val_loss': metrics.get('val_loss', np.nan),
                'train_loss': metrics.get('train_loss', np.nan),
                'best_val_acc': config.get('best_val_acc', np.nan),
                'best_epoch': config.get('best_epoch', np.nan)
            })

        except Exception as e:
            print(f"Error loading {folder.name}: {e}")
            continue

    df = pd.DataFrame(data)

    # Remove any rows with NaN in critical columns
    df = df.dropna(subset=['test_acc', 'qubits', 'template', 'depth'])

    # Convert types
    df['qubits'] = df['qubits'].astype(int)
    df['depth'] = df['depth'].astype(int)
    df['template'] = df['template'].astype(str)

    return df


def calculate_effect_size(groups):
    """
    Calculate eta-squared (η²) effect size for ANOVA.

    η² = SS_between / SS_total
    Small: 0.01, Medium: 0.06, Large: 0.14
    """
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    # Between-group sum of squares
    ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)

    # Total sum of squares
    ss_total = np.sum((all_data - grand_mean)**2)

    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    return eta_squared


def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d for effect size between two groups.

    Small: 0.2, Medium: 0.5, Large: 0.8
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    return d


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def step1_descriptive_statistics(df: pd.DataFrame):
    """
    Step 1: Comprehensive descriptive statistics.
    """
    print_section_header("STEP 1: DESCRIPTIVE STATISTICS")

    print("\n--- Dataset Overview ---")
    print(f"Total number of experiments: {len(df)}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")

    print("\n--- Test Accuracy Summary Statistics ---")
    print(df['test_acc'].describe())

    print("\n--- Distribution of Hyperparameters ---")
    print("\nQubits:")
    print(df['qubits'].value_counts().sort_index())

    print("\nTemplate:")
    print(df['template'].value_counts())

    print("\nDepth:")
    print(df['depth'].value_counts().sort_index())

    print("\n--- Test Accuracy by Hyperparameter ---")

    print("\nBy Qubits:")
    qubits_stats = df.groupby('qubits')['test_acc'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ])
    print(qubits_stats)

    print("\nBy Template:")
    template_stats = df.groupby('template')['test_acc'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ])
    print(template_stats)

    print("\nBy Depth:")
    depth_stats = df.groupby('depth')['test_acc'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ])
    print(depth_stats)

    print("\n--- Combined Grouping ---")
    combined_stats = df.groupby(['template', 'depth', 'qubits'])['test_acc'].agg([
        'count', 'mean', 'std'
    ])
    print(combined_stats)

    return qubits_stats, template_stats, depth_stats


def step2_visualize_distributions(df: pd.DataFrame, output_dir: str = "results/hypothesis_testing"):
    """
    Step 2: Create visualizations of the distributions.
    """
    print_section_header("STEP 2: VISUALIZATION")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Overall distribution
    axes[0, 0].hist(df['test_acc'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Test Accuracy (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Test Accuracy Distribution')
    axes[0, 0].axvline(df['test_acc'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["test_acc"].mean():.2f}%')
    axes[0, 0].legend()

    # 2. Box plot by Qubits
    df.boxplot(column='test_acc', by='qubits', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Number of Qubits')
    axes[0, 1].set_ylabel('Test Accuracy (%)')
    axes[0, 1].set_title('Test Accuracy by Qubits')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=0)

    # 3. Box plot by Template
    df.boxplot(column='test_acc', by='template', ax=axes[0, 2])
    axes[0, 2].set_xlabel('Template')
    axes[0, 2].set_ylabel('Test Accuracy (%)')
    axes[0, 2].set_title('Test Accuracy by Template')
    plt.sca(axes[0, 2])
    plt.xticks(rotation=45, ha='right')

    # 4. Box plot by Depth
    df.boxplot(column='test_acc', by='depth', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Depth')
    axes[1, 0].set_ylabel('Test Accuracy (%)')
    axes[1, 0].set_title('Test Accuracy by Depth')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=0)

    # 5. Violin plot comparison
    data_for_violin = []
    labels_for_violin = []
    for param in ['qubits', 'template', 'depth']:
        for value in df[param].unique():
            data_for_violin.append(df[df[param] == value]['test_acc'].values)
            labels_for_violin.append(f'{param}={value}')

    sns.violinplot(data=df, x='template', y='test_acc', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Template')
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_title('Violin Plot: Test Accuracy by Template')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=45, ha='right')

    # 6. Interaction plot (Template x Depth)
    pivot_data = df.pivot_table(values='test_acc', index='depth', columns='template', aggfunc='mean')
    for col in pivot_data.columns:
        axes[1, 2].plot(pivot_data.index, pivot_data[col], marker='o', label=col)
    axes[1, 2].set_xlabel('Depth')
    axes[1, 2].set_ylabel('Mean Test Accuracy (%)')
    axes[1, 2].set_title('Interaction: Template × Depth')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    output_file = Path(output_dir) / "distributions_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    plt.close()

    # Create additional heatmap for all combinations
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_heatmap = df.pivot_table(
        values='test_acc',
        index=['template', 'depth'],
        columns='qubits',
        aggfunc='mean'
    )
    sns.heatmap(pivot_heatmap, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Test Accuracy (%)'})
    ax.set_title('Heatmap: Mean Test Accuracy by All Hyperparameters')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Template, Depth')

    plt.tight_layout()
    heatmap_file = Path(output_dir) / "heatmap_all_combinations.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {heatmap_file}")
    plt.close()


def step3_check_assumptions(df: pd.DataFrame):
    """
    Step 3: Check ANOVA assumptions (normality and homogeneity of variance).
    """
    print_section_header("STEP 3: ANOVA ASSUMPTION CHECKS")

    print("\n--- A. Normality Tests (Shapiro-Wilk Test) ---")
    print("Null Hypothesis: Data is normally distributed")
    print("If p-value > 0.05, we fail to reject H0 (data is normal)\n")

    # Test overall normality
    stat, p_value = shapiro(df['test_acc'])
    print(f"Overall test accuracy:")
    print(f"  Shapiro-Wilk statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Conclusion: {'Normal distribution' if p_value > 0.05 else 'NOT normally distributed'}")

    # Test normality for each group
    print("\n--- Normality by Qubits ---")
    for qubits in sorted(df['qubits'].unique()):
        data = df[df['qubits'] == qubits]['test_acc']
        if len(data) >= 3:
            stat, p_value = shapiro(data)
            print(f"Qubits={qubits}: W={stat:.4f}, p={p_value:.4f} {'✓ Normal' if p_value > 0.05 else '✗ Not normal'}")

    print("\n--- Normality by Template ---")
    for template in sorted(df['template'].unique()):
        data = df[df['template'] == template]['test_acc']
        if len(data) >= 3:
            stat, p_value = shapiro(data)
            print(f"Template={template}: W={stat:.4f}, p={p_value:.4f} {'✓ Normal' if p_value > 0.05 else '✗ Not normal'}")

    print("\n--- Normality by Depth ---")
    for depth in sorted(df['depth'].unique()):
        data = df[df['depth'] == depth]['test_acc']
        if len(data) >= 3:
            stat, p_value = shapiro(data)
            print(f"Depth={depth}: W={stat:.4f}, p={p_value:.4f} {'✓ Normal' if p_value > 0.05 else '✗ Not normal'}")

    print("\n\n--- B. Homogeneity of Variance (Levene's Test) ---")
    print("Null Hypothesis: Variances are equal across groups")
    print("If p-value > 0.05, we fail to reject H0 (variances are equal)\n")

    # Levene's test for qubits
    groups_qubits = [df[df['qubits'] == q]['test_acc'].values for q in df['qubits'].unique()]
    stat, p_value = levene(*groups_qubits)
    print(f"Qubits groups:")
    print(f"  Levene statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Conclusion: {'Equal variances' if p_value > 0.05 else 'Unequal variances'}")

    # Levene's test for template
    groups_template = [df[df['template'] == t]['test_acc'].values for t in df['template'].unique()]
    stat, p_value = levene(*groups_template)
    print(f"\nTemplate groups:")
    print(f"  Levene statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Conclusion: {'Equal variances' if p_value > 0.05 else 'Unequal variances'}")

    # Levene's test for depth
    groups_depth = [df[df['depth'] == d]['test_acc'].values for d in df['depth'].unique()]
    stat, p_value = levene(*groups_depth)
    print(f"\nDepth groups:")
    print(f"  Levene statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Conclusion: {'Equal variances' if p_value > 0.05 else 'Unequal variances'}")


def step4_one_way_anova(df: pd.DataFrame):
    """
    Step 4: Conduct one-way ANOVA for each hyperparameter.
    """
    print_section_header("STEP 4: ONE-WAY ANOVA TESTS")

    print("\nNull Hypothesis (H0): All group means are equal")
    print("Alternative Hypothesis (H1): At least one group mean is different")
    print("Significance level: α = 0.05")
    print("If p-value < 0.05, we reject H0 (significant difference exists)\n")

    results = {}

    # ANOVA for Qubits
    print("\n" + "-" * 80)
    print("A. ANOVA: Effect of QUBITS on Test Accuracy")
    print("-" * 80)
    groups_qubits = [df[df['qubits'] == q]['test_acc'].values for q in sorted(df['qubits'].unique())]
    f_stat, p_value = f_oneway(*groups_qubits)
    eta_sq = calculate_effect_size(groups_qubits)

    print(f"Groups: {sorted(df['qubits'].unique())}")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Effect size (η²): {eta_sq:.4f}", end=" ")

    if eta_sq < 0.01:
        effect_interpretation = "(negligible)"
    elif eta_sq < 0.06:
        effect_interpretation = "(small)"
    elif eta_sq < 0.14:
        effect_interpretation = "(medium)"
    else:
        effect_interpretation = "(large)"
    print(effect_interpretation)

    if p_value < 0.05:
        print(f"✓ SIGNIFICANT: Reject H0 - Qubits significantly affects test accuracy")
    else:
        print(f"✗ NOT SIGNIFICANT: Fail to reject H0 - Qubits does NOT significantly affect test accuracy")

    results['qubits'] = {'f_stat': f_stat, 'p_value': p_value, 'eta_squared': eta_sq, 'significant': p_value < 0.05}

    # ANOVA for Template
    print("\n" + "-" * 80)
    print("B. ANOVA: Effect of TEMPLATE on Test Accuracy")
    print("-" * 80)
    groups_template = [df[df['template'] == t]['test_acc'].values for t in sorted(df['template'].unique())]
    f_stat, p_value = f_oneway(*groups_template)
    eta_sq = calculate_effect_size(groups_template)

    print(f"Groups: {sorted(df['template'].unique())}")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Effect size (η²): {eta_sq:.4f}", end=" ")

    if eta_sq < 0.01:
        effect_interpretation = "(negligible)"
    elif eta_sq < 0.06:
        effect_interpretation = "(small)"
    elif eta_sq < 0.14:
        effect_interpretation = "(medium)"
    else:
        effect_interpretation = "(large)"
    print(effect_interpretation)

    if p_value < 0.05:
        print(f"✓ SIGNIFICANT: Reject H0 - Template significantly affects test accuracy")
    else:
        print(f"✗ NOT SIGNIFICANT: Fail to reject H0 - Template does NOT significantly affect test accuracy")

    results['template'] = {'f_stat': f_stat, 'p_value': p_value, 'eta_squared': eta_sq, 'significant': p_value < 0.05}

    # ANOVA for Depth
    print("\n" + "-" * 80)
    print("C. ANOVA: Effect of DEPTH on Test Accuracy")
    print("-" * 80)
    groups_depth = [df[df['depth'] == d]['test_acc'].values for d in sorted(df['depth'].unique())]
    f_stat, p_value = f_oneway(*groups_depth)
    eta_sq = calculate_effect_size(groups_depth)

    print(f"Groups: {sorted(df['depth'].unique())}")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Effect size (η²): {eta_sq:.4f}", end=" ")

    if eta_sq < 0.01:
        effect_interpretation = "(negligible)"
    elif eta_sq < 0.06:
        effect_interpretation = "(small)"
    elif eta_sq < 0.14:
        effect_interpretation = "(medium)"
    else:
        effect_interpretation = "(large)"
    print(effect_interpretation)

    if p_value < 0.05:
        print(f"✓ SIGNIFICANT: Reject H0 - Depth significantly affects test accuracy")
    else:
        print(f"✗ NOT SIGNIFICANT: Fail to reject H0 - Depth does NOT significantly affect test accuracy")

    results['depth'] = {'f_stat': f_stat, 'p_value': p_value, 'eta_squared': eta_sq, 'significant': p_value < 0.05}

    return results


def step5_posthoc_analysis(df: pd.DataFrame, anova_results: dict):
    """
    Step 5: Post-hoc pairwise comparisons using Tukey's HSD test.
    """
    print_section_header("STEP 5: POST-HOC PAIRWISE COMPARISONS (Tukey HSD)")

    print("\nTukey's Honestly Significant Difference (HSD) Test")
    print("Compares all pairs of groups to identify which specific groups differ")
    print("Adjusted for multiple comparisons (Family-Wise Error Rate control)\n")

    # Only perform post-hoc if ANOVA was significant
    for param_name, param_col in [('QUBITS', 'qubits'), ('TEMPLATE', 'template'), ('DEPTH', 'depth')]:
        print("\n" + "-" * 80)
        print(f"Post-hoc Analysis for {param_name}")
        print("-" * 80)

        if anova_results[param_col]['significant']:
            print(f"ANOVA was significant (p={anova_results[param_col]['p_value']:.6f})")
            print("Conducting Tukey HSD test...\n")

            tukey = pairwise_tukeyhsd(endog=df['test_acc'], groups=df[param_col], alpha=0.05)
            print(tukey)

            # Calculate Cohen's d for significant pairs
            print("\n--- Effect Sizes (Cohen's d) for Significant Pairs ---")
            summary_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

            for idx, row in summary_df.iterrows():
                if row['reject']:
                    group1_data = df[df[param_col] == row['group1']]['test_acc'].values
                    group2_data = df[df[param_col] == row['group2']]['test_acc'].values

                    cohens_d = calculate_cohens_d(group1_data, group2_data)

                    if abs(cohens_d) < 0.2:
                        effect_interp = "(negligible)"
                    elif abs(cohens_d) < 0.5:
                        effect_interp = "(small)"
                    elif abs(cohens_d) < 0.8:
                        effect_interp = "(medium)"
                    else:
                        effect_interp = "(large)"

                    print(f"{row['group1']} vs {row['group2']}: Cohen's d = {cohens_d:+.4f} {effect_interp}")
        else:
            print(f"ANOVA was NOT significant (p={anova_results[param_col]['p_value']:.6f})")
            print("Skipping post-hoc tests (no significant differences to explore)")


def step6_multifactor_anova(df: pd.DataFrame):
    """
    Step 6: Multi-way ANOVA to test for interaction effects.
    """
    print_section_header("STEP 6: MULTI-WAY ANOVA (Interaction Effects)")

    print("\nTesting for main effects and interaction effects of all three factors:")
    print("  - Qubits (main effect)")
    print("  - Template (main effect)")
    print("  - Depth (main effect)")
    print("  - Qubits × Template (interaction)")
    print("  - Qubits × Depth (interaction)")
    print("  - Template × Depth (interaction)")
    print("  - Qubits × Template × Depth (three-way interaction)\n")

    # Convert categorical variables to proper format
    df_anova = df.copy()
    df_anova['qubits'] = df_anova['qubits'].astype('category')
    df_anova['template'] = df_anova['template'].astype('category')
    df_anova['depth'] = df_anova['depth'].astype('category')

    # Fit the model with all main effects and interactions
    formula = 'test_acc ~ C(qubits) + C(template) + C(depth) + C(qubits):C(template) + C(qubits):C(depth) + C(template):C(depth)'

    try:
        model = ols(formula, data=df_anova).fit()
        anova_table = anova_lm(model, typ=2)

        print("ANOVA Table (Type II):")
        print(anova_table)

        print("\n--- Interpretation of Results ---")
        for idx, row in anova_table.iterrows():
            if idx != 'Residual':
                p_val = row['PR(>F)']
                f_val = row['F']

                if pd.notna(p_val):
                    significance = "✓ SIGNIFICANT" if p_val < 0.05 else "✗ NOT SIGNIFICANT"
                    print(f"{idx:40s}: F={f_val:8.4f}, p={p_val:.6f} {significance}")

        # Calculate R-squared
        print(f"\nModel R-squared: {model.rsquared:.4f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        print(f"\nInterpretation: The model explains {model.rsquared*100:.2f}% of variance in test accuracy")

    except Exception as e:
        print(f"Error fitting multi-way ANOVA model: {e}")
        print("This may occur if there are insufficient samples for some factor combinations")


def step7_summary_and_conclusions(df: pd.DataFrame, anova_results: dict):
    """
    Step 7: Generate comprehensive summary and statistical conclusions.
    """
    print_section_header("STEP 7: SUMMARY AND CONCLUSIONS")

    print("\n--- Research Question ---")
    print("Which hyperparameters (Template, Depth, Qubits) significantly affect")
    print("the test set accuracy on processed_breakhis_balanced_50_25_25/test?")

    print("\n--- Dataset Summary ---")
    print(f"Total experiments analyzed: {len(df)}")
    print(f"Test accuracy range: {df['test_acc'].min():.2f}% - {df['test_acc'].max():.2f}%")
    print(f"Test accuracy mean ± std: {df['test_acc'].mean():.2f}% ± {df['test_acc'].std():.2f}%")

    print("\n--- Statistical Findings ---")

    significant_factors = []
    non_significant_factors = []

    for param_name, param_key in [('Qubits', 'qubits'), ('Template', 'template'), ('Depth', 'depth')]:
        result = anova_results[param_key]
        if result['significant']:
            significant_factors.append(param_name)
            print(f"\n✓ {param_name.upper()} - SIGNIFICANT EFFECT")
            print(f"    F-statistic: {result['f_stat']:.4f}")
            print(f"    p-value: {result['p_value']:.6f}")
            print(f"    Effect size (η²): {result['eta_squared']:.4f}")
        else:
            non_significant_factors.append(param_name)
            print(f"\n✗ {param_name.upper()} - NO SIGNIFICANT EFFECT")
            print(f"    F-statistic: {result['f_stat']:.4f}")
            print(f"    p-value: {result['p_value']:.6f}")
            print(f"    Effect size (η²): {result['eta_squared']:.4f}")

    print("\n--- Final Conclusions ---")

    if significant_factors:
        print(f"\nThe following hyperparameters SIGNIFICANTLY affect test accuracy:")
        for factor in significant_factors:
            print(f"  • {factor}")

    if non_significant_factors:
        print(f"\nThe following hyperparameters do NOT significantly affect test accuracy:")
        for factor in non_significant_factors:
            print(f"  • {factor}")

    print("\n--- Recommendations ---")

    if 'template' in [f.lower() for f in significant_factors]:
        best_template = df.groupby('template')['test_acc'].mean().idxmax()
        best_template_acc = df.groupby('template')['test_acc'].mean().max()
        print(f"\n1. Template choice matters!")
        print(f"   Best template: '{best_template}' with mean accuracy {best_template_acc:.2f}%")

    if 'qubits' in [f.lower() for f in significant_factors]:
        best_qubits = df.groupby('qubits')['test_acc'].mean().idxmax()
        best_qubits_acc = df.groupby('qubits')['test_acc'].mean().max()
        print(f"\n2. Number of qubits matters!")
        print(f"   Best qubit count: {best_qubits} with mean accuracy {best_qubits_acc:.2f}%")

    if 'depth' in [f.lower() for f in significant_factors]:
        best_depth = df.groupby('depth')['test_acc'].mean().idxmax()
        best_depth_acc = df.groupby('depth')['test_acc'].mean().max()
        print(f"\n3. Circuit depth matters!")
        print(f"   Best depth: {best_depth} with mean accuracy {best_depth_acc:.2f}%")

    # Overall best configuration
    print("\n--- Best Overall Configuration ---")
    best_idx = df['test_acc'].idxmax()
    best_config = df.loc[best_idx]
    print(f"Template: {best_config['template']}")
    print(f"Depth: {best_config['depth']}")
    print(f"Qubits: {best_config['qubits']}")
    print(f"Test Accuracy: {best_config['test_acc']:.2f}%")
    print(f"Folder: {best_config['folder']}")

    print("\n" + "=" * 80)


def main():
    """
    Main function to orchestrate all hypothesis testing steps.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE HYPOTHESIS TESTING FOR QUANTUM CNN HYPERPARAMETERS")
    print("=" * 80)
    print("\nDataset: processed_breakhis_balanced_50_25_25/test")
    print("Hyperparameters under investigation:")
    print("  1. Template (quantum circuit template)")
    print("  2. Depth (circuit depth)")
    print("  3. Qubits (number of qubits)")
    print("\nDependent Variable: Test Set Accuracy (%)")
    print("=" * 80)

    # Load data
    print("\nLoading data from results_quantum_config...")
    df = load_quantum_results_to_dataframe()
    print(f"Successfully loaded {len(df)} experiments")

    # Step-by-step analysis
    step1_descriptive_statistics(df)

    step2_visualize_distributions(df)

    step3_check_assumptions(df)

    anova_results = step4_one_way_anova(df)

    step5_posthoc_analysis(df, anova_results)

    step6_multifactor_anova(df)

    step7_summary_and_conclusions(df, anova_results)

    # Save results to CSV
    output_file = "results/hypothesis_testing/statistical_analysis_results.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
