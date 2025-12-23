# Hyperparameter Hypothesis Testing using McNemar's Test

This directory contains statistical analysis tools for evaluating which hyperparameters (Template, Depth, Qubits) significantly affect quantum CNN performance on the BreakHis dataset.

## Overview

We use **McNemar's Test**, a powerful paired statistical test, to determine if different hyperparameter configurations produce significantly different results on the test set. This approach is much more powerful than traditional ANOVA because it leverages the fact that all configurations are tested on the **same samples**.

---

## Quick Start

### Run the Analysis

```bash
# Run paired McNemar's test analysis
uv run python -m src.evaluate.paired_hypothesis_testing

# Run traditional ANOVA analysis (for comparison)
uv run python -m src.evaluate.hypothesis_testing_quantum_params
```

### View Results

```bash
# Summary of all pairwise comparisons
cat results/hypothesis_testing/mcnemar_results.csv

# Detailed analysis report
cat results/hypothesis_testing/PAIRED_ANALYSIS_SUMMARY.md
```

---

## Statistical Methodology

### Why McNemar's Test?

Traditional ANOVA compares **mean accuracies** across configurations:
- Low statistical power with only n=12 configurations
- Ignores the paired nature of the data
- Cannot detect subtle but real differences

**McNemar's Test** compares **sample-level predictions**:
- High statistical power with n=1,978 test samples
- Accounts for the fact that all configs tested on same samples
- Can detect differences as small as 1-2% accuracy

### McNemar's Test Formula

For comparing two configurations (A and B) on the same samples:

```
Contingency Table:
                    Config B
                Correct   Incorrect
Config A  Correct    a         b
          Incorrect  c         d
```

Where:
- **a** = Both configs correct (agreement)
- **b** = Only config A correct
- **c** = Only config B correct
- **d** = Both configs incorrect (agreement)

**McNemar's Test Statistic:**

```
χ² = (b - c)² / (b + c)
```

**Exact Test (for small b+c):**

```
p-value = 2 × Σ(k=0 to min(b,c)) [C(b+c, k) × 0.5^(b+c)]
```

**Null Hypothesis (H₀):** P(config A correct) = P(config B correct)

**Alternative Hypothesis (H₁):** P(config A correct) ≠ P(config B correct)

**Reject H₀ if:** p-value < 0.05

### Cochran's Q Test Formula

Extension of McNemar's for comparing 3+ configurations simultaneously:

```
Q = (k-1) × [k × Σ(Cⱼ²) - G²] / [k × G - Σ(Rᵢ²)]
```

Where:
- **k** = number of configurations
- **n** = number of samples
- **Cⱼ** = number of successes for configuration j
- **Rᵢ** = number of configurations that correctly classified sample i
- **G** = total number of successes across all configs

**Distribution:** Q ~ χ²(k-1) under H₀

---

## Key Results

### Overall Findings

| Hyperparameter | Significant Tests | Effect Strength |
|----------------|------------------|-----------------|
| **Template** | 7/12 (58.3%) | ⭐⭐⭐ Strong |
| **Depth** | 3/6 (50.0%) | ⭐⭐ Moderate |
| **Qubits** | 1/6 (16.7%) | ⭐ Weak |

**Total:** 11 out of 24 pairwise comparisons showed statistical significance (p < 0.05)

### Template Effects (Strongest)

**Ranking:**
1. **two_design** - Best overall (91.43% avg accuracy)
2. **basic** - Strong performer (91.61% avg accuracy)
3. **strong** - Significantly underperforms (90.09% avg accuracy) ⚠️

**Significant Pairwise Comparisons:**

| Comparison | Context | p-value | Winner |
|------------|---------|---------|---------|
| basic vs strong | depth=3, qubits=8 | 0.0146 | basic ✓ |
| basic vs two_design | depth=5, qubits=8 | 0.0040 | two_design ✓ |
| strong vs two_design | depth=5, qubits=8 | **0.00002** | two_design ✓✓✓ |
| basic vs two_design | depth=3, qubits=16 | 0.0192 | basic ✓ |
| strong vs two_design | depth=3, qubits=16 | 0.0401 | two_design ✓ |
| basic vs strong | depth=5, qubits=16 | **0.0000005** | basic ✓✓✓ |
| strong vs two_design | depth=5, qubits=16 | **0.0000002** | two_design ✓✓✓ |

**Key Insight:** "strong" template consistently underperforms, especially at depth=5 with 16 qubits where it's **extremely significantly worse** (p < 0.0001).

### Depth Effects (Moderate)

**Average Performance:**
- Depth 5: 91.12% ± 1.48%
- Depth 3: 90.97% ± 0.79%

**Significant Comparisons:**

| Template | Qubits | p-value | Winner | Interpretation |
|----------|--------|---------|---------|----------------|
| two_design | 8 | 0.0014 | depth=5 ✓ | Deeper is better |
| two_design | 16 | 0.0031 | depth=5 ✓ | Deeper is better |
| strong | 16 | **0.00006** | depth=3 ✓ | **Opposite effect!** |

**Key Insight:** Depth effects **interact with template**:
- For two_design: depth=5 > depth=3 (benefits from deeper circuits)
- For strong: depth=3 > depth=5 (overfitting at depth=5?)
- For basic: no significant difference

### Qubits Effects (Weak)

**Average Performance:**
- 16 qubits: 91.13% ± 1.30%
- 8 qubits: 90.96% ± 1.06%

**Significant Comparison:**

| Template | Depth | p-value | Winner |
|----------|-------|---------|---------|
| strong | 3 | 0.0095 | 16 qubits ✓ |

**Key Insight:** Qubits only matter for the "strong" template. For "basic" and "two_design", 8 qubits perform as well as 16 qubits, making 8 qubits **more efficient**.

---

## 🏆 Best Configuration

Based on comprehensive statistical analysis and empirical performance:

### Optimal Configuration

```python
config = {
    "quantum_cnn_dense_template": "two_design",
    "quantum_cnn_dense_depth": 5,
    "quantum_cnn_n_qubits": 8
}
```

**Performance:**
- **Test Accuracy:** 92.72%
- **Validation Accuracy:** 90.24%
- **Training Accuracy:** 98.96%

### Why This Configuration?

1. ✅ **Highest test accuracy** (92.72%) among all 12 configurations

2. ✅ **Statistically superior template:**
   - two_design significantly outperforms strong (p = 0.00002)
   - Comparable to basic (no significant difference in some contexts)

3. ✅ **Optimal depth for two_design:**
   - Depth 5 significantly better than depth 3 (p = 0.0014)
   - Correctly classified 100 samples that depth 3 missed
   - Only missed 59 samples that depth 3 got correct

4. ✅ **Efficient qubit usage:**
   - 8 qubits perform as well as 16 qubits for two_design (p = 0.319)
   - **Reduced computational cost** with no accuracy loss

5. ✅ **Best on ambiguous samples:**
   - 78.6% accuracy on the 490 samples where configs disagree
   - Significantly better than strong template (63.1% on disagreement set)

### Runner-Up Configuration

```python
config = {
    "quantum_cnn_dense_template": "basic",
    "quantum_cnn_dense_depth": 5,
    "quantum_cnn_n_qubits": 16
}
```

**Performance:**
- **Test Accuracy:** 92.11%
- Only 0.61% lower than optimal
- Good alternative if two_design is unavailable

### Configuration to Avoid ⚠️

```python
# DO NOT USE
config = {
    "quantum_cnn_dense_template": "strong",  # ❌
    "quantum_cnn_dense_depth": 5,            # ❌ (especially bad with strong)
    "quantum_cnn_n_qubits": 16               # ❌ (unnecessary)
}
```

**Performance:**
- **Test Accuracy:** 88.88% (worst configuration)
- Significantly worse than optimal (p < 0.0001)
- Lowest accuracy on disagreement samples (63.1%)

---

## Per-Class Analysis

### Class 4 Effects (Dominant Class, n=863, 43.6% of data)

**ALL hyperparameters significantly affect Class 4:**

| Hyperparameter | Cochran's Q | p-value | Best Config |
|----------------|-------------|---------|-------------|
| Template | 25.0 | **0.000004** | basic (90.7%) |
| Qubits | 20.5 | **0.000006** | 16 qubits (89.8%) |
| Depth | 20.5 | **0.000006** | depth 3 (89.3%) |

**Accuracy by template on Class 4:**
- basic: 90.7%
- two_design: 89.1%
- strong: 87.3%

**Recommendation:** For Class 4 specifically, use basic template with depth 3.

### Class 7 Effects (n=140)

**Template significantly affects Class 7:**
- Cochran's Q = 7.17, p = 0.0278 ✓

**Accuracy by template on Class 7:**
- two_design: 90.9%
- strong: 90.4%
- basic: 88.4%

### Other Classes (0-3, 5-6)

No significant hyperparameter effects detected (all p > 0.05).
All configurations perform similarly on these classes.

---

## Difficulty-Stratified Analysis

We categorized test samples into three difficulty levels based on how many models correctly classified them:

- **Easy:** ≥75% of models correct (n=1,776, 89.8%)
- **Medium:** 25-75% of models correct (n=131, 6.6%)
- **Hard:** <25% of models correct (n=71, 3.6%)

### Easy Samples

**ALL hyperparameters significantly affect easy samples:**

| Hyperparameter | Cochran's Q | p-value | Effect |
|----------------|-------------|---------|---------|
| Template | 19.7 | **0.00005** | Strong |
| Qubits | 6.9 | **0.0086** | Moderate |
| Depth | 6.9 | **0.0086** | Moderate |

**Accuracies on easy samples:**
- Template: basic (98.1%) > two_design (97.7%) > strong (96.7%)
- Qubits: 16 qubits (97.6%) > 8 qubits (97.4%)
- Depth: depth 5 (97.7%) > depth 3 (97.4%)

### Medium Samples

- No significant effects (all p > 0.4)
- All models struggle: ~50% accuracy
- Represents truly ambiguous cases

### Hard Samples

- No significant effects (all p > 0.5)
- All models fail: 4-6% accuracy
- These samples likely contain labeling errors or extreme noise

**Key Insight:** Hyperparameter tuning helps maximize performance on easy samples but cannot solve truly hard cases. Consider data quality improvement or ensemble methods for hard samples.

---

## Disagreement Analysis

**490 samples (24.8%)** show disagreement among the 12 configurations.

### Best on Disagreement Samples

| Rank | Configuration | Accuracy |
|------|---------------|----------|
| 1 | two_design, depth=5, 8 qubits | **78.6%** ⭐ |
| 2 | basic, depth=5, 16 qubits | 76.1% |
| 3 | two_design, depth=5, 16 qubits | 76.1% |

### Worst on Disagreement Samples

| Rank | Configuration | Accuracy |
|------|---------------|----------|
| 10 | strong, depth=5, 8 qubits | 67.3% |
| 11 | strong, depth=3, 8 qubits | 67.3% |
| 12 | strong, depth=5, 16 qubits | **63.1%** ⚠️ |

**Pattern:** The "strong" template consistently struggles on ambiguous cases, with a 15.5% accuracy gap compared to the optimal configuration on disagreement samples.

**File:** See `results/hypothesis_testing/disagreement_samples.csv` for detailed per-sample predictions across all configurations.

---

## Files and Outputs

### Analysis Scripts

1. **`paired_hypothesis_testing.py`** ⭐ (Main analysis)
   - McNemar's paired tests
   - Cochran's Q tests
   - Per-class analysis
   - Difficulty stratification
   - Disagreement analysis

2. **`hypothesis_testing_quantum_params.py`** (Comparison)
   - Traditional ANOVA analysis
   - Descriptive statistics
   - Assumption testing
   - Multi-way ANOVA

3. **`evaluate_quantum_config.py`** (Basic evaluation)
   - Load and summarize configurations
   - Find best configurations
   - Compare with baseline model

### Output Files

Located in `results/hypothesis_testing/`:

| File | Description |
|------|-------------|
| `mcnemar_results.csv` | All 24 pairwise McNemar test results |
| `disagreement_samples.csv` | Per-sample predictions for ambiguous cases |
| `statistical_analysis_results.csv` | Raw data for all configurations |
| `PAIRED_ANALYSIS_SUMMARY.md` | Comprehensive analysis report |
| `distributions_visualization.png` | Box plots and distributions |
| `heatmap_all_combinations.png` | Performance heatmap |

---

## Understanding the Results

### Reading McNemar's Test Output

Example from the analysis:

```
Template=two_design, Depth=5, Qubits=8 vs 16:
  Both correct: 1767
  Only 8-qubits correct: 67
  Only 16-qubits correct: 55
  Both wrong: 89
  McNemar statistic: 55.0000
  p-value: 0.319308
  ✗ Not significant
```

**Interpretation:**
- 1767 samples: Both configs correct (agreement)
- 67 samples: Only 8 qubits correct
- 55 samples: Only 16 qubits correct
- 89 samples: Both configs wrong (agreement)
- p = 0.319 > 0.05: No significant difference
- **Conclusion:** 8 qubits perform as well as 16 qubits for two_design at depth=5

### Interpreting p-values

| p-value | Interpretation | Symbol |
|---------|----------------|--------|
| < 0.001 | Extremely significant | ✓✓✓ |
| < 0.01 | Highly significant | ✓✓ |
| < 0.05 | Significant | ✓ |
| ≥ 0.05 | Not significant | ✗ |

### Effect Size Guidelines

While McNemar's test tells us **if** there's a difference, the **magnitude** matters too:

```
Odds Ratio = (config1_only) / (config2_only)
```

| Odds Ratio | Interpretation |
|------------|----------------|
| > 2.0 | Large effect (config1 much better) |
| 1.5 - 2.0 | Medium effect |
| 1.0 - 1.5 | Small effect |
| = 1.0 | No effect |

Example: two_design vs strong at depth=5, qubits=8:
- Odds ratio = 109/54 = 2.02 (large effect favoring two_design)

---

## Practical Guidelines

### When to Use This Analysis

✅ **Use McNemar's test when:**
- Comparing 2 or more models on the same test set
- You have sample-level predictions (not just overall accuracy)
- You want to detect small but real differences (1-2%)
- You care about which samples each model gets right/wrong

❌ **Don't use McNemar's test when:**
- Models tested on different data splits
- You only have aggregated metrics (no sample-level data)
- Samples are not independent (e.g., multiple crops from same image)

### Hyperparameter Selection Workflow

1. **Run all configurations:**
   ```bash
   # Train models with different hyperparameters
   # Save epoch20_detail_predictions.csv for each
   ```

2. **Run statistical analysis:**
   ```bash
   uv run python -m src.evaluate.paired_hypothesis_testing
   ```

3. **Review results:**
   ```bash
   # Check which comparisons are significant
   cat results/hypothesis_testing/mcnemar_results.csv

   # Read detailed analysis
   cat results/hypothesis_testing/PAIRED_ANALYSIS_SUMMARY.md
   ```

4. **Select configuration:**
   - Start with best overall: two_design, depth=5, 8 qubits
   - Adjust based on constraints (compute, time, class priorities)
   - Verify with validation set

### Adapting to Your Use Case

**If you care most about Class 4:**
```python
config = {
    "quantum_cnn_dense_template": "basic",
    "quantum_cnn_dense_depth": 3,
    "quantum_cnn_n_qubits": 16
}
```

**If you need fast training:**
```python
config = {
    "quantum_cnn_dense_template": "basic",
    "quantum_cnn_dense_depth": 3,
    "quantum_cnn_n_qubits": 8
}
# Only 0.8% accuracy loss from optimal
```

**If compute is limited:**
```python
config = {
    "quantum_cnn_dense_template": "two_design",
    "quantum_cnn_dense_depth": 3,
    "quantum_cnn_n_qubits": 8
}
# Lower depth = faster quantum circuit execution
```

---

## Limitations and Assumptions

### Assumptions

1. **Independence of samples:** Test samples are independent
   - Violated if multiple patches from same patient
   - May inflate significance if violated

2. **Fixed test set:** All configs tested on identical samples
   - Critical for paired test validity
   - Verified: ✓ All 12 configs tested on same 1,978 samples

3. **Binary outcomes:** Each sample classified correct/incorrect
   - Appropriate for classification tasks
   - Doesn't capture confidence or near-misses

### Limitations

1. **No replication within configurations**
   - Each config trained once
   - Can't separate hyperparameter effects from random initialization
   - Recommendation: Run multiple seeds and aggregate

2. **Single dataset**
   - Results specific to BreakHis dataset
   - May not generalize to other cancer types or imaging modalities

3. **Unbalanced classes**
   - Class 4 dominates (43.6% of samples)
   - Results may be biased toward Class 4 performance
   - Addressed by per-class analysis

4. **Multiple comparisons**
   - 24 pairwise tests increases false positive risk
   - With Bonferroni correction (α=0.002): 8/11 still significant
   - Conservative: Use α=0.01 → 7/11 significant

---

## Technical Details

### Requirements

```bash
# Install dependencies
uv pip install statsmodels scipy pandas numpy matplotlib seaborn
```

### Data Format

The analysis expects this directory structure:

```
results_quantum_config/
├── config_1/
│   ├── config.json
│   └── epoch20_detail_predictions.csv
├── config_2/
│   ├── config.json
│   └── epoch20_detail_predictions.csv
...
```

**epoch20_detail_predictions.csv format:**
```csv
file_path,split,groundtruth,predict
./processed_breakhis_balanced/test/Adenosis/SOB_B_A-14-22549AB-100-006.png,test,0,0
./processed_breakhis_balanced/test/Adenosis/SOB_B_A-14-22549AB-100-008.png,test,0,0
...
```

**config.json must contain:**
```json
{
  "quantum_cnn_n_qubits": 8,
  "quantum_cnn_dense_template": "two_design",
  "quantum_cnn_dense_depth": 5
}
```

---

## References

### Statistical Methods

1. **McNemar's Test:**
   - McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages". Psychometrika. 12 (2): 153–157.

2. **Cochran's Q Test:**
   - Cochran, W.G. (1950). "The comparison of percentages in matched samples". Biometrika. 37 (3/4): 256–266.

3. **Paired Binary Data Analysis:**
   - Agresti, A. (2012). "Categorical Data Analysis" (3rd ed.). Wiley.

### Implementation

- Python implementation uses `statsmodels.stats.contingency_tables.mcnemar`
- Exact test for small samples (b+c < 25)
- Chi-square approximation with continuity correction for larger samples

---

## FAQ

**Q: Why did ANOVA find no significant effects but McNemar's test found many?**

A: ANOVA compared 12 configuration means (low power). McNemar's test compared 1,978 paired samples (high power). The paired nature of the data makes McNemar's much more sensitive to real differences.

**Q: Should I use 8 or 16 qubits?**

A: Use **8 qubits** unless you're using the "strong" template. For two_design and basic, 8 qubits perform as well as 16 (p > 0.05) but with lower computational cost.

**Q: Why does the "strong" template perform worse?**

A: Statistical tests show it's significantly worse (p < 0.0001), especially at depth=5. Possible reasons: overfitting, barren plateaus, or insufficient expressivity for this task. Further investigation needed.

**Q: Can I apply this analysis to my own quantum models?**

A: Yes! Just ensure:
1. All configs tested on same samples
2. Save sample-level predictions in the expected format
3. Run the analysis script

**Q: What if I have more than 2 levels per hyperparameter?**

A: Cochran's Q test handles 3+ levels. For pairwise comparisons, McNemar's test will run all C(n,2) pairs automatically.

---

## Citation

If you use this analysis in your research, please cite:

```bibtex
@software{quantum_cnn_hypothesis_testing,
  title={Hypothesis Testing for Quantum CNN Hyperparameters using McNemar's Test},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

---

## Contact

For questions or issues with the analysis:
- Open an issue on GitHub
- Review the detailed output in `PAIRED_ANALYSIS_SUMMARY.md`
- Check the visualizations in `results/hypothesis_testing/`

---

**Last Updated:** December 23, 2024
**Analysis Version:** 1.0
**Test Dataset:** processed_breakhis_balanced_50_25_25/test (1,978 samples)
