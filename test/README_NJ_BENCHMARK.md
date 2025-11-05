# NJ Variants Benchmarking Suite

A comprehensive benchmarking and analysis toolkit for evaluating Neighbor Joining (NJ) tree reconstruction variants against the CTBS method.

## Overview

This suite provides:

1. **Benchmarking** - Evaluate all NJ reconstruction variants across multiple seeds
2. **Statistical Analysis** - Compute summary statistics and perform hypothesis tests
3. **Visualization** - Generate paper-ready figures with publication-quality graphics
4. **Reporting** - Automated markdown reports with tables and statistics

## Files Structure

```
test/
├── nj_benchmark.py              # Main benchmarking script
├── nj_analyzer.py               # Statistical analysis script
├── nj_visualizer.py             # Visualization generation script
├── run_nj_benchmark_pipeline.py # Master pipeline script
├── README_NJ_BENCHMARK.md       # This file
├── results/                     # Output CSV files (generated)
│   ├── nj_benchmark_results.csv
│   ├── nj_summary_statistics.csv
│   ├── nj_statistical_tests.csv
│   ├── nj_roc_data.csv
│   └── nj_analysis_report.md
└── figures/                     # Output visualizations (generated)
    ├── boxplot_multiset_f1.png
    ├── boxplot_unique_f1.png
    ├── violin_multiset_f1.png
    ├── roc_curves.png
    ├── pvalue_heatmap.png
    ├── performance_scatter.png
    ├── method_ranking.png
    └── multi_metric_comparison.png
```

## Installation

Required Python packages:
```bash
pip install pandas numpy scipy matplotlib seaborn tqdm
```

## Quick Start

### Run Complete Pipeline

The easiest way to run everything:

```bash
cd test
python run_nj_benchmark_pipeline.py
```

This will:
1. Benchmark all NJ variants on all seeds
2. Perform statistical analysis
3. Generate all visualizations
4. Create summary report

### Test Run (First 10 Seeds)

To quickly test the pipeline:

```bash
python run_nj_benchmark_pipeline.py --max-seeds 10
```

## Individual Scripts

### 1. Benchmarking (`nj_benchmark.py`)

Evaluates all NJ reconstruction variants:

```bash
python nj_benchmark.py --output results/nj_benchmark_results.csv --max-seeds 10
```

**NJ Variants Tested:**
- `standard` - Standard NJ algorithm
- `full` - Parent-retaining NJ
- `full_cps` - Centrality-guided NJ
- `hybrid` - Hybrid distance-centrality
- `hybrid_inverse_centrality` - Inverse-distance centrality
- `adaptive_centrality` - Adaptive centrality blending
- `adaptive_centrality_nonlinear` - Nonlinear sigmoid blending
- `adaptive_centrality_reversed` - Reversed adaptive weighting
- `hybrid_opt` - Optimized hybrid with Q-matrix
- `hybrid_opt_adaptive` - Adaptive alpha/beta weighting
- `hybrid_opt_v2` - Combined direct + inverse centrality
- `hybrid_opt_refined` - Normalized hybrid with non-linear penalty
- `hybrid_anticentral_opt` - Anti-central optimization
- `hybrid_anticentral_adaptive_v2` - Anti-central adaptive v2
- `hybrid_anticentral_adaptive_v3` - Anti-central adaptive v3

**Output:** CSV file with columns:
- `seed` - Random seed used
- `method` - Reconstruction method name
- `method_type` - 'nj' or 'ctbs'
- `multiset_precision`, `multiset_recall`, `multiset_f1`, `multiset_iou` - Multiset metrics
- `unique_precision`, `unique_recall`, `unique_f1`, `unique_iou` - Unique set metrics
- `multiset_tp`, `multiset_fp`, `multiset_fn` - True/False positives/negatives (multiset)
- `unique_tp`, `unique_fp`, `unique_fn` - True/False positives/negatives (unique)
- `status` - 'success' or 'failed'

### 2. Analysis (`nj_analyzer.py`)

Performs statistical analysis on benchmark results:

```bash
python nj_analyzer.py --input results/nj_benchmark_results.csv
```

**Analysis Performed:**
- Summary statistics (mean, median, std, min, max, quartiles)
- Pairwise statistical tests (Wilcoxon signed-rank test)
- Bonferroni correction for multiple comparisons
- Effect size computation (Cohen's d)
- ROC curve data generation
- AUC (Area Under Curve) computation

**Output Files:**
1. `nj_summary_statistics.csv` - Summary statistics for each method
2. `nj_statistical_tests.csv` - Pairwise test results with p-values
3. `nj_roc_data.csv` - ROC curve data for all methods
4. `nj_analysis_report.md` - Markdown report with tables

### 3. Visualization (`nj_visualizer.py`)

Generates publication-quality figures:

```bash
python nj_visualizer.py --output-dir figures
```

**Figures Generated:**

1. **Box Plot Comparison** - Distribution of F1 scores across methods
2. **Violin Plot** - Detailed distribution for top N methods
3. **ROC Curves** - Receiver Operating Characteristic curves
4. **P-value Heatmap** - Statistical significance visualization
5. **Performance Scatter** - Multiset F1 vs Unique F1
6. **Method Ranking** - Horizontal bar chart of method performance
7. **Multi-Metric Comparison** - Grouped bar chart of multiple metrics

All figures are:
- Publication quality (300 DPI)
- Properly labeled with titles and legends
- Color-coded for clarity
- Grid-enhanced for readability

## Metrics Explained

### Multiset Metrics
Evaluate ancestor sets where multiplicities matter (e.g., cell X appears twice in ancestor set).

### Unique Metrics
Evaluate unique ancestor sets (set semantics, no duplicates).

### Metrics Computed
- **Precision** - TP / (TP + FP)
- **Recall** - TP / (TP + FN)
- **F1 Score** - 2 × (Precision × Recall) / (Precision + Recall)
- **IoU** - Intersection over Union (Jaccard index)

### Statistical Tests
- **Wilcoxon Signed-Rank Test** - Non-parametric paired test
  - Tests if CTBS significantly outperforms NJ variant
  - One-tailed test (alternative: CTBS > NJ)
- **Bonferroni Correction** - Adjusts p-values for multiple comparisons
- **Cohen's d** - Effect size measure

## Usage Examples

### Example 1: Full Benchmark on All Seeds

```bash
cd test
python run_nj_benchmark_pipeline.py
```

Expected runtime: Several hours (depends on number of seeds and CPU)

### Example 2: Quick Test on 5 Seeds

```bash
python run_nj_benchmark_pipeline.py --max-seeds 5
```

Expected runtime: ~30-60 minutes

### Example 3: Custom Output Directories

```bash
python run_nj_benchmark_pipeline.py \
    --output-dir my_results \
    --figures-dir my_figures \
    --max-seeds 20
```

### Example 4: Run Only Analysis (After Benchmark)

```bash
# Already have benchmark results
python nj_analyzer.py --input results/nj_benchmark_results.csv
python nj_visualizer.py
```

### Example 5: Generate Only Specific Figure

```python
from nj_visualizer import create_boxplot_comparison
import pandas as pd

df = pd.read_csv('results/nj_benchmark_results.csv')
df = df[df['status'] == 'success']

create_boxplot_comparison(df, metric='multiset_f1', 
                          output_file='figures/my_custom_boxplot.png')
```

## Interpreting Results

### Summary Statistics Table

```
method                          | multiset_f1_mean | multiset_f1_median | ...
--------------------------------|------------------|--------------------|-
ctbs                            | 0.7234           | 0.7156             | ...
hybrid_anticentral_adaptive_v3  | 0.6891           | 0.6823             | ...
...
```

- **Higher is better** for all metrics
- Compare `mean` and `median` to assess distribution skewness
- Check `std` to assess consistency

### Statistical Tests Table

```
method                 | metric            | pvalue  | pvalue_corrected | cohens_d | significant
-----------------------|-------------------|---------|------------------|----------|------------
hybrid_opt             | multiset_f1       | 0.0023  | 0.0345           | 0.421    | ✓
```

- **pvalue < 0.05** indicates significant difference
- **pvalue_corrected** accounts for multiple comparisons (use this!)
- **cohens_d > 0.5** indicates medium effect size
- **cohens_d > 0.8** indicates large effect size

### Box Plots

- Box shows interquartile range (IQR)
- Line inside box is median
- Diamond marker is mean
- Whiskers extend to 1.5 × IQR
- Points beyond whiskers are outliers

### ROC Curves

- Higher AUC = better performance
- AUC = 0.5 means random performance
- AUC = 1.0 means perfect performance
- Compare curves visually and by AUC values

## Advanced Usage

### Custom Metric Analysis

```python
from nj_analyzer import load_benchmark_results, perform_pairwise_tests

df = load_benchmark_results('results/nj_benchmark_results.csv')

# Test on custom metrics
custom_metrics = ['multiset_precision', 'unique_recall']
tests = perform_pairwise_tests(df, baseline_method='ctbs', 
                               metrics=custom_metrics, test='wilcoxon')
print(tests)
```

### Filter Specific Methods

```python
import pandas as pd

df = pd.read_csv('results/nj_benchmark_results.csv')

# Select only hybrid variants
hybrid_methods = [m for m in df['method'].unique() if 'hybrid' in m]
df_hybrid = df[df['method'].isin(hybrid_methods + ['ctbs'])]

# Analyze only these
from nj_analyzer import compute_summary_statistics
summary = compute_summary_statistics(df_hybrid)
print(summary)
```

### Export for LaTeX

```python
import pandas as pd

df = pd.read_csv('results/nj_summary_statistics.csv')

# Select columns
cols = ['method', 'multiset_f1_mean', 'multiset_f1_std', 'unique_f1_mean', 'unique_f1_std']
df_table = df[cols].round(4)

# Export to LaTeX
latex = df_table.to_latex(index=False, escape=False)
with open('results/table.tex', 'w') as f:
    f.write(latex)
```

## Troubleshooting

### Issue: Import errors
**Solution:** Make sure you're in the `test/` directory when running scripts.

### Issue: Memory errors with large datasets
**Solution:** Use `--max-seeds` to limit the number of seeds tested.

### Issue: Slow benchmark
**Solution:** 
- Run on fewer seeds first (`--max-seeds 10`)
- Consider running on a cluster/HPC
- Parallelize if possible (currently sequential)

### Issue: Missing figures
**Solution:** Ensure matplotlib backend is properly configured. Try:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

## Citation

If you use this benchmarking suite in your research, please cite:

```bibtex
@software{nj_benchmark_suite,
  title = {NJ Variants Benchmarking Suite for Cancer Tree Reconstruction},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/j-paszek/ctbf}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please contact [your contact info] or open an issue on GitHub.

## Changelog

### Version 1.0.0 (2025-01-05)
- Initial release
- 15 NJ variants benchmarked
- Comprehensive statistical analysis
- 8 publication-quality visualizations
- Automated pipeline with master script
