# NJ Benchmark Suite - Quick Start Guide

## What You Get

A complete benchmarking and analysis pipeline for comparing 15 NJ reconstruction variants against your CTBS method:

### 📊 Scripts
1. **nj_benchmark.py** - Run all benchmarks
2. **nj_analyzer.py** - Statistical analysis  
3. **nj_visualizer.py** - Paper-ready figures
4. **run_nj_benchmark_pipeline.py** - Run everything at once
5. **nj_utils.py** - Quick utilities
6. **nj_benchmark_analysis.ipynb** - Interactive Jupyter notebook

### 📈 Outputs
- CSV files with detailed metrics
- Statistical test results (Wilcoxon, Bonferroni corrected)
- 8 publication-quality figures
- Markdown report with tables
- ROC/AUC analysis

### 📐 Metrics Tracked
- Precision, Recall, F1, IoU
- Both multiset and unique set evaluations
- True/False Positives/Negatives
- Effect sizes (Cohen's d)

## Installation

```bash
cd test
pip install -r requirements_nj_benchmark.txt
```

**Note:** The benchmark suite uses `parallel=True` mode by default, so it doesn't require the external cnp2cnp tool. If you encounter path-related errors, see [CNP2CNP_CONFIG.md](CNP2CNP_CONFIG.md).

## Running the Pipeline

### Option 1: Complete Pipeline (Recommended)

```bash
# Full benchmark on all seeds
python run_nj_benchmark_pipeline.py

# Quick test on 10 seeds
python run_nj_benchmark_pipeline.py --max-seeds 10
```

This runs:
1. Benchmark all NJ variants
2. Statistical analysis
3. Generate all figures
4. Create summary report

### Option 2: Step-by-Step

```bash
# 1. Run benchmarks
python nj_benchmark.py --max-seeds 10

# 2. Analyze results
python nj_analyzer.py

# 3. Generate figures
python nj_visualizer.py
```

### Option 3: Interactive Analysis

```bash
jupyter notebook nj_benchmark_analysis.ipynb
```

## Quick Commands

```bash
# View quick summary
python nj_utils.py summary

# Compare two methods
python nj_utils.py compare ctbs hybrid_opt

# Find best method for a metric
python nj_utils.py best multiset_f1

# Check consistency
python nj_utils.py consistency

# Analyze specific seed
python nj_utils.py seed 295

# Export top 10 methods
python nj_utils.py export 10
```

## Output Files

```
test/
├── results/
│   ├── nj_benchmark_results.csv      # Raw benchmark data
│   ├── nj_summary_statistics.csv     # Summary stats
│   ├── nj_statistical_tests.csv      # P-values, effect sizes
│   ├── nj_roc_data.csv               # ROC curve data
│   └── nj_analysis_report.md         # Markdown report
└── figures/
    ├── boxplot_multiset_f1.png       # Box plots
    ├── violin_multiset_f1.png        # Violin plots
    ├── roc_curves.png                # ROC curves
    ├── pvalue_heatmap.png            # Significance heatmap
    ├── performance_scatter.png       # Scatter plot
    ├── method_ranking.png            # Bar chart
    └── multi_metric_comparison.png   # Multi-metric bars
```

## NJ Methods Tested

1. **standard** - Standard NJ
2. **full** - Parent-retaining NJ
3. **full_cps** - Centrality-guided
4. **hybrid** - Hybrid distance-centrality
5. **hybrid_inverse_centrality** - Inverse-distance centrality
6. **adaptive_centrality** - Adaptive blending
7. **adaptive_centrality_nonlinear** - Nonlinear sigmoid
8. **adaptive_centrality_reversed** - Reversed weighting
9. **hybrid_opt** - Optimized hybrid with Q-matrix
10. **hybrid_opt_adaptive** - Adaptive alpha/beta
11. **hybrid_opt_v2** - Combined direct+inverse
12. **hybrid_opt_refined** - Normalized hybrid
13. **hybrid_anticentral_opt** - Anti-central optimization
14. **hybrid_anticentral_adaptive_v2** - Anti-central adaptive v2
15. **hybrid_anticentral_adaptive_v3** - Anti-central adaptive v3

## Example: Test Run (5 minutes)

```bash
# Quick test on 5 seeds
python run_nj_benchmark_pipeline.py --max-seeds 5

# View results
python nj_utils.py summary

# See the figures
open figures/
```

## Interpreting Results

### Statistical Tests Table
- **p-value < 0.05**: Statistically significant difference
- **p-value_corrected**: Use this (Bonferroni corrected)
- **Cohen's d > 0.5**: Medium effect size
- **Cohen's d > 0.8**: Large effect size

### Box Plots
- Box = IQR (25th-75th percentile)
- Line = Median
- Diamond = Mean
- Red dashed line = CTBS baseline

### ROC Curves
- Higher AUC = Better performance
- AUC = 0.5: Random
- AUC = 1.0: Perfect

## Customization

### Test Specific Methods

Edit `nj_benchmark.py`:
```python
def get_all_nj_algorithms():
    return [
        ("hybrid_opt", neighbor_joining_hybrid_opt),
        ("hybrid_opt_v2", neighbor_joining_hybrid_opt_v2),
        # Add your methods here
    ]
```

### Add Custom Metrics

Edit `nj_analyzer.py`:
```python
# In perform_pairwise_tests()
metrics = ['multiset_precision', 'multiset_f1', 'your_custom_metric']
```

### Create Custom Figures

```python
from nj_visualizer import create_boxplot_comparison
import pandas as pd

df = pd.read_csv('results/nj_benchmark_results.csv')
create_boxplot_comparison(df, metric='unique_f1', 
                          output_file='my_custom_plot.png')
```

## Troubleshooting

### "Import pandas could not be resolved"
```bash
pip install pandas numpy scipy matplotlib seaborn tqdm
```

### "No module named 'ctbs'"
Make sure you're running from the `test/` directory.

### Benchmark is slow
Use `--max-seeds 10` for testing. Full benchmark may take hours.

### Out of memory
Process seeds in batches or reduce `--max-seeds`.

## For Paper/Publication

### Key Figures to Include:
1. `boxplot_multiset_f1.png` - Main comparison
2. `pvalue_heatmap.png` - Statistical significance
3. `method_ranking.png` - Performance ranking
4. `roc_curves.png` - ROC analysis

### Tables to Include:
1. `nj_summary_statistics.csv` - Summary stats
2. `nj_statistical_tests.csv` - P-values

### Text to Include:
See `nj_analysis_report.md` for pre-formatted tables.

## Getting Help

- Full documentation: `README_NJ_BENCHMARK.md`
- Example notebook: `nj_benchmark_analysis.ipynb`
- Quick utilities: `python nj_utils.py`

## Citation

Include in your Methods section:

> We benchmarked 15 NJ variants against CTBS using Wilcoxon signed-rank tests 
> with Bonferroni correction. Statistical analysis and visualizations were 
> generated using custom Python scripts based on scipy, pandas, and matplotlib.

## Next Steps

1. ✅ Run quick test: `python run_nj_benchmark_pipeline.py --max-seeds 5`
2. ✅ Check results: `python nj_utils.py summary`
3. ✅ View figures: `open figures/`
4. ✅ Run full benchmark: `python run_nj_benchmark_pipeline.py`
5. ✅ Include in paper: Use figures and tables from `results/` and `figures/`

Good luck with your analysis! 🎉
