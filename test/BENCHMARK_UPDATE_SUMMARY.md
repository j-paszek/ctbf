# Benchmark Suite Update Summary

## Overview
Updated the NJ benchmarking suite to properly track and analyze **both** pure NJ variants and CTBS+NJ hybrid methods.

## Key Changes

### 1. Data Collection (`nj_benchmark.py`)

#### Previous Behavior
- Collected only NJ variant results + single CTBS result
- `reconstructed_tree` was mislabeled as "ctbs" (actually CTBS+NJ hybrid)
- Total methods tracked: 16 (15 NJ variants + 1 "CTBS")

#### New Behavior
- Collects **two results per NJ variant**:
  1. **Pure NJ** (`nj_tree`): Standard NJ reconstruction only
  2. **CTBS+NJ Hybrid** (`reconstructed_tree`): CTBS reconstruction using NJ variant for root finding
- Total methods tracked: **30 methods** (15 pure NJ + 15 CTBS+NJ hybrids)

#### Data Structure Changes
```python
# Each result now includes:
{
    'seed': int,
    'method': str,              # e.g., 'standard' or 'ctbs_standard'
    'method_type': str,         # 'nj_pure' or 'ctbs_hybrid'
    'method_category': str,     # 'NJ' or 'CTBS+NJ'
    'nj_variant': str,          # (for hybrids) which NJ variant was used
    # ... metrics ...
}
```

#### Method Naming Convention
- **Pure NJ**: Original names (e.g., `standard`, `full`, `hybrid`)
- **CTBS+NJ Hybrid**: Prefixed with `ctbs_` (e.g., `ctbs_standard`, `ctbs_full`, `ctbs_hybrid`)

---

### 2. Statistical Analysis (`nj_analyzer.py`)

#### Enhanced Summary Statistics
- Now tracks `method_type` and `method_category`
- Includes `nj_variant` field for hybrid methods
- All statistics computed separately for each of 30 methods

#### Two-Tier Comparison System

**Tier 1: Paired NJ vs CTBS+NJ**
- Compares each NJ variant to its CTBS+NJ counterpart
- Example: `standard` (pure NJ) vs `ctbs_standard` (CTBS+NJ hybrid)
- Test type: `comparison_type='hybrid_vs_pure'`
- Shows impact of adding CTBS to each NJ variant

**Tier 2: Best Method Comparisons**
- Automatically finds best-performing CTBS+NJ hybrid
- Compares all other methods to this best hybrid
- Three comparison types:
  - `best_hybrid_vs_other_hybrid`: Best CTBS+NJ vs other CTBS+NJ
  - `best_hybrid_vs_pure_nj`: Best CTBS+NJ vs pure NJ
  - `best_vs_all`: General comparisons

#### Updated Test Results Format
```python
{
    'baseline_method': str,        # Method used as baseline
    'comparison_method': str,      # Method being compared
    'comparison_type': str,        # Type of comparison
    'metric': str,
    'pvalue': float,
    'cohens_d': float,
    'mean_baseline': float,
    'mean_comparison': float,
    'mean_diff': float,           # baseline - comparison
    'wins': int,                  # Times baseline > comparison
    'ties': int,
    'losses': int,
    # ... more fields ...
}
```

---

### 3. Visualizations (`nj_visualizer.py`)

#### Updated Visualizations (11 total, was 8)

**1-2. Box Plot Comparisons** (multiset & unique F1)
- Now shows top 20 methods (was: all methods)
- Color coding:
  - 🔴 Red: CTBS+NJ hybrid methods
  - 🔵 Teal: Pure NJ methods
- Labels: `C+` prefix indicates CTBS+NJ hybrid (e.g., `C+standard`)

**3. Violin Plot**
- Top 10 methods by median performance
- Shows distribution shapes for each method

**4. ROC Curves**
- Top 8 methods
- Area Under Curve (AUC) computed for ranking

**5. P-value Heatmap**
- Shows statistical significance between method pairs
- Now includes all 30 methods

**6. Performance Scatter**
- Precision vs Recall plot
- Color-coded by method category

**7. Method Ranking Bar Chart**
- Horizontal bars showing mean F1 scores
- Top N methods

**8. Multi-Metric Comparison**
- Grouped bar chart showing 6 metrics simultaneously
- Precision, Recall, F1 for both multiset and unique modes

**9. NJ vs CTBS+NJ Comparison** ✨ NEW
- **Left panel**: Side-by-side bars showing pure NJ vs CTBS+NJ
- **Right panel**: Improvement percentage from adding CTBS
- Green bars = improvement, Red bars = degradation
- Shows top 15 NJ variants

**10-11. Category Comparisons** ✨ NEW
- Violin + Box plots comparing overall categories
- Pure NJ vs CTBS+NJ distributions
- Separate plots for multiset and unique F1

---

### 4. What This Enables

#### Scientific Analysis
1. **Quantify CTBS Impact**: See exactly how much CTBS improves each NJ variant
2. **Best Variant Identification**: Find which NJ variant works best with CTBS
3. **Variance Analysis**: Compare consistency across different NJ variants
4. **Publication-Ready Figures**: 11 high-quality visualizations at 300 DPI

#### Key Questions Answered
- Which NJ variant benefits most from CTBS?
- Does CTBS consistently improve all NJ variants?
- Which CTBS+NJ combination is overall best?
- How much better is the best CTBS+NJ vs the best pure NJ?
- Are improvements statistically significant?

---

## File Changes Summary

### Modified Files
1. **`nj_benchmark.py`** (lines 70-244)
   - Updated `evaluate_method()` to return both NJ and CTBS+NJ results
   - Changed method naming and metadata structure
   - Removed deduplication logic (was incorrectly filtering CTBS results)

2. **`nj_analyzer.py`** (lines 17-315)
   - Enhanced `compute_summary_statistics()` with method type tracking
   - Completely rewrote `perform_pairwise_tests()` for two-tier comparisons
   - Added automatic best-method detection

3. **`nj_visualizer.py`** (lines 38-530)
   - Updated `create_boxplot_comparison()` with top N selection and color coding
   - Added `create_nj_vs_ctbs_comparison()` (85 lines)
   - Added `create_category_comparison()` (60 lines)
   - Updated `create_all_visualizations()` to generate 11 plots

### New File
4. **`BENCHMARK_UPDATE_SUMMARY.md`** (this file)
   - Documentation of all changes

---

## Expected Output Structure

### CSV Files
```
results/
├── nj_benchmark_results.csv          # Raw data: 30 methods × N seeds
├── nj_summary_statistics.csv         # Aggregated stats for 30 methods
├── nj_statistical_tests.csv          # Pairwise comparisons
├── nj_roc_data.csv                   # ROC curve data
└── nj_analysis_report.md             # Markdown summary
```

### Visualization Files
```
figures/
├── boxplot_multiset_f1.png                      # Top 20 methods comparison
├── boxplot_unique_f1.png                        # Top 20 methods comparison
├── violin_multiset_f1.png                       # Top 10 distributions
├── roc_curves.png                               # Top 8 ROC curves
├── pvalue_heatmap.png                           # Statistical significance
├── performance_scatter.png                      # Precision vs Recall
├── method_ranking.png                           # Bar chart ranking
├── multi_metric_comparison.png                  # 6 metrics side-by-side
├── nj_vs_ctbs_comparison.png            ✨ NEW # Paired NJ vs CTBS+NJ
├── category_comparison_multiset_f1.png  ✨ NEW # Category distributions
└── category_comparison_unique_f1.png    ✨ NEW # Category distributions
```

---

## How to Run

### Full Pipeline (Recommended)
```bash
cd /Users/krzysiek/PROJECTS-MIMUW/ctbf/test
python run_nj_benchmark_pipeline.py --max-seeds 10
```

### Individual Steps
```bash
# 1. Run benchmark
python nj_benchmark.py --max-seeds 10 --output results/nj_benchmark_results.csv

# 2. Analyze results
python nj_analyzer.py --input results/nj_benchmark_results.csv

# 3. Create visualizations
python nj_visualizer.py --benchmark-csv results/nj_benchmark_results.csv
```

---

## Verification Checklist

After running the updated pipeline, verify:

- [ ] CSV has 30 unique methods (15 NJ + 15 CTBS+NJ)
- [ ] Each method appears N times (once per seed)
- [ ] `method_type` column has values: `nj_pure` and `ctbs_hybrid`
- [ ] `method_category` column has values: `NJ` and `CTBS+NJ`
- [ ] CTBS+NJ methods are named with `ctbs_` prefix
- [ ] Statistical tests include both `hybrid_vs_pure` and `best_vs_all` comparisons
- [ ] All 11 visualization files are generated
- [ ] Box plots show color distinction (red vs teal)
- [ ] NJ vs CTBS comparison shows improvement percentages

---

## Backward Compatibility

⚠️ **Breaking Changes**:
- Old CSV files will NOT work with new analyzer/visualizer
- Method names have changed (CTBS results now prefixed with `ctbs_`)
- Need to re-run benchmarks to generate new data format

✅ **Compatible**:
- Seed files (`data/f1results.csv`) unchanged
- Config files unchanged
- Core simulation/reconstruction logic unchanged
- cnp2cnp integration unchanged

---

## Performance Notes

- **Benchmark time**: ~3-4 hours for 15 variants × 5 seeds
- **Analysis time**: < 1 minute
- **Visualization time**: < 30 seconds
- **Total data size**: ~500KB for 5 seeds, ~5MB for 50 seeds

---

## Questions & Next Steps

### For User to Consider
1. How many seeds should we use for final analysis? (5, 10, 20, 50?)
2. Should we add more visualizations (e.g., heatmap of all pairwise comparisons)?
3. Do you want to export best-performing methods to a separate file?
4. Should we create separate analysis for multiset vs unique metrics?

### Potential Extensions
- Add confidence intervals to plots
- Create interactive HTML visualizations
- Export LaTeX tables for paper
- Add convergence analysis across seeds
- Compare variance between methods

---

*Last updated: 2025-11-05*
*Author: GitHub Copilot*
