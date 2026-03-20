"""
NJ Variants Statistical Analysis Script

This script performs comprehensive statistical analysis on the benchmark results,
including hypothesis testing and summary statistics generation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, friedmanchisquare
import os


def load_benchmark_results(csv_path="results/nj_benchmark_results.csv"):
    """Load benchmark results from CSV file."""
    df = pd.read_csv(csv_path)
    # Filter only successful runs
    df = df[df['status'] == 'success'].copy()
    return df


def compute_summary_statistics(df):
    """
    Compute summary statistics for each method.
    
    Returns DataFrame with mean, std, median, min, max for each metric.
    Includes method_type and method_category for grouping.
    """
    # Group by method
    metrics = [
        'multiset_precision', 'multiset_recall', 'multiset_f1', 'multiset_iou',
        'unique_precision', 'unique_recall', 'unique_f1', 'unique_iou'
    ]
    
    summary_stats = []
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        
        stats_row = {
            'method': method, 
            'n_samples': len(method_df),
            'method_type': method_df['method_type'].iloc[0] if 'method_type' in method_df.columns else 'unknown',
            'method_category': method_df['method_category'].iloc[0] if 'method_category' in method_df.columns else 'unknown',
        }
        
        # Add NJ variant information if available
        if 'nj_variant' in method_df.columns and pd.notna(method_df['nj_variant'].iloc[0]):
            stats_row['nj_variant'] = method_df['nj_variant'].iloc[0]
        
        for metric in metrics:
            if metric in method_df.columns:
                values = method_df[metric].dropna()
                if len(values) > 0:
                    stats_row[f'{metric}_mean'] = values.mean()
                    stats_row[f'{metric}_std'] = values.std()
                    stats_row[f'{metric}_median'] = values.median()
                    stats_row[f'{metric}_min'] = values.min()
                    stats_row[f'{metric}_max'] = values.max()
                    stats_row[f'{metric}_q25'] = values.quantile(0.25)
                    stats_row[f'{metric}_q75'] = values.quantile(0.75)
        
        summary_stats.append(stats_row)
    
    return pd.DataFrame(summary_stats)


def perform_pairwise_tests(df, baseline_methods=None, metrics=None, test='wilcoxon'):
    """
    Perform pairwise statistical tests comparing methods.
    
    For each NJ variant, compares:
    1. Pure NJ vs corresponding CTBS+NJ hybrid
    2. All CTBS+NJ hybrids vs best performing CTBS+NJ method
    
    Parameters
    ----------
    df : DataFrame
        Benchmark results
    baseline_methods : list or None
        List of baseline method names to compare against (if None, uses best CTBS+NJ method)
    metrics : list
        List of metrics to test (default: all F1 and precision metrics)
    test : str
        Statistical test to use: 'wilcoxon', 'ttest', or 'mannwhitney'
    
    Returns
    -------
    DataFrame with test results including p-values and effect sizes
    """
    if metrics is None:
        metrics = ['multiset_precision', 'multiset_f1', 'unique_precision', 'unique_f1']
    
    test_results = []
    
    # Part 1: Compare each NJ variant with its CTBS+NJ hybrid counterpart
    nj_methods = df[df['method_type'] == 'nj_pure']['method'].unique()
    
    for nj_method in nj_methods:
        ctbs_hybrid_method = f'ctbs_{nj_method}'
        
        # Check if hybrid method exists
        if ctbs_hybrid_method not in df['method'].values:
            continue
        
        nj_df = df[df['method'] == nj_method].set_index('seed')
        ctbs_df = df[df['method'] == ctbs_hybrid_method].set_index('seed')
        
        # Find common seeds
        common_seeds = nj_df.index.intersection(ctbs_df.index)
        
        if len(common_seeds) < 3:
            print(f"Warning: Only {len(common_seeds)} common seeds for {nj_method} vs {ctbs_hybrid_method}. Skipping.")
            continue
        
        for metric in metrics:
            if metric not in nj_df.columns or metric not in ctbs_df.columns:
                continue
            
            # Get paired samples (CTBS+NJ as baseline, pure NJ as comparison)
            baseline_values = ctbs_df.loc[common_seeds, metric].values
            method_values = nj_df.loc[common_seeds, metric].values
            
            # Remove NaN pairs
            valid_mask = ~(np.isnan(baseline_values) | np.isnan(method_values))
            baseline_values = baseline_values[valid_mask]
            method_values = method_values[valid_mask]
            
            if len(baseline_values) < 3:
                continue
            
            # Skip if values are identical (can't perform test)
            if np.allclose(baseline_values, method_values):
                continue
            
            # Perform statistical test
            if test == 'wilcoxon':
                # Wilcoxon signed-rank test (paired, non-parametric)
                try:
                    statistic, pvalue = wilcoxon(baseline_values, method_values, 
                                                alternative='greater', zero_method='zsplit')
                    test_name = 'Wilcoxon'
                except Exception as e:
                    print(f"Error in Wilcoxon test for {nj_method} vs {ctbs_hybrid_method}, {metric}: {e}")
                    continue
            elif test == 'ttest':
                # Paired t-test
                statistic, pvalue = ttest_rel(baseline_values, method_values, alternative='greater')
                test_name = 'Paired t-test'
            elif test == 'mannwhitney':
                # Mann-Whitney U test (unpaired, non-parametric)
                statistic, pvalue = mannwhitneyu(baseline_values, method_values, alternative='greater')
                test_name = 'Mann-Whitney U'
            else:
                raise ValueError(f"Unknown test: {test}")
            
            # Compute effect size (Cohen's d)
            diff = baseline_values - method_values
            cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
            
            # Compute mean difference
            mean_diff = np.mean(baseline_values) - np.mean(method_values)
            
            # Win/Tie/Loss counts (CTBS+NJ wins when it's better)
            wins = np.sum(baseline_values > method_values)
            ties = np.sum(baseline_values == method_values)
            losses = np.sum(baseline_values < method_values)
            
            test_results.append({
                'baseline_method': ctbs_hybrid_method,
                'comparison_method': nj_method,
                'comparison_type': 'hybrid_vs_pure',
                'metric': metric,
                'test': test_name,
                'statistic': statistic,
                'pvalue': pvalue,
                'pvalue_corrected': None,  # Will be filled after Bonferroni correction
                'cohens_d': cohens_d,
                'mean_baseline': np.mean(baseline_values),
                'mean_comparison': np.mean(method_values),
                'mean_diff': mean_diff,
                'median_baseline': np.median(baseline_values),
                'median_comparison': np.median(method_values),
                'wins': wins,
                'ties': ties,
                'losses': losses,
                'n_samples': len(baseline_values),
                'significant_005': pvalue < 0.05,
                'significant_001': pvalue < 0.01,
            })
    
    # Part 2: Find best CTBS+NJ hybrid and compare all others to it
    if baseline_methods is None:
        # Find the best CTBS+NJ method by median F1 score
        ctbs_hybrid_methods = df[df['method_type'] == 'ctbs_hybrid']
        if len(ctbs_hybrid_methods) > 0 and 'multiset_f1' in ctbs_hybrid_methods.columns:
            best_ctbs = ctbs_hybrid_methods.groupby('method')['multiset_f1'].median().idxmax()
            baseline_methods = [best_ctbs]
            print(f"Using best CTBS+NJ hybrid as baseline: {best_ctbs}")
    
    if baseline_methods:
        for baseline_method in baseline_methods:
            baseline_df = df[df['method'] == baseline_method].set_index('seed')
            
            # Compare to all other methods
            other_methods = [m for m in df['method'].unique() if m != baseline_method]
            
            for comp_method in other_methods:
                comp_df = df[df['method'] == comp_method].set_index('seed')
                
                # Find common seeds
                common_seeds = baseline_df.index.intersection(comp_df.index)
                
                if len(common_seeds) < 3:
                    continue
                
                for metric in metrics:
                    if metric not in baseline_df.columns or metric not in comp_df.columns:
                        continue
                    
                    # Get paired samples
                    baseline_values = baseline_df.loc[common_seeds, metric].values
                    comparison_values = comp_df.loc[common_seeds, metric].values
                    
                    # Remove NaN pairs
                    valid_mask = ~(np.isnan(baseline_values) | np.isnan(comparison_values))
                    baseline_values = baseline_values[valid_mask]
                    comparison_values = comparison_values[valid_mask]
                    
                    if len(baseline_values) < 3:
                        continue
                    
                    # Skip if values are identical (can't perform test)
                    if np.allclose(baseline_values, comparison_values):
                        continue
                    
                    # Perform statistical test
                    if test == 'wilcoxon':
                        try:
                            statistic, pvalue = wilcoxon(baseline_values, comparison_values, 
                                                        alternative='greater', zero_method='zsplit')
                            test_name = 'Wilcoxon'
                        except Exception as e:
                            print(f"Error in Wilcoxon test for {baseline_method} vs {comp_method}, {metric}: {e}")
                            continue
                    elif test == 'ttest':
                        statistic, pvalue = ttest_rel(baseline_values, comparison_values, alternative='greater')
                        test_name = 'Paired t-test'
                    elif test == 'mannwhitney':
                        statistic, pvalue = mannwhitneyu(baseline_values, comparison_values, alternative='greater')
                        test_name = 'Mann-Whitney U'
                    else:
                        raise ValueError(f"Unknown test: {test}")
                    
                    # Compute effect size
                    diff = baseline_values - comparison_values
                    cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
                    mean_diff = np.mean(baseline_values) - np.mean(comparison_values)
                    
                    # Win/Tie/Loss counts
                    wins = np.sum(baseline_values > comparison_values)
                    ties = np.sum(baseline_values == comparison_values)
                    losses = np.sum(baseline_values < comparison_values)
                    
                    # Determine comparison type
                    comp_type = 'best_vs_all'
                    if comp_method in df[df['method_type'] == 'ctbs_hybrid']['method'].values:
                        comp_type = 'best_hybrid_vs_other_hybrid'
                    elif comp_method in df[df['method_type'] == 'nj_pure']['method'].values:
                        comp_type = 'best_hybrid_vs_pure_nj'
                    
                    test_results.append({
                        'baseline_method': baseline_method,
                        'comparison_method': comp_method,
                        'comparison_type': comp_type,
                        'metric': metric,
                        'test': test_name,
                        'statistic': statistic,
                        'pvalue': pvalue,
                        'pvalue_corrected': None,
                        'cohens_d': cohens_d,
                        'mean_baseline': np.mean(baseline_values),
                        'mean_comparison': np.mean(comparison_values),
                        'mean_diff': mean_diff,
                        'median_baseline': np.median(baseline_values),
                        'median_comparison': np.median(comparison_values),
                        'wins': wins,
                        'ties': ties,
                        'losses': losses,
                        'n_samples': len(baseline_values),
                        'significant_005': pvalue < 0.05,
                        'significant_001': pvalue < 0.01,
                    })
    
    results_df = pd.DataFrame(test_results)
    
    # Apply Bonferroni correction
    if len(results_df) > 0:
        n_tests = len(results_df)
        results_df['pvalue_corrected'] = results_df['pvalue'] * n_tests
        results_df['pvalue_corrected'] = results_df['pvalue_corrected'].clip(upper=1.0)
        results_df['significant_005_bonf'] = results_df['pvalue_corrected'] < 0.05
        results_df['significant_001_bonf'] = results_df['pvalue_corrected'] < 0.01
    
    return results_df


def compute_roc_data(df, metric='multiset_f1', threshold_range=None):
    """
    Compute ROC-like data for comparing methods.
    
    For each method, compute true positive rate and false positive rate
    at various threshold levels.
    
    Note: This is an approximation - traditional ROC requires binary classification.
    Here we treat "better than median CTBS performance" as positive class.
    """
    if threshold_range is None:
        threshold_range = np.linspace(0, 1, 50)
    
    # Get CTBS baseline
    ctbs_df = df[df['method'] == 'ctbs']
    ctbs_median = ctbs_df[metric].median()
    
    roc_data = []
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        
        for threshold in threshold_range:
            # Count how many samples exceed threshold
            tp = np.sum(method_df[metric] >= threshold)
            fn = np.sum(method_df[metric] < threshold)
            
            # TPR and FPR
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # For FPR, we need some notion of "false positives"
            # Here we use: samples above threshold but below CTBS median
            fp = np.sum((method_df[metric] >= threshold) & (method_df[metric] < ctbs_median))
            tn = np.sum(method_df[metric] < threshold)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            roc_data.append({
                'method': method,
                'metric': metric,
                'threshold': threshold,
                'tpr': tpr,
                'fpr': fpr,
                'tp': tp,
                'fn': fn,
                'fp': fp,
                'tn': tn
            })
    
    return pd.DataFrame(roc_data)


def compute_auc(roc_df, method):
    """
    Compute AUC (Area Under Curve) for a given method from ROC data.
    Uses trapezoidal integration.
    """
    method_roc = roc_df[roc_df['method'] == method].sort_values('fpr')
    
    if len(method_roc) < 2:
        return np.nan
    
    fpr = method_roc['fpr'].values
    tpr = method_roc['tpr'].values
    
    # Trapezoidal integration
    auc = np.trapz(tpr, fpr)
    
    return auc


def analyze_benchmark_results(input_csv="results/nj_benchmark_results.csv",
                              output_summary_csv="results/nj_summary_statistics.csv",
                              output_tests_csv="results/nj_statistical_tests.csv",
                              output_roc_csv="results/nj_roc_data.csv",
                              use_input_dir=True):
    """
    Main analysis function.
    
    Performs comprehensive statistical analysis and saves results.
    
    Parameters
    ----------
    input_csv : str
        Path to benchmark results CSV
    output_summary_csv : str
        Path to output summary statistics CSV
    output_tests_csv : str
        Path to output statistical tests CSV
    output_roc_csv : str
        Path to output ROC data CSV
    use_input_dir : bool
        If True, place output files in same directory as input_csv (default: True)
    """
    print("="*60)
    print("NJ Variants Statistical Analysis")
    print("="*60)
    
    # If use_input_dir, place outputs in same directory as input
    if use_input_dir:
        input_dir = os.path.dirname(input_csv)
        output_summary_csv = os.path.join(input_dir, os.path.basename(output_summary_csv))
        output_tests_csv = os.path.join(input_dir, os.path.basename(output_tests_csv))
        output_roc_csv = os.path.join(input_dir, os.path.basename(output_roc_csv))
    
    # Load data
    print("\n1. Loading benchmark results...")
    df = load_benchmark_results(input_csv)
    print(f"   Loaded {len(df)} successful runs")
    print(f"   Methods: {df['method'].nunique()}")
    print(f"   Seeds: {df['seed'].nunique()}")
    
    # Compute summary statistics
    print("\n2. Computing summary statistics...")
    summary_df = compute_summary_statistics(df)
    os.makedirs(os.path.dirname(output_summary_csv), exist_ok=True)
    summary_df.to_csv(output_summary_csv, index=False)
    print(f"   Saved to: {output_summary_csv}")
    
    # Perform statistical tests
    print("\n3. Performing pairwise statistical tests...")
    test_metrics = ['multiset_precision', 'multiset_f1', 'unique_precision', 'unique_f1']
    tests_df = perform_pairwise_tests(df, baseline_methods=None, metrics=test_metrics, test='wilcoxon')
    tests_df.to_csv(output_tests_csv, index=False)
    print(f"   Saved to: {output_tests_csv}")
    
    # Print significant results
    print("\n4. Significant results (p < 0.05, Bonferroni corrected):")
    if len(tests_df) > 0 and 'significant_005_bonf' in tests_df.columns:
        sig_results = tests_df[tests_df['significant_005_bonf'] == True]
        if len(sig_results) > 0:
            for _, row in sig_results.iterrows():
                print(f"   {row['comparison_method']:30s} | {row['metric']:20s} | "
                      f"p={row['pvalue']:.4f} (corrected: {row['pvalue_corrected']:.4f}) | "
                      f"d={row['cohens_d']:.3f}")
        else:
            print("   No significant results found after Bonferroni correction.")
    else:
        print("   No statistical tests performed (insufficient data).")
    
    # Compute ROC data
    print("\n5. Computing ROC data...")
    roc_df = compute_roc_data(df, metric='multiset_f1')
    roc_df.to_csv(output_roc_csv, index=False)
    print(f"   Saved to: {output_roc_csv}")
    
    # Compute AUC for each method
    print("\n6. Computing AUC values...")
    auc_results = []
    for method in df['method'].unique():
        auc = compute_auc(roc_df, method)
        auc_results.append({'method': method, 'auc': auc})
    
    auc_df = pd.DataFrame(auc_results).sort_values('auc', ascending=False)
    print("\n   AUC Rankings:")
    for _, row in auc_df.iterrows():
        print(f"   {row['method']:30s} | AUC = {row['auc']:.4f}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60 + "\n")
    
    return summary_df, tests_df, roc_df, auc_df


def generate_markdown_report(summary_df, tests_df, auc_df, output_file="results/nj_analysis_report.md"):
    """
    Generate a markdown report summarizing the analysis.
    """
    with open(output_file, 'w') as f:
        f.write("# NJ Variants Benchmark Analysis Report\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write("### F1 Scores (Multiset)\n\n")
        f.write("| Method | Mean | Median | Std | Min | Max |\n")
        f.write("|--------|------|--------|-----|-----|-----|\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"| {row['method']:30s} | "
                   f"{row.get('multiset_f1_mean', np.nan):.4f} | "
                   f"{row.get('multiset_f1_median', np.nan):.4f} | "
                   f"{row.get('multiset_f1_std', np.nan):.4f} | "
                   f"{row.get('multiset_f1_min', np.nan):.4f} | "
                   f"{row.get('multiset_f1_max', np.nan):.4f} |\n")
        
        f.write("\n### F1 Scores (Unique)\n\n")
        f.write("| Method | Mean | Median | Std | Min | Max |\n")
        f.write("|--------|------|--------|-----|-----|-----|\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"| {row['method']:30s} | "
                   f"{row.get('unique_f1_mean', np.nan):.4f} | "
                   f"{row.get('unique_f1_median', np.nan):.4f} | "
                   f"{row.get('unique_f1_std', np.nan):.4f} | "
                   f"{row.get('unique_f1_min', np.nan):.4f} | "
                   f"{row.get('unique_f1_max', np.nan):.4f} |\n")
        
        f.write("\n## Statistical Tests\n\n")
        f.write("Wilcoxon signed-rank test (one-tailed)\n\n")
        
        # Check if we have comparison data
        if len(tests_df) > 0:
            f.write("| Baseline Method | Comparison Method | Comparison Type | Metric | p-value | p-value (Bonf.) | Cohen's d | Significant |\n")
            f.write("|-----------------|-------------------|-----------------|--------|---------|-----------------|-----------|-------------|\n")
            
            for _, row in tests_df.iterrows():
                sig_marker = "✓" if row.get('significant_005_bonf', False) else ""
                baseline = row.get('baseline_method', 'N/A')
                comparison = row.get('comparison_method', 'N/A')
                comp_type = row.get('comparison_type', 'N/A')
                
                f.write(f"| {baseline[:28]:28s} | "
                       f"{comparison[:28]:28s} | "
                       f"{comp_type[:14]:14s} | "
                       f"{row['metric'][:12]:12s} | "
                       f"{row['pvalue']:.4f} | "
                       f"{row['pvalue_corrected']:.4f} | "
                       f"{row['cohens_d']:.3f} | "
                       f"{sig_marker} |\n")
        else:
            f.write("No statistical tests performed (insufficient data).\n\n")
        
        f.write("\n## AUC Rankings\n\n")
        f.write("| Rank | Method | AUC |\n")
        f.write("|------|--------|-----|\n")
        
        for idx, row in auc_df.iterrows():
            f.write(f"| {idx+1} | {row['method']:30s} | {row['auc']:.4f} |\n")
        
        f.write("\n---\n")
        f.write("*Report generated automatically by nj_analyzer.py*\n")
    
    print(f"Markdown report saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze NJ benchmark results')
    parser.add_argument('--input', type=str, default='results/nj_benchmark_results.csv',
                       help='Input benchmark CSV file')
    parser.add_argument('--no-use-input-dir', action='store_true',
                       help='Do not place output files in same directory as input')
    
    args = parser.parse_args()
    
    # Run analysis
    summary_df, tests_df, roc_df, auc_df = analyze_benchmark_results(
        input_csv=args.input,
        use_input_dir=not args.no_use_input_dir
    )
    
    # Generate markdown report
    input_dir = os.path.dirname(args.input)
    report_path = os.path.join(input_dir, 'nj_analysis_report.md')
    generate_markdown_report(summary_df, tests_df, auc_df, output_file=report_path)
