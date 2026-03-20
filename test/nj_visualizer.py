"""
NJ Variants Visualization Script

This script creates paper-ready visualizations of the benchmark results,
including box plots, ROC curves, and comparison plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os


# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def load_data(benchmark_csv, summary_csv, tests_csv, roc_csv):
    """Load all analysis data."""
    df_benchmark = pd.read_csv(benchmark_csv)
    df_benchmark = df_benchmark[df_benchmark['status'] == 'success']
    
    df_summary = pd.read_csv(summary_csv)
    df_tests = pd.read_csv(tests_csv)
    df_roc = pd.read_csv(roc_csv)
    
    return df_benchmark, df_summary, df_tests, df_roc


def create_boxplot_comparison(df, metric='multiset_f1', output_file='figures/boxplot_comparison.png',
                              top_n=20):
    """
    Create box plot comparing top N methods on a specific metric.
    Colors distinguish between NJ pure and CTBS+NJ hybrid methods.
    """
    # Sort methods by median performance and take top N
    method_order = df.groupby('method')[metric].median().sort_values(ascending=False).head(top_n).index.tolist()
    df_plot = df[df['method'].isin(method_order)]
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Prepare data for box plot
    plot_data = []
    plot_labels = []
    colors = []
    
    for method in method_order:
        method_data = df_plot[df_plot['method'] == method][metric].dropna()
        plot_data.append(method_data)
        
        # Create shorter labels for CTBS+NJ methods
        if method.startswith('ctbs_'):
            label = method[5:]  # Remove 'ctbs_' prefix
            plot_labels.append(f'C+{label}')
            colors.append('#FF6B6B')  # Red for CTBS+NJ hybrids
        else:
            plot_labels.append(method)
            colors.append('#4ECDC4')  # Teal for pure NJ
    
    # Create box plot
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='yellow', markersize=4))
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Rotate labels
    ax.set_xticklabels(plot_labels, rotation=60, ha='right', fontsize=8)
    
    # Labels and title
    metric_name = metric.replace('_', ' ').title()
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel('Method (C+ = CTBS+NJ hybrid)', fontsize=12)
    ax.set_title(f'Top {top_n} Reconstruction Methods: {metric_name}', fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', alpha=0.7, label='CTBS+NJ Hybrid'),
        Patch(facecolor='#4ECDC4', alpha=0.7, label='Pure NJ')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Box plot saved to: {output_file}")


def create_violin_plot(df, metric='multiset_f1', output_file='figures/violin_comparison.png',
                       top_n=10):
    """
    Create violin plot for top N methods.
    """
    # Select top N methods by median
    top_methods = df.groupby('method')[metric].median().nlargest(top_n).index.tolist()
    df_top = df[df['method'].isin(top_methods)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create violin plot
    sns.violinplot(data=df_top, x='method', y=metric, ax=ax, inner='box', palette='Set2')
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Labels
    metric_name = metric.replace('_', ' ').title()
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'Distribution of {metric_name} (Top {top_n} Methods)', fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Violin plot saved to: {output_file}")


def create_roc_curves(df_roc, output_file='figures/roc_curves.png', top_n=8):
    """
    Create ROC curves for top N methods.
    """
    # Compute AUC for each method
    auc_values = {}
    for method in df_roc['method'].unique():
        method_roc = df_roc[df_roc['method'] == method].sort_values('fpr')
        if len(method_roc) > 1:
            auc = np.trapz(method_roc['tpr'].values, method_roc['fpr'].values)
            auc_values[method] = auc
    
    # Select top N by AUC
    top_methods = sorted(auc_values.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_method_names = [m[0] for m in top_methods]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_method_names)))
    
    for method, color in zip(top_method_names, colors):
        method_roc = df_roc[df_roc['method'] == method].sort_values('fpr')
        auc = auc_values[method]
        
        # Highlight CTBS
        linewidth = 3 if method == 'ctbs' else 1.5
        linestyle = '-' if method == 'ctbs' else '-'
        
        ax.plot(method_roc['fpr'], method_roc['tpr'], 
               label=f'{method} (AUC={auc:.3f})',
               color=color, linewidth=linewidth, linestyle=linestyle)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.500)')
    
    # Labels and styling
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Method Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to: {output_file}")


def create_pvalue_heatmap(df_tests, output_file='figures/pvalue_heatmap.png'):
    """
    Create heatmap of p-values from statistical tests.
    Shows comparisons between baseline and comparison methods.
    """
    if df_tests.empty:
        print("No test data available. Skipping p-value heatmap.")
        return
    
    # Check if we have the new column structure
    if 'baseline_method' not in df_tests.columns or 'comparison_method' not in df_tests.columns:
        print("Test data does not have baseline/comparison method columns. Skipping heatmap.")
        return
    
    # Create a visualization showing comparisons
    fig, ax = plt.subplots(figsize=(14, max(8, len(df_tests) * 0.3)))
    
    # Create labels for each comparison
    df_tests = df_tests.copy()
    df_tests['comparison_label'] = (df_tests['baseline_method'].str[:20] + ' vs\n' + 
                                     df_tests['comparison_method'].str[:20] + '\n(' + 
                                     df_tests['metric'] + ')')
    
    # Sort by p-value
    df_tests = df_tests.sort_values('pvalue')
    
    # Create horizontal bar plot
    y_pos = np.arange(len(df_tests))
    colors = ['#2ECC71' if p < 0.05 else '#E74C3C' for p in df_tests['pvalue_corrected']]
    
    bars = ax.barh(y_pos, df_tests['pvalue'], color=colors, alpha=0.7)
    
    # Add p-value annotations
    for i, (_, row) in enumerate(df_tests.iterrows()):
        pval_text = f"{row['pvalue']:.4f}"
        if row['pvalue_corrected'] < 0.05:
            pval_text += " *"
        ax.text(row['pvalue'], i, f"  {pval_text}", va='center', fontsize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_tests['comparison_label'], fontsize=7)
    ax.set_xlabel('p-value (Wilcoxon signed-rank test)', fontsize=11)
    ax.set_title('Statistical Tests: Pairwise Comparisons\nGreen: Significant (p<0.05 Bonferroni), Red: Not significant',
                fontsize=13, fontweight='bold')
    ax.axvline(x=0.05, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='p=0.05 threshold')
    ax.set_xlim(0, min(1.0, max(df_tests['pvalue']) * 1.1))
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"P-value heatmap saved to: {output_file}")


def create_performance_scatter(df, output_file='figures/performance_scatter.png'):
    """
    Create scatter plot: multiset F1 vs unique F1.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Aggregate by method
    methods = df['method'].unique()
    
    for method in methods:
        method_df = df[df['method'] == method]
        x = method_df['multiset_f1']
        y = method_df['unique_f1']
        
        # Highlight CTBS
        if method == 'ctbs':
            ax.scatter(x, y, label=method, s=100, alpha=0.8, edgecolors='black', linewidth=2, zorder=10)
        else:
            ax.scatter(x, y, label=method, s=50, alpha=0.6)
    
    # Diagonal line
    max_val = max(df['multiset_f1'].max(), df['unique_f1'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
    
    # Labels
    ax.set_xlabel('Multiset F1 Score', fontsize=12)
    ax.set_ylabel('Unique F1 Score', fontsize=12)
    ax.set_title('Performance Comparison: Multiset vs Unique F1', fontsize=14, fontweight='bold')
    
    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Performance scatter plot saved to: {output_file}")


def create_method_ranking_bar(df_summary, metric='multiset_f1_mean', 
                              output_file='figures/method_ranking.png', top_n=15):
    """
    Create horizontal bar chart ranking methods by performance.
    """
    # Sort and select top N
    df_sorted = df_summary.sort_values(metric, ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors: CTBS in red, others in teal
    colors = ['#FF6B6B' if m == 'ctbs' else '#4ECDC4' for m in df_sorted['method']]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(df_sorted))
    ax.barh(y_pos, df_sorted[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['method'])
    
    metric_name = metric.replace('_', ' ').title()
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_title(f'Method Performance Ranking: {metric_name.replace(" Mean", "")} (Top {top_n})',
                fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, v in enumerate(df_sorted[metric]):
        ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Method ranking bar chart saved to: {output_file}")


def create_multi_metric_comparison(df_summary, output_file='figures/multi_metric_comparison.png',
                                   top_n=10):
    """
    Create grouped bar chart comparing multiple metrics across methods.
    """
    # Select top N methods by multiset F1
    top_methods = df_summary.nlargest(top_n, 'multiset_f1_mean')['method'].tolist()
    df_top = df_summary[df_summary['method'].isin(top_methods)]
    
    # Metrics to compare
    metrics = [
        ('multiset_precision_mean', 'Precision (M)'),
        ('multiset_recall_mean', 'Recall (M)'),
        ('multiset_f1_mean', 'F1 (M)'),
        ('unique_precision_mean', 'Precision (U)'),
        ('unique_recall_mean', 'Recall (U)'),
        ('unique_f1_mean', 'F1 (U)')
    ]
    
    # Prepare data
    methods = df_top['method'].tolist()
    x = np.arange(len(methods))
    width = 0.13
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create grouped bars
    for i, (metric_col, metric_label) in enumerate(metrics):
        values = df_top[metric_col].values
        offset = width * (i - len(metrics)/2 + 0.5)
        ax.bar(x + offset, values, width, label=metric_label, alpha=0.8)
    
    # Labels
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Multi-Metric Performance Comparison (Top {top_n} Methods)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-metric comparison saved to: {output_file}")


def create_nj_vs_ctbs_comparison(df, metric='multiset_f1', 
                                  output_file='figures/nj_vs_ctbs_hybrid_comparison.png',
                                  top_n=15):
    """
    Create a paired comparison showing NJ pure vs CTBS+NJ hybrid for each variant.
    Uses side-by-side bars to show the improvement from CTBS.
    """
    # Get NJ variants that have both pure and hybrid versions
    nj_pure_methods = df[df['method_type'] == 'nj_pure']['method'].unique()
    
    # Collect data for each NJ variant
    comparison_data = []
    for nj_method in nj_pure_methods:
        ctbs_method = f'ctbs_{nj_method}'
        
        if ctbs_method in df['method'].values:
            nj_median = df[df['method'] == nj_method][metric].median()
            ctbs_median = df[df['method'] == ctbs_method][metric].median()
            improvement = ctbs_median - nj_median
            
            comparison_data.append({
                'nj_variant': nj_method,
                'nj_pure': nj_median,
                'ctbs_hybrid': ctbs_median,
                'improvement': improvement,
                'improvement_pct': (improvement / nj_median * 100) if nj_median > 0 else 0
            })
    
    # Convert to DataFrame and sort by CTBS hybrid performance
    comp_df = pd.DataFrame(comparison_data).sort_values('ctbs_hybrid', ascending=False).head(top_n)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Side-by-side comparison
    x = np.arange(len(comp_df))
    width = 0.35
    
    bars1 = ax1.barh(x - width/2, comp_df['nj_pure'], width, label='Pure NJ', 
                     color='#4ECDC4', alpha=0.8)
    bars2 = ax1.barh(x + width/2, comp_df['ctbs_hybrid'], width, label='CTBS+NJ', 
                     color='#FF6B6B', alpha=0.8)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(comp_df['nj_variant'], fontsize=9)
    ax1.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
    ax1.set_title(f'Pure NJ vs CTBS+NJ Hybrid', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Right plot: Improvement percentages
    colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in comp_df['improvement_pct']]
    bars3 = ax2.barh(x, comp_df['improvement_pct'], color=colors, alpha=0.8)
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(comp_df['nj_variant'], fontsize=9)
    ax2.set_xlabel('Improvement (%)', fontsize=11)
    ax2.set_title(f'CTBS Improvement over Pure NJ', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Add value labels on improvement bars
    for i, (bar, val) in enumerate(zip(bars3, comp_df['improvement_pct'])):
        ax2.text(val + (1 if val > 0 else -1), i, f'{val:.1f}%', 
                va='center', ha='left' if val > 0 else 'right', fontsize=8)
    
    plt.suptitle(f'Impact of CTBS on Reconstruction Quality (Top {top_n} Variants, {metric.replace("_", " ").title()})',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"NJ vs CTBS+NJ comparison saved to: {output_file}")


def create_category_comparison(df, metric='multiset_f1',
                               output_file='figures/category_comparison.png'):
    """
    Create violin plots comparing method categories (Pure NJ vs CTBS+NJ).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter data
    df_plot = df[df['method_category'].isin(['NJ', 'CTBS+NJ'])].copy()
    
    # Left plot: Full distribution
    sns.violinplot(data=df_plot, x='method_category', y=metric, ax=axes[0],
                  palette={'NJ': '#4ECDC4', 'CTBS+NJ': '#FF6B6B'}, inner='box')
    axes[0].set_title('Overall Distribution by Category', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Method Category', fontsize=11)
    axes[0].set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_axisbelow(True)
    
    # Right plot: Box plot comparison
    sns.boxplot(data=df_plot, x='method_category', y=metric, ax=axes[1],
               palette={'NJ': '#4ECDC4', 'CTBS+NJ': '#FF6B6B'}, width=0.5)
    axes[1].set_title('Statistical Summary by Category', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Method Category', fontsize=11)
    axes[1].set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_axisbelow(True)
    
    # Add sample sizes
    for i, cat in enumerate(['NJ', 'CTBS+NJ']):
        n = len(df_plot[df_plot['method_category'] == cat])
        axes[0].text(i, axes[0].get_ylim()[1], f'n={n}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Pure NJ vs CTBS+NJ Hybrid Methods ({metric.replace("_", " ").title()})',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Category comparison saved to: {output_file}")


def create_all_visualizations(benchmark_csv='results/nj_benchmark_results.csv',
                              summary_csv='results/nj_summary_statistics.csv',
                              tests_csv='results/nj_statistical_tests.csv',
                              roc_csv='results/nj_roc_data.csv',
                              output_dir='figures',
                              use_input_dir=True):
    """
    Generate all paper-ready visualizations.
    
    Parameters
    ----------
    benchmark_csv : str
        Path to benchmark results CSV
    summary_csv : str
        Path to summary statistics CSV
    tests_csv : str
        Path to statistical tests CSV
    roc_csv : str
        Path to ROC data CSV
    output_dir : str
        Output directory for figures
    use_input_dir : bool
        If True, place figures in subdirectory of benchmark_csv directory (default: True)
    """
    print("="*60)
    print("Generating Paper-Ready Visualizations")
    print("="*60)
    
    # If use_input_dir, place figures in same parent directory as benchmark results
    if use_input_dir:
        result_dir = os.path.dirname(benchmark_csv)
        output_dir = os.path.join(result_dir, os.path.basename(output_dir))
        # Also update paths for other CSVs to be in same directory
        summary_csv = os.path.join(result_dir, os.path.basename(summary_csv))
        tests_csv = os.path.join(result_dir, os.path.basename(tests_csv))
        roc_csv = os.path.join(result_dir, os.path.basename(roc_csv))
    
    # Load data
    print("\nLoading data...")
    df_benchmark, df_summary, df_tests, df_roc = load_data(
        benchmark_csv, summary_csv, tests_csv, roc_csv
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    print("\n1. Creating box plot comparison (multiset F1)...")
    create_boxplot_comparison(df_benchmark, metric='multiset_f1',
                             output_file=f'{output_dir}/boxplot_multiset_f1.png')
    
    print("2. Creating box plot comparison (unique F1)...")
    create_boxplot_comparison(df_benchmark, metric='unique_f1',
                             output_file=f'{output_dir}/boxplot_unique_f1.png')
    
    print("3. Creating violin plot (multiset F1)...")
    create_violin_plot(df_benchmark, metric='multiset_f1',
                      output_file=f'{output_dir}/violin_multiset_f1.png')
    
    print("4. Creating ROC curves...")
    create_roc_curves(df_roc, output_file=f'{output_dir}/roc_curves.png')
    
    print("5. Creating p-value heatmap...")
    create_pvalue_heatmap(df_tests, output_file=f'{output_dir}/pvalue_heatmap.png')
    
    print("6. Creating performance scatter plot...")
    create_performance_scatter(df_benchmark, output_file=f'{output_dir}/performance_scatter.png')
    
    print("7. Creating method ranking bar chart...")
    create_method_ranking_bar(df_summary, metric='multiset_f1_mean',
                             output_file=f'{output_dir}/method_ranking.png')
    
    print("8. Creating multi-metric comparison...")
    create_multi_metric_comparison(df_summary, output_file=f'{output_dir}/multi_metric_comparison.png')
    
    print("9. Creating NJ vs CTBS+NJ comparison...")
    create_nj_vs_ctbs_comparison(df_benchmark, metric='multiset_f1',
                                output_file=f'{output_dir}/nj_vs_ctbs_comparison.png')
    
    print("10. Creating category comparison (multiset F1)...")
    create_category_comparison(df_benchmark, metric='multiset_f1',
                              output_file=f'{output_dir}/category_comparison_multiset_f1.png')
    
    print("11. Creating category comparison (unique F1)...")
    create_category_comparison(df_benchmark, metric='unique_f1',
                              output_file=f'{output_dir}/category_comparison_unique_f1.png')
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for NJ benchmark results')
    parser.add_argument('--benchmark-csv', type=str, default='results/nj_benchmark_results.csv',
                       help='Input benchmark CSV file')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures')
    parser.add_argument('--no-use-input-dir', action='store_true',
                       help='Do not place figures in same directory as input')
    
    args = parser.parse_args()
    
    # Derive other CSV paths from benchmark CSV location
    result_dir = os.path.dirname(args.benchmark_csv)
    summary_csv = os.path.join(result_dir, 'nj_summary_statistics.csv')
    tests_csv = os.path.join(result_dir, 'nj_statistical_tests.csv')
    roc_csv = os.path.join(result_dir, 'nj_roc_data.csv')
    
    create_all_visualizations(
        benchmark_csv=args.benchmark_csv,
        summary_csv=summary_csv,
        tests_csv=tests_csv,
        roc_csv=roc_csv,
        output_dir=args.output_dir,
        use_input_dir=not args.no_use_input_dir
    )
