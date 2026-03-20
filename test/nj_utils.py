"""
Quick Utilities for NJ Benchmark Analysis

Convenient functions for quick data exploration and custom analyses.
"""

import pandas as pd
import numpy as np


def quick_summary(csv_path='results/nj_benchmark_results.csv'):
    """
    Print a quick summary of benchmark results.
    """
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success']
    
    print("="*80)
    print("QUICK SUMMARY")
    print("="*80)
    
    print(f"\nTotal runs: {len(df)}")
    print(f"Methods tested: {df['method'].nunique()}")
    print(f"Seeds tested: {df['seed'].nunique()}")
    
    print("\n" + "-"*80)
    print("TOP 10 METHODS BY MULTISET F1 (MEAN)")
    print("-"*80)
    
    top10 = df.groupby('method')['multiset_f1'].mean().nlargest(10)
    for i, (method, score) in enumerate(top10.items(), 1):
        print(f"{i:2d}. {method:35s} | F1 = {score:.4f}")
    
    print("\n" + "-"*80)
    print("TOP 10 METHODS BY UNIQUE F1 (MEAN)")
    print("-"*80)
    
    top10_unique = df.groupby('method')['unique_f1'].mean().nlargest(10)
    for i, (method, score) in enumerate(top10_unique.items(), 1):
        print(f"{i:2d}. {method:35s} | F1 = {score:.4f}")
    
    # CTBS performance
    if 'ctbs' in df['method'].values:
        ctbs_df = df[df['method'] == 'ctbs']
        print("\n" + "-"*80)
        print("CTBS PERFORMANCE")
        print("-"*80)
        print(f"Multiset F1:  Mean = {ctbs_df['multiset_f1'].mean():.4f}, "
              f"Median = {ctbs_df['multiset_f1'].median():.4f}, "
              f"Std = {ctbs_df['multiset_f1'].std():.4f}")
        print(f"Unique F1:    Mean = {ctbs_df['unique_f1'].mean():.4f}, "
              f"Median = {ctbs_df['unique_f1'].median():.4f}, "
              f"Std = {ctbs_df['unique_f1'].std():.4f}")
    
    print("\n" + "="*80 + "\n")


def compare_methods(method1, method2, csv_path='results/nj_benchmark_results.csv'):
    """
    Compare two methods head-to-head.
    """
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success']
    
    df1 = df[df['method'] == method1].set_index('seed')
    df2 = df[df['method'] == method2].set_index('seed')
    
    # Find common seeds
    common_seeds = df1.index.intersection(df2.index)
    
    print("="*80)
    print(f"HEAD-TO-HEAD COMPARISON: {method1} vs {method2}")
    print("="*80)
    print(f"Common seeds: {len(common_seeds)}\n")
    
    metrics = ['multiset_precision', 'multiset_recall', 'multiset_f1',
               'unique_precision', 'unique_recall', 'unique_f1']
    
    for metric in metrics:
        if metric in df1.columns and metric in df2.columns:
            v1 = df1.loc[common_seeds, metric].values
            v2 = df2.loc[common_seeds, metric].values
            
            # Remove NaN pairs
            valid_mask = ~(np.isnan(v1) | np.isnan(v2))
            v1 = v1[valid_mask]
            v2 = v2[valid_mask]
            
            wins = np.sum(v1 > v2)
            ties = np.sum(v1 == v2)
            losses = np.sum(v1 < v2)
            
            mean_diff = np.mean(v1 - v2)
            
            print(f"{metric:20s} | "
                  f"Mean1: {np.mean(v1):.4f} | "
                  f"Mean2: {np.mean(v2):.4f} | "
                  f"Diff: {mean_diff:+.4f} | "
                  f"W/T/L: {wins}/{ties}/{losses}")
    
    print("\n" + "="*80 + "\n")


def find_best_method(metric='multiset_f1', csv_path='results/nj_benchmark_results.csv'):
    """
    Find the best performing method for a given metric.
    """
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success']
    
    # Aggregate by method
    agg = df.groupby('method')[metric].agg(['mean', 'median', 'std', 'min', 'max'])
    agg = agg.sort_values('mean', ascending=False)
    
    print("="*80)
    print(f"BEST METHODS FOR: {metric}")
    print("="*80)
    print(f"\n{'Rank':<5} {'Method':<35} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-"*80)
    
    for i, (method, row) in enumerate(agg.head(15).iterrows(), 1):
        print(f"{i:<5} {method:<35} "
              f"{row['mean']:<8.4f} {row['median']:<8.4f} {row['std']:<8.4f} "
              f"{row['min']:<8.4f} {row['max']:<8.4f}")
    
    print("\n" + "="*80 + "\n")
    
    return agg


def method_consistency(csv_path='results/nj_benchmark_results.csv', top_n=10):
    """
    Analyze consistency (low variance) of top methods.
    """
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success']
    
    # Compute coefficient of variation (CV = std/mean)
    stats = df.groupby('method')['multiset_f1'].agg(['mean', 'std'])
    stats['cv'] = stats['std'] / stats['mean']
    
    # Sort by mean, then by CV
    stats = stats.sort_values(['mean', 'cv'], ascending=[False, True])
    
    print("="*80)
    print("METHOD CONSISTENCY ANALYSIS (Multiset F1)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Method':<35} {'Mean':<10} {'Std':<10} {'CV':<10}")
    print("-"*80)
    
    for i, (method, row) in enumerate(stats.head(top_n).iterrows(), 1):
        print(f"{i:<5} {method:<35} "
              f"{row['mean']:<10.4f} {row['std']:<10.4f} {row['cv']:<10.4f}")
    
    print("\n" + "="*80)
    print("Note: Lower CV (Coefficient of Variation) = More consistent")
    print("="*80 + "\n")
    
    return stats


def seed_analysis(seed, csv_path='results/nj_benchmark_results.csv'):
    """
    Analyze all methods on a specific seed.
    """
    df = pd.read_csv(csv_path)
    df = df[(df['status'] == 'success') & (df['seed'] == seed)]
    
    if len(df) == 0:
        print(f"No data found for seed {seed}")
        return
    
    print("="*80)
    print(f"SEED ANALYSIS: {seed}")
    print("="*80)
    
    df = df.sort_values('multiset_f1', ascending=False)
    
    print(f"\n{'Rank':<5} {'Method':<35} {'M-F1':<8} {'M-Prec':<8} {'U-F1':<8} {'U-Prec':<8}")
    print("-"*80)
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{i:<5} {row['method']:<35} "
              f"{row['multiset_f1']:<8.4f} {row['multiset_precision']:<8.4f} "
              f"{row['unique_f1']:<8.4f} {row['unique_precision']:<8.4f}")
    
    print("\n" + "="*80 + "\n")


def export_top_methods_table(n=10, output_file='results/top_methods_table.csv',
                             csv_path='results/nj_benchmark_results.csv'):
    """
    Export a table of top N methods with all key metrics.
    """
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success']
    
    metrics = ['multiset_precision', 'multiset_recall', 'multiset_f1',
               'unique_precision', 'unique_recall', 'unique_f1']
    
    summary = df.groupby('method')[metrics].agg(['mean', 'std', 'median'])
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Sort by multiset F1 mean
    summary = summary.sort_values('multiset_f1_mean', ascending=False).head(n)
    
    # Save
    summary.to_csv(output_file)
    print(f"Top {n} methods table saved to: {output_file}")
    
    return summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'summary':
            quick_summary()
        
        elif command == 'compare' and len(sys.argv) >= 4:
            method1 = sys.argv[2]
            method2 = sys.argv[3]
            compare_methods(method1, method2)
        
        elif command == 'best':
            metric = sys.argv[2] if len(sys.argv) > 2 else 'multiset_f1'
            find_best_method(metric)
        
        elif command == 'consistency':
            method_consistency()
        
        elif command == 'seed' and len(sys.argv) >= 3:
            seed = int(sys.argv[2])
            seed_analysis(seed)
        
        elif command == 'export':
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            export_top_methods_table(n)
        
        else:
            print("Unknown command or missing arguments")
            print("\nUsage:")
            print("  python nj_utils.py summary")
            print("  python nj_utils.py compare <method1> <method2>")
            print("  python nj_utils.py best [metric]")
            print("  python nj_utils.py consistency")
            print("  python nj_utils.py seed <seed_number>")
            print("  python nj_utils.py export [top_n]")
    
    else:
        # Default: run quick summary
        quick_summary()
