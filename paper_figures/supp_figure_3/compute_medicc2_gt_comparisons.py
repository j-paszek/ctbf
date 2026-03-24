#!/usr/bin/env python3
"""
Compare MEDICC2 distance matrices with ground truth matrices.

This script:
1. Loads MEDICC2 and ground truth distance matrices
2. Compares them using the same metrics as cnp2cnp/naive comparisons
3. Updates metadata.csv with comparison metrics (prefix: medicc2_vs_gt_)

Usage:
    python compute_medicc2_gt_comparisons.py [--results-dir RESULTS_DIR] [--update-metadata]
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))

from simulation_utils import compare_matrices_programmatically


def _save_metadata_incremental(df, comparisons, metadata_path):
    """Helper function to incrementally save metadata."""
    if not comparisons:
        return
    
    comparisons_df = pd.DataFrame(comparisons)
    df_updated = df.copy()
    
    # For each comparison column, update or add it
    for col in comparisons_df.columns:
        if col == 'run_id':
            continue
        
        # Create a mapping from run_id to value
        value_map = dict(zip(comparisons_df['run_id'], comparisons_df[col]))
        
        if col in df_updated.columns:
            # Update existing column where we have new values
            mask = df_updated['run_id'].isin(value_map.keys())
            df_updated.loc[mask, col] = df_updated.loc[mask, 'run_id'].map(value_map)
        else:
            # Add new column
            df_updated[col] = df_updated['run_id'].map(value_map)
    
    # Save to temporary file first, then rename (atomic operation)
    temp_path = metadata_path.with_suffix('.tmp')
    df_updated.to_csv(temp_path, index=False)
    temp_path.replace(metadata_path)


def process_all_trees(results_dir, update_metadata=True, start_from_run_id=None, skip_existing=True, recompute_comparisons=False, max_workers=None):
    """
    Process all runs and compare MEDICC2 matrices with ground truth.
    
    Parameters
    ----------
    results_dir : str or Path
        Directory containing matrix files and metadata.csv
    update_metadata : bool
        If True, update metadata.csv with MEDICC2 vs GT comparison metrics
    start_from_run_id : int, optional
        Start processing from this run_id (skip all previous runs)
    skip_existing : bool
        If True, skip runs where comparison already exists in metadata
    recompute_comparisons : bool
        If True, recompute comparison metrics even if they exist
    max_workers : int, optional
        Maximum number of parallel workers (not used in this version, kept for API compatibility)
    """
    results_dir = Path(results_dir)
    
    # Load metadata
    metadata_path = results_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {results_dir}")
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} simulation runs from {metadata_path}")
    
    # Process each run
    medicc2_comparisons = []
    skipped_count = 0
    processed_count = 0
    failed_count = 0
    
    # Create progress bar
    pbar = tqdm(total=len(df), desc="Processing MEDICC2 vs GT comparisons", unit="run")
    
    for idx, row in df.iterrows():
        run_id = row.get('run_id', idx + 1)
        
        # Skip if start_from_run_id is specified and we haven't reached it yet
        if start_from_run_id is not None and run_id < start_from_run_id:
            skipped_count += 1
            pbar.update(1)
            pbar.set_postfix({'skipped': skipped_count, 'processed': processed_count, 'failed': failed_count})
            continue
        
        # Construct file paths (check new organized structure first, fall back to old structure)
        matrix_medicc2_dir = results_dir / "matrix_medicc2"
        matrix_gt_dir = results_dir / "matrix_gt"
        
        if matrix_medicc2_dir.exists():
            medicc2_file = matrix_medicc2_dir / f"matrix_medicc2_{run_id:03d}.txt"
            gt_file = matrix_gt_dir / f"matrix_gt_{run_id:03d}.txt"
        else:
            # Fall back to old flat structure
            medicc2_file = results_dir / f"matrix_medicc2_{run_id:03d}.txt"
            gt_file = results_dir / f"matrix_gt_{run_id:03d}.txt"
        
        # Check if both files exist
        if not medicc2_file.exists():
            pbar.write(f"  Warning: MEDICC2 matrix not found for run {run_id}: {medicc2_file}")
            pbar.update(1)
            failed_count += 1
            continue
        
        if not gt_file.exists():
            pbar.write(f"  Warning: Ground truth matrix not found for run {run_id}: {gt_file}")
            pbar.update(1)
            failed_count += 1
            continue
        
        # Check if comparison already exists in metadata
        existing_medicc2_gt = None
        if not recompute_comparisons and skip_existing:
            if 'medicc2_vs_gt_pearson_r' in df.columns:
                existing_medicc2_gt = df.loc[df['run_id'] == run_id, 'medicc2_vs_gt_pearson_r'].values
                if len(existing_medicc2_gt) > 0 and pd.notna(existing_medicc2_gt[0]):
                    existing_medicc2_gt = existing_medicc2_gt[0]
                    skipped_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'skipped': skipped_count, 'processed': processed_count, 'failed': failed_count})
                    continue
        
        # Update progress bar description
        pbar.set_description(f"Processing run {run_id:03d}")
        
        try:
            # Compare MEDICC2 vs ground truth
            comparison_metrics = {}
            
            try:
                medicc2_vs_gt = compare_matrices_programmatically(
                    str(medicc2_file), str(gt_file),
                    permutations=99, fast_mode=True
                )
                # Prefix with 'medicc2_vs_gt_'
                for k, v in medicc2_vs_gt.items():
                    comparison_metrics[f'medicc2_vs_gt_{k}'] = v
                
                comparison_metrics['run_id'] = run_id
                medicc2_comparisons.append(comparison_metrics)
                processed_count += 1
                
                pbar.write(f"  Run {run_id:03d}: Compared MEDICC2 vs ground truth")
                
                # Incrementally save metadata every 10 comparisons to prevent data loss
                if update_metadata and len(medicc2_comparisons) % 10 == 0:
                    try:
                        _save_metadata_incremental(df, medicc2_comparisons, metadata_path)
                        # Reload df to get updated columns
                        df = pd.read_csv(metadata_path)
                    except Exception as e:
                        pbar.write(f"  Warning: Failed to save incremental metadata: {e}")
                
            except Exception as e:
                pbar.write(f"  Run {run_id:03d}: Warning - MEDICC2 vs ground truth comparison failed: {e}")
                failed_count += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'comparisons': len(medicc2_comparisons),
                'skipped': skipped_count,
                'failed': failed_count
            })
            
        except Exception as e:
            pbar.write(f"  Error processing run {run_id:03d}: {e}")
            import traceback
            traceback.print_exc()
            pbar.update(1)
            failed_count += 1
            continue
    
    # Close progress bar
    pbar.close()
    
    # Update metadata if requested
    if update_metadata and medicc2_comparisons:
        print(f"\nUpdating metadata.csv with MEDICC2 vs GT comparisons...")
        print(f"  Merging {len(medicc2_comparisons)} comparison records...")
        
        # Create DataFrame from comparisons
        comparisons_df = pd.DataFrame(medicc2_comparisons)
        
        # Simple merge approach: merge on run_id and update columns
        if 'run_id' not in comparisons_df.columns:
            print("  Warning: run_id not found in comparisons_df")
            return
        
        # Merge with existing metadata
        df_updated = df.copy()
        
        # For each comparison column, update or add it
        for col in comparisons_df.columns:
            if col == 'run_id':
                continue
            
            # Create a mapping from run_id to value
            value_map = dict(zip(comparisons_df['run_id'], comparisons_df[col]))
            
            if col in df_updated.columns:
                # Update existing column where we have new values
                mask = df_updated['run_id'].isin(value_map.keys())
                df_updated.loc[mask, col] = df_updated.loc[mask, 'run_id'].map(value_map)
            else:
                # Add new column
                df_updated[col] = df_updated['run_id'].map(value_map)
        
        # Save updated metadata
        print(f"  Saving updated metadata to: {metadata_path}")
        df_updated.to_csv(metadata_path, index=False)
        print(f"  ✓ Updated metadata saved successfully!")
        
        # Print summary statistics
        print("\n" + "="*70)
        print("Summary: MEDICC2 vs Ground Truth Comparison")
        print("="*70)
        
        if 'medicc2_vs_gt_pearson_r' in df_updated.columns:
            medicc2_corr = df_updated['medicc2_vs_gt_pearson_r'].mean()
            print(f"MEDICC2 vs Ground Truth - Mean Pearson r: {medicc2_corr:.4f}")
        
        # Compare with cnp2cnp and naive if available
        if 'c2c_vs_gt_pearson_r' in df_updated.columns:
            c2c_corr = df_updated['c2c_vs_gt_pearson_r'].mean()
            print(f"cnp2cnp vs Ground Truth - Mean Pearson r: {c2c_corr:.4f}")
        
        if 'naive_vs_gt_pearson_r' in df_updated.columns:
            naive_corr = df_updated['naive_vs_gt_pearson_r'].mean()
            print(f"Naive vs Ground Truth - Mean Pearson r: {naive_corr:.4f}")
        
        # Count how many times each method is better
        if all(col in df_updated.columns for col in ['medicc2_vs_gt_pearson_r', 'c2c_vs_gt_pearson_r', 'naive_vs_gt_pearson_r']):
            valid_mask = (
                df_updated['medicc2_vs_gt_pearson_r'].notna() &
                df_updated['c2c_vs_gt_pearson_r'].notna() &
                df_updated['naive_vs_gt_pearson_r'].notna()
            )
            if valid_mask.sum() > 0:
                df_valid = df_updated[valid_mask]
                medicc2_better_c2c = (df_valid['medicc2_vs_gt_pearson_r'] > df_valid['c2c_vs_gt_pearson_r']).sum()
                medicc2_better_naive = (df_valid['medicc2_vs_gt_pearson_r'] > df_valid['naive_vs_gt_pearson_r']).sum()
                c2c_better_medicc2 = (df_valid['c2c_vs_gt_pearson_r'] > df_valid['medicc2_vs_gt_pearson_r']).sum()
                total = len(df_valid)
                print(f"\nMEDICC2 better than cnp2cnp: {medicc2_better_c2c}/{total} ({100*medicc2_better_c2c/total:.1f}%)")
                print(f"MEDICC2 better than Naive: {medicc2_better_naive}/{total} ({100*medicc2_better_naive/total:.1f}%)")
                print(f"cnp2cnp better than MEDICC2: {c2c_better_medicc2}/{total} ({100*c2c_better_medicc2/total:.1f}%)")
        
        print("="*70)
    
    print(f"\n{'='*70}")
    print("MEDICC2 vs Ground Truth comparison complete!")
    print(f"Processed {processed_count} comparisons")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} runs (already exist or before start_from_run_id)")
    if failed_count > 0:
        print(f"Failed {failed_count} runs (missing files or comparison errors)")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare MEDICC2 distance matrices with ground truth matrices"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='sim_grid_results',
        help='Directory containing matrix files and metadata.csv (default: sim_grid_results)'
    )
    parser.add_argument(
        '--update-metadata',
        action='store_true',
        help='Update metadata.csv with MEDICC2 vs GT comparison metrics'
    )
    parser.add_argument(
        '--start-from-run-id',
        type=int,
        default=None,
        help='Start processing from this run_id (skip all previous runs). Useful for resuming interrupted computations.'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Recompute comparisons even if they already exist in metadata (default: skip existing comparisons)'
    )
    parser.add_argument(
        '--recompute-comparisons',
        action='store_true',
        help='Recompute comparison metrics even if they already exist in metadata (default: skip existing comparisons)'
    )
    
    args = parser.parse_args()
    
    process_all_trees(
        args.results_dir, 
        update_metadata=args.update_metadata,
        start_from_run_id=args.start_from_run_id,
        skip_existing=not args.no_skip_existing,
        recompute_comparisons=args.recompute_comparisons
    )


if __name__ == '__main__':
    main()
