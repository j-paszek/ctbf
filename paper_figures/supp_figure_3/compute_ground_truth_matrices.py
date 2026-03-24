#!/usr/bin/env python3
"""
Compute ground truth distance matrices from existing tree CSV files.

This script:
1. Loads trees from CSV files
2. Computes ground truth distance matrices using the tree structure (event counts)
3. Compares cnp2cnp and naive distance matrices against ground truth
4. Updates metadata.csv with comparison metrics

Usage:
    python compute_ground_truth_matrices.py [--results-dir RESULTS_DIR] [--update-metadata]
"""
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import csv
import itertools
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

from dm_compare import parse_distance_file
from simulation_utils import compare_matrices_programmatically


def load_tree_from_csv(csv_path):
    """
    Load a tree from CSV format.
    
    CSV format:
    - type, node_id, cell_id, generation, genome, parent_id, child_id, events
    - type='node' for nodes, type='edge' for edges
    
    Returns
    -------
    tree : nx.DiGraph
        Directed tree with nodes and edges
    """
    tree = nx.DiGraph()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type'] == 'node':
                node_id = int(row['node_id'])
                cell_id = row['cell_id'] if row['cell_id'] else node_id
                generation = int(row['generation']) if row['generation'] else 0
                genome_str = row['genome']
                genome = [int(x) for x in genome_str.split(',')] if genome_str else []
                
                tree.add_node(node_id, 
                            cell_id=cell_id,
                            generation=generation,
                            genome=genome)
            
            elif row['type'] == 'edge':
                parent_id = int(row['parent_id'])
                child_id = int(row['child_id'])
                events = row['events'] if row['events'] else ''
                
                tree.add_edge(parent_id, child_id, events=events)
    
    return tree


def compute_ground_truth_distance_matrix(tree, node_list=None):
    """
    Compute ground truth distance matrix from tree structure.
    
    Distance is the number of evolutionary events along the path between nodes.
    This replicates the logic from simulator.py's to_distance_matrix method.
    
    Parameters
    ----------
    tree : nx.DiGraph
        The tree structure
    node_list : list, optional
        If provided, only include these node IDs in the distance matrix.
    
    Returns
    -------
    cell_ids : list
        List of cell_id labels
    dist_matrix : np.ndarray
        Distance matrix (n x n)
    """
    undirected_tree = tree.to_undirected()
    
    # Filter nodes
    if node_list is not None:
        nodes = [n for n in node_list if n in tree]
    else:
        nodes = list(tree.nodes())
    
    n = len(nodes)
    
    # Get cell_ids for labeling
    cell_ids = [str(tree.nodes[node].get("cell_id", node)) for node in nodes]
    
    # Precompute all pairwise distances
    dist_matrix = np.zeros((n, n), dtype=int)
    for i, j in itertools.combinations(range(n), 2):
        src, tgt = nodes[i], nodes[j]
        try:
            path = nx.shortest_path(undirected_tree, source=src, target=tgt)
            
            total_events = 0
            for u, v in zip(path[:-1], path[1:]):
                edge_data = tree.get_edge_data(u, v) or tree.get_edge_data(v, u)
                if edge_data is not None and edge_data.get("events"):
                    events_str = edge_data["events"]
                    if events_str:
                        # Count events (semicolon-separated)
                        total_events += len([e for e in events_str.split(";") if e.strip()])
            
            dist_matrix[i, j] = total_events
            dist_matrix[j, i] = total_events
        except nx.NetworkXNoPath:
            # Nodes are not connected (shouldn't happen in a tree, but handle gracefully)
            dist_matrix[i, j] = -1
            dist_matrix[j, i] = -1
    
    return cell_ids, dist_matrix


def write_distance_matrix_phylip(cell_ids, dist_matrix, output_path):
    """
    Write distance matrix in PHYLIP format.
    
    Parameters
    ----------
    cell_ids : list
        List of cell ID labels
    dist_matrix : np.ndarray
        Distance matrix
    output_path : str or Path
        Output file path
    """
    output_path = Path(output_path)
    n = len(cell_ids)
    
    with open(output_path, 'w') as f:
        f.write(f"{n}\n")  # number of nodes first
        for i, cid in enumerate(cell_ids):
            f.write(f"{str(cid):<10}")
            f.write(" ".join(str(int(dist)) for dist in dist_matrix[i]))
            f.write("\n")


def _save_metadata_incremental(df, ground_truth_comparisons, metadata_path):
    """Helper function to incrementally save metadata."""
    if not ground_truth_comparisons:
        return
    
    comparisons_df = pd.DataFrame(ground_truth_comparisons)
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


def _process_single_tree_gt(args):
    """
    Wrapper function to process a single tree (for parallelization).
    
    Parameters
    ----------
    args : tuple
        (tree_path, gt_file, c2c_file, naive_file, run_id, need_gt_matrix, 
         recompute_comparisons, results_dir)
    
    Returns
    -------
    dict
        Result dictionary with run_id and comparison_metrics (if any)
    """
    (tree_path, gt_file, c2c_file, naive_file, run_id, need_gt_matrix,
     recompute_comparisons, results_dir) = args
    
    result = {'run_id': run_id, 'comparison_metrics': {}, 'error': None}
    
    try:
        # Load tree and compute ground truth if needed
        if need_gt_matrix:
            tree = load_tree_from_csv(tree_path)
            cell_ids, gt_matrix = compute_ground_truth_distance_matrix(tree)
            write_distance_matrix_phylip(cell_ids, gt_matrix, gt_file)
        
        # Compare with cnp2cnp and naive if they exist
        if Path(gt_file).exists():
            if Path(c2c_file).exists():
                try:
                    c2c_vs_gt = compare_matrices_programmatically(
                        str(c2c_file), str(gt_file),
                        permutations=99, fast_mode=True
                    )
                    for k, v in c2c_vs_gt.items():
                        result['comparison_metrics'][f'c2c_vs_gt_{k}'] = v
                except Exception as e:
                    pass
            
            if Path(naive_file).exists():
                try:
                    naive_vs_gt = compare_matrices_programmatically(
                        str(naive_file), str(gt_file),
                        permutations=99, fast_mode=True
                    )
                    for k, v in naive_vs_gt.items():
                        result['comparison_metrics'][f'naive_vs_gt_{k}'] = v
                except Exception as e:
                    pass
            
            if Path(c2c_file).exists() and Path(naive_file).exists():
                try:
                    c2c_vs_naive = compare_matrices_programmatically(
                        str(c2c_file), str(naive_file),
                        permutations=99, fast_mode=True
                    )
                    for k, v in c2c_vs_naive.items():
                        result['comparison_metrics'][f'c2c_vs_naive_{k}'] = v
                except Exception as e:
                    pass
    except Exception as e:
        result['error'] = str(e)
    
    return result


def process_all_trees(results_dir, update_metadata=True, start_from_run_id=None, skip_existing=True, recompute_comparisons=False, max_workers=None):
    """
    Process all tree CSV files and compute ground truth distance matrices.
    
    Parameters
    ----------
    results_dir : str or Path
        Directory containing tree CSV files and metadata.csv
    update_metadata : bool
        If True, update metadata.csv with ground truth comparison metrics
    start_from_run_id : int, optional
        Start processing from this run_id (skip all previous runs)
    skip_existing : bool
        If True, skip runs where ground truth matrix already exists
    recompute_comparisons : bool
        If True, recompute comparison metrics even if they exist
    max_workers : int, optional
        Maximum number of parallel workers (default: None = sequential)
    """
    results_dir = Path(results_dir)
    
    # Load metadata
    metadata_path = results_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {results_dir}")
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} simulation runs from {metadata_path}")
    
    # Find all tree files
    tree_files = sorted(results_dir.glob("tree*.csv"))
    print(f"Found {len(tree_files)} tree files")
    
    # Process each tree
    ground_truth_comparisons = []
    skipped_count = 0
    processed_count = 0
    
    # Create progress bar
    pbar = tqdm(total=len(df), desc="Processing runs", unit="run")
    
    for idx, row in df.iterrows():
        tree_file_path = row.get('tree_file_path', '')
        run_id = row.get('run_id', idx + 1)
        
        # Skip if start_from_run_id is specified and we haven't reached it yet
        if start_from_run_id is not None and run_id < start_from_run_id:
            skipped_count += 1
            pbar.update(1)
            pbar.set_postfix({'skipped': skipped_count, 'processed': processed_count})
            continue
        
        # Construct file paths (check new organized structure first, fall back to old structure)
        if tree_file_path:
            # If tree_file_path already includes directory (e.g., "trees/tree001.csv"), use it as-is
            if '/' in tree_file_path or '\\' in tree_file_path:
                tree_path = results_dir / tree_file_path
            else:
                # Old format: just filename, check trees/ directory first
                trees_dir = results_dir / "trees"
                if trees_dir.exists():
                    tree_path = trees_dir / tree_file_path
                else:
                    tree_path = results_dir / tree_file_path
        else:
            # No tree_file_path specified, try to find it
            trees_dir = results_dir / "trees"
            if trees_dir.exists():
                tree_path = trees_dir / f"tree{run_id:03d}.csv"
            else:
                tree_path = results_dir / f"tree{run_id:03d}.csv"
        
        # Try new organized structure first
        matrix_c2c_dir = results_dir / "matrix_c2c"
        matrix_other_dir = results_dir / "matrix_other"
        matrix_gt_dir = results_dir / "matrix_gt"
        
        if matrix_c2c_dir.exists():
            c2c_file = matrix_c2c_dir / f"matrix_c2c_{run_id:03d}.txt"
            naive_file = matrix_other_dir / f"matrix_other_{run_id:03d}.txt"
            # Ensure matrix_gt directory exists
            matrix_gt_dir.mkdir(exist_ok=True)
            gt_file = matrix_gt_dir / f"matrix_gt_{run_id:03d}.txt"
        else:
            # Fall back to old flat structure
            c2c_file = results_dir / f"matrix_c2c_{run_id:03d}.txt"
            naive_file = results_dir / f"matrix_other_{run_id:03d}.txt"
            gt_file = results_dir / f"matrix_gt_{run_id:03d}.txt"
        
        if not tree_path.exists():
            pbar.write(f"  Warning: Tree file not found: {tree_path}")
            pbar.update(1)
            continue
        
        # Check if we need to compute ground truth matrix
        need_gt_matrix = not (skip_existing and gt_file.exists())
        
        # Update progress bar description
        if need_gt_matrix:
            pbar.set_description(f"Processing run {run_id} (computing GT)")
        else:
            pbar.set_description(f"Processing run {run_id} (comparisons only)")
        
        try:
            # Load tree and compute ground truth if needed
            if need_gt_matrix:
                tree = load_tree_from_csv(tree_path)
                pbar.write(f"  Run {run_id}: Loaded tree with {len(tree.nodes())} nodes, {len(tree.edges())} edges")
                
                # Compute ground truth distance matrix
                cell_ids, gt_matrix = compute_ground_truth_distance_matrix(tree)
                pbar.write(f"  Run {run_id}: Computed ground truth matrix: {gt_matrix.shape}")
                
                # Save ground truth matrix
                write_distance_matrix_phylip(cell_ids, gt_matrix, gt_file)
                pbar.write(f"  Run {run_id}: Saved ground truth matrix")
            else:
                skipped_count += 1
            
            # Compare with cnp2cnp and naive if they exist (always do this if GT exists)
            comparison_metrics = {}
            
            # Check if comparisons already exist in metadata
            existing_c2c_gt = None
            existing_naive_gt = None
            if not recompute_comparisons:
                if 'c2c_vs_gt_pearson_r' in df.columns:
                    existing_c2c_gt = df.loc[df['run_id'] == run_id, 'c2c_vs_gt_pearson_r'].values
                    if len(existing_c2c_gt) > 0 and pd.notna(existing_c2c_gt[0]):
                        existing_c2c_gt = existing_c2c_gt[0]
                if 'naive_vs_gt_pearson_r' in df.columns:
                    existing_naive_gt = df.loc[df['run_id'] == run_id, 'naive_vs_gt_pearson_r'].values
                    if len(existing_naive_gt) > 0 and pd.notna(existing_naive_gt[0]):
                        existing_naive_gt = existing_naive_gt[0]
            
            # Compute comparisons if ground truth exists and comparisons are missing (or recompute requested)
            if gt_file.exists():
                if c2c_file.exists() and (recompute_comparisons or existing_c2c_gt is None or pd.isna(existing_c2c_gt)):
                    # Compare cnp2cnp vs ground truth
                    try:
                        c2c_vs_gt = compare_matrices_programmatically(
                            str(c2c_file), str(gt_file),
                            permutations=99, fast_mode=True
                        )
                        # Prefix with 'c2c_vs_gt_'
                        for k, v in c2c_vs_gt.items():
                            comparison_metrics[f'c2c_vs_gt_{k}'] = v
                        pbar.write(f"  Run {run_id}: Compared cnp2cnp vs ground truth")
                    except Exception as e:
                        pbar.write(f"  Run {run_id}: Warning - cnp2cnp vs ground truth comparison failed: {e}")
                
                if naive_file.exists() and (recompute_comparisons or existing_naive_gt is None or pd.isna(existing_naive_gt)):
                    # Compare naive vs ground truth
                    try:
                        naive_vs_gt = compare_matrices_programmatically(
                            str(naive_file), str(gt_file),
                            permutations=99, fast_mode=True
                        )
                        # Prefix with 'naive_vs_gt_'
                        for k, v in naive_vs_gt.items():
                            comparison_metrics[f'naive_vs_gt_{k}'] = v
                        pbar.write(f"  Run {run_id}: Compared naive vs ground truth")
                    except Exception as e:
                        pbar.write(f"  Run {run_id}: Warning - naive vs ground truth comparison failed: {e}")
                
                # Also compare cnp2cnp vs naive (if not already in metadata)
                if c2c_file.exists() and naive_file.exists():
                    existing_c2c_naive = None
                    if 'c2c_vs_naive_pearson_r' in df.columns:
                        existing_c2c_naive = df.loc[df['run_id'] == run_id, 'c2c_vs_naive_pearson_r'].values
                        if len(existing_c2c_naive) > 0 and pd.notna(existing_c2c_naive[0]):
                            existing_c2c_naive = existing_c2c_naive[0]
                    
                    if recompute_comparisons or existing_c2c_naive is None or pd.isna(existing_c2c_naive):
                        try:
                            c2c_vs_naive = compare_matrices_programmatically(
                                str(c2c_file), str(naive_file),
                                permutations=99, fast_mode=True
                            )
                            # Prefix with 'c2c_vs_naive_'
                            for k, v in c2c_vs_naive.items():
                                comparison_metrics[f'c2c_vs_naive_{k}'] = v
                        except Exception as e:
                            pass  # Silently skip if fails
            
            # Store comparison metrics with run_id for merging (only if we computed something)
            if comparison_metrics:
                comparison_metrics['run_id'] = run_id
                ground_truth_comparisons.append(comparison_metrics)
                if need_gt_matrix:
                    processed_count += 1
                else:
                    # Count as processed for comparisons even if we skipped GT matrix
                    processed_count += 1
                
                # Incrementally save metadata every 10 comparisons to prevent data loss
                if update_metadata and len(ground_truth_comparisons) % 10 == 0:
                    try:
                        _save_metadata_incremental(df, ground_truth_comparisons, metadata_path)
                    except Exception as e:
                        pbar.write(f"  Warning: Failed to save incremental metadata: {e}")
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'comparisons': len(ground_truth_comparisons),
                'skipped': skipped_count
            })
            
        except Exception as e:
            pbar.write(f"  Error processing run {run_id} ({tree_path}): {e}")
            import traceback
            traceback.print_exc()
            pbar.update(1)
            continue
    
    # Close progress bar
    pbar.close()
    
    # Update metadata if requested
    if update_metadata and ground_truth_comparisons:
        print(f"\nUpdating metadata.csv with ground truth comparisons...")
        print(f"  Merging {len(ground_truth_comparisons)} comparison records...")
        
        # Create DataFrame from comparisons
        comparisons_df = pd.DataFrame(ground_truth_comparisons)
        
        # Simple merge approach: merge on run_id and update columns
        # First, ensure run_id is in both DataFrames
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
        print("Summary: Ground Truth Comparison")
        print("="*70)
        
        if 'c2c_vs_gt_pearson_r' in df_updated.columns:
            c2c_corr = df_updated['c2c_vs_gt_pearson_r'].mean()
            print(f"cnp2cnp vs Ground Truth - Mean Pearson r: {c2c_corr:.4f}")
        
        if 'naive_vs_gt_pearson_r' in df_updated.columns:
            naive_corr = df_updated['naive_vs_gt_pearson_r'].mean()
            print(f"Naive vs Ground Truth - Mean Pearson r: {naive_corr:.4f}")
        
        if 'c2c_vs_gt_pearson_r' in df_updated.columns and 'naive_vs_gt_pearson_r' in df_updated.columns:
            better = (df_updated['c2c_vs_gt_pearson_r'] > df_updated['naive_vs_gt_pearson_r']).sum()
            total = df_updated['c2c_vs_gt_pearson_r'].notna().sum()
            print(f"cnp2cnp better than Naive: {better}/{total} ({100*better/total:.1f}%)")
        
        print("="*70)
    
        print(f"\n{'='*70}")
        print("Ground truth matrix computation complete!")
        print(f"Processed {processed_count} new trees")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} trees (already exist or before start_from_run_id)")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute ground truth distance matrices from existing tree CSV files"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='sim_grid_results',
        help='Directory containing tree CSV files and metadata.csv (default: sim_grid_results)'
    )
    parser.add_argument(
        '--update-metadata',
        action='store_true',
        help='Update metadata.csv with ground truth comparison metrics'
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
        help='Recompute ground truth matrices even if they already exist (default: skip existing files)'
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
