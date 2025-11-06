"""
NJ Variants Benchmarking Script

This script evaluates all Neighbor Joining (NJ) reconstruction variants
against the CTBS reconstruction method across multiple seeds.

Output: CSV file with detailed metrics for each method and seed.
"""

import pandas as pd
import sys
import os
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ctbs import run_single_test
from evaluator_full import evaluate_4
from reconstructor import (
    neighbor_joining_standard,
    neighbor_joining_full,
    neighbor_joining_full_cps,
    neighbor_joining_hybrid,
    neighbor_joining_hybrid_inverse_centrality,
    neighbor_joining_adaptive_centrality,
    neighbor_joining_adaptive_centrality_nonlinear,
    neighbor_joining_adaptive_centrality_reversed,
    neighbor_joining_hybrid_opt,
    neighbor_joining_hybrid_opt_adaptive,
    neighbor_joining_hybrid_opt_v2,
    neighbor_joining_hybrid_opt_refined,
    neighbor_joining_hybrid_anticentral_opt,
    neighbor_joining_hybrid_anticentral_adaptive_v2,
    neighbor_joining_hybrid_anticentral_adaptive_v3,
)


def get_all_nj_algorithms():
    """
    Returns a list of tuples (algorithm_name, algorithm_function) for all NJ variants.
    """
    return [
        ("standard", neighbor_joining_standard),
        ("full", neighbor_joining_full),
        ("full_cps", neighbor_joining_full_cps),
        ("hybrid", neighbor_joining_hybrid),
        ("hybrid_inverse_centrality", neighbor_joining_hybrid_inverse_centrality),
        ("adaptive_centrality", neighbor_joining_adaptive_centrality),
        ("adaptive_centrality_nonlinear", neighbor_joining_adaptive_centrality_nonlinear),
        ("adaptive_centrality_reversed", neighbor_joining_adaptive_centrality_reversed),
        ("hybrid_opt", neighbor_joining_hybrid_opt),
        ("hybrid_opt_adaptive", neighbor_joining_hybrid_opt_adaptive),
        ("hybrid_opt_v2", neighbor_joining_hybrid_opt_v2),
        ("hybrid_opt_refined", neighbor_joining_hybrid_opt_refined),
        ("hybrid_anticentral_opt", neighbor_joining_hybrid_anticentral_opt),
        ("hybrid_anticentral_adaptive_v2", neighbor_joining_hybrid_anticentral_adaptive_v2),
        ("hybrid_anticentral_adaptive_v3", neighbor_joining_hybrid_anticentral_adaptive_v3),
    ]


def load_seeds(csv_path="data/f1results.csv"):
    """
    Loads unique seeds from the f1results.csv file.
    """
    df = pd.read_csv(csv_path, delimiter="\t")
    all_seeds = df["seed"].unique().tolist()
    return all_seeds


def evaluate_method(method_name, reconstruction_algorithm, seed, config, bedfile,
                    biopsy_size_scalable, biopsy_generations, r_dist):
    """
    Evaluates a single reconstruction method on a given seed.
    
    Returns a list with two dictionaries:
    1. Pure NJ variant results (from nj_tree)
    2. CTBS+NJ hybrid results (from reconstructed_tree, which uses NJ for root finding)
    
    The reconstructed_tree is CTBS reconstruction that uses the specified NJ variant
    for finding the main root of the tree.
    """
    try:
        # Run the reconstruction with parallel=True to avoid cnp2cnp dependency
        simulated_tree, reconstructed_tree, nj_tree = run_single_test(
            seed=seed,
            config=config,
            bedfile=bedfile,
            biopsy_size_scalable=biopsy_size_scalable,
            biopsy_generations=biopsy_generations,
            r_dist=r_dist,
            write_newick=False,
            reconstruction_algorithm=reconstruction_algorithm,
            parallel=True,  # Use parallel distance computation to avoid cnp2cnp tool
        )
        
        # Evaluate CTBS+NJ hybrid (reconstructed_tree uses NJ variant for root finding)
        rec_metrics = evaluate_4(simulated_tree, reconstructed_tree)
        
        # Evaluate pure NJ variant
        nj_metrics = evaluate_4(simulated_tree, nj_tree)
        
        # Result 1: Pure NJ variant
        nj_result = {
            'seed': seed,
            'method': method_name,
            'method_type': 'nj_pure',
            'method_category': 'NJ',
            
            # Multiset metrics
            'multiset_precision': nj_metrics['ancestors_multiset']['precision'],
            'multiset_recall': nj_metrics['ancestors_multiset']['recall'],
            'multiset_f1': nj_metrics['ancestors_multiset']['F1'],
            'multiset_iou': nj_metrics['ancestors_multiset'].get('IoU', None),
            'multiset_tp': nj_metrics['ancestors_multiset'].get('TP', None),
            'multiset_fp': nj_metrics['ancestors_multiset'].get('FP', None),
            'multiset_fn': nj_metrics['ancestors_multiset'].get('FN', None),
            
            # Unique metrics
            'unique_precision': nj_metrics['ancestors_unique']['precision'],
            'unique_recall': nj_metrics['ancestors_unique']['recall'],
            'unique_f1': nj_metrics['ancestors_unique']['F1'],
            'unique_iou': nj_metrics['ancestors_unique'].get('IoU', None),
            'unique_tp': nj_metrics['ancestors_unique'].get('TP', None),
            'unique_fp': nj_metrics['ancestors_unique'].get('FP', None),
            'unique_fn': nj_metrics['ancestors_unique'].get('FN', None),
            
            'status': 'success'
        }
        
        # Result 2: CTBS+NJ hybrid (CTBS with NJ variant for root finding)
        ctbs_hybrid_result = {
            'seed': seed,
            'method': f'ctbs_{method_name}',
            'nj_variant': method_name,
            'method_type': 'ctbs_hybrid',
            'method_category': 'CTBS+NJ',
            
            # Multiset metrics
            'multiset_precision': rec_metrics['ancestors_multiset']['precision'],
            'multiset_recall': rec_metrics['ancestors_multiset']['recall'],
            'multiset_f1': rec_metrics['ancestors_multiset']['F1'],
            'multiset_iou': rec_metrics['ancestors_multiset'].get('IoU', None),
            'multiset_tp': rec_metrics['ancestors_multiset'].get('TP', None),
            'multiset_fp': rec_metrics['ancestors_multiset'].get('FP', None),
            'multiset_fn': rec_metrics['ancestors_multiset'].get('FN', None),
            
            # Unique metrics
            'unique_precision': rec_metrics['ancestors_unique']['precision'],
            'unique_recall': rec_metrics['ancestors_unique']['recall'],
            'unique_f1': rec_metrics['ancestors_unique']['F1'],
            'unique_iou': rec_metrics['ancestors_unique'].get('IoU', None),
            'unique_tp': rec_metrics['ancestors_unique'].get('TP', None),
            'unique_fp': rec_metrics['ancestors_unique'].get('FP', None),
            'unique_fn': rec_metrics['ancestors_unique'].get('FN', None),
            
            'status': 'success'
        }
        
        return [nj_result, ctbs_hybrid_result]
        
    except Exception as e:
        print(f"Error for method {method_name}, seed {seed}: {e}")
        return [{
            'seed': seed,
            'method': method_name,
            'method_type': 'nj_pure',
            'method_category': 'NJ',
            'status': 'failed',
            'error': str(e)
        }, {
            'seed': seed,
            'method': f'ctbs_{method_name}',
            'nj_variant': method_name,
            'method_type': 'ctbs_hybrid',
            'method_category': 'CTBS+NJ',
            'status': 'failed',
            'error': str(e)
        }]


def evaluate_single_task(task_tuple):
    """
    Evaluate a single (algorithm, seed) combination.
    
    Parameters
    ----------
    task_tuple : tuple
        (algo_name, algo_func, seed, config, bedfile, biopsy_size_scalable, 
         biopsy_generations, r_dist)
    
    Returns
    -------
    list
        List with two dictionaries: [nj_pure_result, ctbs_hybrid_result]
    """
    (algo_name, algo_func, seed, config, bedfile, 
     biopsy_size_scalable, biopsy_generations, r_dist) = task_tuple
    
    return evaluate_method(
        method_name=algo_name,
        reconstruction_algorithm=algo_func,
        seed=seed,
        config=config,
        bedfile=bedfile,
        biopsy_size_scalable=biopsy_size_scalable,
        biopsy_generations=biopsy_generations,
        r_dist=r_dist
    )


def run_benchmark(output_csv="results/nj_benchmark_results.csv",
                 config="data/config_for_pic.json",
                 bedfile="data/pic.csv",
                 biopsy_size_scalable=0.5,
                 biopsy_generations=[4, 6, 8],
                 r_dist=4,
                 max_seeds=None,
                 parallel=False,
                 max_workers=None,
                 timestamp_dirs=True):
    """
    Main benchmarking function that evaluates all NJ variants across all seeds.
    
    Parameters
    ----------
    output_csv : str
        Path to output CSV file
    config : str
        Configuration file for simulation
    bedfile : str
        Bedfile for simulation
    biopsy_size_scalable : float
        Fraction of cells to sample in biopsy
    biopsy_generations : list
        List of generation levels for biopsy
    r_dist : int
        Distance threshold for reconstruction
    max_seeds : int or None
        Maximum number of seeds to test (None = all seeds)
    parallel : bool
        If True, run tasks in parallel using ProcessPoolExecutor (default: False)
    max_workers : int or None
        Maximum number of parallel workers (default: 60% of CPUs)
    timestamp_dirs : bool
        If True, add timestamp to output directories (default: True)
    """
    
    # Add timestamp to output paths if requested
    if timestamp_dirs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Update output_csv path to include timestamp
        csv_dir = os.path.dirname(output_csv)
        csv_filename = os.path.basename(output_csv)
        output_csv = os.path.join(csv_dir, timestamp, csv_filename)
    
    # Load seeds
    all_seeds = load_seeds()
    if max_seeds is not None:
        all_seeds = all_seeds[:max_seeds]
    
    print(f"Loaded {len(all_seeds)} seeds for benchmarking")
    
    # Get all NJ algorithms
    algorithms = get_all_nj_algorithms()
    print(f"Testing {len(algorithms)} NJ variants")
    
    # Default to 60% of available cores for efficiency and cost-effectiveness
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        max_workers = max(1, int(cpu_count * 0.6))
        print(f"Using {max_workers} workers (60% of {cpu_count} available cores)")
    
    if parallel:
        # Create a flat list of all (algorithm, seed) tasks
        tasks = []
        for algo_name, algo_func in algorithms:
            for seed in all_seeds:
                tasks.append((
                    algo_name, algo_func, seed, config, bedfile,
                    biopsy_size_scalable, biopsy_generations, r_dist
                ))
        
        total_tasks = len(tasks)
        print(f"Processing {total_tasks} tasks ({len(algorithms)} algorithms × {len(all_seeds)} seeds)")
        print(f"Running in parallel with {max_workers} workers")
        
        # Process all tasks in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to show progress for all tasks
            futures = [executor.submit(evaluate_single_task, task) for task in tasks]
            
            for future in tqdm(futures, desc="Processing tasks", total=total_tasks):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"\nError processing task: {e}")
    else:
        # Sequential processing
        all_results = []
        
        # Iterate through each algorithm
        for algo_name, algo_func in algorithms:
            print(f"\n{'='*60}")
            print(f"Evaluating: {algo_name}")
            print(f"{'='*60}")
            
            # Test on all seeds with progress bar
            for seed in tqdm(all_seeds, desc=f"Testing {algo_name}"):
                results = evaluate_method(
                    method_name=algo_name,
                    reconstruction_algorithm=algo_func,
                    seed=seed,
                    config=config,
                    bedfile=bedfile,
                    biopsy_size_scalable=biopsy_size_scalable,
                    biopsy_generations=biopsy_generations,
                    r_dist=r_dist
                )
                
                # Add both NJ and CTBS+NJ hybrid results
                all_results.extend(results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by method and seed for better organization
    if 'method' in df.columns and 'seed' in df.columns:
        df = df.sort_values(['method', 'seed']).reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Benchmark complete!")
    print(f"Results saved to: {output_csv}")
    print(f"Total rows: {len(df)}")
    print(f"Successful runs: {df[df['status'] == 'success'].shape[0]}")
    print(f"Failed runs: {df[df['status'] == 'failed'].shape[0]}")
    print(f"{'='*60}\n")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark NJ variants')
    parser.add_argument('--output', type=str, default='results/nj_benchmark_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--max-seeds', type=int, default=None,
                       help='Maximum number of seeds to test (default: all)')
    parser.add_argument('--config', type=str, default='data/config_for_pic.json',
                       help='Configuration file')
    parser.add_argument('--bedfile', type=str, default='data/pic.csv',
                       help='Bedfile for simulation')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tasks in parallel (processes all algorithm×seed combinations concurrently)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: 60%% of available CPUs)')
    parser.add_argument('--no-timestamp', action='store_true',
                       help='Disable timestamp in output directories')
    
    args = parser.parse_args()
    
    # Run benchmark
    df = run_benchmark(
        output_csv=args.output,
        config=args.config,
        bedfile=args.bedfile,
        max_seeds=args.max_seeds,
        parallel=args.parallel,
        max_workers=args.max_workers,
        timestamp_dirs=not args.no_timestamp
    )
