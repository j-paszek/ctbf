#!/usr/bin/env python3
"""
Grid simulation script that runs all combinations of parameter values
with multiple repetitions per parameter setting.

Usage:
    python grid_simulation.py [--output-dir OUTPUT_DIR] [--repetitions N] [--config-dict CONFIG_DICT_JSON]
"""
import argparse
import json
import sys
import os
import itertools
import tempfile
import shutil
from pathlib import Path
import csv
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from simulator import CancerCellEvolutionSimulator
from simulation_utils import (
    export_tree_to_csv,
    compute_naive_distance_matrix,
    figure3_cnp2cnp_provenance as regular_cnp2cnp_provenance,
    write_distance_matrix,
)

# Select a cnp2cnp backend. The fallback is still cnp2cnp subprocess mode;
# L1 is computed only as its separately named comparator below.
try:
    from simulation_utils_optimized import (
        distance_matrix_from_biopsy_optimized as distance_matrix_from_biopsy_fast,
        figure3_cnp2cnp_provenance as selected_cnp2cnp_provenance,
    )
    USE_OPTIMIZED = True
except ImportError:
    from ctbs import distance_matrix_from_biopsy as distance_matrix_from_biopsy_fast
    selected_cnp2cnp_provenance = regular_cnp2cnp_provenance
    USE_OPTIMIZED = False


# Default parameter grid file path
DEFAULT_PARAMETER_GRID_FILE = os.path.join(os.path.dirname(__file__), "figure_3_parameter_grid.json")


def load_parameter_grid(grid_file=None):
    """
    Load parameter grid from JSON file.
    
    Parameters
    ----------
    grid_file : str or Path, optional
        Path to JSON file with parameter grid. If None, uses default file.
        
    Returns
    -------
    dict
        Parameter grid dictionary
    """
    if grid_file is None:
        grid_file = DEFAULT_PARAMETER_GRID_FILE
    
    grid_file = Path(grid_file)
    
    if not grid_file.exists():
        raise FileNotFoundError(f"Parameter grid file not found: {grid_file}")
    
    with open(grid_file, 'r') as f:
        param_grid = json.load(f)
    
    return param_grid


def interpret_scenario(config_dict):
    """
    Interpret a parameter configuration to describe the evolutionary scenario.
    
    Parameters
    ----------
    config_dict : dict
        Dictionary with parameter values
        
    Returns
    -------
    str
        Human-readable description of the scenario
    """
    interpretations = []
    
    # Genome complexity
    genome_length = config_dict.get('genome_length', 100)
    if genome_length < 50:
        complexity = "small genome"
    elif genome_length < 100:
        complexity = "medium genome"
    else:
        complexity = "large genome"
    interpretations.append(complexity)
    
    # Evolutionary rate (event probability)
    event_prob = config_dict.get('GENERAL_EVENT_PROB', 0.1)
    if event_prob < 0.05:
        evo_rate = "low evolutionary rate"
    elif event_prob < 0.15:
        evo_rate = "moderate evolutionary rate"
    else:
        evo_rate = "high evolutionary rate"
    interpretations.append(evo_rate)
    
    # Duplication bias (amplification vs loss)
    dup_prob = config_dict.get('GENERAL_DUPLICATION_PROB', 0.5)
    loss_prob = config_dict.get('GENERAL_LOSS_PROB', 0.5)
    
    # Calculate relative bias
    if dup_prob < 0.2:
        bias = "loss-dominant"
    elif dup_prob < 0.4:
        bias = "balanced (slight loss bias)"
    elif dup_prob < 0.6:
        bias = "balanced"
    elif dup_prob < 0.8:
        bias = "balanced (slight amplification bias)"
    else:
        bias = "amplification-dominant"
    interpretations.append(bias)
    
    # Number of generations
    num_gen = config_dict.get('NUMBER_OF_GENERATIONS', 8)
    if num_gen < 6:
        gen_desc = "short evolution"
    elif num_gen < 10:
        gen_desc = "moderate evolution"
    else:
        gen_desc = "long evolution"
    interpretations.append(gen_desc)
    
    # Telomeric regions
    if config_dict.get('MODEL_TELOMERIC_REGIONS', 'False') == 'True' or config_dict.get('MODEL_TELOMERIC_REGIONS', False) is True:
        telomeric = float(config_dict.get('GENERAL_TELOMERIC_INSTABILITY', 0.1))
        if telomeric > 0.15:
            interpretations.append("high telomeric instability")
        elif telomeric >= 0.05:
            interpretations.append("moderate telomeric instability")
        else:
            interpretations.append("low telomeric instability")
    
    # Combine into readable description
    scenario = f"{complexity.capitalize()}, {evo_rate}, {bias}, {gen_desc}"
    if len(interpretations) > 4:
        scenario += f", {interpretations[-1]}"
    
    return scenario


def expand_parameter_grid(param_grid):
    """
    Expand parameter grid into all combinations.
    
    Parameters
    ----------
    param_grid : dict
        Dictionary where keys are parameter names and values are lists of possible values
        
    Returns
    -------
    list
        List of dictionaries, each representing one parameter combination
    """
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    
    combinations = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        combinations.append(config)
    
    return combinations


def create_temp_config(config_dict, temp_dir):
    """
    Create a temporary JSON config file from a dictionary.
    
    Parameters
    ----------
    config_dict : dict
        Configuration dictionary
    temp_dir : Path
        Temporary directory to create config file in
        
    Returns
    -------
    Path
        Path to temporary config file
    """
    config_path = temp_dir / "temp_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    return config_path


def run_single_grid_simulation(config_dict, repetition, run_id, output_dir, base_seed=42, 
                                max_nodes=1000, timeout_seconds=60, max_retries=3, max_cells_for_matrix=150):
    """
    Run a single simulation for the grid.
    
    Parameters
    ----------
    config_dict : dict
        Parameter configuration dictionary
    repetition : int
        Repetition number (0-indexed)
    run_id : int
        Unique run ID (for file naming)
    output_dir : Path
        Output directory
    base_seed : int
        Base seed for random number generation
    max_nodes : int
        Maximum number of nodes allowed in the tree (default: 1000)
    timeout_seconds : int
        Maximum time allowed for a single simulation in seconds (default: 60)
    max_retries : int
        Maximum number of retries with different seeds if tree is too large or times out (default: 3)
    max_cells_for_matrix : int
        Maximum number of cells (leaves) for which to compute distance matrices (default: 150)
        If exceeded, matrices are skipped to avoid hangs (O(n^2) complexity)
        
    Returns
    -------
    dict
        Metadata dictionary for this run, or None if failed after retries
    """
    # Create unique seed from base_seed, run_id, and repetition
    seed = base_seed + run_id * 1000 + repetition
    
    # Create temporary config file (create once, reuse for retries)
    temp_dir = tempfile.mkdtemp()
    temp_dir = Path(temp_dir)
    
    try:
        # Retry loop with different seeds
        for retry in range(max_retries + 1):
            current_seed = seed + retry * 10000  # Use different seed for each retry
            
            # Recreate config file for each retry (in case it was deleted)
            config_path = create_temp_config(config_dict, temp_dir)
            
            try:
                # Run simulation with timeout
                # Note: signal-based timeout only works in sequential execution
                # For parallel execution, timeout is handled at future.result() level
                import signal
                timeout_set = False
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Simulation exceeded {timeout_seconds} seconds")
                
                # Set up timeout (Unix only - for Windows, we'll use a different approach)
                # Only use signal timeout in sequential mode (when max_workers is None or 1)
                # In parallel mode, timeout is handled by future.result(timeout=...)
                if hasattr(signal, 'SIGALRM'):
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(timeout_seconds)
                        timeout_set = True
                    except (ValueError, OSError):
                        # Signal might not work in all contexts (e.g., in worker processes)
                        timeout_set = False
                
                try:
                    # Run simulation
                    sim = CancerCellEvolutionSimulator(str(config_path), seed=current_seed)
                    sim.run_simulation()
                    
                    # Check tree size before proceeding
                    num_nodes = len(sim.tree.nodes())
                    if num_nodes > max_nodes:
                        if timeout_set:
                            signal.alarm(0)  # Cancel timeout
                        if retry < max_retries:
                            print(f"Warning: Run {run_id:03d} (rep {repetition}, retry {retry}) produced {num_nodes} nodes (max: {max_nodes}). Retrying with different seed...")
                            continue
                        else:
                            print(f"Warning: Run {run_id:03d} (rep {repetition}) produced {num_nodes} nodes (max: {max_nodes}) after {max_retries} retries. Skipping.")
                            return None
                    
                    # Cancel timeout if we got here
                    if timeout_set:
                        signal.alarm(0)
                    
                except TimeoutError as e:
                    if timeout_set:
                        signal.alarm(0)  # Cancel timeout
                    if retry < max_retries:
                        print(f"Warning: Run {run_id:03d} (rep {repetition}, retry {retry}) timed out after {timeout_seconds}s. Retrying with different seed...")
                        continue
                    else:
                        print(f"Warning: Run {run_id:03d} (rep {repetition}) timed out after {timeout_seconds}s after {max_retries} retries. Skipping.")
                        return None
                
                # Get all leaf nodes (cells from final generation)
                max_generation = max(
                    data.get('generation', 0) for _, data in sim.tree.nodes(data=True)
                )
                leaf_nodes = [
                    node for node, data in sim.tree.nodes(data=True)
                    if data.get('generation') == max_generation
                ]
                
                if len(leaf_nodes) < 3:
                    print(f"Warning: Run {run_id:03d} (rep {repetition}) produced only {len(leaf_nodes)} leaves. Skipping.")
                    return None
                
                # Get Genotype objects for leaf nodes
                cells = [sim.genotypes[node_id] for node_id in leaf_nodes]
                
                # Skip matrix computation if too many cells (to avoid hangs)
                # Matrix computation complexity is O(n^2), so limit to reasonable size
                if len(cells) > max_cells_for_matrix:
                    print(f"Warning: Run {run_id:03d} (rep {repetition}) has {len(cells)} cells (max: {max_cells_for_matrix}). Skipping matrix computation to avoid hangs.")
                    # Still save the tree, but skip matrices
                    run_id_str = f"{run_id:03d}"
                    trees_dir = output_dir / "trees"
                    trees_dir.mkdir(exist_ok=True)
                    tree_file = trees_dir / f"tree{run_id_str}.csv"
                    export_tree_to_csv(sim.tree, tree_file)
                    
                    # Return metadata without matrices
                    metadata = {
                        'tree_file_path': f"trees/{tree_file.name}",
                        'repetition': repetition,
                        'run_id': run_id,
                        'seed': current_seed,
                        'num_nodes': len(sim.tree.nodes()),
                        'num_leaves': len(leaf_nodes),
                        'matrices_skipped': True,
                        'skip_reason': f'Too many cells: {len(cells)} > {max_cells_for_matrix}'
                    }
                    for key, value in config_dict.items():
                        metadata[key] = value
                    return metadata
                
                # Generate file names
                run_id_str = f"{run_id:03d}"
                
                # Create subdirectories if they don't exist
                trees_dir = output_dir / "trees"
                matrix_c2c_dir = output_dir / "matrix_c2c"
                matrix_other_dir = output_dir / "matrix_other"
                trees_dir.mkdir(exist_ok=True)
                matrix_c2c_dir.mkdir(exist_ok=True)
                matrix_other_dir.mkdir(exist_ok=True)
                
                tree_file = trees_dir / f"tree{run_id_str}.csv"
                c2c_file = matrix_c2c_dir / f"matrix_c2c_{run_id_str}.txt"
                naive_file = matrix_other_dir / f"matrix_other_{run_id_str}.txt"
                
                # Export tree
                export_tree_to_csv(sim.tree, tree_file)
                
                # Compute cnp2cnp distance matrix (using optimized version if available)
                # Disable progress bar for batch processing (too verbose)
                # CRITICAL: When running in parallel mode, limit nested parallelism to avoid deadlocks
                # Use max_threads=1 to avoid nested ProcessPoolExecutor issues
                # (The outer executor already parallelizes across simulations)
                try:
                    if USE_OPTIMIZED:
                        # Limit nested parallelism: use 1 worker to avoid ProcessPoolExecutor nesting issues
                        # The outer executor already provides parallelism across simulations
                        c2c_ids, c2c_matrix = distance_matrix_from_biopsy_fast(
                            cells, 
                            show_progress=False,
                            max_threads=1  # CRITICAL: Avoid nested ProcessPoolExecutor deadlocks
                        )
                    else:
                        c2c_ids, c2c_matrix = distance_matrix_from_biopsy_fast(
                            cells,
                            max_threads=1  # CRITICAL: Avoid nested ProcessPoolExecutor deadlocks
                        )
                except Exception as e:
                    print(f"Warning: Run {run_id:03d} (rep {repetition}) cnp2cnp matrix computation failed: {e}. Skipping matrices.")
                    metadata = {
                        'tree_file_path': f"trees/{tree_file.name}",
                        'repetition': repetition,
                        'run_id': run_id,
                        'seed': current_seed,
                        'num_nodes': len(sim.tree.nodes()),
                        'num_leaves': len(leaf_nodes),
                        'matrices_skipped': True,
                        'skip_reason': f'cnp2cnp computation error: {str(e)[:100]}'
                    }
                    for key, value in config_dict.items():
                        metadata[key] = value
                    return metadata
                
                write_distance_matrix(c2c_ids, c2c_matrix, c2c_file)
                
                # Compute naive distance matrix (disable progress bar for batch processing)
                # Naive computation is sequential, so no nested parallelism issue
                try:
                    naive_ids, naive_matrix = compute_naive_distance_matrix(
                        cells, 
                        desc=f"Run {run_id:03d} naive",
                        show_progress=False
                    )
                except Exception as e:
                    print(f"Warning: Run {run_id:03d} (rep {repetition}) naive matrix computation failed: {e}. Saving partial results.")
                    # Still save c2c matrix even if naive failed
                    metadata = {
                        'tree_file_path': f"trees/{tree_file.name}",
                        'repetition': repetition,
                        'run_id': run_id,
                        'seed': current_seed,
                        'num_nodes': len(sim.tree.nodes()),
                        'num_leaves': len(leaf_nodes),
                        'naive_matrix_skipped': True,
                        'skip_reason': f'naive computation error: {str(e)[:100]}'
                    }
                    for key, value in config_dict.items():
                        metadata[key] = value
                    return metadata
                
                write_distance_matrix(naive_ids, naive_matrix, naive_file)
                
                # Compare matrices (cnp2cnp vs naive)
                # Skip comparison in grid simulation to speed up and avoid hangs
                # Comparison can be done later in post-processing
                comparison_metrics = {}
                # Uncomment below if you want matrix comparisons during grid simulation
                # try:
                #     from simulation_utils import compare_matrices_programmatically
                #     comparison_metrics = compare_matrices_programmatically(
                #         c2c_file, naive_file, 
                #         permutations=99,  # Fast mode for grid search
                #         fast_mode=True
                #     )
                #     # Prefix comparison metrics to avoid conflicts
                #     comparison_metrics = {f'comparison_{k}': v for k, v in comparison_metrics.items()}
                # except Exception as e:
                #     # Silently skip comparison to avoid hanging
                #     pass
                
                # Create metadata entry (store relative path from output_dir)
                metadata = {
                    'tree_file_path': f"trees/{tree_file.name}",
                    'repetition': repetition,
                    'run_id': run_id,
                    'seed': current_seed,  # Use the actual seed that worked
                    'num_nodes': len(sim.tree.nodes()),
                    'num_leaves': len(leaf_nodes),
                }
                distance_provenance = selected_cnp2cnp_provenance()
                metadata.update({
                    'cnp2cnp_provenance_schema': distance_provenance['schema_version'],
                    'cnp2cnp_semantics_version': distance_provenance['semantics_version'],
                    'cnp2cnp_symmetrization': distance_provenance['symmetrization'],
                    'cnp2cnp_formula': distance_provenance['formula'],
                    'cnp2cnp_construction': distance_provenance['construction'],
                    'cnp2cnp_source_revision': distance_provenance['cnp2cnp_source_revision'],
                    'cnp2cnp_source_description': distance_provenance['cnp2cnp_source_description'],
                    'cnp2cnp_source_sha256': json.dumps(
                        distance_provenance['source_sha256'],
                        sort_keys=True,
                    ),
                    'cnp2cnp_command_template': json.dumps(
                        distance_provenance['command_template'],
                    ),
                })
                
                # Add all config parameters
                for key, value in config_dict.items():
                    metadata[key] = value
                
                # Add comparison metrics
                metadata.update(comparison_metrics)
                
                return metadata
                
            except Exception as e:
                if retry < max_retries:
                    print(f"Warning: Run {run_id:03d} (rep {repetition}, retry {retry}) failed: {e}. Retrying with different seed...")
                    continue
                else:
                    print(f"Error in run {run_id:03d} (rep {repetition}): {e}")
                    return None
        
        # If we get here, all retries failed
        return None
    finally:
        # Clean up temporary directory only after all retries are done
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_grid_simulation(param_grid, output_dir, repetitions=10, base_seed=42, param_grid_file=None, 
                        max_workers=None, max_nodes=1000, timeout_seconds=60, max_retries=3, max_cells_for_matrix=150):
    """
    Run grid simulation with all parameter combinations.
    
    Parameters
    ----------
    param_grid : dict
        Parameter grid dictionary
    output_dir : str or Path
        Output directory for results
    repetitions : int
        Number of repetitions per parameter combination
    base_seed : int
        Base seed for random number generation
    param_grid_file : str or Path, optional
        Path to parameter grid JSON file (will be copied to output_dir)
    max_workers : int, optional
        Maximum number of parallel workers (default: None = sequential)
    max_nodes : int
        Maximum number of nodes allowed in a tree (default: 1000)
    timeout_seconds : int
        Maximum time allowed for a single simulation in seconds (default: 60)
    max_retries : int
        Maximum number of retries with different seeds if tree is too large or times out (default: 3)
    max_cells_for_matrix : int
        Maximum number of cells (leaves) for which to compute distance matrices (default: 150)
        If exceeded, matrices are skipped to avoid hangs (O(n^2) complexity)
        
    Returns
    -------
    Path
        Path to metadata.csv file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameter grid JSON to output directory for reproducibility
    if param_grid_file is not None:
        param_grid_file = Path(param_grid_file)
        if param_grid_file.exists():
            import shutil
            output_grid_file = output_dir / "parameter_grid.json"
            shutil.copy(param_grid_file, output_grid_file)
            print(f"Parameter grid saved to: {output_grid_file}")
    else:
        # Save the current parameter grid to output directory
        output_grid_file = output_dir / "parameter_grid.json"
        with open(output_grid_file, 'w') as f:
            json.dump(param_grid, f, indent=2)
        print(f"Parameter grid saved to: {output_grid_file}")
    
    # Expand parameter grid
    print("Expanding parameter grid...")
    combinations = expand_parameter_grid(param_grid)
    total_runs = len(combinations) * repetitions
    print(f"Total parameter combinations: {len(combinations)}")
    print(f"Repetitions per combination: {repetitions}")
    print(f"Total simulation runs: {total_runs}")
    
    # Print safety limits
    print(f"Safety limits: max_nodes={max_nodes}, timeout={timeout_seconds}s, max_retries={max_retries}, max_cells_for_matrix={max_cells_for_matrix}")
    
    # Prepare all simulation tasks
    all_tasks = []
    run_id = 1
    for combo_idx, config_dict in enumerate(combinations):
        for rep in range(repetitions):
            all_tasks.append((config_dict, rep, run_id, output_dir, base_seed, max_nodes, timeout_seconds, max_retries, max_cells_for_matrix))
            run_id += 1
    
    metadata_path = output_dir / "metadata.csv"
    all_metadata = []
    
    # Run simulations (parallel or sequential)
    if max_workers and max_workers > 1:
        print(f"\nRunning simulations in parallel with {max_workers} workers...")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        
        executor = None
        future_to_task = {}
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
            # Submit all tasks
            future_to_task = {
                executor.submit(run_single_grid_simulation, *task): task 
                for task in all_tasks
            }
            
            # Process results with progress bar
            # Use a lock for thread-safe metadata writing
            import threading
            metadata_lock = threading.Lock()
            write_counter = [0]  # Use list to make it mutable in nested function
            
            try:
                for future in tqdm(as_completed(future_to_task), total=len(all_tasks), 
                                 desc="Running simulations", unit="run"):
                    task = future_to_task[future]
                    run_id = task[2]
                    try:
                        # Add timeout at the future.result() level for parallel execution
                        # This ensures the entire task (including retries) doesn't exceed the timeout
                        # Use a more aggressive timeout: (timeout_seconds * (max_retries + 1)) + small buffer
                        # The timeout should account for simulation + matrix computation
                        total_timeout = timeout_seconds * (max_retries + 1) + 60  # Add 60s buffer for matrix computation overhead
                        metadata = future.result(timeout=total_timeout)
                        if metadata is not None:
                            with metadata_lock:
                                all_metadata.append(metadata)
                                write_counter[0] += 1
                                # Write metadata every 10 completions to reduce I/O overhead
                                if write_counter[0] % 10 == 0:
                                    _write_metadata_csv(all_metadata, metadata_path)
                    except TimeoutError as e:
                        print(f"\n  Error in run {run_id:03d}: Task exceeded total timeout ({total_timeout}s). Skipping.")
                        # Note: future.cancel() only works if the future hasn't started executing yet
                        # Once a task is running, we can't cancel it, but the timeout will prevent
                        # the main thread from waiting indefinitely
                        try:
                            future.cancel()
                        except Exception:
                            pass  # Ignore if cancellation fails (task already running)
                    except Exception as e:
                        print(f"\n  Error in run {run_id:03d}: {e}")
                        # Continue processing other tasks even if one fails
                
                # All futures completed - ensure final metadata write
                print("\nAll simulations completed. Writing final metadata...")
                with metadata_lock:
                    _write_metadata_csv(all_metadata, metadata_path)
                
                # Collect any remaining results that might have been missed
                # (though as_completed should have gotten them all)
                remaining_futures = [f for f in future_to_task.keys() if not f.done()]
                if remaining_futures:
                    print(f"\nCollecting {len(remaining_futures)} remaining futures...")
                    for future in remaining_futures:
                        try:
                            # Use a short timeout for cleanup - if it's not done by now, skip it
                            metadata = future.result(timeout=2)
                            if metadata is not None:
                                with metadata_lock:
                                    all_metadata.append(metadata)
                        except (TimeoutError, Exception):
                            # Ignore errors in cleanup - these are likely stuck tasks
                            pass
                
                # Final metadata write after collecting all results
                print("Writing final metadata...")
                with metadata_lock:
                    _write_metadata_csv(all_metadata, metadata_path)
                
            finally:
                # CRITICAL: Shutdown executor properly to avoid hanging worker processes
                if executor is not None:
                    print("Shutting down worker processes...")
                    # Cancel all pending futures first
                    pending_futures = [f for f in future_to_task.keys() if not f.done()]
                    if pending_futures:
                        print(f"Cancelling {len(pending_futures)} pending tasks...")
                        for future in pending_futures:
                            try:
                                future.cancel()
                            except Exception:
                                pass
                    
                    # Shutdown with wait=False to allow faster termination
                    # This sends shutdown signal to workers immediately rather than waiting
                    executor.shutdown(wait=False)
                    # Give workers a moment to clean up, but don't wait indefinitely
                    import time
                    time.sleep(0.5)
                    # Force shutdown if still running
                    executor.shutdown(wait=True)
                    print("Worker processes terminated.")
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user.")
            print(f"Writing partial metadata for {len(all_metadata)} completed runs...")
            _write_metadata_csv(all_metadata, metadata_path)
            
            # CRITICAL: Force immediate shutdown on KeyboardInterrupt
            if executor is not None:
                print("Forcefully shutting down worker processes...")
                try:
                    # Cancel all pending futures
                    pending_futures = [f for f in future_to_task.keys() if not f.done()]
                    for future in pending_futures:
                        try:
                            future.cancel()
                        except Exception:
                            pass
                    # Shutdown without waiting - let OS clean up if needed
                    executor.shutdown(wait=False)
                except Exception as e:
                    print(f"Error during executor shutdown: {e}")
            
            raise
        except Exception as e:
            print(f"\n\nError during parallel simulation: {e}")
            print(f"Writing partial metadata for {len(all_metadata)} completed runs...")
            _write_metadata_csv(all_metadata, metadata_path)
            raise
    else:
        # Sequential execution (original code)
        print("\nRunning simulations sequentially...")
        run_id = 1
        try:
            for combo_idx, config_dict in enumerate(combinations):
                print(f"\n=== Combination {combo_idx + 1}/{len(combinations)} ===")
                print(f"Parameters: genome_length={config_dict.get('genome_length')}, "
                      f"GENERAL_EVENT_PROB={config_dict.get('GENERAL_EVENT_PROB')}, "
                      f"GENERAL_DUPLICATION_PROB={config_dict.get('GENERAL_DUPLICATION_PROB')}")
                
                for rep in range(repetitions):
                    print(f"  Running repetition {rep + 1}/{repetitions} (run_id={run_id:03d})...", end=' ', flush=True)
                    
                    metadata = run_single_grid_simulation(
                        config_dict, rep, run_id, output_dir, base_seed, 
                        max_nodes=max_nodes, timeout_seconds=timeout_seconds, max_retries=max_retries,
                        max_cells_for_matrix=max_cells_for_matrix
                    )
                    
                    if metadata is not None:
                        all_metadata.append(metadata)
                        print("✓")
                    else:
                        print("✗ (skipped)")
                    
                    run_id += 1
                    
                    # Write metadata incrementally (after each run) for safety
                    _write_metadata_csv(all_metadata, metadata_path)
        
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user.")
            print(f"Writing partial metadata for {len(all_metadata)} completed runs...")
            _write_metadata_csv(all_metadata, metadata_path)
            raise
        except Exception as e:
            print(f"\n\nError during simulation: {e}")
            print(f"Writing partial metadata for {len(all_metadata)} completed runs...")
            _write_metadata_csv(all_metadata, metadata_path)
            raise
    
    # Final metadata write (ensures it's up to date)
    if not all_metadata:
        print("\nWarning: No successful simulations!")
        return None
    
    _write_metadata_csv(all_metadata, metadata_path)
    
    # Generate scenario interpretations
    print("\nGenerating scenario interpretations...")
    scenarios_path = output_dir / "scenarios_interpretation.csv"
    _write_scenarios_interpretation(all_metadata, scenarios_path, combinations)
    
    print(f"\n=== Grid Simulation Complete ===")
    print(f"Successful runs: {len(all_metadata)}/{total_runs}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Scenario interpretations saved to: {scenarios_path}")
    print(f"Results directory: {output_dir}")
    
    return metadata_path


def _write_scenarios_interpretation(all_metadata, scenarios_path, combinations):
    """
    Write scenario interpretations CSV file.
    
    Parameters
    ----------
    all_metadata : list
        List of metadata dictionaries
    scenarios_path : Path
        Path to output CSV file
    combinations : list
        List of parameter combination dictionaries
    """
    # Create a mapping from parameter combination to interpretation
    # We need to match each metadata entry to its parameter combination
    scenario_interpretations = []
    
    # Group metadata by parameter combination (excluding run_id, repetition, seed, etc.)
    # Parameters that define a scenario (excluding run-specific ones)
    scenario_params = [
        'genome_length', 'initial_copies', 'NUMBER_OF_GENERATIONS',
        'REPRESENTATION_TYPE', 'OFFSPRING_MODEL', 'OFFSPRING_PARAMETER',
        'GENERAL_EVENT_PROB', 'GENERAL_DUPLICATION_PROB', 
        'GENERAL_DUPLICATION_MULTIPLICITY', 'GENERAL_LOSS_PROB',
        'GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB', 'MODEL_TELOMERIC_REGIONS',
        'GENERAL_TELOMERIC_PERCENTAGE', 'GENERAL_TELOMERIC_INSTABILITY',
        'MODEL_CRUCIAL_FOR_SURVIVAL'
    ]
    
    # Create interpretations for each unique parameter combination
    seen_combinations = {}
    
    for metadata in all_metadata:
        tree_file = metadata.get('tree_file_path', '')
        
        # Extract parameter combination
        config_dict = {k: metadata.get(k) for k in scenario_params if k in metadata}
        
        # Create a hashable key for the combination
        combo_key = tuple(sorted(config_dict.items()))
        
        # Get or create interpretation
        if combo_key not in seen_combinations:
            interpretation = interpret_scenario(config_dict)
            seen_combinations[combo_key] = interpretation
        
        scenario_interpretations.append({
            'tree_file': tree_file,
            'interpretation': seen_combinations[combo_key]
        })
    
    # Write to CSV
    with open(scenarios_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tree_file', 'interpretation'])
        writer.writeheader()
        writer.writerows(scenario_interpretations)
    
    print(f"  Generated {len(scenario_interpretations)} scenario interpretations")


def _write_metadata_csv(all_metadata, metadata_path):
    """
    Helper function to write metadata CSV file.
    Can be called incrementally to save progress.
    """
    if not all_metadata:
        return
    
    # Get all unique keys from metadata entries
    all_keys = set()
    for entry in all_metadata:
        all_keys.update(entry.keys())
    
    # Sort keys: put important ones first
    priority_keys = ['run_id', 'tree_file_path', 'repetition', 'seed', 
                     'genome_length', 'initial_copies', 'num_nodes', 'num_leaves']
    
    # Group comparison metrics together
    comparison_keys = sorted([k for k in all_keys if k.startswith('comparison_')])
    
    # Other keys (config parameters, etc.)
    other_keys = sorted([k for k in all_keys if k not in priority_keys and k not in comparison_keys])
    
    # Order: priority -> comparison metrics -> other config parameters
    fieldnames = ([k for k in priority_keys if k in all_keys] + 
                  comparison_keys + 
                  other_keys)
    
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metadata)


def main():
    parser = argparse.ArgumentParser(
        description="Run grid simulation with parameter combinations"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='sim_grid_results',
        help='Output directory for results (default: sim_grid_results)'
    )
    parser.add_argument(
        '--repetitions',
        type=int,
        default=10,
        help='Number of repetitions per parameter combination (default: 10)'
    )
    parser.add_argument(
        '--parameter-grid',
        type=str,
        default=None,
        help='Path to JSON file with parameter grid (default: figure_3_parameter_grid.json)'
    )
    parser.add_argument(
        '--base-seed',
        type=int,
        default=42,
        help='Base seed for random number generation (default: 42)'
    )
    parser.add_argument(
        '--max-cells-for-matrix',
        type=int,
        default=150,
        help='Maximum number of cells (leaves) for which to compute distance matrices (default: 150). If exceeded, matrices are skipped to avoid hangs.'
    )
    
    args = parser.parse_args()
    
    # Load parameter grid
    try:
        param_grid = load_parameter_grid(args.parameter_grid)
        if args.parameter_grid:
            print(f"Loaded parameter grid from: {args.parameter_grid}")
        else:
            print(f"Loaded default parameter grid from: {DEFAULT_PARAMETER_GRID_FILE}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Print parameter grid summary
    print("\n=== Parameter Grid ===")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    # Run grid simulation
    metadata_path = run_grid_simulation(
        param_grid,
        args.output_dir,
        repetitions=args.repetitions,
        base_seed=args.base_seed,
        param_grid_file=args.parameter_grid if args.parameter_grid else DEFAULT_PARAMETER_GRID_FILE,
        max_cells_for_matrix=args.max_cells_for_matrix
    )
    
    if metadata_path is None:
        print("Grid simulation failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
