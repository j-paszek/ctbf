#!/usr/bin/env python3
"""
Compute MEDICC2 distance matrices from existing tree CSV files.

This script:
1. Loads trees from CSV files
2. Converts CNP profiles to MEDICC2 TSV format
3. Runs MEDICC2 to compute pairwise distances
4. Extracts distance matrices from MEDICC2 output
5. Saves matrices in the same format as other distance matrices

Usage:
    python compute_medicc2_matrices.py [--results-dir RESULTS_DIR] [--medicc2-path PATH] [--update-metadata]
"""
import argparse
import pandas as pd
import numpy as np
import csv
import subprocess
import tempfile
import shutil
from pathlib import Path
import sys
import os
import time
import threading
import queue

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


def check_medicc2_available(medicc2_path=None):
    """Check if MEDICC2 is available.
    
    Priority order:
    1. Installed Python module (import medicc) - PREFERRED
    2. Command line tool (medicc2 in PATH)
    3. Custom path (if provided)
    """
    # FIRST: Try Python import (installed version) - preferred
    try:
        import medicc
        medicc_file = Path(medicc.__file__)
        if 'site-packages' in str(medicc_file) or 'dist-packages' in str(medicc_file):
            import shutil
            medicc2_cmd = shutil.which('medicc2')
            if medicc2_cmd:
                try:
                    result = subprocess.run(
                        ['medicc2', '--version'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        return True, 'command_line', None
                except Exception:
                    pass
            return True, 'python_module', None
    except ImportError:
        pass
    except Exception:
        pass
    
    # SECOND: Try command line (if Python import failed)
    try:
        result = subprocess.run(
            ['medicc2', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, 'command_line', None
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # THIRD: Try custom path (if provided)
    if medicc2_path:
        medicc2_script = Path(medicc2_path) / "medicc2"
        if medicc2_script.exists():
            return True, 'custom_path', medicc2_path

    return False, None, None


def load_tree_from_csv(csv_path):
    """
    Load a tree from CSV format and extract CNP profiles for leaf nodes.
    
    Returns
    -------
    leaf_cells : list of dict
        List of dictionaries with 'cell_id' and 'genome' (CNP array)
    """
    tree_data = []
    leaf_nodes = set()
    all_nodes = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type'] == 'node':
                node_id = int(row['node_id'])
                cell_id = row['cell_id'] if row['cell_id'] else str(node_id)
                generation = int(row['generation']) if row['generation'] else 0
                genome_str = row['genome']
                genome = [int(x) for x in genome_str.split(',')] if genome_str else []
                
                tree_data.append({
                    'node_id': node_id,
                    'cell_id': cell_id,
                    'generation': generation,
                    'genome': genome
                })
                all_nodes.add(node_id)
            
            elif row['type'] == 'edge':
                parent_id = int(row['parent_id'])
                child_id = int(row['child_id'])
                all_nodes.add(parent_id)
                all_nodes.add(child_id)
    
    # Find leaf nodes (nodes that are not parents)
    parent_nodes = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type'] == 'edge':
                parent_id = int(row['parent_id'])
                parent_nodes.add(parent_id)
    
    leaf_nodes = all_nodes - parent_nodes
    
    # Get CNP profiles for leaf nodes
    leaf_cells = []
    for node_data in tree_data:
        if node_data['node_id'] in leaf_nodes:
            leaf_cells.append({
                'cell_id': node_data['cell_id'],
                'genome': node_data['genome']
            })
    
    return leaf_cells


def convert_cnps_to_medicc2_tsv(cells, output_path, genome_length=None, use_total_cn=True):
    """
    Convert CNP profiles to MEDICC2 TSV format.
    
    MEDICC2 expects:
    - sample_id, chrom, start, end, cn_a, cn_b (for allele-specific)
    - OR sample_id, chrom, start, end, cn_a (for total copy numbers)
    
    When using --total-copy-numbers flag, MEDICC2 expects only ONE allele column.
    We use --input-allele-columns cn_a to specify this.
    
    Note: MEDICC2 requires consistent segmentation across all samples.
    We create one segment per genomic position.
    
    Parameters
    ----------
    cells : list of dict
        List with 'cell_id' and 'genome' (CNP array)
    output_path : Path
        Output TSV file path
    genome_length : int, optional
        Genome length (if None, inferred from first cell)
    use_total_cn : bool
        If True, use only cn_a column (for --total-copy-numbers)
    """
    if not cells:
        raise ValueError("No cells provided")
    
    if genome_length is None:
        genome_length = len(cells[0]['genome'])
    
    # MEDICC2 requires consistent segmentation across all samples
    # We'll create one segment per position
    # All cells must have the same segments (same start/end positions)
    
    rows = []
    
    # First, add a diploid reference sample (MEDICC2 requires this)
    # This prevents MEDICC2 from adding it later, which can break the sort order
    diploid_id = 'diploid'
    for pos in range(genome_length):
        if use_total_cn:
            rows.append({
                'sample_id': diploid_id,
                'chrom': 'chrom1',
                'start': pos,
                'end': pos + 1,
                'cn_a': 2  # Diploid = 2 copies
            })
        else:
            rows.append({
                'sample_id': diploid_id,
                'chrom': 'chrom1',
                'start': pos,
                'end': pos + 1,
                'cn_a': 1,  # Diploid = 1 copy per allele
                'cn_b': 1
            })
    
    # Then add all the actual cells
    for cell in cells:
        cell_id = cell['cell_id']
        genome = cell['genome']
        
        # Ensure genome has the correct length
        if len(genome) < genome_length:
            # Pad with diploid (2) if needed
            genome = list(genome) + [2] * (genome_length - len(genome))
        elif len(genome) > genome_length:
            # Truncate if needed
            genome = genome[:genome_length]
        
        # Create segments: one per position
        # Use "chrom1" as chromosome name, positions 0 to genome_length-1
        for pos in range(genome_length):
            cn_total = int(genome[pos])
            
            if use_total_cn:
                # For total copy numbers, use only cn_a column
                rows.append({
                    'sample_id': str(cell_id),
                    'chrom': 'chrom1',
                    'start': pos,
                    'end': pos + 1,  # BED format: end is non-inclusive
                    'cn_a': cn_total
                })
            else:
                # For allele-specific, use both columns
                rows.append({
                    'sample_id': str(cell_id),
                    'chrom': 'chrom1',
                    'start': pos,
                    'end': pos + 1,
                    'cn_a': cn_total,
                    'cn_b': 0
                })
    
    # Write TSV
    df = pd.DataFrame(rows)
    
    # MEDICC2 expects the data to be sorted by sample_id, chrom, start, end
    # This ensures the MultiIndex created by MEDICC2 will be sorted
    df = df.sort_values(['sample_id', 'chrom', 'start', 'end'])
    
    df.to_csv(output_path, sep='\t', index=False)
    
    return output_path


def run_medicc2(input_tsv, output_dir, medicc2_path=None, use_total_cn=True):
    """
    Run MEDICC2 on input TSV file.
    
    Parameters
    ----------
    input_tsv : Path
        Input TSV file path
    output_dir : Path
        Output directory for MEDICC2 results
    medicc2_path : Path, optional
        Path to MEDICC2 installation (if not in PATH)
    use_total_cn : bool
        If True, use --total-copy-numbers flag
        
    Returns
    -------
    pairwise_distances_path : Path or None
        Path to pairwise distances TSV file, or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine MEDICC2 command
    available, method, detected_path = check_medicc2_available(medicc2_path)
    if not available:
        raise RuntimeError("MEDICC2 not available. Please install MEDICC2 or provide --medicc2-path")
    
    # Set up environment for MEDICC2
    env = os.environ.copy()
    
    if method == 'command_line':
        cmd = ['medicc2']
    elif method == 'python_module':
        # Use installed Python module - find the medicc2 entry point script
        try:
            import medicc
            # Try to find medicc2 script via entry points or in site-packages
            import shutil
            medicc2_cmd = shutil.which('medicc2')
            if medicc2_cmd:
                cmd = [medicc2_cmd]
            else:
                # Try to find script in the installed package location
                medicc_path = os.path.dirname(medicc.__file__)
                # Look for medicc2 script in parent directory (site-packages structure)
                parent_dir = os.path.dirname(medicc_path)
                medicc2_script = os.path.join(parent_dir, 'medicc2')
                if os.path.exists(medicc2_script):
                    cmd = [sys.executable, medicc2_script]
                else:
                    # Try bin directory (common for installed packages)
                    import site
                    for site_pkg in site.getsitepackages():
                        bin_dir = os.path.join(os.path.dirname(site_pkg), 'bin')
                        medicc2_bin = os.path.join(bin_dir, 'medicc2')
                        if os.path.exists(medicc2_bin):
                            cmd = [medicc2_bin]
                            break
                    else:
                        raise RuntimeError("MEDICC2 Python module found but cannot locate medicc2 script. Try: pip install -e . in medicc2 directory")
        except Exception as e:
            raise RuntimeError(f"MEDICC2 Python module found but cannot execute: {e}")
    elif method == 'custom_path':
        medicc2_script = Path(medicc2_path) / "medicc2"
        if not medicc2_script.exists():
            raise RuntimeError(f"MEDICC2 script not found at {medicc2_script}")
        # Make sure script is executable
        if not os.access(medicc2_script, os.X_OK):
            # Try with Python
            cmd = [sys.executable, str(medicc2_script)]
        else:
            cmd = [str(medicc2_script)]
        # Add medicc2 directory to Python path
        env['PYTHONPATH'] = str(Path(medicc2_path).parent) + (os.pathsep + env.get('PYTHONPATH', ''))
    else:
        raise RuntimeError(
            "Unsupported MEDICC2 detection method. "
            "Use a working environment with installed MEDICC2 or pass --medicc2-path."
        )
    
    # Build command
    cmd.extend([str(input_tsv), str(output_dir)])
    
    # Add flags
    if use_total_cn:
        cmd.append('--total-copy-numbers')
        cmd.append('--input-allele-columns')  # Specify only one column for total CN
        cmd.append('cn_a')  # Use only cn_a column
    # Note: --topology-only causes issues with output_df being None
    # Instead, we'll let it run normally but skip plots
    cmd.append('--no-plot-tree')  # Skip tree plots
    cmd.append('--plot')  # Set to 'none' to skip plots
    cmd.append('none')
    
    # Run MEDICC2 with progress indication
    # Note: This is a blocking call, so the progress bar won't update during execution
    # But we print status messages before and after
    # We'll use threading to allow periodic status checks
    
    def run_medicc2_process(cmd, env, output_queue):
        """Run MEDICC2 in a separate thread and put result in queue"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=False,  # Don't raise on non-zero exit
                env=env
            )
            output_queue.put(('success', result))
        except subprocess.TimeoutExpired as e:
            output_queue.put(('timeout', e))
        except Exception as e:
            output_queue.put(('error', e))
    
    output_queue = queue.Queue()
    thread = threading.Thread(target=run_medicc2_process, args=(cmd, env, output_queue), daemon=True)
    thread.start()
    
    # Wait for completion with periodic status updates
    start_time_medicc2 = time.time()
    last_status_time = start_time_medicc2
    status_interval = 10  # Print status every 10 seconds
    
    run_name = Path(input_tsv).parent.name
    print(f"  [INFO] Starting MEDICC2 for {run_name}...", flush=True)
    
    while thread.is_alive():
        time.sleep(1)  # Check every second
        elapsed = time.time() - start_time_medicc2
        if time.time() - last_status_time >= status_interval:
            # Print status update (this will appear in the log and terminal)
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            print(f"  [INFO] {run_name}: MEDICC2 still running... ({elapsed_min}m {elapsed_sec}s elapsed)", flush=True)
            last_status_time = time.time()
    
    # Get result from queue
    try:
        status, result = output_queue.get(timeout=1)
    except queue.Empty:
        # Thread finished but no result (shouldn't happen)
        print(f"  [ERROR] {run_name}: MEDICC2 process completed but no result available", flush=True)
        return None
    
    total_elapsed = time.time() - start_time_medicc2
    elapsed_min = int(total_elapsed // 60)
    elapsed_sec = int(total_elapsed % 60)
    
    if status == 'timeout':
        print(f"  [ERROR] {run_name}: MEDICC2 timed out after 1 hour", flush=True)
        return None
    elif status == 'error':
        print(f"  [ERROR] {run_name}: Error running MEDICC2: {result}", flush=True)
        return None
    else:
        print(f"  [INFO] {run_name}: MEDICC2 completed in {elapsed_min}m {elapsed_sec}s", flush=True)
    
    # Process the result
    if result.returncode != 0:
        # Check for common error patterns
        error_msg = ""
        if result.stderr:
            error_msg = result.stderr
            # Check for missing fstlib (compiled dependencies)
            if "fstlib" in error_msg.lower() or "import fstlib" in error_msg or "fstlib.cext" in error_msg:
                # Only print the helpful message, not the full traceback
                print(f"  Warning: MEDICC2 failed - missing compiled dependencies (fstlib/OpenFST)")
                print(f"  MEDICC2 requires compiled C++ extensions. Please install via conda:")
                print(f"    conda install -c bioconda -c conda-forge medicc2")
            else:
                # For other errors, show a brief summary
                print(f"  Warning: MEDICC2 returned non-zero exit code: {result.returncode}")
                # Show first line of error if available
                first_line = error_msg.split('\n')[0] if error_msg else ""
                if first_line:
                    print(f"  Error: {first_line[:100]}")
        else:
            print(f"  Warning: MEDICC2 returned non-zero exit code: {result.returncode}")
        return None
    
    # Find output file
    input_name = Path(input_tsv).stem
    pairwise_distances_path = output_dir / f"{input_name}_pairwise_distances.tsv"
    
    if not pairwise_distances_path.exists():
        # Try alternative naming
        alt_path = output_dir / "pairwise_distances.tsv"
        if alt_path.exists():
            pairwise_distances_path = alt_path
        else:
            print(f"  Warning: MEDICC2 output file not found. Expected: {pairwise_distances_path}")
            print(f"  Available files in {output_dir}: {list(output_dir.glob('*'))}")
            return None
    
    return pairwise_distances_path


def parse_medicc2_pairwise_distances(tsv_path):
    """
    Parse MEDICC2 pairwise distances TSV file.
    
    Format:
        sample1  sample2  sample3
    sample1  0.0  1.0  2.0
    sample2  1.0  0.0  3.0
    ...
    
    Note: Excludes 'diploid' sample to match other distance matrices.
    
    Returns
    -------
    cell_ids : list
        List of cell IDs (excluding 'diploid')
    dist_matrix : np.ndarray
        Distance matrix (n x n), excluding diploid row/column
    """
    df = pd.read_csv(tsv_path, sep='\t', index_col=0)
    
    # Exclude 'diploid' sample to match other distance matrices
    # MEDICC2 includes it as a reference, but we don't need it in our comparisons
    if 'diploid' in df.index:
        df = df.drop(index='diploid')
        df = df.drop(columns='diploid', errors='ignore')  # Also drop column if present
    
    # Get cell IDs (index)
    cell_ids = df.index.tolist()
    
    # Convert to numpy array
    dist_matrix = df.values.astype(float)
    
    # Ensure symmetric (MEDICC2 should already be symmetric, but verify)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    return cell_ids, dist_matrix


def process_all_trees_medicc2(results_dir, medicc2_path=None, start_from_run_id=None, skip_existing=True, update_metadata=False, max_leaves_to_evaluate=None):
    """
    Process all tree CSV files and compute MEDICC2 distance matrices.
    
    Parameters
    ----------
    results_dir : str or Path
        Directory containing tree CSV files and metadata.csv
    medicc2_path : str or Path, optional
        Path to MEDICC2 installation
    start_from_run_id : int, optional
        Start processing from this run_id
    skip_existing : bool
        If True, skip runs where MEDICC2 matrix already exists
    update_metadata : bool
        If True, update metadata.csv with MEDICC2 comparison metrics
    max_leaves_to_evaluate : int, optional
        Maximum number of leaves to process. Trees with more leaves will be skipped.
        If None, process all trees regardless of leaf count.
    """
    results_dir = Path(results_dir)
    
    # Check if MEDICC2 is available
    available, method, detected_path = check_medicc2_available(medicc2_path)
    if not available:
        print("⚠ Warning: MEDICC2 not detected. Cannot compute MEDICC2 matrices.")
        print("  To enable MEDICC2:")
        print("  1. Activate the conda environment with MEDICC2 installed:")
        print("     conda activate medicc2_src  # or your MEDICC2 environment name")
        print("  2. Install MEDICC2: conda install -c bioconda -c conda-forge medicc2")
        print("  3. Or provide --medicc2-path to MEDICC2 installation directory")
        print("  4. Or ensure medicc2 is cloned in the parent directory")
        return
    
    print(f"✓ MEDICC2 detected (method: {method})")
    if detected_path:
        print(f"  Using MEDICC2 from: {detected_path}")
    elif method == 'python_module':
        # Show where it's installed from
        try:
            import medicc
            print(f"  Using installed MEDICC2 from: {medicc.__file__}")
        except:
            pass
    
    # Load metadata
    metadata_path = results_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {results_dir}")
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} simulation runs from {metadata_path}")
    
    # Create MEDICC2 matrix directory
    matrix_medicc2_dir = results_dir / "matrix_medicc2"
    matrix_medicc2_dir.mkdir(exist_ok=True)
    
    # Create temporary directory for MEDICC2 intermediate files
    temp_base = results_dir / "medicc2_temp"
    temp_base.mkdir(exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    skipped_too_many_leaves = 0
    failed_count = 0
    
    # Count how many need to be processed (not skipped)
    total_to_process = 0
    for idx, row in df.iterrows():
        run_id = row.get('run_id', idx + 1)
        if start_from_run_id is not None and run_id < start_from_run_id:
            continue
        tree_file_path = row.get('tree_file_path', '')
        if tree_file_path:
            if '/' in tree_file_path or '\\' in tree_file_path:
                tree_path = results_dir / tree_file_path
            else:
                trees_dir = results_dir / "trees"
                if trees_dir.exists():
                    tree_path = trees_dir / tree_file_path
                else:
                    tree_path = results_dir / tree_file_path
        else:
            trees_dir = results_dir / "trees"
            if trees_dir.exists():
                tree_path = trees_dir / f"tree{run_id:03d}.csv"
            else:
                tree_path = results_dir / f"tree{run_id:03d}.csv"
        medicc2_file = matrix_medicc2_dir / f"matrix_medicc2_{run_id:03d}.txt"
        if not skip_existing or not medicc2_file.exists():
            if tree_path.exists():
                total_to_process += 1
    
    print(f"Total runs to process: {total_to_process} (out of {len(df)} total)")
    if max_leaves_to_evaluate is not None:
        print(f"Maximum leaves per tree: {max_leaves_to_evaluate} (trees with more leaves will be skipped)")
    if total_to_process == 0:
        print("All MEDICC2 matrices already exist. Use --no-skip-existing to recompute.")
        return
    
    # Create progress bar with more detailed information
    pbar = tqdm(total=len(df), desc="Processing MEDICC2", unit="run", 
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                ncols=120)
    
    import time
    start_time = time.time()
    
    for idx, row in df.iterrows():
        tree_file_path = row.get('tree_file_path', '')
        run_id = row.get('run_id', idx + 1)
        
        # Skip if start_from_run_id is specified
        if start_from_run_id is not None and run_id < start_from_run_id:
            skipped_count += 1
            pbar.update(1)
            pbar.set_postfix_str(f'✓:{processed_count} ⊘:{skipped_count} ✗:{failed_count}')
            continue
        
        # Construct file paths
        if tree_file_path:
            if '/' in tree_file_path or '\\' in tree_file_path:
                tree_path = results_dir / tree_file_path
            else:
                trees_dir = results_dir / "trees"
                if trees_dir.exists():
                    tree_path = trees_dir / tree_file_path
                else:
                    tree_path = results_dir / tree_file_path
        else:
            trees_dir = results_dir / "trees"
            if trees_dir.exists():
                tree_path = trees_dir / f"tree{run_id:03d}.csv"
            else:
                tree_path = results_dir / f"tree{run_id:03d}.csv"
        
        medicc2_file = matrix_medicc2_dir / f"matrix_medicc2_{run_id:03d}.txt"
        
        if not tree_path.exists():
            pbar.write(f"  Warning: Tree file not found: {tree_path}")
            pbar.update(1)
            failed_count += 1
            pbar.set_postfix_str(f'✓:{processed_count} ⊘:{skipped_count} ✗:{failed_count}')
            continue
        
        # Check if we need to compute
        if skip_existing and medicc2_file.exists():
            skipped_count += 1
            pbar.update(1)
            pbar.set_postfix_str(f'✓:{processed_count} ⊘:{skipped_count} ✗:{failed_count}')
            continue
        
        # Update description to show current run being processed
        pbar.set_description(f"Processing run {run_id:03d} (MEDICC2)")
        pbar.refresh()  # Force update of progress bar
        
        try:
            # Load tree and extract leaf cell CNPs
            leaf_cells = load_tree_from_csv(tree_path)
            
            if len(leaf_cells) < 2:
                pbar.write(f"  Warning: Run {run_id} has fewer than 2 leaf cells. Skipping.")
                pbar.update(1)
                failed_count += 1
                continue
            
            # Check if tree has too many leaves
            if max_leaves_to_evaluate is not None and len(leaf_cells) > max_leaves_to_evaluate:
                pbar.write(f"  Skipping run {run_id}: {len(leaf_cells)} leaves exceeds maximum ({max_leaves_to_evaluate})")
                pbar.update(1)
                skipped_count += 1
                skipped_too_many_leaves += 1
                pbar.set_postfix_str(f'✓:{processed_count} ⊘:{skipped_count} ✗:{failed_count}')
                continue
            
            # Create temporary directory for this run
            run_temp_dir = temp_base / f"run_{run_id:03d}"
            run_temp_dir.mkdir(exist_ok=True)
            
            try:
                # Convert to MEDICC2 TSV format
                pbar.set_postfix_str(f'Run {run_id:03d}: Converting to TSV...')
                pbar.refresh()
                input_tsv = run_temp_dir / "input.tsv"
                convert_cnps_to_medicc2_tsv(leaf_cells, input_tsv, use_total_cn=True)
                
                # Run MEDICC2 (this can take a while)
                pbar.set_postfix_str(f'Run {run_id:03d}: Running MEDICC2 (this may take a while)...')
                pbar.refresh()
                medicc2_output_dir = run_temp_dir / "medicc2_output"
                pairwise_distances_path = run_medicc2(
                    input_tsv,
                    medicc2_output_dir,
                    medicc2_path=medicc2_path,
                    use_total_cn=True
                )
                
                if pairwise_distances_path is None or not pairwise_distances_path.exists():
                    pbar.write(f"  Warning: MEDICC2 failed for run {run_id} (no output file generated)")
                    pbar.update(1)
                    failed_count += 1
                    # Clean up temp directory
                    shutil.rmtree(run_temp_dir, ignore_errors=True)
                    continue
                
                # Parse MEDICC2 output
                pbar.set_postfix_str(f'Run {run_id:03d}: Parsing MEDICC2 output...')
                pbar.refresh()
                cell_ids, dist_matrix = parse_medicc2_pairwise_distances(pairwise_distances_path)
                
                # Write distance matrix in our format (PHYLIP-like)
                from compute_ground_truth_matrices import write_distance_matrix_phylip
                write_distance_matrix_phylip(cell_ids, dist_matrix, medicc2_file)
                
                processed_count += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_count if processed_count > 0 else 0
                remaining = total_to_process - processed_count
                eta_seconds = avg_time * remaining if remaining > 0 else 0
                eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 0 else "calculating..."
                pbar.write(f"  ✓ Run {run_id:03d}: Computed MEDICC2 matrix ({len(cell_ids)} cells) | ETA: {eta_str}")
                pbar.set_postfix_str(f'✓:{processed_count} ⊘:{skipped_count} ✗:{failed_count} | ETA: {eta_str}')
                
            except RuntimeError as e:
                # MEDICC2 not available or failed to run
                pbar.write(f"  Warning: MEDICC2 error for run {run_id}: {e}")
                failed_count += 1
            except Exception as e:
                pbar.write(f"  Warning: Error processing run {run_id}: {e}")
                failed_count += 1
            finally:
                # Clean up temp directory
                shutil.rmtree(run_temp_dir, ignore_errors=True)
            
        except Exception as e:
            pbar.write(f"  Error loading tree for run {run_id}: {e}")
            failed_count += 1
        
        pbar.update(1)
        # Update postfix with current statistics (only if not already set)
        if processed_count > 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count if processed_count > 0 else 0
            remaining = total_to_process - processed_count
            eta_seconds = avg_time * remaining if remaining > 0 else 0
            eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 0 else "calculating..."
            pbar.set_postfix_str(f'✓:{processed_count} ⊘:{skipped_count} ✗:{failed_count} | ETA: {eta_str}')
        else:
            pbar.set_postfix_str(f'✓:{processed_count} ⊘:{skipped_count} ✗:{failed_count}')
    
    pbar.close()
    
    # Print final summary with timing
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("MEDICC2 matrix computation complete!")
    print(f"Total time: {int(total_time//60)}m {int(total_time%60)}s")
    print(f"Processed: {processed_count} new matrices")
    if processed_count > 0:
        print(f"Average time per matrix: {total_time/processed_count:.2f}s")
    print(f"Processed {processed_count} new trees")
    if skipped_count > 0:
        skip_reasons = []
        if skipped_too_many_leaves > 0:
            skip_reasons.append(f"{skipped_too_many_leaves} too many leaves (>{max_leaves_to_evaluate if max_leaves_to_evaluate is not None else 'N/A'})")
        other_skipped = skipped_count - skipped_too_many_leaves
        if other_skipped > 0:
            skip_reasons.append(f"{other_skipped} already exist or before start_from_run_id")
        print(f"Skipped {skipped_count} trees ({', '.join(skip_reasons)})")
    if failed_count > 0:
        print(f"Failed {failed_count} trees")
        if failed_count == len(df):
            print("\n⚠ All MEDICC2 computations failed. This is likely because:")
            print("  1. You're not in the correct conda environment. Try:")
            print("     conda activate medicc2_src  # or your MEDICC2 environment name")
            print("  2. MEDICC2's compiled dependencies (fstlib/OpenFST) are not available")
            print("  3. MEDICC2 needs to be properly installed via conda:")
            print("     conda install -c bioconda -c conda-forge medicc2")
            print("  4. Or the cloned MEDICC2 directory needs to be compiled/installed")
            print("\n  The pipeline will continue with other distance matrices (GT, cnp2cnp, naive).")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute MEDICC2 distance matrices from existing tree CSV files"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='sim_grid_results',
        help='Directory containing tree CSV files and metadata.csv (default: sim_grid_results)'
    )
    parser.add_argument(
        '--medicc2-path',
        type=str,
        default=None,
        help='Path to MEDICC2 installation directory (if not in PATH)'
    )
    parser.add_argument(
        '--start-from-run-id',
        type=int,
        default=None,
        help='Start processing from this run_id (skip all previous runs)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Recompute MEDICC2 matrices even if they already exist'
    )
    parser.add_argument(
        '--update-metadata',
        action='store_true',
        help='Update metadata.csv with MEDICC2 comparison metrics (not yet implemented)'
    )
    parser.add_argument(
        '--max-leaves-to-evaluate',
        type=int,
        default=None,
        help='Maximum number of leaves to process. Trees with more leaves will be skipped. '
             'Use this to avoid long-running MEDICC2 computations on large trees. '
             'If not specified, all trees will be processed.'
    )
    
    args = parser.parse_args()
    
    process_all_trees_medicc2(
        args.results_dir,
        medicc2_path=args.medicc2_path,
        start_from_run_id=args.start_from_run_id,
        skip_existing=not args.no_skip_existing,
        update_metadata=args.update_metadata,
        max_leaves_to_evaluate=args.max_leaves_to_evaluate
    )


if __name__ == '__main__':
    main()
