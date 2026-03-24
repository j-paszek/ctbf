"""
Optimized utility functions for simulation experiments with direct Python calls.
This version avoids subprocess overhead by calling cnp2cnp functions directly.
"""
import csv
import numpy as np
import networkx as nx
from pathlib import Path
import sys
import os

# Add local cnp2cnp path
local_cnp2cnp = os.path.join(os.path.dirname(__file__), 'cnp2cnp', 'cnp2cnp')
if local_cnp2cnp not in sys.path:
    sys.path.insert(0, local_cnp2cnp)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import tempfile

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Import cnp2cnp functions directly
try:
    from cnpsolver import CNPSolver
    CNP2CNP_AVAILABLE = True
except ImportError:
    CNP2CNP_AVAILABLE = False
    print("Warning: Could not import CNPSolver. Falling back to subprocess method.")


def compute_cnp2cnp_distance_direct(cnp1, cnp2, use_dbl=False):
    """
    Compute cnp2cnp distance directly using Python functions (no subprocess).
    
    Parameters
    ----------
    cnp1 : list or str
        First CNP (list of ints or comma-separated string)
    cnp2 : list or str
        Second CNP (list of ints or comma-separated string)
    use_dbl : bool
        Whether to use double constraint
        
    Returns
    -------
    int
        Distance (number of events)
    """
    if not CNP2CNP_AVAILABLE:
        raise ImportError("CNPSolver not available")
    
    # Convert string to list if needed
    if isinstance(cnp1, str):
        cnp1 = [int(x) for x in cnp1.split(',')]
    if isinstance(cnp2, str):
        cnp2 = [int(x) for x in cnp2.split(',')]
    
    # Get comparable CNPs (removes zeros from first CNP)
    tmp = CNPSolver.get_comparable_cnps(cnp1, cnp2)
    u = tmp[0]
    v = tmp[1]
    
    # Compute approximate events
    evs = CNPSolver.get_approximate_events(u, v, use_dbl=use_dbl)
    
    return len(evs)


def _compute_pair_optimized(args):
    """
    Optimized version that calls Python functions directly instead of subprocess.
    This eliminates subprocess overhead which is the main bottleneck.
    """
    c, d, i, j = args
    
    try:
        # Get CNP strings
        cnp1_str = c.get_cnp()
        cnp2_str = d.get_cnp()
        
        # Convert to lists
        cnp1 = [int(x) for x in cnp1_str.split(',')]
        cnp2 = [int(x) for x in cnp2_str.split(',')]
        
        # Compute distance directly using Python functions
        if CNP2CNP_AVAILABLE:
            dist = compute_cnp2cnp_distance_direct(cnp1, cnp2, use_dbl=False)
        else:
            # Fallback to subprocess if direct import failed
            input_str = f">{c.get_id()}\n{cnp1_str}\n>{d.get_id()}\n{cnp2_str}\n"
            from ctbs import use_cnp2cnp_to_compute_pairwise_distance
            dist = float(use_cnp2cnp_to_compute_pairwise_distance(input_str))
            
    except Exception as e:
        # Fallback: simple Manhattan distance on CNP profiles
        cnp_c = np.array([int(x) for x in c.get_cnp().split(',')])
        cnp_d = np.array([int(x) for x in d.get_cnp().split(',')])
        dist = float(np.sum(np.abs(cnp_c - cnp_d)))
    
    return i, j, dist


def distance_matrix_from_biopsy_optimized(cells, max_threads=None, desc="Computing cnp2cnp distance matrix", show_progress=True):
    """
    Optimized version that calls Python functions directly (no subprocess overhead).
    
    This should be significantly faster than the subprocess-based version.
    
    Parameters
    ----------
    cells : list
        List of Genotype objects
    max_threads : int, optional
        Maximum number of worker threads (default: None = use all CPUs)
    desc : str
        Description for progress bar (ignored if show_progress=False)
    show_progress : bool
        Whether to show progress bar (default: True)
        
    Returns
    -------
    ids : list
        List of cell IDs
    dist_matrix : np.ndarray
        Distance matrix (symmetric, n x n)
    """
    n = len(cells)
    ids = [c.get_id() for c in cells]
    dist_matrix = np.zeros((n, n), dtype=float)
    
    pairs = [(cells[i], cells[j], i, j) for i in range(n) for j in range(i + 1, n)]
    total_pairs = len(pairs)
    
    # Use all available CPUs if max_threads not specified
    if max_threads is None:
        import multiprocessing
        max_threads = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        # Submit all tasks
        future_to_pair = {executor.submit(_compute_pair_optimized, pair): pair for pair in pairs}
        
        # Process results with or without progress bar
        iterator = as_completed(future_to_pair)
        if show_progress:
            iterator = tqdm(iterator, total=total_pairs, desc=desc, unit="pairs")
        
        for future in iterator:
            i, j, dist = future.result()
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return ids, dist_matrix


def export_tree_to_csv(tree, output_path):
    """
    Export a NetworkX tree to CSV format.
    
    The CSV contains:
    - Node information: node_id, cell_id, generation, genome (as comma-separated string)
    - Edge information: parent_id, child_id, events
    
    Parameters
    ----------
    tree : nx.DiGraph
        The tree to export
    output_path : str or Path
        Path to output CSV file
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['type', 'node_id', 'cell_id', 'generation', 'genome', 'parent_id', 'child_id', 'events'])
        
        # Write nodes
        for node_id, data in tree.nodes(data=True):
            cell_id = data.get('cell_id', node_id)
            generation = data.get('generation', '')
            genome = data.get('genome', [])
            genome_str = ','.join(map(str, genome)) if genome is not None else ''
            
            writer.writerow(['node', node_id, cell_id, generation, genome_str, '', '', ''])
        
        # Write edges
        for parent_id, child_id, data in tree.edges(data=True):
            events = data.get('events', '')
            writer.writerow(['edge', '', '', '', '', parent_id, child_id, events])


def compute_naive_distance_matrix(cells, desc="Computing naive distance matrix"):
    """
    Compute a naive distance matrix using Manhattan (L1) distance on CNP profiles.
    This is the "non-cnp2cnp" approach.
    
    Parameters
    ----------
    cells : list
        List of Genotype objects with get_cnp() method
    desc : str
        Description for progress bar
        
    Returns
    -------
    ids : list
        List of cell IDs
    dist_matrix : np.ndarray
        Distance matrix (symmetric, n x n)
    """
    n = len(cells)
    ids = [c.get_id() for c in cells]
    dist_matrix = np.zeros((n, n), dtype=float)
    
    for i in tqdm(range(n), desc=desc, unit="cells"):
        cnp_i = np.array([int(x) for x in cells[i].get_cnp().split(',')])
        for j in range(i + 1, n):
            cnp_j = np.array([int(x) for x in cells[j].get_cnp().split(',')])
            # Manhattan distance (L1 norm)
            dist = float(np.sum(np.abs(cnp_i - cnp_j)))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return ids, dist_matrix


def write_distance_matrix(ids, dist_matrix, output_path):
    """
    Write distance matrix to file in PHYLIP format.
    
    Parameters
    ----------
    ids : list
        List of identifiers (strings)
    dist_matrix : np.ndarray
        Distance matrix (n x n)
    output_path : str or Path
        Path to output file
    """
    output_path = Path(output_path)
    n = len(ids)
    
    with open(output_path, 'w') as f:
        f.write(f"{n}\n")  # number of nodes first
        for i, cid in enumerate(ids):
            f.write(f"{str(cid):<10}")
            f.write(" ".join(str(dist) for dist in dist_matrix[i]))
            f.write("\n")


def compare_matrices_programmatically(matrix1_path, matrix2_path, permutations=99, fast_mode=True):
    """
    Compare two distance matrices using optimized dm_compare functionality.
    Uses the optimized version from simulation_utils.
    """
    # Import from simulation_utils which has the optimized version
    from simulation_utils import compare_matrices_programmatically as compare_fast
    return compare_fast(matrix1_path, matrix2_path, permutations=permutations, fast_mode=fast_mode)
