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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from _ctbs_config import configured_cnp2cnp_file, configured_cnp2cnp_module_dir

configured_cnp2cnp = str(configured_cnp2cnp_module_dir())
if configured_cnp2cnp not in sys.path:
    sys.path.insert(0, configured_cnp2cnp)

from ctbs import bounded_process_map, resolve_distance_worker_count
from distance_semantics import (
    cnp2cnp_provenance,
    minimum_bidirectional_distance,
    validate_distance_matrix,
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Import cnp2cnp functions directly from the location configured for CTBF.
try:
    from cnpsolver import CNPSolver
    CNP2CNP_AVAILABLE = True
except ImportError:
    CNP2CNP_AVAILABLE = False
    print(
        "Warning: Could not import CNPSolver from configured cnp2cnp; "
        "using the checked subprocess cnp2cnp backend (never L1)."
    )


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


def compute_symmetric_cnp2cnp_distance_direct(cnp1, cnp2, use_dbl=False):
    """Compute the paper-facing min of both directional direct calls."""
    forward = compute_cnp2cnp_distance_direct(cnp1, cnp2, use_dbl=use_dbl)
    reverse = compute_cnp2cnp_distance_direct(cnp2, cnp1, use_dbl=use_dbl)
    return minimum_bidirectional_distance(forward, reverse)


def figure3_cnp2cnp_provenance():
    return cnp2cnp_provenance(
        configured_cnp2cnp_file(),
        construction=(
            "direct_bidirectional_api"
            if CNP2CNP_AVAILABLE
            else "bidirectional_pair_mode"
        ),
    )


def _compute_pair_optimized(args):
    """
    Optimized version that calls Python functions directly instead of subprocess.
    This eliminates subprocess overhead which is the main bottleneck.
    """
    c, d, i, j = args
    
    cnp1 = [int(x) for x in c.get_cnp().split(',')]
    cnp2 = [int(x) for x in d.get_cnp().split(',')]

    if CNP2CNP_AVAILABLE:
        dist = compute_symmetric_cnp2cnp_distance_direct(
            cnp1,
            cnp2,
            use_dbl=False,
        )
    else:
        from ctbs import compute_symmetric_cnp2cnp_distance

        dist = compute_symmetric_cnp2cnp_distance(c, d)
    
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
    if not CNP2CNP_AVAILABLE:
        from ctbs import distance_matrix_from_biopsy

        return distance_matrix_from_biopsy(cells, max_threads=max_threads)

    n = len(cells)
    ids = [c.get_id() for c in cells]
    dist_matrix = np.zeros((n, n), dtype=float)
    
    total_pairs = n * (n - 1) // 2
    worker_count = resolve_distance_worker_count(max_threads, total_pairs)
    pairs = (
        (cells[i], cells[j], i, j)
        for i in range(n)
        for j in range(i + 1, n)
    )
    iterator = bounded_process_map(
        _compute_pair_optimized,
        pairs,
        max_workers=worker_count,
        task_count=total_pairs,
    )
    if show_progress:
        iterator = tqdm(iterator, total=total_pairs, desc=desc, unit="pairs")

    for i, j, dist in iterator:
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist
    
    return validate_distance_matrix(ids, dist_matrix)


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
