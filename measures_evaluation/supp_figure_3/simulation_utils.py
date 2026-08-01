"""
Utility functions for simulation experiments.
"""
import csv
import numpy as np
import networkx as nx
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ctbs import (
    compute_symmetric_cnp2cnp_distance,
    load_ctbs_runtime_config,
)
from distance_semantics import cnp2cnp_provenance, validate_distance_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Dummy tqdm if not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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


def _compute_pair_wrapper(args):
    """
    Wrapper for computing pairwise distance that can be used with ProcessPoolExecutor.
    This replicates the logic from ctbs._compute_pair.
    """
    c, d, i, j = args
    dist = compute_symmetric_cnp2cnp_distance(c, d)
    return i, j, dist


def figure3_cnp2cnp_provenance():
    runtime_config = load_ctbs_runtime_config()
    return cnp2cnp_provenance(
        runtime_config.cnp2cnp_file,
        construction="bidirectional_pair_mode",
    )


def distance_matrix_from_biopsy_with_progress(cells, max_threads=None, desc="Computing cnp2cnp distance matrix"):
    """
    Build a distance matrix for a list of cells using cnp2cnp with progress bar.
    
    Parameters
    ----------
    cells : list
        List of Genotype objects
    max_threads : int, optional
        Maximum number of worker threads
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
    
    pairs = [(cells[i], cells[j], i, j) for i in range(n) for j in range(i + 1, n)]
    total_pairs = len(pairs)
    
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        # Submit all tasks
        future_to_pair = {executor.submit(_compute_pair_wrapper, pair): pair for pair in pairs}
        
        # Process results with progress bar
        for future in tqdm(as_completed(future_to_pair), total=total_pairs, desc=desc, unit="pairs"):
            i, j, dist = future.result()
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return validate_distance_matrix(ids, dist_matrix)


def compute_naive_distance_matrix(cells, desc="Computing naive distance matrix", show_progress=True):
    """
    Compute a naive distance matrix using Manhattan (L1) distance on CNP profiles.
    This is the "non-cnp2cnp" approach.
    
    Parameters
    ----------
    cells : list
        List of Genotype objects with get_cnp() method
    desc : str
        Description for progress bar
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
    
    iterator = tqdm(range(n), desc=desc, unit="cells") if show_progress else range(n)
    
    for i in iterator:
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


def _upper_triangular_vector(D):
    """Return the upper triangular (i<j) entries as a 1D vector."""
    return D[np.triu_indices(D.shape[0], k=1)]


def _pearson_corr(x, y):
    """Compute Pearson correlation."""
    x_ = (x - np.mean(x))
    y_ = (y - np.mean(y))
    denom = np.sqrt((x_**2).sum() * (y_**2).sum())
    if denom == 0:
        return np.nan
    return float((x_ * y_).sum() / denom)


def _fast_spearman_corr(x, y):
    """Fast Spearman correlation using scipy if available, otherwise fallback."""
    if SCIPY_AVAILABLE:
        return stats.spearmanr(x, y)[0]
    else:
        # Fallback to manual calculation
        import pandas as pd
        rx = pd.Series(x).rank(method="average").to_numpy()
        ry = pd.Series(y).rank(method="average").to_numpy()
        x_ = (rx - rx.mean())
        y_ = (ry - ry.mean())
        denom = np.sqrt((x_**2).sum() * (y_**2).sum())
        if denom == 0:
            return np.nan
        return float((x_ * y_).sum() / denom)


def compute_ultrametric_deviation_score(D):
    """
    Compute Ultrametric Deviation Score (UDS) for a distance matrix.
    
    For each triplet (i,j,k), compute:
    UD(i,j,k) = |max(D_ij, D_ik, D_jk) - second_largest(D_ij, D_ik, D_jk)|
    
    Returns mean UD over all triplets.
    """
    n = D.shape[0]
    ud_values = []
    
    # Iterate over all triplets
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d_ij = D[i, j]
                d_ik = D[i, k]
                d_jk = D[j, k]
                
                # Get max and second largest
                distances = sorted([d_ij, d_ik, d_jk])
                max_dist = distances[2]
                second_largest = distances[1]
                
                ud = abs(max_dist - second_largest)
                ud_values.append(ud)
    
    return np.mean(ud_values) if ud_values else np.nan


def compute_distance_spectrum_shape(D):
    """
    Compute distance spectrum shape metrics: skewness, kurtosis, 
    bimodality coefficient, and coefficient of variation.
    """
    v = _upper_triangular_vector(D)
    
    mean_dist = np.mean(v)
    std_dist = np.std(v)
    cv = std_dist / mean_dist if mean_dist > 0 else np.nan
    
    # Skewness and kurtosis
    if SCIPY_AVAILABLE:
        skewness = float(stats.skew(v))
        kurtosis = float(stats.kurtosis(v))
    else:
        # Manual calculation
        n = len(v)
        if n < 3:
            skewness = np.nan
            kurtosis = np.nan
        else:
            # Skewness
            m3 = np.mean((v - mean_dist) ** 3)
            skewness = float(m3 / (std_dist ** 3)) if std_dist > 0 else np.nan
            
            # Kurtosis (excess kurtosis)
            m4 = np.mean((v - mean_dist) ** 4)
            kurtosis = float(m4 / (std_dist ** 4) - 3) if std_dist > 0 else np.nan
    
    # Bimodality coefficient (BC)
    # BC = (skewness^2 + 1) / (kurtosis + 3)
    # BC > 0.555 indicates bimodality
    if not np.isnan(skewness) and not np.isnan(kurtosis):
        bc = float((skewness ** 2 + 1) / (kurtosis + 3)) if (kurtosis + 3) != 0 else np.nan
    else:
        bc = np.nan
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'bimodality_coefficient': bc,
        'coefficient_of_variation': cv
    }


def compute_local_compactness_global_spread(D, k=None):
    """
    Compute Local Compactness vs Global Spread (LCGS) metrics.
    
    For each cell i, compute mean distance to k nearest neighbors.
    Then compute statistics over all cells.
    """
    n = D.shape[0]
    
    # Default k: use sqrt(n) or 5, whichever is smaller
    if k is None:
        k = min(int(np.sqrt(n)), 5, n-1)
    k = max(1, min(k, n-1))  # Ensure 1 <= k < n
    
    r_values = []
    
    for i in range(n):
        # Get distances from cell i to all others
        distances = D[i, :].copy()
        distances[i] = np.inf  # Exclude self
        
        # Get k nearest neighbors
        k_nearest_indices = np.argpartition(distances, k)[:k]
        k_nearest_distances = distances[k_nearest_indices]
        
        # Mean distance to k nearest neighbors
        r_i = np.mean(k_nearest_distances)
        r_values.append(r_i)
    
    r_values = np.array(r_values)
    
    mean_r = float(np.mean(r_values))
    std_r = float(np.std(r_values))
    spread = float(np.max(r_values) - np.min(r_values))
    
    # LCGS ratio
    lcgs = float(std_r / mean_r) if mean_r > 0 else np.nan
    
    return {
        'local_compactness_mean': mean_r,
        'local_compactness_std': std_r,
        'local_compactness_spread': spread,
        'lcgs_ratio': lcgs
    }


def compute_eigenvalue_decay(D):
    """
    Compute eigenvalue decay metrics from distance matrix.
    Uses classical MDS (centered distance matrix).
    """
    n = D.shape[0]
    
    # Convert to centered kernel matrix (classical MDS)
    # K = -0.5 * (D^2 - row_means - col_means + grand_mean)
    D_squared = D ** 2
    
    row_means = np.mean(D_squared, axis=1, keepdims=True)
    col_means = np.mean(D_squared, axis=0, keepdims=True)
    grand_mean = np.mean(D_squared)
    
    K = -0.5 * (D_squared - row_means - col_means + grand_mean)
    
    # Compute eigenvalues
    try:
        eigenvals = np.linalg.eigvalsh(K)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
        eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
        
        if len(eigenvals) == 0:
            return {
                'eigenvalue_decay_rate': np.nan,
                'num_significant_eigenvalues': 0,
                'eigenvalue_entropy': np.nan
            }
        
        # Normalize eigenvalues
        eigenvals_norm = eigenvals / np.sum(eigenvals)
        
        # Decay rate: ratio of first to second eigenvalue
        if len(eigenvals) >= 2:
            decay_rate = float(eigenvals[0] / eigenvals[1]) if eigenvals[1] > 0 else np.inf
        else:
            decay_rate = np.inf
        
        # Number of significant eigenvalues (explaining > 1% of variance)
        significant = np.sum(eigenvals_norm > 0.01)
        
        # Entropy of eigenvalue distribution (measure of structure)
        # Higher entropy = more uniform = less structure
        entropy = -np.sum(eigenvals_norm * np.log(eigenvals_norm + 1e-10))
        
        return {
            'eigenvalue_decay_rate': decay_rate,
            'num_significant_eigenvalues': int(significant),
            'eigenvalue_entropy': float(entropy)
        }
    except Exception as e:
        return {
            'eigenvalue_decay_rate': np.nan,
            'num_significant_eigenvalues': 0,
            'eigenvalue_entropy': np.nan
        }


def compute_triangle_thickness_index(D):
    """
    Compute Triangle Thickness Index (TTI).
    
    For each triplet (i,j,k), compute:
    TT(i,j,k) = D_ij + D_ik + D_jk
    
    Then compute mean, std, and TTI = std(TT) / mean(TT)
    """
    n = D.shape[0]
    tt_values = []
    
    # Iterate over all triplets
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tt = D[i, j] + D[i, k] + D[j, k]
                tt_values.append(tt)
    
    if not tt_values:
        return {
            'triangle_thickness_mean': np.nan,
            'triangle_thickness_std': np.nan,
            'triangle_thickness_index': np.nan
        }
    
    tt_values = np.array(tt_values)
    mean_tt = float(np.mean(tt_values))
    std_tt = float(np.std(tt_values))
    tti = float(std_tt / mean_tt) if mean_tt > 0 else np.nan
    
    return {
        'triangle_thickness_mean': mean_tt,
        'triangle_thickness_std': std_tt,
        'triangle_thickness_index': tti
    }


def _fast_triad_score_vectorized(D1, D2):
    """
    Optimized vectorized version of triad_score.
    This is much faster than the nested loop version.
    """
    n = D1.shape[0]
    
    # For each i, compute all differences d(i,j) - d(i,k) for j != k
    # We can vectorize this by working with rows
    agrees = 0
    total = 0
    
    for i in range(n):
        # Get row i from both matrices
        row1 = D1[i, :]
        row2 = D2[i, :]
        
        # Create difference matrices: row1[j] - row1[k] for all j, k
        # Using broadcasting: row1[:, None] - row1[None, :]
        diff1 = row1[:, None] - row1[None, :]  # shape (n, n)
        diff2 = row2[:, None] - row2[None, :]  # shape (n, n)
        
        # We only want j < k (upper triangle, excluding diagonal)
        # Also exclude j == i or k == i
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask[i, :] = False  # Exclude j == i
        mask[:, i] = False  # Exclude k == i
        
        # Apply mask
        diff1_masked = diff1[mask]
        diff2_masked = diff2[mask]
        
        # Skip ties (where either difference is zero)
        non_tie_mask = (diff1_masked != 0) & (diff2_masked != 0)
        diff1_final = diff1_masked[non_tie_mask]
        diff2_final = diff2_masked[non_tie_mask]
        
        # Count agreements (same sign)
        signs_match = np.sign(diff1_final) == np.sign(diff2_final)
        agrees += np.sum(signs_match)
        total += len(diff1_final)
    
    return agrees / total if total > 0 else np.nan


def compare_matrices_programmatically(matrix1_path, matrix2_path, permutations=99, fast_mode=True):
    """
    Compare two distance matrices using dm_compare functionality.
    Optimized version with faster computation and configurable permutations.
    
    Parameters
    ----------
    matrix1_path : str or Path
        Path to first distance matrix file
    matrix2_path : str or Path
        Path to second distance matrix file
    permutations : int
        Number of permutations for statistical tests (default: 99 for faster computation)
        Original default was 999, but 99 is usually sufficient and much faster
    fast_mode : bool
        If True, use optimized algorithms and skip slow computations (default: True)
        
    Returns
    -------
    dict
        Dictionary with comparison metrics
    """
    # Import dm_compare functions
    from dm_compare import (
        parse_distance_file, align_matrices, scale_and_rmse
    )
    
    labels1, D1 = parse_distance_file(matrix1_path)
    labels2, D2 = parse_distance_file(matrix2_path)
    
    common, A, B = align_matrices(labels1, D1, labels2, D2)
    
    # Fast Mantel test with reduced permutations
    v1 = _upper_triangular_vector(A)
    v2 = _upper_triangular_vector(B)
    
    mantel_r = _fast_spearman_corr(v1, v2)
    
    # For p-value, use fewer permutations in fast mode
    if fast_mode and permutations > 99:
        pval_permutations = 99
    else:
        pval_permutations = permutations
    
    if pval_permutations > 0:
        rng = np.random.default_rng(42)
        n = A.shape[0]
        null_vals = np.empty(pval_permutations, dtype=float)
        for b in tqdm(range(pval_permutations), desc="Mantel test", leave=False, disable=not TQDM_AVAILABLE):
            perm = rng.permutation(n)
            Bp = B[np.ix_(perm, perm)]
            v2p = _upper_triangular_vector(Bp)
            null_vals[b] = _fast_spearman_corr(v1, v2p)
        more_extreme = np.sum(np.abs(null_vals) >= (0 if np.isnan(mantel_r) else abs(mantel_r)))
        mantel_p = (more_extreme + 1.0) / (pval_permutations + 1.0)
    else:
        mantel_p = np.nan
    
    # Fast Kendall tau (using scipy if available)
    if SCIPY_AVAILABLE:
        kendall_tau = stats.kendalltau(v1, v2)[0]
    else:
        # Fallback to manual calculation
        import pandas as pd
        from math import comb
        rx = pd.Series(v1).rank(method="average")
        ry = pd.Series(v2).rank(method="average")
        n = len(rx)
        concordant = discordant = 0
        for i in range(n-1):
            dx = rx.iloc[i+1:] - rx.iloc[i]
            dy = ry.iloc[i+1:] - ry.iloc[i]
            s = np.sign(dx * dy)
            concordant += np.sum(s > 0)
            discordant += np.sum(s < 0)
        denom = comb(n, 2)
        kendall_tau = (concordant - discordant) / denom if denom > 0 else np.nan
    
    # Fast triad concordance (vectorized)
    if fast_mode:
        triad_score = _fast_triad_score_vectorized(A, B)
        # Skip p-value computation for triad in fast mode (it's very slow)
        triad_p = np.nan
    else:
        # Use original method with reduced permutations
        from dm_compare import triad_concordance
        triads = triad_concordance(A, B, permutations=min(permutations, 99), random_state=42)
        triad_score = triads['score']
        triad_p = triads['p_value']
    
    alpha, beta, rmse = scale_and_rmse(A, B)
    
    # Calculate R-squared for linear model
    yhat = alpha + beta * v1
    ss_res = np.sum((v2 - yhat) ** 2)
    ss_tot = np.sum((v2 - np.mean(v2)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(v2 - v1))
    
    # Mean Absolute Percentage Error (MAPE) - avoiding division by zero
    mape = np.mean(np.abs((v2 - v1) / (v1 + 1e-10))) * 100  # Add small epsilon to avoid division by zero
    
    # Distance matrix statistics for both matrices
    def matrix_stats(D, prefix):
        v = _upper_triangular_vector(D)
        
        # Calculate mode (most frequent value, using binned histogram approach)
        # For continuous data, we use the mode of a histogram
        if len(v) > 0:
            # Use histogram to find mode bin
            hist, bin_edges = np.histogram(v, bins=min(50, len(np.unique(v))))
            mode_bin_idx = np.argmax(hist)
            # Mode is the center of the most frequent bin
            mode_value = float((bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2)
        else:
            mode_value = np.nan
        
        return {
            f'{prefix}_mean': float(np.mean(v)),
            f'{prefix}_std': float(np.std(v)),
            f'{prefix}_min': float(np.min(v)),
            f'{prefix}_max': float(np.max(v)),
            f'{prefix}_median': float(np.median(v)),
            f'{prefix}_mode': mode_value,
        }
    
    stats_A = matrix_stats(A, 'matrix1')
    stats_B = matrix_stats(B, 'matrix2')
    
    # Pearson correlation (in addition to Spearman)
    pearson_r = _pearson_corr(v1, v2) if not SCIPY_AVAILABLE else stats.pearsonr(v1, v2)[0]
    
    # Triangle inequality violations (phylogenetic metric)
    # A proper distance should satisfy: d(i,j) <= d(i,k) + d(k,j) for all i,j,k
    # This is O(n³) so we can skip it in fast mode for large matrices
    n = A.shape[0]
    if fast_mode and n > 50:
        # Skip triangle inequality check for large matrices in fast mode
        violations_A = np.nan
        violations_B = np.nan
        triangle_violation_rate_A = np.nan
        triangle_violation_rate_B = np.nan
    else:
        violations_A = 0
        violations_B = 0
        total_triangles = 0
        
        # Vectorized triangle inequality check (faster than nested loops)
        for i in range(n):
            for j in range(i+1, n):
                # Check all k: d(i,j) <= d(i,k) + d(k,j)
                # Vectorize over k
                d_ij_A = A[i, j]
                d_ij_B = B[i, j]
                
                # Get all d(i,k) and d(k,j) for k != i, j
                k_mask = np.ones(n, dtype=bool)
                k_mask[i] = False
                k_mask[j] = False
                k_indices = np.where(k_mask)[0]
                
                if len(k_indices) > 0:
                    d_ik_A = A[i, k_indices]
                    d_kj_A = A[k_indices, j]
                    d_ik_B = B[i, k_indices]
                    d_kj_B = B[k_indices, j]
                    
                    # Check: d(i,j) > d(i,k) + d(k,j) is a violation
                    violations_A += np.sum(d_ij_A > d_ik_A + d_kj_A + 1e-10)
                    violations_B += np.sum(d_ij_B > d_ik_B + d_kj_B + 1e-10)
                    total_triangles += len(k_indices)
        
        triangle_violation_rate_A = violations_A / total_triangles if total_triangles > 0 else 0.0
        triangle_violation_rate_B = violations_B / total_triangles if total_triangles > 0 else 0.0
    
    # Compute new tree/matrix property metrics for both matrices
    # These metrics analyze properties of each matrix independently
    
    # Ultrametric Deviation Score (UDS)
    # Skip for large matrices in fast mode (O(n³))
    if fast_mode and n > 50:
        uds_A = np.nan
        uds_B = np.nan
    else:
        uds_A = compute_ultrametric_deviation_score(A)
        uds_B = compute_ultrametric_deviation_score(B)
    
    # Distance spectrum shape (for both matrices)
    spectrum_A = compute_distance_spectrum_shape(A)
    spectrum_B = compute_distance_spectrum_shape(B)
    
    # Local compactness vs global spread (for both matrices)
    lcgs_A = compute_local_compactness_global_spread(A)
    lcgs_B = compute_local_compactness_global_spread(B)
    
    # Eigenvalue decay (for both matrices)
    eigen_A = compute_eigenvalue_decay(A)
    eigen_B = compute_eigenvalue_decay(B)
    
    # Triangle thickness index (for both matrices)
    # Skip for large matrices in fast mode (O(n³))
    if fast_mode and n > 50:
        tti_A = {'triangle_thickness_mean': np.nan, 'triangle_thickness_std': np.nan, 'triangle_thickness_index': np.nan}
        tti_B = {'triangle_thickness_mean': np.nan, 'triangle_thickness_std': np.nan, 'triangle_thickness_index': np.nan}
    else:
        tti_A = compute_triangle_thickness_index(A)
        tti_B = compute_triangle_thickness_index(B)
    
    # Build result dictionary with all metrics
    result = {
        'shared_leaves': len(common),
        'mantel_spearman_r': mantel_r,
        'mantel_spearman_p': mantel_p,
        'pearson_r': pearson_r,
        'kendall_tau': kendall_tau,
        'triad_concordance': triad_score,
        'triad_p': triad_p,
        'linear_alpha': alpha,
        'linear_beta': beta,
        'linear_rmse': rmse,
        'linear_r_squared': r_squared,
        'mae': mae,
        'mape_percent': mape,
        'triangle_violations_matrix1': violations_A,
        'triangle_violations_matrix2': violations_B,
        'triangle_violation_rate_matrix1': triangle_violation_rate_A,
        'triangle_violation_rate_matrix2': triangle_violation_rate_B,
        **stats_A,
        **stats_B,
        # New tree/matrix property metrics
        'uds_matrix1': uds_A,
        'uds_matrix2': uds_B,
        # Distance spectrum shape - matrix1
        'matrix1_skewness': spectrum_A['skewness'],
        'matrix1_kurtosis': spectrum_A['kurtosis'],
        'matrix1_bimodality_coefficient': spectrum_A['bimodality_coefficient'],
        'matrix1_coefficient_of_variation': spectrum_A['coefficient_of_variation'],
        # Distance spectrum shape - matrix2
        'matrix2_skewness': spectrum_B['skewness'],
        'matrix2_kurtosis': spectrum_B['kurtosis'],
        'matrix2_bimodality_coefficient': spectrum_B['bimodality_coefficient'],
        'matrix2_coefficient_of_variation': spectrum_B['coefficient_of_variation'],
        # Local compactness vs global spread - matrix1
        'matrix1_local_compactness_mean': lcgs_A['local_compactness_mean'],
        'matrix1_local_compactness_std': lcgs_A['local_compactness_std'],
        'matrix1_local_compactness_spread': lcgs_A['local_compactness_spread'],
        'matrix1_lcgs_ratio': lcgs_A['lcgs_ratio'],
        # Local compactness vs global spread - matrix2
        'matrix2_local_compactness_mean': lcgs_B['local_compactness_mean'],
        'matrix2_local_compactness_std': lcgs_B['local_compactness_std'],
        'matrix2_local_compactness_spread': lcgs_B['local_compactness_spread'],
        'matrix2_lcgs_ratio': lcgs_B['lcgs_ratio'],
        # Eigenvalue decay - matrix1
        'matrix1_eigenvalue_decay_rate': eigen_A['eigenvalue_decay_rate'],
        'matrix1_num_significant_eigenvalues': eigen_A['num_significant_eigenvalues'],
        'matrix1_eigenvalue_entropy': eigen_A['eigenvalue_entropy'],
        # Eigenvalue decay - matrix2
        'matrix2_eigenvalue_decay_rate': eigen_B['eigenvalue_decay_rate'],
        'matrix2_num_significant_eigenvalues': eigen_B['num_significant_eigenvalues'],
        'matrix2_eigenvalue_entropy': eigen_B['eigenvalue_entropy'],
        # Triangle thickness index - matrix1
        'matrix1_triangle_thickness_mean': tti_A['triangle_thickness_mean'],
        'matrix1_triangle_thickness_std': tti_A['triangle_thickness_std'],
        'matrix1_triangle_thickness_index': tti_A['triangle_thickness_index'],
        # Triangle thickness index - matrix2
        'matrix2_triangle_thickness_mean': tti_B['triangle_thickness_mean'],
        'matrix2_triangle_thickness_std': tti_B['triangle_thickness_std'],
        'matrix2_triangle_thickness_index': tti_B['triangle_thickness_index'],
    }
    
    return result
