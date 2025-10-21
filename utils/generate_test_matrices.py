#!/usr/bin/env python3
"""
Generate synthetic distance matrices based on cnp_distance_matrix.txt
for testing dm_compare.py with different levels of similarity.
"""

import numpy as np
import argparse

def parse_distance_file(path):
    """Parse the original distance matrix file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    
    n = int(lines[0].split()[0])
    labels = []
    rows = []
    
    for ln in lines[1:1+n]:
        parts = ln.split()
        labels.append(parts[0])
        row = [float(x) for x in parts[1:n+1]]
        rows.append(row)
    
    D = np.array(rows, dtype=float)
    return labels, D, n

def make_symmetric(D):
    """Ensure matrix is symmetric."""
    return (D + D.T) / 2

def ensure_zero_diagonal(D):
    """Ensure diagonal is zero."""
    np.fill_diagonal(D, 0)
    return D

def ensure_non_negative(D):
    """Ensure all distances are non-negative."""
    return np.maximum(D, 0)

def generate_slightly_different(D, noise_level=0.5, random_state=42):
    """
    Generate a slightly different distance matrix by adding small random noise.
    
    Parameters:
    - noise_level: standard deviation of Gaussian noise to add
    - About 15-20% of entries will change by ±1
    """
    rng = np.random.default_rng(random_state)
    n = D.shape[0]
    
    # Add small Gaussian noise
    noise = rng.normal(0, noise_level, size=(n, n))
    D_new = D + noise
    
    # Make symmetric and enforce constraints
    D_new = make_symmetric(D_new)
    D_new = ensure_zero_diagonal(D_new)
    D_new = ensure_non_negative(D_new)
    
    # Round to integers to match original format
    D_new = np.round(D_new).astype(int)
    
    return D_new

def generate_more_different(D, noise_level=1.5, scale_factor=1.2, shift=0.3, random_state=42):
    """
    Generate a more different distance matrix with larger changes.
    
    Parameters:
    - noise_level: standard deviation of Gaussian noise
    - scale_factor: multiply some distances by this factor
    - shift: add systematic shift to some distances
    """
    rng = np.random.default_rng(random_state)
    n = D.shape[0]
    
    # Start with scaled version
    D_new = D * scale_factor
    
    # Add systematic shift to some entries
    shift_mask = rng.random((n, n)) > 0.6  # 40% of entries
    D_new = D_new + shift_mask * shift
    
    # Add larger random noise
    noise = rng.normal(0, noise_level, size=(n, n))
    D_new = D_new + noise
    
    # Introduce some more dramatic changes (swap some distance relationships)
    swap_prob = 0.1  # 10% chance to perturb relationships
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < swap_prob:
                # Add or subtract a larger amount
                perturbation = rng.choice([-2, -1, 1, 2])
                D_new[i, j] += perturbation
    
    # Make symmetric and enforce constraints
    D_new = make_symmetric(D_new)
    D_new = ensure_zero_diagonal(D_new)
    D_new = ensure_non_negative(D_new)
    
    # Round to integers
    D_new = np.round(D_new).astype(int)
    
    return D_new

def write_distance_matrix(labels, D, output_path):
    """Write distance matrix to file in the standard format."""
    n = len(labels)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{n}\n")
        for i, label in enumerate(labels):
            row_str = ' '.join(str(int(D[i, j])) for j in range(n))
            f.write(f"{label:<10} {row_str}\n")

def compute_difference_stats(D_orig, D_new):
    """Compute statistics about how different two matrices are."""
    n = D_orig.shape[0]
    
    # Get upper triangle (excluding diagonal)
    mask = np.triu_indices(n, k=1)
    orig_vals = D_orig[mask]
    new_vals = D_new[mask]
    
    # Compute differences
    diff = new_vals - orig_vals
    abs_diff = np.abs(diff)
    
    # Count changes
    changed = np.sum(abs_diff > 0)
    total = len(orig_vals)
    pct_changed = 100 * changed / total
    
    # Statistics
    mean_abs_diff = np.mean(abs_diff)
    max_abs_diff = np.max(abs_diff)
    
    # Correlation
    correlation = np.corrcoef(orig_vals, new_vals)[0, 1]
    
    return {
        'pct_changed': pct_changed,
        'mean_abs_diff': mean_abs_diff,
        'max_abs_diff': max_abs_diff,
        'correlation': correlation,
        'changed_entries': changed,
        'total_entries': total
    }

def main():
    parser = argparse.ArgumentParser(
        description="Generate test distance matrices with controlled differences"
    )
    parser.add_argument(
        '--input', 
        default='cnp_distance_matrix.txt',
        help='Input distance matrix file (default: cnp_distance_matrix.txt)'
    )
    parser.add_argument(
        '--output-slightly',
        default='cnp_distance_matrix_slightly_different.txt',
        help='Output file for slightly different matrix'
    )
    parser.add_argument(
        '--output-more',
        default='cnp_distance_matrix_more_different.txt',
        help='Output file for more different matrix'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()
    
    # Parse original matrix
    print(f"Reading original matrix from: {args.input}")
    labels, D_orig, n = parse_distance_file(args.input)
    print(f"Matrix size: {n}x{n}")
    print(f"Labels: {', '.join(labels[:5])}{'...' if n > 5 else ''}\n")
    
    # Generate slightly different matrix
    print("Generating slightly different matrix...")
    D_slight = generate_slightly_different(D_orig, noise_level=0.5, random_state=args.seed)
    write_distance_matrix(labels, D_slight, args.output_slightly)
    stats_slight = compute_difference_stats(D_orig, D_slight)
    
    print(f"Saved to: {args.output_slightly}")
    print(f"  Changed entries: {stats_slight['changed_entries']}/{stats_slight['total_entries']} ({stats_slight['pct_changed']:.1f}%)")
    print(f"  Mean absolute difference: {stats_slight['mean_abs_diff']:.3f}")
    print(f"  Max absolute difference: {stats_slight['max_abs_diff']:.0f}")
    print(f"  Correlation with original: {stats_slight['correlation']:.4f}\n")
    
    # Generate more different matrix
    print("Generating more different matrix...")
    D_more = generate_more_different(
        D_orig, 
        noise_level=1.5, 
        scale_factor=1.2, 
        shift=0.3,
        random_state=args.seed + 1
    )
    write_distance_matrix(labels, D_more, args.output_more)
    stats_more = compute_difference_stats(D_orig, D_more)
    
    print(f"Saved to: {args.output_more}")
    print(f"  Changed entries: {stats_more['changed_entries']}/{stats_more['total_entries']} ({stats_more['pct_changed']:.1f}%)")
    print(f"  Mean absolute difference: {stats_more['mean_abs_diff']:.3f}")
    print(f"  Max absolute difference: {stats_more['max_abs_diff']:.0f}")
    print(f"  Correlation with original: {stats_more['correlation']:.4f}\n")
    
    print("✅ Test matrices generated successfully!")
    print("\nYou can now test dm_compare.py with:")
    print(f"  python dm_compare.py {args.input} {args.output_slightly} --visualize")
    print(f"  python dm_compare.py {args.input} {args.output_more} --visualize")

if __name__ == '__main__':
    main()
