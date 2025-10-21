
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Optional imports for visualization and advanced statistics
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def parse_distance_file(path):
    """
    Parse a distance matrix file with the format:
      - First line: integer N = number of leaves
      - Next N lines: "<label>  d_11 d_12 ... d_1N" (label followed by N numbers)
    Returns:
      labels: list of labels (as strings)
      D: (N,N) numpy array of distances (float)
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    try:
        n = int(lines[0].split()[0])
    except Exception as e:
        raise ValueError(f"Couldn't parse the first line as N from {path!r}: {e}")

    rows = []
    labels = []
    for ln in lines[1:1+n]:
        parts = ln.split()
        if len(parts) < n+1:
            raise ValueError(f"Line has {len(parts)} tokens but expected >= {n+1}: {ln!r}")
        labels.append(parts[0])
        row = [float(x) for x in parts[1:n+1]]
        rows.append(row)

    D = np.array(rows, dtype=float)
    if D.shape != (n, n):
        raise ValueError(f"Parsed matrix shape {D.shape} != ({n},{n})")

    return labels, D

def align_matrices(labels1, D1, labels2, D2):
    """
    Align two distance matrices on the intersection of labels.
    Returns:
      common_labels, D1a, D2a
    """
    idx1 = {lab:i for i,lab in enumerate(labels1)}
    idx2 = {lab:i for i,lab in enumerate(labels2)}
    common = sorted(set(labels1).intersection(labels2), key=lambda x: (str(x)))
    if len(common) < 3:
        raise ValueError(f"Need at least 3 shared labels to compare (got {len(common)}).")
    i1 = [idx1[lab] for lab in common]
    i2 = [idx2[lab] for lab in common]
    return common, D1[np.ix_(i1,i1)], D2[np.ix_(i2,i2)]

def _upper_triangular_vector(D):
    """Return the upper triangular (i<j) entries as a 1D vector."""
    return D[np.triu_indices(D.shape[0], k=1)]

def _spearman_rank_corr(x, y):
    """Compute Spearman rank correlation between two 1D arrays (no ties handling beyond average rank)."""
    # rank with average method for ties
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    # Pearson on ranks
    x_ = (rx - rx.mean())
    y_ = (ry - ry.mean())
    denom = np.sqrt((x_**2).sum() * (y_**2).sum())
    if denom == 0:
        return np.nan
    return float((x_ * y_).sum() / denom)

def _pearson_corr(x, y):
    x_ = (x - np.mean(x))
    y_ = (y - np.mean(y))
    denom = np.sqrt((x_**2).sum() * (y_**2).sum())
    if denom == 0:
        return np.nan
    return float((x_ * y_).sum() / denom)

def mantel_test(D1, D2, method="spearman", permutations=999, random_state=42):
    """
    Mantel test between two distance matrices.
    - method: 'spearman' or 'pearson'
    - permutations: number of label permutations for p-value
    Returns dict with: stat, p_value, method, permutations
    """
    assert D1.shape == D2.shape and D1.shape[0] == D1.shape[1]
    rng = np.random.default_rng(random_state)
    v1 = _upper_triangular_vector(D1)
    v2 = _upper_triangular_vector(D2)

    corr_fn = _spearman_rank_corr if method == "spearman" else _pearson_corr
    obs = corr_fn(v1, v2)

    # Permute labels of D2 (simultaneously rows/cols)
    n = D1.shape[0]
    null_vals = np.empty(permutations, dtype=float)
    for b in range(permutations):
        perm = rng.permutation(n)
        D2p = D2[np.ix_(perm, perm)]
        v2p = _upper_triangular_vector(D2p)
        null_vals[b] = corr_fn(v1, v2p)
    # two-sided p-value
    more_extreme = np.sum(np.abs(null_vals) >= (0 if np.isnan(obs) else abs(obs)))
    pval = (more_extreme + 1.0) / (permutations + 1.0)
    return {"stat": obs, "p_value": pval, "method": method, "permutations": permutations}

def kendall_tau_on_pairs(D1, D2):
    """
    Kendall correlation between pairwise distances (over unordered pairs i<j).
    (This is equivalent to Kendall tau between the upper-triangle vectors.)
    """
    x = _upper_triangular_vector(D1)
    y = _upper_triangular_vector(D2)
    # Handle ties using pandas
    rx = pd.Series(x).rank(method="average")
    ry = pd.Series(y).rank(method="average")
    # Compute tau-a (ignoring ties), which is robust and simple
    from math import comb
    n = len(rx)
    concordant = discordant = 0
    for i in range(n-1):
        dx = rx.iloc[i+1:] - rx.iloc[i]
        dy = ry.iloc[i+1:] - ry.iloc[i]
        s = np.sign(dx * dy)
        concordant += np.sum(s > 0)
        discordant += np.sum(s < 0)
    denom = comb(n, 2)
    return (concordant - discordant) / denom if denom > 0 else np.nan

def triad_concordance(D1, D2, permutations=999, random_state=42):
    """
    For each triple (i,j,k), compare the ordering of d(i,j) vs d(i,k) in both matrices.
    Score = fraction of non-tied triads where the sign matches.
    """
    n = D1.shape[0]
    rng = np.random.default_rng(random_state)

    def triad_score(Da, Db):
        agrees = 0
        total = 0
        for i in range(n):
            for j in range(n):
                if j == i: 
                    continue
                for k in range(j+1, n):
                    if k == i: 
                        continue
                    a = Da[i, j] - Da[i, k]
                    b = Db[i, j] - Db[i, k]
                    if a == 0 or b == 0:
                        continue  # skip ties
                    total += 1
                    if np.sign(a) == np.sign(b):
                        agrees += 1
        return agrees / total if total > 0 else np.nan

    obs = triad_score(D1, D2)
    null_vals = np.empty(permutations, dtype=float)
    idx = np.arange(n)
    for b in range(permutations):
        perm = rng.permutation(idx)
        D2p = D2[np.ix_(perm, perm)]
        null_vals[b] = triad_score(D1, D2p)
    if np.isnan(obs):
        pval = np.nan
    else:
        ge = np.sum(null_vals >= obs)
        pval = (ge + 1.0) / (permutations + 1.0)
    return {"score": obs, "p_value": pval, "permutations": permutations}

def scale_and_rmse(Dref, D):
    """
    Fit a linear mapping D ~ alpha + beta * Dref (least squares on upper triangle).
    Returns alpha, beta, RMSE (on upper triangle).
    """
    x = _upper_triangular_vector(Dref)
    y = _upper_triangular_vector(D)
    X = np.vstack([np.ones_like(x), x]).T
    beta_hat, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = beta_hat[0], beta_hat[1]
    yhat = alpha + beta * x
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    return float(alpha), float(beta), rmse

def visualize_linear_model(Dref, D, alpha, beta, rmse, output_dir=".", prefix="dm_comparison"):
    """
    Create visualizations for the linear model D ~ alpha + beta * Dref.
    Saves plots to files showing the relationship and model diagnostics.
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib and/or seaborn not available. Skipping visualization.")
        print("To enable visualization, install: pip install matplotlib seaborn")
        return
    
    x = _upper_triangular_vector(Dref)
    y = _upper_triangular_vector(D)
    yhat = alpha + beta * x
    residuals = y - yhat
    
    # Set up the plotting style
    plt.style.use('default')
    if 'sns' in globals():
        sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distance Matrix Comparison: Linear Model Analysis', fontsize=14, fontweight='bold')
    
    # 1. Scatter plot with fitted line
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.6, s=20, color='steelblue', label='Data points')
    ax1.plot(x, yhat, 'r-', linewidth=2, label=f'Fit: y = {alpha:.3f} + {beta:.3f}x')
    ax1.set_xlabel('Reference Matrix Distances')
    ax1.set_ylabel('Comparison Matrix Distances')
    ax1.set_title('Pairwise Distance Relationship')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add R² calculation
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    ax1.text(0.05, 0.95, f'R² = {r_squared:.4f}\nRMSE = {rmse:.4f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Residuals vs fitted values
    ax2 = axes[0, 1]
    ax2.scatter(yhat, residuals, alpha=0.6, s=20, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Fitted Values')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram of residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)
    
    # Add normal distribution overlay for comparison
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = len(residuals) * (residuals.max() - residuals.min()) / 30 * \
             (1 / (residual_std * np.sqrt(2 * np.pi))) * \
             np.exp(-0.5 * ((x_norm - residual_mean) / residual_std) ** 2)
    ax3.plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal fit')
    ax3.legend()
    
    # 4. Q-Q plot for residuals normality check
    ax4 = axes[1, 1]
    if SCIPY_AVAILABLE:
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot: Residuals vs Normal')
        ax4.grid(True, alpha=0.3)
    else:
        # Simple Q-Q plot fallback without scipy
        sorted_residuals = np.sort(residuals)
        n = len(sorted_residuals)
        # Approximate normal quantiles
        theoretical_quantiles = np.sqrt(2) * np.sqrt(-np.log(1 - (np.arange(1, n+1) - 0.5) / n))
        theoretical_quantiles = np.where(np.arange(1, n+1) <= n/2, -theoretical_quantiles[::-1], theoretical_quantiles)
        ax4.scatter(theoretical_quantiles, sorted_residuals, alpha=0.7, color='orange')
        ax4.plot(theoretical_quantiles, theoretical_quantiles * residual_std + residual_mean, 'r-')
        ax4.set_xlabel('Theoretical Quantiles')
        ax4.set_ylabel('Sample Quantiles')
        ax4.set_title('Q-Q Plot: Residuals vs Normal (approx)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plot_filename = output_path / f"{prefix}_linear_model_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Linear model visualization saved to: {plot_filename}")
    
    # Create a second figure for distance matrix heatmaps
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle('Distance Matrix Heatmaps', fontsize=14, fontweight='bold')
    
    # Reference matrix
    im1 = axes2[0].imshow(Dref, cmap='viridis', aspect='auto')
    axes2[0].set_title('Reference Matrix')
    axes2[0].set_xlabel('Leaf Index')
    axes2[0].set_ylabel('Leaf Index')
    plt.colorbar(im1, ax=axes2[0], shrink=0.6)
    
    # Comparison matrix
    im2 = axes2[1].imshow(D, cmap='viridis', aspect='auto')
    axes2[1].set_title('Comparison Matrix')
    axes2[1].set_xlabel('Leaf Index')
    axes2[1].set_ylabel('Leaf Index')
    plt.colorbar(im2, ax=axes2[1], shrink=0.6)
    
    # Difference matrix
    diff_matrix = D - (alpha + beta * Dref)
    im3 = axes2[2].imshow(diff_matrix, cmap='RdBu_r', aspect='auto')
    axes2[2].set_title('Residual Matrix\n(Comparison - Fitted)')
    axes2[2].set_xlabel('Leaf Index')
    axes2[2].set_ylabel('Leaf Index')
    plt.colorbar(im3, ax=axes2[2], shrink=0.6)
    
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_filename = output_path / f"{prefix}_distance_matrices_heatmap.png"
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    print(f"Distance matrix heatmaps saved to: {heatmap_filename}")
    
    plt.show()
    
    # Print summary statistics
    print("\n# Linear Model Diagnostics")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Residual standard error: {np.std(residuals):.4f}")
    print(f"Mean absolute error: {np.mean(np.abs(residuals)):.4f}")
    print(f"Max absolute residual: {np.max(np.abs(residuals)):.4f}")
    
    return {
        "r_squared": r_squared,
        "residual_std": np.std(residuals),
        "mae": np.mean(np.abs(residuals)),
        "max_abs_residual": np.max(np.abs(residuals))
    }

def sanity_checks(D):
    if D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix is not square.")
    if not np.allclose(D, D.T):
        raise ValueError("Distance matrix is not symmetric.")
    if not np.allclose(np.diag(D), 0):
        raise ValueError("Distance matrix diagonal must be zero (self-distances).")

def summarize(labels, D, name):
    n = len(labels)
    v = D[np.triu_indices(n, k=1)]
    return {
        "name": name,
        "n_leaves": n,
        "min": float(np.min(v)),
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "max": float(np.max(v)),
    }

def main():
    ap = argparse.ArgumentParser(description="Compare two distance matrices (evolutionary trees).")
    ap.add_argument("file1", type=str, help="Path to first distance-matrix file")
    ap.add_argument("file2", type=str, help="Path to second distance-matrix file")
    ap.add_argument("--permutations", type=int, default=999, help="Permutations for p-values (default: 999)")
    ap.add_argument("--mantel", choices=["spearman","pearson"], default="spearman", help="Correlation type for Mantel test")
    ap.add_argument("--visualize", action="store_true", help="Create visualizations of the linear model analysis")
    ap.add_argument("--output-dir", type=str, default=".", help="Directory to save visualization plots (default: current directory)")
    ap.add_argument("--plot-prefix", type=str, default="dm_comparison", help="Prefix for saved plot filenames")
    args = ap.parse_args()

    labels1, D1 = parse_distance_file(args.file1)
    labels2, D2 = parse_distance_file(args.file2)

    sanity_checks(D1)
    sanity_checks(D2)

    common, A, B = align_matrices(labels1, D1, labels2, D2)

    s1 = summarize(common, A, "Matrix 1 (aligned)")
    s2 = summarize(common, B, "Matrix 2 (aligned)")

    mantel = mantel_test(A, B, method=args.mantel, permutations=args.permutations, random_state=42)
    tau_pairs = kendall_tau_on_pairs(A, B)
    triads = triad_concordance(A, B, permutations=args.permutations, random_state=42)
    alpha, beta, rmse = scale_and_rmse(A, B)

    print("# Alignment")
    print(f"Shared leaves: {len(common)}")
    print("Labels (first 10):", ", ".join(map(str, common[:10])) + (" ..." if len(common)>10 else ""))
    print()
    print("# Summaries (upper triangle)")
    for s in (s1, s2):
        print(f"{s['name']}: n={s['n_leaves']}  min={s['min']:.3f}  median={s['median']:.3f}  mean={s['mean']:.3f}  max={s['max']:.3f}")
    print()
    print("# Similarity metrics")
    print(f"Mantel ({mantel['method']}): r={mantel['stat']:.4f}  p={mantel['p_value']:.4g}  (permutations={mantel['permutations']})")
    print(f"Kendall tau on pairwise distances (tau-a approx): {tau_pairs:.4f}  (no p-value computed)")
    print(f"Triad-order concordance: score={triads['score']:.4f}  p={triads['p_value']:.4g}  (permutations={triads['permutations']})")
    print()
    print("# Scale compatibility (B ~ alpha + beta*A)")
    print(f"alpha={alpha:.4f}, beta={beta:.4f}, RMSE={rmse:.4f}")
    print()
    
    # Create visualizations if requested
    if args.visualize:
        model_diagnostics = visualize_linear_model(A, B, alpha, beta, rmse, 
                                                 output_dir=args.output_dir, 
                                                 prefix=args.plot_prefix)
    
    print("Interpretation notes:")
    print(" - Mantel r close to 1 with small p suggests the two matrices encode very similar relative distances.")
    print(" - Triad concordance near 1 with small p indicates that, for most triples, the 'which is farther' relationship agrees.")
    print(" - The linear fit (alpha, beta) shows if one matrix is a scaled/shifted version of the other (expect alpha≈0, beta>0).")
    if args.visualize:
        print(" - Visualizations have been saved to help interpret the linear model fit and residuals.")

if __name__ == '__main__':
    main()
