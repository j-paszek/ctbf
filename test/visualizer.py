import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


TEST_VARIANTS = [
    "r2bss025",
    "r2bss05",
    "r2bss075",
    "r4bss05",
    "r4bss075",
    "r4bss05high",
    "r4bss05highdm",
]

METRIC_PAIRS = [
    ("3-f1", "nj"),
    ("3-f1", "rec"),
    ("grf", "nj"),
    ("grf", "rec"),
]


def load_ranking_table(test_variant: str, metric: str, mode: str) -> pd.DataFrame:
    filename = f"ranking_{test_variant}_{metric}_{mode}.csv"
    filename = os.path.join("results", filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find {filename}")

    df = pd.read_csv(filename, index_col=0)
    if "algorithm" in df.columns:
        df = df.set_index("algorithm")
    return df


def build_wins_minus_losses_matrix(test_variants, metric, mode, algorithms=None):
    tables = {}

    # If algorithm order not provided → use order from first CSV
    if algorithms is None:
        first_tv = test_variants[0]
        df0 = load_ranking_table(first_tv, metric, mode)
        algorithms = df0.index.tolist()


    mat = pd.DataFrame(index=test_variants, columns=algorithms, dtype=float)

    for tv in test_variants:
        df = load_ranking_table(tv, metric, mode)
        for alg in algorithms:
            if alg in df.index:
                mat.loc[tv, alg] = df.loc[alg, "wins"] - df.loc[alg, "losses"]
            else:
                mat.loc[tv, alg] = 0.0

    return mat, algorithms


def plot_stacked_heatmaps():
    """
    Loads ranking CSVs using the existing helper functions
    (load_ranking_table + build_wins_minus_losses_matrix)
    and generates a single stacked heatmap figure using
    _plot_stacked_heatmaps_core().
    """

    test_variants = TEST_VARIANTS
    # extract algorithm order from first CSV
    first_metric, first_mode = METRIC_PAIRS[0]
    df0 = load_ranking_table(test_variants[0], first_metric, first_mode)
    algorithms = df0.index.tolist()

    results_dict = {}

    # Build matrices
    for metric, mode in METRIC_PAIRS:
        mat, _ = build_wins_minus_losses_matrix(test_variants, metric, mode, algorithms)
        key = f"{metric}_{mode}"
        results_dict[key] = mat.values.astype(int)

    # Now call your original core plotting function
    _plot_stacked_heatmaps_core(
        results_dict,
        algorithms,
        test_variants,
        out_file="stacked_heatmaps.png"
    )

    print("Saved: stacked_heatmaps.png")


def _plot_stacked_heatmaps_core(results_dict, algorithms, test_variants, out_file):
    """
    Internal plotting backend.
    """
    fig = plt.figure(figsize=(22, 16))

    gs = gridspec.GridSpec(
        nrows=4, ncols=2,
        width_ratios=[1, 0.04],
        wspace=0.15,
        hspace=0.35
    )

    heatmap_titles = [
        ("3-F1 — NJ", "3-f1_nj"),
        ("3-F1 — Rec", "3-f1_rec"),
        ("GRF — NJ", "grf_nj"),
        ("GRF — Rec", "grf_rec"),
    ]

    for row, (title, key) in enumerate(heatmap_titles):
        ax = fig.add_subplot(gs[row, 0])
        cax = fig.add_subplot(gs[row, 1])

        short_labels = [alg.replace("neighbor_joining_", "") for alg in algorithms]

        sns.heatmap(
            results_dict[key],
            annot=True,
            fmt="d",
            cmap="coolwarm",
            center=0,
            xticklabels=short_labels if row == 3 else [],  # only bottom row
            yticklabels=test_variants,
            ax=ax,
            cbar_ax=cax
        )

        ax.set_title(title, fontsize=14)
        if row < 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Algorithms", fontsize=14)

    fig.suptitle("Algorithm Comparison Heatmaps (wins - losses)", fontsize=20)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    plot_stacked_heatmaps()
