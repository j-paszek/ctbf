import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

TOOLS_DIR = Path(__file__).resolve().parent
TEST_DIR = TOOLS_DIR.parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from json_case_results import (  # noqa: E402
    DEFAULT_CASES_ROOT,
    TEST_VARIANTS,
    algorithms_for_variant,
    wins_minus_losses_matrix_from_json,
    write_ranking_tables_from_json,
)
from reconstructor_algorithm_config import (  # noqa: E402
    HIGHLIGHTED_HEATMAP_ALGORITHMS,
    algorithm_label,
    resolve_comparison_algorithm_names,
)


HEATMAP_SPECS = [
    ("3-f1", "full_cnp", "AD-F1 - Pairwise NJ-like inference (wins-losses)", "3-f1_nj"),
    ("3-f1", "biopsy_guided_top", "AD-F1 - Biopsy guided inference (wins-losses)", "3-f1_rec"),
    ("grf", "full_cnp", "GRF - Pairwise NJ-like inference (wins-losses)", "grf_nj"),
    ("grf", "biopsy_guided_top", "GRF - Biopsy guided inference (wins-losses)", "grf_rec"),
]


def _algorithms_for_mode(cases_root, variants, mode, selected_algorithms=None):
    available = algorithms_for_variant(cases_root, variants[0], mode)
    if selected_algorithms is None:
        return available
    return [algorithm for algorithm in selected_algorithms if algorithm in available]


def _style_algorithm_tick_labels(ax, algorithms):
    highlighted = set(HIGHLIGHTED_HEATMAP_ALGORITHMS)
    for tick_label, algorithm in zip(ax.get_yticklabels(), algorithms):
        if algorithm in highlighted:
            tick_label.set_fontweight("bold")


def plot_side_by_side_heatmaps_from_json(
    cases_root,
    variants,
    output_file,
    alpha=0.05,
    selected_algorithms=None,
):
    matrices = {}
    algorithms_by_key = {}
    for metric, mode, _, key in HEATMAP_SPECS:
        algorithms = _algorithms_for_mode(cases_root, variants, mode, selected_algorithms)
        matrix, algorithms = wins_minus_losses_matrix_from_json(
            cases_root,
            variants,
            mode,
            metric,
            algorithms=algorithms,
            alpha=alpha,
        )
        matrices[key] = matrix
        algorithms_by_key[key] = algorithms

    fig = plt.figure(figsize=(28, 16))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[1, 1, 0.05],
        hspace=0.25,
        wspace=0.20,
        figure=fig,
    )
    cbar_ax = fig.add_subplot(gs[:, 2])

    for idx, (_, _, title, key) in enumerate(HEATMAP_SPECS):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        algorithms = algorithms_by_key[key]
        short_algs = [algorithm_label(algorithm) for algorithm in algorithms]
        matrix = matrices[key].loc[variants, algorithms].T.astype(float)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="coolwarm",
            center=0,
            xticklabels=variants,
            yticklabels=short_algs,
            ax=ax,
            cbar=(idx == len(HEATMAP_SPECS) - 1),
            cbar_ax=cbar_ax if idx == len(HEATMAP_SPECS) - 1 else None,
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Test variants")
        ax.set_ylabel("Algorithms")
        _style_algorithm_tick_labels(ax, algorithms)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate heatmaps_side_by_side from frozen JSON fixtures.")
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    parser.add_argument("--rankings-dir", type=Path, default=PROJECT_ROOT / "test" / "data" / "results_from_json")
    parser.add_argument("--output-file", type=Path, default=PROJECT_ROOT / "test" / "heatmaps_side_by_side_from_json.png")
    parser.add_argument("--variant", action="append", choices=TEST_VARIANTS,
                        help="Variant to include. Can be passed multiple times. Defaults to all variants.")
    parser.add_argument("--algorithm-group", action="append",
                        help="Comparison group from reconstructor_algorithm_config.py. Can be passed multiple times.")
    parser.add_argument("--algorithm-name", action="append",
                        help="Explicit algorithm/result row to include. Can be passed multiple times.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--skip-ranking-csv", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = args.variant or TEST_VARIANTS
    selected_algorithms = resolve_comparison_algorithm_names(args.algorithm_group, args.algorithm_name)
    if not args.skip_ranking_csv:
        written = write_ranking_tables_from_json(
            args.cases_root,
            args.rankings_dir,
            variants=variants,
            alpha=args.alpha,
        )
        print(f"Wrote {len(written)} ranking CSV files to {args.rankings_dir}")
    output_file = plot_side_by_side_heatmaps_from_json(
        args.cases_root,
        variants,
        args.output_file,
        alpha=args.alpha,
        selected_algorithms=selected_algorithms,
    )
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
