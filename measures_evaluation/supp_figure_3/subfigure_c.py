#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _shared import (
    PARAM_X,
    PARAM_Y,
    default_figures_dir,
    default_results_dir,
    ensure_dir,
    humanize_parameter_name,
    load_metadata,
    validate_preprint_results,
)


def create_subfigure_c(results_dir: Path | str, output_path: Path | str) -> Path:
    validate_preprint_results(results_dir)
    df = load_metadata(results_dir)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    method_columns = [
        ("c2c_vs_gt_pearson_r", "cnp2cnp vs GT"),
        ("naive_vs_gt_pearson_r", "Naive vs GT"),
        ("medicc2_vs_gt_pearson_r", "MEDICC2 vs GT"),
    ]

    missing = [col for col, _ in method_columns if col not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing required columns in metadata.csv for subfigure C:\n"
            + "\n".join(missing)
        )

    pivots = []
    for col, _ in method_columns:
        pivots.append(
            df.pivot_table(
                values=col,
                index=PARAM_Y,
                columns=PARAM_X,
                aggfunc="mean",
            )
        )

    vmin = min(pivot.min().min() for pivot in pivots)
    vmax = max(pivot.max().max() for pivot in pivots)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2))

    for ax, pivot, (_, title) in zip(axes, pivots, method_columns):
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar_kws={"label": "Pearson Correlation"},
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_xlabel(humanize_parameter_name(PARAM_X), fontweight="bold", fontsize=14)
        ax.set_ylabel(humanize_parameter_name(PARAM_Y), fontweight="bold", fontsize=14)
        ax.set_title(title, fontweight="bold", fontsize=15)
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Appendix Figure 3 subfigure C")
    parser.add_argument("--results-dir", type=str, default=str(default_results_dir()))
    parser.add_argument("--output-path", type=str, default=str(default_figures_dir() / "subfigure_c.png"))
    args = parser.parse_args()

    output_path = create_subfigure_c(args.results_dir, args.output_path)
    print(f"Saved subfigure C to {output_path}")


if __name__ == "__main__":
    main()
