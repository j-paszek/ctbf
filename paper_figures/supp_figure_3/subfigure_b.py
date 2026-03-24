#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _shared import (
    METHOD_STYLES,
    default_figures_dir,
    default_results_dir,
    ensure_dir,
    load_distance_vectors,
    resolve_preprint_run_id,
)


def create_subfigure_b(
    results_dir: Path | str,
    output_path: Path | str,
    run_id: int | None = None,
) -> Path:
    run_id = resolve_preprint_run_id(results_dir, requested_run_id=run_id)
    distances = load_distance_vectors(results_dir, run_id=run_id, include_naive=False)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    all_vals = np.concatenate([dist for _, dist in distances])
    bins = np.linspace(all_vals.min(), all_vals.max(), 30)
    x_range = np.linspace(all_vals.min(), all_vals.max(), 200)

    for label, dist in distances:
        color = METHOD_STYLES[label]["color"]
        ax.hist(dist, bins=bins, alpha=0.5, edgecolor="black", color=color, label=label, density=True)
        kde = stats.gaussian_kde(dist)
        ax.plot(x_range, kde(x_range), color=color, linewidth=2, linestyle="--", alpha=0.8, label=f"{label} KDE")
        ax.axvline(np.mean(dist), color=color, linestyle=":", linewidth=2, alpha=0.8)

    ax.set_xlabel("Distance", fontweight="bold", fontsize=16)
    ax.set_ylabel("Density", fontweight="bold", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Appendix Figure 3 subfigure B")
    parser.add_argument("--results-dir", type=str, default=str(default_results_dir()))
    parser.add_argument("--output-path", type=str, default=str(default_figures_dir() / "subfigure_b.png"))
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run ID override. By default, resolves the captioned preprint parameter set.",
    )
    args = parser.parse_args()

    output_path = create_subfigure_b(args.results_dir, args.output_path, run_id=args.run_id)
    print(f"Saved subfigure B to {output_path}")


if __name__ == "__main__":
    main()
