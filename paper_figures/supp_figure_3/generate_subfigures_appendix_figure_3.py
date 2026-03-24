#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _shared import (
    default_figures_dir,
    default_generated_results_dir,
    default_published_results_dir,
    published_panel_description,
    resolve_preprint_run_id,
    validate_preprint_results,
)
from generate_figure_3_data import generate_data
from subfigure_a import create_subfigure_a
from subfigure_b import create_subfigure_b
from subfigure_c import create_subfigure_c


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all data and subfigures required for the Figure 3 study"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory. Defaults depend on --data-source.",
    )
    parser.add_argument("--figures-dir", type=str, default=str(default_figures_dir()))
    parser.add_argument(
        "--parameter-grid",
        type=str,
        default=str(SCRIPT_DIR / "figure_3_parameter_grid.json"),
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run ID override for panels A/B. By default, resolves the captioned preprint setup.",
    )
    parser.add_argument(
        "--data-source",
        choices=["frozen", "existing", "generate"],
        default="frozen",
        help=(
            "Where to get the data from: "
            "`frozen` uses the minimal published bundle, "
            "`existing` renders from an existing results directory, "
            "`generate` runs the pipeline first."
        ),
    )
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--cpu-fraction", type=float, default=0.6)
    parser.add_argument("--medicc2-path", type=str, default=None)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-cells-for-matrix", type=int, default=150)
    parser.add_argument("--max-leaves-to-evaluate", type=int, default=150)
    args = parser.parse_args()

    data_source = "existing" if args.skip_data else args.data_source
    if data_source == "frozen":
        results_dir = Path(args.results_dir) if args.results_dir is not None else default_published_results_dir()
    else:
        results_dir = Path(args.results_dir) if args.results_dir is not None else default_generated_results_dir()
    figures_dir = Path(args.figures_dir)

    if data_source == "generate":
        generate_data(
            results_dir=results_dir,
            parameter_grid_path=args.parameter_grid,
            repetitions=args.repetitions,
            base_seed=args.base_seed,
            max_workers=args.max_workers,
            cpu_fraction=args.cpu_fraction,
            medicc2_path=args.medicc2_path,
            max_nodes=args.max_nodes,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            max_cells_for_matrix=args.max_cells_for_matrix,
            max_leaves_to_evaluate=args.max_leaves_to_evaluate,
            reuse_existing=True,
        )
    else:
        if data_source == "frozen":
            print(f"Using frozen Figure 3 data bundle from {results_dir}")
        else:
            print(f"Using existing results from {results_dir}")
        validate_preprint_results(results_dir)

    resolved_run_id = resolve_preprint_run_id(results_dir, requested_run_id=args.run_id)
    print(
        "Rendering Figure 3 panels using panel A/B setup: "
        f"{published_panel_description()} (run_id={resolved_run_id})"
    )

    subfigure_a_path = create_subfigure_a(results_dir, figures_dir / "subfigure_a.png", run_id=resolved_run_id)
    subfigure_b_path = create_subfigure_b(results_dir, figures_dir / "subfigure_b.png", run_id=resolved_run_id)
    subfigure_c_path = create_subfigure_c(results_dir, figures_dir / "subfigure_c.png")

    print("Generated Figure 3 study subfigures:")
    print(f"  A: {subfigure_a_path}")
    print(f"  B: {subfigure_b_path}")
    print(f"  C: {subfigure_c_path}")


if __name__ == "__main__":
    main()
