#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

from compute_ground_truth_matrices import process_all_trees as compute_ground_truth
from compute_medicc2_gt_comparisons import process_all_trees as compute_medicc2_gt
from compute_medicc2_matrices import process_all_trees_medicc2
from grid_simulation import load_parameter_grid, run_grid_simulation

from _shared import (
    default_generated_results_dir,
    get_matrix_paths,
    resolve_preprint_run_id,
    validate_preprint_results,
)


def get_parallel_workers(fraction: float = 0.6) -> int:
    total_cores = multiprocessing.cpu_count()
    return max(1, int(total_cores * fraction))


def validate_required_outputs(results_dir: Path, run_id: int | None = None) -> None:
    validate_preprint_results(results_dir)
    run_id = resolve_preprint_run_id(results_dir, requested_run_id=run_id)
    matrix_paths = get_matrix_paths(results_dir, run_id)
    missing = [path for path in matrix_paths.values() if not path.exists()]
    if missing:
        missing_str = "\n".join(str(path) for path in missing)
        raise RuntimeError(
            "Required matrices for Figure 3 study are missing.\n"
            f"Expected run_id {run_id} to contain GT/cnp2cnp/Naive/MEDICC2 matrices.\n"
            f"Missing:\n{missing_str}"
        )

    metadata_path = results_dir / "metadata.csv"
    if not metadata_path.exists():
        raise RuntimeError(f"metadata.csv missing in {results_dir}")


def generate_data(
    results_dir: Path | str | None = None,
    parameter_grid_path: Path | str | None = None,
    repetitions: int = 1,
    base_seed: int = 42,
    max_workers: int | None = None,
    cpu_fraction: float = 0.6,
    medicc2_path: str | None = None,
    max_nodes: int = 1000,
    timeout_seconds: int = 60,
    max_retries: int = 3,
    max_cells_for_matrix: int = 150,
    max_leaves_to_evaluate: int = 150,
    reuse_existing: bool = True,
) -> Path:
    results_dir = Path(results_dir) if results_dir is not None else default_generated_results_dir()
    parameter_grid_path = (
        Path(parameter_grid_path)
        if parameter_grid_path is not None
        else SCRIPT_DIR / "figure_3_parameter_grid.json"
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        max_workers = get_parallel_workers(cpu_fraction)

    metadata_path = results_dir / "metadata.csv"
    if metadata_path.exists() and reuse_existing:
        validate_preprint_results(results_dir)
        print(f"Using existing results in {results_dir}")
    else:
        print(f"Generating simulation data in {results_dir}")
        parameter_grid = load_parameter_grid(parameter_grid_path)
        run_grid_simulation(
            parameter_grid,
            results_dir,
            repetitions=repetitions,
            base_seed=base_seed,
            param_grid_file=str(parameter_grid_path),
            max_workers=max_workers,
            max_nodes=max_nodes,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_cells_for_matrix=max_cells_for_matrix,
        )

    print("Computing ground truth matrices and GT comparison metrics...")
    compute_ground_truth(
        results_dir,
        update_metadata=True,
        skip_existing=True,
        recompute_comparisons=False,
        max_workers=max_workers,
    )

    print("Computing MEDICC2 matrices...")
    process_all_trees_medicc2(
        results_dir,
        medicc2_path=medicc2_path,
        skip_existing=True,
        update_metadata=False,
        max_leaves_to_evaluate=max_leaves_to_evaluate,
    )

    print("Computing MEDICC2 vs GT comparison metrics...")
    compute_medicc2_gt(
        results_dir,
        update_metadata=True,
        skip_existing=True,
        recompute_comparisons=False,
        max_workers=max_workers,
    )

    validate_required_outputs(results_dir)
    print(f"Figure 3 study data ready in {results_dir}")
    return results_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all data required for the Figure 3 study"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(default_generated_results_dir()),
        help="Directory where generated results will be stored",
    )
    parser.add_argument(
        "--parameter-grid",
        type=str,
        default=str(SCRIPT_DIR / "figure_3_parameter_grid.json"),
        help="Parameter grid JSON used for generation",
    )
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
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Ignore existing results and regenerate from scratch",
    )
    args = parser.parse_args()

    generate_data(
        results_dir=args.results_dir,
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
        reuse_existing=not args.force_regenerate,
    )


if __name__ == "__main__":
    main()
