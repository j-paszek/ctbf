from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

utils_dir = REPO_ROOT / "utils"
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

from dm_compare import parse_distance_file


PUBLISHED_RUN_ID = 47
PUBLISHED_PANEL_PARAMS = {
    "GENERAL_EVENT_PROB": 0.05,
    "GENERAL_DUPLICATION_PROB": 0.5,
    "GENERAL_DUPLICATION_MULTIPLICITY": 1,
    "GENERAL_LOSS_PROB": 0.5,
    "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": 0.5,
    "genome_length": 20,
}
PARAM_X = "GENERAL_DUPLICATION_MULTIPLICITY"
PARAM_Y = "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB"

METADATA_EXCLUDE_COLS = {
    "tree_file_path",
    "run_id",
    "repetition",
    "seed",
    "num_nodes",
    "num_leaves",
    "matrices_skipped",
    "skip_reason",
}
METADATA_EXCLUDE_PATTERNS = ("comparison_", "c2c_vs_", "naive_vs_", "medicc2_vs_")

METHOD_STYLES = {
    "GT": {"color": "green"},
    "cnp2cnp": {"color": "steelblue"},
    "Naive": {"color": "coral"},
    "MEDICC2": {"color": "purple"},
}


def default_results_dir() -> Path:
    return default_published_results_dir()


def default_published_results_dir() -> Path:
    return SCRIPT_DIR / "published_figure_3_data"


def default_generated_results_dir() -> Path:
    return SCRIPT_DIR / "results"


def default_figures_dir() -> Path:
    return SCRIPT_DIR / "figures"


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_metadata(results_dir: Path | str) -> pd.DataFrame:
    results_dir = Path(results_dir)
    metadata_path = results_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {results_dir}")
    return pd.read_csv(metadata_path)


def simulation_parameter_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in METADATA_EXCLUDE_COLS
        and not any(pattern in col for pattern in METADATA_EXCLUDE_PATTERNS)
    ]


def extract_upper_triangular(dist_matrix: np.ndarray) -> np.ndarray:
    return dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]


def get_matrix_paths(results_dir: Path | str, run_id: int) -> dict[str, Path]:
    results_dir = Path(results_dir)
    run_id_str = f"{run_id:03d}"
    return {
        "GT": results_dir / "matrix_gt" / f"matrix_gt_{run_id_str}.txt",
        "cnp2cnp": results_dir / "matrix_c2c" / f"matrix_c2c_{run_id_str}.txt",
        "Naive": results_dir / "matrix_other" / f"matrix_other_{run_id_str}.txt",
        "MEDICC2": results_dir / "matrix_medicc2" / f"matrix_medicc2_{run_id_str}.txt",
    }


def has_required_matrices(results_dir: Path | str, run_id: int, include_naive: bool = True) -> bool:
    matrix_paths = get_matrix_paths(results_dir, run_id)
    required = ["GT", "cnp2cnp", "MEDICC2"]
    if include_naive:
        required.append("Naive")
    return all(matrix_paths[name].exists() for name in required)


def published_panel_matches(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for key, value in PUBLISHED_PANEL_PARAMS.items():
        if key not in df.columns:
            raise RuntimeError(f"Required metadata column missing for published Figure 3 selection: {key}")
        mask &= df[key] == value
    return df.loc[mask].copy()


def published_panel_description() -> str:
    ordered_keys = [
        "GENERAL_EVENT_PROB",
        "GENERAL_DUPLICATION_PROB",
        "GENERAL_DUPLICATION_MULTIPLICITY",
        "GENERAL_LOSS_PROB",
        "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB",
        "genome_length",
    ]
    return ", ".join(f"{key}={PUBLISHED_PANEL_PARAMS[key]}" for key in ordered_keys)


def validate_preprint_results(results_dir: Path | str) -> pd.DataFrame:
    df = load_metadata(results_dir)
    param_cols = simulation_parameter_columns(df)
    duplicate_groups = (
        df.groupby(param_cols, dropna=False)
        .size()
        .reset_index(name="count")
    )
    duplicate_groups = duplicate_groups[duplicate_groups["count"] > 1]
    if not duplicate_groups.empty:
        preview = duplicate_groups.head(3).to_dict("records")
        raise RuntimeError(
            "Results directory is not preprint-compatible for exact Figure 3 reproduction. "
            "Found repeated runs for the same parameter combination. "
            "Regenerate with a single repetition (`--repetitions 1`). "
            f"Example duplicate groups: {preview}"
        )

    matches = published_panel_matches(df)
    if len(matches) != 1:
        match_ids = matches["run_id"].tolist()
        raise RuntimeError(
            "Results directory is not preprint-compatible for exact Figure 3 reproduction. "
            "Expected exactly one run matching the published panel A/B setup "
            f"({published_panel_description()}), found {len(matches)} matches: {match_ids}"
        )
    return df


def resolve_preprint_run_id(results_dir: Path | str, requested_run_id: int | None = None) -> int:
    if requested_run_id is not None:
        return requested_run_id

    df = load_metadata(results_dir)
    matches = published_panel_matches(df)
    candidate_ids = [
        int(run_id)
        for run_id in matches["run_id"].tolist()
        if has_required_matrices(results_dir, int(run_id), include_naive=True)
    ]

    if PUBLISHED_RUN_ID in candidate_ids:
        return PUBLISHED_RUN_ID
    if len(candidate_ids) == 1:
        return candidate_ids[0]
    if not candidate_ids:
        raise RuntimeError(
            "Could not resolve the published Figure 3 panel A/B run. "
            "No run matching the caption parameters has all required GT/cnp2cnp/Naive/MEDICC2 matrices. "
            f"Caption parameters: {published_panel_description()}"
        )
    raise RuntimeError(
        "Could not resolve a unique published Figure 3 panel A/B run. "
        f"Multiple matching runs with complete matrices were found: {candidate_ids}. "
        "Use `--run-id` to choose one explicitly, or regenerate with `--repetitions 1` "
        "to reproduce the preprint setup exactly."
    )


def load_distance_vectors(
    results_dir: Path | str,
    run_id: int,
    include_naive: bool = True,
) -> list[tuple[str, np.ndarray]]:
    matrix_paths = get_matrix_paths(results_dir, run_id)
    method_order = ["GT", "cnp2cnp", "MEDICC2"]
    if include_naive:
        method_order = ["GT", "cnp2cnp", "Naive", "MEDICC2"]

    vectors = []
    for method in method_order:
        path = matrix_paths[method]
        if not path.exists():
            raise FileNotFoundError(f"Required matrix missing for {method}: {path}")
        _, dist_matrix = parse_distance_file(str(path))
        vectors.append((method, extract_upper_triangular(dist_matrix)))

    return vectors


def humanize_parameter_name(param: str) -> str:
    return param.replace("_", " ").title()
