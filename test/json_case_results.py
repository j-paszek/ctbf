import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
DEFAULT_CASES_ROOT = Path(
    os.environ.get(
        "CTBF_ALGORITHM_CASES_ROOT",
        PROJECT_ROOT / "test" / "data" / "algorithm_cases",
    )
)
LEGACY_TEST_VARIANTS = [
    "r2bss025",
    "r2bss05",
    "r2bss075",
    "r4bss05",
    "r4bss075",
    "r4bss05high",
    "r4bss05highdm",
]
ADAPTIVE_RADIUS_TEST_VARIANTS = [
    "rAbss025",
    "rAbss05",
    "rAbss075",
    "rAbss05high",
    "rAbss05highdm",
]
TEST_VARIANTS = LEGACY_TEST_VARIANTS
ALL_TEST_VARIANTS = LEGACY_TEST_VARIANTS + ADAPTIVE_RADIUS_TEST_VARIANTS
MODES = ["full_cnp", "biopsy_guided_top"]
MODE_TO_HEATMAP_SUFFIX = {
    "full_cnp": "nj",
    "biopsy_guided_top": "rec",
}
METRIC_PATHS = {
    "3-precision": ("metrics", "ancestors_unique_restricted", "precision"),
    "3-f1": ("metrics", "ancestors_unique_restricted", "F1"),
    "grf": ("metrics", "grf"),
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def metric_value(result, metric):
    value = result
    for key in METRIC_PATHS[metric]:
        value = value[key]
    return float(value)


def variant_seed_dirs(cases_root=DEFAULT_CASES_ROOT, variant=None):
    variants = [variant] if variant is not None else TEST_VARIANTS
    for variant_name in variants:
        variant_dir = Path(cases_root) / variant_name
        if not variant_dir.exists():
            continue
        for seed_dir in sorted(
            [path for path in variant_dir.iterdir() if path.is_dir()],
            key=lambda path: int(path.name),
        ):
            if (seed_dir / "input.json").exists():
                yield variant_name, int(seed_dir.name), seed_dir


def algorithms_for_variant(cases_root, variant, mode):
    algorithms = set()
    for _, _, seed_dir in variant_seed_dirs(cases_root, variant):
        mode_dir = seed_dir / mode
        if not mode_dir.exists():
            continue
        algorithms.update(path.stem for path in mode_dir.glob("*.json"))
    return sorted(algorithms)


def load_result(cases_root, variant, seed, mode, algorithm):
    return load_json(Path(cases_root) / variant / str(seed) / mode / f"{algorithm}.json")


def metric_frame(cases_root, variant, mode, metric, algorithms=None):
    algorithms = algorithms or algorithms_for_variant(cases_root, variant, mode)
    rows = []
    for _, seed, seed_dir in variant_seed_dirs(cases_root, variant):
        row = {"seed": seed}
        for algorithm in algorithms:
            result_path = seed_dir / mode / f"{algorithm}.json"
            if result_path.exists():
                row[algorithm] = metric_value(load_json(result_path), metric)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _wilcoxon_summary(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    diffs = x - y
    if len(x) == 0 or np.allclose(diffs, 0):
        p_value = 1.0
    else:
        _, p_value = wilcoxon(x, y)
    return {
        "p_value": float(p_value),
        "alg1_mean": float(x.mean()) if len(x) else 0.0,
        "alg2_mean": float(y.mean()) if len(y) else 0.0,
    }


def decide_winner(summary, alpha=0.05):
    if summary["p_value"] > alpha:
        return "tie"
    if summary["alg1_mean"] > summary["alg2_mean"]:
        return "alg1"
    if summary["alg2_mean"] > summary["alg1_mean"]:
        return "alg2"
    return "tie"


def pairwise_ranking_from_json(cases_root, variant, mode, metric, algorithms=None, alpha=0.05):
    algorithms = algorithms or algorithms_for_variant(cases_root, variant, mode)
    table = pd.DataFrame(
        0,
        index=algorithms,
        columns=["wins", "losses", "ties", "score"],
        dtype=int,
    )
    frame = metric_frame(cases_root, variant, mode, metric, algorithms)

    for alg1, alg2 in combinations(algorithms, 2):
        paired = frame[["seed", alg1, alg2]].dropna()
        summary = _wilcoxon_summary(paired[alg1], paired[alg2])
        winner = decide_winner(summary, alpha=alpha)
        if winner == "alg1":
            table.loc[alg1, "wins"] += 1
            table.loc[alg2, "losses"] += 1
        elif winner == "alg2":
            table.loc[alg2, "wins"] += 1
            table.loc[alg1, "losses"] += 1
        else:
            table.loc[alg1, "ties"] += 1
            table.loc[alg2, "ties"] += 1

    table["score"] = table["wins"] - table["losses"]
    return table


def write_ranking_tables_from_json(cases_root, output_dir, variants=None, metrics=None, modes=None, alpha=0.05):
    variants = variants or TEST_VARIANTS
    metrics = metrics or list(METRIC_PATHS)
    modes = modes or MODES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for variant in variants:
        for mode in modes:
            algorithms = algorithms_for_variant(cases_root, variant, mode)
            suffix = MODE_TO_HEATMAP_SUFFIX[mode]
            for metric in metrics:
                ranking = pairwise_ranking_from_json(
                    cases_root,
                    variant,
                    mode,
                    metric,
                    algorithms=algorithms,
                    alpha=alpha,
                )
                path = output_dir / f"ranking_{variant}_{metric}_{suffix}.csv"
                ranking.to_csv(path)
                written.append(path)
    return written


def wins_minus_losses_matrix_from_json(cases_root, variants, mode, metric, algorithms=None, alpha=0.05):
    if algorithms is None:
        algorithms = algorithms_for_variant(cases_root, variants[0], mode)
    matrix = pd.DataFrame(index=variants, columns=algorithms, dtype=float)
    for variant in variants:
        ranking = pairwise_ranking_from_json(
            cases_root,
            variant,
            mode,
            metric,
            algorithms=algorithms,
            alpha=alpha,
        )
        for algorithm in algorithms:
            matrix.loc[variant, algorithm] = ranking.loc[algorithm, "score"] if algorithm in ranking.index else 0.0
    return matrix, algorithms
