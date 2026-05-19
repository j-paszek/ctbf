#!/usr/bin/env python
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
import traceback

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
TEST_DIR = TOOLS_DIR.parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from freeze_algorithm_variant_cases import (  # noqa: E402
    ADAPTIVE_RADIUS_GRID,
    ADAPTIVE_RADIUS_VARIANT_NAMES,
    DEFAULT_OUTPUT_ROOT,
    VARIANT_PRESETS,
    existing_seeds_for_variant,
    get_algorithms_to_test,
    input_path,
    load_json,
    mean_pairwise_input_distance,
    resolve_reference_algorithm_indices,
    result_path,
    source_variant_name,
    write_seed_case,
)


DEFAULT_GRID_ALGORITHM_NAMES = [
    "neighbor_joining_baseline",
    "neighbor_joining_hybrid_opt",
    "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
]


def alpha_token(alpha):
    return f"{float(alpha):g}".replace("-", "m").replace(".", "p")


def adaptive_cases_root(output_root, alpha):
    return Path(output_root) / f"alpha_{alpha_token(alpha)}" / "cases"


def metric_values(result):
    return {
        "ad_f1": float(result["metrics"]["ancestors_unique_restricted"]["F1"]),
        "ad_precision": float(result["metrics"]["ancestors_unique_restricted"]["precision"]),
        "grf": float(result["metrics"]["grf"]),
    }


def case_radius_row(input_case):
    matrix = np.asarray(input_case["distance_matrices"]["cnp2cnp"]["matrix"], dtype=float)
    d_mean = mean_pairwise_input_distance(matrix)
    r_dist = float(input_case["r_dist"])
    return {
        "r_dist": r_dist,
        "d_mean": d_mean,
        "r_over_d": r_dist / d_mean if d_mean else 0.0,
    }


def seed_values(source_cases_root, variant_name, explicit_seeds, limit):
    seeds = explicit_seeds if explicit_seeds is not None else existing_seeds_for_variant(source_cases_root, variant_name)
    if limit is not None:
        seeds = seeds[:limit]
    return list(seeds)


def resolve_algorithms(algorithm_names, algorithm_indexes):
    names = algorithm_names
    if names is None and algorithm_indexes is not None:
        names = []
    indexes = resolve_reference_algorithm_indices(
        algorithm_indexes=algorithm_indexes,
        algorithm_names=names,
    )
    algorithms = get_algorithms_to_test()
    return [algorithms[index] for index in indexes]


def collect_result_row(
    *,
    cases_root,
    variant,
    seed,
    mode,
    algorithm_name,
    radius_mode,
    alpha,
    source_variant,
):
    current_input_path = input_path(cases_root, variant, seed)
    current_result_path = result_path(cases_root, variant, seed, mode, algorithm_name)
    if not current_input_path.exists() or not current_result_path.exists():
        return None

    input_case = load_json(current_input_path)
    result = load_json(current_result_path)
    row = {
        "radius_mode": radius_mode,
        "alpha": alpha,
        "variant": variant,
        "source_variant": source_variant,
        "seed": seed,
        "mode": mode,
        "algorithm": algorithm_name,
    }
    row.update(case_radius_row(input_case))
    row.update(metric_values(result))
    return row


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["radius_mode"],
            row["alpha"],
            row["variant"],
            row["source_variant"],
            row["mode"],
            row["algorithm"],
        )
        groups[key].append(row)

    aggregate = []
    for key, group_rows in groups.items():
        radius_mode, alpha, variant, source_variant, mode, algorithm = key
        aggregate.append({
            "radius_mode": radius_mode,
            "alpha": alpha,
            "variant": variant,
            "source_variant": source_variant,
            "mode": mode,
            "algorithm": algorithm,
            "cases": len(group_rows),
            "mean_r_dist": mean(row["r_dist"] for row in group_rows),
            "mean_d": mean(row["d_mean"] for row in group_rows),
            "mean_r_over_d": mean(row["r_over_d"] for row in group_rows),
            "mean_ad_f1": mean(row["ad_f1"] for row in group_rows),
            "mean_ad_precision": mean(row["ad_precision"] for row in group_rows),
            "mean_grf": mean(row["grf"] for row in group_rows),
        })
    return sorted(
        aggregate,
        key=lambda row: (
            row["radius_mode"],
            str(row["alpha"]),
            row["variant"],
            row["mode"],
            row["algorithm"],
        ),
    )


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    if value == "":
        return ""
    return f"{float(value):.6f}"


def summary_markdown(aggregate_rows, *, source_cases_root, output_root, alphas, variants, limit):
    lines = [
        "# Adaptive Radius Grid",
        "",
        f"- Source fixed cases root: `{source_cases_root}`",
        f"- Scratch output root: `{output_root}`",
        f"- Adaptive formula: `rA = alpha * d`",
        "- `d` is the mean pairwise cnp2cnp input distance for each frozen case.",
        f"- Alpha grid: `{', '.join(str(alpha) for alpha in alphas)}`",
        f"- Variants: `{', '.join(variants)}`",
        f"- Limit per variant: `{limit if limit is not None else 'all source seeds'}`",
        "",
        "## Biopsy-Guided AD-F1",
        "",
        "| radius mode | alpha | variant | source variant | algorithm | cases | mean r | mean d | mean r/d | mean AD-F1 | mean GRF |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        if row["mode"] != "biopsy_guided_top":
            continue
        lines.append(
            "| {radius_mode} | {alpha} | {variant} | {source_variant} | {algorithm} | {cases} | {mean_r} | {mean_d} | {mean_r_over_d} | {mean_ad_f1} | {mean_grf} |".format(
                radius_mode=row["radius_mode"],
                alpha=row["alpha"],
                variant=row["variant"],
                source_variant=row["source_variant"],
                algorithm=row["algorithm"],
                cases=row["cases"],
                mean_r=format_float(row["mean_r_dist"]),
                mean_d=format_float(row["mean_d"]),
                mean_r_over_d=format_float(row["mean_r_over_d"]),
                mean_ad_f1=format_float(row["mean_ad_f1"]),
                mean_grf=format_float(row["mean_grf"]),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Derive adaptive-radius frozen cases from fixed frozen inputs and run "
            "a small alpha grid in scratch outputs."
        )
    )
    parser.add_argument("--source-cases-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        choices=ADAPTIVE_RADIUS_VARIANT_NAMES,
        help="Adaptive variant to run. Can be passed multiple times. Defaults to all adaptive variants.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        action="append",
        default=None,
        help="Adaptive scale alpha. Defaults to the built-in small grid.",
    )
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit seeds per adaptive variant.")
    parser.add_argument("--algorithm-name", action="append", default=None)
    parser.add_argument("--algorithm-index", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = args.variant or ADAPTIVE_RADIUS_VARIANT_NAMES
    alphas = args.alpha or list(ADAPTIVE_RADIUS_GRID)
    if args.algorithm_name is None and args.algorithm_index is None:
        algorithm_names = DEFAULT_GRID_ALGORITHM_NAMES
    else:
        algorithm_names = args.algorithm_name
    algorithms = resolve_algorithms(algorithm_names, args.algorithm_index)

    plan_lines = [
        f"Source cases root: {args.source_cases_root}",
        f"Output root: {args.output_root}",
        "Variants: " + ", ".join(variants),
        "Alphas: " + ", ".join(str(alpha) for alpha in alphas),
        "Algorithms: " + ", ".join(getattr(algorithm, "__name__", str(algorithm)) for algorithm in algorithms),
    ]
    print("\n".join(plan_lines))
    if args.dry_run:
        return 0

    failures = []
    detail_rows = []
    fixed_seen = set()
    for alpha in alphas:
        cases_root = adaptive_cases_root(args.output_root, alpha)
        for variant_name in variants:
            variant = VARIANT_PRESETS[variant_name]
            fixed_variant = source_variant_name(variant_name, variant)
            seeds = seed_values(args.source_cases_root, fixed_variant, args.seed, args.limit)
            for seed in seeds:
                try:
                    write_seed_case(
                        cases_root,
                        variant_name,
                        variant,
                        seed,
                        algorithms,
                        args.overwrite,
                        input_only=False,
                        results_only=False,
                        skip_existing=args.skip_existing,
                        source_cases_root=args.source_cases_root,
                        adaptive_radius_scale=alpha,
                    )
                except Exception as exc:
                    failures.append({
                        "alpha": alpha,
                        "variant": variant_name,
                        "seed": seed,
                        "error": str(exc),
                    })
                    print(f"Failed alpha={alpha} {variant_name}/{seed}: {exc}", file=sys.stderr)
                    traceback.print_exc()
                    if args.fail_fast:
                        raise

                for algorithm in algorithms:
                    algorithm_name = getattr(algorithm, "__name__", str(algorithm))
                    adaptive_row = collect_result_row(
                        cases_root=cases_root,
                        variant=variant_name,
                        seed=seed,
                        mode="biopsy_guided_top",
                        algorithm_name=algorithm_name,
                        radius_mode="adaptive",
                        alpha=alpha,
                        source_variant=fixed_variant,
                    )
                    if adaptive_row is not None:
                        detail_rows.append(adaptive_row)
                    fixed_key = (variant_name, fixed_variant, seed, algorithm_name)
                    if fixed_key not in fixed_seen:
                        fixed_seen.add(fixed_key)
                        fixed_row = collect_result_row(
                            cases_root=args.source_cases_root,
                            variant=fixed_variant,
                            seed=seed,
                            mode="biopsy_guided_top",
                            algorithm_name=algorithm_name,
                            radius_mode="fixed",
                            alpha="",
                            source_variant=fixed_variant,
                        )
                        if fixed_row is not None:
                            fixed_row["variant"] = variant_name
                            detail_rows.append(fixed_row)

    aggregate = aggregate_rows(detail_rows)
    write_csv(args.output_root / "adaptive_radius_grid_case_metrics.csv", detail_rows)
    write_csv(args.output_root / "adaptive_radius_grid_summary.csv", aggregate)
    markdown = summary_markdown(
        aggregate,
        source_cases_root=args.source_cases_root,
        output_root=args.output_root,
        alphas=alphas,
        variants=variants,
        limit=args.limit,
    )
    (args.output_root / "adaptive_radius_grid_summary.md").write_text(markdown)
    print(markdown)

    if failures:
        write_csv(args.output_root / "adaptive_radius_grid_failures.csv", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
