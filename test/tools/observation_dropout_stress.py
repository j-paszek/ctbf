#!/usr/bin/env python
import argparse
import copy
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
    DEFAULT_OUTPUT_ROOT,
    LEGACY_VARIANT_NAMES,
    VARIANT_PRESETS,
    cnp2cnp_distance_matrix,
    genotypes_from_json,
    get_algorithms_to_test,
    input_path,
    load_json,
    reconstruction_result,
    resolve_reference_algorithm_indices,
    result_path,
    write_json,
)


DEFAULT_STRESS_ALGORITHM_NAMES = [
    "neighbor_joining_baseline",
    "neighbor_joining_hybrid_opt",
    "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
]
DEFAULT_DROPOUT_RATE = 0.05
DEFAULT_STRESS_SEED = 20260519
STRESS_KIND = "observed_positive_bin_dropout_to_zero"


def dropout_token(dropout_rate):
    return f"obsdrop{float(dropout_rate):g}".replace("-", "m").replace(".", "p")


def stable_cell_seed(stress_seed, case_seed, cell_id):
    return int(stress_seed) + int(case_seed) * 1_000_003 + int(cell_id) * 9_176


def dropout_genome(genome, *, dropout_rate, rng):
    observed = np.asarray(genome, dtype=int).copy()
    positive_positions = np.flatnonzero(observed > 0)
    if dropout_rate <= 0.0 or positive_positions.size == 0:
        return observed.tolist(), 0, int(positive_positions.size)

    dropped_mask = rng.random(positive_positions.size) < float(dropout_rate)
    dropped_positions = positive_positions[dropped_mask]
    observed[dropped_positions] = 0
    return observed.tolist(), int(dropped_positions.size), int(positive_positions.size)


def unique_cells_from_biopsies(input_case):
    unique = {}
    for biopsy in input_case["biopsies"]:
        for cell in biopsy["cells"]:
            unique[cell["cell_id"]] = cell
    return [unique[cell_id] for cell_id in sorted(unique)]


def l1_distance_matrix(cells):
    ids = [cell["cell_id"] for cell in cells]
    genomes = [np.asarray(cell["genome"], dtype=float) for cell in cells]
    matrix = np.zeros((len(cells), len(cells)), dtype=float)
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            distance = float(np.abs(genomes[i] - genomes[j]).sum())
            matrix[i, j] = distance
            matrix[j, i] = distance
    return ids, matrix


def recompute_observed_distance_matrix(input_case, distance_mode):
    cells = unique_cells_from_biopsies(input_case)
    if distance_mode == "cnp2cnp":
        return cnp2cnp_distance_matrix(genotypes_from_json(cells))
    if distance_mode == "l1":
        return l1_distance_matrix(cells)
    raise ValueError(f"Unsupported distance mode: {distance_mode}")


def perturb_input_case(source_input_case, *, dropout_rate, stress_seed, distance_mode="cnp2cnp"):
    input_case = copy.deepcopy(source_input_case)
    case_seed = int(input_case["seed"])
    stress_label = dropout_token(dropout_rate)
    genome_by_cell_id = {}
    dropped_by_cell_id = defaultdict(int)
    positive_by_cell_id = defaultdict(int)

    for clean_cell in unique_cells_from_biopsies(source_input_case):
        cell_id = int(clean_cell["cell_id"])
        rng = np.random.default_rng(stable_cell_seed(stress_seed, case_seed, cell_id))
        perturbed_genome, dropped, positives = dropout_genome(
            clean_cell["genome"],
            dropout_rate=dropout_rate,
            rng=rng,
        )
        genome_by_cell_id[cell_id] = perturbed_genome
        dropped_by_cell_id[cell_id] += dropped
        positive_by_cell_id[cell_id] += positives

    for biopsy in input_case["biopsies"]:
        for cell in biopsy["cells"]:
            cell["genome"] = list(genome_by_cell_id[int(cell["cell_id"])])

    ids, matrix = recompute_observed_distance_matrix(input_case, distance_mode)
    input_case["distance_matrices"]["cnp2cnp"] = {
        "ids": ids,
        "matrix": matrix,
    }
    input_case["case_id"] = f"{source_input_case['case_id']}_{stress_label}"
    input_case["source_case_id"] = source_input_case["case_id"]
    input_case["stress_label"] = stress_label
    input_case["observation_perturbation"] = {
        "kind": STRESS_KIND,
        "dropout_rate": float(dropout_rate),
        "stress_seed": int(stress_seed),
        "distance_mode": distance_mode,
        "source_variant": source_input_case["variant"],
        "source_case_id": source_input_case["case_id"],
        "perturbed_unique_cells": len(genome_by_cell_id),
        "positive_bins": int(sum(positive_by_cell_id.values())),
        "dropped_bins": int(sum(dropped_by_cell_id.values())),
        "dropped_bin_rate": (
            float(sum(dropped_by_cell_id.values())) / float(sum(positive_by_cell_id.values()))
            if sum(positive_by_cell_id.values())
            else 0.0
        ),
    }
    return input_case


def resolve_variants(selected_variants):
    names = selected_variants or LEGACY_VARIANT_NAMES
    unknown = sorted(set(names) - set(VARIANT_PRESETS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(VARIANT_PRESETS)}")
    return names


def existing_seeds_for_variant(cases_root, variant_name):
    variant_dir = Path(cases_root) / variant_name
    if not variant_dir.exists():
        return []
    seeds = []
    for seed_dir in variant_dir.iterdir():
        if seed_dir.is_dir() and seed_dir.name.isdigit() and (seed_dir / "input.json").exists():
            seeds.append(int(seed_dir.name))
    return sorted(seeds)


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


def metric_row(input_case, result, mode):
    perturbation = input_case["observation_perturbation"]
    return {
        "stress_label": input_case["stress_label"],
        "variant": input_case["variant"],
        "seed": input_case["seed"],
        "mode": mode,
        "algorithm": result["algorithm"],
        "dropout_rate": perturbation["dropout_rate"],
        "distance_mode": perturbation["distance_mode"],
        "perturbed_unique_cells": perturbation["perturbed_unique_cells"],
        "positive_bins": perturbation["positive_bins"],
        "dropped_bins": perturbation["dropped_bins"],
        "dropped_bin_rate": perturbation["dropped_bin_rate"],
        "ad_f1": result["metrics"]["ancestors_unique_restricted"]["F1"],
        "ad_precision": result["metrics"]["ancestors_unique_restricted"]["precision"],
        "grf": result["metrics"]["grf"],
    }


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stress_label"], row["variant"], row["mode"], row["algorithm"])].append(row)

    aggregated = []
    for (stress_label, variant, mode, algorithm), group in groups.items():
        aggregated.append({
            "stress_label": stress_label,
            "variant": variant,
            "mode": mode,
            "algorithm": algorithm,
            "cases": len(group),
            "mean_dropped_bin_rate": mean(row["dropped_bin_rate"] for row in group),
            "mean_ad_f1": mean(row["ad_f1"] for row in group),
            "mean_ad_precision": mean(row["ad_precision"] for row in group),
            "mean_grf": mean(row["grf"] for row in group),
        })
    return sorted(
        aggregated,
        key=lambda row: (row["stress_label"], row["variant"], row["mode"], row["algorithm"]),
    )


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def format_float(value):
    return f"{float(value):.6f}"


def summary_markdown(aggregate, *, source_cases_root, output_root, dropout_rate, stress_seed, distance_mode, limit):
    lines = [
        "# Observation Dropout Stress",
        "",
        f"- Source clean cases root: `{source_cases_root}`",
        f"- Scratch output root: `{output_root}`",
        f"- Stress kind: `{STRESS_KIND}`",
        f"- Dropout rate: `{dropout_rate}`",
        f"- Stress seed: `{stress_seed}`",
        f"- Observed distance mode: `{distance_mode}`",
        f"- Limit per variant: `{limit if limit is not None else 'all source seeds'}`",
        "",
        "## Mean Metrics",
        "",
        "| stress | variant | mode | algorithm | cases | dropped-bin rate | AD-F1 | AD-precision | GRF |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| {stress_label} | {variant} | {mode} | {algorithm} | {cases} | {drop_rate} | {ad_f1} | {ad_precision} | {grf} |".format(
                stress_label=row["stress_label"],
                variant=row["variant"],
                mode=row["mode"],
                algorithm=row["algorithm"],
                cases=row["cases"],
                drop_rate=format_float(row["mean_dropped_bin_rate"]),
                ad_f1=format_float(row["mean_ad_f1"]),
                ad_precision=format_float(row["mean_ad_precision"]),
                grf=format_float(row["mean_grf"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_stress_case(
    *,
    source_cases_root,
    output_root,
    variant,
    seed,
    algorithms,
    dropout_rate,
    stress_seed,
    distance_mode,
    overwrite,
    skip_existing,
):
    current_input_path = input_path(output_root, variant, seed)
    if skip_existing and current_input_path.exists():
        input_case = load_json(current_input_path)
    else:
        source_input = load_json(input_path(source_cases_root, variant, seed))
        input_case = perturb_input_case(
            source_input,
            dropout_rate=dropout_rate,
            stress_seed=stress_seed,
            distance_mode=distance_mode,
        )
        write_json(current_input_path, input_case, overwrite)

    rows = []
    for algorithm in algorithms:
        algorithm_name = getattr(algorithm, "__name__", str(algorithm))
        for mode in ["full_cnp", "biopsy_guided_top"]:
            output_path = result_path(output_root, variant, seed, mode, algorithm_name)
            if skip_existing and output_path.exists():
                result = load_json(output_path)
            else:
                result = reconstruction_result(input_case, algorithm, mode)
                write_json(output_path, result, overwrite)
            rows.append(metric_row(input_case, result, mode))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create scratch frozen JSON cases with deterministic observed-CNP dropout "
            "and replay selected reconstruction rows."
        )
    )
    parser.add_argument("--source-cases-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANT_PRESETS),
                        help="Variant to stress. Defaults to legacy fixed-r variants.")
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit seeds per variant.")
    parser.add_argument("--dropout-rate", type=float, default=DEFAULT_DROPOUT_RATE)
    parser.add_argument("--stress-seed", type=int, default=DEFAULT_STRESS_SEED)
    parser.add_argument("--distance-mode", choices=["cnp2cnp", "l1"], default="cnp2cnp")
    parser.add_argument("--algorithm-name", action="append", default=None)
    parser.add_argument("--algorithm-index", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = resolve_variants(args.variant)
    if args.algorithm_name is None and args.algorithm_index is None:
        algorithm_names = DEFAULT_STRESS_ALGORITHM_NAMES
    else:
        algorithm_names = args.algorithm_name
    algorithms = resolve_algorithms(algorithm_names, args.algorithm_index)

    seeds_by_variant = {}
    for variant in variants:
        seeds = args.seed if args.seed is not None else existing_seeds_for_variant(args.source_cases_root, variant)
        if args.limit is not None:
            seeds = seeds[:args.limit]
        seeds_by_variant[variant] = list(seeds)

    print(f"Source cases root: {args.source_cases_root}")
    print(f"Output root: {args.output_root}")
    print("Variants:", ", ".join(variants))
    for variant in variants:
        print(f"Seeds for {variant}: {len(seeds_by_variant[variant])}", ", ".join(str(s) for s in seeds_by_variant[variant]))
    print("Algorithms:", ", ".join(getattr(algorithm, "__name__", str(algorithm)) for algorithm in algorithms))
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Stress seed: {args.stress_seed}")
    print(f"Distance mode: {args.distance_mode}")
    if args.dry_run:
        return 0

    failures = []
    rows = []
    for variant in variants:
        for seed in seeds_by_variant[variant]:
            try:
                rows.extend(write_stress_case(
                    source_cases_root=args.source_cases_root,
                    output_root=args.output_root,
                    variant=variant,
                    seed=seed,
                    algorithms=algorithms,
                    dropout_rate=args.dropout_rate,
                    stress_seed=args.stress_seed,
                    distance_mode=args.distance_mode,
                    overwrite=args.overwrite,
                    skip_existing=args.skip_existing,
                ))
                print(f"Wrote stress case {variant}/{seed}")
            except Exception as exc:
                failures.append({"variant": variant, "seed": seed, "error": str(exc)})
                print(f"Failed {variant}/{seed}: {exc}", file=sys.stderr)
                traceback.print_exc()
                if args.fail_fast:
                    raise

    aggregate = aggregate_rows(rows)
    write_csv(args.output_root / "observation_dropout_case_metrics.csv", rows)
    write_csv(args.output_root / "observation_dropout_summary.csv", aggregate)
    markdown = summary_markdown(
        aggregate,
        source_cases_root=args.source_cases_root,
        output_root=args.output_root,
        dropout_rate=args.dropout_rate,
        stress_seed=args.stress_seed,
        distance_mode=args.distance_mode,
        limit=args.limit,
    )
    (args.output_root / "observation_dropout_summary.md").write_text(markdown)
    print(markdown)

    if failures:
        write_csv(args.output_root / "observation_dropout_failures.csv", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
