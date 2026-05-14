#!/usr/bin/env python
import argparse
import copy
from dataclasses import dataclass
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

from reconstructor import (  # noqa: E402
    build_evolution_tree,
    neighbor_joining_standard,
    resolve_biopsy_guided_config,
)
from reconstructor_algorithm_config import ALGORITHM_CONFIG_BY_NAME  # noqa: E402
from reconstructor_registry import resolve_reconstruction_algorithm  # noqa: E402

from freeze_algorithm_variant_cases import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    VARIANT_PRESETS,
    cell_lists_from_input,
    existing_seeds_for_variant,
    load_json,
    metric_summary,
    node_link_data,
    result_path,
    to_newick,
    true_tree_from_input,
    write_json,
)


@dataclass(frozen=True)
class BiopsyPresetBenchmarkSpec:
    benchmark_name: str
    preset_name: str
    neighbor_joining: object
    top_reconstruction_algorithm: str


def _resolve_top_reconstruction_algorithm(name):
    if name in {None, "neighbor_joining_standard"}:
        return neighbor_joining_standard, "neighbor_joining_standard"
    return resolve_reconstruction_algorithm(name), name


def _build_biopsy_preset_benchmarks():
    benchmarks = {}
    for name, config in ALGORITHM_CONFIG_BY_NAME.items():
        preset_name = config.procedure.biopsy_guided_preset
        if preset_name is None:
            continue
        top_algorithm, top_algorithm_name = _resolve_top_reconstruction_algorithm(
            config.procedure.top_reconstruction_algorithm
        )
        benchmarks[name] = BiopsyPresetBenchmarkSpec(
            benchmark_name=name,
            preset_name=preset_name,
            neighbor_joining=top_algorithm,
            top_reconstruction_algorithm=top_algorithm_name,
        )
    return benchmarks


BIOPSY_PRESET_BENCHMARKS = _build_biopsy_preset_benchmarks()


def resolve_variants(selected_variants):
    names = selected_variants or list(VARIANT_PRESETS)
    unknown = sorted(set(names) - set(VARIANT_PRESETS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(VARIANT_PRESETS)}")
    return names


def resolve_presets(selected_presets):
    names = selected_presets or list(BIOPSY_PRESET_BENCHMARKS)
    unknown = sorted(set(names) - set(BIOPSY_PRESET_BENCHMARKS))
    if unknown:
        raise ValueError(
            f"Unknown preset benchmark names: {unknown}. "
            f"Available: {sorted(BIOPSY_PRESET_BENCHMARKS)}"
        )
    return [BIOPSY_PRESET_BENCHMARKS[name] for name in names]


def input_path(cases_root, variant_name, seed):
    return Path(cases_root) / variant_name / str(seed) / "input.json"


def biopsy_preset_result(input_case, spec):
    cell_lists = cell_lists_from_input(input_case)
    matrix = input_case["distance_matrices"]["cnp2cnp"]
    reconstructed_tree, _, reconstructed_root = build_evolution_tree(
        copy.deepcopy(cell_lists),
        r=input_case["r_dist"],
        seed=input_case["seed"],
        inids=matrix["ids"],
        indm=np.array(matrix["matrix"], dtype=float),
        neighbor_joining=spec.neighbor_joining,
        biopsy_guided_config=resolve_biopsy_guided_config(spec.preset_name),
    )
    true_tree = true_tree_from_input(input_case)
    return {
        "case_id": input_case["case_id"],
        "variant": input_case["variant"],
        "seed": input_case["seed"],
        "algorithm": spec.benchmark_name,
        "mode": "biopsy_guided_top",
        "biopsy_guided_preset": spec.preset_name,
        "neighbor_joining": spec.top_reconstruction_algorithm,
        "root": reconstructed_root,
        "newick": to_newick(reconstructed_tree),
        "reconstructed_tree": node_link_data(reconstructed_tree),
        "metrics": metric_summary(true_tree, reconstructed_tree),
    }


def write_case_presets(cases_root, variant_name, seed, presets, overwrite=False, skip_existing=False):
    input_case = load_json(input_path(cases_root, variant_name, seed))
    written = []
    skipped = []
    for spec in presets:
        output_path = result_path(cases_root, variant_name, seed, "biopsy_guided_top", spec.benchmark_name)
        if skip_existing and output_path.exists():
            skipped.append(output_path)
            continue
        write_json(
            output_path,
            biopsy_preset_result(input_case, spec),
            overwrite=overwrite,
        )
        written.append(output_path)
    return written, skipped


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a fast biopsy-preset benchmark from frozen JSON cases. "
            "Only new biopsy-guided preset result JSON files are written; "
            "frozen biopsies, true trees, and distance matrices are reused."
        )
    )
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANT_PRESETS),
                        help="Variant to run. Can be passed multiple times. Defaults to all variants.")
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="Seed to run. Can be passed multiple times. Defaults to existing frozen seeds.")
    parser.add_argument("--preset", action="append", choices=sorted(BIOPSY_PRESET_BENCHMARKS),
                        help="Benchmark preset row to write. Defaults to all preset rows.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of seeds per variant after filtering.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Do not rewrite result JSON files that already exist.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = resolve_variants(args.variant)
    presets = resolve_presets(args.preset)
    seeds_by_variant = {}
    for variant_name in variants:
        seeds = args.seed if args.seed is not None else existing_seeds_for_variant(args.cases_root, variant_name)
        if args.limit is not None:
            seeds = seeds[:args.limit]
        seeds_by_variant[variant_name] = seeds

    print("Variants:", ", ".join(variants))
    for variant_name in variants:
        seeds = seeds_by_variant[variant_name]
        print(f"Seeds for {variant_name}: {len(seeds)}", ", ".join(str(seed) for seed in seeds))
    print("Preset rows:", ", ".join(spec.benchmark_name for spec in presets))
    print("Cases root:", args.cases_root)
    if args.dry_run:
        return

    failures = []
    total_written = 0
    total_skipped = 0
    for variant_name in variants:
        for seed in seeds_by_variant[variant_name]:
            try:
                written, skipped = write_case_presets(
                    args.cases_root,
                    variant_name,
                    seed,
                    presets,
                    overwrite=args.overwrite,
                    skip_existing=args.skip_existing,
                )
                total_written += len(written)
                total_skipped += len(skipped)
                print(f"{variant_name}/{seed}: wrote {len(written)}, skipped {len(skipped)}")
            except Exception as exc:
                failures.append({"variant": variant_name, "seed": seed, "error": str(exc)})
                print(f"Failed {variant_name}/{seed}: {exc}", file=sys.stderr)
                traceback.print_exc()
                if args.fail_fast:
                    raise

    print(f"Wrote {total_written} result files; skipped {total_skipped}.")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure['variant']}/{failure['seed']}: {failure['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
