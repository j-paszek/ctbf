"""Population-only preflight for the CTBF v5 length/height trend program.

This owner-run boundary executes only CTBF simulation for the provisional
factor rows.  It writes compact population and resource summaries and never
calls cnp2cnp, samples a reconstruction condition, reconstructs, evaluates, or
reads an existing result corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from algorithm_evaluation.hypothesis_trend_pilot import (
    ANALYSIS_ROLE,
    DEFAULT_EXPECTED_CNA_STARTS,
    DEFAULT_HEIGHTS,
    DEFAULT_HEIGHT_TREND_LENGTH,
    DEFAULT_LENGTHS,
    DEFAULT_LENGTH_TREND_HEIGHT,
    PROJECT_ROOT,
    build_pilot_report,
)
from algorithm_evaluation.paper_pipeline_contract import (
    fraction_prefix_size,
    read_json,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import measured_stage
from simulator import CancerCellEvolutionSimulator
from simulator_config import SIMULATOR_SEMANTIC_VERSION, choose_crucial_mask


SCHEMA_VERSION = "ctbf-hypothesis-trend-population-preflight-v1"
PREFLIGHT_ROLE = "discovery_population_resource_preflight"
SEED_NAMESPACE = "ctbf-v5-hypothesis-trend-population-preflight-v1"
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "simulator_examples" / "paper_v5" / "clean_balanced.json"
CRUCIAL_MASK_FRACTION = 0.10
CRUCIAL_MASK_SEED = 20260803
DEFAULT_BASE_SEED = 20260805
DEFAULT_FIXED_FRACTION = 0.5
DEFAULT_STATIC_NODE_CAP = 150_000
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_seed(base_seed: int, length: int, height: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    material = f"{SEED_NAMESPACE}\0{base_seed}\0L{length}\0H{height}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def build_population_case_plan(
    *,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    heights: Sequence[int] = DEFAULT_HEIGHTS,
    expected_cna_starts: float = DEFAULT_EXPECTED_CNA_STARTS,
    length_trend_height: int = DEFAULT_LENGTH_TREND_HEIGHT,
    height_trend_length: int = DEFAULT_HEIGHT_TREND_LENGTH,
) -> list[dict[str, Any]]:
    t0 = build_pilot_report(
        lengths=lengths,
        heights=heights,
        expected_cna_starts=expected_cna_starts,
        length_trend_height=length_trend_height,
        height_trend_length=height_trend_length,
    )
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for trend_name in ("length_trend", "height_trend"):
        for row in t0["prospective_plan"][trend_name]:
            key = (row["genome_length"], row["number_of_generations"])
            if key not in unique:
                unique[key] = {
                    **row,
                    "trend_membership": [trend_name],
                }
            else:
                unique[key]["trend_membership"].append(trend_name)
    return [
        unique[key]
        for key in sorted(unique, key=lambda value: (value[1], value[0]))
    ]


def _profile(cell: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(cell.genome).tolist())


def _canonical_generation_cells(
    simulator: CancerCellEvolutionSimulator,
    generation: int,
) -> list[Any]:
    cells = [
        genotype
        for genotype in simulator.genotypes.values()
        if genotype.generation == generation
    ]
    return simulator.canonicalize_biopsy_genotypes(cells)


def _simulation_summary(
    *,
    config_path: Path,
    seed: int,
    schedule: Sequence[int],
    fixed_fraction: float,
) -> dict[str, Any]:
    simulator = CancerCellEvolutionSimulator(str(config_path), seed=seed)
    simulator.run_simulation()

    tree = simulator.tree
    generation_counts = Counter(
        int(attributes["generation"])
        for _node, attributes in tree.nodes(data=True)
    )
    profiles = [_profile(genotype) for genotype in simulator.genotypes.values()]
    cell_ids = [genotype.cell_id for genotype in simulator.genotypes.values()]
    schedule_rows = []
    fraction_union_profiles: set[tuple[int, ...]] = set()
    for generation in schedule:
        canonical = sorted(
            _canonical_generation_cells(simulator, int(generation)),
            key=lambda cell: (_profile(cell), str(cell.cell_id)),
        )
        selected_count = fraction_prefix_size(len(canonical), fixed_fraction)
        selected = canonical[:selected_count]
        fraction_union_profiles.update(_profile(cell) for cell in selected)
        schedule_rows.append(
            {
                "generation": int(generation),
                "occurrence_count": int(generation_counts.get(int(generation), 0)),
                "unique_state_count": len(canonical),
                "fixed_fraction_selected_count": selected_count,
            }
        )

    min_unique_states = min(
        (row["unique_state_count"] for row in schedule_rows),
        default=0,
    )
    fraction_union_count = len(fraction_union_profiles)
    fraction_selected_occurrence_count = sum(
        row["fixed_fraction_selected_count"] for row in schedule_rows
    )
    fraction_union_upper_bound = min(
        fraction_selected_occurrence_count,
        len(set(profiles)),
    )
    simulator_diagnostics = tree.graph.get(
        "simulation_diagnostics",
        simulator.diagnostics_snapshot(),
    )
    return {
        "truth_node_count": tree.number_of_nodes(),
        "truth_edge_count": tree.number_of_edges(),
        "realized_max_generation": max(generation_counts, default=0),
        "node_count_by_generation": {
            str(generation): generation_counts[generation]
            for generation in sorted(generation_counts)
        },
        "unique_exact_profile_count": len(set(profiles)),
        "unique_cell_id_count": len(set(cell_ids)),
        "repeated_profile_occurrence_count": len(profiles) - len(set(profiles)),
        "repeated_cell_id_occurrence_count": len(cell_ids) - len(set(cell_ids)),
        "schedule": schedule_rows,
        "maximum_common_fixed_budget_per_level": min_unique_states,
        "fixed_fraction": fixed_fraction,
        "fixed_fraction_selected_occurrence_count": fraction_selected_occurrence_count,
        "fixed_fraction_canonical_prefix_profile_union_count": fraction_union_count,
        "fixed_fraction_profile_union_upper_bound": fraction_union_upper_bound,
        "fixed_fraction_unordered_profile_pair_upper_bound": (
            fraction_union_upper_bound * (fraction_union_upper_bound - 1) // 2
        ),
        "fixed_fraction_bidirectional_cnp2cnp_call_bound": (
            fraction_union_upper_bound * (fraction_union_upper_bound - 1)
        ),
        "simulation_diagnostics": simulator_diagnostics,
    }


def _case_config(
    base_config: Mapping[str, Any],
    *,
    length: int,
    height: int,
    expected_cna_starts: float,
) -> dict[str, Any]:
    config = dict(base_config)
    config["GENOME_LENGTH"] = int(length)
    config["CRUCIAL_BIN_INDICES"] = list(
        choose_crucial_mask(
            int(length),
            CRUCIAL_MASK_FRACTION,
            CRUCIAL_MASK_SEED,
        )
    )
    config["NUMBER_OF_GENERATIONS"] = int(height)
    config["CNA_EVENT_PROBABILITY"] = float(expected_cna_starts) / int(length)
    return config


def run_population_preflight(
    *,
    base_config_path: Path | str = DEFAULT_BASE_CONFIG,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    heights: Sequence[int] = DEFAULT_HEIGHTS,
    expected_cna_starts: float = DEFAULT_EXPECTED_CNA_STARTS,
    length_trend_height: int = DEFAULT_LENGTH_TREND_HEIGHT,
    height_trend_length: int = DEFAULT_HEIGHT_TREND_LENGTH,
    base_seed: int = DEFAULT_BASE_SEED,
    fixed_fraction: float = DEFAULT_FIXED_FRACTION,
    static_node_cap: int = DEFAULT_STATIC_NODE_CAP,
    timeout_seconds_per_case: int = DEFAULT_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.is_file():
        raise ValueError(f"Base config is not a file: {base_config_path}")
    base_config = read_json(base_config_path)
    if base_config.get("SIMULATOR_SEMANTIC_VERSION") != SIMULATOR_SEMANTIC_VERSION:
        raise ValueError("Population preflight requires a CTBF v5 simulator config.")
    if (
        base_config.get("OFFSPRING_MODEL") != "constant"
        or base_config.get("OFFSPRING_PARAMETER") != 1
        or base_config.get("BASELINE_DESCENDANT_ATTEMPTS") != 1
    ):
        raise ValueError(
            "Population preflight currently requires the clean one-baseline-plus-one-"
            "constant descendant-attempt contract used by its binary static bound."
        )
    if not 0 < float(fixed_fraction) <= 1:
        raise ValueError("fixed_fraction must lie in (0,1].")
    if isinstance(static_node_cap, bool) or not isinstance(static_node_cap, int) or static_node_cap <= 0:
        raise ValueError("static_node_cap must be a positive integer.")
    if timeout_seconds_per_case <= 0 or rss_limit_bytes <= 0:
        raise ValueError("timeout and RSS limit must be positive.")

    plan = build_population_case_plan(
        lengths=lengths,
        heights=heights,
        expected_cna_starts=expected_cna_starts,
        length_trend_height=length_trend_height,
        height_trend_length=height_trend_length,
    )
    records = []
    with tempfile.TemporaryDirectory(prefix="ctbf-hypothesis-population-") as directory:
        temporary_root = Path(directory)
        for row in plan:
            length = int(row["genome_length"])
            height = int(row["number_of_generations"])
            case_key = f"population-L{length}-H{height}"
            seed = _case_seed(int(base_seed), length, height)
            static_bound = int(row["unreduced_binary_total_node_bound"])
            common = {
                "case_key": case_key,
                "trend_membership": list(row["trend_membership"]),
                "genome_length": length,
                "number_of_generations": height,
                "relative_biopsy_generations": list(row["relative_biopsy_generations"]),
                "expected_cna_starts_per_attempted_child": float(expected_cna_starts),
                "cna_event_probability": float(expected_cna_starts) / length,
                "simulation_seed": seed,
                "unreduced_binary_total_node_bound": static_bound,
            }
            if static_bound > static_node_cap:
                records.append(
                    {
                        **common,
                        "status": "not_run_static_bound",
                        "message": (
                            f"Unreduced bound {static_bound} exceeds static cap "
                            f"{static_node_cap}."
                        ),
                        "runtime": None,
                        "summary": None,
                    }
                )
                continue

            config = _case_config(
                base_config,
                length=length,
                height=height,
                expected_cna_starts=float(expected_cna_starts),
            )
            config_path = temporary_root / f"{case_key}.json"
            write_json_atomic(config_path, config)
            summary, runtime, error = measured_stage(
                lambda config_path=config_path, seed=seed, schedule=tuple(
                    row["relative_biopsy_generations"]
                ): _simulation_summary(
                    config_path=config_path,
                    seed=seed,
                    schedule=schedule,
                    fixed_fraction=float(fixed_fraction),
                ),
                timeout_seconds=int(timeout_seconds_per_case),
                rss_limit_bytes=int(rss_limit_bytes),
            )
            if error is None:
                records.append(
                    {
                        **common,
                        "status": "success",
                        "message": None,
                        "runtime": runtime,
                        "summary": summary,
                    }
                )
            else:
                records.append(
                    {
                        **common,
                        "status": "failure",
                        "message": str(error)[:4096],
                        "error_type": type(error).__name__,
                        "runtime": runtime,
                        "summary": None,
                    }
                )
            gc.collect()

    status_counts = Counter(record["status"] for record in records)
    successful = [record for record in records if record["status"] == "success"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": PREFLIGHT_ROLE,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "scientific_role": {
            "parent_program_role": ANALYSIS_ROLE,
            "paper_evidence_allowed": False,
            "simulation_only": True,
            "cnp2cnp_run": False,
            "reconstruction_run": False,
            "evaluation_run": False,
        },
        "input": {
            "base_config": base_config_path.relative_to(PROJECT_ROOT).as_posix()
            if base_config_path.is_relative_to(PROJECT_ROOT)
            else str(base_config_path),
            "base_config_sha256": _file_sha256(base_config_path),
            "base_seed": int(base_seed),
            "seed_namespace": SEED_NAMESPACE,
            "lengths": sorted({int(value) for value in lengths}),
            "heights": sorted({int(value) for value in heights}),
            "expected_cna_starts_per_attempted_child": float(expected_cna_starts),
            "fixed_fraction": float(fixed_fraction),
            "static_node_cap": int(static_node_cap),
            "timeout_seconds_per_case": int(timeout_seconds_per_case),
            "rss_limit_bytes": int(rss_limit_bytes),
        },
        "summary": {
            "planned_case_count": len(records),
            "status_counts": dict(sorted(status_counts.items())),
            "successful_truth_node_range": (
                {
                    "minimum": min(
                        record["summary"]["truth_node_count"] for record in successful
                    ),
                    "maximum": max(
                        record["summary"]["truth_node_count"] for record in successful
                    ),
                }
                if successful
                else None
            ),
            "successful_fixed_fraction_profile_union_upper_bound_range": (
                {
                    "minimum": min(
                        record["summary"]["fixed_fraction_profile_union_upper_bound"]
                        for record in successful
                    ),
                    "maximum": max(
                        record["summary"]["fixed_fraction_profile_union_upper_bound"]
                        for record in successful
                    ),
                }
                if successful
                else None
            ),
        },
        "cases": records,
        "provenance": {
            "source_module": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "source_module_sha256": _file_sha256(Path(__file__)),
            "reads_existing_result_corpus": False,
            "writes_truth_trees_or_profiles": False,
        },
    }
    validate_population_report(report)
    return report


def validate_population_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown population-preflight schema.")
    if report.get("analysis_role") != PREFLIGHT_ROLE:
        raise ValueError("Population preflight has the wrong analysis role.")
    role = report.get("scientific_role")
    if not isinstance(role, Mapping) or role.get("paper_evidence_allowed") is not False:
        raise ValueError("Population preflight must be ineligible as paper evidence.")
    if any(role.get(field) is not False for field in ("cnp2cnp_run", "reconstruction_run", "evaluation_run")):
        raise ValueError("Population preflight must not run downstream stages.")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Population preflight must contain case records.")
    keys = [record.get("case_key") for record in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Population preflight contains duplicate case keys.")
    for record in cases:
        if record.get("status") not in {"success", "failure", "not_run_static_bound"}:
            raise ValueError("Population preflight contains an unknown status.")
        if record["status"] == "success" and not isinstance(record.get("summary"), Mapping):
            raise ValueError("Successful population case lacks a summary.")
    json.dumps(report, sort_keys=True, allow_nan=False)


def compact_summary(report: Mapping[str, Any], output: Path) -> dict[str, Any]:
    cases = {}
    for record in report["cases"]:
        summary = record.get("summary")
        cases[record["case_key"]] = {
            "status": record["status"],
            "truth_node_count": summary.get("truth_node_count") if summary else None,
            "fixed_fraction_profile_union_upper_bound": (
                summary.get("fixed_fraction_profile_union_upper_bound")
                if summary
                else None
            ),
            "fixed_fraction_canonical_prefix_profile_union_count": (
                summary.get("fixed_fraction_canonical_prefix_profile_union_count")
                if summary
                else None
            ),
            "maximum_common_fixed_budget_per_level": (
                summary.get("maximum_common_fixed_budget_per_level") if summary else None
            ),
            "wall_time_seconds": (
                record["runtime"]["wall_time_ns"] / 1_000_000_000
                if record.get("runtime")
                else None
            ),
            "message": record.get("message"),
        }
    return {
        "analysis_role": report["analysis_role"],
        "schema_version": report["schema_version"],
        "output": str(output.resolve()),
        "status_counts": report["summary"]["status_counts"],
        "cases": cases,
        "next_stage": (
            "freeze_fixed_budget_and_one_case_cnp2cnp_preflight_only_after_review"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the simulation-only population/resource preflight for the "
            "provisional CTBF v5 length/height trends."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--heights", type=int, nargs="+", default=list(DEFAULT_HEIGHTS))
    parser.add_argument("--expected-cna-starts", type=float, default=DEFAULT_EXPECTED_CNA_STARTS)
    parser.add_argument("--length-trend-height", type=int, default=DEFAULT_LENGTH_TREND_HEIGHT)
    parser.add_argument("--height-trend-length", type=int, default=DEFAULT_HEIGHT_TREND_LENGTH)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--fixed-fraction", type=float, default=DEFAULT_FIXED_FRACTION)
    parser.add_argument("--static-node-cap", type=int, default=DEFAULT_STATIC_NODE_CAP)
    parser.add_argument("--timeout-seconds-per-case", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new versioned path or pass --overwrite."
        )
    report = run_population_preflight(
        base_config_path=arguments.base_config,
        lengths=arguments.lengths,
        heights=arguments.heights,
        expected_cna_starts=arguments.expected_cna_starts,
        length_trend_height=arguments.length_trend_height,
        height_trend_length=arguments.height_trend_length,
        base_seed=arguments.base_seed,
        fixed_fraction=arguments.fixed_fraction,
        static_node_cap=arguments.static_node_cap,
        timeout_seconds_per_case=arguments.timeout_seconds_per_case,
        rss_limit_bytes=arguments.rss_limit_bytes,
    )
    write_json_atomic(output, report)
    print(json.dumps(compact_summary(report, output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
