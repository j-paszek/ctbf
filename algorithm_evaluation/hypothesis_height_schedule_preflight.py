"""Simulation-only feasibility preflight for the approved height schedule.

The fixed-budget height design observes six distinct canonical CNP genotype
states at three biopsy levels placed at 50%, 75%, and 100% of each configured
height.  This runner checks only whether those states exist across multiple
simulation seeds.  It never selects or serializes profiles, invokes cnp2cnp,
reconstructs a tree, or evaluates an outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from algorithm_evaluation.hypothesis_trend_pilot import PROJECT_ROOT
from algorithm_evaluation.hypothesis_trend_population_preflight import (
    _canonical_generation_cells,
    _case_config,
    _file_sha256,
)
from algorithm_evaluation.paper_pipeline_contract import read_json, write_json_atomic
from algorithm_evaluation.paper_pipeline_runner import measured_stage
from simulator import CancerCellEvolutionSimulator


SCHEMA_VERSION = "ctbf-hypothesis-height-schedule-availability-v1"
ANALYSIS_ROLE = "discovery_height_schedule_availability_preflight"
SEED_NAMESPACE = "ctbf-v5-hypothesis-height-schedule-availability-v1"
SAMPLING_UNIT = "canonical_unique_cnp_genotype_state"

APPROVED_RELATIVE_POSITIONS = (0.5, 0.75, 1.0)
APPROVED_SCHEDULES = {
    8: (4, 6, 8),
    12: (6, 9, 12),
    16: (8, 12, 16),
}
DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT / "simulator_examples" / "paper_v5" / "clean_balanced.json"
)
DEFAULT_GENOME_LENGTH = 50
DEFAULT_EXPECTED_CNA_STARTS = 0.1
DEFAULT_STATES_PER_LEVEL = 6
DEFAULT_REPLICATES = 12
DEFAULT_BASE_SEED = 20260810
DEFAULT_STATIC_NODE_CAP = 150_000
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_MAX_FAILURES = 6
MAX_REPLICATES = 24


SimulationCompute = Callable[[Path, int, Sequence[int], int], dict[str, Any]]


def _simulation_seed(base_seed: int, replicate_index: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    if (
        isinstance(replicate_index, bool)
        or not isinstance(replicate_index, int)
        or replicate_index < 0
    ):
        raise ValueError("replicate_index must be a nonnegative integer.")
    material = (
        f"{SEED_NAMESPACE}\0simulation\0{base_seed}\0{replicate_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _availability_summary(
    config_path: Path,
    seed: int,
    schedule: Sequence[int],
    states_per_level: int,
) -> dict[str, Any]:
    simulator = CancerCellEvolutionSimulator(str(config_path), seed=seed)
    simulator.run_simulation()

    generation_counts = Counter(
        int(attributes["generation"])
        for _node, attributes in simulator.tree.nodes(data=True)
    )
    rows = []
    for generation in schedule:
        unique_state_count = len(
            _canonical_generation_cells(simulator, int(generation))
        )
        rows.append(
            {
                "generation": int(generation),
                "occurrence_count": int(generation_counts.get(int(generation), 0)),
                "unique_state_count": unique_state_count,
                "fixed_budget_shortfall": max(
                    0,
                    int(states_per_level) - unique_state_count,
                ),
                "eligible_for_fixed_budget": (
                    unique_state_count >= int(states_per_level)
                ),
            }
        )

    minimum_unique_state_count = min(
        (row["unique_state_count"] for row in rows),
        default=0,
    )
    return {
        "truth_node_count": simulator.tree.number_of_nodes(),
        "truth_edge_count": simulator.tree.number_of_edges(),
        "realized_max_generation": max(generation_counts, default=0),
        "schedule": rows,
        "minimum_unique_state_count": minimum_unique_state_count,
        "eligible_for_fixed_budget": all(
            row["eligible_for_fixed_budget"] for row in rows
        ),
        "insufficient_generation_count": sum(
            not row["eligible_for_fixed_budget"] for row in rows
        ),
    }


def _typed_error(error: BaseException | None) -> dict[str, str] | None:
    if error is None:
        return None
    return {
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }


def _validate_case_summary(
    summary: Mapping[str, Any],
    *,
    schedule: Sequence[int],
    states_per_level: int,
) -> None:
    rows = summary.get("schedule")
    if not isinstance(rows, list) or len(rows) != len(schedule):
        raise ValueError("Availability summary has the wrong biopsy-level count.")
    observed_generations = [row.get("generation") for row in rows]
    if observed_generations != list(schedule):
        raise ValueError("Availability summary has the wrong generation schedule.")
    for row in rows:
        for field in ("occurrence_count", "unique_state_count", "fixed_budget_shortfall"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Availability summary has invalid {field}.")
        expected_eligibility = row["unique_state_count"] >= states_per_level
        if row.get("eligible_for_fixed_budget") is not expected_eligibility:
            raise ValueError("Per-level fixed-budget eligibility is inconsistent.")
        if row["fixed_budget_shortfall"] != max(
            0,
            states_per_level - row["unique_state_count"],
        ):
            raise ValueError("Per-level fixed-budget shortfall is inconsistent.")
    expected_minimum = min(row["unique_state_count"] for row in rows)
    if summary.get("minimum_unique_state_count") != expected_minimum:
        raise ValueError("Availability summary has an inconsistent minimum.")
    expected_eligibility = all(
        row["eligible_for_fixed_budget"] for row in rows
    )
    if summary.get("eligible_for_fixed_budget") is not expected_eligibility:
        raise ValueError("Case fixed-budget eligibility is inconsistent.")
    if summary.get("insufficient_generation_count") != sum(
        not row["eligible_for_fixed_budget"] for row in rows
    ):
        raise ValueError("Case insufficient-generation count is inconsistent.")
    for field in ("truth_node_count", "truth_edge_count", "realized_max_generation"):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Availability summary has invalid {field}.")


def _number_range(values: Sequence[int]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def _aggregate_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
) -> dict[str, Any]:
    by_height = {}
    for height, schedule in APPROVED_SCHEDULES.items():
        records = [record for record in cases if record["height"] == height]
        completed = [
            record
            for record in records
            if record["status"] in {"eligible", "insufficient_unique_states"}
        ]
        by_height[str(height)] = {
            "schedule": list(schedule),
            "planned_case_count": replicates,
            "attempted_case_count": len(records),
            "status_counts": dict(
                sorted(Counter(record["status"] for record in records).items())
            ),
            "eligible_case_count": sum(
                record["status"] == "eligible" for record in records
            ),
            "minimum_unique_state_count": _number_range(
                [
                    int(record["summary"]["minimum_unique_state_count"])
                    for record in completed
                ]
            ),
            "truth_node_count": _number_range(
                [int(record["summary"]["truth_node_count"]) for record in completed]
            ),
        }

    by_replicate: dict[int, set[int]] = {}
    for record in cases:
        if record["status"] == "eligible":
            by_replicate.setdefault(int(record["replicate_index"]), set()).add(
                int(record["height"])
            )
    complete_eligible_blocks = sum(
        heights == set(APPROVED_SCHEDULES) for heights in by_replicate.values()
    )
    planned_case_count = replicates * len(APPROVED_SCHEDULES)
    all_cases_eligible = (
        len(cases) == planned_case_count
        and all(record["status"] == "eligible" for record in cases)
    )
    return {
        "by_height": by_height,
        "complete_eligible_replicate_block_count": complete_eligible_blocks,
        "planned_replicate_block_count": replicates,
        "all_planned_cases_eligible": all_cases_eligible,
        "availability_supported_for_owner_review": all_cases_eligible,
    }


def run_height_schedule_preflight(
    *,
    base_config_path: Path | str = DEFAULT_BASE_CONFIG,
    genome_length: int = DEFAULT_GENOME_LENGTH,
    expected_cna_starts: float = DEFAULT_EXPECTED_CNA_STARTS,
    states_per_level: int = DEFAULT_STATES_PER_LEVEL,
    replicates: int = DEFAULT_REPLICATES,
    base_seed: int = DEFAULT_BASE_SEED,
    static_node_cap: int = DEFAULT_STATIC_NODE_CAP,
    timeout_seconds_per_case: int = DEFAULT_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    max_failures: int = DEFAULT_MAX_FAILURES,
    simulation_compute: SimulationCompute | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Run the approved schedule's population-availability check."""
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.is_file():
        raise ValueError(f"Base config is not a file: {base_config_path}")
    base_config = read_json(base_config_path)
    if base_config.get("SIMULATOR_SEMANTIC_VERSION") != "ctbf-cnp-state-simulator-v5":
        raise ValueError("Height schedule preflight requires a CTBF v5 config.")
    if (
        base_config.get("OFFSPRING_MODEL") != "constant"
        or base_config.get("OFFSPRING_PARAMETER") != 1
        or base_config.get("BASELINE_DESCENDANT_ATTEMPTS") != 1
    ):
        raise ValueError(
            "Height schedule preflight requires the clean binary-bound offspring contract."
        )
    if genome_length != DEFAULT_GENOME_LENGTH:
        raise ValueError(f"Approved height preflight requires length {DEFAULT_GENOME_LENGTH}.")
    if expected_cna_starts != DEFAULT_EXPECTED_CNA_STARTS:
        raise ValueError(
            "Approved height preflight requires expected CNA starts "
            f"{DEFAULT_EXPECTED_CNA_STARTS}."
        )
    if states_per_level != DEFAULT_STATES_PER_LEVEL:
        raise ValueError(
            f"Approved height preflight requires {DEFAULT_STATES_PER_LEVEL} states per level."
        )
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise ValueError("replicates must be an integer.")
    if not 1 <= replicates <= MAX_REPLICATES:
        raise ValueError(f"replicates must be in [1,{MAX_REPLICATES}].")
    _simulation_seed(base_seed, 0)
    for field, value in (
        ("static_node_cap", static_node_cap),
        ("timeout_seconds_per_case", timeout_seconds_per_case),
        ("rss_limit_bytes", rss_limit_bytes),
        ("max_failures", max_failures),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer.")

    compute = simulation_compute or _availability_summary
    cases = []
    failure_count = 0
    stopped_early = False
    with tempfile.TemporaryDirectory(prefix="ctbf-height-schedule-") as directory:
        temporary_root = Path(directory)
        for replicate_index in range(replicates):
            simulation_seed = _simulation_seed(base_seed, replicate_index)
            for height, schedule in APPROVED_SCHEDULES.items():
                static_bound = (2 ** (height + 1)) - 1
                common = {
                    "case_key": f"height-schedule-H{height}-R{replicate_index + 1:03d}",
                    "height": height,
                    "replicate_index": replicate_index,
                    "simulation_seed": simulation_seed,
                    "schedule": list(schedule),
                    "unreduced_binary_total_node_bound": static_bound,
                }
                if static_bound > static_node_cap:
                    cases.append(
                        {
                            **common,
                            "status": "not_run_static_bound",
                            "runtime": None,
                            "summary": None,
                            "error": {
                                "type": "StaticNodeBoundExceeded",
                                "message": (
                                    f"Unreduced bound {static_bound} exceeds static cap "
                                    f"{static_node_cap}."
                                ),
                            },
                        }
                    )
                    continue

                config = _case_config(
                    base_config,
                    length=genome_length,
                    height=height,
                    expected_cna_starts=expected_cna_starts,
                )
                config_path = temporary_root / f"{common['case_key']}.json"
                write_json_atomic(config_path, config)

                def execute_case(
                    config_path: Path = config_path,
                    simulation_seed: int = simulation_seed,
                    schedule: Sequence[int] = schedule,
                ) -> dict[str, Any]:
                    summary = compute(
                        config_path,
                        simulation_seed,
                        schedule,
                        states_per_level,
                    )
                    _validate_case_summary(
                        summary,
                        schedule=schedule,
                        states_per_level=states_per_level,
                    )
                    return summary

                summary, runtime, error = measured_stage(
                    execute_case,
                    timeout_seconds=timeout_seconds_per_case,
                    rss_limit_bytes=rss_limit_bytes,
                )
                if error is not None:
                    cases.append(
                        {
                            **common,
                            "status": "simulation_failure",
                            "runtime": runtime,
                            "summary": None,
                            "error": _typed_error(error),
                        }
                    )
                    failure_count += 1
                    if failure_count >= max_failures:
                        stopped_early = True
                        break
                else:
                    cases.append(
                        {
                            **common,
                            "status": (
                                "eligible"
                                if summary["eligible_for_fixed_budget"]
                                else "insufficient_unique_states"
                            ),
                            "runtime": runtime,
                            "summary": summary,
                            "error": None,
                        }
                    )
                gc.collect()
            if stopped_early:
                break

    aggregate = _aggregate_cases(cases, replicates=replicates)
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "status": "stopped_at_failure_cap" if stopped_early else "complete",
        "scientific_role": {
            "paper_evidence_allowed": False,
            "design_feasibility_only": True,
            "simulation_run": simulation_compute is None,
            "injected_simulation_for_test": simulation_compute is not None,
            "profile_selection_run": False,
            "percentage_sampling_run": False,
            "cnp2cnp_run": False,
            "reconstruction_run": False,
            "evaluation_run": False,
        },
        "input": {
            "base_config": (
                base_config_path.relative_to(PROJECT_ROOT).as_posix()
                if base_config_path.is_relative_to(PROJECT_ROOT)
                else str(base_config_path)
            ),
            "base_config_sha256": _file_sha256(base_config_path),
            "genome_length": genome_length,
            "expected_cna_starts_per_attempted_child": expected_cna_starts,
            "sampling_unit": SAMPLING_UNIT,
            "population_abundance_weighted": False,
            "percentage_sampling": False,
            "states_per_level": states_per_level,
            "biopsy_level_count": len(APPROVED_RELATIVE_POSITIONS),
            "relative_biopsy_positions": list(APPROVED_RELATIVE_POSITIONS),
            "generation_schedule_by_height": {
                str(height): list(schedule)
                for height, schedule in APPROVED_SCHEDULES.items()
            },
            "replicates": replicates,
            "base_seed": base_seed,
            "seed_namespace": SEED_NAMESPACE,
            "common_simulation_seed_within_replicate_across_heights": True,
            "availability_seeds_must_not_be_reused_for_reconstruction": True,
            "static_node_cap": static_node_cap,
            "timeout_seconds_per_case": timeout_seconds_per_case,
            "rss_limit_bytes": rss_limit_bytes,
            "max_failures": max_failures,
        },
        "resource_bound": {
            "planned_case_count": replicates * len(APPROVED_SCHEDULES),
            "attempted_case_count": len(cases),
            "maximum_observation_count_if_later_sampled": (
                states_per_level * len(APPROVED_RELATIVE_POSITIONS)
            ),
            "profiles_selected_in_this_preflight": 0,
            "distance_entries_computed_in_this_preflight": 0,
            "sequential_execution": True,
        },
        "cases": cases,
        "aggregate": aggregate,
        "provenance": {
            "source_module": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "source_module_sha256": _file_sha256(Path(__file__)),
            "reads_existing_result_corpus": False,
            "writes_truth_trees_or_profiles": False,
        },
    }
    validate_height_schedule_report(report)
    return report


def validate_height_schedule_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown height-schedule availability schema.")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("Height-schedule preflight has the wrong role.")
    if report.get("status") not in {"complete", "stopped_at_failure_cap"}:
        raise ValueError("Height-schedule preflight has an unknown status.")
    role = report.get("scientific_role")
    if not isinstance(role, Mapping) or role.get("paper_evidence_allowed") is not False:
        raise ValueError("Height-schedule preflight must remain design-only.")
    for field in (
        "profile_selection_run",
        "percentage_sampling_run",
        "cnp2cnp_run",
        "reconstruction_run",
        "evaluation_run",
    ):
        if role.get(field) is not False:
            raise ValueError(f"Height-schedule preflight must keep {field}=false.")

    inputs = report.get("input")
    if not isinstance(inputs, Mapping):
        raise ValueError("Height-schedule preflight lacks its input contract.")
    expected_inputs = {
        "sampling_unit": SAMPLING_UNIT,
        "population_abundance_weighted": False,
        "percentage_sampling": False,
        "states_per_level": DEFAULT_STATES_PER_LEVEL,
        "biopsy_level_count": 3,
        "relative_biopsy_positions": list(APPROVED_RELATIVE_POSITIONS),
        "generation_schedule_by_height": {
            str(height): list(schedule)
            for height, schedule in APPROVED_SCHEDULES.items()
        },
        "availability_seeds_must_not_be_reused_for_reconstruction": True,
    }
    mismatches = {
        field: {"expected": expected, "observed": inputs.get(field)}
        for field, expected in expected_inputs.items()
        if inputs.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"Height-schedule input contract mismatch: {mismatches}")

    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Height-schedule preflight must contain case records.")
    case_keys = [record.get("case_key") for record in cases]
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("Height-schedule preflight contains duplicate case keys.")
    allowed_statuses = {
        "eligible",
        "insufficient_unique_states",
        "simulation_failure",
        "not_run_static_bound",
    }
    for record in cases:
        if record.get("status") not in allowed_statuses:
            raise ValueError("Height-schedule case has an unknown status.")
        height = record.get("height")
        if height not in APPROVED_SCHEDULES:
            raise ValueError("Height-schedule case has an unapproved height.")
        if record.get("schedule") != list(APPROVED_SCHEDULES[height]):
            raise ValueError("Height-schedule case has an unapproved schedule.")
        if record["status"] in {"eligible", "insufficient_unique_states"}:
            summary = record.get("summary")
            if not isinstance(summary, Mapping):
                raise ValueError("Completed height-schedule case lacks a summary.")
            _validate_case_summary(
                summary,
                schedule=APPROVED_SCHEDULES[height],
                states_per_level=DEFAULT_STATES_PER_LEVEL,
            )
            if (record["status"] == "eligible") is not bool(
                summary["eligible_for_fixed_budget"]
            ):
                raise ValueError("Height-schedule case status is inconsistent.")
        elif record.get("summary") is not None:
            raise ValueError("Unrun/failed height-schedule case must not have a summary.")
    json.dumps(report, sort_keys=True, allow_nan=False)


def compact_summary(report: Mapping[str, Any], output: Path) -> dict[str, Any]:
    by_height = {
        height: {
            "schedule": block["schedule"],
            "status_counts": block["status_counts"],
            "eligible_case_count": block["eligible_case_count"],
            "planned_case_count": block["planned_case_count"],
            "minimum_unique_state_count": block["minimum_unique_state_count"],
            "truth_node_count": block["truth_node_count"],
        }
        for height, block in report["aggregate"]["by_height"].items()
    }
    supported = report["aggregate"]["availability_supported_for_owner_review"]
    return {
        "analysis_role": report["analysis_role"],
        "schema_version": report["schema_version"],
        "status": report["status"],
        "output": str(output.resolve()),
        "sampling_unit": report["input"]["sampling_unit"],
        "states_per_level": report["input"]["states_per_level"],
        "relative_biopsy_positions": report["input"]["relative_biopsy_positions"],
        "planned_case_count": report["resource_bound"]["planned_case_count"],
        "attempted_case_count": report["resource_bound"]["attempted_case_count"],
        "by_height": by_height,
        "complete_eligible_replicate_block_count": report["aggregate"][
            "complete_eligible_replicate_block_count"
        ],
        "availability_supported_for_owner_review": supported,
        "next_stage": (
            "owner_review_then_freeze_fresh_reconstruction_seeds"
            if supported
            else "owner_review_schedule_unavailable_or_operational_failures"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check six-state availability at 50%, 75%, and 100% of heights "
            "8, 12, and 16 using simulation only."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--static-node-cap", type=int, default=DEFAULT_STATIC_NODE_CAP)
    parser.add_argument(
        "--timeout-seconds-per-case",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new versioned path or pass --overwrite."
        )
    report = run_height_schedule_preflight(
        base_config_path=arguments.base_config,
        replicates=arguments.replicates,
        base_seed=arguments.base_seed,
        static_node_cap=arguments.static_node_cap,
        timeout_seconds_per_case=arguments.timeout_seconds_per_case,
        rss_limit_bytes=arguments.rss_limit_bytes,
        max_failures=arguments.max_failures,
    )
    write_json_atomic(output, report)
    print(json.dumps(compact_summary(report, output), indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
