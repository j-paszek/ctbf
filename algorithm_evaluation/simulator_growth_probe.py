"""Compact owner-run CTBF v5 representative-growth and CN0-burden probe.

The probe runs simulation only.  It writes aggregate counts and small per-run
summaries; it never serializes truth trees, profiles, biopsies, distances, or
reconstruction results.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import numpy as np

from algorithm_evaluation.paper_pipeline_contract import (
    canonical_json_sha256,
    read_json,
    write_json_atomic,
)
from simulator import CancerCellEvolutionSimulator, SimulationResourceLimitExceeded
from simulator_config import SIMULATOR_SEMANTIC_VERSION, load_simulator_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ctbf-simulator-growth-probe-v2"
ANALYSIS_ROLE = "nonpaper_simulator_growth_intuition_probe"
SEED_NAMESPACE = "ctbf-v5-simulator-growth-probe-v1"
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "simulator_examples" / "default.json"
DEFAULT_GENERATIONS = 64
DEFAULT_REPLICATES = 100
DEFAULT_BASE_SEED = 20260812
MAX_REPLICATES = 100
STANDARD_CRUCIAL_BIN_INDICES = [4, 7, 38, 40, 41, 49, 54, 61, 69, 72]


SimulatorFactory = Callable[[Mapping[str, Any], int], CancerCellEvolutionSimulator]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replicate_seed(base_seed: int, replicate_index: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    if (
        isinstance(replicate_index, bool)
        or not isinstance(replicate_index, int)
        or replicate_index < 0
    ):
        raise ValueError("replicate_index must be a nonnegative integer.")
    material = (
        f"{SEED_NAMESPACE}\0{base_seed}\0{replicate_index}".encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _validate_standard_base_config(config: Mapping[str, Any]) -> None:
    required_values = {
        "SIMULATOR_SEMANTIC_VERSION": SIMULATOR_SEMANTIC_VERSION,
        "GENOME_LENGTH": 100,
        "NUMBER_OF_CHROMOSOMES": 1,
        "INITIAL_COPY_NUMBER": 2,
        "CRUCIAL_BIN_INDICES": STANDARD_CRUCIAL_BIN_INDICES,
        "OFFSPRING_MODEL": "constant",
        "OFFSPRING_PARAMETER": 1,
        "BASELINE_DESCENDANT_ATTEMPTS": 1,
        "CNA_EVENT_PROBABILITY": 0.001,
        "CNA_INITIATION_SCHEDULE": {"MODEL": "constant"},
        "GAIN_GIVEN_CNA_PROBABILITY": 0.5,
        "INTERVAL_CNA_PROBABILITY": 0.1,
        "INTERVAL_GAIN_OPERATOR_PROBABILITIES": {
            "unit": 0.8,
            "additive": 0.2,
            "multiplicative": 0,
        },
        "ADDITIVE_GAIN_LAMBDA": 0,
        "MULTIPLICATIVE_FACTOR_PROBABILITIES": {"2": 1},
        "WGD_PROBABILITY": 0,
        "REPRESENTATION_TYPE": "representative",
        "STATE_LINEAGE_REGULATION": {"MODEL": "none"},
        "RESOURCE_GUARD": {
            "MAX_REPRESENTATIVES_PER_GENERATION": 2000,
            "MAX_TOTAL_NODES": 40000,
        },
        "TELOMERIC_INSTABILITY_ENABLED": False,
        "TELOMERIC_INSTABILITY_INCREMENT": 0,
        "CRUCIAL_SURVIVAL_ENABLED": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in required_values.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Growth probe requires the approved CTBF v5 standard base config; "
            f"mismatches={mismatches!r}."
        )


def _number_summary(values: list[int]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "observation_count": len(values),
        "minimum": int(np.min(array)),
        "q25": float(np.quantile(array, 0.25, method="linear")),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "maximum": int(np.max(array)),
    }


def _histogram_value_at_rank(histogram: Mapping[int, int], rank: int) -> int:
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += int(count)
        if rank < cumulative:
            return int(value)
    raise ValueError("Histogram rank is outside the observed values.")


def _histogram_quantile(
    histogram: Mapping[int, int],
    quantile: float,
) -> float:
    observation_count = sum(int(count) for count in histogram.values())
    if observation_count < 1:
        raise ValueError("Cannot calculate a quantile of an empty histogram.")
    position = (observation_count - 1) * float(quantile)
    lower_rank = math.floor(position)
    upper_rank = math.ceil(position)
    lower = _histogram_value_at_rank(histogram, lower_rank)
    upper = _histogram_value_at_rank(histogram, upper_rank)
    return float(lower + (upper - lower) * (position - lower_rank))


def _histogram_number_summary(
    histogram: Mapping[int, int],
    *,
    scale: float = 1.0,
) -> dict[str, Any] | None:
    observation_count = sum(int(count) for count in histogram.values())
    if observation_count < 1:
        return None
    minimum = min(int(value) for value in histogram)
    maximum = max(int(value) for value in histogram)
    mean = sum(
        int(value) * int(count) for value, count in histogram.items()
    ) / observation_count
    return {
        "observation_count": observation_count,
        "minimum": minimum * scale,
        "q25": _histogram_quantile(histogram, 0.25) * scale,
        "median": _histogram_quantile(histogram, 0.5) * scale,
        "mean": mean * scale,
        "q75": _histogram_quantile(histogram, 0.75) * scale,
        "p95": _histogram_quantile(histogram, 0.95) * scale,
        "maximum": maximum * scale,
    }


def _merge_serialized_histogram(
    destination: Counter[int],
    serialized: Mapping[str, Any],
) -> None:
    destination.update(
        {int(value): int(count) for value, count in serialized.items()}
    )


def _zero_burden_summary(
    records: list[dict[str, Any]],
    *,
    generations: int,
    genome_length: int,
) -> list[dict[str, Any]]:
    rows = []
    for generation in range(generations + 1):
        complete_records = [
            record
            for record in records
            if int(record["complete_through_generation"]) >= generation
        ]
        zero_counts: Counter[int] = Counter()
        longest_runs: Counter[int] = Counter()
        for record in complete_records:
            generation_audit = record["_zero_burden_by_generation"].get(
                str(generation),
                {},
            )
            _merge_serialized_histogram(
                zero_counts,
                generation_audit.get("zero_bin_count_histogram", {}),
            )
            _merge_serialized_histogram(
                longest_runs,
                generation_audit.get(
                    "longest_contiguous_zero_run_histogram",
                    {},
                ),
            )

        profile_count = sum(zero_counts.values())
        thresholds = {}
        for fraction in (0.10, 0.25, 0.50):
            minimum_zero_bins = math.ceil(genome_length * fraction)
            count = sum(
                observed
                for zero_bins, observed in zero_counts.items()
                if zero_bins >= minimum_zero_bins
            )
            thresholds[f"{int(fraction * 100)}pct"] = {
                "minimum_zero_bins": minimum_zero_bins,
                "profile_count": int(count),
                "profile_fraction": (
                    None if profile_count == 0 else count / profile_count
                ),
            }

        rows.append(
            {
                "generation": generation,
                "complete_run_count": len(complete_records),
                "profile_count": int(profile_count),
                "zero_bin_fraction": _histogram_number_summary(
                    zero_counts,
                    scale=1.0 / genome_length,
                ),
                "longest_contiguous_zero_run_bins": (
                    _histogram_number_summary(longest_runs)
                ),
                "longest_contiguous_zero_run_fraction": (
                    _histogram_number_summary(
                        longest_runs,
                        scale=1.0 / genome_length,
                    )
                ),
                "profiles_at_or_above_zero_fraction": thresholds,
                "all_zero_profile_count": int(zero_counts.get(genome_length, 0)),
            }
        )
    return rows


def _event_selection_summary(
    records: list[dict[str, Any]],
    diagnostic_totals: Mapping[str, int],
) -> dict[str, Any]:
    interval_histograms = {
        stage: {"gain": Counter(), "loss": Counter()}
        for stage in ("proposed", "retained")
    }
    for record in records:
        audit = record["_interval_footprint_length_histograms"]
        for stage in ("proposed", "retained"):
            for direction in ("gain", "loss"):
                _merge_serialized_histogram(
                    interval_histograms[stage][direction],
                    audit[stage]["totals"].get(direction, {}),
                )

    balance = {}
    for stage in ("proposed", "retained"):
        gain = int(diagnostic_totals.get(f"{stage}_segmental_gain_events", 0))
        loss = int(diagnostic_totals.get(f"{stage}_segmental_loss_events", 0))
        total = gain + loss
        balance[stage] = {
            "segmental_event_count": total,
            "gain_count": gain,
            "loss_count": loss,
            "gain_fraction": None if total == 0 else gain / total,
            "loss_fraction": None if total == 0 else loss / total,
        }

    attempted_children = int(diagnostic_totals.get("attempted_children", 0))
    viability_rejections = int(diagnostic_totals.get("viability_rejections", 0))
    rejection_counts = {
        "all_zero_genome": int(
            diagnostic_totals.get("all_zero_genome_rejections", 0)
        ),
        "crucial_bin_zero": int(
            diagnostic_totals.get("crucial_bin_zero_rejections", 0)
        ),
    }
    return {
        "segmental_gain_loss_balance": balance,
        "interval_footprint_length": {
            stage: {
                direction: _histogram_number_summary(
                    interval_histograms[stage][direction]
                )
                for direction in ("gain", "loss")
            }
            for stage in ("proposed", "retained")
        },
        "interval_footprint_count": {
            stage: {
                direction: int(sum(interval_histograms[stage][direction].values()))
                for direction in ("gain", "loss")
            }
            for stage in ("proposed", "retained")
        },
        "viability": {
            "attempted_child_count": attempted_children,
            "rejection_count": viability_rejections,
            "rejection_fraction": (
                None
                if attempted_children == 0
                else viability_rejections / attempted_children
            ),
            "rejection_count_by_reason": rejection_counts,
            "rejection_fraction_by_reason": {
                reason: None if attempted_children == 0 else count / attempted_children
                for reason, count in rejection_counts.items()
            },
        },
    }


def _truth_event_breakdown(simulator: CancerCellEvolutionSimulator) -> Counter:
    counts: Counter[str] = Counter()
    for _, _, edge_data in simulator.tree.edges(data=True):
        for event in edge_data.get("events", ()):
            counts["truth_events"] += 1
            event_class = event["event_class"]
            if event_class == "whole_genome_doubling":
                counts["whole_genome_doublings"] += 1
                continue
            counts[f"{event_class}_{event['direction']}"] += 1
            if event_class == "interval_mode_cna":
                counts[f"interval_footprint_{event['footprint_direction']}"] += 1
                counts[
                    "interval_singleton"
                    if event["footprint_length"] == 1
                    else "interval_multi_position"
                ] += 1
                if event["direction"] == "gain":
                    counts[f"interval_gain_operator_{event['operator']}"] += 1
                    counts[f"interval_gain_magnitude_{event['magnitude']}"] += 1
    return counts


def _run_one(
    config: Mapping[str, Any],
    *,
    replicate_index: int,
    seed: int,
    generations: int,
    simulator_factory: SimulatorFactory,
) -> dict[str, Any]:
    simulator = simulator_factory(config, seed)
    failure: SimulationResourceLimitExceeded | None = None
    try:
        simulator.run_simulation()
    except SimulationResourceLimitExceeded as exc:
        failure = exc

    counts = Counter(
        int(data["generation"])
        for _, data in simulator.tree.nodes(data=True)
        if data.get("generation") is not None
    )
    outcome = dict(simulator.tree.graph["simulation_outcome"])
    status = str(outcome["status"])
    if failure is not None:
        complete_through_generation = int(failure.generation) - 1
        partial_generation = int(failure.generation)
    elif status == "extinct":
        complete_through_generation = int(outcome["extinction_generation"])
        partial_generation = None
    else:
        complete_through_generation = generations
        partial_generation = None

    diagnostics = simulator.diagnostics_snapshot()

    return {
        "replicate_index": replicate_index,
        "seed": seed,
        "status": status,
        "complete_through_generation": complete_through_generation,
        "partial_generation": partial_generation,
        "nodes_by_generation": {
            str(generation): int(counts.get(generation, 0))
            for generation in range(0, max(counts, default=0) + 1)
        },
        "total_retained_nodes": len(simulator.tree),
        "simulation_outcome": outcome,
        "diagnostic_totals": diagnostics["totals"],
        "truth_event_breakdown": dict(
            sorted(_truth_event_breakdown(simulator).items())
        ),
        "_zero_burden_by_generation": diagnostics[
            "zero_burden_by_generation"
        ],
        "_interval_footprint_length_histograms": diagnostics[
            "interval_footprint_length_histograms"
        ],
    }


def _prediction_parameters(config: Mapping[str, Any]) -> dict[str, float]:
    inputs = load_simulator_inputs(config)
    no_start_probability = math.prod(
        1.0 - genome_bin.cna_event_probability
        for genome_bin in inputs.genome_bins
    )
    at_least_one_start = 1.0 - no_start_probability
    attempts = float(
        int(config["BASELINE_DESCENDANT_ATTEMPTS"])
        + int(config["OFFSPRING_PARAMETER"])
    )
    if attempts != 2.0:
        raise ValueError("Growth prediction is defined here for exactly two attempts.")
    return {
        "probability_at_least_one_segmental_start_per_attempt": at_least_one_start,
        "earlier_first_order_growth_factor": 1.0 + attempts * at_least_one_start,
        "distinct_child_no_recurrence_growth_factor": (
            attempts * at_least_one_start
            + 1.0
            - at_least_one_start**attempts
        ),
    }


def _aggregate_report(
    records: list[dict[str, Any]],
    *,
    generations: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    prediction = _prediction_parameters(config)
    earlier_factor = prediction["earlier_first_order_growth_factor"]
    corrected_factor = prediction["distinct_child_no_recurrence_growth_factor"]
    rows = []
    for generation in range(generations + 1):
        complete_values = [
            int(record["nodes_by_generation"].get(str(generation), 0))
            for record in records
            if record["complete_through_generation"] >= generation
        ]
        partial_values = [
            int(record["nodes_by_generation"].get(str(generation), 0))
            for record in records
            if record["partial_generation"] == generation
        ]
        summary = _number_summary(complete_values)
        earlier_prediction = earlier_factor**generation
        corrected_prediction = corrected_factor**generation
        rows.append(
            {
                "generation": generation,
                "complete_run_count": len(complete_values),
                "partial_failure_run_count": len(partial_values),
                "complete_counts": summary,
                "partial_failure_counts": _number_summary(partial_values),
                "earlier_first_order_prediction": earlier_prediction,
                "distinct_child_no_recurrence_prediction": corrected_prediction,
                "median_to_earlier_prediction_ratio": (
                    None
                    if summary is None
                    else summary["median"] / earlier_prediction
                ),
                "median_to_distinct_child_prediction_ratio": (
                    None
                    if summary is None
                    else summary["median"] / corrected_prediction
                ),
            }
        )

    diagnostic_totals: Counter[str] = Counter()
    truth_event_totals: Counter[str] = Counter()
    for record in records:
        diagnostic_totals.update(record["diagnostic_totals"])
        truth_event_totals.update(record["truth_event_breakdown"])
    status_counts = Counter(record["status"] for record in records)
    failure_generations = [
        int(record["partial_generation"])
        for record in records
        if record["partial_generation"] is not None
    ]
    return {
        "outcome_counts": dict(sorted(status_counts.items())),
        "resource_failure_generation": _number_summary(failure_generations),
        "total_retained_nodes": _number_summary(
            [int(record["total_retained_nodes"]) for record in records]
        ),
        "prediction": prediction,
        "diagnostic_totals": dict(sorted(diagnostic_totals.items())),
        "truth_event_totals": dict(sorted(truth_event_totals.items())),
        "event_selection": _event_selection_summary(records, diagnostic_totals),
        "zero_burden_by_generation": _zero_burden_summary(
            records,
            generations=generations,
            genome_length=int(config["GENOME_LENGTH"]),
        ),
        "by_generation": rows,
    }


def run_growth_probe(
    *,
    base_config_path: Path | str = DEFAULT_BASE_CONFIG,
    generations: int = DEFAULT_GENERATIONS,
    replicates: int = DEFAULT_REPLICATES,
    base_seed: int = DEFAULT_BASE_SEED,
    simulator_factory: SimulatorFactory | None = None,
    created_at_utc: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.is_file():
        raise ValueError(f"Base config is not a file: {base_config_path}")
    if isinstance(generations, bool) or not isinstance(generations, int):
        raise ValueError("generations must be an integer.")
    if generations < 1:
        raise ValueError("generations must be >= 1.")
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise ValueError("replicates must be an integer.")
    if not 1 <= replicates <= MAX_REPLICATES:
        raise ValueError(f"replicates must be in [1,{MAX_REPLICATES}].")

    base_config = read_json(base_config_path)
    _validate_standard_base_config(base_config)
    config = dict(base_config)
    config["NUMBER_OF_GENERATIONS"] = generations
    load_simulator_inputs(config)
    factory = simulator_factory or (
        lambda mapping, seed: CancerCellEvolutionSimulator(mapping, seed=seed)
    )

    records = []
    for replicate_index in range(replicates):
        seed = _replicate_seed(base_seed, replicate_index)
        records.append(
            _run_one(
                config,
                replicate_index=replicate_index,
                seed=seed,
                generations=generations,
                simulator_factory=factory,
            )
        )
        if progress and ((replicate_index + 1) % 10 == 0 or replicate_index == 0):
            print(
                f"completed {replicate_index + 1}/{replicates} trees",
                file=sys.stderr,
                flush=True,
            )
        gc.collect()

    summary = _aggregate_report(
        records,
        generations=generations,
        config=config,
    )
    public_records = [
        {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
        for record in records
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "scientific_role": {
            "paper_evidence_allowed": False,
            "simulation_only": True,
            "truth_trees_serialized": False,
            "biopsies_sampled": False,
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
            "effective_config": config,
            "effective_config_canonical_sha256": canonical_json_sha256(config),
            "generations": generations,
            "replicates": replicates,
            "base_seed": base_seed,
            "seed_namespace": SEED_NAMESPACE,
            "summary_quantile_method": "numpy_linear",
        },
        "summary": summary,
        "runs": public_records,
        "provenance": {
            "source_module": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "source_module_sha256": _file_sha256(Path(__file__)),
            "simulator_semantic_version": SIMULATOR_SEMANTIC_VERSION,
            "numpy_version": np.__version__,
            "bit_generator": type(np.random.default_rng().bit_generator).__name__,
            "reads_existing_result_corpus": False,
            "writes_truth_trees_or_profiles": False,
        },
    }
    validate_growth_report(report)
    return report


def validate_growth_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown simulator-growth report schema.")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("Simulator-growth report has the wrong analysis role.")
    role = report.get("scientific_role")
    if not isinstance(role, Mapping) or role.get("paper_evidence_allowed") is not False:
        raise ValueError("Simulator-growth probe must be ineligible as paper evidence.")
    if any(
        role.get(field) is not False
        for field in ("biopsies_sampled", "cnp2cnp_run", "reconstruction_run", "evaluation_run")
    ):
        raise ValueError("Simulator-growth probe must stop before downstream stages.")
    records = report.get("runs")
    if not isinstance(records, list) or not records:
        raise ValueError("Simulator-growth report must contain run records.")
    expected_indices = list(range(len(records)))
    if [record.get("replicate_index") for record in records] != expected_indices:
        raise ValueError("Simulator-growth runs are missing or out of order.")
    if any(any(str(key).startswith("_") for key in record) for record in records):
        raise ValueError("Simulator-growth report exposes a private working field.")
    zero_rows = report.get("summary", {}).get("zero_burden_by_generation", [])
    if any(row.get("all_zero_profile_count") != 0 for row in zero_rows):
        raise ValueError("CTBF v5 retained an all-zero profile.")
    json.dumps(report, sort_keys=True, allow_nan=False)


def format_generation_table(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"outcomes: {json.dumps(summary['outcome_counts'], sort_keys=True)}",
        (
            "prediction factors: earlier_first_order="
            f"{summary['prediction']['earlier_first_order_growth_factor']:.6f}, "
            "distinct_child_no_recurrence="
            f"{summary['prediction']['distinct_child_no_recurrence_growth_factor']:.6f}"
        ),
        "",
        "| generation | complete n | partial n | mean | median | min | q25 | q75 | max | earlier prediction | corrected prediction | median/earlier |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["by_generation"]:
        stats = row["complete_counts"]
        values = (
            ("NA",) * 6
            if stats is None
            else (
                f"{stats['mean']:.2f}",
                f"{stats['median']:.2f}",
                str(stats["minimum"]),
                f"{stats['q25']:.2f}",
                f"{stats['q75']:.2f}",
                str(stats["maximum"]),
            )
        )
        ratio = row["median_to_earlier_prediction_ratio"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["generation"]),
                    str(row["complete_run_count"]),
                    str(row["partial_failure_run_count"]),
                    *values,
                    f"{row['earlier_first_order_prediction']:.2f}",
                    f"{row['distinct_child_no_recurrence_prediction']:.2f}",
                    "NA" if ratio is None else f"{ratio:.3f}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "CN0 burden across retained profiles from complete runs:",
            "",
            "| generation | profiles | median zero % | p95 zero % | max zero % | median longest run | max longest run | >=10% | >=25% | >=50% |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["zero_burden_by_generation"]:
        fraction = row["zero_bin_fraction"]
        run = row["longest_contiguous_zero_run_bins"]
        thresholds = row["profiles_at_or_above_zero_fraction"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["generation"]),
                    str(row["profile_count"]),
                    "NA" if fraction is None else f"{100 * fraction['median']:.2f}",
                    "NA" if fraction is None else f"{100 * fraction['p95']:.2f}",
                    "NA" if fraction is None else f"{100 * fraction['maximum']:.2f}",
                    "NA" if run is None else f"{run['median']:.2f}",
                    "NA" if run is None else f"{run['maximum']:.0f}",
                    str(thresholds["10pct"]["profile_count"]),
                    str(thresholds["25pct"]["profile_count"]),
                    str(thresholds["50pct"]["profile_count"]),
                )
            )
            + " |"
        )

    selection = summary["event_selection"]
    proposed = selection["segmental_gain_loss_balance"]["proposed"]
    retained = selection["segmental_gain_loss_balance"]["retained"]
    viability = selection["viability"]
    lines.extend(
        [
            "",
            (
                "segmental gain/loss: proposed="
                f"{proposed['gain_count']}/{proposed['loss_count']}, "
                "retained="
                f"{retained['gain_count']}/{retained['loss_count']}"
            ),
            (
                "viability rejections: total="
                f"{viability['rejection_count']}/"
                f"{viability['attempted_child_count']}, reasons="
                f"{json.dumps(viability['rejection_count_by_reason'], sort_keys=True)}"
            ),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a compact non-paper CTBF v5 representative-growth probe."
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    report = run_growth_probe(
        base_config_path=args.base_config,
        generations=args.generations,
        replicates=args.replicates,
        base_seed=args.base_seed,
        progress=args.progress,
    )
    write_json_atomic(args.output.expanduser().resolve(), report)
    print(format_generation_table(report))


if __name__ == "__main__":
    main()
