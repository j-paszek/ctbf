"""Deterministic post-raw-close analysis for CTBF v5 paper artifacts.

The analyzer refuses to read result rows until the raw checksum closure is
valid.  G2-01-A uses the status/schema audit for its toy smoke case; the same
interface is the entry point for registered estimands after an owner-run raw
experiment closes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

import networkx as nx
import numpy as np

from evaluation_contract import validate_evaluation_result

from algorithm_evaluation.paper_pipeline_contract import (
    ANALYSIS_SCHEMA_VERSION,
    EXPECTED_INVENTORY_SCHEMA_VERSION,
    REGISTERED_ARM_SPECS,
    REGISTERED_CLEAN_EXPERIMENT,
    condition_id,
    file_sha256,
    fixed_label_ad_f1,
    read_json,
    validate_checksum_closure,
    validate_status_record,
    write_json_atomic,
)


def _relative_paths(inventory: Mapping[str, Any], key: str) -> list[str]:
    paths = inventory.get(key)
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError(f"Expected inventory field {key!r} must be a list of paths.")
    return paths


def audit_closed_raw_artifacts(output_root: Path | str) -> dict[str, Any]:
    """Validate raw closure and return schema/status counts without aggregation."""
    output_root = Path(output_root).resolve()
    validate_checksum_closure(
        output_root,
        "raw_checksums.sha256",
        include_analysis=False,
    )
    inventory = read_json(output_root / "expected_inventory.json")
    if inventory.get("schema_version") != EXPECTED_INVENTORY_SCHEMA_VERSION:
        raise ValueError("Unknown expected-inventory schema.")

    missing = [
        relative
        for relative in _relative_paths(inventory, "raw_files")
        if not (output_root / relative).is_file()
    ]
    if missing:
        raise ValueError(f"Raw expected inventory is incomplete ({len(missing)} missing).")

    status_counts = Counter()
    status_code_counts = Counter()
    for relative in _relative_paths(inventory, "status_files"):
        record = read_json(output_root / relative)
        validate_status_record(record)
        status_counts[record["status"]] += 1
        status_code_counts[record["code"]] += 1

    nested_locations = inventory.get("nested_status_records", [])
    if not isinstance(nested_locations, list):
        raise ValueError("Expected inventory nested_status_records must be a list.")
    for location in nested_locations:
        record = read_json(output_root / location["path"])
        field = location["field"]
        if field not in record and location.get("optional_on_success"):
            continue
        nested = record.get(field)
        if not isinstance(nested, dict):
            raise ValueError(f"Missing nested status {location['path']}:{field}.")
        validate_status_record(nested)
        status_counts[nested["status"]] += 1
        status_code_counts[nested["code"]] += 1

    evaluation_counts = Counter()
    coverage_below_one = 0
    for relative in _relative_paths(inventory, "evaluation_files"):
        result = read_json(output_root / relative)
        validate_evaluation_result(result)
        evaluation_counts[result["status"]] += 1
        if result["status"] == "success":
            coverage = result["inputs"]["observation_label_coverage"]["fraction"]
            coverage_below_one += float(coverage) < 1.0

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "closed_raw_schema_and_status_audit",
        "raw_checksum_file_sha256": file_sha256(output_root / "raw_checksums.sha256"),
        "expected_raw_file_count": len(_relative_paths(inventory, "raw_files")),
        "status_counts": dict(sorted(status_counts.items())),
        "status_code_counts": dict(sorted(status_code_counts.items())),
        "evaluation_counts": dict(sorted(evaluation_counts.items())),
        "successful_evaluations_below_full_coverage": coverage_below_one,
        "raw_artifact_metrics_inspected_only_after_checksum_close": True,
    }


def _deserialize_tree(serialized: Mapping[str, Any]) -> nx.DiGraph:
    if serialized.get("directed") is not True or serialized.get("multigraph") is True:
        raise ValueError("Analysis requires a serialized simple directed tree.")
    tree = nx.DiGraph()
    tree.graph.update(serialized.get("graph_attributes", {}))
    for raw_node in serialized.get("nodes", []):
        node = dict(raw_node)
        node_id = node.pop("id")
        tree.add_node(node_id, **node)
    for raw_edge in serialized.get("links", []):
        edge = dict(raw_edge)
        source = edge.pop("source")
        target = edge.pop("target")
        tree.add_edge(source, target, **edge)
    return tree


def _numeric_summary(values) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _bootstrap_interval(
    values,
    *,
    repetitions: int,
    seed: int,
    chunk_size: int = 10_000,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {
            "status": "undefined_no_complete_blocks",
            "lower": None,
            "upper": None,
            "repetitions": repetitions,
        }
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    means = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "status": "success",
        "lower": float(lower),
        "upper": float(upper),
        "repetitions": repetitions,
        "interval": "two_sided_percentile_95_percent",
        "quantile_method": "numpy_default_linear",
    }


def _sign_flip_test(
    values,
    *,
    repetitions: int,
    seed: int,
    chunk_size: int = 20_000,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {
            "status": "undefined_no_complete_blocks",
            "p_value": None,
            "repetitions": repetitions,
        }
    observed = abs(float(values.mean()))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    extreme = 0
    for start in range(0, repetitions, chunk_size):
        count = min(chunk_size, repetitions - start)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8)
        signs = signs * 2 - 1
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
    return {
        "status": "success",
        "p_value": float((extreme + 1) / (repetitions + 1)),
        "extreme_draw_count": extreme,
        "repetitions": repetitions,
        "tail": "two_sided_absolute_mean",
        "monte_carlo_correction": "plus_one_numerator_and_denominator",
    }


def _cohen_dz(values) -> dict[str, Any]:
    values = [float(value) for value in values]
    if len(values) < 2:
        return {"status": "undefined_fewer_than_two_blocks", "value": None}
    deviation = statistics.stdev(values)
    if deviation == 0:
        return {"status": "undefined_zero_standard_deviation", "value": None}
    return {"status": "success", "value": statistics.fmean(values) / deviation}


def _holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    available = sorted(
        ((name, value) for name, value in p_values.items() if value is not None),
        key=lambda item: (item[1], item[0]),
    )
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    total = len(p_values)
    for rank, (name, value) in enumerate(available):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return adjusted


class _CleanArtifactReader:
    def __init__(self, output_root: Path, inventory: Mapping[str, Any]):
        self.output_root = output_root
        self.inventory = inventory
        self.cases = list(inventory.get("cases", []))
        if len(self.cases) != inventory.get("case_count"):
            raise ValueError("Expected inventory case metadata is incomplete.")
        self._arm_cache = {}
        self._input_cache = {}
        self._truth_cache = {}

    def arm(self, case_id_value: str, condition: str, arm_id: str):
        key = (case_id_value, condition, arm_id)
        if key not in self._arm_cache:
            base = (
                self.output_root
                / "cases"
                / case_id_value
                / "conditions"
                / condition
                / "arms"
                / arm_id
            )
            status = read_json(base / "status.json")
            validate_status_record(status)
            evaluation = read_json(base / "evaluation.json")
            validate_evaluation_result(evaluation)
            self._arm_cache[key] = (status, evaluation)
        return self._arm_cache[key]

    def metric(self, case_id_value: str, condition: str, arm_id: str, metric: str):
        status, evaluation = self.arm(case_id_value, condition, arm_id)
        if status["status"] != "success" or evaluation["status"] != "success":
            return None
        if evaluation["inputs"]["observation_label_coverage"]["fraction"] < 1.0:
            return None
        return float(evaluation["metrics"][metric])

    def input(self, case_id_value: str, condition: str):
        key = (case_id_value, condition)
        if key not in self._input_cache:
            self._input_cache[key] = read_json(
                self.output_root
                / "cases"
                / case_id_value
                / "conditions"
                / condition
                / "input.json"
            )
        return self._input_cache[key]

    def reconstruction_tree(self, case_id_value: str, condition: str, arm_id: str):
        record = read_json(
            self.output_root
            / "cases"
            / case_id_value
            / "conditions"
            / condition
            / "arms"
            / arm_id
            / "reconstruction.json"
        )
        if record.get("status") != "success":
            return None
        return _deserialize_tree(record["tree"])

    def truth_tree(self, case_id_value: str):
        if case_id_value not in self._truth_cache:
            record = read_json(
                self.output_root / "cases" / case_id_value / "truth.json"
            )
            self._truth_cache[case_id_value] = _deserialize_tree(record["tree"])
        return self._truth_cache[case_id_value]


def _paired_rows(
    reader: _CleanArtifactReader,
    *,
    condition: str,
    left_arm: str,
    right_arm: str,
    metric: str,
):
    rows = []
    for case in reader.cases:
        left = reader.metric(case["case_id"], condition, left_arm, metric)
        right = reader.metric(case["case_id"], condition, right_arm, metric)
        if left is None or right is None:
            continue
        rows.append(
            {
                "case_id": case["case_id"],
                "replicate": int(case["replicate"]),
                "regime_id": case["regime_id"],
                "effect": left - right,
            }
        )
    return rows


def _complete_block_values(rows, regimes):
    by_replicate = defaultdict(dict)
    for row in rows:
        by_replicate[row["replicate"]][row["regime_id"]] = row["effect"]
    return {
        replicate: statistics.fmean(values[regime] for regime in regimes)
        for replicate, values in by_replicate.items()
        if set(values) == set(regimes)
    }


def _effect_analysis(
    rows,
    *,
    regimes,
    material_threshold: float,
    bootstrap_repetitions: int,
    sign_flip_repetitions: int,
    bootstrap_seed: int,
    sign_flip_seed: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    block_values = _complete_block_values(rows, regimes)
    ordered_blocks = [block_values[key] for key in sorted(block_values)]
    effects = [row["effect"] for row in rows]
    by_regime = defaultdict(list)
    for row in rows:
        by_regime[row["regime_id"]].append(row["effect"])
    interval = _bootstrap_interval(
        ordered_blocks,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    sign_flip = _sign_flip_test(
        ordered_blocks,
        repetitions=sign_flip_repetitions,
        seed=sign_flip_seed,
    )
    mean = statistics.fmean(ordered_blocks) if ordered_blocks else None
    return {
        "complete_case_pair_count": len(rows),
        "complete_block_count": len(ordered_blocks),
        "case_effect_summary": _numeric_summary(effects),
        "block_effect_summary": _numeric_summary(ordered_blocks),
        "regime_effect_summaries": {
            regime: _numeric_summary(by_regime.get(regime, [])) for regime in regimes
        },
        "bootstrap_interval": interval,
        "sign_flip": sign_flip,
        "cohen_dz": _cohen_dz(ordered_blocks),
        "wins_ties_losses": {
            "wins": sum(value > tie_tolerance for value in effects),
            "ties": sum(abs(value) <= tie_tolerance for value in effects),
            "losses": sum(value < -tie_tolerance for value in effects),
            "tolerance": tie_tolerance,
        },
        "material_threshold": material_threshold,
        "unadjusted_statistical_support": bool(
            mean is not None
            and mean >= material_threshold
            and interval["lower"] is not None
            and interval["lower"] > 0.0
            and sign_flip["p_value"] is not None
            and sign_flip["p_value"] < 0.05
        ),
    }


def _fixed_labels_from_input(payload: Mapping[str, Any]) -> list[Any]:
    return sorted(
        {
            state["state_label"]
            for level in payload["levels"]
            for state in level["states"]
        },
        key=lambda value: (f"{type(value).__module__}.{type(value).__qualname__}", str(value)),
    )


def _fixed_label_rows(reader: _CleanArtifactReader, contrast_id: str, low: str, high: str):
    rows = []
    records = []
    for case in reader.cases:
        case_id_value = case["case_id"]
        fixed_labels = _fixed_labels_from_input(reader.input(case_id_value, low))
        low_status, _low_evaluation = reader.arm(case_id_value, low, "temporal_minimum")
        high_status, _high_evaluation = reader.arm(case_id_value, high, "temporal_minimum")
        record = {
            "schema_version": "ctbf-v5-fixed-label-contrast-record-v1",
            "contrast_id": contrast_id,
            "case_id": case_id_value,
            "replicate": case["replicate"],
            "regime_id": case["regime_id"],
            "low_condition": low,
            "high_condition": high,
            "fixed_label_source_condition": low,
        }
        if low_status["status"] != "success" or high_status["status"] != "success":
            record.update({"status": "not_run_dependency", "dependency": "native_arm_status"})
            records.append(record)
            continue
        try:
            truth = reader.truth_tree(case_id_value)
            low_tree = reader.reconstruction_tree(case_id_value, low, "temporal_minimum")
            high_tree = reader.reconstruction_tree(case_id_value, high, "temporal_minimum")
            if low_tree is None or high_tree is None:
                raise ValueError("Successful native arm is missing its reconstruction tree.")
            low_result = fixed_label_ad_f1(truth, low_tree, fixed_labels)
            high_result = fixed_label_ad_f1(truth, high_tree, fixed_labels)
            effect = high_result["metrics"]["ad_f1"] - low_result["metrics"]["ad_f1"]
            record.update(
                {
                    "status": "success",
                    "low": low_result,
                    "high": high_result,
                    "effect_high_minus_low": effect,
                }
            )
            rows.append(
                {
                    "case_id": case_id_value,
                    "replicate": int(case["replicate"]),
                    "regime_id": case["regime_id"],
                    "effect": effect,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "failure",
                    "failure": {
                        "type": type(exc).__name__,
                        "message": str(exc)[:4096],
                    },
                }
            )
        records.append(record)
    return rows, records


def _failure_gate(reader: _CleanArtifactReader, regimes, anchor: str, contract):
    pair_counts = {}
    arm_failure_rates = {}
    for regime in regimes:
        cases = [case for case in reader.cases if case["regime_id"] == regime]
        complete_pairs = 0
        failures = {"temporal_minimum": 0, "temporal_minimum_no_time": 0}
        for case in cases:
            statuses = {}
            for arm_id in failures:
                status, evaluation = reader.arm(case["case_id"], anchor, arm_id)
                statuses[arm_id] = status["status"] == "success" and evaluation["status"] == "success"
                failures[arm_id] += not statuses[arm_id]
            complete_pairs += all(statuses.values())
        pair_counts[regime] = complete_pairs
        arm_failure_rates[regime] = {
            arm_id: failures[arm_id] / len(cases) if cases else None
            for arm_id in failures
        }
    primary_rows = _paired_rows(
        reader,
        condition=anchor,
        left_arm="temporal_minimum",
        right_arm="temporal_minimum_no_time",
        metric="ad_f1",
    )
    complete_blocks = len(_complete_block_values(primary_rows, regimes))
    maximum_rate = max(
        rate
        for values in arm_failure_rates.values()
        for rate in values.values()
        if rate is not None
    )
    passed = (
        complete_blocks >= contract["minimum_complete_three_regime_blocks"]
        and all(
            count >= contract["minimum_complete_primary_pairs_per_regime"]
            for count in pair_counts.values()
        )
        and maximum_rate <= contract["maximum_primary_arm_failure_rate_per_regime"]
    )
    return {
        "passed": passed,
        "complete_three_regime_blocks": complete_blocks,
        "complete_primary_pairs_per_regime": pair_counts,
        "primary_arm_failure_rates_per_regime": arm_failure_rates,
        "maximum_primary_arm_failure_rate": maximum_rate,
        "thresholds": contract,
    }


def _resource_summaries(output_root: Path, inventory: Mapping[str, Any]):
    by_arm = defaultdict(lambda: defaultdict(list))
    for evaluation_path in inventory["evaluation_files"]:
        path = Path(evaluation_path)
        arm_id = path.parts[-2]
        resource = read_json(output_root / path.parent / "resources.json")
        for stage in ("reconstruction", "evaluation"):
            runtime = resource.get(stage)
            if not isinstance(runtime, dict):
                continue
            wall = runtime.get("wall_time_ns")
            if isinstance(wall, int):
                by_arm[arm_id][f"{stage}_wall_time_ns"].append(wall)
            memory = runtime.get("memory")
            if isinstance(memory, dict) and isinstance(memory.get("peak_rss_bytes"), int):
                by_arm[arm_id][f"{stage}_peak_rss_bytes"].append(
                    memory["peak_rss_bytes"]
                )
    arm_summary = {
        arm_id: {
            field: _numeric_summary(values)
            for field, values in sorted(fields.items())
        }
        for arm_id, fields in sorted(by_arm.items())
    }

    provider_wall = []
    provider_peak = []
    external_process_counts = []
    simulation_wall = []
    simulation_peak = []
    for case in inventory["cases"]:
        case_status = read_json(
            output_root / "cases" / case["case_id"] / "case_status.json"
        )
        simulation = case_status.get("simulation_runtime")
        if isinstance(simulation, dict):
            if isinstance(simulation.get("wall_time_ns"), int):
                simulation_wall.append(simulation["wall_time_ns"])
            memory = simulation.get("memory")
            if isinstance(memory, dict) and isinstance(memory.get("peak_rss_bytes"), int):
                simulation_peak.append(memory["peak_rss_bytes"])
        record = read_json(
            output_root
            / "cases"
            / case["case_id"]
            / "distances"
            / "minimum_bidirectional.json"
        )
        resources = record.get("resources")
        if isinstance(resources, dict):
            if isinstance(resources.get("wall_time_ns"), int):
                provider_wall.append(resources["wall_time_ns"])
            memory = resources.get("memory")
            if isinstance(memory, dict) and isinstance(memory.get("peak_rss_bytes"), int):
                provider_peak.append(memory["peak_rss_bytes"])
        provenance = record.get("provenance")
        if isinstance(provenance, dict) and isinstance(
            provenance.get("external_process_count"), int
        ):
            external_process_counts.append(provenance["external_process_count"])
    return {
        "simulation": {
            "wall_time_ns": _numeric_summary(simulation_wall),
            "peak_rss_bytes": _numeric_summary(simulation_peak),
        },
        "arms": arm_summary,
        "distance_provider": {
            "wall_time_ns": _numeric_summary(provider_wall),
            "peak_rss_bytes": _numeric_summary(provider_peak),
            "external_process_count": _numeric_summary(external_process_counts),
        },
    }


def analyze_clean_confirmation(
    output_root: Path,
    audit: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(output_root / "design_manifest.snapshot.json")
    clean = manifest["experiments"]["clean_confirmation"]
    inventory = read_json(output_root / "expected_inventory.json")
    if inventory["experiment_id"] != REGISTERED_CLEAN_EXPERIMENT:
        raise ValueError("Clean analyzer received a different experiment inventory.")
    if inventory["arm_record_count"] != clean["expected_arm_records"]:
        raise ValueError("Clean raw inventory arm count differs from preregistration.")
    reader = _CleanArtifactReader(output_root, inventory)
    regimes = list(clean["regime_ids"])
    anchor = condition_id(0.5, "L3")
    bootstrap_repetitions = int(clean["analysis"]["bootstrap_repetitions"])
    sign_flip_repetitions = int(clean["analysis"]["sign_flip_repetitions"])
    seed_record = next(
        record
        for record in manifest["seed_contract"]["experiments"]
        if record["experiment_id"] == REGISTERED_CLEAN_EXPERIMENT
    )
    bootstrap_seed = seed_record["analysis_seeds"]["block_bootstrap"]
    sign_flip_seed = seed_record["analysis_seeds"]["sign_flip"]
    tolerance = clean["analysis"]["win_tie_loss_tolerance"]

    def analyze_effect(rows, threshold):
        return _effect_analysis(
            rows,
            regimes=regimes,
            material_threshold=threshold,
            bootstrap_repetitions=bootstrap_repetitions,
            sign_flip_repetitions=sign_flip_repetitions,
            bootstrap_seed=bootstrap_seed,
            sign_flip_seed=sign_flip_seed,
            tie_tolerance=tolerance,
        )

    primary_ad = analyze_effect(
        _paired_rows(
            reader,
            condition=anchor,
            left_arm="temporal_minimum",
            right_arm="temporal_minimum_no_time",
            metric="ad_f1",
        ),
        clean["primary_hypothesis"]["minimum_material_mean_gain"],
    )
    primary_grf = analyze_effect(
        _paired_rows(
            reader,
            condition=anchor,
            left_arm="temporal_minimum",
            right_arm="temporal_minimum_no_time",
            metric="grf",
        ),
        clean["primary_hypothesis"]["minimum_material_mean_gain"],
    )
    gate = _failure_gate(reader, regimes, anchor, clean["failure_gate"])
    primary_ad["failure_gate_passed"] = gate["passed"]
    primary_ad["registered_support"] = bool(
        primary_ad["unadjusted_statistical_support"] and gate["passed"]
    )

    nested_specs = {
        "fraction_endpoint": (condition_id(0.25, "L3"), condition_id(1.0, "L3")),
        "level_endpoint": (condition_id(0.5, "L2"), condition_id(0.5, "L5")),
    }
    nested = {}
    fixed_records = []
    for contrast, (low, high) in nested_specs.items():
        rows, records = _fixed_label_rows(reader, contrast, low, high)
        fixed_records.extend(records)
        nested[contrast] = analyze_effect(
            rows,
            clean["nested_hypotheses"]["material_mean_gain"],
        )
    nested_adjusted = _holm_adjust(
        {
            name: analysis["sign_flip"]["p_value"]
            for name, analysis in nested.items()
        }
    )
    for name, analysis in nested.items():
        analysis["holm_adjusted_p_value"] = nested_adjusted[name]
        analysis["failure_gate_passed"] = gate["passed"]
        mean = analysis["block_effect_summary"]["mean"]
        interval = analysis["bootstrap_interval"]
        analysis["registered_support"] = bool(
            gate["passed"]
            and mean is not None
            and mean >= analysis["material_threshold"]
            and interval["lower"] is not None
            and interval["lower"] > 0
            and nested_adjusted[name] is not None
            and nested_adjusted[name] < 0.05
        )

    fully_specs = {
        "temporal_minus_rooted_labeled_nj": (
            "temporal_minimum",
            "rooted_labeled_nj",
        ),
        "anticentral_minus_rooted_labeled_nj": (
            "anticentral_parsimony",
            "rooted_labeled_nj",
        ),
        "temporal_minus_anticentral": (
            "temporal_minimum",
            "anticentral_parsimony",
        ),
    }
    fully = {
        name: analyze_effect(
            _paired_rows(
                reader,
                condition=anchor,
                left_arm=left,
                right_arm=right,
                metric="ad_f1",
            ),
            clean["secondary_families"]["fully_labeled_anchor"]["material_mean_gain"],
        )
        for name, (left, right) in fully_specs.items()
    }
    fully_adjusted = _holm_adjust(
        {name: result["sign_flip"]["p_value"] for name, result in fully.items()}
    )
    for name, result in fully.items():
        result["holm_adjusted_p_value"] = fully_adjusted[name]
        mean = result["block_effect_summary"]["mean"]
        interval = result["bootstrap_interval"]
        result["registered_support"] = bool(
            mean is not None
            and mean >= result["material_threshold"]
            and interval["lower"] is not None
            and interval["lower"] > 0
            and fully_adjusted[name] is not None
            and fully_adjusted[name] < 0.05
        )

    partial = analyze_effect(
        _paired_rows(
            reader,
            condition=anchor,
            left_arm="biopsy_guided_classical",
            right_arm="classical_partial",
            metric="grf",
        ),
        clean["secondary_families"]["partial_anchor"]["material_mean_gain"],
    )

    absolute = {}
    for arm_id, _algorithm in REGISTERED_ARM_SPECS:
        absolute[arm_id] = {}
        for metric in ("ad_f1", "grf"):
            values = [
                value
                for case in reader.cases
                if (value := reader.metric(case["case_id"], anchor, arm_id, metric))
                is not None
            ]
            absolute[arm_id][metric] = _numeric_summary(values)

    degeneracy = Counter()
    coverage = []
    for case in reader.cases:
        for arm_id, _algorithm in REGISTERED_ARM_SPECS:
            _status, evaluation = reader.arm(case["case_id"], anchor, arm_id)
            if evaluation["status"] == "success":
                degeneracy[evaluation["metrics"]["ad_f1_degeneracy"]] += 1
                coverage.append(
                    evaluation["inputs"]["observation_label_coverage"]["fraction"]
                )

    summary = {
        **dict(audit),
        "analysis_kind": "registered_clean_confirmation_estimands",
        "experiment_id": REGISTERED_CLEAN_EXPERIMENT,
        "analysis_seeds": {
            "block_bootstrap": bootstrap_seed,
            "sign_flip": sign_flip_seed,
            "common_random_draws_reused_across_contrasts": True,
        },
        "primary_order_effect": {
            "metric": "ad_f1",
            "contrast": "temporal_minimum_minus_temporal_minimum_no_time",
            "condition": anchor,
            "analysis": primary_ad,
            "paired_grf_diagnostic": primary_grf,
        },
        "failure_gate": gate,
        "nested_fixed_label_effects": nested,
        "secondary_fully_labeled_anchor": fully,
        "secondary_partial_anchor": partial,
        "absolute_anchor_summaries": absolute,
        "anchor_diagnostics": {
            "ad_f1_degeneracy_counts": dict(sorted(degeneracy.items())),
            "coverage_summary": _numeric_summary(coverage),
        },
        "resource_summaries": _resource_summaries(output_root, inventory),
        "fixed_label_case_record_count": len(fixed_records),
    }
    return summary, fixed_records


def write_analysis(output_root: Path | str, *, run_kind: str) -> Path:
    """Write a deterministic analysis only after validating raw closure."""
    output_root = Path(output_root).resolve()
    summary = audit_closed_raw_artifacts(output_root)
    summary["run_kind"] = run_kind
    fixed_records = []
    if run_kind == "registered_clean_confirmation":
        summary, fixed_records = analyze_clean_confirmation(output_root, summary)
        summary["run_kind"] = run_kind
    destination = output_root / "analysis" / ANALYSIS_SCHEMA_VERSION / "summary.json"
    for record in fixed_records:
        write_json_atomic(
            output_root
            / "analysis"
            / ANALYSIS_SCHEMA_VERSION
            / "fixed_label"
            / record["contrast_id"]
            / f"{record['case_id']}.json",
            record,
        )
    write_json_atomic(destination, summary)
    return destination


__all__ = [
    "analyze_clean_confirmation",
    "audit_closed_raw_artifacts",
    "write_analysis",
]
