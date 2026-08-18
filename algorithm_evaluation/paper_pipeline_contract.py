"""Strict CTBF v5 paper-runner contracts for G2-01-A.

This module contains only deterministic validation and transformation helpers.
It deliberately does not simulate, reconstruct, or inspect aggregate held-out
results.  The runner composes these helpers with the existing CTBF v5 module
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    DISTANCE_PROVENANCE_SCHEMA_VERSION,
    stable_distance_label_key,
    validate_distance_matrix,
)
from evaluation_contract import (
    AD_F1_SEMANTICS_VERSION,
    EVALUATION_RESULT_SCHEMA_VERSION,
    GRF_SEMANTICS_VERSION,
    LABEL_NORMALIZATION_VERSION,
    TREE_VALIDATION_VERSION,
    validate_rooted_labeled_tree,
)
from evaluator_full import normalize_cell_labels, prf1_iou, unique_ancestor_pair_set
from reconstructor_registry import resolve_reconstruction_algorithm
from simulator_config import load_simulator_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "experimental_description"
    / "g0_05a_v5_preregistration_manifest.json"
)

MANIFEST_SCHEMA_VERSION = "ctbf-v5-paper-preregistration-manifest-v1"
MANIFEST_ID = "ctbf-v5-paper-preregistration-v1"
# Set only when the owner approves the new CTBF v5 preregistration bytes.
APPROVED_MANIFEST_SHA256 = None
APPROVED_PROTOCOL_SHA256 = None

EXPERIMENT_STATUS_SCHEMA_VERSION = "ctbf-v5-experiment-status-v1"
CASE_SCHEMA_VERSION = "ctbf-v5-paper-case-v1"
OBSERVATION_SCHEMA_VERSION = "ctbf-v5-observation-set-v1"
FIXED_LABEL_AD_F1_SCHEMA_VERSION = "ctbf-fixed-label-ad-f1-v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "ctbf-v5-source-manifest-v1"
ENVIRONMENT_SCHEMA_VERSION = "ctbf-v5-environment-v1"
EXPECTED_INVENTORY_SCHEMA_VERSION = "ctbf-v5-expected-inventory-v1"
ANALYSIS_SCHEMA_VERSION = "ctbf-v5-paper-analysis-v1"

REGISTERED_ARM_SPECS = (
    ("classical_partial", "neighbor_joining_classical"),
    ("biopsy_guided_classical", "neighbor_joining_classical"),
    ("rooted_labeled_nj", "rooted_labeled_nj"),
    ("temporal_minimum", "temporal_cnp_arborescence"),
    ("temporal_minimum_no_time", "temporal_cnp_arborescence_no_time"),
    (
        "anticentral_parsimony",
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
    ),
)
REGISTERED_CLEAN_EXPERIMENT = "ctbf-v5-clean-confirmation-v1"
CALIBRATION_NAMESPACE = "ctbf-v5-g0-05a-calibration-v1"
CALIBRATION_STREAMS = ("simulation", "crucial", "wgd", "distance")

SOURCE_FREEZE_PATHS = (
    "experimental_description/v5_confirmatory_protocol.md",
    "experimental_description/g0_05a_v5_preregistration_manifest.json",
    "simulator_examples/paper_v5/clean_balanced.json",
    "simulator_examples/paper_v5/clean_interval_gain.json",
    "simulator_examples/paper_v5/clean_telomeric.json",
    "simulator_examples/paper_v5/wgd_1pct.json",
    "simulator_examples/paper_v5/crucial_10pct_control.json",
    "simulator_examples/paper_v5/crucial_10pct_survival.json",
    "algorithm_evaluation/paper_pipeline_contract.py",
    "algorithm_evaluation/paper_pipeline_runner.py",
    "algorithm_evaluation/paper_pipeline_analysis.py",
    "ctbf_constraints.py",
    "simulator.py",
    "simulator_config.py",
    "simulator_events.py",
    "ctbs.py",
    "ctbs_config.json",
    "ctbs_utils.py",
    "distance_semantics.py",
    "reconstructor.py",
    "reconstructor_algorithm_specs.py",
    "reconstructor_registry.py",
    "reconstructor_algorithms.py",
    "reconstructor_biopsy_blocks.py",
    "reconstructor_biopsy_guided.py",
    "reconstructor_biopsy_presets.py",
    "reconstructor_engine.py",
    "reconstructor_plausibility.py",
    "reconstructor_temporal.py",
    "reconstructor_utils.py",
    "evaluation_contract.py",
    "evaluator.py",
    "evaluator_full.py",
)

REQUIRED_ROOT_FILES = (
    "design_manifest.snapshot.json",
    "source_manifest.json",
    "environment.json",
    "run_status.json",
    "expected_inventory.json",
    "raw_checksums.sha256",
    "complete_checksums.sha256",
)


class PaperContractError(ValueError):
    """Typed refusal raised before or during a v5 paper run."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _error(code: str, message: str, **details: Any) -> PaperContractError:
    return PaperContractError(code, message, details=details)


def json_safe(value: Any) -> Any:
    """Return strict-JSON data, rejecting non-finite numbers."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise _error("nonfinite_json", "Non-finite values are forbidden in paper artifacts.")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise _error(
        "unsupported_json_value",
        f"Paper artifacts cannot serialize {type(value).__name__}.",
    )


def read_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise _error("invalid_json_root", f"{path} must contain a JSON object.")
    return value


def write_json_atomic(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(value)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
    os.replace(temporary, path)


def file_sha256(path: Path | str) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def derive_seed(namespace: str, stream: str, replicate: int) -> int:
    payload = f"{namespace}:{stream}:{int(replicate)}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "big")


def derive_analysis_seed(namespace: str, analysis_name: str) -> int:
    payload = f"{namespace}:analysis:{analysis_name}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "big")


def derive_generation_permutation_seed(
    experiment_id: str,
    sampling_seed: int,
    regime_id: str,
    generation: int,
) -> int:
    payload = (
        f"{experiment_id}:generation-permutation:{int(sampling_seed)}:"
        f"{regime_id}:{int(generation)}"
    ).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def canonical_seed_table(experiment: Mapping[str, Any]) -> list[dict[str, int]]:
    experiment_id = str(experiment["experiment_id"])
    streams = tuple(experiment["streams"])
    return [
        {
            "replicate": replicate,
            **{
                f"{stream}_seed": derive_seed(experiment_id, stream, replicate)
                for stream in streams
            },
        }
        for replicate in range(1, int(experiment["replicates"]) + 1)
    ]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return bytes_sha256(payload)


def _mapping_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    return {
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    }


def _require_exact_pair_difference(
    project_root: Path,
    left_path: str,
    right_path: str,
    expected_field: str,
) -> None:
    left = read_json(project_root / left_path)
    right = read_json(project_root / right_path)
    differences = _mapping_differences(left, right)
    if differences != {expected_field}:
        raise _error(
            "config_pair_difference",
            f"{left_path} and {right_path} must differ only in {expected_field}.",
            differences=sorted(differences),
        )


def _validate_strict_simulator_config(
    project_root: Path,
    config_path: str,
    *,
    expected_sha256: str,
    semantic_version: str,
    bed_path: str | None = None,
) -> Any:
    resolved_config = project_root / config_path
    resolved_bed = None if bed_path is None else project_root / bed_path
    inputs = load_simulator_inputs(
        str(resolved_config),
        None if resolved_bed is None else str(resolved_bed),
    )
    if inputs.config_sha256 != expected_sha256:
        raise _error("strict_config_hash", f"Strict parser hash mismatch for {config_path}.")
    if inputs.config.semantic_version != semantic_version:
        raise _error("simulator_semantics", f"Semantic version mismatch in {config_path}.")
    return inputs


def _validate_seed_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    seed_contract = manifest["seed_contract"]
    all_numeric: list[int] = []
    table_digests: dict[str, str] = {}
    for experiment in seed_contract["experiments"]:
        table = canonical_seed_table(experiment)
        digest = canonical_json_sha256(table)
        experiment_id = experiment["experiment_id"]
        if digest != experiment["canonical_seed_table_sha256"]:
            raise _error(
                "seed_table_digest",
                f"Seed-table digest mismatch for {experiment_id}.",
                expected=experiment["canonical_seed_table_sha256"],
                actual=digest,
            )
        table_digests[experiment_id] = digest
        for record in table:
            all_numeric.extend(
                value for key, value in record.items() if key.endswith("_seed")
            )
        for name, expected in experiment.get("analysis_seeds", {}).items():
            actual = derive_analysis_seed(experiment_id, name)
            if actual != expected:
                raise _error(
                    "analysis_seed",
                    f"Analysis seed mismatch for {experiment_id}:{name}.",
                    expected=expected,
                    actual=actual,
                )
            all_numeric.append(actual)

    expected_run_count = int(seed_contract["registered_run_stream_seed_count"])
    expected_analysis_count = int(seed_contract["registered_analysis_seed_count"])
    actual_total = len(all_numeric)
    if actual_total != expected_run_count + expected_analysis_count:
        raise _error(
            "seed_count",
            "Registered numeric seed count does not match the manifest.",
            actual=actual_total,
        )
    if len(set(all_numeric)) != len(all_numeric):
        raise _error("seed_collision", "Registered run and analysis seeds must be globally unique.")

    calibration = manifest["calibration_disclosure"]
    calibration_seeds = {
        derive_seed(calibration["namespace"], stream, index)
        for stream in calibration["streams"]
        for index in range(
            int(calibration["indices_per_stream"][0]),
            int(calibration["indices_per_stream"][1]) + 1,
        )
    }
    if len(calibration_seeds) != calibration["seed_count"]:
        raise _error("calibration_seed_collision", "Calibration seeds are not unique.")
    overlap = set(all_numeric) & calibration_seeds
    if overlap or calibration["held_out_numeric_seed_intersection_count"] != 0:
        raise _error(
            "seed_namespace_overlap",
            "Held-out and calibration numeric seeds must be disjoint.",
            overlap_count=len(overlap),
        )
    return {
        "run_seed_count": expected_run_count,
        "analysis_seed_count": expected_analysis_count,
        "calibration_seed_count": len(calibration_seeds),
        "table_digests": table_digests,
    }


def _validate_observation_contract(manifest: Mapping[str, Any]) -> None:
    design = manifest["shared_observation_design"]
    generations = design["maximal_generations"]
    if generations != [3, 4, 5, 6, 7]:
        raise _error("observation_generations", "Maximal generations changed after approval.")
    fractions = design["fractions"]
    if fractions != [0.25, 0.5, 0.75, 1.0]:
        raise _error("observation_fractions", "Observation fractions changed after approval.")
    schedules = design["level_schedules"]
    expected = {
        "L2": [3, 7],
        "L3": [3, 5, 7],
        "L4": [3, 4, 5, 7],
        "L5": [3, 4, 5, 6, 7],
    }
    if schedules != expected:
        raise _error("observation_schedules", "Level schedules changed after approval.")
    if int(design["condition_count"]) != len(fractions) * len(schedules):
        raise _error("observation_condition_count", "Observation condition count is invalid.")
    maximal = set(generations)
    if any(not set(schedule).issubset(maximal) for schedule in schedules.values()):
        raise _error("observation_schedule_membership", "A schedule is not a maximal-generation subset.")


def _validate_arm_contract(manifest: Mapping[str, Any]) -> None:
    actual = tuple(
        (record.get("arm_id"), record.get("algorithm"))
        for record in manifest["reconstruction_portfolio"]
    )
    if actual != REGISTERED_ARM_SPECS:
        raise _error("arm_portfolio", "The registered reconstruction portfolio changed.")
    for _arm_id, algorithm_name in actual:
        algorithm = resolve_reconstruction_algorithm(algorithm_name)
        if algorithm.__name__ != algorithm_name:
            raise _error(
                "algorithm_registry",
                f"Registry resolved {algorithm_name} as {algorithm.__name__}.",
            )


def _validate_artifact_and_metric_contracts(manifest: Mapping[str, Any]) -> None:
    artifact = manifest["artifact_contract"]
    expected_schemas = {
        "status_schema": EXPERIMENT_STATUS_SCHEMA_VERSION,
        "case_schema": CASE_SCHEMA_VERSION,
        "observation_schema": OBSERVATION_SCHEMA_VERSION,
        "evaluation_schema": EVALUATION_RESULT_SCHEMA_VERSION,
    }
    for field, expected in expected_schemas.items():
        if artifact.get(field) != expected:
            raise _error("artifact_schema", f"Manifest {field} is not {expected}.")
    if tuple(artifact.get("required_root_files", ())) != REQUIRED_ROOT_FILES:
        raise _error("artifact_root_files", "Required root artifact inventory changed.")
    if artifact.get("strict_json") is not True or artifact.get("nonfinite_json_allowed") is not False:
        raise _error("strict_json_contract", "Paper artifacts must use strict finite JSON.")

    evaluator = manifest["evaluator_contract"]
    expected_evaluator = {
        "native_result_schema": EVALUATION_RESULT_SCHEMA_VERSION,
        "label_normalization": LABEL_NORMALIZATION_VERSION,
        "tree_validation": TREE_VALIDATION_VERSION,
        "ad_f1": AD_F1_SEMANTICS_VERSION,
        "grf": GRF_SEMANTICS_VERSION,
        "fixed_label_nested_metric": FIXED_LABEL_AD_F1_SCHEMA_VERSION,
        "required_observation_coverage_fraction": 1.0,
    }
    for field, expected in expected_evaluator.items():
        if evaluator.get(field) != expected:
            raise _error("evaluator_contract", f"Manifest evaluator field {field} changed.")


def _validate_experiment_counts(manifest: Mapping[str, Any]) -> None:
    clean = manifest["experiments"]["clean_confirmation"]
    if clean["experiment_id"] != REGISTERED_CLEAN_EXPERIMENT:
        raise _error("clean_experiment_id", "The clean experiment id changed.")
    truth_cases = int(clean["replicate_blocks"]) * len(clean["regime_ids"])
    expected_arms = truth_cases * int(clean["conditions_per_truth"]) * int(clean["arms_per_condition"])
    if truth_cases != clean["truth_cases"] or expected_arms != clean["expected_arm_records"]:
        raise _error("clean_expected_count", "Clean experiment expected counts are inconsistent.")
    if int(clean["conditions_per_truth"]) != manifest["shared_observation_design"]["condition_count"]:
        raise _error("clean_condition_count", "Clean condition count disagrees with shared design.")
    if int(clean["arms_per_condition"]) != len(REGISTERED_ARM_SPECS):
        raise _error("clean_arm_count", "Clean arm count disagrees with the portfolio.")


def validate_manifest(
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
    enforce_approved_bytes: bool = True,
) -> dict[str, Any]:
    """Validate the complete preregistration boundary without writing output."""
    manifest_path = Path(manifest_path).resolve()
    if enforce_approved_bytes:
        if APPROVED_MANIFEST_SHA256 is None or APPROVED_PROTOCOL_SHA256 is None:
            raise _error(
                "v5_preregistration_not_frozen",
                "CTBF v5 paper execution is disabled until the owner approves "
                "new manifest and protocol bytes.",
            )
        if file_sha256(manifest_path) != APPROVED_MANIFEST_SHA256:
            raise _error(
                "manifest_bytes",
                "The preregistration manifest bytes differ from the owner-approved lock.",
            )
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise _error("manifest_schema", "Unsupported v5 paper manifest schema.")
    if manifest.get("manifest_id") != MANIFEST_ID:
        raise _error("manifest_id", "Unexpected preregistration manifest id.")
    if manifest.get("product_version") != "CTBF v5":
        raise _error("product_version", "Paper evidence must target CTBF v5.")
    if manifest.get("status") != "preregistered_before_held_out_generation":
        raise _error("manifest_status", "Manifest is not in the approved preregistered state.")
    if manifest.get("held_out_outputs_generated") is not False:
        raise _error("held_out_state", "Manifest reports that held-out outputs already exist.")
    if manifest.get("held_out_outputs_inspected") is not False:
        raise _error("held_out_state", "Manifest reports that held-out outputs were inspected.")
    if manifest.get("legacy_artifacts_admissible") is not False:
        raise _error(
            "legacy_artifact_policy",
            "Legacy CTBF v1--v4 evidence must remain excluded.",
        )
    approval = manifest["scientific_lock"]["owner_approval"]
    if approval.get("status") != "approved" or not approval.get("date") or not approval.get("record"):
        raise _error("owner_approval", "The frozen owner-approval record changed.")

    protocol_path = project_root / manifest["protocol"]
    if APPROVED_PROTOCOL_SHA256 is None or file_sha256(protocol_path) != APPROVED_PROTOCOL_SHA256:
        raise _error("protocol_bytes", "The owner-approved protocol bytes changed.")

    for group in ("clean_regimes",):
        for asset in manifest["input_assets"][group]:
            path = project_root / asset["path"]
            if file_sha256(path) != asset["sha256"]:
                raise _error("input_asset_hash", f"Hash mismatch for {asset['path']}.")
            config = read_json(path)
            if config.get("SIMULATOR_SEMANTIC_VERSION") != manifest["simulator_semantic_version"]:
                raise _error("simulator_semantics", f"Semantic version mismatch in {asset['path']}.")
            _validate_strict_simulator_config(
                project_root,
                asset["path"],
                expected_sha256=asset["sha256"],
                semantic_version=manifest["simulator_semantic_version"],
            )

    wgd = manifest["input_assets"]["wgd_pair"]
    crucial = manifest["input_assets"]["crucial_pair"]
    for path_field, hash_field in (
        ("control_path", "control_sha256"),
        ("wgd_path", "wgd_sha256"),
    ):
        if file_sha256(project_root / wgd[path_field]) != wgd[hash_field]:
            raise _error("input_asset_hash", f"Hash mismatch for {wgd[path_field]}.")
    for path_field, hash_field in (
        ("control_path", "control_sha256"),
        ("survival_path", "survival_sha256"),
    ):
        if file_sha256(project_root / crucial[path_field]) != crucial[hash_field]:
            raise _error("input_asset_hash", f"Hash mismatch for {crucial[path_field]}.")
    _require_exact_pair_difference(
        project_root, wgd["control_path"], wgd["wgd_path"], "WGD_PROBABILITY"
    )
    _require_exact_pair_difference(
        project_root,
        crucial["control_path"],
        crucial["survival_path"],
        "CRUCIAL_SURVIVAL_ENABLED",
    )
    _validate_strict_simulator_config(
        project_root,
        wgd["wgd_path"],
        expected_sha256=wgd["wgd_sha256"],
        semantic_version=manifest["simulator_semantic_version"],
    )
    for config_field, hash_field in (
        ("control_path", "control_sha256"),
        ("survival_path", "survival_sha256"),
    ):
        _validate_strict_simulator_config(
            project_root,
            crucial[config_field],
            expected_sha256=crucial[hash_field],
            semantic_version=manifest["simulator_semantic_version"],
        )
    selected = crucial["selected_zero_based_indices"]
    if len(selected) != crucial["selected_rows"] or len(set(selected)) != len(selected):
        raise _error("crucial_mask", "The preregistered crucial mask is malformed.")
    if any(index < 0 or index >= crucial["eligible_rows"] for index in selected):
        raise _error("crucial_mask", "A crucial-mask index is out of range.")
    crucial_inputs = load_simulator_inputs(
        str(project_root / crucial["survival_path"]),
    )
    actual_crucial_indices = [
        index
        for index, genome_bin in enumerate(crucial_inputs.genome_bins)
        if genome_bin.crucial
    ]
    if actual_crucial_indices != selected:
        raise _error(
            "crucial_mask",
            "Config crucial rows disagree with the preregistered mask indices.",
            actual=actual_crucial_indices,
            expected=selected,
        )

    _validate_observation_contract(manifest)
    _validate_arm_contract(manifest)
    _validate_artifact_and_metric_contracts(manifest)
    _validate_experiment_counts(manifest)
    seed_report = _validate_seed_contract(manifest)

    distance = manifest["distance_contract"]
    if (
        distance.get("semantic_version") != CNP2CNP_SEMANTICS_VERSION
        or distance.get("provenance_schema") != DISTANCE_PROVENANCE_SCHEMA_VERSION
        or distance.get("formula") != "min(d_any(u,v),d_any(v,u))"
        or distance.get("fallback_allowed") is not False
        or distance.get("ordered_triangle_fast_included") is not False
        or distance.get("directed_variant_included") is not False
    ):
        raise _error("distance_contract", "The bidirectional-minimum distance contract changed.")
    resources = manifest["resource_limits"]
    if resources.get("workers") != 1 or resources.get("max_unique_states_or_occurrences_per_synthetic_reconstruction_condition") != 248:
        raise _error("resource_contract", "Registered worker or reconstruction bound changed.")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": file_sha256(manifest_path),
        "protocol_sha256": file_sha256(protocol_path),
        "seed_validation": seed_report,
        "arm_count": len(REGISTERED_ARM_SPECS),
        "condition_count": manifest["shared_observation_design"]["condition_count"],
    }


@dataclass(frozen=True)
class ObservationCondition:
    condition_id: str
    fraction: float
    schedule_id: str
    generations: tuple[int, ...]
    cells_by_generation: tuple[tuple[Any, ...], ...]

    @property
    def occurrence_count(self) -> int:
        return sum(len(cells) for cells in self.cells_by_generation)

    @property
    def unique_labels(self) -> tuple[Any, ...]:
        labels = {
            cell.cell_id
            for cells in self.cells_by_generation
            for cell in cells
        }
        return tuple(sorted(labels, key=stable_distance_label_key))


@dataclass(frozen=True)
class NestedObservationDesign:
    maximal_cells_by_generation: Mapping[int, tuple[Any, ...]]
    conditions: Mapping[str, ObservationCondition]
    permutation_seeds: Mapping[int, int]


@dataclass(frozen=True)
class NestedDropoutDesign:
    """One frozen positive-bin draw field evaluated at nested severities."""

    profiles_by_rate: Mapping[float, Mapping[Any, tuple[int, ...]]]
    dropped_positive_bins_by_rate: Mapping[float, int]
    positive_bin_count: int


def _canonical_cell_key(cell: Any) -> tuple[Any, ...]:
    cnp = tuple(int(value) for value in np.asarray(cell.genome).tolist())
    label = str(cell.cell_id).strip()
    if not label:
        raise _error("empty_state_label", "Observed state labels must be non-empty.")
    return cnp, label


def condition_id(fraction: float, schedule_id: str) -> str:
    token = f"{float(fraction):.2f}".replace(".", "p")
    return f"f{token}_{schedule_id}"


def fraction_prefix_size(size: int, fraction: float) -> int:
    if size == 0:
        return 0
    return min(size, max(1, math.floor(float(fraction) * size)))


def sample_nested_observations(
    generation_cells: Mapping[int, Sequence[Any]],
    *,
    experiment_id: str,
    sampling_seed: int,
    regime_id: str,
    observation_contract: Mapping[str, Any],
) -> NestedObservationDesign:
    """Apply the registered one-permutation, nested-prefix observation design."""
    maximal_generations = tuple(observation_contract["maximal_generations"])
    fractions = tuple(float(value) for value in observation_contract["fractions"])
    schedules = observation_contract["level_schedules"]
    permuted: dict[int, tuple[Any, ...]] = {}
    permutation_seeds: dict[int, int] = {}
    label_to_cnp: dict[Any, tuple[int, ...]] = {}
    cnp_to_label: dict[tuple[int, ...], Any] = {}

    for generation in maximal_generations:
        canonical = sorted(tuple(generation_cells.get(generation, ())), key=_canonical_cell_key)
        if not canonical:
            raise _error(
                "empty_required_generation",
                f"Required generation {generation} is empty.",
                generation=generation,
            )
        for cell in canonical:
            cnp = _canonical_cell_key(cell)[0]
            label = cell.cell_id
            if label in label_to_cnp and label_to_cnp[label] != cnp:
                raise _error("state_label_collision", "One state label maps to multiple CNPs.")
            if cnp in cnp_to_label and cnp_to_label[cnp] != label:
                raise _error("state_profile_collision", "Distinct labels map to one exact CNP.")
            label_to_cnp[label] = cnp
            cnp_to_label[cnp] = label
        seed = derive_generation_permutation_seed(
            experiment_id, sampling_seed, regime_id, generation
        )
        permutation = np.random.Generator(np.random.PCG64(seed)).permutation(len(canonical))
        permuted[generation] = tuple(canonical[int(index)] for index in permutation)
        permutation_seeds[generation] = seed

    conditions: dict[str, ObservationCondition] = {}
    for fraction in fractions:
        selected_by_generation = {
            generation: tuple(
                sorted(
                    permuted[generation][
                        : fraction_prefix_size(len(permuted[generation]), fraction)
                    ],
                    key=_canonical_cell_key,
                )
            )
            for generation in maximal_generations
        }
        for schedule_id, generations in schedules.items():
            identifier = condition_id(fraction, schedule_id)
            conditions[identifier] = ObservationCondition(
                condition_id=identifier,
                fraction=fraction,
                schedule_id=schedule_id,
                generations=tuple(generations),
                cells_by_generation=tuple(
                    selected_by_generation[generation] for generation in generations
                ),
            )

    if len(conditions) != int(observation_contract["condition_count"]):
        raise _error("condition_count", "Nested sampler produced the wrong condition count.")
    maximal_canonical = {
        generation: tuple(sorted(cells, key=_canonical_cell_key))
        for generation, cells in permuted.items()
    }
    return NestedObservationDesign(
        maximal_cells_by_generation=maximal_canonical,
        conditions=conditions,
        permutation_seeds=permutation_seeds,
    )


def nested_positive_bin_dropout(
    state_profiles: Mapping[Any, Sequence[int]],
    *,
    dropout_seed: int,
    rates: Sequence[float],
) -> NestedDropoutDesign:
    """Apply the preregistered one-draw-field nested positive-bin dropout.

    Labels are opaque result/alignment keys.  Draw order is profile-first, so a
    bijective relabeling cannot change a state's perturbation.  Exact-profile
    collisions are a typed failure and are never silently merged.
    """
    normalized_rates = tuple(float(rate) for rate in rates)
    if normalized_rates != tuple(sorted(set(normalized_rates))):
        raise _error("dropout_rates", "Dropout rates must be sorted and unique.")
    if any(rate < 0.0 or rate > 1.0 for rate in normalized_rates):
        raise _error("dropout_rates", "Dropout rates must lie in [0, 1].")
    ordered = sorted(
        (
            (label, tuple(int(value) for value in profile))
            for label, profile in state_profiles.items()
        ),
        key=lambda item: (item[1], str(item[0]).strip()),
    )
    if not ordered:
        raise _error("dropout_states_empty", "Dropout requires at least one state.")
    lengths = {len(profile) for _label, profile in ordered}
    if len(lengths) != 1:
        raise _error("dropout_profile_length", "Dropout profiles must have equal length.")
    original_profiles = [profile for _label, profile in ordered]
    if len(set(original_profiles)) != len(original_profiles):
        raise _error("state_profile_collision", "Source states must have injective CNPs.")
    if any(value < 0 for profile in original_profiles for value in profile):
        raise _error("dropout_negative_copy_number", "CNP values must be nonnegative.")

    rng = np.random.Generator(np.random.PCG64(int(dropout_seed)))
    uniforms: dict[tuple[Any, int], float] = {}
    for label, profile in ordered:
        for index, value in enumerate(profile):
            if value > 0:
                uniforms[(label, index)] = float(rng.random())

    profiles_by_rate: dict[float, dict[Any, tuple[int, ...]]] = {}
    dropped_by_rate: dict[float, int] = {}
    for rate in normalized_rates:
        perturbed: dict[Any, tuple[int, ...]] = {}
        profile_owner: dict[tuple[int, ...], Any] = {}
        dropped = 0
        for label, profile in ordered:
            values = list(profile)
            for index, value in enumerate(profile):
                if value > 0 and uniforms[(label, index)] < rate:
                    values[index] = 0
                    dropped += 1
            result_profile = tuple(values)
            previous = profile_owner.get(result_profile)
            if previous is not None and previous != label:
                raise _error(
                    "perturbed_state_collision",
                    "Distinct labels became one exact perturbed CNP.",
                    rate=rate,
                    labels=json_safe([previous, label]),
                )
            profile_owner[result_profile] = label
            perturbed[label] = result_profile
        profiles_by_rate[rate] = perturbed
        dropped_by_rate[rate] = dropped
    return NestedDropoutDesign(
        profiles_by_rate=profiles_by_rate,
        dropped_positive_bins_by_rate=dropped_by_rate,
        positive_bin_count=len(uniforms),
    )


def aligned_distance_submatrix(
    maximal_ids: Sequence[Any],
    maximal_matrix: Any,
    requested_ids: Iterable[Any],
):
    ids, matrix = validate_distance_matrix(maximal_ids, maximal_matrix)
    requested = sorted(set(requested_ids), key=stable_distance_label_key)
    positions = {cell_id: index for index, cell_id in enumerate(ids)}
    missing = [cell_id for cell_id in requested if cell_id not in positions]
    if missing:
        raise _error(
            "distance_submatrix_labels",
            "Condition labels are absent from the maximal distance matrix.",
            missing=json_safe(missing),
        )
    indices = [positions[cell_id] for cell_id in requested]
    submatrix = matrix[np.ix_(indices, indices)]
    validated_ids, validated = validate_distance_matrix(requested, submatrix)
    return validated_ids, validated


def fixed_label_ad_f1(
    true_tree: Any,
    reconstructed_tree: Any,
    fixed_labels: Iterable[Any],
) -> dict[str, Any]:
    """Compute bilateral endpoint-filtered AD-F1 without tree projection."""
    labels = normalize_cell_labels(fixed_labels)
    if not labels:
        raise _error("fixed_labels_empty", "Fixed-label AD-F1 needs at least one label.")
    true_validated = validate_rooted_labeled_tree(true_tree, "true_tree")
    reconstructed_validated = validate_rooted_labeled_tree(
        reconstructed_tree, "reconstructed_tree"
    )
    missing = labels - frozenset(true_validated.context.labels.values())
    if missing:
        raise _error(
            "fixed_labels_missing_from_truth",
            "Every fixed label must occur in the true tree.",
            labels=sorted(missing),
        )
    label_to_id: dict[str, int] = {}
    true_pairs = unique_ancestor_pair_set(
        true_validated.context,
        restrict_labels=labels,
        label_to_id=label_to_id,
    )
    reconstructed_pairs = unique_ancestor_pair_set(
        reconstructed_validated.context,
        restrict_labels=labels,
        label_to_id=label_to_id,
    )
    tp = len(true_pairs & reconstructed_pairs)
    fp = len(reconstructed_pairs - true_pairs)
    fn = len(true_pairs - reconstructed_pairs)
    precision, recall, f1, iou = prf1_iou(tp, fp, fn)
    if not true_pairs and not reconstructed_pairs:
        degeneracy = "empty_truth_and_reconstruction"
    elif not true_pairs:
        degeneracy = "empty_truth"
    elif not reconstructed_pairs:
        degeneracy = "empty_reconstruction"
    else:
        degeneracy = "none"
    result = {
        "schema_version": FIXED_LABEL_AD_F1_SCHEMA_VERSION,
        "status": "success",
        "fixed_labels": sorted(labels),
        "semantics": {
            "direction": "higher_is_better",
            "pair_multiplicity": "unique_label_pairs",
            "endpoint_filter": "bilateral_both_endpoints_in_fixed_labels",
            "tree_operation": "none_no_projection_contraction_relabeling_or_cross_mapping",
            "empty_empty_value": 0.0,
            "fixed_label_grf_defined": False,
        },
        "metrics": {
            "ad_f1": float(f1),
            "ad_precision": float(precision),
            "ad_recall": float(recall),
            "ad_iou": float(iou),
            "counts": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "true_unique_pair_count": len(true_pairs),
                "reconstructed_unique_pair_count": len(reconstructed_pairs),
            },
            "degeneracy": degeneracy,
        },
    }
    validate_fixed_label_result(result)
    return result


def validate_fixed_label_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != FIXED_LABEL_AD_F1_SCHEMA_VERSION:
        raise _error("fixed_label_schema", "Unknown fixed-label AD-F1 schema.")
    if result.get("status") != "success":
        raise _error("fixed_label_status", "Fixed-label result must be successful.")
    labels = result.get("fixed_labels")
    if not isinstance(labels, list) or labels != sorted(set(labels)) or not labels:
        raise _error("fixed_label_labels", "Fixed labels must be sorted, unique, and nonempty.")
    metrics = result.get("metrics", {})
    counts = metrics.get("counts", {})
    for field in ("tp", "fp", "fn", "true_unique_pair_count", "reconstructed_unique_pair_count"):
        value = counts.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _error("fixed_label_counts", f"Invalid fixed-label count {field}.")
    if counts["tp"] + counts["fn"] != counts["true_unique_pair_count"]:
        raise _error("fixed_label_counts", "Fixed-label truth counts disagree.")
    if counts["tp"] + counts["fp"] != counts["reconstructed_unique_pair_count"]:
        raise _error("fixed_label_counts", "Fixed-label reconstruction counts disagree.")
    expected = prf1_iou(counts["tp"], counts["fp"], counts["fn"])
    for field, value in zip(("ad_precision", "ad_recall", "ad_f1", "ad_iou"), expected):
        actual = metrics.get(field)
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise _error("fixed_label_metric", f"Invalid fixed-label metric {field}.")
        if not math.isfinite(float(actual)) or not math.isclose(float(actual), value, abs_tol=1e-12):
            raise _error("fixed_label_metric", f"Fixed-label metric {field} disagrees with counts.")
    json.dumps(json_safe(result), sort_keys=True, allow_nan=False)


def status_record(
    *,
    entity_type: str,
    entity_id: str,
    status: str,
    stage: str,
    code: str,
    dependency: str | None = None,
    attempts: Sequence[Mapping[str, Any]] = (),
    runtime: Mapping[str, Any] | None = None,
    exception: BaseException | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": EXPERIMENT_STATUS_SCHEMA_VERSION,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "status": status,
        "stage": stage,
        "code": code,
        "dependency": dependency,
        "attempts": list(attempts),
        "runtime": dict(runtime or {}),
        "exception": None,
        "message": message or "",
    }
    if exception is not None:
        record["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception)[:4096],
        }
        if not record["message"]:
            record["message"] = str(exception)[:4096]
    validate_status_record(record)
    return record


def validate_status_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != EXPERIMENT_STATUS_SCHEMA_VERSION:
        raise _error("status_schema", "Unknown experiment-status schema.")
    if record.get("status") not in {"success", "failure", "not_run_dependency"}:
        raise _error("status_value", "Status must be success, failure, or not_run_dependency.")
    for field in ("entity_type", "entity_id", "stage", "code"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise _error("status_field", f"Status field {field} must be nonempty.")
    if record["status"] == "not_run_dependency" and not record.get("dependency"):
        raise _error("status_dependency", "Dependency failure must name its dependency.")
    if not isinstance(record.get("attempts"), list) or not isinstance(record.get("runtime"), dict):
        raise _error("status_shape", "Status attempts/runtime have invalid shapes.")
    json.dumps(json_safe(record), sort_keys=True, allow_nan=False)


def ensure_new_empty_output_root(output_root: Path | str) -> Path:
    output_root = Path(output_root).resolve()
    if output_root.exists():
        if not output_root.is_dir():
            raise _error("output_root_not_directory", f"{output_root} is not a directory.")
        if any(output_root.iterdir()):
            raise _error(
                "output_root_not_empty",
                f"Output root must be new and empty: {output_root}",
            )
    else:
        output_root.mkdir(parents=True)
    return output_root


def _git_output(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def source_freeze_manifest(
    *,
    project_root: Path = PROJECT_ROOT,
    extra_paths: Sequence[str] = (),
    external_paths: Mapping[str, Path | str] | None = None,
) -> dict[str, Any]:
    paths = tuple(dict.fromkeys((*SOURCE_FREEZE_PATHS, *extra_paths)))
    missing = [relative for relative in paths if not (project_root / relative).is_file()]
    if missing:
        raise _error(
            "source_freeze_missing",
            "A required source-freeze file is missing.",
            paths=missing,
        )
    status = _git_output(project_root, "status", "--porcelain=v1", "--untracked-files=all")
    worktree_diff = _git_output(project_root, "diff", "--binary", "--no-ext-diff")
    index_diff = _git_output(project_root, "diff", "--cached", "--binary", "--no-ext-diff")
    untracked_result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=project_root,
        capture_output=True,
        check=True,
    )
    untracked_paths = [
        raw.decode("utf-8")
        for raw in untracked_result.stdout.split(b"\0")
        if raw
    ]
    untracked_hashes = {}
    for relative in untracked_paths:
        path = project_root / relative
        if not path.is_file():
            raise _error(
                "source_freeze_untracked",
                f"Untracked inventory entry is not a regular file: {relative}",
            )
        untracked_hashes[relative] = file_sha256(path)
    external = {}
    for name, raw_path in sorted((external_paths or {}).items()):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise _error("external_tool_missing", f"Required external tool is missing: {path}")
        external[name] = {"path": str(path), "sha256": file_sha256(path)}
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "git": {
            "head": _git_output(project_root, "rev-parse", "HEAD").strip(),
            "status_porcelain_v1": status.splitlines(),
            "worktree_diff_sha256": bytes_sha256(worktree_diff.encode("utf-8")),
            "index_diff_sha256": bytes_sha256(index_diff.encode("utf-8")),
            "untracked_file_sha256": dict(sorted(untracked_hashes.items())),
        },
        "files": {
            relative: file_sha256(project_root / relative)
            for relative in sorted(paths)
        },
        "external_tools": external,
    }


def checksum_entries(
    output_root: Path | str,
    *,
    include_analysis: bool,
) -> list[tuple[str, str]]:
    output_root = Path(output_root)
    files = []
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        parts = Path(relative).parts
        if relative in {"raw_checksums.sha256", "complete_checksums.sha256"}:
            continue
        if not include_analysis and parts and parts[0] == "analysis":
            continue
        if relative.endswith(".tmp") or (parts and parts[0] == "work"):
            continue
        files.append((relative, file_sha256(path)))
    return sorted(files, key=lambda item: item[0].encode("utf-8"))


def write_checksum_file(
    output_root: Path | str,
    filename: str,
    *,
    include_analysis: bool,
) -> list[tuple[str, str]]:
    output_root = Path(output_root)
    entries = checksum_entries(output_root, include_analysis=include_analysis)
    destination = output_root / filename
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for relative, digest in entries:
            stream.write(f"{digest}  {relative}\n")
    os.replace(temporary, destination)
    return entries


def read_checksum_file(path: Path | str) -> list[tuple[str, str]]:
    entries = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            if len(line) < 67 or line[64:66] != "  ":
                raise _error("checksum_format", f"Malformed checksum line {line_number}.")
            digest, relative = line[:64], line[66:]
            if any(character not in "0123456789abcdef" for character in digest):
                raise _error("checksum_format", f"Invalid digest on line {line_number}.")
            entries.append((relative, digest))
    if entries != sorted(entries, key=lambda item: item[0].encode("utf-8")):
        raise _error("checksum_order", "Checksum paths are not bytewise sorted.")
    if len({relative for relative, _digest in entries}) != len(entries):
        raise _error("checksum_duplicate", "Checksum file contains duplicate paths.")
    return entries


def validate_checksum_closure(
    output_root: Path | str,
    filename: str,
    *,
    include_analysis: bool,
) -> None:
    output_root = Path(output_root)
    expected = checksum_entries(output_root, include_analysis=include_analysis)
    actual = read_checksum_file(output_root / filename)
    expected_by_path = dict(expected)
    actual_by_path = dict(actual)
    if expected_by_path != actual_by_path:
        changed = sorted(
            set(expected_by_path) | set(actual_by_path),
            key=lambda value: value.encode("utf-8"),
        )
        changed = [
            value
            for value in changed
            if expected_by_path.get(value) != actual_by_path.get(value)
        ]
        raise _error(
            "checksum_closure",
            f"{filename} does not close over the expected artifact bytes.",
            paths=changed,
        )
