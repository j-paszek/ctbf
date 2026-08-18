"""Shared contract for the CTBF v5 fully labeled shortlist robustness test."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.paper_pipeline_runner import validate_reconstruction_input
from algorithm_evaluation.v5_algorithm_development_common import (
    ARM_SPEC_BY_ID,
    BANK_CONFIG_NAME,
    DevelopmentArmSpec,
    deserialize_distance,
    deserialize_truth,
    ensure_new_output_root,
    observed_labels,
    serialize_distance,
    serialize_truth,
    truth_prefix,
    write_json,
)
from distance_semantics import validate_distance_label_coverage


BANK_SCHEMA_VERSION = "ctbf-v5-shortlist-robustness-bank-v3"
PREVIOUS_BANK_SCHEMA_VERSION = "ctbf-v5-shortlist-robustness-bank-v2"
LEGACY_BANK_SCHEMA_VERSION = "ctbf-v5-shortlist-robustness-bank-v1"
CASE_METADATA_SCHEMA_VERSION = "ctbf-v5-shortlist-robustness-case-v1"
RUN_SCHEMA_VERSION = "ctbf-v5-shortlist-robustness-run-v3"
PREVIOUS_RUN_SCHEMA_VERSION = "ctbf-v5-shortlist-robustness-run-v2"
REPORT_SCHEMA_VERSION = "ctbf-v5-shortlist-robustness-report-v3"
BANK_MANIFEST_NAME = "bank_manifest.json"
RESULT_NAME = "result.json"

DISTANCE_EXECUTION_SCHEMA_VERSION = "ctbf-v5-distance-execution-v2"
LEGACY_DISTANCE_EXECUTION_SCHEMA_VERSION = "ctbf-v5-distance-execution-v1"
DISTANCE_EXECUTION_SEMANTICS = (
    "fresh_condition_workers_with_sequential_opposite_order_matrices"
)
LEGACY_DISTANCE_EXECUTION_SEMANTICS = (
    "independent_condition_workers_with_sequential_opposite_order_matrices"
)

DEFAULT_BANK_ID = "ctbf-v5-g1-06-shortlist-robustness-bank-v1"
SEED_NAMESPACE = "ctbf-v5-g1-06-shortlist-robustness-v1"
DEFAULT_BASE_SEED = 20260817
DEFAULT_BLOCK_COUNT = 100
HEIGHTS = (14, 24, 34, 38)
PLACEMENT_POLICIES = ("spread", "late", "random")
TARGET_FRACTION = 0.5
BIOPSY_LOWER_BOUND = 6
SAMPLING_RULE = "min(N,max(6,ceil(0.5*N)))"

ORDERED_A_ID = "biopsy_guided_full_anticentral_binary_r2_bottom_deferred_tie"
ORDERED_B_ID = "biopsy_guided_full_rooted_labeled_q_r2_bottom_deferred_tie"
ORDERED_C_ID = "biopsy_guided_full_anticentral_binary_r4"
POOLED_D_ID = (
    "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony"
)
SHORTLIST_ARM_IDS = (ORDERED_A_ID, ORDERED_B_ID, ORDERED_C_ID, POOLED_D_ID)
POOLED_E_ID = "neighbor_joining_baseline"
POOLED_F_ID = "neighbor_joining_hybrid_opt_refined"
ORDERED_G_ID = "biopsy_guided_full_anticentral_binary_r2"
FULL_EXTENSION_ARM_IDS = (POOLED_E_ID, POOLED_F_ID, ORDERED_G_ID)
FULL_V2_ARM_IDS = SHORTLIST_ARM_IDS + FULL_EXTENSION_ARM_IDS

PARTIAL_X_ID = "classical_partial"
PARTIAL_Y_ID = (
    "biopsy_guided_top_anticentral_binary_r2_bottom_deferred_tie"
)
PARTIAL_Z_ID = "biopsy_guided_classical_r2_bottom_deferred_tie"
PARTIAL_V_ID = "biopsy_guided_top_anticentral_binary_r2"
PARTIAL_W_ID = "biopsy_guided_top_anticentral_binary_r4"
PARTIAL_U_ID = "biopsy_guided_classical_r2"
PARTIAL_V2_ARM_IDS = (
    PARTIAL_X_ID,
    PARTIAL_Y_ID,
    PARTIAL_Z_ID,
    PARTIAL_V_ID,
    PARTIAL_W_ID,
    PARTIAL_U_ID,
)
V2_EXTENSION_ARM_IDS = FULL_EXTENSION_ARM_IDS + PARTIAL_V2_ARM_IDS
V2_COMPLETE_ARM_IDS = FULL_V2_ARM_IDS + PARTIAL_V2_ARM_IDS

ARM_SET_BY_NAME = {
    "abcd": SHORTLIST_ARM_IDS,
    "full-extension": FULL_EXTENSION_ARM_IDS,
    "partial-comparison": PARTIAL_V2_ARM_IDS,
    "v2-extensions": V2_EXTENSION_ARM_IDS,
    "v2-complete": V2_COMPLETE_ARM_IDS,
}

SHORT_LABEL_BY_ARM = {
    ORDERED_A_ID: "A",
    ORDERED_B_ID: "B",
    ORDERED_C_ID: "C",
    POOLED_D_ID: "D",
    POOLED_E_ID: "E",
    POOLED_F_ID: "F",
    ORDERED_G_ID: "G",
    PARTIAL_X_ID: "X",
    PARTIAL_Y_ID: "Y",
    PARTIAL_Z_ID: "Z",
    PARTIAL_V_ID: "V",
    PARTIAL_W_ID: "W",
    PARTIAL_U_ID: "U",
}
DECLARED_METRICS = ("ad_f1", "grf", "ad_precision", "ad_recall")
PARTIAL_DECLARED_METRICS = ("grf",)


def validate_positive_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")


def derived_seed(
    stream: str,
    base_seed: int,
    block_index: int,
    *coordinates: int,
) -> int:
    """Derive an independent deterministic seed in the new shortlist namespace."""
    if not stream:
        raise ValueError("Seed stream must be nonempty.")
    values = (base_seed, block_index, *coordinates)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("Seed coordinates must be nonnegative integers.")
    material = "\0".join(
        [SEED_NAMESPACE, stream, *(str(value) for value in values)]
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def spread_schedule(height: int) -> tuple[int, int, int]:
    schedule = (math.ceil(0.6 * height), math.ceil(0.8 * height), height)
    if tuple(sorted(set(schedule))) != schedule:
        raise ValueError(f"Spread schedule collapses at H{height}: {schedule}.")
    return schedule


def late_schedule(height: int) -> tuple[int, int, int]:
    schedule = (height - 2, height - 1, height)
    if schedule[0] <= 0:
        raise ValueError(f"Late schedule is invalid at H{height}.")
    return schedule


def random_schedule(
    *,
    height: int,
    base_seed: int,
    block_index: int,
) -> tuple[tuple[int, int, int], int]:
    start = math.ceil(0.6 * height)
    candidates = np.arange(start, height, dtype=int)
    if candidates.size < 2:
        raise ValueError(f"Random placement window is too small at H{height}.")
    seed = derived_seed("placement_schedule", base_seed, block_index, height)
    selected = np.random.Generator(np.random.PCG64(seed)).choice(
        candidates,
        size=2,
        replace=False,
    )
    schedule = (*sorted(int(value) for value in selected), int(height))
    return schedule, seed


def placement_schedule(
    policy: str,
    *,
    height: int,
    base_seed: int,
    block_index: int,
) -> tuple[tuple[int, int, int], int | None]:
    if policy == "spread":
        return spread_schedule(height), None
    if policy == "late":
        return late_schedule(height), None
    if policy == "random":
        return random_schedule(
            height=height,
            base_seed=base_seed,
            block_index=block_index,
        )
    raise ValueError(f"Unknown placement policy {policy!r}.")


def case_id(block_index: int, height: int, policy: str) -> str:
    if policy not in PLACEMENT_POLICIES:
        raise ValueError(f"Unknown placement policy {policy!r}.")
    return f"short-b{block_index + 1:03d}-H{height}-{policy}"


def condition_paths(block_index: int, height: int, policy: str) -> dict[str, str]:
    base = Path("cases") / case_id(block_index, height, policy)
    return {
        "metadata": (base / "case.json").as_posix(),
        "input": (base / "input.json").as_posix(),
        "distance": (base / "distance.json").as_posix(),
    }


def truth_path(block_index: int) -> str:
    return (Path("truths") / f"block_{block_index + 1:03d}.json").as_posix()


def resolve_shortlist_arm_ids(arm_set: str) -> tuple[str, ...]:
    try:
        return ARM_SET_BY_NAME[str(arm_set)]
    except KeyError as error:
        available = ", ".join(ARM_SET_BY_NAME)
        raise ValueError(
            f"Unknown shortlist arm set {arm_set!r}; available: {available}."
        ) from error


def shortlist_specs(
    arm_ids: Sequence[str] = SHORTLIST_ARM_IDS,
) -> tuple[DevelopmentArmSpec, ...]:
    normalized = tuple(str(arm_id) for arm_id in arm_ids)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("Shortlist arm ids must be a nonempty unique sequence.")
    unknown = [arm_id for arm_id in normalized if arm_id not in V2_COMPLETE_ARM_IDS]
    if unknown:
        raise ValueError(f"Unknown v2 shortlist arm ids: {unknown!r}.")
    specs = tuple(ARM_SPEC_BY_ID[arm_id] for arm_id in normalized)
    for spec in specs:
        if spec.arm_id in FULL_V2_ARM_IDS:
            expected_primary = "ad_f1"
            expected_complementary = ("grf", "ad_precision", "ad_recall")
        else:
            expected_primary = "grf"
            expected_complementary = ()
        if spec.primary_metric != expected_primary:
            raise RuntimeError(
                f"Shortlist arm {spec.arm_id} changed primary metric."
            )
        if tuple(spec.complementary_metrics) != expected_complementary:
            raise RuntimeError(
                f"Shortlist arm {spec.arm_id} changed complementary metrics."
            )
    return specs


def load_bank_manifest(
    bank_root: Path | str,
    *,
    expected_block_count: int | None = DEFAULT_BLOCK_COUNT,
) -> tuple[Path, dict[str, Any]]:
    root = Path(bank_root).expanduser().resolve()
    manifest = read_json(root / BANK_MANIFEST_NAME)
    if manifest.get("schema_version") not in {
        BANK_SCHEMA_VERSION,
        PREVIOUS_BANK_SCHEMA_VERSION,
        LEGACY_BANK_SCHEMA_VERSION,
    }:
        raise ValueError("Unknown shortlist-robustness bank schema.")
    if manifest.get("bank_id") != DEFAULT_BANK_ID:
        raise ValueError("Unknown shortlist-robustness bank id.")
    if manifest.get("seed_namespace") != SEED_NAMESPACE:
        raise ValueError("Unknown shortlist-robustness seed namespace.")
    if manifest.get("status") != "complete":
        raise ValueError("Shortlist-robustness bank is not complete.")
    block_count = manifest.get("block_count")
    validate_positive_integer(block_count, "manifest block_count")
    if expected_block_count is not None and block_count != expected_block_count:
        raise ValueError(
            "Shortlist bank has the wrong truth-block count: "
            f"expected {expected_block_count}, observed {block_count}."
        )
    heights = tuple(manifest.get("heights", []))
    policies = tuple(manifest.get("placement_policies", []))
    if not heights or any(height not in HEIGHTS for height in heights):
        raise ValueError("Shortlist bank declares invalid heights.")
    if not policies or any(policy not in PLACEMENT_POLICIES for policy in policies):
        raise ValueError("Shortlist bank declares invalid placement policies.")
    mode = manifest.get("contract_mode")
    if mode == "production_100_block_four_height_three_placement_shortlist":
        if (
            int(block_count) != DEFAULT_BLOCK_COUNT
            or heights != HEIGHTS
            or policies != PLACEMENT_POLICIES
        ):
            raise ValueError("Production shortlist bank factorial changed.")
    elif mode == "technical_h38_late_resource_preflight":
        if heights != (38,) or policies != ("late",):
            raise ValueError("Shortlist resource-preflight factorial changed.")
    elif mode != "nonproduction_full_factorial_smoke":
        raise ValueError("Unknown shortlist bank contract mode.")
    if manifest.get("shortlist_arm_ids") != list(SHORTLIST_ARM_IDS):
        raise ValueError("Shortlist bank arm declaration changed.")
    if manifest.get("schema_version") == BANK_SCHEMA_VERSION and manifest.get(
        "v2_reproduction_arm_ids"
    ) != list(V2_COMPLETE_ARM_IDS):
        raise ValueError("Shortlist bank v2 reproduction roster changed.")
    if manifest.get("sampling_rule") != SAMPLING_RULE:
        raise ValueError("Shortlist bank sampling rule changed.")
    if manifest.get("simulation_height") != max(heights):
        raise ValueError("Shortlist bank simulation height is inconsistent.")
    if not (root / BANK_CONFIG_NAME).is_file():
        raise ValueError("Shortlist bank simulator config is missing.")
    declared_count = int(block_count) * len(heights) * len(policies)
    if manifest.get("declared_condition_count") != declared_count:
        raise ValueError("Shortlist bank declared-condition count is inconsistent.")
    inventory = manifest.get("condition_inventory")
    if not isinstance(inventory, list) or len(inventory) != declared_count:
        raise ValueError("Shortlist bank condition inventory is incomplete.")
    expected_ids = {
        case_id(block_index, int(height), str(policy))
        for block_index in range(int(block_count))
        for height in heights
        for policy in policies
    }
    observed_ids = [record.get("case_id") for record in inventory]
    if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != expected_ids:
        raise ValueError("Shortlist bank condition ids are incomplete or duplicated.")
    inventory_by_id = {}
    for record in inventory:
        record_id = str(record["case_id"])
        if record.get("status") not in {"available", "unavailable"}:
            raise ValueError("Shortlist bank has an invalid condition status.")
        expected_schedule, expected_seed = placement_schedule(
            str(record["placement_policy"]),
            height=int(record["height"]),
            base_seed=int(manifest["base_seed"]),
            block_index=int(record["block_index"]),
        )
        if record.get("generations") != list(expected_schedule):
            raise ValueError(f"Shortlist condition {record_id} schedule changed.")
        if record.get("placement_schedule_seed") != expected_seed:
            raise ValueError(
                f"Shortlist condition {record_id} placement seed changed."
            )
        inventory_by_id[record_id] = record
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Shortlist bank available-case inventory is invalid.")
    available_ids = {
        record["case_id"]
        for record in inventory
        if record.get("status") == "available"
    }
    if {case.get("case_id") for case in cases} != available_ids:
        raise ValueError("Shortlist bank available-case inventory is inconsistent.")
    for case in cases:
        declaration = inventory_by_id[str(case["case_id"])]
        for field in (
            "block_index",
            "height",
            "placement_policy",
            "generations",
            "placement_schedule_seed",
            "status",
        ):
            if case.get(field) != declaration.get(field):
                raise ValueError(
                    f"Shortlist available case {case['case_id']} changed {field}."
                )
    if manifest.get("available_condition_count") != len(cases):
        raise ValueError("Shortlist bank available-condition count is inconsistent.")
    if manifest.get("unavailable_condition_count") != declared_count - len(cases):
        raise ValueError("Shortlist bank unavailable-condition count is inconsistent.")
    if manifest.get("completed_condition_count") != declared_count:
        raise ValueError("Shortlist bank completion count is inconsistent.")
    execution_by_block = manifest.get("distance_execution_by_block")
    if execution_by_block is not None:
        if manifest.get("distance_execution_semantics") not in {
            DISTANCE_EXECUTION_SEMANTICS,
            LEGACY_DISTANCE_EXECUTION_SEMANTICS,
        }:
            raise ValueError("Shortlist bank distance-execution semantics changed.")
        if not isinstance(execution_by_block, list) or len(execution_by_block) != int(
            block_count
        ):
            raise ValueError("Shortlist bank distance-execution history is incomplete.")
        for block_index, record in enumerate(execution_by_block):
            if (
                not isinstance(record, Mapping)
                or record.get("schema_version") not in {
                    DISTANCE_EXECUTION_SCHEMA_VERSION,
                    LEGACY_DISTANCE_EXECUTION_SCHEMA_VERSION,
                }
                or record.get("block_index") != block_index
            ):
                raise ValueError("Shortlist bank distance-execution history changed.")
            validate_positive_integer(
                record.get("requested_worker_count"),
                "distance requested_worker_count",
            )
            requested = int(record["requested_worker_count"])
            effective = record.get("effective_worker_count")
            if (
                isinstance(effective, bool)
                or not isinstance(effective, int)
                or not 0 <= effective <= requested
            ):
                raise ValueError("Shortlist bank effective worker count is invalid.")
    for case in cases:
        for field in ("metadata_path", "input_path", "distance_path", "truth_path"):
            path = root / str(case[field])
            if not path.is_file():
                raise ValueError(f"Shortlist bank is missing {path}.")
    return root, manifest


def read_case_assets(
    bank_root: Path,
    case_record: Mapping[str, Any],
    *,
    truth_cache: dict[str, nx.DiGraph] | None = None,
) -> tuple[dict[str, Any], Any, nx.DiGraph, dict[str, Any]]:
    input_payload = read_json(bank_root / str(case_record["input_path"]))
    validate_reconstruction_input(input_payload)
    distance_payload = read_json(bank_root / str(case_record["distance_path"]))
    distance = deserialize_distance(distance_payload)
    metadata = read_json(bank_root / str(case_record["metadata_path"]))
    truth_relative = str(case_record["truth_path"])
    cache = truth_cache if truth_cache is not None else {}
    if truth_relative not in cache:
        truth_payload = read_json(bank_root / truth_relative)
        if truth_payload.get("block_index") != int(case_record["block_index"]):
            raise ValueError("Stored shortlist truth block index changed.")
        cache[truth_relative] = deserialize_truth(truth_payload)
    truth = truth_prefix(cache[truth_relative], int(case_record["height"]))
    if input_payload.get("case_id") != case_record["case_id"]:
        raise ValueError("Stored shortlist reconstruction-input id changed.")
    if distance_payload.get("case_id") != case_record["case_id"]:
        raise ValueError("Stored shortlist distance id changed.")
    if metadata.get("schema_version") != CASE_METADATA_SCHEMA_VERSION:
        raise ValueError("Unknown shortlist case-metadata schema.")
    for field in ("case_id", "block_index", "height", "placement_policy"):
        if metadata.get(field) != case_record.get(field):
            raise ValueError(f"Stored shortlist metadata field {field} changed.")
    validate_distance_label_coverage(
        distance.ids,
        observed_labels(input_payload),
        allow_extra=False,
    )
    return input_payload, distance, truth, metadata


__all__ = [
    "BANK_CONFIG_NAME",
    "BANK_MANIFEST_NAME",
    "BANK_SCHEMA_VERSION",
    "BIOPSY_LOWER_BOUND",
    "CASE_METADATA_SCHEMA_VERSION",
    "DECLARED_METRICS",
    "DEFAULT_BANK_ID",
    "DEFAULT_BASE_SEED",
    "DEFAULT_BLOCK_COUNT",
    "DISTANCE_EXECUTION_SCHEMA_VERSION",
    "DISTANCE_EXECUTION_SEMANTICS",
    "LEGACY_BANK_SCHEMA_VERSION",
    "LEGACY_DISTANCE_EXECUTION_SCHEMA_VERSION",
    "LEGACY_DISTANCE_EXECUTION_SEMANTICS",
    "HEIGHTS",
    "ORDERED_A_ID",
    "ORDERED_B_ID",
    "ORDERED_C_ID",
    "PLACEMENT_POLICIES",
    "POOLED_D_ID",
    "REPORT_SCHEMA_VERSION",
    "RESULT_NAME",
    "RUN_SCHEMA_VERSION",
    "SAMPLING_RULE",
    "SEED_NAMESPACE",
    "SHORTLIST_ARM_IDS",
    "SHORT_LABEL_BY_ARM",
    "TARGET_FRACTION",
    "case_id",
    "condition_paths",
    "derived_seed",
    "ensure_new_output_root",
    "late_schedule",
    "load_bank_manifest",
    "placement_schedule",
    "random_schedule",
    "read_case_assets",
    "serialize_distance",
    "serialize_truth",
    "shortlist_specs",
    "spread_schedule",
    "truth_path",
    "validate_positive_integer",
    "write_json",
]
