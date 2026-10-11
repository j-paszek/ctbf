"""Frozen contracts and reusable helpers for CTBF v5 confirmation."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from algorithm_evaluation.paper_pipeline_contract import (
    PaperContractError,
    file_sha256,
    json_safe,
    read_json,
    write_json_atomic,
)
from algorithm_evaluation.v5_algorithm_development_common import ARM_SPEC_BY_ID
from distance_semantics import CNP2CNP_SEMANTICS_VERSION
from simulator_config import load_simulator_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "experimental_description"
    / "g2_01a_v5_confirmatory_manifest.json"
)
OWNER_APPROVAL_PATH = (
    PROJECT_ROOT / "algorithm_evaluation" / "v5_confirmation_approval.py"
)

MANIFEST_SCHEMA_VERSION = "ctbf-v5-confirmatory-design-manifest-v1"
MANIFEST_ID = "ctbf-v5-g2-01a-independent-confirmation-v1"
BANK_SCHEMA_VERSION = "ctbf-v5-confirmatory-bank-v1"
RUN_SCHEMA_VERSION = "ctbf-v5-confirmatory-run-v1"
REPORT_SCHEMA_VERSION = "ctbf-v5-confirmatory-report-v1"
SOURCE_LOCK_SCHEMA_VERSION = "ctbf-v5-confirmatory-source-lock-v1"
BANK_MANIFEST_NAME = "bank_manifest.json"
RUN_MANIFEST_NAME = "run_manifest.json"
BANK_CHECKSUM_NAME = "bank_checksums.sha256"
RUN_CHECKSUM_NAME = "run_checksums.sha256"
SIMULATOR_CONFIG_NAME = "simulator_config.json"

CONFIRMATION_NAMESPACE = "ctbf-v5-g2-01a-independent-confirmation-v1"
BLOCK_COUNT = 100
SIMULATION_HEIGHT = 34
TARGET_FRACTION = 0.5
BIOPSY_LOWER_BOUND = 6
SAMPLING_RULE = "min(N,max(6,ceil(0.5*N)))"
CNP2CNP_REQUIRED_SOURCE_FILES = (
    "bioinfoutil.py",
    "cnp2cnp.py",
    "cnpsolver.py",
    "genomesimulator.py",
    "node.py",
)
INTERNAL_RUNTIME_PACKAGES = ("networkx", "numpy", "plotly", "psutil")
STANDARD_CONDITION_IDS = ("H14_rule_y", "H24_rule_y", "H34_rule_y")
BIOPSY_COUNT_CONDITION_IDS = (
    "H24_L2",
    "H24_rule_y",
    "H24_L4",
    "H24_L5",
)
PLACEMENT_CONDITION_IDS = ("H24_rule_y", "H24_late", "H24_random")

M0_ID = "classical_partial"
M2_ID = (
    "biopsy_guided_top_anticentral_binary_adaptive_median_prior_nn_"
    "bottom_deferred_tie"
)
M1_ID = "neighbor_joining_baseline"
POOLED_Q_ID = "rooted_labeled_nj"
M5_ID = "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony"
M6_ID = (
    "biopsy_guided_full_anticentral_binary_adaptive_median_prior_nn_"
    "bottom_deferred_tie"
)
M7_ID = (
    "biopsy_guided_full_rooted_labeled_q_adaptive_median_prior_nn_"
    "bottom_deferred_tie"
)
PAPER_ARM_IDS = (M0_ID, M2_ID, M1_ID, POOLED_Q_ID, M5_ID, M6_ID, M7_ID)
PAPER_ID_BY_ARM = {
    M0_ID: "M0",
    M2_ID: "M2",
    M1_ID: "M1",
    POOLED_Q_ID: "Pooled-Q",
    M5_ID: "M5",
    M6_ID: "M6",
    M7_ID: "M7",
}


@dataclass(frozen=True)
class ConfirmationCondition:
    condition_id: str
    height: int
    family: str
    fixed_schedule: tuple[int, ...] | None


CONDITIONS = (
    ConfirmationCondition("H14_rule_y", 14, "standard", (9, 12, 14)),
    ConfirmationCondition("H24_L2", 24, "biopsy_count", (15, 24)),
    ConfirmationCondition(
        "H24_rule_y",
        24,
        "standard_biopsy_count_and_placement_anchor",
        (15, 20, 24),
    ),
    ConfirmationCondition("H24_L4", 24, "biopsy_count", (15, 18, 20, 24)),
    ConfirmationCondition(
        "H24_L5", 24, "biopsy_count", (15, 18, 20, 22, 24)
    ),
    ConfirmationCondition("H24_late", 24, "placement", (22, 23, 24)),
    ConfirmationCondition("H24_random", 24, "placement", None),
    ConfirmationCondition("H34_rule_y", 34, "standard", (21, 28, 34)),
)
CONDITION_BY_ID = {condition.condition_id: condition for condition in CONDITIONS}


def _contract_error(code: str, message: str, **details: Any) -> PaperContractError:
    return PaperContractError(code, message, details=details)


def read_owner_approval_lock(
    path: Path | str = OWNER_APPROVAL_PATH,
) -> dict[str, Any]:
    """Parse the mutable owner record without importing executable code."""
    source = Path(path).expanduser().resolve()
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    allowed = {
        "APPROVED_MANIFEST_SHA256",
        "APPROVED_PROTOCOL_SHA256",
        "APPROVED_SOURCE_LOCK_SHA256",
        "OWNER_APPROVAL_DATE",
        "OWNER_APPROVAL_RECORD",
    }
    values: dict[str, Any] = {}
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in allowed
            and node.value is not None
        ):
            raise _contract_error(
                "owner_approval_shape",
                "Owner approval file must contain only its docstring and five data fields.",
            )
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError) as error:
            raise _contract_error(
                "owner_approval_shape",
                "Owner approval values must be string or null literals.",
            ) from error
        if value is not None and not isinstance(value, str):
            raise _contract_error(
                "owner_approval_shape",
                "Owner approval values must be string or null literals.",
            )
        if node.target.id in values:
            raise _contract_error(
                "owner_approval_shape",
                "Owner approval field repeats.",
            )
        values[node.target.id] = value
    if set(values) != allowed:
        raise _contract_error("owner_approval_shape", "Owner approval fields changed.")
    populated = sum(value is not None for value in values.values())
    if populated not in {0, len(allowed)}:
        raise _contract_error(
            "owner_approval_incomplete",
            "Owner approval must be entirely unset or entirely populated.",
        )
    return {
        "status": "approved" if populated else "pending",
        "path": str(source),
        "sha256": file_sha256(source),
        **values,
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_seed_for_namespace(
    namespace: str,
    stream: str,
    block_index: int,
    *coordinates: int,
) -> int:
    if not namespace:
        raise ValueError("Seed namespace must be nonempty.")
    if not stream:
        raise ValueError("Seed stream must be nonempty.")
    values = (block_index, *coordinates)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("Seed coordinates must be nonnegative integers.")
    material = "\0".join(
        [namespace, stream, *(str(value) for value in values)]
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def derive_seed(stream: str, block_index: int, *coordinates: int) -> int:
    return derive_seed_for_namespace(
        CONFIRMATION_NAMESPACE,
        stream,
        block_index,
        *coordinates,
    )


def derive_analysis_seed(name: str) -> int:
    if not name:
        raise ValueError("Analysis seed name must be nonempty.")
    material = "\0".join(
        [CONFIRMATION_NAMESPACE, "analysis", name]
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def random_h24_schedule(
    block_index: int,
    *,
    namespace: str = CONFIRMATION_NAMESPACE,
) -> tuple[tuple[int, int, int], int]:
    seed = derive_seed_for_namespace(namespace, "random_schedule", block_index, 24)
    selected = np.random.Generator(np.random.PCG64(seed)).choice(
        np.arange(15, 24, dtype=int),
        size=2,
        replace=False,
    )
    schedule = (*sorted(int(value) for value in selected), 24)
    return schedule, seed


def resolve_schedule(
    condition: ConfirmationCondition,
    block_index: int,
    *,
    namespace: str = CONFIRMATION_NAMESPACE,
) -> tuple[tuple[int, ...], int | None]:
    if condition.condition_id == "H24_random":
        return random_h24_schedule(block_index, namespace=namespace)
    if condition.fixed_schedule is None:  # pragma: no cover - guarded above
        raise ValueError(f"Condition {condition.condition_id} has no schedule.")
    return condition.fixed_schedule, None


def confirmation_case_id(block_index: int, condition_id: str) -> str:
    if condition_id not in CONDITION_BY_ID:
        raise ValueError(f"Unknown confirmation condition {condition_id!r}.")
    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
        raise ValueError("block_index must be a nonnegative integer.")
    return f"conf-b{block_index + 1:03d}-{condition_id}"


def hybrid_sample_size(available: int) -> int:
    if isinstance(available, bool) or not isinstance(available, int) or available < 0:
        raise ValueError("available must be a nonnegative integer.")
    if available == 0:
        return 0
    return min(
        available,
        max(BIOPSY_LOWER_BOUND, int(math.ceil(TARGET_FRACTION * available))),
    )


def inspect_cnp2cnp_source(runfile: Path | str) -> dict[str, Any]:
    """Return the confirmation-relevant identity of a cnp2cnp checkout."""
    executable = Path(runfile).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Missing cnp2cnp executable: {executable}.")
    if executable.name != "cnp2cnp.py":
        raise ValueError("The frozen cnp2cnp entry point must be named cnp2cnp.py.")
    source_root = executable.parent
    sources = {name: source_root / name for name in CNP2CNP_REQUIRED_SOURCE_FILES}
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The cnp2cnp checkout is missing frozen source dependencies: "
            + ", ".join(missing)
        )

    revision = None
    description = None
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip() or None
        description = subprocess.run(
            ["git", "-C", str(source_root), "describe", "--always", "--dirty", "--tags"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return {
        "executable_path": str(executable),
        "executable_name": executable.name,
        "source_sha256": {
            name: file_sha256(path) for name, path in sorted(sources.items())
        },
        "git_revision": revision,
        "git_description": description,
    }


def validate_cnp2cnp_source(
    manifest: Mapping[str, Any],
    runfile: Path | str,
) -> dict[str, Any]:
    """Reject a runtime whose executable source differs from the frozen tool."""
    declaration = manifest.get("distance_contract", {}).get("external_tool_lock", {})
    actual = inspect_cnp2cnp_source(runfile)
    for field in ("executable_name", "source_sha256", "git_revision"):
        if actual.get(field) != declaration.get(field):
            raise _contract_error(
                "cnp2cnp_source",
                f"cnp2cnp {field} differs from the confirmatory lock.",
                expected=declaration.get(field),
                actual=actual.get(field),
            )
    return actual


def inspect_internal_runtime() -> dict[str, Any]:
    """Return locked semantics plus non-locked execution provenance."""
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name) for name in INTERNAL_RUNTIME_PACKAGES
        },
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def validate_internal_runtime(manifest: Mapping[str, Any]) -> dict[str, Any]:
    declaration = manifest.get("internal_runtime_contract", {})
    actual = inspect_internal_runtime()
    for field in ("python_implementation", "python_version", "packages"):
        if actual.get(field) != declaration.get(field):
            raise _contract_error(
                "internal_runtime",
                f"Internal runtime {field} differs from the confirmatory lock.",
                expected=declaration.get(field),
                actual=actual.get(field),
            )
    return actual


def condition_paths(block_index: int, condition_id: str) -> dict[str, str]:
    base = Path("cases") / confirmation_case_id(block_index, condition_id)
    return {
        "metadata": (base / "case.json").as_posix(),
        "input": (base / "input.json").as_posix(),
    }


def truth_path(block_index: int) -> str:
    return (Path("truths") / f"block_{block_index + 1:03d}.json").as_posix()


def distance_path(block_index: int) -> str:
    return (Path("distances") / f"block_{block_index + 1:03d}.json").as_posix()


def record_path(case_id: str, arm_id: str) -> str:
    return (Path("records") / case_id / f"{arm_id}.json").as_posix()


def _expected_manifest_conditions() -> list[dict[str, Any]]:
    result = []
    for condition in CONDITIONS:
        schedule: list[int] | str
        if condition.fixed_schedule is None:
            schedule = "two_without_replacement_from_15_through_23_then_24"
        else:
            schedule = list(condition.fixed_schedule)
        result.append(
            {
                "condition_id": condition.condition_id,
                "height": condition.height,
                "family": condition.family,
                "schedule": schedule,
            }
        )
    return result


def _validate_seed_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    contract = manifest["seed_contract"]
    if contract.get("namespace") != CONFIRMATION_NAMESPACE:
        raise _contract_error("seed_namespace", "Confirmation seed namespace changed.")
    excluded = set(contract.get("development_namespaces_excluded", ()))
    if CONFIRMATION_NAMESPACE in excluded or not excluded:
        raise _contract_error("seed_namespace", "Development namespace exclusion is invalid.")
    expected_analysis = {
        name: derive_analysis_seed(name)
        for name in ("primary_bootstrap", "primary_sign_flip", "secondary_bootstrap")
    }
    if contract.get("analysis_seeds") != expected_analysis:
        raise _contract_error("analysis_seed", "Registered analysis seeds changed.")

    values: list[int] = []
    required_generations: set[int] = set()
    for block_index in range(BLOCK_COUNT):
        values.append(derive_seed("simulation", block_index))
        values.append(derive_seed("random_schedule", block_index, 24))
        for ordinal, condition in enumerate(CONDITIONS):
            schedule, _seed = resolve_schedule(condition, block_index)
            required_generations.update(schedule)
            values.append(derive_seed("reconstruction", block_index, ordinal))
        values.extend(
            derive_seed("sampling", block_index, generation)
            for generation in required_generations
        )
    values.extend(expected_analysis.values())
    if len(values) != len(set(values)):
        raise _contract_error("seed_collision", "Confirmation seeds are not unique.")
    return {
        "registered_numeric_seed_count": len(values),
        "numeric_seed_collision_count": 0,
        "namespace_is_distinct_from_declared_development_namespaces": True,
    }


def _validate_portfolio(manifest: Mapping[str, Any]) -> None:
    records = manifest.get("reconstruction_portfolio")
    if not isinstance(records, list):
        raise _contract_error("portfolio", "Reconstruction portfolio must be a list.")
    arm_ids = tuple(record.get("arm_id") for record in records)
    if arm_ids != PAPER_ARM_IDS:
        raise _contract_error("portfolio", "The seven-method paper portfolio changed.")
    if tuple(record.get("paper_id") for record in records) != tuple(
        PAPER_ID_BY_ARM[arm_id] for arm_id in PAPER_ARM_IDS
    ):
        raise _contract_error("portfolio", "Paper method identifiers changed.")
    for record in records:
        spec = ARM_SPEC_BY_ID[record["arm_id"]]
        if record.get("problem") != spec.problem:
            raise _contract_error(
                "portfolio_problem",
                f"Problem contract changed for {record['arm_id']}.",
            )
        if record.get("primary_metric") != spec.primary_metric:
            raise _contract_error(
                "portfolio_metric",
                f"Primary metric changed for {record['arm_id']}.",
            )


def _validate_observations(manifest: Mapping[str, Any]) -> None:
    design = manifest.get("observation_design", {})
    if design.get("sampling_rule") != SAMPLING_RULE:
        raise _contract_error("sampling_rule", "Confirmation sampling rule changed.")
    if design.get("target_fraction") != TARGET_FRACTION:
        raise _contract_error("sampling_fraction", "Target fraction changed.")
    if design.get("lower_bound") != BIOPSY_LOWER_BOUND:
        raise _contract_error("sampling_lower_bound", "Biopsy lower bound changed.")
    if design.get("conditions") != _expected_manifest_conditions():
        raise _contract_error("conditions", "Confirmation condition declarations changed.")
    if design.get("condition_count_per_block") != len(CONDITIONS):
        raise _contract_error("condition_count", "Condition count changed.")
    if tuple(design.get("standard_condition_ids", ())) != STANDARD_CONDITION_IDS:
        raise _contract_error("conditions", "Standard conditions changed.")
    if tuple(design.get("biopsy_count_condition_ids", ())) != BIOPSY_COUNT_CONDITION_IDS:
        raise _contract_error("conditions", "Biopsy-count conditions changed.")
    if tuple(design.get("placement_condition_ids", ())) != PLACEMENT_CONDITION_IDS:
        raise _contract_error("conditions", "Placement conditions changed.")


def _validate_exact_analysis_contract(manifest: Mapping[str, Any]) -> None:
    primary = manifest.get("primary_hypothesis", {})
    expected = {
        "contrast": "M2_minus_M0",
        "conditions": list(STANDARD_CONDITION_IDS),
        "metric": "grf",
        "independent_unit": "truth_block",
        "block_effect": "unweighted_mean_of_three_height_paired_differences",
        "minimum_complete_blocks": 90,
        "maximum_arm_failure_rate_per_height": 0.1,
        "minimum_material_mean_gain": 0.01,
        "confidence_interval": "two_sided_percentile_95_percent_block_bootstrap",
        "bootstrap_repetitions": 100000,
        "test": "one_sided_monte_carlo_block_sign_flip_positive_mean",
        "sign_flip_repetitions": 100000,
        "alpha": 0.05,
        "tie_tolerance": 1e-12,
        "multiplicity_adjustment": "none_single_primary_test",
    }
    if primary != expected:
        raise _contract_error("primary_hypothesis", "Primary analysis contract changed.")


def _validate_distance_tool_declaration(manifest: Mapping[str, Any]) -> None:
    declaration = manifest.get("distance_contract", {}).get("external_tool_lock", {})
    if declaration.get("executable_name") != "cnp2cnp.py":
        raise _contract_error("cnp2cnp_source", "cnp2cnp entry-point lock changed.")
    hashes = declaration.get("source_sha256")
    if not isinstance(hashes, Mapping) or tuple(sorted(hashes)) != tuple(
        sorted(CNP2CNP_REQUIRED_SOURCE_FILES)
    ):
        raise _contract_error("cnp2cnp_source", "cnp2cnp source inventory changed.")
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in hashes.values()
    ):
        raise _contract_error("cnp2cnp_source", "cnp2cnp source hash is malformed.")
    revision = declaration.get("git_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise _contract_error("cnp2cnp_source", "cnp2cnp revision lock is malformed.")


def _validate_internal_runtime_declaration(manifest: Mapping[str, Any]) -> None:
    declaration = manifest.get("internal_runtime_contract", {})
    if declaration.get("python_implementation") != "CPython":
        raise _contract_error("internal_runtime", "Python implementation lock changed.")
    version = declaration.get("python_version")
    if not isinstance(version, str) or not version:
        raise _contract_error("internal_runtime", "Python version lock is missing.")
    packages = declaration.get("packages")
    if not isinstance(packages, Mapping) or tuple(sorted(packages)) != tuple(
        sorted(INTERNAL_RUNTIME_PACKAGES)
    ):
        raise _contract_error("internal_runtime", "Package-version inventory changed.")
    if any(not isinstance(value, str) or not value for value in packages.values()):
        raise _contract_error("internal_runtime", "Package-version lock is malformed.")


def _validate_source_lock(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
    allow_pending: bool,
) -> dict[str, Any]:
    declaration = manifest.get("source_lock", {})
    if allow_pending and declaration.get("status") == "pending_implementation_completion":
        if declaration.get("sha256") is not None:
            raise _contract_error("source_lock", "Pending source lock unexpectedly has a hash.")
        return {"status": "pending_implementation_completion", "file_count": None}
    if declaration.get("status") != "frozen":
        raise _contract_error("source_lock", "Confirmation source lock is not frozen.")
    path = project_root / str(declaration.get("path"))
    if not path.is_file():
        raise _contract_error("source_lock", f"Missing source lock: {path}.")
    actual_digest = file_sha256(path)
    if actual_digest != declaration.get("sha256"):
        raise _contract_error("source_lock", "Source-lock digest changed.")
    lock = read_json(path)
    if lock.get("schema_version") != SOURCE_LOCK_SCHEMA_VERSION:
        raise _contract_error("source_lock", "Unknown source-lock schema.")
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise _contract_error("source_lock", "Source lock has no files.")
    paths = [record.get("path") for record in files]
    if len(paths) != len(set(paths)):
        raise _contract_error("source_lock", "Source lock repeats a path.")
    for record in files:
        source = project_root / str(record.get("path"))
        if not source.is_file() or file_sha256(source) != record.get("sha256"):
            raise _contract_error(
                "source_bytes",
                f"Frozen source changed: {record.get('path')}.",
            )
    return {"status": "frozen", "file_count": len(files), "sha256": actual_digest}


def validate_manifest(
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
    require_approved: bool = True,
) -> dict[str, Any]:
    """Validate the draft or owner-locked confirmation without generating data."""
    path = Path(manifest_path).expanduser().resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise _contract_error("manifest_schema", "Unknown confirmation manifest schema.")
    if manifest.get("manifest_id") != MANIFEST_ID:
        raise _contract_error("manifest_id", "Confirmation manifest id changed.")
    if manifest.get("status") != "frozen_design_candidate":
        raise _contract_error("manifest_status", "Confirmation design status changed.")
    approval_policy = manifest.get("owner_approval", {})
    if approval_policy != {
        "required": True,
        "mechanism": "external_exact_hash_lock",
        "lock_path": "algorithm_evaluation/v5_confirmation_approval.py",
    }:
        raise _contract_error("owner_approval", "Owner-approval policy changed.")
    if manifest.get("product_version") != "CTBF v5":
        raise _contract_error("product_version", "Confirmation must target CTBF v5.")
    if manifest.get("simulator_semantic_version") != "ctbf-cnp-state-simulator-v5":
        raise _contract_error("simulator_semantics", "Simulator semantics changed.")
    if manifest.get("legacy_artifacts_admissible") is not False:
        raise _contract_error("legacy_policy", "Legacy artifacts must remain excluded.")
    if manifest.get("held_out_outputs_generated") is not False:
        raise _contract_error("held_out_state", "Manifest reports generated held-out data.")
    if manifest.get("held_out_outputs_inspected") is not False:
        raise _contract_error("held_out_state", "Manifest reports inspected held-out data.")

    protocol = manifest.get("protocol", {})
    protocol_path = project_root / str(protocol.get("path"))
    if not protocol_path.is_file() or file_sha256(protocol_path) != protocol.get("sha256"):
        raise _contract_error("protocol_bytes", "Protocol digest changed.")

    simulator = manifest.get("simulator", {})
    config_path = project_root / str(simulator.get("config_path"))
    if not config_path.is_file() or file_sha256(config_path) != simulator.get("config_sha256"):
        raise _contract_error("simulator_config", "Simulator-config digest changed.")
    if simulator.get("simulation_height") != SIMULATION_HEIGHT:
        raise _contract_error("simulation_height", "Simulation height changed.")
    if simulator.get("block_count") != BLOCK_COUNT:
        raise _contract_error("block_count", "Truth-block count changed.")
    inputs = load_simulator_inputs(str(config_path))
    if inputs.config.semantic_version != manifest["simulator_semantic_version"]:
        raise _contract_error("simulator_semantics", "Parsed config semantics changed.")
    if inputs.config.number_of_generations != SIMULATION_HEIGHT:
        raise _contract_error("simulation_height", "Parsed simulator height changed.")

    _validate_observations(manifest)
    _validate_portfolio(manifest)
    _validate_exact_analysis_contract(manifest)
    _validate_distance_tool_declaration(manifest)
    _validate_internal_runtime_declaration(manifest)
    seeds = _validate_seed_contract(manifest)
    source = _validate_source_lock(
        manifest,
        project_root=project_root,
        allow_pending=not require_approved,
    )

    artifact = manifest.get("artifact_contract", {})
    if artifact.get("bank_schema") != BANK_SCHEMA_VERSION:
        raise _contract_error("artifact_schema", "Bank schema changed.")
    if artifact.get("run_schema") != RUN_SCHEMA_VERSION:
        raise _contract_error("artifact_schema", "Run schema changed.")
    if artifact.get("report_schema") != REPORT_SCHEMA_VERSION:
        raise _contract_error("artifact_schema", "Report schema changed.")
    if artifact.get("expected_condition_records") != BLOCK_COUNT * len(CONDITIONS):
        raise _contract_error("artifact_count", "Expected condition count changed.")
    if artifact.get("expected_internal_arm_records") != (
        BLOCK_COUNT * len(CONDITIONS) * len(PAPER_ARM_IDS)
    ):
        raise _contract_error("artifact_count", "Expected arm-record count changed.")
    distance = manifest.get("distance_contract", {})
    if (
        distance.get("semantic_version") != CNP2CNP_SEMANTICS_VERSION
        or distance.get("formula") != "min(d_any(u,v),d_any(v,u))"
        or distance.get("one_maximal_matrix_per_truth_block") is not True
        or distance.get("condition_matrix") != "exact_label_aligned_submatrix"
        or distance.get("fallback_allowed") is not False
        or distance.get("truth_distance_allowed") is not False
    ):
        raise _contract_error("distance_contract", "Distance contract changed.")

    manifest_digest = file_sha256(path)
    owner_approval = read_owner_approval_lock(
        project_root / approval_policy["lock_path"]
    )
    if require_approved:
        if owner_approval["status"] != "approved":
            raise _contract_error(
                "confirmation_not_locked",
                "Held-out execution is disabled until approval hashes are installed.",
            )
        if manifest_digest != owner_approval["APPROVED_MANIFEST_SHA256"]:
            raise _contract_error("manifest_bytes", "Approved manifest bytes changed.")
        if protocol.get("sha256") != owner_approval["APPROVED_PROTOCOL_SHA256"]:
            raise _contract_error("protocol_bytes", "Approved protocol bytes changed.")
        if (
            manifest["source_lock"].get("sha256")
            != owner_approval["APPROVED_SOURCE_LOCK_SHA256"]
        ):
            raise _contract_error("source_lock", "Approved source-lock bytes changed.")

    return {
        "status": (
            "valid_approved" if require_approved else "valid_frozen_candidate"
        ),
        "manifest_path": str(path),
        "manifest_sha256": manifest_digest,
        "protocol_sha256": protocol["sha256"],
        "source_lock": source,
        "owner_approval": owner_approval,
        "seed_validation": seeds,
        "block_count": BLOCK_COUNT,
        "condition_count": len(CONDITIONS),
        "arm_count": len(PAPER_ARM_IDS),
        "manifest": manifest,
    }


def ensure_new_output_root(path: Path | str) -> Path:
    root = Path(path).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Output root already exists: {root}.")
    root.mkdir(parents=True)
    return root


def checksum_rows(root: Path, *, exclude: Iterable[str] = ()) -> list[tuple[str, str]]:
    excluded = set(exclude)
    rows = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix().encode("utf-8"),
    ):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append((file_sha256(path), relative))
    return rows


def write_checksum_closure(root: Path, filename: str) -> Path:
    destination = root / filename
    rows = checksum_rows(root, exclude=(filename,))
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for digest, relative in rows:
            handle.write(f"{digest}  {relative}\n")
    os.replace(temporary, destination)
    return destination


def validate_checksum_closure(root: Path, filename: str) -> dict[str, Any]:
    path = root / filename
    if not path.is_file():
        raise ValueError(f"Missing checksum closure: {path}.")
    expected = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise ValueError(f"Malformed checksum line in {path}.")
        expected.append((digest, relative))
    actual = checksum_rows(root, exclude=(filename,))
    if expected != actual:
        raise ValueError(f"Checksum closure changed: {path}.")
    return {"file_count": len(actual), "sha256": file_sha256(path)}


def write_json(path: Path | str, value: Any) -> None:
    write_json_atomic(path, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the frozen CTBF v5 confirmation contract without data generation."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--require-approved", action="store_true")
    arguments = parser.parse_args(argv)
    result = validate_manifest(
        arguments.manifest,
        require_approved=arguments.require_approved,
    )
    print(
        f"complete: {result['status']}; manifest {result['manifest_sha256']}; "
        f"{result['source_lock']['file_count']} frozen source files; "
        f"{result['seed_validation']['registered_numeric_seed_count']} unique seeds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BANK_CHECKSUM_NAME",
    "BANK_MANIFEST_NAME",
    "BANK_SCHEMA_VERSION",
    "BIOPSY_COUNT_CONDITION_IDS",
    "BIOPSY_LOWER_BOUND",
    "BLOCK_COUNT",
    "CNP2CNP_REQUIRED_SOURCE_FILES",
    "CONDITIONS",
    "CONDITION_BY_ID",
    "CONFIRMATION_NAMESPACE",
    "DEFAULT_MANIFEST_PATH",
    "M0_ID",
    "M1_ID",
    "M2_ID",
    "M5_ID",
    "M6_ID",
    "M7_ID",
    "INTERNAL_RUNTIME_PACKAGES",
    "PAPER_ARM_IDS",
    "PAPER_ID_BY_ARM",
    "PLACEMENT_CONDITION_IDS",
    "POOLED_Q_ID",
    "OWNER_APPROVAL_PATH",
    "PROJECT_ROOT",
    "REPORT_SCHEMA_VERSION",
    "RUN_CHECKSUM_NAME",
    "RUN_MANIFEST_NAME",
    "RUN_SCHEMA_VERSION",
    "SAMPLING_RULE",
    "SIMULATION_HEIGHT",
    "SIMULATOR_CONFIG_NAME",
    "STANDARD_CONDITION_IDS",
    "TARGET_FRACTION",
    "canonical_json_sha256",
    "checksum_rows",
    "condition_paths",
    "confirmation_case_id",
    "derive_analysis_seed",
    "derive_seed",
    "derive_seed_for_namespace",
    "distance_path",
    "ensure_new_output_root",
    "hybrid_sample_size",
    "inspect_cnp2cnp_source",
    "inspect_internal_runtime",
    "main",
    "random_h24_schedule",
    "read_owner_approval_lock",
    "record_path",
    "resolve_schedule",
    "truth_path",
    "validate_checksum_closure",
    "validate_cnp2cnp_source",
    "validate_internal_runtime",
    "validate_manifest",
    "write_checksum_closure",
    "write_json",
]
