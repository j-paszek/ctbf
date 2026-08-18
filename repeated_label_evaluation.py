"""Isolated contracts and tiny exact references for repeated-label metrics.

The implementations in this module are deliberately bounded reference
algorithms.  They exist to discriminate hand-checkable cases and validate
future optional backends; they are not scalable benchmark implementations.
They consume completed ``cell_id``-labeled trees and never use graph-local
node ids as cross-tree identity.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

from evaluation_contract import (
    LABEL_NORMALIZATION_VERSION,
    EvaluationContractError,
    validate_rooted_labeled_tree,
)


REPEATED_LABEL_RESULT_SCHEMA_VERSION = (
    "ctbf-repeated-label-tree-evaluation-result-v1"
)
REPEATED_LABEL_TREE_POLICY_VERSION = "ctbf-repeated-label-full-tree-policy-v1"
UTED_EXACT_REFERENCE_SEMANTICS_VERSION = (
    "ctbf-uted-root-mapped-unit-cost-reference-v1"
)
EPS_EXACT_REFERENCE_SEMANTICS_VERSION = (
    "ctbf-eps-label-matching-edge-count-reference-v1"
)
REFERENCE_MAX_NODES_PER_TREE = 8
UNLABELED_LITERAL = "__CTBF_UNLABELED_LITERAL_V1__"
EXTERNAL_UTED_SOURCE_AUDIT_SCHEMA_VERSION = "ctbf-external-uted-source-audit-v1"
EXTERNAL_UTED_SEMANTIC_PROBE_SCHEMA_VERSION = (
    "ctbf-external-uted-semantic-probe-v1"
)
DEFAULT_EXTERNAL_UTED_ROOT = Path(__file__).resolve().parent.parent / "uted"
UTED_AUDITED_REVISION = "b16f3a510b6c1db588202555c3d2a6b6981be60a"
UTED_AUDITED_SOURCE_SHA256 = {
    "LICENSE.md": "8951bda6e616df7418ff3f80d9699f96da704df64f3bc9f0d5bb75a103680456",
    "README.md": "7f7d348ca27b8a5e2f8966c407919f1d0ea8ea060d31e697f30f60ab7c8d13d5",
    "uted.py": "67b3a3fb52c3dfba1a8e5eafa42056a77528e1de4e92353bc72328b4ed5c57e1",
    "uted_test.py": "dceaf9e97259eb22c5b65b055bb76a4549cfa88c4e008b3b5054d0a1d76aca66",
}
EXTERNAL_EDIST_SOURCE_AUDIT_SCHEMA_VERSION = "ctbf-external-edist-source-audit-v1"
EXTERNAL_EDIST_SEMANTIC_PROBE_SCHEMA_VERSION = (
    "ctbf-external-edist-semantic-probe-v1"
)
EXTERNAL_CUTED_EVALUATION_SEMANTICS_VERSION = (
    "ctbf-edist-1.2.2-cuted-unit-cost-external-v1"
)
EXTERNAL_CUTED_DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_EXTERNAL_EDIST_ROOT = Path(__file__).resolve().parent.parent / "edist"
EDIST_AUDITED_REVISION = "8392f391dfb2a85ea78a9c1f1d57dadbfe48fcb8"
EDIST_AUDITED_SOURCE_SHA256 = {
    "LICENSE.md": "8951bda6e616df7418ff3f80d9699f96da704df64f3bc9f0d5bb75a103680456",
    "README.md": "83aa1916c6b103b01f72c456b258e1f01ed84d211b9c781417fffdc7c6695229",
    "edist/uted.pyx": (
        "96f18f1b8f8a21b49224c1f2dca3e47177603c88ebfd98f6ee916ffb7882a21a"
    ),
    "setup.py": "2530f089103f04c2ce1d72a4a4af01e521dad11d5f4626f6217ebee2f8b5fdd7",
    "tests/uted_test.py": (
        "3917a49cac88c693f8c81d307c03b876a1e84802cf001a0dfbbf56d876899762"
    ),
}
EXTERNAL_EPS_SOURCE_AUDIT_SCHEMA_VERSION = "ctbf-external-eps-source-audit-v2"
EXTERNAL_EPS_APPROX_SEMANTICS_VERSION = (
    "ctbf-eps-four-approx-bidirectional-max-external-v1"
)
EXTERNAL_EPS_DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_EXTERNAL_EPS_ROOT = (
    Path(__file__).resolve().parent.parent / "edge-preservation-similarity"
)
EPS_AUDITED_REVISION = "84bb2601c0b7686cc0b646dc9a98057dbc2eaae9"
EPS_AUDITED_SOURCE_SHA256 = {
    "LICENSE": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
    "edge_preservation_similarity/compute_eps.py": (
        "96f9627c27e9e0aafb9f78d414055273eb711a50631ffbb058735b6d1a8807db"
    ),
    "edge_preservation_similarity/utils.py": (
        "fa25ea5d1599bd45f4a287211f669548460365d4a11bdafc5cbb4530eb664824"
    ),
}


class RepeatedLabelEvaluationError(ValueError):
    """A typed adapter-input, backend, or result-schema failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.details = details or {}


_COMMON_TREE_POLICY = {
    "version": REPEATED_LABEL_TREE_POLICY_VERSION,
    "tree_scope": "full_validated_rooted_directed_tree",
    "label_source": "node_attribute_cell_id",
    "label_normalization": LABEL_NORMALIZATION_VERSION,
    "unlabeled_node_policy": "one_literal_label_not_a_wildcard",
    "unlabeled_literal": UNLABELED_LITERAL,
    "copied_or_inferred_node_policy": "retain_every_vertex_no_contraction",
    "occurrence_correspondence": "optimized_by_metric_not_preassigned",
    "ignored_node_data": "all_attributes_except_cell_id",
    "graph_node_id_policy": "graph_local_only_never_cross_tree_identity",
}


_METRIC_CONTRACTS = {
    "uted_exact_reference": {
        "family": "uted",
        "variant": "unrestricted_exact_reference",
        "semantics_version": UTED_EXACT_REFERENCE_SEMANTICS_VERSION,
        "direction": "lower_is_better",
        "exactness": "exact_within_declared_size_limit",
        "implementation": "ctbf_exhaustive_mapping_reference_v1",
        "implementation_status": "implemented_reference_only",
        "tree_policy": _COMMON_TREE_POLICY,
        "mapping_policy": {
            "root": "true_root_must_map_to_reconstructed_root",
            "structure": "injective_ancestry_relation_preserving_unordered_mapping",
        },
        "cost_policy": {
            "label_substitution": "zero_if_equal_else_one",
            "node_deletion": 1.0,
            "node_insertion": 1.0,
            "cnp_distance": "not_used_in_unit_reference",
        },
        "normalization": "raw_cost_divided_by_sum_of_input_node_counts",
        "timeout_policy": "bounded_by_reference_node_limit_not_wall_clock",
        "max_nodes_per_tree": REFERENCE_MAX_NODES_PER_TREE,
    },
    "cuted_edist": {
        "family": "cuted",
        "variant": "zhang_constrained_uted",
        "semantics_version": EXTERNAL_CUTED_EVALUATION_SEMANTICS_VERSION,
        "direction": "lower_is_better",
        "exactness": "exact_for_constrained_uted_not_unrestricted_uted",
        "implementation": "external_edist_uted_subprocess_adapter",
        "implementation_status": "external_runner_available",
        "dependency_identity": {
            "package_version": "1.2.2",
            "revision": EDIST_AUDITED_REVISION,
            "license": "GPL-3.0-or-later",
            "source_audit": "verified",
            "isolated_semantic_probe": "passed",
        },
        "tree_policy": _COMMON_TREE_POLICY,
        "mapping_policy": {
            "constraint": "zhang_constrained_mapping",
            "root": "not_forced_backend_permits_root_deletion_or_insertion",
            "siblings": "unordered",
        },
        "cost_policy": {
            "node_insertion": 1.0,
            "node_deletion": 1.0,
            "label_substitution": "zero_if_equal_else_one",
            "unlabeled_literal": "ordinary_label_not_wildcard",
        },
        "normalization": "raw_cost_divided_by_sum_of_input_node_counts",
        "timeout_policy": "caller_configurable_default_30_seconds",
        "execution_policy": (
            "explicit_external_interpreter_no_vendoring_no_fallback"
        ),
        "paper_role": "sensitivity_candidate_pending_bounded_runtime_check",
    },
    "eps_exact_reference": {
        "family": "eps",
        "variant": "exact_reference",
        "semantics_version": EPS_EXACT_REFERENCE_SEMANTICS_VERSION,
        "direction": "higher_is_better",
        "exactness": "exact_within_declared_size_limit",
        "implementation": "ctbf_exhaustive_label_matching_reference_v1",
        "implementation_status": "implemented_reference_only",
        "tree_policy": _COMMON_TREE_POLICY,
        "mapping_policy": {
            "nodes": "partial_injective_equal_label_matching",
            "preserved_relation": "directed_parent_child_edge",
            "root": "not_forced",
        },
        "cost_policy": "exact_label_equality_only_no_graded_substitution",
        "normalization": "raw_preserved_edges_divided_by_max_input_edge_count",
        "zero_edge_policy": "normalized_value_null_with_explicit_degeneracy",
        "timeout_policy": "bounded_by_reference_node_limit_not_wall_clock",
        "max_nodes_per_tree": REFERENCE_MAX_NODES_PER_TREE,
    },
    "eps_approx_external": {
        "family": "eps",
        "variant": "published_four_approximation",
        "semantics_version": EXTERNAL_EPS_APPROX_SEMANTICS_VERSION,
        "direction": "higher_is_better",
        "exactness": "four_approximation",
        "implementation": "external_published_eps_subprocess_adapter",
        "implementation_status": "external_runner_available",
        "dependency_identity": {
            "revision": EPS_AUDITED_REVISION,
            "license": "GPL-3.0",
            "gurobi_version": [13, 0, 2],
            "networkx_version": "3.4.2",
            "numpy_version": "2.2.6",
        },
        "tree_policy": _COMMON_TREE_POLICY,
        "mapping_policy": "published_local_match_graph_approximation",
        "cost_policy": "exact_label_equality_only_no_graded_substitution",
        "normalization": "raw_preserved_edges_divided_by_max_input_edge_count",
        "zero_edge_policy": "normalized_value_null_with_explicit_degeneracy",
        "direction_combination": "maximum_of_forward_and_reverse_as_upstream_cli",
        "timeout_policy": "caller_configurable_default_30_seconds",
        "execution_policy": (
            "explicit_external_interpreter_no_vendoring_no_fallback"
        ),
        "paper_role": "sensitivity_candidate_pending_bounded_runtime_check",
    },
}


IMPLEMENTED_REFERENCE_METRICS = frozenset(
    {"uted_exact_reference", "eps_exact_reference"}
)
SUCCESS_CAPABLE_METRICS = IMPLEMENTED_REFERENCE_METRICS | {
    "cuted_edist",
    "eps_approx_external",
}
CANDIDATE_METRIC_IDS = tuple(_METRIC_CONTRACTS)


def candidate_metric_contract(metric_id: str) -> Dict[str, Any]:
    """Return an owned copy of one candidate's complete current contract."""
    try:
        return deepcopy(_METRIC_CONTRACTS[metric_id])
    except KeyError as exc:
        raise ValueError(f"Unknown repeated-label metric id: {metric_id!r}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_checkout_identity(repository_root: Path) -> Tuple[Optional[str], Optional[bool]]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository_root),
                    "status",
                    "--short",
                    "--untracked-files=no",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    return revision or None, dirty


def inspect_external_uted_source(
    repository_root: Any = DEFAULT_EXTERNAL_UTED_ROOT,
) -> Dict[str, Any]:
    """Audit the owner-local GPL checkout without importing or running it."""
    repository_root = Path(repository_root).expanduser().resolve()
    required_paths = {
        relative: repository_root / relative
        for relative in UTED_AUDITED_SOURCE_SHA256
    }
    missing = sorted(
        relative
        for relative, path in required_paths.items()
        if not path.is_file()
    )
    hashes = {
        relative: _sha256_file(path)
        for relative, path in required_paths.items()
        if path.is_file()
    }
    revision, dirty = _git_checkout_identity(repository_root)
    source_matches = (
        not missing
        and revision == UTED_AUDITED_REVISION
        and hashes == UTED_AUDITED_SOURCE_SHA256
        and dirty is False
    )
    dependencies = {
        module: importlib.util.find_spec(module) is not None
        for module in ("numpy", "scipy")
    }
    if missing:
        status = "source_unavailable"
    elif not source_matches:
        status = "source_drift"
    elif not all(dependencies.values()):
        status = "source_verified_backend_dependency_missing"
    else:
        status = "source_verified_backend_dependency_present_unexecuted"
    result = {
        "schema_version": EXTERNAL_UTED_SOURCE_AUDIT_SCHEMA_VERSION,
        "status": status,
        "repository_root": str(repository_root),
        "license": "GPL-3.0-or-later",
        "integration_policy": "external_optional_checkout_no_source_vendoring",
        "execution_policy": "read_only_source_audit_does_not_import_or_run_backend",
        "revision": revision,
        "expected_revision": UTED_AUDITED_REVISION,
        "checkout_dirty": dirty,
        "source_sha256": hashes,
        "expected_source_sha256": dict(UTED_AUDITED_SOURCE_SHA256),
        "source_matches_audited_identity": source_matches,
        "missing_paths": missing,
        "dependency_available": dependencies,
        "backend_executed": False,
        "upstream_tests_executed": False,
        "semantic_probe_executed": False,
        "requirements_are_version_pinned": False,
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


_EXTERNAL_UTED_PROBES = (
    {
        "probe_id": "published_unrestricted_mapping_example",
        "expected_distance": 1.0,
        "left_nodes": ["a", "b", "c", "d", "e"],
        "left_adjacency": [[1, 4], [2, 3], [], [], []],
        "right_nodes": ["a", "e", "d", "c"],
        "right_adjacency": [[1, 2, 3], [], [], []],
    },
    {
        "probe_id": "insert_parent_of_two_siblings",
        "expected_distance": 1.0,
        "left_nodes": ["A", "B", "C"],
        "left_adjacency": [[1, 2], [], []],
        "right_nodes": ["A", "A", "B", "C"],
        "right_adjacency": [[1], [2, 3], [], []],
    },
)


_EXTERNAL_UTED_PROBE_PROGRAM = r"""
import importlib.util
import json
from pathlib import Path
import sys

source_path = Path(sys.argv[1]) / "uted.py"
spec = importlib.util.spec_from_file_location("_ctbf_external_uted", source_path)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load external uted.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
payload = json.load(sys.stdin)
records = []
for probe in payload["probes"]:
    orientations = {}
    for orientation, prefix_a, prefix_b in (
        ("forward", "left", "right"),
        ("reverse", "right", "left"),
    ):
        distance, alignment, search_size = module.uted_astar(
            probe[prefix_a + "_nodes"],
            probe[prefix_a + "_adjacency"],
            probe[prefix_b + "_nodes"],
            probe[prefix_b + "_adjacency"],
        )
        orientations[orientation] = {
            "distance": float(distance),
            "alignment": [int(value) for value in alignment],
            "search_size": int(search_size),
        }
    records.append({
        "probe_id": probe["probe_id"],
        "expected_distance": float(probe["expected_distance"]),
        **orientations,
    })
print(json.dumps({"records": records}, allow_nan=False, sort_keys=True))
"""


def probe_external_uted_semantics(
    repository_root: Any = DEFAULT_EXTERNAL_UTED_ROOT,
    *,
    timeout_seconds: float = 10.0,
) -> Dict[str, Any]:
    """Execute two tiny pinned probes in an isolated child interpreter."""
    repository_root = Path(repository_root).expanduser().resolve()
    source_audit = inspect_external_uted_source(repository_root)
    executable_status = "source_verified_backend_dependency_present_unexecuted"
    result = {
        "schema_version": EXTERNAL_UTED_SEMANTIC_PROBE_SCHEMA_VERSION,
        "status": "source_not_executable",
        "repository_root": str(repository_root),
        "backend": "uted.uted_astar",
        "source_audit": source_audit,
        "timeout_seconds": float(timeout_seconds),
        "backend_executed": False,
        "records": [],
    }
    if source_audit["status"] != executable_status:
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result

    payload = {"probes": list(_EXTERNAL_UTED_PROBES)}
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _EXTERNAL_UTED_PROBE_PROGRAM,
                str(repository_root),
            ],
            check=True,
            capture_output=True,
            input=json.dumps(payload, allow_nan=False),
            text=True,
            timeout=timeout_seconds,
        )
        child_result = json.loads(completed.stdout)
        records = child_result["records"]
    except subprocess.TimeoutExpired:
        result["status"] = "backend_timeout"
        result["backend_executed"] = True
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError) as exc:
        result["status"] = "backend_execution_failed"
        result["backend_executed"] = True
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result

    all_passed = True
    for record in records:
        expected = record["expected_distance"]
        forward = record["forward"]["distance"]
        reverse = record["reverse"]["distance"]
        record["symmetric"] = math.isclose(
            forward,
            reverse,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        record["matches_expected"] = math.isclose(
            forward,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) and math.isclose(
            reverse,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        all_passed = all_passed and record["symmetric"] and record["matches_expected"]

    result["status"] = "semantic_probe_passed" if all_passed else "semantic_probe_failed"
    result["backend_executed"] = True
    result["records"] = records
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def _compiled_edist_extensions(repository_root: Path) -> Dict[str, list]:
    package_root = repository_root / "edist"
    extensions = {}
    for module in ("ted", "uted"):
        matches = sorted(
            {
                *package_root.glob(f"{module}*.so"),
                *package_root.glob(f"{module}*.pyd"),
            }
        )
        extensions[module] = [
            {
                "path": str(path.relative_to(repository_root)),
                "sha256": _sha256_file(path),
            }
            for path in matches
            if path.is_file()
        ]
    return extensions


def inspect_external_edist_source(
    repository_root: Any = DEFAULT_EXTERNAL_EDIST_ROOT,
) -> Dict[str, Any]:
    """Audit the owner-local CUTED source and optional in-place build."""
    repository_root = Path(repository_root).expanduser().resolve()
    required_paths = {
        relative: repository_root / relative
        for relative in EDIST_AUDITED_SOURCE_SHA256
    }
    missing = sorted(
        relative
        for relative, path in required_paths.items()
        if not path.is_file()
    )
    hashes = {
        relative: _sha256_file(path)
        for relative, path in required_paths.items()
        if path.is_file()
    }
    revision, dirty = _git_checkout_identity(repository_root)
    source_matches = (
        not missing
        and revision == EDIST_AUDITED_REVISION
        and hashes == EDIST_AUDITED_SOURCE_SHA256
        and dirty is False
    )
    dependencies = {
        module: importlib.util.find_spec(module) is not None
        for module in ("Cython", "numpy", "scipy")
    }
    compiled_extensions = _compiled_edist_extensions(repository_root)
    build_present = all(compiled_extensions.values())
    if missing:
        status = "source_unavailable"
    elif not source_matches:
        status = "source_drift"
    elif not build_present and not dependencies["Cython"]:
        status = "source_verified_build_dependency_missing"
    elif not build_present:
        status = "source_verified_backend_not_built"
    else:
        status = "source_verified_backend_build_present_unexecuted"
    result = {
        "schema_version": EXTERNAL_EDIST_SOURCE_AUDIT_SCHEMA_VERSION,
        "status": status,
        "repository_root": str(repository_root),
        "package_version": "1.2.2",
        "algorithm": "zhang_1996_constrained_unordered_tree_edit_distance",
        "license": "GPL-3.0-or-later",
        "integration_policy": "external_optional_checkout_no_source_vendoring",
        "execution_policy": "source_audit_does_not_import_or_run_backend",
        "revision": revision,
        "expected_revision": EDIST_AUDITED_REVISION,
        "checkout_dirty": dirty,
        "source_sha256": hashes,
        "expected_source_sha256": dict(EDIST_AUDITED_SOURCE_SHA256),
        "source_matches_audited_identity": source_matches,
        "missing_paths": missing,
        "dependency_available": dependencies,
        "compiled_extensions": compiled_extensions,
        "compiled_build_present": build_present,
        "backend_executed": False,
        "upstream_tests_executed": False,
        "requirements_are_version_pinned": False,
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


_EXTERNAL_EDIST_PROBES = (
    {
        "probe_id": "sibling_permutation",
        "expected_distance": 0.0,
        "left_nodes": ["A", "B", "C"],
        "left_adjacency": [[1, 2], [], []],
        "right_nodes": ["A", "C", "B"],
        "right_adjacency": [[1, 2], [], []],
        "cost_mode": "unit",
    },
    {
        "probe_id": "published_unrestricted_distinguishing_example",
        "expected_distance": 3.0,
        "left_nodes": ["a", "b", "c", "d", "e"],
        "left_adjacency": [[1, 4], [2, 3], [], [], []],
        "right_nodes": ["a", "e", "d", "c"],
        "right_adjacency": [[1, 2, 3], [], [], []],
        "cost_mode": "unit",
    },
    {
        "probe_id": "root_deletion_is_permitted",
        "expected_distance": 1.0,
        "left_nodes": ["A", "B"],
        "left_adjacency": [[1], []],
        "right_nodes": ["B"],
        "right_adjacency": [[]],
        "cost_mode": "unit",
    },
    {
        "probe_id": "custom_quarter_substitution",
        "expected_distance": 0.25,
        "left_nodes": ["A"],
        "left_adjacency": [[]],
        "right_nodes": ["B"],
        "right_adjacency": [[]],
        "cost_mode": "quarter_substitution",
    },
    {
        "probe_id": "insert_parent_of_two_siblings_exploratory",
        "expected_distance": None,
        "left_nodes": ["A", "B", "C"],
        "left_adjacency": [[1, 2], [], []],
        "right_nodes": ["A", "A", "B", "C"],
        "right_adjacency": [[1], [2, 3], [], []],
        "cost_mode": "unit",
    },
)


_EXTERNAL_EDIST_PROBE_PROGRAM = r"""
import json
from pathlib import Path
import sys

repository_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository_root))
import edist.uted as backend

module_path = Path(backend.__file__).resolve()
if not module_path.is_relative_to(repository_root):
    raise RuntimeError("edist.uted did not load from the audited checkout")

payload = json.load(sys.stdin)
records = []
for probe in payload["probes"]:
    delta = None
    if probe["cost_mode"] == "quarter_substitution":
        def delta(left, right):
            if left is None or right is None:
                return 1.0
            return 0.0 if left == right else 0.25
    orientations = {}
    for orientation, prefix_a, prefix_b in (
        ("forward", "left", "right"),
        ("reverse", "right", "left"),
    ):
        distance = backend.uted(
            probe[prefix_a + "_nodes"],
            probe[prefix_a + "_adjacency"],
            probe[prefix_b + "_nodes"],
            probe[prefix_b + "_adjacency"],
            delta,
        )
        orientations[orientation] = {"distance": float(distance)}
    records.append({
        "probe_id": probe["probe_id"],
        "expected_distance": probe["expected_distance"],
        **orientations,
    })
print(json.dumps({
    "backend_module_path": str(module_path),
    "backend_version": getattr(backend, "__version__", None),
    "records": records,
}, allow_nan=False, sort_keys=True))
"""


_EXTERNAL_EDIST_EVALUATION_PROGRAM = r"""
import json
from pathlib import Path
import sys

repository_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository_root))
import edist.uted as backend

module_path = Path(backend.__file__).resolve()
if not module_path.is_relative_to(repository_root):
    raise RuntimeError("edist.uted did not load from the audited checkout")

payload = json.load(sys.stdin)
distance = float(backend.uted(
    payload["left_nodes"],
    payload["left_adjacency"],
    payload["right_nodes"],
    payload["right_adjacency"],
))
print(json.dumps({
    "backend_module_path": str(module_path),
    "backend_version": getattr(backend, "__version__", None),
    "distance": distance,
}, allow_nan=False, sort_keys=True))
"""


_EXTERNAL_EPS_APPROX_EVALUATION_PROGRAM = r"""
import contextlib
import io
import json
from pathlib import Path
import sys

repository_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository_root))

diagnostic_stdout = io.StringIO()
with contextlib.redirect_stdout(diagnostic_stdout):
    import gurobipy
    import networkx as nx
    import numpy as np
    from edge_preservation_similarity import compute_eps as backend

    module_path = Path(backend.__file__).resolve()
    if not module_path.is_relative_to(repository_root):
        raise RuntimeError(
            "edge_preservation_similarity did not load from the audited checkout"
        )

    payload = json.load(sys.stdin)

    def build_graph(prefix):
        graph = nx.DiGraph()
        labels = payload[prefix + "_nodes"]
        for node_id, label in enumerate(labels):
            graph.add_node(node_id, lbl=label)
        for child_id, parent_id in enumerate(payload[prefix + "_parents"]):
            if parent_id is not None:
                graph.add_edge(parent_id, child_id)
        return graph

    left = build_graph("left")
    right = build_graph("right")
    orientations = {}
    for orientation, first, second in (
        ("forward", left, right),
        ("reverse", right, left),
    ):
        score, duration, time_limit_exceeded = backend.compute_similarity(
            "EDGE-PRESERVATION-SIM-APPROX",
            first.copy(),
            second.copy(),
            0,
            False,
        )
        orientations[orientation] = {
            "raw_value": float(score),
            "duration_seconds": float(duration),
            "time_limit_exceeded": bool(time_limit_exceeded),
        }

print(json.dumps({
    "backend_module_path": str(module_path),
    "gurobi_version": list(gurobipy.gurobi.version()),
    "networkx_version": nx.__version__,
    "numpy_version": np.__version__,
    "diagnostic_stdout_line_count": len(
        diagnostic_stdout.getvalue().splitlines()
    ),
    "orientations": orientations,
}, allow_nan=False, sort_keys=True))
"""


def probe_external_edist_semantics(
    repository_root: Any = DEFAULT_EXTERNAL_EDIST_ROOT,
    *,
    python_executable: Any = sys.executable,
    timeout_seconds: float = 10.0,
    additional_probes: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute fixed CUTED probes in an explicit isolated interpreter."""
    repository_root = Path(repository_root).expanduser().resolve()
    python_executable = Path(python_executable).expanduser()
    if not python_executable.is_absolute():
        python_executable = Path.cwd() / python_executable
    source_audit = inspect_external_edist_source(repository_root)
    executable_status = "source_verified_backend_build_present_unexecuted"
    result = {
        "schema_version": EXTERNAL_EDIST_SEMANTIC_PROBE_SCHEMA_VERSION,
        "status": "source_not_executable",
        "repository_root": str(repository_root),
        "python_executable": str(python_executable),
        "backend": "edist.uted.uted",
        "algorithm": "zhang_1996_constrained_unordered_tree_edit_distance",
        "source_audit": source_audit,
        "timeout_seconds": float(timeout_seconds),
        "backend_executed": False,
        "records": [],
    }
    if source_audit["status"] != executable_status:
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result

    probes = list(_EXTERNAL_EDIST_PROBES)
    if additional_probes is not None:
        probes.extend(deepcopy(list(additional_probes)))
    payload = {"probes": probes}
    try:
        completed = subprocess.run(
            [
                str(python_executable),
                "-I",
                "-B",
                "-c",
                _EXTERNAL_EDIST_PROBE_PROGRAM,
                str(repository_root),
            ],
            check=True,
            capture_output=True,
            input=json.dumps(payload, allow_nan=False),
            text=True,
            timeout=timeout_seconds,
        )
        child_result = json.loads(completed.stdout)
        records = child_result["records"]
    except subprocess.TimeoutExpired:
        result["status"] = "backend_timeout"
        result["backend_executed"] = True
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result
    except subprocess.CalledProcessError as exc:
        result["status"] = "backend_execution_failed"
        result["backend_executed"] = True
        result["failure"] = {
            "type": type(exc).__name__,
            "message": "External edist probe process exited unsuccessfully.",
            "returncode": exc.returncode,
            "stderr": exc.stderr[-4000:],
            "stdout": exc.stdout[-4000:],
        }
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError) as exc:
        result["status"] = "backend_execution_failed"
        result["backend_executed"] = True
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result

    all_passed = True
    for record in records:
        expected = record["expected_distance"]
        forward = record["forward"]["distance"]
        reverse = record["reverse"]["distance"]
        record["symmetric"] = math.isclose(
            forward,
            reverse,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        if expected is None:
            record["matches_expected"] = None
        else:
            record["matches_expected"] = math.isclose(
                forward,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ) and math.isclose(
                reverse,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        passed = record["symmetric"] and record["matches_expected"] is not False
        all_passed = all_passed and passed

    result["status"] = "semantic_probe_passed" if all_passed else "semantic_probe_failed"
    result["backend_executed"] = True
    result["backend_module_path"] = child_result["backend_module_path"]
    result["backend_version"] = child_result["backend_version"]
    result["records"] = records
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def inspect_external_eps_source(
    repository_root: Any = DEFAULT_EXTERNAL_EPS_ROOT,
) -> Dict[str, Any]:
    """Audit the owner-local GPL checkout without importing or running it."""
    repository_root = Path(repository_root).expanduser().resolve()
    required_paths = {
        relative: repository_root / relative
        for relative in EPS_AUDITED_SOURCE_SHA256
    }
    required_paths["requirements.txt"] = repository_root / "requirements.txt"
    missing = sorted(
        relative
        for relative, path in required_paths.items()
        if not path.is_file()
    )
    hashes = {
        relative: _sha256_file(path)
        for relative, path in required_paths.items()
        if relative in EPS_AUDITED_SOURCE_SHA256 and path.is_file()
    }
    revision, dirty = _git_checkout_identity(repository_root)
    source_matches = (
        not missing
        and revision == EPS_AUDITED_REVISION
        and hashes == EPS_AUDITED_SOURCE_SHA256
        and dirty is False
    )
    ctbf_environment_gurobipy_available = (
        importlib.util.find_spec("gurobipy") is not None
    )
    if missing:
        status = "source_unavailable"
    elif not source_matches:
        status = "source_drift"
    else:
        status = "source_verified_external_dependency_unchecked"
    result = {
        "schema_version": EXTERNAL_EPS_SOURCE_AUDIT_SCHEMA_VERSION,
        "status": status,
        "repository_root": str(repository_root),
        "license": "GPL-3.0",
        "integration_policy": "external_optional_checkout_no_source_vendoring",
        "execution_policy": "read_only_source_audit_does_not_import_or_run_backend",
        "revision": revision,
        "expected_revision": EPS_AUDITED_REVISION,
        "checkout_dirty": dirty,
        "source_sha256": hashes,
        "expected_source_sha256": dict(EPS_AUDITED_SOURCE_SHA256),
        "source_matches_audited_identity": source_matches,
        "missing_paths": missing,
        "ctbf_environment_gurobipy_available": (
            ctbf_environment_gurobipy_available
        ),
        "external_environment_checked": False,
        "gurobi_license_checked": False,
        "backend_executed": False,
        "requirements_are_version_pinned": False,
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


@dataclass(frozen=True)
class _CanonicalTree:
    labels: Tuple[str, ...]
    parents: Tuple[Optional[int], ...]
    edge_count: int
    structural_digest: str
    metadata: Dict[str, Any]

    @property
    def node_count(self) -> int:
        return len(self.labels)


def _canonical_subtree(context, node: Any) -> tuple:
    label = context.labels.get(node, UNLABELED_LITERAL)
    children = tuple(
        sorted(
            (_canonical_subtree(context, child) for child in context.children.get(node, ())),
        )
    )
    return label, children


def _flatten_canonical_subtree(
    subtree: tuple,
    parent: Optional[int],
    labels: list,
    parents: list,
) -> None:
    label, children = subtree
    node_index = len(labels)
    labels.append(label)
    parents.append(parent)
    for child in children:
        _flatten_canonical_subtree(child, node_index, labels, parents)


def _canonical_tree(tree: Any, role: str) -> _CanonicalTree:
    try:
        validated = validate_rooted_labeled_tree(tree, role)
    except EvaluationContractError as exc:
        raise RepeatedLabelEvaluationError(
            exc.code,
            str(exc),
            stage=exc.stage,
            details=exc.details,
        ) from exc

    subtree = _canonical_subtree(validated.context, validated.root)
    labels: list[str] = []
    parents: list[Optional[int]] = []
    _flatten_canonical_subtree(subtree, None, labels, parents)
    serialized = json.dumps(
        subtree,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    structural_digest = hashlib.sha256(serialized).hexdigest()
    metadata = {
        **validated.metadata,
        "canonical_structure_sha256": structural_digest,
    }
    return _CanonicalTree(
        labels=tuple(labels),
        parents=tuple(parents),
        edge_count=validated.metadata["edge_count"],
        structural_digest=structural_digest,
        metadata=metadata,
    )


def _external_edist_adjacency(tree: _CanonicalTree) -> list:
    children = [[] for _ in range(tree.node_count)]
    for child, parent in enumerate(tree.parents):
        if parent is not None:
            children[parent].append(child)
    return children


def build_external_edist_unit_probe(
    probe_id: str,
    true_tree: Any,
    reconstructed_tree: Any,
) -> Dict[str, Any]:
    """Convert one validated pair into a unit-cost exploratory CUTED probe."""
    if not isinstance(probe_id, str) or not probe_id:
        raise ValueError("probe_id must be a non-empty string.")
    left = _canonical_tree(true_tree, "true_tree")
    right = _canonical_tree(reconstructed_tree, "reconstructed_tree")

    return {
        "probe_id": probe_id,
        "expected_distance": None,
        "left_nodes": list(left.labels),
        "left_adjacency": _external_edist_adjacency(left),
        "right_nodes": list(right.labels),
        "right_adjacency": _external_edist_adjacency(right),
        "cost_mode": "unit",
    }


def _check_reference_size(metric_id: str, *trees: _CanonicalTree) -> None:
    oversized = [
        tree.node_count
        for tree in trees
        if tree.node_count > REFERENCE_MAX_NODES_PER_TREE
    ]
    if oversized:
        raise RepeatedLabelEvaluationError(
            "reference_size_limit_exceeded",
            (
                f"{metric_id} accepts at most {REFERENCE_MAX_NODES_PER_TREE} "
                "nodes per tree."
            ),
            stage="reference_backend",
            details={
                "max_nodes_per_tree": REFERENCE_MAX_NODES_PER_TREE,
                "input_node_counts": [tree.node_count for tree in trees],
            },
        )


def _relation_matrix(tree: _CanonicalTree) -> Tuple[Tuple[str, ...], ...]:
    ancestors = []
    for node in range(tree.node_count):
        node_ancestors = set()
        parent = tree.parents[node]
        while parent is not None:
            node_ancestors.add(parent)
            parent = tree.parents[parent]
        ancestors.append(node_ancestors)

    relations = []
    for left in range(tree.node_count):
        row = []
        for right in range(tree.node_count):
            if left == right:
                relation = "same"
            elif left in ancestors[right]:
                relation = "ancestor"
            elif right in ancestors[left]:
                relation = "descendant"
            else:
                relation = "incomparable"
            row.append(relation)
        relations.append(tuple(row))
    return tuple(relations)


def _uted_exact_unit_cost(left: _CanonicalTree, right: _CanonicalTree) -> float:
    """Exhaustively minimize a root-mapped unordered edit mapping."""
    left_relations = _relation_matrix(left)
    right_relations = _relation_matrix(right)
    root_cost = 0.0 if left.labels[0] == right.labels[0] else 1.0
    best = root_cost + (left.node_count - 1) + (right.node_count - 1)
    mapping: Dict[int, int] = {0: 0}
    used_right = {0}

    def compatible(left_node: int, right_node: int) -> bool:
        return all(
            left_relations[left_node][mapped_left]
            == right_relations[right_node][mapped_right]
            for mapped_left, mapped_right in mapping.items()
        )

    def search(left_node: int, cost: float) -> None:
        nonlocal best
        if cost >= best:
            return
        if left_node == left.node_count:
            total = cost + (right.node_count - len(used_right))
            if total < best:
                best = total
            return

        candidates = [
            right_node
            for right_node in range(1, right.node_count)
            if right_node not in used_right and compatible(left_node, right_node)
        ]
        candidates.sort(
            key=lambda right_node: (
                left.labels[left_node] != right.labels[right_node],
                right_node,
            )
        )
        for right_node in candidates:
            mapping[left_node] = right_node
            used_right.add(right_node)
            substitution = 0.0 if left.labels[left_node] == right.labels[right_node] else 1.0
            search(left_node + 1, cost + substitution)
            used_right.remove(right_node)
            del mapping[left_node]

        search(left_node + 1, cost + 1.0)

    if left.node_count > 1:
        search(1, root_cost)
    return float(best)


def _eps_exact_preserved_edges(left: _CanonicalTree, right: _CanonicalTree) -> int:
    """Exhaustively maximize edges under a partial equal-label node matching."""
    right_edges = {
        (parent, node)
        for node, parent in enumerate(right.parents)
        if parent is not None
    }
    maximum_possible = min(left.edge_count, right.edge_count)
    mapping: Dict[int, int] = {}
    used_right = set()
    best = 0

    def search(left_node: int, preserved: int) -> None:
        nonlocal best
        if best == maximum_possible:
            return
        if left_node == left.node_count:
            best = max(best, preserved)
            return

        candidates = [
            right_node
            for right_node, label in enumerate(right.labels)
            if label == left.labels[left_node] and right_node not in used_right
        ]
        candidates.sort(key=lambda right_node: (right_node != left_node, right_node))
        parent = left.parents[left_node]
        for right_node in candidates:
            increment = 0
            if parent is not None and parent in mapping:
                increment = int((mapping[parent], right_node) in right_edges)
            mapping[left_node] = right_node
            used_right.add(right_node)
            search(left_node + 1, preserved + increment)
            used_right.remove(right_node)
            del mapping[left_node]

        search(left_node + 1, preserved)

    search(0, 0)
    return best


def _input_metadata(left: _CanonicalTree, right: _CanonicalTree) -> Dict[str, Any]:
    return {
        "true_tree": left.metadata,
        "reconstructed_tree": right.metadata,
    }


def _failure_result(
    metric_id: str,
    error: RepeatedLabelEvaluationError,
    inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "schema_version": REPEATED_LABEL_RESULT_SCHEMA_VERSION,
        "status": "failure",
        "metric_id": metric_id,
        "metric_contract": candidate_metric_contract(metric_id),
        "inputs": inputs or {},
        "failure": {
            "code": error.code,
            "stage": error.stage,
            "message": str(error),
            "details": error.details,
        },
    }
    validate_repeated_label_result(result)
    return result


def evaluate_external_cuted_tree_pair_result(
    true_tree: Any,
    reconstructed_tree: Any,
    *,
    repository_root: Any = DEFAULT_EXTERNAL_EDIST_ROOT,
    python_executable: Any,
    timeout_seconds: float = EXTERNAL_CUTED_DEFAULT_TIMEOUT_SECONDS,
    source_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run unit-cost CUTED in an independently installed edist interpreter.

    ``source_audit`` lets a bounded batch reuse one prior source/hash audit.
    The external interpreter remains explicit and no fallback metric is used.
    """
    metric_id = "cuted_edist"
    try:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0.0
        ):
            raise RepeatedLabelEvaluationError(
                "invalid_external_timeout",
                "External CUTED timeout must be finite and positive.",
                stage="backend_setup",
                details={"timeout_seconds": timeout_seconds},
            )
        timeout_seconds = float(timeout_seconds)

        left = _canonical_tree(true_tree, "true_tree")
        right = _canonical_tree(reconstructed_tree, "reconstructed_tree")
        inputs = _input_metadata(left, right)

        repository_root = Path(repository_root).expanduser().resolve()
        python_executable = Path(python_executable).expanduser()
        if not python_executable.is_absolute():
            python_executable = Path.cwd() / python_executable
        if not python_executable.is_file():
            raise RepeatedLabelEvaluationError(
                "external_interpreter_unavailable",
                "The explicit external edist interpreter does not exist.",
                stage="backend_setup",
                details={"python_executable": str(python_executable)},
            )

        if source_audit is None:
            source_audit = inspect_external_edist_source(repository_root)
        else:
            source_audit = deepcopy(source_audit)
        expected_status = "source_verified_backend_build_present_unexecuted"
        audit_matches = (
            source_audit.get("status") == expected_status
            and source_audit.get("repository_root") == str(repository_root)
            and source_audit.get("revision") == EDIST_AUDITED_REVISION
            and source_audit.get("source_matches_audited_identity") is True
        )
        if not audit_matches:
            raise RepeatedLabelEvaluationError(
                "external_backend_unavailable",
                "The external edist checkout/build did not pass its pinned audit.",
                stage="backend_setup",
                details={
                    "repository_root": str(repository_root),
                    "source_audit_status": source_audit.get("status"),
                    "source_revision": source_audit.get("revision"),
                },
            )

        payload = {
            "left_nodes": list(left.labels),
            "left_adjacency": _external_edist_adjacency(left),
            "right_nodes": list(right.labels),
            "right_adjacency": _external_edist_adjacency(right),
        }
        try:
            completed = subprocess.run(
                [
                    str(python_executable),
                    "-I",
                    "-B",
                    "-c",
                    _EXTERNAL_EDIST_EVALUATION_PROGRAM,
                    str(repository_root),
                ],
                check=True,
                capture_output=True,
                input=json.dumps(payload, allow_nan=False),
                text=True,
                timeout=timeout_seconds,
            )
            child_result = json.loads(completed.stdout)
            raw_value = child_result["distance"]
            module_path = Path(child_result["backend_module_path"]).resolve()
        except subprocess.TimeoutExpired as exc:
            raise RepeatedLabelEvaluationError(
                "external_backend_timeout",
                f"External CUTED exceeded its {timeout_seconds}-second timeout.",
                stage="backend_execution",
                details={"timeout_seconds": timeout_seconds},
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RepeatedLabelEvaluationError(
                "external_backend_failed",
                "External edist exited unsuccessfully.",
                stage="backend_execution",
                details={
                    "returncode": exc.returncode,
                    "stdout_tail": (exc.stdout or "")[-4000:],
                    "stderr_tail": (exc.stderr or "")[-4000:],
                },
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise RepeatedLabelEvaluationError(
                "external_backend_failed",
                "External edist could not be executed.",
                stage="backend_execution",
                details={"error_type": type(exc).__name__, "message": str(exc)},
            ) from exc
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RepeatedLabelEvaluationError(
                "external_backend_invalid_output",
                "External edist returned an invalid result record.",
                stage="backend_output",
                details={"error_type": type(exc).__name__, "message": str(exc)},
            ) from exc

        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
            or float(raw_value) < 0.0
            or not module_path.is_file()
            or not module_path.is_relative_to(repository_root)
        ):
            raise RepeatedLabelEvaluationError(
                "external_backend_invalid_output",
                "External edist returned an invalid distance or module identity.",
                stage="backend_output",
                details={
                    "raw_value": raw_value,
                    "backend_module_path": str(module_path),
                },
            )

        raw_value = float(raw_value)
        denominator = float(left.node_count + right.node_count)
        result = {
            "schema_version": REPEATED_LABEL_RESULT_SCHEMA_VERSION,
            "status": "success",
            "metric_id": metric_id,
            "metric_contract": candidate_metric_contract(metric_id),
            "inputs": inputs,
            "metric": {
                "raw_value": raw_value,
                "normalization_denominator": denominator,
                "value": raw_value / denominator,
                "degeneracy": "none",
            },
            "external_execution": {
                "backend": "edist.uted.uted",
                "repository_root": str(repository_root),
                "python_executable": str(python_executable),
                "source_revision": source_audit["revision"],
                "backend_module_path": str(module_path),
                "backend_module_sha256": _sha256_file(module_path),
                "backend_version": child_result.get("backend_version"),
                "timeout_seconds": timeout_seconds,
                "cost_mode": "unit",
                "fallback_used": False,
            },
        }
        validate_repeated_label_result(result)
        return result
    except RepeatedLabelEvaluationError as error:
        captured_inputs = locals().get("inputs")
        return _failure_result(metric_id, error, captured_inputs)


def evaluate_external_eps_approx_tree_pair_result(
    true_tree: Any,
    reconstructed_tree: Any,
    *,
    repository_root: Any = DEFAULT_EXTERNAL_EPS_ROOT,
    python_executable: Any,
    timeout_seconds: float = EXTERNAL_EPS_DEFAULT_TIMEOUT_SECONDS,
    source_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the published approximate EPS in both directions externally."""
    metric_id = "eps_approx_external"
    try:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0.0
        ):
            raise RepeatedLabelEvaluationError(
                "invalid_external_timeout",
                "External EPS timeout must be finite and positive.",
                stage="backend_setup",
                details={"timeout_seconds": timeout_seconds},
            )
        timeout_seconds = float(timeout_seconds)

        left = _canonical_tree(true_tree, "true_tree")
        right = _canonical_tree(reconstructed_tree, "reconstructed_tree")
        inputs = _input_metadata(left, right)

        repository_root = Path(repository_root).expanduser().resolve()
        python_executable = Path(python_executable).expanduser()
        if not python_executable.is_absolute():
            python_executable = Path.cwd() / python_executable
        if not python_executable.is_file():
            raise RepeatedLabelEvaluationError(
                "external_interpreter_unavailable",
                "The explicit external EPS interpreter does not exist.",
                stage="backend_setup",
                details={"python_executable": str(python_executable)},
            )

        if source_audit is None:
            source_audit = inspect_external_eps_source(repository_root)
        else:
            source_audit = deepcopy(source_audit)
        audit_matches = (
            source_audit.get("status")
            == "source_verified_external_dependency_unchecked"
            and source_audit.get("repository_root") == str(repository_root)
            and source_audit.get("revision") == EPS_AUDITED_REVISION
            and source_audit.get("source_matches_audited_identity") is True
        )
        if not audit_matches:
            raise RepeatedLabelEvaluationError(
                "external_backend_unavailable",
                "The external EPS checkout did not pass its pinned source audit.",
                stage="backend_setup",
                details={
                    "repository_root": str(repository_root),
                    "source_audit_status": source_audit.get("status"),
                    "source_revision": source_audit.get("revision"),
                },
            )

        payload = {
            "left_nodes": list(left.labels),
            "left_parents": list(left.parents),
            "right_nodes": list(right.labels),
            "right_parents": list(right.parents),
        }
        try:
            completed = subprocess.run(
                [
                    str(python_executable),
                    "-I",
                    "-B",
                    "-c",
                    _EXTERNAL_EPS_APPROX_EVALUATION_PROGRAM,
                    str(repository_root),
                ],
                check=True,
                capture_output=True,
                input=json.dumps(payload, allow_nan=False),
                text=True,
                timeout=timeout_seconds,
            )
            stdout_lines = completed.stdout.splitlines()
            if not stdout_lines:
                raise ValueError("External EPS produced no stdout record.")
            child_result = json.loads(stdout_lines[-1])
            module_path = Path(child_result["backend_module_path"]).resolve()
            orientations = child_result["orientations"]
        except subprocess.TimeoutExpired as exc:
            raise RepeatedLabelEvaluationError(
                "external_backend_timeout",
                f"External EPS exceeded its {timeout_seconds}-second timeout.",
                stage="backend_execution",
                details={"timeout_seconds": timeout_seconds},
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RepeatedLabelEvaluationError(
                "external_backend_failed",
                "External EPS exited unsuccessfully.",
                stage="backend_execution",
                details={
                    "returncode": exc.returncode,
                    "stdout_tail": (exc.stdout or "")[-4000:],
                    "stderr_tail": (exc.stderr or "")[-4000:],
                },
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise RepeatedLabelEvaluationError(
                "external_backend_failed",
                "External EPS could not be executed.",
                stage="backend_execution",
                details={"error_type": type(exc).__name__, "message": str(exc)},
            ) from exc
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RepeatedLabelEvaluationError(
                "external_backend_invalid_output",
                "External EPS returned an invalid result record.",
                stage="backend_output",
                details={"error_type": type(exc).__name__, "message": str(exc)},
            ) from exc

        expected_versions = {
            "gurobi_version": [13, 0, 2],
            "networkx_version": "3.4.2",
            "numpy_version": "2.2.6",
        }
        if any(
            child_result.get(field) != expected
            for field, expected in expected_versions.items()
        ):
            raise RepeatedLabelEvaluationError(
                "external_backend_version_mismatch",
                "External EPS dependency versions differ from the frozen environment.",
                stage="backend_output",
                details={
                    field: child_result.get(field)
                    for field in expected_versions
                },
            )
        if not module_path.is_file() or not module_path.is_relative_to(
            repository_root
        ):
            raise RepeatedLabelEvaluationError(
                "external_backend_invalid_output",
                "External EPS returned an invalid module identity.",
                stage="backend_output",
                details={"backend_module_path": str(module_path)},
            )

        directional_values = {}
        directional_durations = {}
        for orientation in ("forward", "reverse"):
            record = orientations.get(orientation)
            if not isinstance(record, dict):
                raise RepeatedLabelEvaluationError(
                    "external_backend_invalid_output",
                    "External EPS omitted a directional result.",
                    stage="backend_output",
                    details={"orientation": orientation},
                )
            raw_value = record.get("raw_value")
            duration = record.get("duration_seconds")
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
                or float(raw_value) < 0.0
                or not math.isclose(
                    float(raw_value),
                    round(float(raw_value)),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or float(duration) < 0.0
                or record.get("time_limit_exceeded") is not False
            ):
                raise RepeatedLabelEvaluationError(
                    "external_backend_invalid_output",
                    "External EPS returned an invalid directional value.",
                    stage="backend_output",
                    details={"orientation": orientation, "record": record},
                )
            directional_values[orientation] = float(raw_value)
            directional_durations[orientation] = float(duration)

        raw_value = max(directional_values.values())
        denominator = float(max(left.edge_count, right.edge_count))
        if raw_value > denominator:
            raise RepeatedLabelEvaluationError(
                "external_backend_invalid_output",
                "External EPS preserved more edges than the declared denominator.",
                stage="backend_output",
                details={
                    "raw_value": raw_value,
                    "normalization_denominator": denominator,
                },
            )
        if denominator == 0.0:
            value = None
            degeneracy = "zero_edge_denominator"
        else:
            value = raw_value / denominator
            degeneracy = "none"

        result = {
            "schema_version": REPEATED_LABEL_RESULT_SCHEMA_VERSION,
            "status": "success",
            "metric_id": metric_id,
            "metric_contract": candidate_metric_contract(metric_id),
            "inputs": inputs,
            "metric": {
                "raw_value": raw_value,
                "normalization_denominator": denominator,
                "value": value,
                "degeneracy": degeneracy,
            },
            "external_execution": {
                "backend": (
                    "edge_preservation_similarity.compute_eps.compute_similarity"
                ),
                "algorithm": "EDGE-PRESERVATION-SIM-APPROX",
                "repository_root": str(repository_root),
                "python_executable": str(python_executable),
                "source_revision": source_audit["revision"],
                "backend_module_path": str(module_path),
                "backend_module_sha256": _sha256_file(module_path),
                **expected_versions,
                "timeout_seconds": timeout_seconds,
                "direction_combination": "maximum_of_forward_and_reverse",
                "directional_raw_values": directional_values,
                "directional_duration_seconds": directional_durations,
                "diagnostic_stdout_line_count": child_result.get(
                    "diagnostic_stdout_line_count"
                ),
                "fallback_used": False,
            },
        }
        validate_repeated_label_result(result)
        return result
    except RepeatedLabelEvaluationError as error:
        captured_inputs = locals().get("inputs")
        return _failure_result(metric_id, error, captured_inputs)


def evaluate_repeated_label_tree_pair_result(
    true_tree: Any,
    reconstructed_tree: Any,
    metric_id: str,
) -> Dict[str, Any]:
    """Return one strict status-bearing candidate-evaluator result.

    CUTED and approximate EPS return ``external_execution_required`` unless
    their explicit runners are called. This prevents accidental external
    execution or substitution of another tree metric.
    """
    contract = candidate_metric_contract(metric_id)
    try:
        left = _canonical_tree(true_tree, "true_tree")
        right = _canonical_tree(reconstructed_tree, "reconstructed_tree")
        inputs = _input_metadata(left, right)
        if contract["implementation_status"] != "implemented_reference_only":
            failure_code = (
                "external_execution_required"
                if contract["implementation_status"] == "external_runner_available"
                else "backend_not_integrated"
            )
            raise RepeatedLabelEvaluationError(
                failure_code,
                f"The declared {metric_id} backend requires explicit external execution.",
                stage="backend",
                details={
                    "implementation": contract["implementation"],
                    "implementation_status": contract["implementation_status"],
                },
            )
        _check_reference_size(metric_id, left, right)

        if metric_id == "uted_exact_reference":
            raw_value = _uted_exact_unit_cost(left, right)
            denominator = float(left.node_count + right.node_count)
            value: Optional[float] = raw_value / denominator
            degeneracy = "none"
        elif metric_id == "eps_exact_reference":
            raw_value = float(_eps_exact_preserved_edges(left, right))
            denominator = float(max(left.edge_count, right.edge_count))
            if denominator == 0.0:
                value = None
                degeneracy = "zero_edge_denominator"
            else:
                value = raw_value / denominator
                degeneracy = "none"
        else:  # guarded by the implementation status above
            raise AssertionError(f"Missing implemented dispatcher for {metric_id}")

        result = {
            "schema_version": REPEATED_LABEL_RESULT_SCHEMA_VERSION,
            "status": "success",
            "metric_id": metric_id,
            "metric_contract": contract,
            "inputs": inputs,
            "metric": {
                "raw_value": raw_value,
                "normalization_denominator": denominator,
                "value": value,
                "degeneracy": degeneracy,
            },
        }
        validate_repeated_label_result(result)
        return result
    except RepeatedLabelEvaluationError as error:
        captured_inputs = locals().get("inputs")
        return _failure_result(metric_id, error, captured_inputs)


def _schema_error(message: str, **details: Any) -> RepeatedLabelEvaluationError:
    return RepeatedLabelEvaluationError(
        "invalid_repeated_label_result",
        message,
        stage="result_schema",
        details=details,
    )


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _schema_error(f"{field} must be numeric.", field=field)
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise _schema_error(f"{field} must be finite and nonnegative.", field=field)
    return numeric


def validate_repeated_label_result(result: Dict[str, Any]) -> None:
    """Raise when a candidate result is not strict and self-interpretable."""
    if not isinstance(result, dict):
        raise _schema_error("Repeated-label result must be a mapping.")
    if result.get("schema_version") != REPEATED_LABEL_RESULT_SCHEMA_VERSION:
        raise _schema_error("Unknown or missing repeated-label result schema.")
    metric_id = result.get("metric_id")
    if metric_id not in _METRIC_CONTRACTS:
        raise _schema_error("Unknown or missing metric id.")
    if result.get("metric_contract") != candidate_metric_contract(metric_id):
        raise _schema_error("Metric contract is missing or altered.")
    inputs = result.get("inputs")
    if not isinstance(inputs, dict):
        raise _schema_error("Inputs metadata must be a mapping.")

    status = result.get("status")
    if status == "failure":
        failure = result.get("failure")
        if not isinstance(failure, dict):
            raise _schema_error("Failure result must contain a failure mapping.")
        for field in ("code", "stage", "message"):
            if not isinstance(failure.get(field), str) or not failure[field]:
                raise _schema_error(f"Failure field {field} must be a non-empty string.")
        if not isinstance(failure.get("details"), dict):
            raise _schema_error("Failure details must be a mapping.")
    elif status == "success":
        if metric_id not in SUCCESS_CAPABLE_METRICS:
            raise _schema_error("The declared metric has no success-capable runner.")
        metric = result.get("metric")
        if not isinstance(metric, dict):
            raise _schema_error("Success result must contain a metric mapping.")
        raw_value = _finite_nonnegative(metric.get("raw_value"), "metric.raw_value")
        denominator = _finite_nonnegative(
            metric.get("normalization_denominator"),
            "metric.normalization_denominator",
        )
        value = metric.get("value")
        degeneracy = metric.get("degeneracy")
        if denominator == 0.0:
            if metric_id not in {"eps_exact_reference", "eps_approx_external"}:
                raise _schema_error("Only zero-edge EPS may have a zero denominator.")
            if (
                raw_value != 0.0
                or value is not None
                or degeneracy != "zero_edge_denominator"
            ):
                raise _schema_error("Zero-edge EPS must be null and explicitly degenerate.")
        else:
            numeric_value = _finite_nonnegative(value, "metric.value")
            if numeric_value > 1.0:
                raise _schema_error("Normalized metric value must be in [0, 1].")
            if not math.isclose(
                numeric_value,
                raw_value / denominator,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise _schema_error("Normalized value disagrees with raw value and denominator.")
            if degeneracy != "none":
                raise _schema_error("Positive-denominator result cannot be degenerate.")
        if metric_id == "cuted_edist":
            execution = result.get("external_execution")
            if not isinstance(execution, dict):
                raise _schema_error(
                    "External CUTED success requires execution provenance."
                )
            expected = {
                "backend": "edist.uted.uted",
                "source_revision": EDIST_AUDITED_REVISION,
                "cost_mode": "unit",
                "fallback_used": False,
            }
            for field, expected_value in expected.items():
                if execution.get(field) != expected_value:
                    raise _schema_error(
                        f"External CUTED execution field {field} is invalid."
                    )
            for field in (
                "repository_root",
                "python_executable",
                "backend_module_path",
                "backend_module_sha256",
            ):
                if not isinstance(execution.get(field), str) or not execution[field]:
                    raise _schema_error(
                        f"External CUTED execution field {field} is missing."
                    )
            backend_hash = execution["backend_module_sha256"]
            if len(backend_hash) != 64 or any(
                character not in "0123456789abcdef" for character in backend_hash
            ):
                raise _schema_error("External CUTED module hash is invalid.")
            timeout = execution.get("timeout_seconds")
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(float(timeout))
                or float(timeout) <= 0.0
            ):
                raise _schema_error("External CUTED timeout is invalid.")
        elif metric_id == "eps_approx_external":
            execution = result.get("external_execution")
            if not isinstance(execution, dict):
                raise _schema_error(
                    "External approximate EPS success requires execution provenance."
                )
            expected = {
                "backend": (
                    "edge_preservation_similarity.compute_eps.compute_similarity"
                ),
                "algorithm": "EDGE-PRESERVATION-SIM-APPROX",
                "source_revision": EPS_AUDITED_REVISION,
                "backend_module_sha256": EPS_AUDITED_SOURCE_SHA256[
                    "edge_preservation_similarity/compute_eps.py"
                ],
                "gurobi_version": [13, 0, 2],
                "networkx_version": "3.4.2",
                "numpy_version": "2.2.6",
                "direction_combination": "maximum_of_forward_and_reverse",
                "fallback_used": False,
            }
            for field, expected_value in expected.items():
                if execution.get(field) != expected_value:
                    raise _schema_error(
                        f"External EPS execution field {field} is invalid."
                    )
            for field in (
                "repository_root",
                "python_executable",
                "backend_module_path",
            ):
                if not isinstance(execution.get(field), str) or not execution[field]:
                    raise _schema_error(
                        f"External EPS execution field {field} is missing."
                    )
            timeout = execution.get("timeout_seconds")
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(float(timeout))
                or float(timeout) <= 0.0
            ):
                raise _schema_error("External EPS timeout is invalid.")
            directional_values = execution.get("directional_raw_values")
            directional_durations = execution.get(
                "directional_duration_seconds"
            )
            if (
                not isinstance(directional_values, dict)
                or set(directional_values) != {"forward", "reverse"}
                or not isinstance(directional_durations, dict)
                or set(directional_durations) != {"forward", "reverse"}
            ):
                raise _schema_error("External EPS directional records are invalid.")
            checked_values = [
                _finite_nonnegative(
                    directional_values[orientation],
                    f"external_execution.directional_raw_values.{orientation}",
                )
                for orientation in ("forward", "reverse")
            ]
            for orientation in ("forward", "reverse"):
                _finite_nonnegative(
                    directional_durations[orientation],
                    (
                        "external_execution.directional_duration_seconds."
                        f"{orientation}"
                    ),
                )
            if not math.isclose(
                raw_value,
                max(checked_values),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise _schema_error(
                    "External EPS raw value is not the directional maximum."
                )
        elif "external_execution" in result:
            raise _schema_error(
                "Reference metric success must not claim external execution."
            )
    else:
        raise _schema_error("Result status must be success or failure.")

    try:
        json.dumps(result, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise _schema_error(f"Result is not strict JSON: {exc}") from exc
