"""Read-only bounded replay for the versioned CTBF evaluator contract."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_contract import (  # noqa: E402
    EVALUATION_RESULT_SCHEMA_VERSION,
    evaluate_tree_pair_result,
    validate_evaluation_result,
)
from evaluator_full import normalize_cell_labels  # noqa: E402


INTEGRITY_REPORT_SCHEMA_VERSION = "ctbf-evaluator-integrity-report-v1"
DEFAULT_BOUNDED_ROOT = (
    PROJECT_ROOT / "experimental_results" / "g0_02_c_g0_03_b_discovery_v1_1"
)


def _load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deserialize_frozen_tree(serialized):
    """Read the explicit bounded-discovery tree format without mutating it."""
    tree = nx.DiGraph() if serialized.get("directed", True) else nx.Graph()
    for node in serialized.get("nodes", []):
        tree.add_node(node["node_id"], **node.get("attributes", {}))
    for edge in serialized.get("edges", []):
        tree.add_edge(
            edge["source"],
            edge["target"],
            **edge.get("attributes", {}),
        )
    return tree


def observed_labels_from_record(record):
    observations = (
        observation.get("cell_id")
        for biopsy in record.get("replay_input", {}).get("biopsies", [])
        for observation in biopsy.get("observations", [])
    )
    return normalize_cell_labels(observations) or frozenset()


def _metric_drift(issues, record_name, arm_name, field, stored, recomputed, tolerance):
    try:
        matches = math.isclose(
            float(stored),
            float(recomputed),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        issues.append(
            {
                "code": "stored_metric_drift",
                "record": record_name,
                "arm": arm_name,
                "field": field,
                "stored": stored,
                "recomputed": recomputed,
            }
        )


def audit_record(record, record_name, tolerance=1e-12):
    """Recompute all successful arms in one frozen record."""
    issues = []
    checked_arm_count = 0
    if not isinstance(record, dict):
        return checked_arm_count, [
            {"code": "invalid_record", "record": record_name, "message": "Expected a mapping."}
        ]
    if record.get("status") != "complete":
        issues.append(
            {
                "code": "record_not_complete",
                "record": record_name,
                "stored_status": record.get("status"),
            }
        )
        return checked_arm_count, issues

    replay_input = record.get("replay_input")
    if not isinstance(replay_input, dict) or not isinstance(replay_input.get("truth_tree"), dict):
        issues.append({"code": "missing_replay_truth", "record": record_name})
        return checked_arm_count, issues
    observed_labels = observed_labels_from_record(record)
    if not observed_labels:
        issues.append({"code": "missing_observation_labels", "record": record_name})
        return checked_arm_count, issues
    try:
        true_tree = deserialize_frozen_tree(replay_input["truth_tree"])
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(
            {
                "code": "truth_deserialization_failure",
                "record": record_name,
                "message": str(exc),
            }
        )
        return checked_arm_count, issues

    arms = record.get("arms")
    if not isinstance(arms, dict) or not arms:
        issues.append({"code": "missing_arms", "record": record_name})
        return checked_arm_count, issues
    for arm_name in sorted(arms):
        arm = arms[arm_name]
        if not isinstance(arm, dict):
            issues.append(
                {"code": "invalid_arm", "record": record_name, "arm": arm_name}
            )
            continue
        if arm.get("status") != "success":
            issues.append(
                {
                    "code": "arm_not_successful",
                    "record": record_name,
                    "arm": arm_name,
                    "stored_status": arm.get("status"),
                }
            )
            continue
        if not isinstance(arm.get("tree"), dict):
            issues.append(
                {"code": "missing_reconstructed_tree", "record": record_name, "arm": arm_name}
            )
            continue
        checked_arm_count += 1
        try:
            reconstructed_tree = deserialize_frozen_tree(arm["tree"])
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                {
                    "code": "reconstructed_deserialization_failure",
                    "record": record_name,
                    "arm": arm_name,
                    "message": str(exc),
                }
            )
            continue

        result = evaluate_tree_pair_result(true_tree, reconstructed_tree, observed_labels)
        validate_evaluation_result(result)
        if result["status"] != "success":
            issues.append(
                {
                    "code": "evaluator_contract_failure",
                    "record": record_name,
                    "arm": arm_name,
                    "failure": result["failure"],
                }
            )
            continue

        stored_metrics = arm.get("metrics", {})
        recomputed = result["metrics"]
        _metric_drift(
            issues,
            record_name,
            arm_name,
            "ad_f1",
            stored_metrics.get("ad_f1"),
            recomputed["ad_f1"],
            tolerance,
        )
        _metric_drift(
            issues,
            record_name,
            arm_name,
            "grf",
            stored_metrics.get("grf"),
            recomputed["grf"],
            tolerance,
        )
        stored_counts = stored_metrics.get("ad_f1_counts", {})
        expected_counts = {
            "TP": recomputed["ad_counts"]["tp"],
            "FP": recomputed["ad_counts"]["fp"],
            "FN": recomputed["ad_counts"]["fn"],
            "num_unique_pairs_true": recomputed["ad_counts"]["true_unique_pair_count"],
            "num_unique_pairs_rec": recomputed["ad_counts"]["reconstructed_unique_pair_count"],
        }
        for field, expected in expected_counts.items():
            if stored_counts.get(field) != expected:
                issues.append(
                    {
                        "code": "stored_count_drift",
                        "record": record_name,
                        "arm": arm_name,
                        "field": field,
                        "stored": stored_counts.get(field),
                        "recomputed": expected,
                    }
                )
    return checked_arm_count, issues


def verify_checksum_inventory(root):
    path = root / "checksums.json"
    if not path.is_file():
        return 0, [{"code": "missing_checksum_inventory", "path": str(path)}]
    try:
        inventory = _load_json(path)
    except (OSError, ValueError) as exc:
        return 0, [{"code": "invalid_checksum_inventory", "message": str(exc)}]
    if not isinstance(inventory, dict):
        return 0, [{"code": "invalid_checksum_inventory", "message": "Expected a mapping."}]

    issues = []
    root_resolved = root.resolve()
    for relative_path, expected in sorted(inventory.items()):
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            issues.append({"code": "checksum_path_escape", "path": relative_path})
            continue
        if not candidate.is_file():
            issues.append({"code": "checksum_file_missing", "path": relative_path})
            continue
        actual = _sha256(candidate)
        if actual != expected:
            issues.append(
                {
                    "code": "checksum_mismatch",
                    "path": relative_path,
                    "expected": expected,
                    "actual": actual,
                }
            )
    return len(inventory), issues


def run_integrity(root, max_records=30, tolerance=1e-12, verify_checksums=True):
    root = Path(root)
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    issues = []
    checksum_file_count = 0
    if not root.is_dir():
        issues.append({"code": "bounded_root_missing", "path": str(root)})
        record_paths = []
    else:
        if verify_checksums:
            checksum_file_count, checksum_issues = verify_checksum_inventory(root)
            issues.extend(checksum_issues)
        record_paths = sorted((root / "records").glob("*/*.json"))

    selected_paths = record_paths[:max_records]
    if not selected_paths:
        issues.append({"code": "no_bounded_records", "path": str(root / "records")})

    checked_arm_count = 0
    for path in selected_paths:
        relative_name = str(path.relative_to(root))
        try:
            record = _load_json(path)
        except (OSError, ValueError) as exc:
            issues.append(
                {"code": "record_load_failure", "record": relative_name, "message": str(exc)}
            )
            continue
        arm_count, record_issues = audit_record(record, relative_name, tolerance=tolerance)
        checked_arm_count += arm_count
        issues.extend(record_issues)

    report = {
        "schema_version": INTEGRITY_REPORT_SCHEMA_VERSION,
        "evaluator_schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
        "status": "pass" if not issues else "failure",
        "bounded_root": str(root),
        "available_record_count": len(record_paths),
        "checked_record_count": len(selected_paths),
        "checked_arm_count": checked_arm_count,
        "checksum_file_count": checksum_file_count,
        "absolute_tolerance": tolerance,
        "issue_count": len(issues),
        "issues": issues,
    }
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Verify frozen bounded-discovery checksums and replay AD-F1/GRF "
            "through the versioned evaluator schema without writing artifacts."
        )
    )
    parser.add_argument("--bounded-root", type=Path, default=DEFAULT_BOUNDED_ROOT)
    parser.add_argument("--max-records", type=int, default=30)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip the frozen artifact checksum inventory (primarily for focused tests).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.max_records <= 0:
        raise SystemExit("--max-records must be positive")
    if args.absolute_tolerance < 0 or not math.isfinite(args.absolute_tolerance):
        raise SystemExit("--absolute-tolerance must be finite and nonnegative")
    report = run_integrity(
        args.bounded_root,
        max_records=args.max_records,
        tolerance=args.absolute_tolerance,
        verify_checksums=not args.skip_checksums,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
