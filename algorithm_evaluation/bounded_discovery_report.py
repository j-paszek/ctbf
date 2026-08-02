#!/usr/bin/env python3
"""Read-only detailed reporting for frozen bounded-discovery JSON records."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_evaluation.bounded_discovery import (  # noqa: E402
    DEFAULT_MANIFEST,
    _read_json,
    _write_json_atomic,
    file_sha256,
    load_all_records,
    paired_metric_summary,
    validate_manifest,
)


REPORT_SCHEMA_VERSION = "ctbf-bounded-reconstruction-detailed-report-v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experimental_description"
    / "g0_02_c_bounded_discovery_results.json"
)


def numeric_summary(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def aggregate_arm_metrics(records, arm_ids):
    result = {}
    for arm_id in arm_ids:
        successful = [
            record["arms"][arm_id]
            for record in records
            if record.get("status") == "complete"
            and record["arms"][arm_id].get("status") == "success"
        ]
        result[arm_id] = {
            metric: numeric_summary([arm["metrics"][metric] for arm in successful])
            for metric in ("ad_f1", "grf")
        }
    return result


def aggregate_fast_order_audit(records):
    order_names = (
        "canonical",
        "reverse_canonical",
        "rotate_canonical_left_by_one",
    )
    result = {
        "record_count": 0,
        "records_with_any_matrix_change": 0,
        "orders": {
            name: {
                "records_with_changed_pairs": 0,
                "changed_unordered_pairs_total": 0,
                "ordered_tree_changed_records": 0,
                "no_time_tree_changed_records": 0,
            }
            for name in order_names
        },
    }
    for record in records:
        if record.get("status") != "complete":
            continue
        result["record_count"] += 1
        audit = record["fast_order_audit"]
        result["records_with_any_matrix_change"] += int(
            audit["matrix"]["any_matrix_change"]
        )
        matrix_orders = audit["matrix"]["orders"]
        ordered_trees = audit["tree"]["ordered"]["orders"]
        no_time_trees = audit["tree"]["no_time"]["orders"]
        if not (
            len(matrix_orders) == len(ordered_trees) == len(no_time_trees) == 3
        ):
            raise ValueError("Fast permutation panel must contain exactly three orders.")
        for index, name in enumerate(order_names):
            changed_pairs = matrix_orders[index]["changed_unordered_pairs_vs_first"]
            result["orders"][name]["changed_unordered_pairs_total"] += changed_pairs
            result["orders"][name]["records_with_changed_pairs"] += int(
                changed_pairs > 0
            )
            result["orders"][name]["ordered_tree_changed_records"] += int(
                ordered_trees[index]["topology_changed_vs_first"]
            )
            result["orders"][name]["no_time_tree_changed_records"] += int(
                no_time_trees[index]["topology_changed_vs_first"]
            )
    return result


def aggregate_tree_identity(records):
    comparisons = {
        "fast_equals_minimum": ("temporal_fast", "temporal_minimum"),
        "directed_equals_minimum": ("temporal_directed", "temporal_minimum"),
        "directed_no_time_equals_minimum_no_time": (
            "temporal_directed_no_time",
            "temporal_minimum_no_time",
        ),
        "temporal_equals_no_time": (
            "temporal_minimum",
            "temporal_minimum_no_time",
        ),
        "temporal_equals_anticentral": (
            "temporal_minimum",
            "anticentral_parsimony",
        ),
    }
    result = {}
    for name, (left, right) in comparisons.items():
        complete = 0
        equal = 0
        for record in records:
            if record.get("status") != "complete":
                continue
            left_record = record["arms"].get(left, {})
            right_record = record["arms"].get(right, {})
            if left_record.get("status") != "success" or right_record.get("status") != "success":
                continue
            complete += 1
            equal += int(left_record["tree"] == right_record["tree"])
        result[name] = {
            "complete_pairs": complete,
            "equal_trees": equal,
            "different_trees": complete - equal,
        }
    return result


def aggregate_input_summary(records):
    fields = (
        "truth_node_count",
        "observed_occurrence_count",
        "observed_unique_state_count",
        "recurrent_observation_count",
    )
    result = {
        field: numeric_summary(
            [
                record["input_summary"][field]
                for record in records
                if record.get("status") == "complete"
            ]
        )
        for field in fields
    }
    result["trivial_single_state_records"] = sum(
        record.get("status") == "complete"
        and record["input_summary"]["observed_unique_state_count"] <= 1
        for record in records
    )
    return result


def aggregate_truth_selection(records):
    totals = Counter()
    for record in records:
        if record.get("status") == "complete":
            totals.update(record["direction_truth_selection"])
    return dict(totals)


def verify_checksums(output_root):
    output_root = Path(output_root)
    checksum_path = output_root / "checksums.json"
    expected = _read_json(checksum_path)
    actual_paths = {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*.json")
        if path.name != "checksums.json"
        and "work" not in path.relative_to(output_root).parts
    }
    expected_paths = set(expected)
    mismatches = sorted(
        relative
        for relative in expected_paths & actual_paths
        if file_sha256(output_root / relative) != expected[relative]
    )
    return {
        "status": (
            "valid"
            if not mismatches and expected_paths == actual_paths
            else "invalid"
        ),
        "checked_file_count": len(expected_paths & actual_paths),
        "missing_files": sorted(expected_paths - actual_paths),
        "unlisted_json_files": sorted(actual_paths - expected_paths),
        "hash_mismatches": mismatches,
    }


def build_detailed_report(
    records,
    manifest,
    summary,
    *,
    output_root,
):
    analysis = manifest["analysis_contract"]
    arm_ids = [arm["id"] for arm in manifest["portfolio_arms"]]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "record_count": len(records),
        "source_artifacts": {
            "summary": str(Path(output_root) / "summary.json"),
            "summary_sha256": file_sha256(Path(output_root) / "summary.json"),
            "checksums": str(Path(output_root) / "checksums.json"),
            "checksums_sha256": file_sha256(Path(output_root) / "checksums.json"),
            "checksum_verification": verify_checksums(output_root),
        },
        "absolute_arm_metrics": aggregate_arm_metrics(records, arm_ids),
        "additional_paired_comparisons": {
            "temporal_vs_anticentral_ad_f1": paired_metric_summary(
                records,
                "temporal_minimum",
                "anticentral_parsimony",
                "ad_f1",
                analysis,
            ),
            "temporal_vs_anticentral_grf": paired_metric_summary(
                records,
                "temporal_minimum",
                "anticentral_parsimony",
                "grf",
                analysis,
            ),
            "legacy_vs_rooted_ad_f1": paired_metric_summary(
                records,
                "legacy_closest_pair",
                "rooted_labeled_nj",
                "ad_f1",
                analysis,
            ),
        },
        "fast_order_sensitivity": aggregate_fast_order_audit(records),
        "tree_identity": aggregate_tree_identity(records),
        "input_summary": aggregate_input_summary(records),
        "direction_truth_selection": aggregate_truth_selection(records),
        "frozen_gate_results": summary["promotion_gates"],
        "discovery_only": True,
    }
    return report
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = validate_manifest(_read_json(args.manifest))
    output_root = (
        Path(args.output_root)
        if args.output_root
        else PROJECT_ROOT / manifest["result_root"]
    )
    records, missing = load_all_records(output_root, manifest)
    if missing:
        raise ValueError(f"Detailed report requires all records; missing {len(missing)}.")
    summary = _read_json(output_root / "summary.json")
    report = build_detailed_report(
        records,
        manifest,
        summary,
        output_root=output_root,
    )
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "aggregate_arm_metrics",
    "aggregate_fast_order_audit",
    "aggregate_tree_identity",
    "build_detailed_report",
    "verify_checksums",
]
