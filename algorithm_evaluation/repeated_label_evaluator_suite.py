"""Tiny frozen discrimination suite for repeated-label tree evaluators.

This suite is intentionally hand-checkable.  It does not read benchmark or
paper-result directories and is safe to run as a focused preflight.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from evaluation_contract import evaluate_tree_pair_result, validate_evaluation_result
from repeated_label_evaluation import (
    CANDIDATE_METRIC_IDS,
    DEFAULT_EXTERNAL_EDIST_ROOT,
    DEFAULT_EXTERNAL_EPS_ROOT,
    DEFAULT_EXTERNAL_UTED_ROOT,
    EXTERNAL_CUTED_DEFAULT_TIMEOUT_SECONDS,
    EXTERNAL_EPS_DEFAULT_TIMEOUT_SECONDS,
    build_external_edist_unit_probe,
    candidate_metric_contract,
    evaluate_external_cuted_tree_pair_result,
    evaluate_external_eps_approx_tree_pair_result,
    evaluate_repeated_label_tree_pair_result,
    inspect_external_edist_source,
    inspect_external_eps_source,
    inspect_external_uted_source,
    probe_external_edist_semantics,
    probe_external_uted_semantics,
    validate_repeated_label_result,
)


DISCRIMINATIVE_SUITE_SCHEMA_VERSION = (
    "ctbf-repeated-label-evaluator-discrimination-suite-v1"
)
DISCRIMINATIVE_SUITE_ID = "ctbf-g1-04-b-tiny-suite-v1"


def _tree(nodes: Iterable[tuple], edges: Iterable[tuple]) -> Dict[str, Any]:
    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": node_id, "cell_id": cell_id, "genome": list(genome)}
            for node_id, cell_id, genome in nodes
        ],
        "links": [
            {"source": source, "target": target}
            for source, target in edges
        ],
    }


def _case(
    case_id: str,
    feature: str,
    expected_relation: str,
    true_tree: Dict[str, Any],
    reconstructed_tree: Dict[str, Any],
    observation_labels: Iterable[str],
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "feature": feature,
        "expected_relation": expected_relation,
        "true_tree": true_tree,
        "reconstructed_tree": reconstructed_tree,
        "observation_labels": list(observation_labels),
    }


def build_discriminative_cases() -> Tuple[Dict[str, Any], ...]:
    """Return fresh owned trees for the ten frozen G1-04-B cases."""
    a = (2, 2)
    b = (2, 3)
    c = (3, 3)
    d = (3, 2)
    cases = (
        _case(
            "node_id_relabeling",
            "Graph-local node ids change bijectively without changing labels or topology.",
            "equivalent",
            _tree(((0, "A", a), (1, "B", b), (2, "C", c)), ((0, 1), (0, 2))),
            _tree(
                (("root-x", "A", a), ("leaf-z", "B", b), ("leaf-y", "C", c)),
                (("root-x", "leaf-z"), ("root-x", "leaf-y")),
            ),
            ("A", "B", "C"),
        ),
        _case(
            "sibling_permutation",
            "Sibling and serialized edge order change without changing the unordered tree.",
            "equivalent",
            _tree(
                (("r", "A", a), ("b", "B", b), ("c", "C", c), ("d", "D", d)),
                (("r", "b"), ("r", "c"), ("b", "d")),
            ),
            _tree(
                (("d", "D", d), ("c", "C", c), ("b", "B", b), ("r", "A", a)),
                (("b", "d"), ("r", "c"), ("r", "b")),
            ),
            ("A", "B", "C", "D"),
        ),
        _case(
            "displaced_repeated_state",
            "A repeated A occurrence moves across the B branching relation.",
            "different",
            _tree(
                (("r", "A", a), ("b", "B", b), ("a2", "A", a), ("c", "C", c)),
                (("r", "b"), ("b", "a2"), ("b", "c")),
            ),
            _tree(
                ((10, "A", a), (11, "A", a), (12, "B", b), (13, "C", c)),
                ((10, 11), (11, 12), (11, 13)),
            ),
            ("A", "B", "C"),
        ),
        _case(
            "copied_state_internal_node",
            "One repeated A occurrence refines the A-to-B edge.",
            "different",
            _tree((("r", "A", a), ("b", "B", b)), (("r", "b"),)),
            _tree(
                (("r2", "A", a), ("copy", "A", a), ("b2", "B", b)),
                (("r2", "copy"), ("copy", "b2")),
            ),
            ("A", "B"),
        ),
        _case(
            "unlabeled_refinement",
            "An explicitly unlabeled internal occurrence refines A-to-B.",
            "different",
            _tree((("r", "A", a), ("b", "B", b)), (("r", "b"),)),
            _tree(
                (("r2", "A", a), ("u", None, a), ("b2", "B", b)),
                (("r2", "u"), ("u", "b2")),
            ),
            ("A", "B"),
        ),
        _case(
            "missing_observation",
            "The reconstructed tree omits observed state B from an A-to-B-to-C path.",
            "different",
            _tree(
                (("r", "A", a), ("b", "B", b), ("c", "C", c)),
                (("r", "b"), ("b", "c")),
            ),
            _tree((("r2", "A", a), ("c2", "C", c)), (("r2", "c2"),)),
            ("A", "B", "C"),
        ),
        _case(
            "wrong_ancestry",
            "B and C reverse order below the same A root.",
            "different",
            _tree(
                (("r", "A", a), ("b", "B", b), ("c", "C", c)),
                (("r", "b"), ("b", "c")),
            ),
            _tree(
                (("r2", "A", a), ("b2", "B", b), ("c2", "C", c)),
                (("r2", "c2"), ("c2", "b2")),
            ),
            ("A", "B", "C"),
        ),
        _case(
            "one_bin_cnp_perturbation",
            "One terminal profile differs by one copy in one bin and receives a new state label.",
            "different",
            _tree((("r", "R", a), ("b", "B", b)), (("r", "b"),)),
            _tree((("r2", "R", a), ("bp", "B_plus_one", (2, 4))), (("r2", "bp"),)),
            ("R", "B"),
        ),
        _case(
            "same_cnp_path",
            "Truth contains an A-to-A-to-B recurrent-label path absent from reconstruction.",
            "different",
            _tree(
                (("a1", "A", a), ("a2", "A", a), ("b", "B", b)),
                (("a1", "a2"), ("a2", "b")),
            ),
            _tree((("ar", "A", a), ("br", "B", b)), (("ar", "br"),)),
            ("A", "B"),
        ),
        _case(
            "copied_state_incident_branches",
            "A copied A occurrence is inserted above two labeled child branches.",
            "different",
            _tree(
                (("a", "A", a), ("b", "B", b), ("c", "C", c)),
                (("a", "b"), ("a", "c")),
            ),
            _tree(
                (("a1", "A", a), ("a2", "A", a), ("b2", "B", b), ("c2", "C", c)),
                (("a1", "a2"), ("a2", "b2"), ("a2", "c2")),
            ),
            ("A", "B", "C"),
        ),
    )
    return deepcopy(cases)


def build_suite_report(
    metric_ids: Iterable[str] = CANDIDATE_METRIC_IDS,
    *,
    external_cuted_options: Dict[str, Any] | None = None,
    external_eps_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metric_ids = tuple(metric_ids)
    contracts = {
        metric_id: candidate_metric_contract(metric_id)
        for metric_id in metric_ids
    }
    records = []
    native_status_counts = {"success": 0, "failure": 0}
    candidate_status_counts = {
        metric_id: {"success": 0, "failure": 0}
        for metric_id in metric_ids
    }

    for case in build_discriminative_cases():
        native = evaluate_tree_pair_result(
            case["true_tree"],
            case["reconstructed_tree"],
            case["observation_labels"],
        )
        validate_evaluation_result(native)
        native_status_counts[native["status"]] += 1
        candidate_results = {}
        for metric_id in metric_ids:
            if metric_id == "cuted_edist" and external_cuted_options is not None:
                candidate = evaluate_external_cuted_tree_pair_result(
                    case["true_tree"],
                    case["reconstructed_tree"],
                    **external_cuted_options,
                )
            elif (
                metric_id == "eps_approx_external"
                and external_eps_options is not None
            ):
                candidate = evaluate_external_eps_approx_tree_pair_result(
                    case["true_tree"],
                    case["reconstructed_tree"],
                    **external_eps_options,
                )
            else:
                candidate = evaluate_repeated_label_tree_pair_result(
                    case["true_tree"],
                    case["reconstructed_tree"],
                    metric_id,
                )
            validate_repeated_label_result(candidate)
            candidate_status_counts[metric_id][candidate["status"]] += 1
            candidate_results[metric_id] = candidate
        records.append(
            {
                "case_id": case["case_id"],
                "feature": case["feature"],
                "expected_relation": case["expected_relation"],
                "observation_labels": case["observation_labels"],
                "native_ad_f1_grf": native,
                "candidate_results": candidate_results,
            }
        )

    return {
        "schema_version": DISCRIMINATIVE_SUITE_SCHEMA_VERSION,
        "suite_id": DISCRIMINATIVE_SUITE_ID,
        "case_count": len(records),
        "metric_contracts": contracts,
        "records": records,
        "summary": {
            "case_count": len(records),
            "native_status_counts": native_status_counts,
            "candidate_status_counts": candidate_status_counts,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the complete small JSON report.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=CANDIDATE_METRIC_IDS,
        dest="metrics",
        help="Candidate metric to include; repeat to select several (default: all).",
    )
    parser.add_argument(
        "--audit-eps-source",
        action="store_true",
        help="Add a read-only audit of the owner-local external EPS checkout.",
    )
    parser.add_argument(
        "--run-external-eps-approx",
        action="store_true",
        help="Evaluate the ten tiny cases with the external approximate EPS tool.",
    )
    parser.add_argument(
        "--eps-repository-root",
        type=Path,
        default=DEFAULT_EXTERNAL_EPS_ROOT,
        help="External EPS checkout used by its audit or explicit runner.",
    )
    parser.add_argument(
        "--eps-python",
        type=Path,
        help="Explicit interpreter for external approximate EPS execution.",
    )
    parser.add_argument(
        "--eps-timeout-seconds",
        type=float,
        default=EXTERNAL_EPS_DEFAULT_TIMEOUT_SECONDS,
        help="Per-pair external EPS timeout (default: 30 seconds).",
    )
    parser.add_argument(
        "--audit-uted-source",
        action="store_true",
        help="Add a read-only audit of the owner-local external UTED checkout.",
    )
    parser.add_argument(
        "--probe-uted-source",
        action="store_true",
        help="Run two tiny semantic probes against the audited external UTED source.",
    )
    parser.add_argument(
        "--uted-repository-root",
        type=Path,
        default=DEFAULT_EXTERNAL_UTED_ROOT,
        help="External UTED checkout used only with its audit or probe option.",
    )
    parser.add_argument(
        "--audit-edist-source",
        action="store_true",
        help="Add a read-only audit of the owner-local edist CUTED checkout.",
    )
    parser.add_argument(
        "--probe-edist-source",
        action="store_true",
        help="Run fixed CUTED probes when the audited Cython build is present.",
    )
    parser.add_argument(
        "--run-external-cuted",
        action="store_true",
        help="Evaluate the ten tiny cases through the external edist runner.",
    )
    parser.add_argument(
        "--edist-repository-root",
        type=Path,
        default=DEFAULT_EXTERNAL_EDIST_ROOT,
        help="External edist checkout used only with its audit or probe option.",
    )
    parser.add_argument(
        "--edist-python",
        type=Path,
        help="Explicit edist interpreter for probe or external CUTED execution.",
    )
    parser.add_argument(
        "--edist-timeout-seconds",
        type=float,
        default=EXTERNAL_CUTED_DEFAULT_TIMEOUT_SECONDS,
        help="Per-pair external CUTED timeout (default: 30 seconds).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.run_external_cuted and args.edist_python is None:
        raise SystemExit("--run-external-cuted requires --edist-python")
    if args.run_external_eps_approx and args.eps_python is None:
        raise SystemExit("--run-external-eps-approx requires --eps-python")
    eps_source_audit = None
    if args.audit_eps_source or args.run_external_eps_approx:
        eps_source_audit = inspect_external_eps_source(args.eps_repository_root)
    edist_source_audit = None
    if (
        args.audit_edist_source
        or args.probe_edist_source
        or args.run_external_cuted
    ):
        edist_source_audit = inspect_external_edist_source(
            args.edist_repository_root
        )
    external_cuted_options = None
    if args.run_external_cuted:
        external_cuted_options = {
            "repository_root": args.edist_repository_root,
            "python_executable": args.edist_python,
            "timeout_seconds": args.edist_timeout_seconds,
            "source_audit": edist_source_audit,
        }
    external_eps_options = None
    if args.run_external_eps_approx:
        external_eps_options = {
            "repository_root": args.eps_repository_root,
            "python_executable": args.eps_python,
            "timeout_seconds": args.eps_timeout_seconds,
            "source_audit": eps_source_audit,
        }
    report = build_suite_report(
        args.metrics or CANDIDATE_METRIC_IDS,
        external_cuted_options=external_cuted_options,
        external_eps_options=external_eps_options,
    )
    if eps_source_audit is not None:
        report["external_eps_source_audit"] = eps_source_audit
        report["summary"]["external_eps_source_status"] = (
            eps_source_audit["status"]
        )
    if args.audit_uted_source or args.probe_uted_source:
        source_audit = inspect_external_uted_source(args.uted_repository_root)
        report["external_uted_source_audit"] = source_audit
        report["summary"]["external_uted_source_status"] = source_audit["status"]
    if args.probe_uted_source:
        semantic_probe = probe_external_uted_semantics(args.uted_repository_root)
        report["external_uted_semantic_probe"] = semantic_probe
        report["summary"]["external_uted_semantic_probe_status"] = (
            semantic_probe["status"]
        )
    if edist_source_audit is not None:
        report["external_edist_source_audit"] = edist_source_audit
        report["summary"]["external_edist_source_status"] = (
            edist_source_audit["status"]
        )
    if args.probe_edist_source:
        probe_kwargs = {
            "additional_probes": [
                build_external_edist_unit_probe(
                    f"frozen_suite:{case['case_id']}",
                    case["true_tree"],
                    case["reconstructed_tree"],
                )
                for case in build_discriminative_cases()
            ]
        }
        if args.edist_python is not None:
            probe_kwargs["python_executable"] = args.edist_python
        semantic_probe = probe_external_edist_semantics(
            args.edist_repository_root,
            **probe_kwargs,
        )
        report["external_edist_semantic_probe"] = semantic_probe
        report["summary"]["external_edist_semantic_probe_status"] = (
            semantic_probe["status"]
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report["summary"], indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
