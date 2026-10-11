"""Reevaluate stored CTBF/MEDICC2/NestedBD trees against state-trimmed truth.

This command has no simulation, reconstruction, cnp2cnp, Java or posterior
summary execution path. It writes only a separate, resumable audit directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
from pathlib import Path
import re
import sys

from algorithm_evaluation.paper_pipeline_contract import file_sha256, read_json, write_json_atomic
from algorithm_evaluation.paper_pipeline_runner import (
    StageTimeoutError, deserialize_tree, measured_stage, serialize_tree, validate_reconstruction_input,
)
from algorithm_evaluation.process_isolation import FreshSpawnTaskPool, fresh_process_contract
from algorithm_evaluation.v5_algorithm_development_common import observed_labels, truth_prefix
from algorithm_evaluation.v5_confirmation_bank import TRUTH_SCHEMA_VERSION
from algorithm_evaluation.v5_trimmed_truth_sources import (
    CONFIRMATION, DEFAULT_NESTED, PROJECT, collect_sources, object_digest, read_ref,
)
from algorithm_evaluation.v5_trimmed_truth_report import build_report, write_report
from evaluation_contract import validate_evaluation_result, validate_rooted_labeled_tree
from trimmed_truth_evaluation import (
    evaluate_prepared_trimmed_truth, prepare_trimmed_truth, truth_target_descriptor,
)


SCHEMA = "ctbf-v5-trimmed-truth-run-v1"
DEFAULT_OUTPUT = CONFIRMATION.with_name("trimmed_truth_v1")
SOURCE_FILES = (
    "trimmed_truth_evaluation.py", "evaluation_contract.py", "evaluator.py", "evaluator_full.py",
    "algorithm_evaluation/v5_trimmed_truth_sources.py", "algorithm_evaluation/v5_trimmed_truth_report.py",
    "algorithm_evaluation/v5_trimmed_truth_run.py", "algorithm_evaluation/process_isolation.py",
    "algorithm_evaluation/paper_pipeline_runner.py", "algorithm_evaluation/paper_pipeline_contract.py",
    "algorithm_evaluation/v5_algorithm_development_common.py", "algorithm_evaluation/v5_confirmation_report.py",
    "algorithm_evaluation/v5_confirmation_common.py",
)


def _failure(error: Exception, stage: str) -> dict:
    return {"stage": stage, "code": getattr(error, "code", type(error).__name__),
            "message": str(error)[:4096]}


def _read_saved(case: dict, reference: dict) -> tuple[dict | None, dict | None]:
    """Validate record identity before using its native result and stored tree."""
    if reference["source_kind"] == "nested":
        if reference["status"] != "success":
            return None, None
        row = read_ref(reference["record_ref"])
        expected = {key: case[key] for key in ("case_id", "block_index", "height")}
        expected.update(method="NestedBD-Long", replicate=reference["replicate"], seed=reference["seed"])
        if any(row.get(key) != value for key, value in expected.items()) or row["status"] != reference["status"]:
            raise ValueError("Sealed NestedBD record identity differs.")
        native = read_ref(reference["evaluation_ref"])
        graph = read_ref(reference["tree_ref"])
        if row["grf"] != reference["grf"] or row["grf"] != native["metrics"]["grf"]:
            raise ValueError("NestedBD stored full score differs between sealed artifacts.")
    else:
        row = read_ref(reference["record_ref"])
        schema = ("ctbf-v5-confirmatory-arm-record-v1" if reference["source_kind"] == "ctbf"
                  else "ctbf-v5-confirmatory-medicc2-record-v1")
        if row.get("schema_version") != schema:
            raise ValueError("Unexpected frozen reconstruction record schema.")
        expected = {key: case[key] for key in ("case_id", "block_index", "condition_id", "height")}
        expected.update(arm_id=reference["arm_id"], status=reference["status"])
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("Frozen record identity differs from its declaration.")
        if row["status"] != "success":
            return row, None
        native, graph = row["native_evaluation"], row["tree"]
        if row["metrics"]["grf"] != native["metrics"]["grf"]:
            raise ValueError("Source declared GRF differs from its native full-truth result.")
    validate_evaluation_result(native)
    if native["status"] != "success":
        raise ValueError("Successful source record has a failed native evaluation.")
    return row, {"evaluation": native, "tree": graph}


def evaluate_case(task: dict, timeout_seconds: int, rss_limit_bytes: int) -> dict:
    case = task["case"]
    saved = []
    for reference in task["records"]:
        raw, evaluation = _read_saved(case, reference)
        row = {**{key: case[key] for key in ("case_id", "block_index", "condition_id", "height")},
               **{key: reference[key] for key in ("method", "problem", "replicate", "seed")},
               "source_status": reference["status"], "source": reference,
               "status": "source_failure" if evaluation is None else "pending",
               "failure": (raw or reference).get("failure"), "grf_full": None, "grf_trim": None, "ad_f1": None}
        if evaluation is not None:
            row["grf_full"] = evaluation["evaluation"]["metrics"]["grf"]
            row["ad_f1"] = evaluation["evaluation"]["metrics"]["ad_f1"]
        saved.append((row, evaluation))
    result = {"schema_version": SCHEMA, "case_id": case["case_id"], "task_digest": object_digest(task),
              "trimmed_tree": None, "projection": None, "records": [row for row, _ in saved]}
    if not any(evaluation is not None for _, evaluation in saved):
        return result
    if case["status"] != "available":
        raise ValueError("Successful reconstruction claims an unavailable bank condition.")
    # Source integrity failures abort, instead of becoming apparent algorithm failures.
    payload, truth_payload = read_ref(task["input_ref"]), read_ref(task["truth_ref"])
    if payload.get("case_id") != case["case_id"]:
        raise ValueError("Observable input case identity differs.")
    validate_reconstruction_input(payload)
    if truth_payload.get("schema_version") != TRUTH_SCHEMA_VERSION or truth_payload["block_index"] != case["block_index"]:
        raise ValueError("True-tree block identity differs.")
    validate_rooted_labeled_tree(truth_payload["tree"], "true_tree")

    def audit():
        truth = truth_prefix(deserialize_tree(truth_payload["tree"]), case["height"])
        prepared = prepare_trimmed_truth(truth, observed_labels(payload))
        result["trimmed_tree"] = serialize_tree(prepared.trimmed.context.graph)
        result["projection"] = {"original_truth": prepared.original.metadata,
                                "trimmed_truth": prepared.trimmed.metadata,
                                "observation_labels": list(prepared.observation_labels),
                                "ad_relation_set_invariant": True,
                                "true_ad_pair_count": prepared.true_ad_pair_count}
        for row, source in saved:
            if source is None:
                continue
            try:
                evaluation = evaluate_prepared_trimmed_truth(prepared, source["tree"], source["evaluation"])
                row.update(status="success", failure=None, grf_trim=evaluation["grf_trim"], audit=evaluation)
            except (StageTimeoutError, MemoryError):
                # The deadline/resource exception ends the whole condition.
                # Do not swallow its one-shot alarm and continue with other methods.
                raise
            except Exception as error:
                row.update(status="audit_failure", failure=_failure(error, "trimmed_evaluation"))
        return True

    _value, resources, error = measured_stage(audit, timeout_seconds=timeout_seconds, rss_limit_bytes=rss_limit_bytes)
    result["audit_resources"] = resources
    if error is not None:
        for row, source in saved:
            if source is not None:
                row.update(status="audit_failure", grf_trim=None, failure=_failure(error, "case_evaluation"))
                row.pop("audit", None)
    return result


def verify_hashes(inventory: dict) -> None:
    for path, digest in inventory.items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Protected source changed: {path}")


def guard_output(output: Path, roots: list[str]) -> None:
    for source in map(Path, roots):
        source = source.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Audit output overlaps a protected source root.")
    if output == PROJECT or PROJECT.is_relative_to(output):
        raise ValueError("Audit output cannot be the project root or its ancestor.")


@contextmanager
def owner_lock(output: Path):
    with (output / ".runner.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("A trimmed-truth controller is already running here.") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _case_path(output: Path, case_id: str) -> Path:
    if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", case_id):
        raise ValueError("Case id is not a safe audit filename.")
    return output / "cases" / (case_id + ".json")


def _load_case(path: Path, task: dict) -> dict:
    envelope = read_json(path)
    result = envelope["case_result"]
    if object_digest(result) != envelope["sha256"] or result.get("task_digest") != object_digest(task):
        raise ValueError(f"Cached case digest/selection differs: {path}")
    expected = [(row["method"], row["replicate"], row["seed"]) for row in task["records"]]
    actual = [(row["method"], row["replicate"], row["seed"]) for row in result["records"]]
    if result.get("schema_version") != SCHEMA or result["case_id"] != task["case"]["case_id"] or actual != expected:
        raise ValueError("Cached case record identity differs.")
    if any(row["status"] not in {"success", "source_failure", "audit_failure"} for row in result["records"]):
        raise ValueError("Cached case is not terminal.")
    return result


def run_audit(sources: dict, output: Path, *, workers: int = 4, timeout_seconds: int = 300,
              rss_limit_bytes: int = 4 * 1024**3, bootstrap_draws: int = 100000,
              resume: bool = False, progress: bool = False, report_only: bool = False) -> dict:
    if not 1 <= workers <= 8 or timeout_seconds <= 0 or rss_limit_bytes <= 0 or bootstrap_draws <= 0:
        raise ValueError("Invalid worker/resource/bootstrap settings.")
    output = output.expanduser().resolve()
    guard_output(output, sources["protected_roots"])
    code_hashes = {str(PROJECT / path): file_sha256(PROJECT / path) for path in SOURCE_FILES}
    contract = {"schema_version": SCHEMA, "truth_target": truth_target_descriptor(), "sources": sources,
                "code_sha256": code_hashes, "timeout_seconds_per_case": timeout_seconds,
                "rss_limit_bytes_per_case": rss_limit_bytes, "bootstrap_draws": bootstrap_draws,
                "execution": fresh_process_contract("case_trimmed_truth_evaluation"),
                "runtime": {"python": sys.version, "packages": {
                    name: importlib.metadata.version(name) for name in ("numpy", "networkx", "psutil")}},
                "full_grf_recomputed": False, "reconstruction_permitted": False}
    if output.exists() and not resume:
        raise FileExistsError("Output exists. Use --resume with the same selection.")
    if report_only and not output.exists():
        raise ValueError("Report-only mode requires an existing audit.")
    output.mkdir(parents=True, exist_ok=True)
    with owner_lock(output):
        contract_path = output / "analysis_contract.json"
        if contract_path.exists():
            if read_json(contract_path) != contract:
                raise ValueError("Audit sources, semantics, code, runtime or settings changed.")
        else:
            if any(path.name != ".runner.lock" for path in output.iterdir()):
                raise ValueError("Nonempty output has no audit contract.")
            write_json_atomic(contract_path, contract)
        if progress:
            print(f"verifying {len(sources['protected_sha256'])} consumed source files (read-only)", flush=True)
        verify_hashes(sources["protected_sha256"])
        marker_path = output / "_COMPLETE.json"
        if marker_path.exists():
            marker = read_json(marker_path)
            if marker["contract_sha256"] != file_sha256(contract_path):
                raise ValueError("Completion contract digest differs.")
            verify_hashes({str(output / path): digest for path, digest in marker["artifacts"].items()})
            return read_json(output / "summary.json")
        results, pending = {}, []
        for task in sources["tasks"]:
            case_id = task["case"]["case_id"]
            path = _case_path(output, case_id)
            if path.exists():
                results[case_id] = _load_case(path, task)
            else:
                pending.append(task)
        if report_only and pending:
            raise ValueError(f"{len(pending)} cases still need evaluation; report-only does not schedule work.")
        if progress:
            print(f"trimmed audit: {len(sources['tasks'])} conditions; {sources['expected_record_count']} records; "
                  f"{len(results)} cached; {len(pending)} pending; {workers} fresh workers", flush=True)
        if pending:
            with FreshSpawnTaskPool(workers, buffer_factor=2) as pool:
                arguments = ((task, timeout_seconds, rss_limit_bytes) for task in pending)
                for task, result in zip(pending, pool.map_ordered(evaluate_case, arguments, timeout_seconds=timeout_seconds + 90)):
                    case_id = task["case"]["case_id"]
                    path = _case_path(output, case_id)
                    if path.exists():
                        raise ValueError("Refusing to replace an existing case result.")
                    write_json_atomic(path, {"sha256": object_digest(result), "case_result": result})
                    results[case_id] = _load_case(path, task)
                    if progress:
                        failures = sum(row["status"] != "success" for row in result["records"])
                        print(f"{len(results)}/{len(sources['tasks'])}: {case_id}; {len(result['records'])} records; {failures} failures", flush=True)
        verify_hashes(sources["protected_sha256"])
        verify_hashes(code_hashes)
        records = [row for task in sources["tasks"] for row in results[task["case"]["case_id"]]["records"]]
        if len(records) != sources["expected_record_count"]:
            raise ValueError("Not all declared source outcomes were retained.")
        if progress:
            print(f"reporting: {len(records)} records; paired history bootstrap with {bootstrap_draws} draws", flush=True)
        report = build_report(records, bootstrap_draws=bootstrap_draws)
        report.update(selected_subset=sources["selected_subset"], expected_record_count=sources["expected_record_count"],
                      consumed_sources_unchanged=True, truth_target=truth_target_descriptor(),
                      completed_condition_count=len(results))
        write_report(output, records, report)
        artifacts = {str(_case_path(output, case_id).relative_to(output)): file_sha256(_case_path(output, case_id))
                     for case_id in results}
        for name in ("summary.json", "scores.csv", "case_scores.csv", "report.md"):
            artifacts[name] = file_sha256(output / name)
        write_json_atomic(marker_path, {"schema_version": SCHEMA, "contract_sha256": file_sha256(contract_path),
                                       "artifacts": artifacts})
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--bank-root", type=Path, default=CONFIRMATION / "bank")
    parser.add_argument("--internal-root", type=Path, default=CONFIRMATION / "internal")
    parser.add_argument("--medicc2-root", type=Path, default=CONFIRMATION / "medicc2")
    parser.add_argument("--nestedbd-root", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--skip-medicc2", action="store_true")
    parser.add_argument("--skip-nestedbd", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-id", action="append", help="Explicit small verification subset; repeat for multiple cases.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluation-timeout-seconds", type=int, default=300, help="Budget for all reevaluations of one condition.")
    parser.add_argument("--rss-limit-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--bootstrap-draws", type=int, default=100000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.progress:
        print("loading frozen inventories; evaluation only, no reconstruction or inference", flush=True)
    sources = collect_sources(args.bank_root, args.internal_root,
                              None if args.skip_medicc2 else args.medicc2_root,
                              None if args.skip_nestedbd else args.nestedbd_root, args.case_id)
    report = run_audit(sources, args.output_root, workers=args.workers,
                       timeout_seconds=args.evaluation_timeout_seconds, rss_limit_bytes=args.rss_limit_bytes,
                       bootstrap_draws=args.bootstrap_draws, resume=args.resume or args.command == "report",
                       progress=args.progress, report_only=args.command == "report")
    statuses = report["status_counts"]
    print(f"complete: {report['raw_record_count']} trimmed-truth records; "
          f"{statuses.get('audit_failure', 0)} audit failures; {statuses.get('source_failure', 0)} source failures; "
          "full scores preserved; no reconstruction")
    print(f"report: {args.output_root.resolve() / 'report.md'}")
    return 0 if not statuses.get("audit_failure", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
