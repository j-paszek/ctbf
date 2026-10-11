"""Reporting-only recovery of the source-pinned ten-history NestedBD benchmark.

No inference, reconstruction, distance computation or checkpoint restoration is
available here. The original runner and its outputs remain read-only. Only
GRF/ext-GRF floating-point replay roundoff is tolerated, with original CTBF
scores retained and all other evaluator fields required to match exactly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import fcntl
import importlib.metadata
import math
from pathlib import Path
import sys

from algorithm_evaluation import v5_nestedbd_long_benchmark as benchmark


pilot = benchmark.pilot
topology = benchmark.topology
SCHEMA = "ctbf-v5-nestedbd-long-benchmark-report-recovery-v1"
DEFAULT_OUTPUT = benchmark.DEFAULT_OUTPUT.with_name("benchmark_v1_report_v2")
REPLAY_ABSOLUTE_TOLERANCE = 1e-14
POLICY = {
    "inference_permitted": False,
    "source_results_read_only": True,
    "replay_absolute_tolerance": REPLAY_ABSOLUTE_TOLERANCE,
    "replay_relative_tolerance": 0.0,
    "tolerant_fields": ["metrics.grf", "metrics.ext_grf"],
    "all_other_evaluator_fields_exact": True,
    "original_ctbf_scores_retained": True,
    "existing_sealed_summaries_reused": True,
    "missing_or_interrupted_inference": "refuse_without_starting_inference",
    "terminal_failures": "retain_without_retry",
}


def runtime_identity() -> dict:
    return {"python": sys.version, "packages": {
        name: importlib.metadata.version(name) for name in ("numpy", "networkx", "psutil")}}


@contextmanager
def source_read_lock(source: Path):
    """Exclude a concurrent original controller without writing its lock file."""
    with (source / ".runner.lock").open("r") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("the original benchmark controller is still running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sealed_files(directory: Path, *, verify: bool = True) -> dict:
    seal_path = directory / "_SEALED.json"
    seal = pilot.read_json(seal_path)
    files = {str(pilot.safe_child(directory, name)): digest for name, digest in seal.items()}
    if verify:
        pilot.verify_hashes(files)
    files[str(seal_path)] = pilot.sha256(seal_path)
    return files


def cached_source_summary(source: Path, case: dict, seed: dict, protected: dict) -> dict | None:
    unit = source / "summaries" / case["case_id"] / f"seed_{seed['seed']}"
    seals = sorted(unit.glob("attempt_*/_SEALED.json"))
    if len(seals) > 1:
        raise ValueError("multiple sealed source summaries for one seed")
    if not seals:
        return None
    directory = seals[0].parent
    protected.update(sealed_files(directory))
    result = pilot.read_json(directory / "result.json")
    expected = benchmark.base_record(case, "NestedBD-Long", replicate=seed["replicate"], seed=seed["seed"])
    if any(result.get(key) != expected[key] for key in
           ("case_id", "block_index", "height", "method", "replicate", "seed")):
        raise ValueError("source summary identity differs")
    if result.get("attempt") != str(directory):
        raise ValueError("source summary attempt path differs")
    return result


def saved_outcome(backend, source: Path, case: dict, seed: int, protected: dict) -> dict:
    """Read terminal attempts only. Missing work is never scheduled implicitly."""
    if case["eligibility"] != "compatible":
        return {"case_id": case["case_id"], "seed": seed, "status": "ineligible",
                "reason": case["eligibility"]}
    for phase in ("technical", "pilot"):
        unit = source / "inference" / case["case_id"] / f"seed_{seed}" / phase
        directory, record = benchmark.choose_inference_attempt(backend, unit, case, phase, seed)
        if record is None:
            raise ValueError(f"incomplete {phase} attempt for {case['case_id']} seed {seed}; "
                             "report-only recovery never starts inference")
        if record.get("attempt") != str(directory):
            raise ValueError("saved inference attempt path differs")
        for seal in sorted(unit.glob("attempt_*/_SEALED.json")):
            # choose_inference_attempt has just verified all these seals.
            protected.update(sealed_files(seal.parent, verify=False))
        if record["status"] != "technical_complete":
            return {"case_id": case["case_id"], "seed": seed, "status": "failure",
                    "stage": phase, "reason": record["status"], "attempt": str(directory)}
    return {"case_id": case["case_id"], "seed": seed, "status": "success", "run": record}


def load_source(source: Path, bio: Path):
    contract_path = source / "analysis_contract.json"
    contract = pilot.read_json(contract_path)
    if (contract.get("schema_version") != benchmark.SCHEMA
            or contract.get("rules") != benchmark.RULES
            or contract.get("bio_root") != str(bio)
            or contract.get("runtime") != runtime_identity()):
        raise ValueError("original benchmark contract or runtime differs")
    protected = dict(contract["protected_sha256"])
    pilot.verify_hashes(protected)
    protected[str(contract_path)] = pilot.sha256(contract_path)
    cases, selected_files = benchmark.load_selection(Path(contract["bank_root"]), Path(contract["internal_root"]))
    if ([{key: value for key, value in case.items() if key != "records"} for case in cases]
            != contract["cases"]):
        raise ValueError("original prespecified input selection differs")
    if any(protected.get(path) != digest for path, digest in selected_files.items()):
        raise ValueError("selected input identity differs from the original contract")
    backend = pilot.load_backend(bio)
    if any(protected.get(path) != digest for path, digest in pilot.backend_snapshot(backend).items()):
        raise ValueError("source-pinned external runtime changed")
    for directory in (source / "preparation", source / "inputs"):
        protected.update(sealed_files(directory))
    for case in cases:
        if case["source_case"] is not None:
            path = source / "inputs" / case["case_id"] / "input.json"
            if pilot.sha256(path) != case["input_sha256"]:
                raise ValueError("stored lossless input copy differs")
    numeric = sorted((source / "numeric_gate").glob("attempt_*/_SEALED.json"))
    if not numeric:
        raise ValueError("missing saved arithmetic gate; reporting will not execute it")
    for seal in numeric:
        protected.update(sealed_files(seal.parent))
    if not pilot.read_json(numeric[0].parent / "report.json")["passes_observed_range"]:
        raise ValueError("saved arithmetic gate did not pass")
    outcomes, summaries = [], {}
    for case in cases:
        for seed in case["seeds"]:
            outcomes.append(saved_outcome(backend, source, case, seed["seed"], protected))
            summary = cached_source_summary(source, case, seed, protected)
            if summary is not None:
                summaries[(case["case_id"], seed["seed"])] = summary
    return contract, cases, outcomes, summaries, protected, backend


def validate_replay(stored: dict, replay: dict) -> dict:
    """Allow bounded finite roundoff in two fields, never general JSON fuzzing."""
    benchmark.validate_evaluation_result(stored)
    benchmark.validate_evaluation_result(replay)
    if stored["status"] != "success" or replay["status"] != "success":
        raise ValueError("frozen CTBF evaluator replay was not successful")
    normalized = copy.deepcopy(replay)
    differences = {}
    for name in ("grf", "ext_grf"):
        left, right = stored["metrics"][name], replay["metrics"][name]
        if (not math.isfinite(left) or not math.isfinite(right)
                or not math.isclose(left, right, rel_tol=0.0, abs_tol=REPLAY_ABSOLUTE_TOLERANCE)):
            raise ValueError(f"frozen CTBF evaluator replay differs beyond roundoff: metrics.{name}")
        if left != right:
            differences[name] = {"stored": left, "replayed": right, "absolute_difference": abs(left - right)}
        normalized["metrics"][name] = left
    if normalized != stored:
        raise ValueError("frozen CTBF evaluator replay differs outside GRF roundoff fields")
    return {"bitwise_exact": not differences, "absolute_tolerance": REPLAY_ABSOLUTE_TOLERANCE,
            "relative_tolerance": 0.0, "all_other_fields_exact": True,
            "stored_scores_retained": True, "roundoff": differences}


def frozen_comparator(bank: Path, case: dict, method: str) -> dict:
    result = benchmark.base_record(case, method)
    path = Path(case["internal_records"][method])
    stored = pilot.read_json(path)
    if (stored["case_id"] != case["case_id"] or stored["arm_id"] != topology.CTBF_ARMS[method]
            or stored["block_index"] != case["block_index"] or stored["height"] != case["height"]
            or stored["schema_version"] != "ctbf-v5-confirmatory-arm-record-v1"):
        raise ValueError("frozen CTBF comparator identity differs")
    result["source_record"] = str(path)
    if stored["status"] != "success":
        result["failure"] = stored.get("failure") or {"code": "source_dependency_failure"}
        return result
    native = stored["native_evaluation"]
    tree = benchmark.deserialize_tree(stored["tree"])
    audit = validate_replay(native, benchmark.evaluation_for(bank, case, tree))
    if (native["metrics"]["grf"] != stored["metrics"]["grf"]
            or benchmark.tree_summary(tree) != stored["tree_summary"]
            or native["inputs"]["observation_label_coverage"]["fraction"] != 1.0):
        raise ValueError("frozen CTBF score, topology or observation coverage differs")
    result.update(status="success", grf=stored["metrics"]["grf"], tree_summary=stored["tree_summary"],
                  frozen_evaluator_replay_exact=audit["bitwise_exact"],
                  frozen_evaluator_replay_validated=True, replay_audit=audit)
    return result


def replay_summary(records: list[dict]) -> dict:
    audits = [row["replay_audit"] for row in records if "replay_audit" in row]
    return {"successful_comparators_checked": len(audits),
            "roundoff_only_comparators": sum(not row["bitwise_exact"] for row in audits),
            "maximum_absolute_difference": max((entry["absolute_difference"]
                for audit in audits for entry in audit["roundoff"].values()), default=0.0),
            "absolute_tolerance": REPLAY_ABSOLUTE_TOLERANCE,
            "original_scores_retained": True, "all_other_fields_exact": True}


def guard_output(output: Path, source: Path, contract: dict) -> None:
    pilot.output_guard(output, Path(contract["bank_root"]), Path(contract["bio_root"]))
    for root in (source, Path(contract["internal_root"]), Path(contract["previous_pilot"]),
                 benchmark.calibration.DEFAULT_OUTPUT.resolve(), pilot.DEFAULT_OUTPUT.resolve()):
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("recovery output overlaps protected evidence")


def prepare_output(output: Path, source: Path, original: dict, protected: dict, *, resume: bool) -> dict:
    guard_output(output, source, original)
    recovery_contract = {"schema_version": SCHEMA, "policy": POLICY, "source_root": str(source),
        "original_contract_sha256": pilot.sha256(source / "analysis_contract.json"),
        "rules": original["rules"], "runtime": runtime_identity(),
        "protected_sha256": {**protected, **topology.source_snapshot()}}
    if output.exists():
        if not resume:
            raise FileExistsError("recovery output exists; use --resume")
        if pilot.read_json(output / "analysis_contract.json") != recovery_contract:
            raise ValueError("recovery contract, source or runtime changed")
    else:
        output.mkdir(parents=True)
        pilot.write_json(output / "analysis_contract.json", recovery_contract)
    return recovery_contract


def finish_report(backend, bank: Path, source: Path, output: Path, cases: list[dict],
                  outcomes: list[dict], summaries: dict, contract: dict, comparators: list[dict]) -> dict:
    lookup = {(row["case_id"], row["seed"]): row for row in outcomes}
    records = list(comparators)
    reused = 0
    for case in cases:
        for seed in case["seeds"]:
            key = (case["case_id"], seed["seed"])
            if key in summaries:
                record = dict(summaries[key], reused_sealed_source_summary=True)
                reused += 1
            else:
                record = benchmark.summarize_external(backend, bank, output, case, seed["replicate"], lookup[key])
            records.append(record)
    artifacts = {}
    for record in records:
        if record.get("attempt") and not record.get("reused_sealed_source_summary"):
            artifacts.update(sealed_files(Path(record["attempt"])))
    report = {"schema_version": SCHEMA, "status": "fixed_budget_benchmark_complete_not_convergence",
        "scientific_role": benchmark.RULES["scientific_role"], "source_root": str(source),
        "records": records, "cases": [{key: value for key, value in case.items() if key != "records"} for case in cases],
        "inference_outcomes": outcomes, "independent_histories": len(benchmark.BLOCKS),
        "planned_external_runs": len(cases) * 2, "mcmc_runs_started": 0,
        "reused_sealed_source_summaries": reused, "replay_audit": replay_summary(comparators),
        "external_status_counts": dict(Counter(row["status"] for row in records if row["method"] == "NestedBD-Long")),
        "paired_effects": benchmark.paired_effects(records),
        "clade_agreement": benchmark.clade_agreement(backend, outcomes),
        "convergence_established": False, "protected_files_unchanged": True, "artifact_sha256": artifacts}
    pilot.verify_hashes(contract["protected_sha256"])
    parent = output / "reports"
    previous = sorted(parent.glob("report_*")) if parent.exists() else []
    number = max((int(path.name.removeprefix("report_")) for path in previous), default=0) + 1
    directory = parent / f"report_{number:03d}"
    directory.mkdir(parents=True)
    pilot.write_json(directory / "report.json", report)
    audit = report["replay_audit"]
    note = ("\n## Reporting-only recovery audit\n\n"
        "No MCMC, CTBF reconstruction or distance calculation was started. Original evidence is unchanged.\n\n"
        f"Reused {reused} sealed external summaries from the original output. "
        f"Validated {audit['successful_comparators_checked']} frozen CTBF comparator records. "
        f"{audit['roundoff_only_comparators']} had GRF/ext-GRF roundoff, with maximum absolute difference "
        f"{audit['maximum_absolute_difference']:.3g} (limit {REPLAY_ABSOLUTE_TOLERANCE:g}). "
        "Every other evaluator field and topology summary matched exactly. Original CTBF scores, "
        "not recalculated replacements, enter every table. Per-record differences are retained in JSON.\n")
    with (directory / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(benchmark.report_text(report) + note)
    with (directory / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
        fields = ["case_id", "block_index", "height", "method", "replicate", "seed", "status", "grf"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    topology._seal(directory)
    pilot.write_json(output / "_COMPLETE.json", {"report_directory": str(directory.relative_to(output)),
                     "sha256": pilot.sha256(directory / "report.json")})
    return report


def recover(source: Path, bio: Path, output: Path, *, resume: bool, check: bool = False) -> dict:
    source, bio, output = (path.expanduser().resolve() for path in (source, bio, output))
    with source_read_lock(source):
        original, cases, outcomes, summaries, protected, backend = load_source(source, bio)
        if check:
            rows = [frozen_comparator(Path(original["bank_root"]), case, method)
                    for case in cases for method in topology.CTBF_ARMS]
            pilot.verify_hashes(protected)
            return {"check_only": True, "saved_external_outcomes": len(outcomes),
                    "inference_status_counts": dict(Counter(row["status"] for row in outcomes)),
                    "existing_summaries": len(summaries), "replay_audit": replay_summary(rows),
                    "mcmc_runs_started": 0, "protected_files_unchanged": True}
        contract = prepare_output(output, source, original, protected, resume=resume)
        with benchmark.owner_lock(output):
            result = benchmark.verify_completed(output, contract)
            if result is not None:
                return result
            # Check every comparator before spending time on any missing MCC summary.
            rows = [frozen_comparator(Path(original["bank_root"]), case, method)
                    for case in cases for method in topology.CTBF_ARMS]
            return finish_report(backend, Path(original["bank_root"]), source, output, cases,
                                 outcomes, summaries, contract, rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--bio-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=benchmark.DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = recover(args.source_root, args.bio_root, args.output_root, resume=args.resume,
                     check=args.command == "check")
    audit = result["replay_audit"]
    if args.command == "check":
        print(f"complete: report-only recovery check; {result['saved_external_outcomes']} saved external outcomes; "
              f"{audit['successful_comparators_checked']} CTBF replays validated; "
              f"max GRF roundoff {audit['maximum_absolute_difference']:.3g}; no MCMC or native summarization")
        return 0
    print(f"complete: NestedBD-Long report recovered; {len(result['records'])} comparison rows; "
          f"external statuses {result['external_status_counts']}; original scores and results preserved; no MCMC")
    output = args.output_root.expanduser().resolve()
    marker = pilot.read_json(output / "_COMPLETE.json")
    print("report: " + str(pilot.safe_child(output, marker["report_directory"]) / "report.md"))
    return 0 if result["external_status_counts"] == {"success": result["planned_external_runs"]} else 2


if __name__ == "__main__":
    raise SystemExit(main())
