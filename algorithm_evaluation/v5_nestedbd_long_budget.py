"""Bounded 2M-versus-10M sampling diagnostic, not a new accuracy benchmark.

Reuse the first two histories' saved 2M diagnostics. The owner starts eight
uninterrupted 10M runs on exactly the same inputs/seeds. No truth, distance,
reconstruction score, checkpoint restoration or automatic promotion is used.
The source-pinned earlier runners and optional ctbfbio runtime are read-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
from pathlib import Path
import signal
import threading
import xml.etree.ElementTree as ET

from algorithm_evaluation import v5_nestedbd_long_benchmark as benchmark
from algorithm_evaluation import v5_nestedbd_long_benchmark_report as recovery
from algorithm_evaluation.v5_nestedbd_long_budget_report import make_report, report_text


pilot, calibration, topology = benchmark.pilot, benchmark.calibration, benchmark.topology
SCHEMA = "ctbf-v5-nestedbd-long-budget-diagnostic-v1"
DEFAULT_OUTPUT = benchmark.DEFAULT_OUTPUT.with_name("budget_diagnostic_v1")
BLOCKS = (0, 1)
CONDITIONS = benchmark.CONDITIONS
SETTINGS = {
    "technical_iterations": 10000, "long_iterations": 10000000,
    "technical_timeout_seconds": 300, "long_timeout_seconds": 18000,
    "log_every": 1000, "burnin_percent": 10, "state_write_every": 10000,
    "rss_limit_bytes_per_run": 4 * 1024**3,
    "capture_limit_bytes_per_run": 8 * 1024**2,
    "maximum_parallel_runs": 2, "coupled_chains_per_run": 4,
}
RULES = {
    "selection": "first_two_original_histories_H14_H24_before_diagnostic_inspection",
    "block_indices": list(BLOCKS), "conditions": list(CONDITIONS),
    "seeds": "both_original_declared_seeds_unchanged",
    "baseline_iterations": 2000000, "settings": SETTINGS,
    "clock": benchmark.CLOCK, "main_benchmark_iterations": 2000000,
    "changed_xml": "run.chainLength_only_relative_to_original_2M_configuration",
    "native_checkpoint_restoration": False,
    "execution_policy": pilot.POLICY,
    "terminal_failures": "retain_without_retry_or_replacement",
    "accuracy_scoring": False, "automatic_budget_selection": False,
    "convergence_certification": False,
    "secondary_burnin_windows": [25, 50],
    "same_seed_prefix_identity_assumed": False,
}


def selected_ids() -> list[str]:
    return [benchmark.confirmation.confirmation_case_id(block, condition)
            for block in BLOCKS for condition in CONDITIONS]


def _bound(path: Path, expected: str | None, protected: dict) -> None:
    if expected is None or pilot.sha256(path) != expected:
        raise ValueError("Changed or unbound baseline file: " + str(path))
    protected[str(path)] = expected


def load_baseline(root: Path, bio: Path):
    """Follow closed references, reading inputs/diagnostics but never accuracy.

    The recovered report is hashed, not deserialized. Its recovery contract
    binds the selected original attempt seals, inputs and implementation.
    Unselected inference traces, truths and matrices are never opened.
    """
    marker_path, contract_path = root / "_COMPLETE.json", root / "analysis_contract.json"
    marker, recovered = pilot.read_json(marker_path), pilot.read_json(contract_path)
    protected = {str(path): pilot.sha256(path) for path in (marker_path, contract_path)}
    report_path = pilot.safe_child(root, marker["report_directory"]) / "report.json"
    _bound(report_path, marker["sha256"], protected)
    source = Path(recovered["source_root"]).resolve()
    trusted = recovered["protected_sha256"]
    original_path = source / "analysis_contract.json"
    _bound(original_path, trusted.get(str(original_path)), protected)
    original = pilot.read_json(original_path)
    if (original.get("schema_version") != benchmark.SCHEMA or original.get("rules") != benchmark.RULES
            or original.get("bio_root") != str(bio)):
        raise ValueError("Original 2M benchmark configuration or runtime root differs.")
    backend = pilot.load_backend(bio)
    for path, digest in pilot.backend_snapshot(backend).items():
        if original["protected_sha256"].get(path) != digest:
            raise ValueError("Pinned external implementation changed: " + path)
        protected[path] = digest
    # Keep the exact old Python implementation, without hashing unused data.
    for path, digest in original["protected_sha256"].items():
        if Path(path).suffix == ".py":
            _bound(Path(path), digest, protected)
    declarations = {case["case_id"]: case for case in original["cases"]}
    if len(declarations) != len(original["cases"]):
        raise ValueError("Duplicate original case declaration.")
    cases, old_runs = [], []
    with recovery.source_read_lock(source):
        for case_id in selected_ids():
            raw = declarations[case_id]
            block = int(case_id.split("-")[1][1:]) - 1
            height = 14 if case_id.endswith(CONDITIONS[0]) else 24
            generations = [9, 12, 14] if height == 14 else [15, 20, 24]
            expected_seeds = [{"replicate": rep, "seed": benchmark.seed_for(case_id, rep)} for rep in (1, 2)]
            if (raw["block_index"] != block or raw["height"] != height
                    or raw["condition_id"] != case_id.split("-", 2)[2]
                    or raw["generations"] != generations or raw["seeds"] != expected_seeds):
                raise ValueError("Original selection, biopsy times or seeds differ.")
            if raw["source_case"] is None:
                raise ValueError("Prespecified diagnostic input was not available; do not replace it.")
            path = Path(raw["input_path"])
            if original["protected_sha256"].get(str(path)) != raw["input_sha256"]:
                raise ValueError("Original input declaration is not checksum-bound.")
            _bound(path, trusted.get(str(path)), protected)
            payload = pilot.read_json(path)
            if (payload.get("case_id") != case_id or payload.get("condition_id") != raw["condition_id"]
                    or [row["generation"] for row in payload["levels"]] != generations):
                raise ValueError("Observable input identity differs.")
            records = pilot.input_records(payload)
            summary = pilot.compatibility(records)
            if summary != raw["input_summary"] or summary["status"] != raw["eligibility"]:
                raise ValueError("Original compatibility summary differs.")
            case = {key: raw[key] for key in ("case_id", "block_index", "height", "condition_id", "generations",
                                              "seeds", "input_path", "input_sha256", "eligibility", "input_summary")}
            case["records"] = records
            cases.append(case)
            for seed in case["seeds"]:
                unit = source / "inference" / case_id / f"seed_{seed['seed']}" / "pilot"
                directory, run = benchmark.choose_inference_attempt(backend, unit, case, "pilot", seed["seed"])
                if run is None:
                    raise ValueError("Original 2M run is unfinished; never start or replace it here.")
                seal_path = directory / "_SEALED.json"
                _bound(seal_path, trusted.get(str(seal_path)), protected)
                for relative, digest in pilot.read_json(seal_path).items():
                    path = pilot.safe_child(directory, relative)
                    _bound(path, digest, protected)
                if run.get("attempt") != str(directory):
                    raise ValueError("Original attempt path differs.")
                old_runs.append({**run, "block_index": block, "height": height, "replicate": seed["replicate"]})
    return cases, old_runs, protected, backend, source, Path(original["bank_root"])


def phase_config(case: dict, phase: str) -> dict:
    if phase not in ("technical", "long"):
        raise ValueError("Unknown diagnostic phase.")
    return {"forward_cycle_positions": case["generations"],
            "chain_length": SETTINGS[phase + "_iterations"],
            "timeout_seconds_per_run": SETTINGS[phase + "_timeout_seconds"],
            **{key: SETTINGS[key] for key in ("log_every", "burnin_percent", "rss_limit_bytes_per_run", "capture_limit_bytes_per_run")}}


def make_xml(backend, case: dict, phase: str):
    """Clone the original calibrated 2M XML; change exactly its chainLength."""
    _released, original, inherited = calibration.make_xml(backend, case, "pilot", benchmark.CLOCK)
    run = original.getroot().find("./run")
    if run is None or run.get("chainLength") != str(RULES["baseline_iterations"]):
        raise ValueError("Original XML does not declare the fixed 2M length.")
    tree = copy.deepcopy(original)
    tree.getroot().find("./run").set("chainLength", str(phase_config(case, phase)["chain_length"]))
    restored = copy.deepcopy(tree)
    restored.getroot().find("./run").set("chainLength", run.get("chainLength"))
    if ET.tostring(restored.getroot()) != ET.tostring(original.getroot()):
        raise ValueError("A field other than chainLength changed.")
    pilot.verify_xml_input(tree, case)
    return original, tree, {"inherited_configuration": inherited,
            "only_additional_xml_change": "run.chainLength", "before": RULES["baseline_iterations"],
            "after": phase_config(case, phase)["chain_length"], "all_other_xml_unchanged": True,
            "original_xml_sha256": hashlib.sha256(ET.tostring(original.getroot())).hexdigest()}


def run_identity(case: dict, phase: str, seed: int) -> dict:
    return {"schema_version": SCHEMA, "case_id": case["case_id"], "block_index": case["block_index"],
            "height": case["height"], "phase": phase, "seed": seed,
            "replicate": next(row["replicate"] for row in case["seeds"] if row["seed"] == seed),
            "input_sha256": case["input_sha256"], "config": phase_config(case, phase),
            "clock": benchmark.CLOCK, "native_checkpoint_restoration": False,
            "execution_policy": pilot.POLICY}


def choose_attempt(backend, unit: Path, case: dict, phase: str, seed: int):
    attempts = sorted(unit.glob("attempt_*")) if unit.exists() else []
    terminal = []
    expected = run_identity(case, phase, seed)
    for directory in attempts:
        started = directory / "execution_started.json"
        if started.is_file() and backend.pilot.process_alive(pilot.read_json(started)):
            raise ValueError("An earlier diagnostic JVM is still running: " + str(directory))
        if not (directory / "_SEALED.json").is_file():
            continue  # preserve interruption, never restore a native state
        topology._verify_seal(directory)
        record = pilot.read_json(directory / "report.json")
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Cached diagnostic identity or configuration differs.")
        if record["status"] == "technical_complete":
            actual = ET.parse(directory / "input.xml")
            _, wanted, _ = make_xml(backend, case, phase)
            if ET.tostring(actual.getroot()) != ET.tostring(wanted.getroot()):
                raise ValueError("Cached diagnostic XML differs.")
        if record["status"] != "interrupted":
            terminal.append((directory, record))
    if len(terminal) > 1:
        raise ValueError("Multiple terminal results for one diagnostic seed.")
    if terminal:
        return terminal[0]
    number = max((int(path.name.removeprefix("attempt_")) for path in attempts), default=0) + 1
    return unit / f"attempt_{number:03d}", None


def execute_attempt(backend, directory: Path, case: dict, phase: str, seed: int, stop: threading.Event) -> dict:
    directory.mkdir(parents=True)
    config = phase_config(case, phase)
    record = {**run_identity(case, phase, seed), "attempt": str(directory), "status": "failed",
              "convergence_established": False}
    try:
        original, tree, proof = make_xml(backend, case, phase)
        original.write(directory / "original_2m.xml", encoding="utf-8", xml_declaration=True)
        tree.write(directory / "input.xml", encoding="utf-8", xml_declaration=True)
        pilot.write_json(directory / "xml_changes.json", proof)
        command = backend.native_command(directory, seed)
        record["command"] = command
        print(f"start: {case['case_id']} {phase} seed {seed}; {config['chain_length']:,} iterations", flush=True)
        execution = backend.pilot.execute(command, directory, config, stop)
        record.update(execution=execution, status=execution["status"])
        if execution["status"] == "completed":
            taxa = set(pilot.sequences(case))
            record["finite_state_checks"] = backend.verify_run(directory, config, taxa)
            if phase == "long":
                analysis = backend.pilot.analyse(directory, config, taxa)
                review = backend.audit.review
                rows = backend.pilot.trace_rows(directory / "shared_pilot.log", config)
                record["scalar_windows"] = review.scalar_windows(
                    directory, rows, analysis["cold_chain_statistics"], directory, seed)
                windows = review.tree_windows(directory / "shared_pilot.trees", sorted(taxa), config)
                pilot.write_json(directory / "clade_windows.json", {
                    str(burn): {"samples": window["samples"], "counts": {str(k): v for k, v in window["counts"].items()}}
                    for burn, window in windows.items()})
            record["status"] = "technical_complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    pilot.write_json(directory / "report.json", record)
    topology._seal(directory)
    print(f"finished: {case['case_id']} {phase} seed {seed}: {record['status']}", flush=True)
    return record


def inference_task(backend, output: Path, case: dict, seed: int, stop: threading.Event):
    if stop.is_set():
        return None
    if case["eligibility"] != "compatible":
        return {**run_identity(case, "long", seed), "status": "ineligible", "error": case["eligibility"]}
    for phase in ("technical", "long"):
        directory, record = choose_attempt(backend,
            output / "inference" / case["case_id"] / f"seed_{seed}" / phase, case, phase, seed)
        if record is None:
            if stop.is_set():
                return None
            record = execute_attempt(backend, directory, case, phase, seed, stop)
        if record["status"] == "interrupted" or stop.is_set():
            return None
        if record["status"] != "technical_complete":
            return record
    return record


def prepare(root: Path, bio: Path, output: Path, *, resume: bool):
    root, bio, output = (path.expanduser().resolve() for path in (root, bio, output))
    cases, old, protected, backend, source, bank = load_baseline(root, bio)
    pilot.output_guard(output, bank, bio)
    for other in (root, source):
        if output.is_relative_to(other) or other.is_relative_to(output):
            raise ValueError("Output overlaps previous benchmark evidence.")
    protected.update({str(Path(__file__).resolve()): pilot.sha256(Path(__file__).resolve()),
                      str(Path(__file__).with_name("v5_nestedbd_long_budget_report.py")):
                      pilot.sha256(Path(__file__).with_name("v5_nestedbd_long_budget_report.py"))})
    contract = {"schema_version": SCHEMA, "rules": RULES, "source_root": str(root), "bio_root": str(bio),
                "cases": [{key: value for key, value in case.items() if key != "records"} for case in cases],
                "baseline_runs": old, "protected_sha256": protected,
                "runtime": recovery.runtime_identity(), "truth_or_accuracy_deserialized": False}
    if output.exists() and not resume:
        raise FileExistsError("Output exists; use --resume for the identical diagnostic.")
    output.mkdir(parents=True, exist_ok=True)
    with benchmark.owner_lock(output):
        path = output / "analysis_contract.json"
        if path.exists():
            if pilot.read_json(path) != contract:
                raise ValueError("Diagnostic sources, rules, code or runtime changed.")
            topology._verify_seal(output / "inputs")
            for case in cases:
                if pilot.sha256(output / "inputs" / case["case_id"] / "input.json") != case["input_sha256"]:
                    raise ValueError("Stored diagnostic input copy differs from the original.")
        else:
            if any(path.name != ".runner.lock" for path in output.iterdir()):
                raise ValueError("Nonempty output lacks its contract.")
            pilot.write_json(path, contract)
            for case in cases:
                pilot.write_input_copy(output, case)
            topology._seal(output / "inputs")
        pilot.verify_hashes(protected)
    return cases, contract, backend


def complete_report(output: Path, backend, cases: list[dict], old: list[dict], runs: list[dict], contract: dict):
    expected = {(case["case_id"], seed["seed"]) for case in cases for seed in case["seeds"]}
    if len(runs) != len(expected) or {(row["case_id"], row["seed"]) for row in runs} != expected:
        raise ValueError("Missing or duplicate diagnostic outcome.")
    report = make_report(backend, cases, old, runs)
    pilot.verify_hashes(contract["protected_sha256"])
    report.update(schema_version=SCHEMA, protected_files_unchanged=True, accuracy_scoring=False,
                  automatic_budget_selection=False, main_benchmark_iterations=2000000)
    parent = output / "reports"
    previous = sorted(parent.glob("report_*")) if parent.exists() else []
    number = max((int(path.name.removeprefix("report_")) for path in previous), default=0) + 1
    directory = parent / f"report_{number:03d}"
    directory.mkdir(parents=True)
    pilot.write_json(directory / "report.json", report)
    with (directory / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(report_text(report))
    topology._seal(directory)
    artifacts = {}
    # New, bounded attempt directories only, never a scan of old result roots.
    for unit in (output / "inference", output / "numeric_gate"):
        for seal in unit.glob("**/_SEALED.json"):
            topology._verify_seal(seal.parent)
            artifacts[str(seal)] = pilot.sha256(seal)
            artifacts.update({str(pilot.safe_child(seal.parent, name)): digest for name, digest in pilot.read_json(seal).items()})
    for name, digest in pilot.read_json(directory / "_SEALED.json").items():
        artifacts[str(pilot.safe_child(directory, name))] = digest
    artifacts[str(directory / "_SEALED.json")] = pilot.sha256(directory / "_SEALED.json")
    pilot.write_json(output / "_COMPLETE.json", {"schema_version": SCHEMA,
                     "contract_sha256": pilot.sha256(output / "analysis_contract.json"),
                     "report_directory": str(directory.relative_to(output)),
                     "report_sha256": pilot.sha256(directory / "report.json"), "artifacts": artifacts})
    return report


def run_prepared(backend, output: Path, cases: list[dict], contract: dict, *, parallel: int = 2,
                 check: bool = False, report_only: bool = False):
    if parallel not in (1, 2):
        raise ValueError("Only one or two concurrent four-chain JVMs are permitted.")
    with benchmark.owner_lock(output):
        marker_path = output / "_COMPLETE.json"
        if marker_path.exists():
            marker = pilot.read_json(marker_path)
            if marker["contract_sha256"] != pilot.sha256(output / "analysis_contract.json"):
                raise ValueError("Completion contract changed.")
            pilot.verify_hashes(marker["artifacts"])
            pilot.verify_hashes(contract["protected_sha256"])
            path = pilot.safe_child(output, marker["report_directory"]) / "report.json"
            if pilot.sha256(path) != marker["report_sha256"]:
                raise ValueError("Completed report changed.")
            if check:
                return {"bounded_check": True, "success": True, "already_complete": True}
            return pilot.read_json(path)
        if report_only:
            runs = []
            for case in cases:
                for seed in case["seeds"]:
                    if case["eligibility"] != "compatible":
                        runs.append({**run_identity(case, "long", seed["seed"]),
                                     "status": "ineligible", "error": case["eligibility"]})
                        continue
                    for phase in ("technical", "long"):
                        _, record = choose_attempt(backend, output / "inference" / case["case_id"] /
                            f"seed_{seed['seed']}" / phase, case, phase, seed["seed"])
                        if record is None:
                            raise ValueError("Unfinished diagnostic; report mode never starts inference.")
                        if record["status"] != "technical_complete":
                            break
                    runs.append(record)
            return complete_report(output, backend, cases, contract["baseline_runs"], runs, contract)
        numeric = benchmark.numeric_gate(backend, output)
        if not numeric["passes_observed_range"]:
            raise ValueError("Unchanged binary arithmetic gate failed; no inference.")
        stop = threading.Event()
        handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if check:
                case = cases[0]
                seed = case["seeds"][0]["seed"]
                directory, record = choose_attempt(backend, output / "inference" / case["case_id"] /
                    f"seed_{seed}" / "technical", case, "technical", seed)
                if record is None:
                    record = execute_attempt(backend, directory, case, "technical", seed, stop)
                pilot.verify_hashes(contract["protected_sha256"])
                return {"bounded_check": True, "success": record["status"] == "technical_complete", "record": record}
            runs = []
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [pool.submit(inference_task, backend, output, case, seed["seed"], stop)
                           for case in cases for seed in case["seeds"]]
                try:
                    for future in as_completed(futures):
                        record = future.result()
                        if record is not None:
                            runs.append(record)
                            print(f"progress: {len(runs)}/{len(futures)} diagnostic outcomes terminal", flush=True)
                except BaseException:
                    stop.set()
                    for future in futures:
                        future.cancel()
                    raise
            if stop.is_set():
                pilot.verify_hashes(contract["protected_sha256"])
                print("interrupted: sealed results retained; unfinished runs restart from zero with --resume")
                return None
            return complete_report(output, backend, cases, contract["baseline_runs"], runs, contract)
        finally:
            stop.set()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run", "report"))
    parser.add_argument("--bio-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=recovery.DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parallel-runs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_root.expanduser().resolve()
    if args.command == "report" and not (output / "analysis_contract.json").is_file():
        raise ValueError("Report mode requires an existing diagnostic.")
    print("verifying fixed inputs, saved 2M diagnostics and pinned runtime; no accuracy scoring", flush=True)
    cases, contract, backend = prepare(args.baseline_root, args.bio_root, output,
                                       resume=args.resume or args.command == "report")
    if args.command == "prepare":
        print(f"complete: budget diagnostic prepared; {len(cases)} inputs; {2 * len(cases)} planned 10M runs; no inference")
        return 0
    result = run_prepared(backend, output, cases, contract, parallel=args.parallel_runs,
                          check=args.command == "check", report_only=args.command == "report")
    if result is None:
        return 130
    if args.command == "check":
        print(f"complete: bounded 10k budget-runner check {'passed' if result['success'] else 'FAILED'}; no long runs or accuracy scoring")
        return 0 if result["success"] else 2
    print(f"complete: NestedBD-Long budget diagnostic; {result['new_success_count']}/{result['planned_new_runs']} 10M runs; "
          "protected results unchanged; main budget 2M unchanged; convergence not established")
    marker = pilot.read_json(output / "_COMPLETE.json")
    print("report: " + str(pilot.safe_child(output, marker["report_directory"]) / "report.md"))
    return 0 if result["new_success_count"] == result["planned_new_runs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
