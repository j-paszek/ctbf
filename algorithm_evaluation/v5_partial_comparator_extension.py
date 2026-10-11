"""Owner-run additive completion of M4 and the 100-history NestedBD comparison.

No old result, scientific executor or method definition is modified. The optional
technical check uses one new H14 input and 10k iterations in a separate root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import signal
import threading

from algorithm_evaluation import v5_partial_comparator_extension_common as common
from algorithm_evaluation import v5_partial_comparator_extension_report as reporting
from algorithm_evaluation.paper_pipeline_contract import write_json_atomic
from algorithm_evaluation.paper_pipeline_runner import deserialize_tree, measured_stage
from algorithm_evaluation.process_isolation import FreshSpawnTaskPool, fresh_process_contract
from algorithm_evaluation.v5_algorithm_development_common import ARM_SPEC_BY_ID
from trimmed_truth_evaluation import prepare_trimmed_truth, evaluate_prepared_trimmed_truth


pilot, benchmark, topology = common.pilot, common.benchmark, common.topology
nested10m = common.nested10m  # compatibility alias; refreshed after CLI profile selection
frozen, calibration = common.frozen, common.calibration

# Only the audited pre-repair orchestration revisions may cross this boundary.
# Scientific executors, model rules, inputs and results remain exactly pinned.
_PRE_RESUME_REPAIR = {
    "v5_partial_comparator_extension.py": "1f8a2532725a8caa27d5977c7bf8124f342e4e1cd67fec8a048ea6198938ea10",
    "v5_partial_comparator_extension_2m_nested.py": "7c46f71c5a991a9ced9f716a45b0eeddf8a2ed7431c91e27896e455b93e24025",
}
_RESUME_REPAIR_FILE = "resume_repair_v1.json"


def resume_repair(previous: dict, current: dict) -> dict | None:
    """Recognize only this operational repair without rewriting the old contract."""
    if previous == current:
        return None
    old, new = previous.get("protected_sha256", {}), current["protected_sha256"]
    if ({k: v for k, v in previous.items() if k != "protected_sha256"}
            != {k: v for k, v in current.items() if k != "protected_sha256"}
            or set(old) != set(new)):
        raise ValueError("Extension inputs, rules, implementation, mode or runtime changed.")
    replacements = {}
    for name, old_digest in old.items():
        if old_digest == new[name]:
            continue
        path = Path(name)
        if (path != Path(__file__).resolve().with_name(path.name)
                or _PRE_RESUME_REPAIR.get(path.name) != old_digest
                or new[name] != pilot.sha256(path)):
            raise ValueError("Unapproved extension source change: " + name)
        replacements[name] = {"before": old_digest, "after": new[name]}
    if not replacements:
        raise ValueError("Unrecognized extension contract change.")
    return {"schema_version": "ctbf-v5-partial-comparator-resume-repair-v1",
            "source_replacements": replacements,
            "scope": "terminal_outcome_reuse_and_selected_budget_reporting",
            "scientific_rules_and_results_unchanged": True}


def nested_score_task_seconds() -> int:
    return (common.nested10m.SUMMARY["treeannotator_timeout_seconds"]
            + common.LIMITS["evaluation_seconds"] + common.LIMITS["trim_seconds"] + 120)


def prepare(args, *, check=False):
    paths = [Path(getattr(args, name)).expanduser().resolve()
             for name in ("bank_root", "trimmed_root", "nested_root", "bio_root", "output_root", "budget_root")]
    bank, trim, nested, bio, output, budget_root = paths
    common.output_guard(output, bank, bio, [trim, nested, benchmark.DEFAULT_OUTPUT,
                                          budget_root, common.OLD_OUTPUT])
    m4, cases, rows, previous, backend, protected, roots = common.prepare_sources(
        bank, trim, nested, bio, budget_root, check=check)
    common.output_guard(output, bank, bio, roots)
    contract = {"schema_version": common.SCHEMA, "rules": common.RULES,
                "mode": "one_input_technical_check" if check else "100_history_comparison",
                "bank_root": str(bank), "bio_root": str(bio), "trimmed_root": str(trim),
                "nested_root": str(nested), "budget_root": str(budget_root), "protected_sha256": protected,
                "runtime": common.recovery.runtime_identity(), "m4_cases": m4,
                "nested_cases": [{k: v for k, v in c.items() if k != "records"} for c in cases],
                "m4_execution": fresh_process_contract("M4_reconstruction_full_and_trimmed_evaluation"),
                "nested_evaluation_execution": fresh_process_contract("native_MCC_full_and_trimmed_evaluation"),
                "reused_score_digest": common.object_digest(rows),
                "reused_nested_outcomes": previous["reused_outcomes"],
                "nested_score_task_timeout_seconds": nested_score_task_seconds(),
                "concurrency_is_operational_not_scientific": True}
    repair = None
    if output.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume to verify completed work.")
        repair = resume_repair(pilot.read_json(output / "analysis_contract.json"), contract)
        if repair is not None:
            repair["original_contract_sha256"] = pilot.sha256(output / "analysis_contract.json")
    else:
        output.mkdir(parents=True)
        write_json_atomic(output / "analysis_contract.json", contract)
    # Atomic, bounded input manifests make an interrupted preparation resumable.
    with benchmark.owner_lock(output):
        repair_path = output / _RESUME_REPAIR_FILE
        if repair_path.exists():
            if pilot.read_json(repair_path) != repair:
                raise ValueError("The recorded resume repair differs from the current implementation.")
        elif repair is not None:
            write_json_atomic(repair_path, repair)
            print("resume repair recorded: original contract and completed results retained", flush=True)
        for case in cases:
            path = output / "inputs" / f"{case['case_id']}.json"
            manifest = {"case_id": case["case_id"], "input_path": case.get("input_path"),
                        "input_sha256": case.get("input_sha256"), "records": case["records"],
                        "input_summary": case["input_summary"], "eligibility": case["eligibility"],
                        "seeds": case["seeds"]}
            if path.exists():
                if common.bundle_read(path) != manifest:
                    raise ValueError("Prepared observable input changed.")
            else:
                common.bundle_write(path, manifest)
    return output, contract, cases, rows, previous, backend


def base_score(case: dict, method: str, *, replicate=None, seed=None) -> dict:
    return {**{key: case[key] for key in ("case_id", "block_index", "condition_id", "height")},
            "problem": "partial", "method": method, "replicate": replicate, "seed": seed,
            "source_status": "not_run", "status": "failure", "grf_full": None,
            "grf_trim": None, "ad_f1": None, "failure": None,
            "inference_budget_iterations": common.nested10m.ITERATIONS if method == "NestedBD-Long" else None,
            "provenance": "new_additive_extension"}


def trimmed_bundle(bank: Path, case: dict, tree, labels: list, native: dict):
    truth = frozen._load_truth_after_reconstruction(bank, case)
    return evaluate_prepared_trimmed_truth(prepare_trimmed_truth(truth, labels), tree, native)


def apply_trim(score: dict, bank: Path, case: dict, tree, labels, native: dict) -> dict:
    """Full scores survive even an independent trimmed-target audit failure."""
    score.update(grf_full=native["metrics"]["grf"], ad_f1=native["metrics"]["ad_f1"])
    result, resources, error = measured_stage(
        lambda: trimmed_bundle(bank, case, tree, labels, native),
        timeout_seconds=common.LIMITS["trim_seconds"], rss_limit_bytes=common.LIMITS["rss_bytes"])
    if error is None:
        error = frozen._resource_audit_error(resources, "trimmed_evaluation")
    if error is not None or result is None:
        score.update(status="audit_failure", failure={"stage": "trimmed_evaluation",
                     "type": type(error).__name__, "message": str(error)[:4096]})
    else:
        score.update(status="success", grf_trim=result["grf_trim"])
    return {"trimmed": result, "trim_resources": resources}


def m4_task(bank_root: str, case: dict) -> dict:
    """Exactly one case per spawned process; cached D is aligned by old helper."""
    spec = ARM_SPEC_BY_ID[common.M4_ARM]
    if case["status"] != "available":
        record = frozen._dependency_record(case, spec, dependency="bank_case", details=case.get("failure"))
    else:
        record = frozen._execute_record(bank_root, case, spec, None,
            common.LIMITS["reconstruction_seconds"], common.LIMITS["evaluation_seconds"], common.LIMITS["rss_bytes"])
    score = base_score(case, "M4")
    score.update(source_status=record["status"], status=record["status"], failure=record.get("failure"))
    result = {"score": score, "record": record}
    if record["status"] == "success":
        try:
            labels = record["native_evaluation"]["observation_labels"]
            result.update(apply_trim(score, Path(bank_root), case, deserialize_tree(record["tree"]),
                                     labels, record["native_evaluation"]))
        except Exception as error:
            score.update(status="audit_failure", failure={"stage": "trimmed_evaluation",
                         "type": type(error).__name__, "message": str(error)[:4096]})
    return result


def score_path(output: Path, case_id: str, method: str, seed=None) -> Path:
    return output / "scores" / method / (case_id + (f"-seed_{seed}" if seed is not None else "") + ".json")


def cached_score(path: Path, case: dict, method: str, seed=None, replicate=None):
    if not path.exists():
        return None
    bundle = common.bundle_read(path)
    expected = base_score(case, method, seed=seed, replicate=replicate)
    if any(bundle["score"].get(key) != expected[key]
           for key in ("case_id", "block_index", "condition_id", "height", "method", "seed", "replicate")):
        raise ValueError("Cached score identity differs.")
    if (method == "NestedBD-Long"
            and bundle["score"].get("inference_budget_iterations") != common.nested10m.ITERATIONS):
        raise ValueError("Cached score is not from the selected uniform NestedBD budget.")
    return bundle


def run_m4(output: Path, contract: dict, *, workers: int) -> list[dict]:
    scores, pending = [], []
    for case in contract["m4_cases"]:
        cached = cached_score(score_path(output, case["case_id"], "M4"), case, "M4")
        if cached is None:
            pending.append(case)
        else:
            scores.append(cached["score"])
    print(f"M4: {len(scores)} reused; {len(pending)} pending; {workers} fresh-process workers", flush=True)
    if pending:
        with FreshSpawnTaskPool(workers, buffer_factor=2) as pool:
            tasks = ((contract["bank_root"], case) for case in pending)
            for case, result in zip(pending, pool.map_ordered(m4_task, tasks, timeout_seconds=1020), strict=True):
                common.bundle_write(score_path(output, case["case_id"], "M4"), result)
                scores.append(result["score"])
                print(f"M4 {len(scores)}/{len(contract['m4_cases'])}: {case['case_id']}: {result['score']['status']}", flush=True)
    return scores


def run_inference(backend, output: Path, cases: list[dict], *, parallel: int, reused: list[dict]):
    numeric = benchmark.numeric_gate(backend, output)
    if not numeric["passes_observed_range"]:
        raise ValueError("Unchanged native arithmetic gate failed; no MCMC started.")
    outcomes = list(reused)
    keys = {(row["case_id"], row["seed"]) for row in reused}
    planned = {(c["case_id"], s["seed"]) for c in cases for s in c["seeds"]}
    if len(keys) != len(reused) or not keys.issubset(planned):
        raise ValueError("Reusable 10M outcomes are duplicated or outside the declared scope.")
    stop = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    label = common.nested10m.LABEL
    print(f"NestedBD {label}: {len(reused)} prior runs reused; "
          f"{len(planned)-len(reused)} extension seed outcomes", flush=True)
    print("Existing sealed outcomes are verified and reused; only missing/interrupted attempts start BEAST.",
          flush=True)
    try:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(nested10m.inference_task, backend, output, case, seed["seed"], stop)
                       for case in cases for seed in case["seeds"] if (case["case_id"], seed["seed"]) not in keys]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        outcomes.append(result)
                        print(f"NestedBD {label} {len(outcomes)}/{len(planned)} terminal outcomes: "
                              f"{result['case_id']} seed {result['seed']}: {result['status']}", flush=True)
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        if stop.is_set():
            print("interrupted: completed attempts retained; --resume restarts unfinished attempts from zero", flush=True)
            return None
        return sorted(outcomes, key=lambda row: (row["case_id"], row["seed"]))
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def external_score_task(bio_root: str, bank_root: str, output_root: str, case: dict,
                        replicate: int, outcome: dict, budget_profile: str) -> dict:
    # Spawn imports the 10M default. The controller's globals are not inherited.
    common.configure_profile(budget_profile)
    backend = pilot.load_backend(Path(bio_root))
    result = common.nested10m.summarize_external(
        backend, Path(bank_root), Path(output_root), case, replicate, outcome)
    score = base_score(case, "NestedBD-Long", replicate=replicate, seed=outcome["seed"])
    score.update(status=result["status"], source_status=result["status"],
                 grf_full=result.get("grf"), failure=result.get("failure"), provenance=outcome["provenance"])
    bundle = {"score": score, "external": result}
    if result["status"] == "success":
        directory = Path(result["attempt"])
        topology._verify_seal(directory)
        native = pilot.read_json(directory / "evaluation.json")
        tree = deserialize_tree(pilot.read_json(directory / "tree.json"))
        try:
            bundle.update(apply_trim(score, Path(bank_root), case["source_case"], tree,
                                    sorted({r["state_label"] for r in case["records"]}), native))
        except Exception as error:
            score.update(status="audit_failure", failure={"stage": "trimmed_evaluation",
                         "type": type(error).__name__, "message": str(error)[:4096]})
    return bundle


def summarize_new(output: Path, contract: dict, cases: list[dict], outcomes: list[dict], *, workers: int):
    lookup = {(row["case_id"], row["seed"]): row for row in outcomes}
    expected = {(case["case_id"], seed["seed"]) for case in cases for seed in case["seeds"]}
    if len(outcomes) != len(expected) or set(lookup) != expected:
        raise ValueError("Not all declared new-scope outcomes are terminal.")
    bundles, tasks, coordinates = [], [], []
    for case in cases:
        for seed in case["seeds"]:
            path = score_path(output, case["case_id"], "NestedBD-Long", seed["seed"])
            cached = cached_score(path, case, "NestedBD-Long", seed["seed"], seed["replicate"])
            if cached is not None:
                bundles.append(cached)
            else:
                tasks.append((contract["bio_root"], contract["bank_root"], str(output), case,
                              seed["replicate"], lookup[case["case_id"], seed["seed"]], common._PROFILE))
                coordinates.append(path)
    if tasks:
        with FreshSpawnTaskPool(min(workers, 2), buffer_factor=2) as pool:
            for path, bundle in zip(coordinates, pool.map_ordered(
                    external_score_task, tasks,
                    timeout_seconds=nested_score_task_seconds()), strict=True):
                common.bundle_write(path, bundle)
                bundles.append(bundle)
                print(f"NestedBD summaries {len(bundles)}/{len(outcomes)}: {bundle['score']['status']}", flush=True)
    return bundles


def diagnostics(backend, cases: list[dict], outcomes: list[dict], previous: dict, bundles: list[dict]) -> dict:
    clades = {}
    lookup = {(row["case_id"], row["seed"]): row for row in outcomes}
    for case in cases:
        runs = [lookup[case["case_id"], seed["seed"]] for seed in case["seeds"]]
        if all(row["status"] == "success" for row in runs):
            windows = [pilot.read_json(Path(row["run"]["attempt"]) / "clade_windows.json") for row in runs]
            clades[case["case_id"]] = backend.audit.review.compare_windows(
                *[{int(k): value for k, value in window.items()} for window in windows])
    records = [bundle["external"] for bundle in bundles]
    return {"convergence_established": False, "iterations_per_seed": common.nested10m.ITERATIONS,
            "clade_agreement": clades, "seed_records": records,
            "inference_outcomes": outcomes, "old_2m_results_modified": False,
            "reused_prior_inference_runs": sum(
                common.nested10m.REUSED_PROVENANCE is not None
                and row["provenance"] == common.nested10m.REUSED_PROVENANCE
                for row in outcomes),
            "old_2m_diagnostics_included": False}


def finish_report(output: Path, contract: dict, old_rows: list[dict], m4: list[dict],
                  bundles: list[dict], diagnostic: dict):
    rows = old_rows + m4 + [bundle["score"] for bundle in bundles]
    report = reporting.build_report(rows, bootstrap_draws=common.RULES["bootstrap_draws"])
    pilot.verify_hashes(contract["protected_sha256"])
    reused = sum(common.nested10m.REUSED_PROVENANCE is not None
                 and b["score"]["provenance"] == common.nested10m.REUSED_PROVENANCE
                 for b in bundles)
    report.update(contract_sha256=pilot.sha256(output / "analysis_contract.json"),
                  protected_results_unchanged=True, nested_new_seed_records=len(bundles)-reused,
                  nested_reused_seed_records=reused,
                  nested_all_scores_recomputed_at_10m=common._PROFILE == "10m",
                  nested_all_scores_recomputed_at_selected_budget=True)
    parent = output / "reports"
    previous = sorted(parent.glob("report_*")) if parent.exists() else []
    number = max((int(path.name.removeprefix("report_")) for path in previous), default=0) + 1
    directory = parent / f"report_{number:03d}"
    directory.mkdir(parents=True)
    reporting.write_report(directory, rows, report)
    write_json_atomic(directory / "diagnostics.json", diagnostic)
    topology._seal(directory)
    # Bind new per-case score bundles and all immutable native attempt seals.
    artifacts = {str(path.relative_to(output)): pilot.sha256(path)
                 for path in sorted((output / "scores").glob("*/*.json"))}
    for bundle in bundles:
        attempt = Path(bundle["external"]["attempt"])
        artifacts[str((attempt / "_SEALED.json").relative_to(output))] = pilot.sha256(attempt / "_SEALED.json")
    for case in contract["nested_cases"]:
        for seed in case["seeds"]:
            for phase in ("technical", "pilot" if common._PROFILE == "2m" else "long"):
                unit = output / "inference" / case["case_id"] / f"seed_{seed['seed']}" / phase
                for seal in sorted(unit.glob("attempt_*/_SEALED.json")):
                    artifacts[str(seal.relative_to(output))] = pilot.sha256(seal)
    if (output / _RESUME_REPAIR_FILE).is_file():
        artifacts[_RESUME_REPAIR_FILE] = pilot.sha256(output / _RESUME_REPAIR_FILE)
    marker = {"schema_version": common.SCHEMA, "contract_sha256": report["contract_sha256"],
              "report_directory": str(directory.relative_to(output)),
              "report_sha256": pilot.sha256(directory / "report.json"), "artifacts": artifacts}
    write_json_atomic(output / "_COMPLETE.json", marker)
    return report


def completed(output: Path, contract: dict):
    path = output / "_COMPLETE.json"
    if not path.exists():
        return None
    marker = pilot.read_json(path)
    if marker.get("schema_version") != common.SCHEMA or marker["contract_sha256"] != pilot.sha256(output / "analysis_contract.json"):
        raise ValueError("Completion contract changed.")
    directory = common.safe_child(output, marker["report_directory"])
    topology._verify_seal(directory)
    if pilot.sha256(directory / "report.json") != marker["report_sha256"]:
        raise ValueError("Completed report changed.")
    for relative, digest in marker["artifacts"].items():
        common.bind(common.safe_child(output, relative), {}, digest)
    pilot.verify_hashes(contract["protected_sha256"])
    return pilot.read_json(directory / "report.json")


def technical_check(output, contract, cases, backend, *, workers):
    if len(cases) != 1 or contract["mode"] != "one_input_technical_check":
        raise ValueError("The technical check requires its own one-input root.")
    m4 = run_m4(output, contract, workers=workers)
    numeric = benchmark.numeric_gate(backend, output)
    if not numeric["passes_observed_range"]:
        raise ValueError("Native arithmetic gate failed.")
    case = cases[0]
    if case["eligibility"] != "compatible":
        raise ValueError("First declared new case is unsupported; do not silently replace it.")
    seed = case["seeds"][0]["seed"]
    unit = output / "technical_check" / case["case_id"]
    if common._PROFILE == "2m":
        directory, run = common.nested10m.choose_attempt(backend, unit, case, "technical", seed)
    else:
        directory, run = common.nested10m.budget.choose_attempt(
            backend, unit, case, "technical", seed)
    if run is None:
        if common._PROFILE == "2m":
            run = calibration.execute_attempt(
                backend, directory, case, "technical", seed, benchmark.CLOCK, threading.Event())
        else:
            run = common.nested10m.budget.execute_attempt(
                backend, directory, case, "technical", seed, threading.Event())
    pilot.verify_hashes(contract["protected_sha256"])
    return {"success": run["status"] == "technical_complete" and m4[0]["status"] == "success",
            "m4": m4, "nested_technical": run, "no_long_inference": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run", "report"))
    parser.add_argument("--nested-budget", choices=("2m", "10m"), default="10m",
                        help="uniform NestedBD budget; 2m starts a fresh 2M series")
    parser.add_argument("--bank-root", type=Path, default=common.DEFAULT_BANK)
    parser.add_argument("--trimmed-root", type=Path, default=common.DEFAULT_TRIM)
    parser.add_argument("--nested-root", type=Path, default=common.DEFAULT_NESTED)
    parser.add_argument("--budget-root", type=Path, default=common.DEFAULT_BUDGET,
                        help="completed eight-run 10M diagnostic; used only with --nested-budget 10m")
    parser.add_argument("--bio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--parallel-runs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    # Avoid resetting test-scoped inventory overrides when the default profile
    # is already active. A 2M invocation always switches profiles in a fresh
    # process and therefore restores the full declared 2M roster.
    if args.nested_budget != common._PROFILE:
        common.configure_profile(args.nested_budget)
    global nested10m
    nested10m = common.nested10m
    if args.output_root is None:
        if args.command == "check":
            parser.error("check needs a separate explicit --output-root; never use the production root")
        args.output_root = common.DEFAULT_OUTPUT
    if args.command == "report" and not (args.output_root.expanduser().resolve() / "analysis_contract.json").is_file():
        parser.error("report requires an existing prepared run and never starts inference")
    output, contract, cases, old_rows, previous, backend = prepare(args, check=args.command == "check")
    with benchmark.owner_lock(output):
        if args.command == "prepare":
            print(f"prepared: {len(contract['m4_cases'])} M4 conditions; {len(cases)*2} "
                  f"NestedBD {common.nested10m.LABEL} seed records "
                  f"({len(previous['reused_outcomes'])} reused inference runs); no inference")
            return 0
        if args.command == "check":
            result = technical_check(output, contract, cases, backend, workers=args.workers)
            print(f"complete: partial-comparator technical check "
                  f"{'success' if result['success'] else 'FAILED'}; one M4; one 10k NestedBD; "
                  f"no {common.nested10m.LABEL} inference")
            return 0 if result["success"] else 1
        report = completed(output, contract)
        if report is None:
            if args.command == "report":
                # Reporting never starts missing reconstruction or MCMC work.
                missing = [c for c in contract["m4_cases"] if not score_path(output, c["case_id"], "M4").is_file()]
                if missing:
                    raise ValueError("M4 records are unfinished; report cannot start inference.")
                outcomes = []
                reused = {(row["case_id"], row["seed"]): row for row in previous["reused_outcomes"]}
                for case in cases:
                    for seed in case["seeds"]:
                        key = case["case_id"], seed["seed"]
                        outcomes.append(reused[key] if key in reused else common.nested10m.cached_inference(
                            backend, output, case, seed["seed"]))
                m4 = [cached_score(score_path(output, c["case_id"], "M4"), c, "M4")["score"] for c in contract["m4_cases"]]
            else:
                m4 = run_m4(output, contract, workers=args.workers)
                outcomes = run_inference(backend, output, cases, parallel=args.parallel_runs,
                                         reused=previous["reused_outcomes"])
                if outcomes is None:
                    return 130
            bundles = summarize_new(output, contract, cases, outcomes, workers=args.workers)
            diagnostic = diagnostics(backend, cases, outcomes, previous, bundles)
            report = finish_report(output, contract, old_rows, m4, bundles, diagnostic)
        failures = sum(row["status"] != "success" for row in report["case_scores"])
        print(f"complete: partial comparator extension {common.nested10m.LABEL}; "
              f"{report['raw_record_count']} records; "
              f"{report['nested_new_seed_records']} new + {report['nested_reused_seed_records']} reused NestedBD seed records; "
              f"{failures} incomplete case-method pairs; protected results unchanged; convergence not established")
        print("report: " + str(output / pilot.read_json(output / "_COMPLETE.json")["report_directory"] / "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
