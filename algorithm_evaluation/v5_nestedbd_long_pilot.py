"""Lossless, owner-run NestedBD-Long feasibility on frozen v5 development inputs.

This is not a new reconstruction arm or an accuracy evaluation. The optional
ctbfbio runtime supplies pinned external binaries and previously checked helpers.
It is loaded only at an explicit command boundary and is never modified here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import difflib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import xml.etree.ElementTree as ET


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = PROJECT / "algorithm_evaluation/results/v5_development/bank_v2"
DEFAULT_OUTPUT = PROJECT / "algorithm_evaluation/results/v5_development/nestedbd_long_pilot_v1"
CASE_IDS = ("dev-b001-H14", "dev-b001-H24", "dev-b001-H34")
SCHEDULES = ((9, 12, 14), (15, 20, 24), (21, 28, 34))
SEEDS = (2026092601, 2026092602)
SCHEMA = "ctbf-v5-nestedbd-long-lossless-pilot-v1"
SAMPLING_RULE = "min(N,max(6,ceil(0.5*N)))"
STATE_MAXIMUM = 9
POLICY = "uninterrupted_restart_incomplete_from_zero"
SETTINGS = {
    "technical_iterations": 10000,
    "pilot_iterations": 2000000,
    "log_every": 1000,
    "state_write_every": 10000,
    "burnin_percent": 10,
    "technical_timeout_seconds": 300,
    "pilot_timeout_seconds": 7200,
    "rss_limit_bytes_per_run": 4 * 1024**3,
    "capture_limit_bytes_per_run": 8 * 1024**2,
    "coupled_chains_per_run": 4,
    "maximum_parallel_runs": 2,
    "time_unit": "one_simulator_generation",
    "clock_and_biological_priors": "unchanged_released_template_not_rate_calibrated",
    "state_count": STATE_MAXIMUM + 1,
    "bins_trimmed": False,
    "values_clipped": False,
    "observations_resampled": False,
    "native_checkpoint_restoration": False,
    "accuracy_metrics_computed": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def safe_child(root: Path, relative: str) -> Path:
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts:
        raise ValueError("unsafe relative path")
    resolved = (root / item).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("path escapes its root")
    return resolved


def verify_hashes(paths: dict[str, str]) -> None:
    for name, expected in paths.items():
        if sha256(Path(name)) != expected:
            raise ValueError("protected input or implementation changed: " + name)


def output_guard(output: Path, bank: Path, bio: Path) -> None:
    for protected in (bank, bio, PROJECT / "algorithm_evaluation/results/v5_confirmation"):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("output overlaps a protected input/runtime/result root")
    if output == PROJECT or output in PROJECT.parents:
        raise ValueError("broad output root refused")


def input_records(payload: dict, expected_bins: int = 100) -> list[dict]:
    """Validate and assign observable occurrence IDs, preserving every entry.

    High CN is deliberately retained here. Compatibility is a separate gate,
    never a transformation or a reason to replace the selected case.
    """
    if payload.get("schema_version") != "ctbf-v5-reconstruction-input-v1":
        raise ValueError("unsupported reconstruction input schema")
    if payload.get("sampling_rule") != SAMPLING_RULE:
        raise ValueError("unexpected frozen sampling rule")
    levels = payload.get("levels", [])
    if len(levels) != 3:
        raise ValueError("this pilot requires the three original biopsies")
    records, state_profiles, profile_states = [], {}, {}
    last_generation = -1
    for index, level in enumerate(levels):
        generation = level.get("generation")
        if (type(generation) is not int or generation <= last_generation
                or level.get("biopsy_level") != index):
            raise ValueError("biopsy indices/generations are not strictly ordered")
        last_generation = generation
        seen_labels, seen_profiles = set(), set()
        if not level.get("states"):
            raise ValueError("empty biopsy is not supported by this pilot")
        for state in level["states"]:
            if set(state) != {"state_label", "cnp"}:
                raise ValueError("only observed state labels and CNPs may enter inference")
            raw_label, values = state["state_label"], state["cnp"]
            if type(raw_label) not in (int, str) or not str(raw_label).strip():
                raise ValueError("invalid state label")
            label = str(raw_label).strip()
            if (not isinstance(values, list) or len(values) != expected_bins
                    or any(type(cn) is not int or cn < 0 for cn in values)):
                raise ValueError("invalid integer CNP or changed bin count")
            profile = tuple(values)
            if label in seen_labels or profile in seen_profiles:
                raise ValueError("input must already be collapsed within each biopsy")
            if (label in state_profiles and state_profiles[label] != profile
                    or profile in profile_states and profile_states[profile] != label):
                raise ValueError("state labels and exact CNPs are not aligned")
            seen_labels.add(label)
            seen_profiles.add(profile)
            state_profiles[label], profile_states[profile] = profile, label
            digest = hashlib.sha256(",".join(map(str, values)).encode("ascii")).hexdigest()
            records.append({"taxon": f"B{index}_cnp_sha256_{digest}",
                            "biopsy_level": index, "generation": generation,
                            "state_label": label, "cnp": values})
    if len({row["taxon"] for row in records}) != len(records):
        raise ValueError("occurrence identifiers collide")
    return records


def compatibility(records: list[dict]) -> dict:
    values = [cn for row in records for cn in row["cnp"]]
    unsupported = sum(cn > STATE_MAXIMUM for cn in values)
    return {"occurrences": len(records),
            "distinct_states": len({tuple(row["cnp"]) for row in records}),
            "biopsy_sizes": [sum(row["biopsy_level"] == i for row in records) for i in range(3)],
            "bins": len(records[0]["cnp"]), "minimum_cn": min(values),
            "maximum_cn": max(values), "entries_above_9": unsupported,
            "status": "compatible" if not unsupported else "blocked_copy_number_range",
            "profiles_transformed": False}


def load_cases(bank: Path) -> tuple[list[dict], dict[str, str]]:
    manifest_path = bank / "bank_manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("schema_version") != "ctbf-v5-algorithm-development-bank-v2"
            or manifest.get("bank_id") != "ctbf-v5-g1-06-development-bank-v2"
            or manifest.get("status") != "complete"):
        raise ValueError("the completed frozen development bank v2 is required")
    config_path = safe_child(bank, manifest["simulator_config_path"])
    config = read_json(config_path)
    if (config.get("SIMULATOR_SEMANTIC_VERSION") != "ctbf-cnp-state-simulator-v5"
            or config.get("GENOME_LENGTH") != 100
            or config.get("NUMBER_OF_CHROMOSOMES") != 1):
        raise ValueError("expected the original 100-bin, one-chromosome v5 layout")
    protected = {str(path): sha256(path) for path in (manifest_path, config_path)}
    cases = []
    for case_id, schedule in zip(CASE_IDS, SCHEDULES):
        matches = [case for case in manifest["cases"] if case["case_id"] == case_id]
        if len(matches) != 1 or matches[0]["block_index"] != 0:
            raise ValueError("prespecified case missing or duplicated: " + case_id)
        path = safe_child(bank, matches[0]["input_path"])
        payload = read_json(path)
        if (payload.get("case_id") != case_id
                or tuple(level["generation"] for level in payload["levels"]) != schedule):
            raise ValueError("case identity or original biopsy schedule changed")
        records = input_records(payload)
        protected[str(path)] = sha256(path)
        cases.append({"case_id": case_id, "input_path": str(path),
                      "input_sha256": protected[str(path)], "generations": list(schedule),
                      "records": records, "input_summary": compatibility(records)})
    return cases, protected


def load_backend(bio: Path):
    scripts = bio / "workflow/scripts"
    if not (scripts / "nestedbd_long_corrected_v1.py").is_file():
        raise ValueError("the existing corrected ctbfbio helpers are required")
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(scripts))
    try:
        backend = importlib.import_module("nestedbd_long_corrected_v1")
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = old_bytecode
    if Path(backend.__file__).resolve().parent != scripts.resolve():
        raise ValueError("ctbfbio helpers were already loaded from another root")
    backend.runtime.verify_runtime(backend.runtime.RUNTIME)
    return backend


def backend_snapshot(backend) -> dict[str, str]:
    paths = {Path(__file__).resolve(), backend.shared.SOURCE / backend.gate.TEMPLATE_PATH,
             backend.runtime.RUNTIME / "runtime_manifest.json"}
    for name, module in tuple(sys.modules.items()):
        if name.startswith("nestedbd_long_") and getattr(module, "__file__", None):
            paths.add(Path(module.__file__).resolve())
    runtime_manifest = backend.runtime.verify_runtime(backend.runtime.RUNTIME)
    paths.update(safe_child(backend.runtime.RUNTIME, relative)
                 for relative in runtime_manifest["files"])
    return {str(path): sha256(path) for path in sorted(paths)}


def phase_config(case: dict, phase: str) -> dict:
    if phase not in ("technical", "pilot"):
        raise ValueError("unknown phase")
    return {"forward_cycle_positions": case["generations"],
            "chain_length": SETTINGS[phase + "_iterations"],
            "log_every": SETTINGS["log_every"],
            "burnin_percent": SETTINGS["burnin_percent"],
            "timeout_seconds_per_run": SETTINGS[phase + "_timeout_seconds"],
            "rss_limit_bytes_per_run": SETTINGS["rss_limit_bytes_per_run"],
            "capture_limit_bytes_per_run": SETTINGS["capture_limit_bytes_per_run"]}


def sequences(case: dict) -> dict[str, tuple[int, ...]]:
    return {row["taxon"]: tuple(row["cnp"]) for row in case["records"]}


def verify_xml_input(tree: ET.ElementTree, case: dict) -> None:
    root = tree.getroot()
    actual = {}
    for node in root.findall("./data/sequence"):
        if node.get("id") != node.get("taxon") or node.get("taxon") in actual:
            raise ValueError("XML occurrence identity changed")
        actual[node.get("taxon")] = tuple(map(int, node.text.strip().split(",")))
    if actual != sequences(case):
        raise ValueError("XML lost, duplicated, trimmed, or clipped original CNP data")
    dates = root.find(".//trait[@id='dateTrait.t:bears']")
    actual_dates = {}
    for item in dates.text.strip().split(","):
        name, value = item.strip().split("=")
        if name in actual_dates:
            raise ValueError("duplicate taxon date")
        actual_dates[name] = float(value)
    expected = {row["taxon"]: case["generations"][-1] - row["generation"]
                for row in case["records"]}
    if dates.get("traitname") != "date-backward" or actual_dates != expected:
        raise ValueError("sampling-generation times changed")


def make_xml(backend, case: dict, phase: str):
    if case["input_summary"]["status"] != "compatible":
        raise ValueError("unsupported copy numbers retained unchanged; no inference")
    baseline, tree, proof = backend.corrected_xml(
        sequences(case), phase_config(case, phase), SETTINGS["state_write_every"])
    verify_xml_input(tree, case)
    root = tree.getroot()
    if (root.find(".//distribution[@id='treeLikelihood.bears']").get("nstates") != "10"
            or root.find(".//substModel[@id='BD']").get("nstate") != "10"):
        raise ValueError("external finite-state configuration differs")
    return baseline, tree, proof


def numerical_control(backend, directory: Path) -> dict:
    """Check the unchanged released binary using real BEAST classes, not stubs."""
    directory.mkdir()
    runtime = backend.runtime
    source = directory / "NestedBDProbe.java"
    with source.open("x", encoding="utf-8") as handle:
        handle.write(runtime.probe.SOURCES["NestedBDProbe.java"])
    classes = directory / "classes"
    classes.mkdir()
    javac = shutil.which("javac")
    if javac is None:
        raise RuntimeError("javac is required for the original-binary arithmetic gate")
    subprocess.run([javac, "-cp", runtime.classpath(runtime.RUNTIME), "-d", str(classes), str(source)],
                   check=True, capture_output=True, timeout=60)
    prefix = runtime.java_prefix(runtime.RUNTIME)
    command = [*prefix[:-1], runtime.classpath(runtime.RUNTIME) + os.pathsep + str(classes),
               "NestedBDProbe", str(STATE_MAXIMUM), *map(str, runtime.probe.PROBE_DISTANCES)]
    path = directory / "original_binary_arithmetic.tsv"
    with path.open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=60)
    result = runtime.probe.summarize_native_output(path, STATE_MAXIMUM)
    result["scope"] = "unchanged_released_BD_binary_with_real_BEAST_classes_no_type_stubs"
    write_json(directory / "report.json", result)
    backend.pilot.seal(directory)
    return result


def write_input_copy(root: Path, case: dict) -> None:
    directory = root / "inputs" / case["case_id"]
    directory.mkdir(parents=True)
    with (directory / "input.json").open("xb") as handle:
        handle.write(Path(case["input_path"]).read_bytes())
    if sha256(directory / "input.json") != case["input_sha256"]:
        raise ValueError("frozen input copy differs")
    with (directory / "occurrence_map.tsv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["taxon", "biopsy_level", "generation", "state_label"])
        writer.writerows([row[key] for key in ("taxon", "biopsy_level", "generation", "state_label")]
                         for row in case["records"])
    write_json(directory / "input_summary.json", case["input_summary"])


def contract(cases: list[dict], protected: dict, check: bool, parallel: int) -> dict:
    return {"schema_version": SCHEMA, "bounded_check": check,
            "case_selection": "first_development_block_standard_heights_before_score_inspection",
            "cases": [{key: value for key, value in case.items() if key != "records"} for case in cases],
            "independent_seeds": list(SEEDS), "settings": SETTINGS,
            "parallel_runs": parallel, "execution_policy": POLICY,
            "protected_sha256": protected,
            "truth_distances_and_scores_loaded": False,
            "convergence_established": False, "accuracy_metrics_computed": False}


def prepare(root: Path, cases: list[dict], expected: dict, backend, resume: bool) -> None:
    if root.exists():
        if not resume:
            raise ValueError("output exists; use --resume for exact completed-run reuse")
        if read_json(root / "analysis_contract.json") != expected:
            raise ValueError("resumed pilot contract differs")
        backend.pilot.verify_seal(root / "inputs")
        for case in cases:
            if sha256(root / "inputs" / case["case_id"] / "input.json") != case["input_sha256"]:
                raise ValueError("preserved input differs from the frozen source")
        return
    root.mkdir(parents=True)
    write_json(root / "analysis_contract.json", expected)
    for case in cases:
        write_input_copy(root, case)
    backend.pilot.seal(root / "inputs")


def run_attempt(backend, directory: Path, case: dict, phase: str, seed: int,
                stop: threading.Event) -> dict:
    directory.mkdir(parents=True)
    config = phase_config(case, phase)
    report = {"schema_version": SCHEMA, "case_id": case["case_id"], "phase": phase,
              "seed": seed, "config": config, "input_sha256": case["input_sha256"],
              "attempt": str(directory), "status": "failed", "execution_policy": POLICY,
              "native_checkpoint_restoration": False, "convergence_established": False}
    try:
        baseline, tree, proof = make_xml(backend, case, phase)
        baseline.write(directory / "baseline.xml", encoding="utf-8", xml_declaration=True)
        tree.write(directory / "input.xml", encoding="utf-8", xml_declaration=True)
        verify_xml_input(ET.parse(directory / "input.xml"), case)
        write_json(directory / "xml_changes.json", proof)
        with (directory / "xml_changes.diff").open("x", encoding="utf-8") as handle:
            handle.write("".join(difflib.unified_diff(
                ET.tostring(baseline.getroot(), encoding="unicode").splitlines(keepends=True),
                ET.tostring(tree.getroot(), encoding="unicode").splitlines(keepends=True),
                fromfile="released-data-template.xml", tofile="corrected-input.xml")))
        command = backend.native_command(directory, seed)
        report["command"] = command
        print(f"start: {case['case_id']} {phase} seed {seed}; {config['chain_length']:,} iterations", flush=True)
        execution = backend.pilot.execute(command, directory, config, stop)
        report["execution"] = execution
        report["status"] = execution["status"]
        if execution["status"] == "completed":
            taxa = set(sequences(case))
            report["finite_state_checks"] = backend.verify_run(directory, config, taxa)
            if phase == "pilot":
                analysis = backend.pilot.analyse(directory, config, taxa)
                diagnostics = backend.audit.review
                rows = backend.pilot.trace_rows(directory / "shared_pilot.log", config)
                report["scalar_windows"] = diagnostics.scalar_windows(
                    directory, rows, analysis["cold_chain_statistics"], directory, seed)
                windows = diagnostics.tree_windows(directory / "shared_pilot.trees", sorted(taxa), config)
                write_json(directory / "clade_windows.json", {
                    str(b): {"samples": w["samples"], "counts": {str(k): v for k, v in w["counts"].items()}}
                    for b, w in windows.items()})
            report["status"] = "technical_complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    write_json(directory / "report.json", report)
    backend.pilot.seal(directory)
    print(f"finished: {case['case_id']} {phase} seed {seed}: {report['status']}", flush=True)
    return report


def existing_attempt(backend, seed_root: Path, case: dict, phase: str, seed: int):
    # An unsealed failed report is not permission to retry a scientific error.
    for path in sorted(seed_root.glob("attempt_*/report.json")):
        if not (path.parent / "_SEALED.json").exists():
            raise ValueError("unsealed existing report requires review: " + str(path))
    directory, report = backend.choose_attempt(seed_root)
    if report is not None:
        for key, expected in {"schema_version": SCHEMA, "case_id": case["case_id"],
                              "phase": phase, "seed": seed, "config": phase_config(case, phase),
                              "input_sha256": case["input_sha256"],
                              "execution_policy": POLICY, "native_checkpoint_restoration": False}.items():
            if report.get(key) != expected:
                raise ValueError("completed attempt identity/configuration differs: " + key)
        verify_xml_input(ET.parse(directory / "input.xml"), case)
    return directory, report


def run_phase(backend, root: Path, cases: list[dict], phase: str, seeds: tuple[int, ...],
              parallel: int, stop: threading.Event) -> list[dict]:
    results, jobs = [], []
    for case in cases:
        for seed in seeds:
            directory, report = existing_attempt(
                backend, root / phase / case["case_id"] / f"seed_{seed}", case, phase, seed)
            if report is None:
                jobs.append((directory, case, seed))
            else:
                results.append(report)
                print(f"reuse: {case['case_id']} {phase} seed {seed}", flush=True)

    def task(directory, case, seed):
        if stop.is_set():
            return None
        result = run_attempt(backend, directory, case, phase, seed, stop)
        if result["status"] != "technical_complete":
            stop.set()
        return result

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(task, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    return sorted(results, key=lambda r: (r["case_id"], r["seed"]))


def pair_diagnostics(backend, runs: list[dict]) -> dict:
    result = {}
    for case_id in CASE_IDS:
        pair = sorted((r for r in runs if r["case_id"] == case_id
                       and r["status"] == "technical_complete"), key=lambda r: r["seed"])
        if len(pair) != 2:
            continue
        windows = [read_json(Path(run["attempt"]) / "clade_windows.json") for run in pair]
        normalized = [{int(b): {"samples": w["samples"], "counts": w["counts"]}
                       for b, w in win.items()} for win in windows]
        result[case_id] = backend.audit.review.compare_windows(*normalized)
    return result


def report_markdown(report: dict) -> str:
    lines = ["# Lossless frozen-input NestedBD-Long pilot", "", f"Status: {report['status']}.", "",
             "All original CNP bins, integer values and biopsy-indexed observations are retained.",
             "No simulation, CTBF reconstruction, accuracy scoring or consensus selection was performed.",
             "These three heights share one development truth block, not three independent histories.", "",
             "## Inputs", "", "| Case | Biopsy sizes | Bins | CN range | Compatibility |",
             "| --- | --- | ---: | --- | --- |"]
    for case in report["cases"]:
        s = case["input_summary"]
        lines.append(f"| {case['case_id']} | {s['biopsy_sizes']} | {s['bins']} | "
                     f"{s['minimum_cn']}–{s['maximum_cn']} | {s['status']} |")
    lines += ["", "## Independent pilot runs", "",
              "| Case | Seed | Status | Hours | Peak JVM RSS (MiB) | Posterior ESS | Tree-height ESS |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for run in report["pilot_runs"]:
        execution = run.get("execution", {})
        stats = run.get("scalar_windows", {}).get("10", {}).get("official_beast_statistics", {})
        def number(value):
            return "NA" if value is None else f"{value:.2f}"
        lines.append(f"| {run['case_id']} | {run['seed']} | {run['status']} | "
                     f"{number(execution.get('wall_seconds', 0) / 3600)} | "
                     f"{number(execution.get('peak_rss_bytes', 0) / 1024**2)} | "
                     f"{number(stats.get('posterior', {}).get('ESS'))} | "
                     f"{number(stats.get('TreeHeight', {}).get('ESS'))} |")
    lines += ["", "## Between-run clade agreement", "",
              "| Case | Burn-in | Majority clades (run 1 / run 2) | Shared | Maximum frequency difference |",
              "| --- | ---: | --- | ---: | ---: |"]
    for case, windows in report["tree_agreement"].items():
        for burnin in ("10", "25", "50"):
            d = windows[burnin]
            counts = " / ".join(map(str, d["majority_clades_per_seed"]))
            lines.append(f"| {case} | {burnin}% | {counts} | {d['shared_majority_clades']} | "
                         f"{d['maximum_clade_frequency_difference']:.3f} |")
    lines += ["", "## Interpretation", "",
              "10% burn-in is primary. The 25% and 50% windows are diagnostic sensitivities.",
              "Completion and finite values are not a convergence certificate. Review all scalar",
              "traces and independent-run clades before choosing any further experiment or scoring trees.",
              "No high-ESS seed is selected and no samples are pooled. Small inputs need not converge.", "",
              "One analysis time unit is one simulator generation. Only observed sampling generations",
              "enter the XML, as backward ages relative to the latest biopsy. Released priors and",
              "the fixed mean clock are unchanged and are not calibrated to CTBF event probabilities.",
              "The birth–death sampling model is not the CTBF representative-state sampling model.",
              "Numerical times provide more information than CTBF's biopsy ordering. This is a",
              "contextual feasibility analysis, not a matched-information temporal ablation.", "",
              "The current state space is 0–9. Higher observed values block the complete pilot without",
              "clipping, dropping a case or selecting replacements. Larger state spaces need a separate",
              "original-binary numerical audit and decision. The original inputs are still exported intact.", "",
              "Resource measurements cover each JVM process tree only. Keep CTBF distance computation",
              "in any future whole-pipeline comparison. Resume reuses complete independent runs,",
              "not MCMC checkpoints. Interrupted runs restart from zero in a new preserved attempt.", ""]
    if report.get("reason"):
        lines += ["Reason: " + report["reason"], ""]
    return "\n".join(lines)


def write_summary(root: Path, result: dict, complete: bool) -> Path:
    directory = root / "reports"
    directory.mkdir(exist_ok=True)
    numbers = [int(p.stem.split("_")[-1]) for p in directory.glob("report_*.json")]
    path = directory / f"report_{max(numbers, default=0) + 1:03d}.json"
    write_json(path, result)
    with path.with_suffix(".md").open("x", encoding="utf-8") as handle:
        handle.write(report_markdown(result))
    if complete:
        write_json(root / "_COMPLETE.json", {"report": str(path.relative_to(root)),
                                           "sha256": sha256(path),
                                           "markdown_sha256": sha256(path.with_suffix(".md"))})
    return path


def run(bank: Path, bio: Path, root: Path, *, check: bool, parallel: int, resume: bool) -> int:
    import psutil

    bank, bio, root = (p.expanduser().resolve() for p in (bank, bio, root))
    output_guard(root, bank, bio)
    cases, protected = load_cases(bank)
    backend = load_backend(bio)
    protected.update(backend_snapshot(backend))
    expected = contract(cases, protected, check, parallel)
    prepare(root, cases, expected, backend, resume)
    lock = root / "_RUNNING.json"
    if lock.exists():
        if backend.pilot.process_alive(read_json(lock)):
            raise ValueError("another pilot controller is running")
        lock.rename(root / f"stale-lock-{time.time_ns()}.json")
    # O_EXCL prevents a concurrent controller from acquiring this output root.
    write_json(lock, {"pid": os.getpid(), "create_time": psutil.Process().create_time()})
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    result = {"schema_version": SCHEMA, "status": "blocked", "bounded_check": check,
              "cases": expected["cases"], "technical_runs": [], "pilot_runs": [],
              "tree_agreement": {}, "convergence_established": False,
              "accuracy_metrics_computed": False, "protected_files_unchanged": False}
    try:
        if (root / "_COMPLETE.json").exists():
            marker = read_json(root / "_COMPLETE.json")
            path = safe_child(root, marker["report"])
            if (sha256(path) != marker["sha256"]
                    or sha256(path.with_suffix(".md")) != marker["markdown_sha256"]):
                raise ValueError("completed report changed")
            stored = read_json(path)
            for previous_run in stored["technical_runs"] + stored["pilot_runs"]:
                case = next(c for c in cases if c["case_id"] == previous_run["case_id"])
                directory, verified = existing_attempt(backend,
                    root / previous_run["phase"] / case["case_id"] / f"seed_{previous_run['seed']}",
                    case, previous_run["phase"], previous_run["seed"])
                if verified != previous_run or str(directory) != previous_run["attempt"]:
                    raise ValueError("completed attempt differs from final report")
            backend.pilot.verify_seal(root / "numeric_gate")
            verify_hashes(protected)
            print("complete: existing lossless pilot verified; no inference rerun")
            print("report: " + str(path.with_suffix(".md")))
            return 0
        if any(c["input_summary"]["status"] != "compatible" for c in cases):
            result["reason"] = "copy number outside 0–9; all inputs preserved, no inference or replacement"
        else:
            numeric_root = root / "numeric_gate"
            if numeric_root.exists():
                backend.pilot.verify_seal(numeric_root)
                numeric = read_json(numeric_root / "report.json")
            else:
                numeric = numerical_control(backend, numeric_root)
            result["numerical_control"] = numeric
            if not numeric["passes_observed_range"]:
                result["reason"] = "unchanged released binary failed arithmetic checks; no MCMC"
            else:
                selected, seeds = (cases[:1], SEEDS[:1]) if check else (cases, SEEDS)
                technical = run_phase(backend, root, selected, "technical", seeds, parallel, stop)
                result["technical_runs"] = technical
                expected_count = len(selected) * len(seeds)
                if (len(technical) == expected_count
                        and all(r["status"] == "technical_complete" for r in technical)
                        and not stop.is_set()):
                    if check:
                        result["status"] = "bounded_check_complete_not_convergence"
                    else:
                        runs = run_phase(backend, root, cases, "pilot", SEEDS, parallel, stop)
                        result["pilot_runs"] = runs
                        result["tree_agreement"] = pair_diagnostics(backend, runs)
                        if (len(runs) == expected_count
                                and all(r["status"] == "technical_complete" for r in runs)):
                            result["status"] = "pilot_complete_pending_convergence_review"
                        else:
                            result["reason"] = "pilot incomplete or failed; review retained attempts"
                else:
                    result["reason"] = "technical phase incomplete or failed; no long runs started"
        verify_hashes(protected)
        result["protected_files_unchanged"] = True
        complete = result["status"] in ("bounded_check_complete_not_convergence",
                                          "pilot_complete_pending_convergence_review")
        path = write_summary(root, result, complete)
        successes = sum(r["status"] == "technical_complete" for r in result["pilot_runs"])
        print(f"complete: NestedBD-Long lossless pilot {result['status']}; {successes}/6 pilot runs; "
              "100 bins retained; protected inputs unchanged; no accuracy scoring")
        print("report: " + str(path.with_suffix(".md")))
        return 0 if complete else 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "run"))
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--bio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--parallel-runs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "check" and args.output_root is None:
        parser.error("check requires a separate --output-root")
    try:
        return run(args.bank_root, args.bio_root, args.output_root or DEFAULT_OUTPUT,
                   check=args.stage == "check", parallel=args.parallel_runs, resume=args.resume)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
