"""Owner-run, ten-history NestedBD-Long extension of the frozen paper inputs.

This additive comparator is not part of the original registered confirmation.
Selection is fixed before input inspection. Existing evidence and pinned helpers
are read-only. Only the owner-run command starts the forty 2M-iteration runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
from pathlib import Path
import signal
import statistics
import sys
import threading
import xml.etree.ElementTree as ET

import numpy as np

from algorithm_evaluation import v5_confirmation_common as confirmation
from algorithm_evaluation import v5_nestedbd_long_clock_calibration as calibration
from algorithm_evaluation import v5_nestedbd_long_pilot as pilot
from algorithm_evaluation import v5_nestedbd_long_topology_comparison as topology
from algorithm_evaluation.paper_pipeline_runner import deserialize_tree, serialize_tree
from algorithm_evaluation.v5_algorithm_development_common import tree_summary
from algorithm_evaluation.v5_confirmation_run import _load_truth_after_reconstruction
from evaluation_contract import evaluate_tree_pair_result, validate_evaluation_result


SCHEMA = "ctbf-v5-nestedbd-long-ten-history-benchmark-v1"
DEFAULT_CONFIRMATION = pilot.PROJECT / "algorithm_evaluation/results/v5_confirmation/confirmation_v1"
DEFAULT_BANK = DEFAULT_CONFIRMATION / "bank"
DEFAULT_INTERNAL = DEFAULT_CONFIRMATION / "internal"
DEFAULT_OUTPUT = pilot.PROJECT / "algorithm_evaluation/results/v5_nestedbd_long/benchmark_v1"
BLOCKS = tuple(range(10))
CONDITIONS = ("H14_rule_y", "H24_rule_y")
SEED_NAMESPACE = "ctbf-v5-nestedbd-long-ten-history-v1"
CLOCK = {"label": "nominal", "factor": 1.0, "mean_clock_rate": 0.001061875}
RULES = {
    "selection": "confirmation_blocks_001_through_010_before_dimensions_or_scores",
    "block_indices": list(BLOCKS), "conditions": list(CONDITIONS),
    "seed_namespace": SEED_NAMESPACE, "replicates_per_condition": 2,
    "inference": copy.deepcopy(calibration.SETTINGS),
    "summary_and_mapping": copy.deepcopy(topology.RULES),
    "clock": CLOCK, "technical_gate": "10k_full_input_per_case_and_seed",
    "eligibility": "all_original_100_bins_and_occurrences_CN0_through_9",
    "replacement_or_clipping": False,
    "failure_policy": "retain_terminal_failures_do_not_stop_unrelated_cases",
    "resume_policy": "reuse_terminal_results_restart_interruptions_only_from_zero",
    "ctbf_comparators": topology.CTBF_ARMS,
    "ctbf_execution": "reuse_sealed_frozen_records_and_replay_native_evaluation",
    "effect_direction": "CTBF_GRF_minus_NestedBD_GRF",
    "independent_unit": "one_simulated_history",
    "block_effect": "mean_of_four_paired_differences_two_heights_by_two_seeds",
    "missing_data": "no_imputation_report_conditional_complete_pairs_and_blocks",
    "bootstrap": "descriptive_95_percent_percentile_CI_resample_complete_histories",
    "bootstrap_draws": 10000, "bootstrap_seed": 2026092701,
    "win_tie_loss_tolerance": 1e-12,
    "automatic_extension_or_promotion": False, "p_values": False,
    "whole_pipeline_resource_comparison": False,
    "scientific_role": "post_registration_fixed_budget_contextual_comparator",
}
# The inherited summary describes the earlier two development cases. Replace
# only its selection fields, not summary/mapping/metric semantics.
RULES["summary_and_mapping"].update(
    cases="twenty_prespecified_confirmation_conditions",
    seeds="two_unique_prospectively_derived_seeds_per_condition",
    independent_histories=10,
    ctbf_distance="frozen_internal_record_already_uses_exact_aligned_D")


def seed_for(case_id: str, replicate: int) -> int:
    digest = hashlib.sha256(f"{SEED_NAMESPACE}|{case_id}|{replicate}".encode()).hexdigest()
    return int(digest[:12], 16) % 2147483646 + 1


def selected_ids() -> list[str]:
    return [confirmation.confirmation_case_id(block, condition)
            for block in BLOCKS for condition in CONDITIONS]


def checksum_index(root: Path, name: str) -> dict[str, str]:
    """Read an existing small inventory, without scanning its result corpus."""
    result = {}
    for line in (root / name).read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (not separator or len(digest) != 64 or not relative
                or any(c not in "0123456789abcdef" for c in digest)
                or relative in result):
            raise ValueError("malformed or duplicate checksum entry")
        pilot.safe_child(root, relative)
        result[relative] = digest
    return result


def protect_member(root: Path, relative: str, inventory: dict, protected: dict) -> Path:
    path = pilot.safe_child(root, relative)
    if relative not in inventory or pilot.sha256(path) != inventory[relative]:
        raise ValueError("selected checksum-closed member changed: " + str(path))
    protected[str(path)] = inventory[relative]
    return path


def load_selection(bank: Path, internal: Path) -> tuple[list[dict], dict]:
    """Freeze original-order blocks 1--10, inspecting only their observable data.

    Truths, distances and internal records are hashed but not deserialized.
    This validates selected members, not a new audit of the entire 800-case bank.
    """
    bank_inventory = checksum_index(bank, confirmation.BANK_CHECKSUM_NAME)
    run_inventory = checksum_index(internal, confirmation.RUN_CHECKSUM_NAME)
    protected = {str(root / name): pilot.sha256(root / name) for root, name in (
        (bank, confirmation.BANK_CHECKSUM_NAME), (internal, confirmation.RUN_CHECKSUM_NAME))}
    bpath = protect_member(bank, confirmation.BANK_MANIFEST_NAME, bank_inventory, protected)
    rpath = protect_member(internal, confirmation.RUN_MANIFEST_NAME, run_inventory, protected)
    b, r = pilot.read_json(bpath), pilot.read_json(rpath)
    approval = confirmation.read_owner_approval_lock()
    protected[approval["path"]] = approval["sha256"]
    if (approval["status"] != "approved" or b.get("schema_version") != confirmation.BANK_SCHEMA_VERSION
            or r.get("schema_version") != confirmation.RUN_SCHEMA_VERSION
            or b.get("status") != "complete" or r.get("status") != "complete"
            or b.get("block_count") != 100
            or b.get("scientific_role") != "prospectively_locked_independent_confirmation"
            or r.get("scientific_role") != b["scientific_role"]
            or r.get("bank_id") != b.get("bank_id")
            or r.get("bank_checksum_sha256") != protected[str(bank / confirmation.BANK_CHECKSUM_NAME)]):
        raise ValueError("the completed owner-approved paper bank/internal run is required")
    for field, approved in (("design_manifest_sha256", "APPROVED_MANIFEST_SHA256"),
                            ("protocol_sha256", "APPROVED_PROTOCOL_SHA256"),
                            ("source_lock_sha256", "APPROVED_SOURCE_LOCK_SHA256")):
        if b.get(field) != approval[approved] or r.get(field) != approval[approved]:
            raise ValueError("confirmation identity differs: " + field)
    snapshot = protect_member(bank, "owner_approval.snapshot.py", bank_inventory, protected)
    if pilot.sha256(snapshot) != approval["sha256"] or b.get("owner_approval_sha256") != approval["sha256"]:
        raise ValueError("owner approval snapshot differs")
    config_path = protect_member(bank, "simulator_config.json", bank_inventory, protected)
    config = pilot.read_json(config_path)
    if config != pilot.read_json(calibration.REFERENCE_CONFIG):
        raise ValueError("the frozen standard simulator configuration is required")
    protected[str(calibration.REFERENCE_CONFIG)] = pilot.sha256(calibration.REFERENCE_CONFIG)
    if calibration.calibration_moments(config)["nominal_mean_clock"] != CLOCK["mean_clock_rate"]:
        raise ValueError("nominal clock no longer matches the declared simulator approximation")
    cases, seeds = [], []
    for case_id in selected_ids():
        declarations = [x for x in b["condition_inventory"] if x["case_id"] == case_id]
        if len(declarations) != 1:
            raise ValueError("missing/duplicate prespecified condition: " + case_id)
        declaration = declarations[0]
        block = int(case_id.split("-")[1][1:]) - 1
        condition = case_id.split("-", 2)[2]
        height = 14 if condition == CONDITIONS[0] else 24
        generations = [9, 12, 14] if height == 14 else [15, 20, 24]
        if (declaration["block_index"] != block or declaration["condition_id"] != condition
                or declaration["height"] != height or declaration["generations"] != generations):
            raise ValueError("condition coordinates or biopsy schedule differ")
        case = {"case_id": case_id, "block_index": block, "height": height,
                "condition_id": condition, "generations": generations,
                "seeds": [{"replicate": index, "seed": seed_for(case_id, index)} for index in (1, 2)],
                "eligibility": "bank_unavailable", "input_summary": None, "records": [],
                "source_case": None, "internal_records": {}, "bank_failure": declaration.get("failure")}
        seeds += [item["seed"] for item in case["seeds"]]
        for method, arm_id in topology.CTBF_ARMS.items():
            entries = [x for x in r["record_inventory"] if x["case_id"] == case_id and x["arm_id"] == arm_id]
            if len(entries) != 1:
                raise ValueError("missing/duplicate frozen CTBF comparator record")
            entry = entries[0]
            path = protect_member(internal, entry["path"], run_inventory, protected)
            if entry["sha256"] != protected[str(path)]:
                raise ValueError("internal record inventory differs from closure")
            case["internal_records"][method] = str(path)
        if declaration["status"] == "available":
            found = [x for x in b["cases"] if x["case_id"] == case_id]
            if len(found) != 1:
                raise ValueError("available case missing from bank")
            source = found[0]
            for key in ("case_id", "block_index", "condition_id", "height", "generations"):
                if source[key] != case[key]:
                    raise ValueError("available case coordinates differ")
            for field in ("input_path", "metadata_path", "truth_path", "distance_path"):
                protect_member(bank, source[field], bank_inventory, protected)
            path = pilot.safe_child(bank, source["input_path"])
            payload = pilot.read_json(path)
            if (payload.get("case_id") != case_id or payload.get("condition_id") != condition
                    or [level["generation"] for level in payload["levels"]] != generations):
                raise ValueError("observable input identity or schedule differs")
            records = pilot.input_records(payload)
            summary = pilot.compatibility(records)
            case.update(input_path=str(path), input_sha256=pilot.sha256(path), source_case=source,
                        records=records, input_summary=summary, eligibility=summary["status"])
        elif declaration["status"] != "unavailable":
            raise ValueError("unknown bank availability status")
        cases.append(case)
    if len(seeds) != len(set(seeds)):
        raise ValueError("prespecified inference seeds collide")
    return cases, protected


def prior_snapshot(root: Path) -> dict:
    """Preserve the completed pilot and its dependencies without score selection."""
    topology._verify_seal(root)
    contract = pilot.read_json(root / "analysis_contract.json")
    if contract.get("schema_version") != topology.SCHEMA:
        raise ValueError("the completed topology pilot is required")
    protected = dict(contract["protected_sha256"])
    for relative, digest in pilot.read_json(root / "_SEALED.json").items():
        protected[str(pilot.safe_child(root, relative))] = digest
    protected[str(root / "_SEALED.json")] = pilot.sha256(root / "_SEALED.json")
    pilot.verify_hashes(protected)
    return protected


def preparation_text(cases: list[dict]) -> str:
    lines = ["# NestedBD-Long: prespecified ten-history input inventory", "",
             "Blocks 001–010, H14_rule_y/H24_rule_y. Selection precedes dimensions and outcomes.",
             "This is an additive post-registration comparator. Original confirmation evidence is unchanged.",
             "No unsupported case is clipped, trimmed or replaced. No accuracy records are read during preparation.", "",
             "| Case | Biopsy sizes | Occurrences | Distinct states | Bins | CN range | Eligibility |",
             "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for case in cases:
        summary = case["input_summary"] or {}
        lines.append(f"| {case['case_id']} | {summary.get('biopsy_sizes', 'NA')} | "
                     f"{summary.get('occurrences', 'NA')} | {summary.get('distinct_states', 'NA')} | "
                     f"{summary.get('bins', 'NA')} | {summary.get('minimum_cn', 'NA')}–"
                     f"{summary.get('maximum_cn', 'NA')} | {case['eligibility']} |")
    lines += ["", "## Frozen execution", "",
              "Two unique prederived seeds per condition: 40 planned external records.",
              "Each eligible seed has a 10k full-input correctness check before its 2M run.",
              "Clock 0.001061875/generation; 10% burn-in; native SA-aware MCC; original heights.",
              "At most two concurrent four-chain JVMs (eight MCMC chains total).",
              "Per JVM: 2-GiB heap, 4-GiB process-tree RSS, 300 s/short check, 3600 s/2M run.",
              "A timeout is a retained failure, not permission to change the input, budget or seed.",
              "The inference limits allow up to 20 hours for 40 long runs at two-way concurrency,",
              "plus up to 100 minutes of short checks and postprocessing. This is a budget ceiling,",
              "not a runtime prediction. A slow large case does not stop unrelated cases.",
              "Only interruption restarts from zero. Completed failures are not retried.",
              "No empirical rerun, H34, prior/heating tuning, simulator change, or new algorithm.",
              "M0/M2 reuse frozen records. No resource superiority is claimed from cached D.", ""]
    return "\n".join(lines)


def prepare(bank: Path, internal: Path, bio: Path, previous: Path, output: Path,
            *, parallel: int, resume: bool):
    bank, internal, bio, previous, output = (
        path.expanduser().resolve() for path in (bank, internal, bio, previous, output))
    if parallel not in (1, 2):
        raise ValueError("only one or two concurrent JVMs are permitted")
    pilot.output_guard(output, bank, bio)
    for root in (internal, previous, calibration.DEFAULT_OUTPUT.resolve(), pilot.DEFAULT_OUTPUT.resolve()):
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("output overlaps protected evidence")
    cases, protected = load_selection(bank, internal)
    protected.update(prior_snapshot(previous))
    backend = pilot.load_backend(bio)
    protected.update(pilot.backend_snapshot(backend))
    protected.update(topology.source_snapshot())
    contract = {"schema_version": SCHEMA, "rules": RULES, "parallel_runs": parallel,
                "bank_root": str(bank), "internal_root": str(internal), "bio_root": str(bio),
                "previous_pilot": str(previous), "protected_sha256": protected,
                "cases": [{key: value for key, value in case.items() if key != "records"} for case in cases],
                "runtime": {"python": sys.version, "packages": {
                    name: importlib.metadata.version(name) for name in ("numpy", "networkx", "psutil")}},
                "integrity_scope": "selected_closed_members_and_prior_pilot_not_a_full_bank_rescan",
                "truth_or_scores_deserialized_during_preparation": False}
    if output.exists():
        if not resume:
            raise FileExistsError("output exists; use --resume to verify it")
        if pilot.read_json(output / "analysis_contract.json") != contract:
            raise ValueError("prospective benchmark contract, source or runtime changed")
        topology._verify_seal(output / "preparation")
        backend.pilot.verify_seal(output / "inputs")
        for case in cases:
            if case["source_case"] is not None:
                if pilot.sha256(output / "inputs" / case["case_id"] / "input.json") != case["input_sha256"]:
                    raise ValueError("stored input copy differs")
    else:
        output.mkdir(parents=True)
        pilot.write_json(output / "analysis_contract.json", contract)
        (output / "inputs").mkdir()
        for case in cases:
            if case["source_case"] is not None:
                pilot.write_input_copy(output, case)
        backend.pilot.seal(output / "inputs")
        directory = output / "preparation"
        directory.mkdir()
        pilot.write_json(directory / "inventory.json", {"case_count": len(cases),
                         "compatible": sum(case["eligibility"] == "compatible" for case in cases),
                         "planned_external_records": 2 * len(cases), "cases": contract["cases"]})
        with (directory / "inventory.md").open("x", encoding="utf-8") as handle:
            handle.write(preparation_text(cases))
        topology._seal(directory)
    pilot.verify_hashes(protected)
    return cases, contract, backend


@contextmanager
def owner_lock(output: Path):
    """OS lock releases after interruption, with no stale-PID mutation needed."""
    with (output / ".runner.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another benchmark controller holds the output lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def choose_inference_attempt(backend, unit: Path, case: dict, phase: str, seed: int):
    previous = sorted(unit.glob("attempt_*")) if unit.exists() else []
    terminal = []
    identity = calibration.run_identity(case, phase, seed, CLOCK)
    for directory in previous:
        started = directory / "execution_started.json"
        if started.is_file() and backend.pilot.process_alive(pilot.read_json(started)):
            raise ValueError("a previous BEAST process is still running: " + str(directory))
        if not (directory / "_SEALED.json").is_file():
            continue  # preserve interrupted files, never restore native state
        backend.pilot.verify_seal(directory)
        report = pilot.read_json(directory / "report.json")
        if any(report.get(key) != value for key, value in identity.items()):
            raise ValueError("sealed inference attempt identity changed")
        if report["status"] == "technical_complete":
            actual = ET.parse(directory / "input.xml")
            pilot.verify_xml_input(actual, case)
            _, expected, _ = calibration.make_xml(backend, case, phase, CLOCK)
            if ET.tostring(actual.getroot()) != ET.tostring(expected.getroot()):
                raise ValueError("sealed inference XML differs from the fixed configuration")
        if report["status"] != "interrupted":
            terminal.append((directory, report))
    if len(terminal) > 1:
        raise ValueError("more than one terminal outcome for a prespecified seed")
    if terminal:
        return terminal[0]
    number = max((int(path.name.removeprefix("attempt_")) for path in previous), default=0) + 1
    return unit / f"attempt_{number:03d}", None


def inference_task(backend, output: Path, case: dict, seed: int, stop: threading.Event) -> dict | None:
    if stop.is_set():
        return None
    if case["eligibility"] != "compatible":
        return {"case_id": case["case_id"], "seed": seed, "status": "ineligible",
                "reason": case["eligibility"]}
    for phase in ("technical", "pilot"):
        directory, report = choose_inference_attempt(
            backend, output / "inference" / case["case_id"] / f"seed_{seed}" / phase,
            case, phase, seed)
        if report is None:
            if stop.is_set():
                return None
            report = calibration.execute_attempt(backend, directory, case, phase, seed, CLOCK, stop)
        if report["status"] == "interrupted" or stop.is_set():
            return None
        if report["status"] != "technical_complete":
            return {"case_id": case["case_id"], "seed": seed, "status": "failure",
                    "stage": phase, "reason": report["status"], "attempt": str(directory)}
    return {"case_id": case["case_id"], "seed": seed, "status": "success", "run": report}


def numeric_gate(backend, output: Path) -> dict:
    unit = output / "numeric_gate"
    previous = sorted(unit.glob("attempt_*")) if unit.exists() else []
    for directory in previous:
        if (directory / "_SEALED.json").is_file():
            backend.pilot.verify_seal(directory)
            return pilot.read_json(directory / "report.json")
    unit.mkdir(exist_ok=True)
    directory = unit / f"attempt_{len(previous) + 1:03d}"
    return pilot.numerical_control(backend, directory)


def evaluation_for(bank: Path, case: dict, tree) -> dict:
    truth = _load_truth_after_reconstruction(bank, case["source_case"])
    labels = sorted({row["state_label"] for row in case["records"]})
    result = evaluate_tree_pair_result(truth, tree, labels)
    validate_evaluation_result(result)
    return result


def base_record(case: dict, method: str, *, replicate=None, seed=None) -> dict:
    return {"case_id": case["case_id"], "block_index": case["block_index"], "height": case["height"],
            "method": method, "replicate": replicate, "seed": seed, "status": "failure",
            "grf": None, "failure": None}


def frozen_comparator(bank: Path, case: dict, method: str) -> dict:
    result = base_record(case, method)
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
    validate_evaluation_result(native)
    tree = deserialize_tree(stored["tree"])
    replay = evaluation_for(bank, case, tree)
    if (replay != native or native["status"] != "success"
            or native["metrics"]["grf"] != stored["metrics"]["grf"]
            or tree_summary(tree) != stored["tree_summary"]
            or native["inputs"]["observation_label_coverage"]["fraction"] != 1.0):
        raise ValueError("frozen CTBF evaluator replay differs")
    result.update(status="success", grf=native["metrics"]["grf"],
                  tree_summary=stored["tree_summary"], frozen_evaluator_replay_exact=True)
    return result


def summarize_external(backend, bank: Path, output: Path, case: dict,
                       replicate: int, outcome: dict) -> dict:
    result = base_record(case, "NestedBD-Long", replicate=replicate, seed=outcome["seed"])
    directory, cached = topology._attempt(
        output / "summaries" / case["case_id"] / f"seed_{outcome['seed']}")
    if cached is not None:
        for key in ("case_id", "block_index", "height", "method", "replicate", "seed"):
            if cached.get(key) != result[key]:
                raise ValueError("cached external summary identity differs")
        return cached
    result["attempt"] = str(directory)
    if outcome["status"] != "success":
        result.update(status=outcome["status"], failure={"stage": outcome.get("stage", "eligibility"),
                      "code": outcome["reason"], "inference_attempt": outcome.get("attempt")})
    else:
        run = outcome["run"]
        stage = "summary_and_projection"
        # Diagnostics remain available even if summary/import later fails.
        result["diagnostics"] = {"convergence_established": False,
            "scalar_windows": run["scalar_windows"], "inference_execution": run["execution"],
            "inference_attempt": run["attempt"]}
        try:
            source = Path(run["attempt"]) / topology.RULES["stream"]
            result["summary_execution"] = topology.native_summary(backend, source, directory)
            result["sample_audit"] = topology.validate_sampled_summary(source, directory / "mcc.tree", case["records"])
            tree, projection = topology.project_summary(list(topology.nexus_trees(directory / "mcc.tree"))[0][1], case["records"])
            result["projection"] = projection
            pilot.write_json(directory / "tree.json", serialize_tree(tree))
            stage = "evaluation"
            native = evaluation_for(bank, case, tree)
            pilot.write_json(directory / "evaluation.json", native)
            if native["status"] != "success":
                result["failure"] = native["failure"]
            elif native["inputs"]["observation_label_coverage"]["fraction"] != 1.0:
                result["failure"] = {"stage": stage, "code": "incomplete_coverage"}
            else:
                result.update(status="success", grf=native["metrics"]["grf"], tree_summary=tree_summary(tree))
        except Exception as error:
            result["failure"] = {"stage": stage, "code": type(error).__name__, "message": str(error)}
    pilot.write_json(directory / "result.json", result)
    topology._seal(directory)
    return result


def effect_summary(values: list[float]) -> dict:
    tolerance = RULES["win_tie_loss_tolerance"]
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "wins": sum(value > tolerance for value in values),
            "ties": sum(abs(value) <= tolerance for value in values),
            "losses": sum(value < -tolerance for value in values)}


def paired_effects(records: list[dict]) -> dict:
    """Positive means CTBF is better. Histories, not seeds/heights, are units."""
    lookup = {}
    for row in records:
        key = (row["block_index"], row["height"], row["method"], row["replicate"])
        if key in lookup:
            raise ValueError("duplicate paired-effect cell")
        lookup[key] = row
    result = {"direction": RULES["effect_direction"], "per_height_and_seed": [], "history_paired": []}
    for method in topology.CTBF_ARMS:
        differences = {}
        for height in (14, 24):
            for replicate in (1, 2):
                values = []
                for block in BLOCKS:
                    left = lookup[(block, height, method, None)]
                    right = lookup[(block, height, "NestedBD-Long", replicate)]
                    if left["status"] == right["status"] == "success":
                        value = left["grf"] - right["grf"]
                        values.append(value)
                        differences[(block, height, replicate)] = value
                result["per_height_and_seed"].append({"method": method, "height": height,
                    "replicate": replicate, "missing_pairs": len(BLOCKS) - len(values), **effect_summary(values)})
        blocks = [{"block_index": block, "difference": statistics.fmean(
            differences[(block, height, replicate)] for height in (14, 24) for replicate in (1, 2))}
            for block in BLOCKS if all((block, height, replicate) in differences
                for height in (14, 24) for replicate in (1, 2))]
        values = [block["difference"] for block in blocks]
        interval = None
        if len(values) >= 2:
            rng = np.random.default_rng(RULES["bootstrap_seed"])
            indices = rng.integers(0, len(values), size=(RULES["bootstrap_draws"], len(values)))
            draws = np.asarray(values)[indices].mean(axis=1)
            interval = np.quantile(draws, [.025, .975]).tolist()
        result["history_paired"].append({"method": method, **effect_summary(values),
            "missing_histories": len(BLOCKS) - len(values), "blocks": blocks,
            "descriptive_bootstrap_95_percent_interval": interval})
    return result


def clade_agreement(backend, outcomes: list[dict]) -> dict:
    result = {}
    for case_id in selected_ids():
        runs = {row["seed"]: row["run"] for row in outcomes
                if row["case_id"] == case_id and row["status"] == "success"}
        seeds = [seed_for(case_id, replicate) for replicate in (1, 2)]
        if not all(seed in runs for seed in seeds):
            continue
        # Completion order is scheduler-dependent, not the seed-replicate order
        # used by the report's two diagnostic columns.
        windows = [pilot.read_json(Path(runs[seed]["attempt"]) / "clade_windows.json") for seed in seeds]
        normalized = [{int(b): window for b, window in values.items()} for values in windows]
        result[case_id] = backend.audit.review.compare_windows(*normalized)
    return result


def report_text(report: dict) -> str:
    def num(value):
        return "NA" if value is None else f"{value:.6f}"
    lines = ["# NestedBD-Long: ten-history frozen-input contextual comparison", "",
        "Post-registration comparator extension. The original confirmation report is unchanged.",
        "40 external seed-runs were prespecified on 20 conditions from 10 histories.",
        "Finite-budget output accuracy, not converged posterior inference. No favored seed or pooled posterior.",
        "Rooted GRF similarity is higher-is-better and is not a percentage of topology recovered.", "",
        f"External status counts: {report['external_status_counts']}. Protected inputs/evidence unchanged.", "",
        "## Paired effects by height and seed replicate", "",
        "Positive differences and wins favor the named CTBF method over NestedBD-Long.",
        "Each row has at most ten paired histories, not twenty independent MCMC samples.", "",
        "| CTBF method | Height | Seed replicate | Pairs | Missing | Mean difference | Median | W/T/L |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in report["paired_effects"]["per_height_and_seed"]:
        lines.append(f"| {row['method']} | {row['height']} | {row['replicate']} | {row['n']} | "
            f"{row['missing_pairs']} | {num(row['mean'])} | {num(row['median'])} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |")
    lines += ["", "## History-paired descriptive summary", "",
        "Average the four paired GRF differences within each complete history (two depths × two seeds).",
        "This averages scores, not trees or posterior samples. Resample histories for the descriptive interval.",
        "With at most ten histories, intervals are exploratory and may be unstable. No p-values or promotion rule.", "",
        "| CTBF method | Complete histories | Missing | Mean difference | Median | W/T/L | Bootstrap 95% interval |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in report["paired_effects"]["history_paired"]:
        ci = row["descriptive_bootstrap_95_percent_interval"]
        interval = "NA" if ci is None else f"[{num(ci[0])}, {num(ci[1])}]"
        lines.append(f"| {row['method']} | {row['n']} | {row['missing_histories']} | {num(row['mean'])} | "
            f"{num(row['median'])} | {row['wins']}/{row['ties']}/{row['losses']} | {interval} |")
    lines += ["", "## Every prespecified input and both external seeds", "",
        "| Case | Occurrences / states | M0 GRF | M2 GRF | NestedBD seed 1 | NestedBD seed 2 | Absolute seed-score gap |",
        "| --- | --- | ---: | ---: | --- | --- | ---: |"]
    for case in report["cases"]:
        rows = [row for row in report["records"] if row["case_id"] == case["case_id"]]
        ctbf = {row["method"]: row for row in rows if row["method"] in topology.CTBF_ARMS}
        external = sorted((row for row in rows if row["method"] == "NestedBD-Long"), key=lambda row: row["replicate"])
        scores = [num(row["grf"]) if row["status"] == "success" else row["status"] for row in external]
        gap = abs(external[0]["grf"] - external[1]["grf"]) if all(row["status"] == "success" for row in external) else None
        sizes = case["input_summary"] or {}
        lines.append(f"| {case['case_id']} | {sizes.get('occurrences', 'NA')} / {sizes.get('distinct_states', 'NA')} | "
            f"{num(ctbf['M0']['grf'])} | {num(ctbf['M2']['grf'])} | {scores[0]} | {scores[1]} | {num(gap)} |")
    lines += ["", "## External inference diagnostics (primary 10% burn-in)", "",
        "Resource columns describe the original BEAST JVM runs only, not a CTBF speed comparison.",
        "ESS and clade agreement are diagnostics, not automatic convergence certification.", "",
        "| Case | Replicate / seed | Posterior ESS | Likelihood ESS | Height ESS | Posterior half 1 / half 2 | Minutes | Peak MiB |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for row in report["records"]:
        if "diagnostics" not in row:
            continue
        window = row["diagnostics"]["scalar_windows"]["10"]
        stats = window["official_beast_statistics"]
        resource = row["diagnostics"]["inference_execution"]
        lines.append(f"| {row['case_id']} | {row['replicate']} / {row['seed']} | "
            + " | ".join(num(stats[key]["ESS"]) for key in ("posterior", "likelihood", "TreeHeight"))
            + f" | {num(window['posterior_first_half_mean'])} / {num(window['posterior_second_half_mean'])} | "
            f"{num(resource.get('wall_seconds', 0)/60)} | {num(resource.get('peak_rss_bytes', 0)/1024**2)} |")
    lines += ["", "## Between-seed descendant-tip clade diagnostics", "",
        "These retain the previous pilot's descendant-tip clade definition. MCC selection itself is SA-aware.", "",
        "| Case | Shared majority clades | Majority clades (seed 1 / seed 2) | Maximum frequency difference |",
        "| --- | ---: | --- | ---: |"]
    for case_id, windows in report["clade_agreement"].items():
        primary = windows["10"]
        lines.append(f"| {case_id} | {primary['shared_majority_clades']} | "
            f"{primary['majority_clades_per_seed']} | {num(primary['maximum_clade_frequency_difference'])} |")
    lines += ["", "## Limits", "",
        "- Both seed outputs and all failures remain visible. Complete-pair/block effects are conditional on success, with no failure-score imputation.",
        "- Two nested depths are not independent histories. This is a bounded post-registration comparator, not a replacement of the original confirmatory design.",
        "- Preserve all bins/values/observations, recurrent labels, the sampled-ancestor projection and the full true prefix. Do not infer accuracy from clade agreement alone.",
        "- The one-state/all-diploid case, if present, remains included. Input information and tree sizes differ across histories.",
        "- Numerical generation times give NestedBD-Long more information than CTBF biopsy ordering. This is not a biopsy-time ablation.",
        "- Only the mean clock was approximately calibrated. OrigTime=1, unbounded origin/diversification priors, representative-versus-population sampling mismatch and CN9 cap remain limitations.",
        "- M0/M2 records are reused without reconstruction or D computation. No whole-pipeline time/RSS superiority is claimed; a future comparison must include per-condition D cost.",
        "- AD quantities in native evaluator audit files are not endpoints for this partial-tree comparison. No new ancestral CNP labels are assigned to NestedBD latent nodes.", ""]
    failures = [row for row in report["records"] if row["status"] != "success"]
    if failures:
        lines += ["## Retained failures", ""]
        lines += [f"- {row['case_id']} {row['method']} seed {row['seed']}: {row['failure']}" for row in failures]
    return "\n".join(lines) + "\n"


def verify_completed(output: Path, contract: dict) -> dict | None:
    marker_path = output / "_COMPLETE.json"
    if not marker_path.exists():
        return None
    marker = pilot.read_json(marker_path)
    directory = pilot.safe_child(output, marker["report_directory"])
    topology._verify_seal(directory)
    report_path = directory / "report.json"
    if pilot.sha256(report_path) != marker["sha256"]:
        raise ValueError("completed benchmark report changed")
    report = pilot.read_json(report_path)
    pilot.verify_hashes(report["artifact_sha256"])
    pilot.verify_hashes(contract["protected_sha256"])
    return report


def finish_report(backend, bank: Path, output: Path, cases: list[dict], outcomes: list[dict], contract: dict) -> dict:
    lookup = {(row["case_id"], row["seed"]): row for row in outcomes}
    if len(lookup) != len(cases) * 2:
        raise ValueError("not all planned external outcomes are terminal")
    records = []
    for case in cases:
        for method in topology.CTBF_ARMS:
            records.append(frozen_comparator(bank, case, method))
        for seed in case["seeds"]:
            records.append(summarize_external(backend, bank, output, case, seed["replicate"],
                                              lookup[(case["case_id"], seed["seed"])]))
    artifacts = {}
    # Only selected, bounded attempt directories produced by this workflow.
    attempts = {Path(row["attempt"]) for row in records if row.get("attempt")}
    attempts.update(path.parent for path in (output / "numeric_gate").glob("attempt_*/_SEALED.json"))
    for case in cases:
        for seed in case["seeds"]:
            for phase in ("technical", "pilot"):
                unit = output / "inference" / case["case_id"] / f"seed_{seed['seed']}" / phase
                attempts.update(path.parent for path in unit.glob("attempt_*/_SEALED.json"))
    for outcome in outcomes:
        if outcome["status"] == "success":
            attempts.add(Path(outcome["run"]["attempt"]))
        elif outcome.get("attempt"):
            attempts.add(Path(outcome["attempt"]))
    for attempt in attempts:
        topology._verify_seal(attempt)
        artifacts[str(attempt / "_SEALED.json")] = pilot.sha256(attempt / "_SEALED.json")
        artifacts.update({str(pilot.safe_child(attempt, name)): value
                          for name, value in pilot.read_json(attempt / "_SEALED.json").items()})
    pilot.verify_hashes(contract["protected_sha256"])
    report = {"schema_version": SCHEMA, "status": "fixed_budget_benchmark_complete_not_convergence",
        "scientific_role": RULES["scientific_role"], "records": records,
        "cases": contract["cases"], "inference_outcomes": outcomes,
        "independent_histories": len(BLOCKS), "planned_external_runs": len(cases) * 2,
        "external_status_counts": dict(Counter(row["status"] for row in records if row["method"] == "NestedBD-Long")),
        "paired_effects": paired_effects(records), "clade_agreement": clade_agreement(backend, outcomes),
        "convergence_established": False, "protected_files_unchanged": True,
        "artifact_sha256": artifacts}
    parent = output / "reports"
    previous = sorted(parent.glob("report_*")) if parent.exists() else []
    number = max((int(path.name.removeprefix("report_")) for path in previous), default=0) + 1
    directory = parent / f"report_{number:03d}"
    directory.mkdir(parents=True)
    pilot.write_json(directory / "report.json", report)
    with (directory / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(report_text(report))
    with (directory / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
        fields = ["case_id", "block_index", "height", "method", "replicate", "seed", "status", "grf"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    topology._seal(directory)
    pilot.write_json(output / "_COMPLETE.json", {"report_directory": str(directory.relative_to(output)),
                     "sha256": pilot.sha256(directory / "report.json")})
    return report


def run_prepared(backend, bank: Path, output: Path, cases: list[dict], contract: dict,
                 *, check: bool = False) -> dict | None:
    with owner_lock(output):
        if not check:
            complete = verify_completed(output, contract)
            if complete is not None:
                return complete
        numeric = numeric_gate(backend, output)
        if not numeric["passes_observed_range"]:
            raise ValueError("unchanged binary arithmetic gate failed; no inference")
        stop = threading.Event()
        handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if check:
                eligible = [case for case in cases if case["eligibility"] == "compatible"]
                if not eligible:
                    raise ValueError("no eligible input for bounded check")
                # Output-blind smallest/largest occurrence counts exercise edge and size cases.
                selected = {case["case_id"]: case for case in (
                    min(eligible, key=lambda case: (case["input_summary"]["occurrences"], case["case_id"])),
                    max(eligible, key=lambda case: (case["input_summary"]["occurrences"], case["case_id"])))}
                checks = []
                for case in selected.values():
                    seed = case["seeds"][0]["seed"]
                    directory, record = choose_inference_attempt(backend,
                        output / "bounded_check" / case["case_id"], case, "technical", seed)
                    if record is None:
                        record = calibration.execute_attempt(backend, directory, case, "technical", seed, CLOCK, stop)
                    checks.append(record)
                pilot.verify_hashes(contract["protected_sha256"])
                return {"bounded_check": True, "records": checks,
                        "success": all(row["status"] == "technical_complete" for row in checks)}
            outcomes = []
            with ThreadPoolExecutor(max_workers=contract["parallel_runs"]) as pool:
                futures = [pool.submit(inference_task, backend, output, case, row["seed"], stop)
                           for case in cases for row in case["seeds"]]
                try:
                    for future in as_completed(futures):
                        outcome = future.result()
                        if outcome is not None:
                            outcomes.append(outcome)
                            print(f"progress: {len(outcomes)}/{len(futures)} external outcomes terminal; "
                                  f"{outcome['case_id']} seed {outcome['seed']}: {outcome['status']}", flush=True)
                except BaseException:
                    # Integrity/controller errors are not ordinary per-seed failures.
                    # Stop children before executor shutdown waits for worker threads.
                    stop.set()
                    for future in futures:
                        future.cancel()
                    raise
            if stop.is_set():
                pilot.verify_hashes(contract["protected_sha256"])
                print("interrupted: completed attempts preserved; --resume restarts unfinished seeds from zero")
                return None
            # No new accuracy is evaluated until every prespecified external job is terminal.
            return finish_report(backend, bank, output, cases, outcomes, contract)
        finally:
            stop.set()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run"))
    parser.add_argument("--bio-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--internal-root", type=Path, default=DEFAULT_INTERNAL)
    parser.add_argument("--previous-pilot", type=Path, default=topology.DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parallel-runs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cases, contract, backend = prepare(args.bank_root, args.internal_root, args.bio_root,
        args.previous_pilot, args.output_root, parallel=args.parallel_runs, resume=args.resume)
    output = args.output_root.expanduser().resolve()
    if args.command == "prepare":
        print(f"complete: prespecified ten-history input inventory; {len(cases)} conditions; "
              f"{sum(case['eligibility'] == 'compatible' for case in cases)} compatible; "
              f"{2 * len(cases)} planned external runs; no inference")
        print("inventory: " + str(output / "preparation/inventory.md"))
        return 0
    result = run_prepared(backend, Path(contract["bank_root"]), output, cases, contract,
                          check=args.command == "check")
    if result is None:
        return 130
    if args.command == "check":
        print(f"complete: bounded full-input checks; {sum(row['status'] == 'technical_complete' for row in result['records'])}"
              f"/{len(result['records'])} passed; no long runs or accuracy scoring")
        return 0 if result["success"] else 2
    print(f"complete: NestedBD-Long ten-history benchmark; {result['planned_external_runs']} planned external runs; "
          f"{len(result['records'])} comparison rows; external statuses {result['external_status_counts']}; "
          "protected results unchanged; convergence not established")
    marker = pilot.read_json(output / "_COMPLETE.json")
    print("report: " + str(pilot.safe_child(output, marker["report_directory"]) / "report.md"))
    return 0 if result["external_status_counts"] == {"success": result["planned_external_runs"]} else 2


if __name__ == "__main__":
    raise SystemExit(main())
