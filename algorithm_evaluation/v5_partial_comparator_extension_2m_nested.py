"""Two-million-iteration adapter for the 100-history partial comparator.

This module keeps the original calibrated 2M NestedBD-Long configuration and
the original topology-summary rules. It is deliberately separate from the
uniform-10M adapter so that the two budgets can never be mixed in one output
root.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from algorithm_evaluation import v5_nestedbd_long_benchmark as benchmark
from algorithm_evaluation import v5_nestedbd_long_clock_calibration as calibration
from algorithm_evaluation import v5_nestedbd_long_topology_comparison as topology
from algorithm_evaluation.paper_pipeline_runner import serialize_tree


pilot = calibration.pilot
SETTINGS = dict(calibration.SETTINGS)
LABEL = "2M"
ITERATIONS = SETTINGS["pilot_iterations"]
SUMMARY_KEYS = (
    "summary", "criterion", "exact_score_tie", "burnin_percent", "total_samples",
    "discarded_samples", "retained_samples", "log_every", "stream", "heights",
    "sampled_ancestor_projection", "ambiguous_sampled_ancestors",
    "ordinary_zero_length_internal_edges", "root", "recurrence",
    "treeannotator_timeout_seconds",
)
SUMMARY = {key: topology.RULES[key] for key in SUMMARY_KEYS}
SUMMARY.update(
    total_samples=ITERATIONS // SETTINGS["log_every"] + 1,
    treeannotator_timeout_seconds=topology.RULES["treeannotator_timeout_seconds"],
)
SUMMARY["discarded_samples"] = SUMMARY["total_samples"] * SUMMARY["burnin_percent"] // 100
SUMMARY["retained_samples"] = SUMMARY["total_samples"] - SUMMARY["discarded_samples"]
REUSED_PROVENANCE = None
NEW_PROVENANCE = "new_2m_extension"


def choose_attempt(backend, unit: Path, case: dict, phase: str, seed: int):
    """Reuse sealed terminal outcomes; only interruptions may start over.

    The older calibration selector deliberately stops on scientific failures
    and retries resource failures. Neither policy implements this extension's
    fixed-budget contract. Keep the calibrated executor and its output intact.
    """
    expected = calibration.run_identity(case, phase, seed, benchmark.CLOCK)
    attempts = sorted(unit.glob("attempt_*"))
    terminal = []
    for directory in attempts:
        started = directory / "execution_started.json"
        if started.is_file() and backend.pilot.process_alive(pilot.read_json(started)):
            raise ValueError("An earlier BEAST process is still running: " + str(directory))
        report_path = directory / "report.json"
        seal_path = directory / "_SEALED.json"
        if not seal_path.is_file():
            if report_path.exists():
                raise ValueError("Unsealed existing report requires review: " + str(report_path))
            continue  # retain incomplete files; never restore a native checkpoint
        topology._verify_seal(directory)
        seal = pilot.read_json(seal_path)
        if "report.json" not in seal:
            raise ValueError("Sealed 2M attempt lacks its report checksum.")
        record = pilot.read_json(report_path)
        if (any(record.get(key) != value for key, value in expected.items())
                or record.get("attempt") != str(directory)):
            raise ValueError("Cached 2M attempt identity or configuration differs.")
        if record.get("status") not in {
                "technical_complete", "failed", "timeout", "rss_limit",
                "capture_limit", "interrupted"}:
            raise ValueError("Unrecognized 2M attempt status requires review.")
        xml_path = directory / "input.xml"
        if xml_path.exists() or record["status"] == "technical_complete":
            if "input.xml" not in seal:
                raise ValueError("Sealed 2M attempt lacks its XML checksum.")
            actual = ET.parse(xml_path)
            pilot.verify_xml_input(actual, case)
            _, wanted, _ = calibration.make_xml(backend, case, phase, benchmark.CLOCK)
            if ET.tostring(actual.getroot()) != ET.tostring(wanted.getroot()):
                raise ValueError("Cached 2M XML differs from the fixed configuration.")
        if record["status"] != "interrupted":
            terminal.append((directory, record))
    if len(terminal) > 1:
        raise ValueError("Multiple terminal results for one 2M seed and phase.")
    if terminal:
        return terminal[0]
    number = max((int(path.name.removeprefix("attempt_")) for path in attempts), default=0) + 1
    return unit / f"attempt_{number:03d}", None


def validate_run(case: dict, seed: int, run: dict) -> None:
    expected = calibration.run_identity(case, "pilot", seed, benchmark.CLOCK)
    if (any(run.get(key) != value for key, value in expected.items())
            or run.get("status") != "technical_complete"):
        raise ValueError("Expected the exact completed 2M run, not a mixed-budget result.")


def outcome_for(case: dict, seed: int, run: dict) -> dict:
    result = {"case_id": case["case_id"], "seed": seed, "provenance": NEW_PROVENANCE}
    if run["status"] == "technical_complete":
        validate_run(case, seed, run)
        result.update(status="success", run=run)
    else:
        result.update(status="failure", stage=run.get("phase", "pilot"),
                      reason=run["status"], attempt=run["attempt"])
    return result


def inference_task(backend, output: Path, case: dict, seed: int, stop):
    if stop.is_set():
        return None
    if case["eligibility"] != "compatible":
        return {"case_id": case["case_id"], "seed": seed, "status": "ineligible",
                "reason": case["eligibility"], "provenance": NEW_PROVENANCE}
    for phase in ("technical", "pilot"):
        unit = output / "inference" / case["case_id"] / f"seed_{seed}" / phase
        directory, run = choose_attempt(backend, unit, case, phase, seed)
        if run is None:
            if stop.is_set():
                return None
            run = calibration.execute_attempt(
                backend, directory, case, phase, seed, benchmark.CLOCK, stop)
        if run["status"] == "interrupted" or stop.is_set():
            return None
        if run["status"] != "technical_complete":
            return {"case_id": case["case_id"], "seed": seed, "status": "failure",
                    "stage": phase, "reason": run["status"],
                    "attempt": str(directory), "provenance": NEW_PROVENANCE}
    return outcome_for(case, seed, run)


def cached_inference(backend, output: Path, case: dict, seed: int) -> dict:
    """Report-only path. It never creates an attempt or starts inference."""
    if case["eligibility"] != "compatible":
        return {"case_id": case["case_id"], "seed": seed, "status": "ineligible",
                "reason": case["eligibility"], "provenance": NEW_PROVENANCE}
    for phase in ("technical", "pilot"):
        unit = output / "inference" / case["case_id"] / f"seed_{seed}" / phase
        directory, run = choose_attempt(backend, unit, case, phase, seed)
        if run is None:
            raise ValueError("NestedBD 2M inference unfinished; report will not start it.")
        if run["status"] != "technical_complete":
            return outcome_for(case, seed, run)
    return outcome_for(case, seed, run)


def validate_sampled_summary(source: Path, summary: Path, records: list[dict]) -> dict:
    """Require the complete 2M tree stream and the fixed retained window."""
    trees = list(topology.nexus_trees(summary))
    if len(trees) != 1:
        raise ValueError("TreeAnnotator must emit exactly one summary tree")
    key = topology.topology_key(trees[0][1])
    expected = Counter(record["taxon"] for record in records)
    first_match, matches, count = None, 0, 0
    for count, (name, tree) in enumerate(topology.nexus_trees(source), 1):
        if name != f"STATE_{(count - 1) * SUMMARY['log_every']}":
            raise ValueError("2M source tree states are incomplete or misaligned")
        if Counter(topology.tip_labels(tree)) != expected:
            raise ValueError("source tree taxon coverage differs")
        if count > SUMMARY["discarded_samples"] and topology.topology_key(tree) == key:
            first_match = first_match or name
            matches += 1
    if count != SUMMARY["total_samples"] or not matches:
        raise ValueError("summary topology absent from the complete retained 2M cold stream")
    return {"samples": count, "discarded": SUMMARY["discarded_samples"],
            "retained": count - SUMMARY["discarded_samples"],
            "matching_retained_topologies": matches,
            "first_retained_state_with_summary_topology": first_match,
            "matching_state_is_not_a_claim_of_exact_selected_height_sample": True}


def native_summary(backend, source: Path, directory: Path) -> dict:
    output = directory / "mcc.tree"
    command = [*backend.runtime.java_prefix(backend.runtime.RUNTIME),
               "beast.app.treeannotator.TreeAnnotator", "-burnin",
               str(SUMMARY["burnin_percent"]), "-heights", SUMMARY["heights"],
               str(source), str(output)]
    pilot.write_json(directory / "command.json", command)
    with (directory / "stdout.txt").open("x") as stdout, \
            (directory / "stderr.txt").open("x") as stderr:
        completed = subprocess.run(command, cwd=directory, stdout=stdout, stderr=stderr,
                                   timeout=SUMMARY["treeannotator_timeout_seconds"], check=False)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise ValueError("native TreeAnnotator failed; inspect preserved stdout/stderr")
    return {"command": command, "returncode": completed.returncode,
            "summary_sha256": pilot.sha256(output)}


def summarize_external(backend, bank: Path, output: Path, case: dict,
                       replicate: int, outcome: dict) -> dict:
    if outcome["status"] == "success":
        validate_run(case, outcome["seed"], outcome["run"])
    result = benchmark.base_record(case, "NestedBD-Long", replicate=replicate,
                                   seed=outcome["seed"])
    result.update(inference_budget_iterations=ITERATIONS, summary_rules=SUMMARY,
                  inference_provenance=outcome["provenance"],
                  input_sha256=case.get("input_sha256"))
    directory, cached = topology._attempt(
        output / "summaries" / case["case_id"] / f"seed_{outcome['seed']}")
    if cached is not None:
        for key in result:
            if key not in {"status", "grf", "failure"} and cached.get(key) != result[key]:
                raise ValueError("Cached 2M external summary identity differs.")
        return cached
    result["attempt"] = str(directory)
    if outcome["status"] != "success":
        result.update(status=outcome["status"], failure={
            "stage": outcome.get("stage", "eligibility"), "code": outcome["reason"],
            "inference_attempt": outcome.get("attempt")})
    else:
        run = outcome["run"]
        stage = "summary_and_projection"
        result["diagnostics"] = {
            "convergence_established": False,
            "scalar_windows": run["scalar_windows"],
            "inference_execution": run["execution"],
            "inference_attempt": run["attempt"],
        }
        try:
            source = Path(run["attempt"]) / SUMMARY["stream"]
            result["summary_execution"] = native_summary(backend, source, directory)
            result["sample_audit"] = validate_sampled_summary(
                source, directory / "mcc.tree", case["records"])
            parsed = next(topology.nexus_trees(directory / "mcc.tree"))[1]
            tree, result["projection"] = topology.project_summary(parsed, case["records"])
            pilot.write_json(directory / "tree.json", serialize_tree(tree))
            stage = "evaluation"
            native = benchmark.evaluation_for(bank, case, tree)
            pilot.write_json(directory / "evaluation.json", native)
            if native["status"] != "success":
                result["failure"] = native["failure"]
            elif native["inputs"]["observation_label_coverage"]["fraction"] != 1.0:
                result["failure"] = {"stage": stage, "code": "incomplete_coverage"}
            else:
                result.update(status="success", grf=native["metrics"]["grf"],
                              tree_summary=benchmark.tree_summary(tree))
        except Exception as error:
            result["failure"] = {"stage": stage, "code": type(error).__name__,
                                  "message": str(error)}
    pilot.write_json(directory / "result.json", result)
    topology._seal(directory)
    return result
