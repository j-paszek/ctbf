"""Uniform 10M adapter without modifying the source-pinned 2M runners.

Inference delegates to the validated 10M budget runner. Only summary sample
counts and its operational timeout differ from the original topology adapter.
No imported module's settings are mutated.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess

from algorithm_evaluation import v5_nestedbd_long_budget as budget
from algorithm_evaluation.paper_pipeline_runner import serialize_tree


pilot, benchmark, topology = budget.pilot, budget.benchmark, budget.topology
SETTINGS = dict(budget.SETTINGS)
LABEL = "10M"
ITERATIONS = SETTINGS["long_iterations"]
SUMMARY_KEYS = (
    "summary", "criterion", "exact_score_tie", "burnin_percent", "total_samples",
    "discarded_samples", "retained_samples", "log_every", "stream", "heights",
    "sampled_ancestor_projection", "ambiguous_sampled_ancestors",
    "ordinary_zero_length_internal_edges", "root", "recurrence", "treeannotator_timeout_seconds",
)
SUMMARY = {key: topology.RULES[key] for key in SUMMARY_KEYS}
SUMMARY.update(total_samples=ITERATIONS // SETTINGS["log_every"] + 1,
               treeannotator_timeout_seconds=900)
SUMMARY["discarded_samples"] = SUMMARY["total_samples"] * SUMMARY["burnin_percent"] // 100
SUMMARY["retained_samples"] = SUMMARY["total_samples"] - SUMMARY["discarded_samples"]
REUSED_PROVENANCE = "reused_10m_budget_diagnostic"
NEW_PROVENANCE = "new_10m_extension"


def validate_run(case: dict, seed: int, run: dict) -> None:
    expected = budget.run_identity(case, "long", seed)
    if (any(run.get(key) != value for key, value in expected.items())
            or run.get("status") != "technical_complete"):
        raise ValueError("Expected the exact completed 10M run, not a mixed-budget result.")


def outcome_for(case: dict, seed: int, run: dict, *, reused=False) -> dict:
    provenance = REUSED_PROVENANCE if reused else NEW_PROVENANCE
    result = {"case_id": case["case_id"], "seed": seed, "provenance": provenance}
    if run["status"] == "technical_complete":
        validate_run(case, seed, run)
        result.update(status="success", run=run)
    else:
        result.update(status="failure", stage=run["phase"], reason=run["status"], attempt=run["attempt"])
    return result


def load_reused(root: Path, source_root: Path, bio: Path, cases: list[dict], backend,
                protected: dict) -> list[dict]:
    """Follow the sealed eight-run diagnostic, never select by accuracy/ESS."""
    marker_path, contract_path = root / "_COMPLETE.json", root / "analysis_contract.json"
    marker = pilot.read_json(marker_path)
    if marker.get("schema_version") != budget.SCHEMA:
        raise ValueError("Expected the completed 10M budget diagnostic.")
    protected[str(marker_path)] = pilot.sha256(marker_path)
    budget._bound(contract_path, marker.get("contract_sha256"), protected)
    contract = pilot.read_json(contract_path)
    if (contract.get("rules") != budget.RULES
            or Path(contract["source_root"]).resolve() != source_root
            or contract.get("bio_root") != str(bio)
            or contract.get("runtime") != budget.recovery.runtime_identity()):
        raise ValueError("10M diagnostic source, runtime or configuration differs.")
    for path, digest in contract["protected_sha256"].items():
        if Path(path).suffix == ".py":
            budget._bound(Path(path), digest, protected)
    report_path = pilot.safe_child(root, marker["report_directory"]) / "report.json"
    budget._bound(report_path, marker.get("report_sha256"), protected)
    report = pilot.read_json(report_path)
    declared = {c["case_id"]: c for c in contract["cases"]}
    selected = {c["case_id"]: c for c in cases}
    if (len(declared) != len(contract["cases"]) or len(selected) != len(cases)
            or set(declared) != set(budget.selected_ids()) or set(selected) != set(declared)):
        raise ValueError("The eight reused 10M runs must be exactly the first two histories.")
    reported = {(r["case_id"], r["seed"]): r for r in report["new_runs"]}
    expected = {(c["case_id"], s["seed"]) for c in cases for s in c["seeds"]}
    if len(reported) != len(report["new_runs"]) or set(reported) != expected:
        raise ValueError("Missing, duplicate or undeclared reusable 10M run.")
    outcomes = []
    with budget.recovery.source_read_lock(root):
        for case in cases:
            original = declared[case["case_id"]]
            for key in ("input_path", "input_sha256", "generations", "seeds", "block_index", "height", "condition_id"):
                if case.get(key) != original.get(key):
                    raise ValueError("Reusable 10M input identity differs: " + key)
            budget._bound(Path(case["input_path"]), case["input_sha256"], protected)
            for seed in case["seeds"]:
                unit = root / "inference" / case["case_id"] / f"seed_{seed['seed']}" / "long"
                directory, run = budget.choose_attempt(backend, unit, case, "long", seed["seed"])
                if run is None:
                    raise ValueError("A reused 10M run is unfinished; never replace it here.")
                validate_run(case, seed["seed"], run)
                if run != reported[case["case_id"], seed["seed"]] or run["attempt"] != str(directory):
                    raise ValueError("Reusable 10M report differs from the sealed attempt.")
                seal = directory / "_SEALED.json"
                budget._bound(seal, marker["artifacts"].get(str(seal)), protected)
                # The complete attempt seal was verified by choose_attempt.
                # Bind consumed members as well for the final read-only audit.
                for name in ("report.json", "input.xml", "shared_pilot.log", "shared_pilot.trees", "clade_windows.json"):
                    path = directory / name
                    budget._bound(path, marker["artifacts"].get(str(path)), protected)
                outcomes.append(outcome_for(case, seed["seed"], run, reused=True))
    return outcomes


def inference_task(backend, output: Path, case: dict, seed: int, stop):
    if stop.is_set():
        return None
    if case["eligibility"] != "compatible":
        return {"case_id": case["case_id"], "seed": seed, "status": "ineligible",
                "reason": case["eligibility"], "provenance": NEW_PROVENANCE}
    run = budget.inference_task(backend, output, case, seed, stop)
    return None if run is None else outcome_for(case, seed, run)


def cached_inference(backend, output: Path, case: dict, seed: int) -> dict:
    """Report-only path: no directory creation and no inference execution."""
    if case["eligibility"] != "compatible":
        return {"case_id": case["case_id"], "seed": seed, "status": "ineligible",
                "reason": case["eligibility"], "provenance": NEW_PROVENANCE}
    for phase in ("technical", "long"):
        unit = output / "inference" / case["case_id"] / f"seed_{seed}" / phase
        _directory, run = budget.choose_attempt(backend, unit, case, phase, seed)
        if run is None:
            raise ValueError("NestedBD 10M inference unfinished; report will not start it.")
        if run["status"] != "technical_complete":
            return outcome_for(case, seed, run)
    return outcome_for(case, seed, run)


def validate_sampled_summary(source: Path, summary: Path, records: list[dict]) -> dict:
    """Original mapping/audit, with the exact 10M retained-sample window."""
    trees = list(topology.nexus_trees(summary))
    if len(trees) != 1:
        raise ValueError("TreeAnnotator must emit exactly one summary tree")
    key = topology.topology_key(trees[0][1])
    expected = Counter(record["taxon"] for record in records)
    first_match, matches, count = None, 0, 0
    for count, (name, tree) in enumerate(topology.nexus_trees(source), 1):
        if name != f"STATE_{(count - 1) * SUMMARY['log_every']}":
            raise ValueError("10M source tree states are incomplete or misaligned")
        if Counter(topology.tip_labels(tree)) != expected:
            raise ValueError("source tree taxon coverage differs")
        if count > SUMMARY["discarded_samples"] and topology.topology_key(tree) == key:
            first_match = first_match or name
            matches += 1
    if count != SUMMARY["total_samples"] or not matches:
        raise ValueError("summary topology absent from the complete retained 10M cold stream")
    return {"samples": count, "discarded": SUMMARY["discarded_samples"],
            "retained": count - SUMMARY["discarded_samples"],
            "matching_retained_topologies": matches,
            "first_retained_state_with_summary_topology": first_match,
            "matching_state_is_not_a_claim_of_exact_selected_height_sample": True}


def native_summary(backend, source: Path, directory: Path) -> dict:
    output = directory / "mcc.tree"
    command = [*backend.runtime.java_prefix(backend.runtime.RUNTIME),
               "beast.app.treeannotator.TreeAnnotator", "-burnin", str(SUMMARY["burnin_percent"]),
               "-heights", SUMMARY["heights"], str(source), str(output)]
    pilot.write_json(directory / "command.json", command)
    with (directory / "stdout.txt").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
        completed = subprocess.run(command, cwd=directory, stdout=stdout, stderr=stderr,
                                   timeout=SUMMARY["treeannotator_timeout_seconds"], check=False)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise ValueError("native TreeAnnotator failed; inspect preserved stdout/stderr")
    return {"command": command, "returncode": completed.returncode,
            "summary_sha256": pilot.sha256(output)}


def summarize_external(backend, bank: Path, output: Path, case: dict,
                       replicate: int, outcome: dict) -> dict:
    result = benchmark.base_record(case, "NestedBD-Long", replicate=replicate, seed=outcome["seed"])
    result.update(inference_budget_iterations=ITERATIONS, summary_rules=SUMMARY,
                  inference_provenance=outcome["provenance"], input_sha256=case.get("input_sha256"))
    if outcome["status"] == "success":
        validate_run(case, outcome["seed"], outcome["run"])
        result["source_inference_attempt"] = outcome["run"]["attempt"]
    directory, cached = topology._attempt(output / "summaries" / case["case_id"] / f"seed_{outcome['seed']}")
    if cached is not None:
        for key in result:
            if key not in {"status", "grf", "failure"} and cached.get(key) != result[key]:
                raise ValueError("Cached 10M external summary identity differs.")
        return cached
    result["attempt"] = str(directory)
    if outcome["status"] != "success":
        result.update(status=outcome["status"], failure={"stage": outcome.get("stage", "eligibility"),
                      "code": outcome["reason"], "inference_attempt": outcome.get("attempt")})
    else:
        run = outcome["run"]
        stage = "summary_and_projection"
        result["diagnostics"] = {"convergence_established": False,
            "scalar_windows": run["scalar_windows"], "inference_execution": run["execution"],
            "inference_attempt": run["attempt"]}
        try:
            source = Path(run["attempt"]) / SUMMARY["stream"]
            result["summary_execution"] = native_summary(backend, source, directory)
            result["sample_audit"] = validate_sampled_summary(source, directory / "mcc.tree", case["records"])
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
                result.update(status="success", grf=native["metrics"]["grf"], tree_summary=benchmark.tree_summary(tree))
        except Exception as error:
            result["failure"] = {"stage": stage, "code": type(error).__name__, "message": str(error)}
    pilot.write_json(directory / "result.json", result)
    topology._seal(directory)
    return result
