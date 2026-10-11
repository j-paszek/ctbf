"""Fixed-budget, topology-only comparison of four stored nominal-clock runs.

No MCMC, clock selection, posterior pooling, source edits or new data generation.
The summary/projection contract is written before any truth is deserialized.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import importlib.metadata
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterator

import networkx as nx

from algorithm_evaluation import v5_nestedbd_long_pilot as pilot
from algorithm_evaluation import v5_nestedbd_long_clock_calibration as calibration


SCHEMA = "ctbf-v5-nestedbd-long-topology-comparison-v1"
DEFAULT_OUTPUT = pilot.DEFAULT_OUTPUT.with_name("nestedbd_long_topology_comparison_v1")
CTBF_ARMS = {
    "M0": "classical_partial",
    "M2": "biopsy_guided_top_anticentral_binary_adaptive_median_prior_nn_bottom_deferred_tie",
}
RULES = {
    "clock": "nominal", "mean_clock_rate": 0.001061875,
    "cases": list(calibration.SELECTED_CASES), "seeds": list(pilot.SEEDS),
    "summary": "BEAST_2.6.7_sampled_ancestor_aware_maximum_clade_credibility",
    "criterion": "native_sum_log_clade_credibilities",
    "exact_score_tie": "first_retained_tree_in_native_stream_order",
    "burnin_percent": 10, "total_samples": 2001, "discarded_samples": 200,
    "retained_samples": 1801, "log_every": 1000,
    "stream": "shared_pilot.trees", "heights": "keep",
    "sampled_ancestor_projection": "zero_length_terminal_tip_label_to_parent",
    "ambiguous_sampled_ancestors": "fail_without_repair",
    "ordinary_zero_length_internal_edges": "preserve",
    "root": "preserve_inferred_root_even_if_observed",
    "recurrence": "preserve_occurrences_map_taxa_to_exact_state_labels",
    "truth": "full_original_truth_prefix_no_observation_pruning",
    "reported_metric": "native_rooted_grf_similarity_higher_is_better",
    "ctbf_arms": CTBF_ARMS,
    "ctbf_seed": "original_case_metadata_reconstruction_seed",
    "ctbf_distance": "original_exact_aligned_cached_case_matrix",
    "independent_histories": 1, "inferential_statistics": False,
    "convergence_established": False, "pool_seeds": False,
    "select_best_seed_or_clock": False, "whole_pipeline_speed_comparison": False,
    "treeannotator_timeout_seconds": 180,
}


@dataclass(frozen=True)
class Node:
    label: str | None
    length: float | None
    children: tuple["Node", ...] = ()


def parse_newick(text: str, translation: dict[str, str]) -> Node:
    """Read the numeric/named-tip, annotated BEAST grammar, failing closed.

    Branch lengths, not posterior height annotations, identify sampled ancestors.
    No epsilon contraction or internal-edge contraction is performed.
    """
    if re.search(r"\b(?:NaN|Infinity)\b", text, re.I) or "[&U]" in text.upper():
        raise ValueError("nonfinite or explicitly unrooted tree")
    text = re.sub(r"\[[^\[\]]*\]", "", text)
    if "[" in text or "]" in text:
        raise ValueError("unsupported nested annotation")
    tokens = re.findall(r"'(?:[^']|'')*'|[(),:;]|[^\s(),:;]+", text)
    position = 0

    def visit() -> Node:
        nonlocal position
        children, label = [], None
        if tokens[position] == "(":
            position += 1
            children.append(visit())
            while tokens[position] == ",":
                position += 1
                children.append(visit())
            if tokens[position] != ")" or len(children) != 2:
                raise ValueError("expected native binary sampled-ancestor encoding")
            position += 1
        else:
            token = tokens[position]
            if token in "(),:;":
                raise ValueError("missing taxon")
            token = token[1:-1].replace("''", "'") if token.startswith("'") else token
            label = translation[token] if translation else token
            position += 1
        length = None
        if tokens[position] == ":":
            length = float(tokens[position + 1])
            if not math.isfinite(length) or length < 0:
                raise ValueError("negative or nonfinite branch length")
            position += 2
        return Node(label, length, tuple(children))

    try:
        root = visit()
        if tokens[position:] != [";"]:
            raise ValueError("unexpected internal label or Newick suffix")
    except (IndexError, KeyError) as error:
        raise ValueError("incomplete tree or unknown translated taxon") from error
    return root


def nexus_trees(path: Path) -> Iterator[tuple[str, Node]]:
    """Stream only the pinned tool's NEXUS translation/tree statements."""
    translation, translating, pending = {}, False, ""
    seen_tree = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.lower() == "translate":
                if translation or translating or seen_tree:
                    raise ValueError("repeated or late translation block")
                translating = True
                continue
            if translating:
                if line == ";":
                    translating = False
                    continue
                match = re.fullmatch(r"(\d+)\s+('?[^,;]+'?)[,;]?", line)
                if match is None or match[1] in translation:
                    raise ValueError("invalid NEXUS translation")
                translation[match[1]] = match[2].strip().strip("'\"")
                if line.endswith(";"):
                    translating = False
                continue
            if pending or re.match(r"tree\s+", line, re.I):
                pending += " " + line
                if not pending.endswith(";"):
                    continue
                match = re.fullmatch(r"\s*tree\s+(\S+)\s*=\s*(.+;)", pending, re.I)
                if match is None or len(set(translation.values())) != len(translation):
                    raise ValueError("invalid tree statement or duplicate translation")
                seen_tree = True
                yield match[1], parse_newick(match[2], translation)
                pending = ""
    if pending or translating or not seen_tree:
        raise ValueError("incomplete NEXUS stream")


def tip_labels(node: Node) -> list[str]:
    if not node.children:
        return [node.label]
    return [label for child in node.children for label in tip_labels(child)]


def topology_key(node: Node) -> tuple:
    """Occurrence topology plus direct-ancestor identity, not heights/support."""
    if not node.children:
        return ("tip", node.label, node.length == 0)
    return ("internal", tuple(sorted(topology_key(child) for child in node.children)))


def project_summary(root: Node, records: list[dict]) -> tuple[nx.DiGraph, dict]:
    expected = {record["taxon"]: record for record in records}
    labels = tip_labels(root)
    if len(expected) != len(records) or Counter(labels) != Counter(expected.keys()):
        raise ValueError("summary taxa do not exactly cover input occurrences")
    tree, transfers = nx.DiGraph(), []

    def assign(node_id: str, taxon: str) -> None:
        record = expected[taxon]
        tree.nodes[node_id].update(cell_id=record["state_label"], observed_taxon=taxon,
                                   biopsy_level=record["biopsy_level"])

    def visit(node: Node, parent: str | None) -> str:
        node_id = f"nestedbd_{len(tree):06d}"
        tree.add_node(node_id, cell_id=None)
        if parent is not None:
            if node.length is None:
                raise ValueError("missing nonroot branch length")
            tree.add_edge(parent, node_id, weight=node.length)
        if not node.children:
            assign(node_id, node.label)
            return node_id
        direct = [child for child in node.children if not child.children and child.length == 0]
        if len(direct) > 1:
            raise ValueError("ambiguous multiple zero-length sampled-ancestor tips")
        if direct:
            assign(node_id, direct[0].label)
            transfers.append({"taxon": direct[0].label, "parent_occurrence": node_id})
        for child in node.children:
            if not direct or child is not direct[0]:
                visit(child, node_id)
        return node_id

    root_id = visit(root, None)
    if not nx.is_arborescence(tree):
        raise ValueError("projection is not one rooted arborescence")
    observed = [data["cell_id"] for _, data in tree.nodes(data=True) if data["cell_id"] is not None]
    if Counter(observed) != Counter(record["state_label"] for record in records):
        raise ValueError("projection changed observed state-label multiplicity")
    return tree, {"projection_version": SCHEMA, "sampled_ancestors": transfers,
                  "sampled_ancestor_count": len(transfers), "input_occurrences": len(records),
                  "root_preserved": True, "root_is_observed": tree.nodes[root_id]["cell_id"] is not None,
                  "other_edges_contracted": 0, "state_multiplicity_preserved": True}


def validate_sampled_summary(source: Path, summary: Path, records: list[dict]) -> dict:
    trees = list(nexus_trees(summary))
    if len(trees) != 1:
        raise ValueError("TreeAnnotator must emit exactly one summary tree")
    key = topology_key(trees[0][1])
    expected = Counter(record["taxon"] for record in records)
    matches, count = [], 0
    for count, (name, tree) in enumerate(nexus_trees(source), 1):
        if name != f"STATE_{(count - 1) * RULES['log_every']}":
            raise ValueError("source tree states are incomplete or misaligned")
        if Counter(tip_labels(tree)) != expected:
            raise ValueError("source tree taxon coverage differs")
        if count > RULES["discarded_samples"] and topology_key(tree) == key:
            matches.append(name)
    if count != RULES["total_samples"] or not matches:
        raise ValueError("summary topology absent from the complete retained cold stream")
    return {"samples": count, "discarded": RULES["discarded_samples"],
            "retained": count - RULES["discarded_samples"],
            "matching_retained_topologies": len(matches),
            "first_retained_state_with_summary_topology": matches[0],
            "matching_state_is_not_a_claim_of_exact_selected_height_sample": True}


def native_summary(backend, source: Path, directory: Path) -> dict:
    output = directory / "mcc.tree"
    command = [*backend.runtime.java_prefix(backend.runtime.RUNTIME),
               "beast.app.treeannotator.TreeAnnotator", "-burnin", "10", "-heights", "keep",
               str(source), str(output)]
    pilot.write_json(directory / "command.json", command)
    with (directory / "stdout.txt").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
        completed = subprocess.run(command, cwd=directory, stdout=stdout, stderr=stderr,
                                   timeout=RULES["treeannotator_timeout_seconds"], check=False)
    # Some BEAST 2.6.7 parsing errors return status zero. Require a valid output too.
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise ValueError("native TreeAnnotator failed; inspect preserved stdout/stderr")
    return {"command": command, "returncode": completed.returncode,
            "summary_sha256": pilot.sha256(output)}


def calibration_source(root: Path, cases: list[dict]) -> tuple[dict, dict[str, str]]:
    marker = pilot.read_json(root / "_COMPLETE.json")
    report_path = pilot.safe_child(root, marker["report"])
    if (pilot.sha256(report_path) != marker["sha256"]
            or pilot.sha256(report_path.with_suffix(".md")) != marker["markdown_sha256"]):
        raise ValueError("clock calibration summary seal differs")
    report = pilot.read_json(report_path)
    contract = pilot.read_json(root / "analysis_contract.json")
    if (report.get("schema_version") != calibration.SCHEMA
            or report.get("status") != "clock_sensitivity_complete_pending_review"
            or report.get("bounded_check") is not False
            or contract.get("settings") != calibration.SETTINGS
            or report.get("accuracy_metrics_computed") is not False
            or len(report.get("pilot_runs", [])) != 12
            or any(run["status"] != "technical_complete" for run in report["pilot_runs"])):
        raise ValueError("the completed twelve-run clock sensitivity is required")
    protected = dict(contract["protected_sha256"])
    for path in (root / "_COMPLETE.json", report_path, report_path.with_suffix(".md"),
                 root / "analysis_contract.json"):
        protected[str(path)] = pilot.sha256(path)
    selected = {}
    for case in cases:
        for seed in pilot.SEEDS:
            matches = [run for run in report["pilot_runs"]
                       if run["case_id"] == case["case_id"] and run["seed"] == seed
                       and run["clock"]["label"] == "nominal"]
            if len(matches) != 1:
                raise ValueError("missing or duplicated nominal run")
            run = matches[0]
            if (run["clock"]["mean_clock_rate"] != RULES["mean_clock_rate"]
                    or run["input_sha256"] != case["input_sha256"]
                    or run["config"] != calibration.phase_config(case, "pilot")):
                raise ValueError("selected run clock, input or schedule differs")
            attempt = Path(run["attempt"]).resolve()
            if not attempt.is_relative_to(root):
                raise ValueError("selected run is outside the clock calibration root")
            seal = pilot.read_json(attempt / "_SEALED.json")
            for required in ("shared_pilot.trees", "input.xml", "report.json"):
                if required not in seal:
                    raise ValueError("selected run seal lacks a required artifact")
            protected[str(attempt / "_SEALED.json")] = pilot.sha256(attempt / "_SEALED.json")
            protected.update({str(pilot.safe_child(attempt, name)): value for name, value in seal.items()})
            selected[(case["case_id"], seed)] = run
    pilot.verify_hashes(protected)
    return selected, protected


def source_snapshot() -> dict[str, str]:
    # Import the exact inference/evaluation boundary now, before the contract is saved.
    from algorithm_evaluation import v5_algorithm_development_common  # noqa: F401
    import evaluation_contract  # noqa: F401
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_relative_to(pilot.PROJECT):
                paths.add(path)
    return {str(path): pilot.sha256(path) for path in sorted(paths)}


def _seal(directory: Path) -> None:
    # This is only our newly created, bounded attempt/report directory.
    pilot.write_json(directory / "_SEALED.json", {
        str(path.relative_to(directory)): pilot.sha256(path)
        for path in sorted(directory.rglob("*")) if path.is_file()})


def _verify_seal(directory: Path) -> None:
    seal = pilot.read_json(directory / "_SEALED.json")
    pilot.verify_hashes({str(pilot.safe_child(directory, name)): value for name, value in seal.items()})


def _attempt(unit: Path) -> tuple[Path, dict | None]:
    previous = sorted(unit.glob("attempt_*")) if unit.exists() else []
    completed = [path for path in previous if (path / "_SEALED.json").is_file()]
    if len(completed) > 1:
        raise ValueError("multiple sealed results for one case/method")
    if completed:
        _verify_seal(completed[0])
        return completed[0], pilot.read_json(completed[0] / "result.json")
    numbers = [int(path.name.removeprefix("attempt_")) for path in previous]
    target = unit / f"attempt_{max(numbers, default=0) + 1:03d}"
    target.mkdir(parents=True)
    return target, None


def _evaluate(bank: Path, case_record: dict, tree: nx.DiGraph, labels: list) -> dict:
    from algorithm_evaluation.v5_algorithm_development_common import deserialize_truth, truth_prefix
    from evaluation_contract import evaluate_tree_pair_result, validate_evaluation_result
    truth_payload = pilot.read_json(pilot.safe_child(bank, case_record["truth_path"]))
    if truth_payload.get("block_index") != case_record["block_index"]:
        raise ValueError("truth block identity differs")
    truth = truth_prefix(deserialize_truth(truth_payload), case_record["height"])
    result = evaluate_tree_pair_result(truth, tree, labels)
    validate_evaluation_result(result)
    return result


def execute_arm(backend, bank: Path, case: dict, case_record: dict, method: str,
                run: dict | None, directory: Path) -> dict:
    from algorithm_evaluation import v5_algorithm_development_common as common
    from algorithm_evaluation.paper_pipeline_runner import serialize_tree
    result = {"case_id": case["case_id"], "height": case_record["height"],
              "block_index": case_record["block_index"], "method": method,
              "seed": None if run is None else run["seed"], "attempt": str(directory),
              "status": "failure", "grf": None}
    stage = "reconstruction_or_summary"
    try:
        if run is not None:
            source = Path(run["attempt"]) / RULES["stream"]
            result["summary_execution"] = native_summary(backend, source, directory)
            result["sample_audit"] = validate_sampled_summary(source, directory / "mcc.tree", case["records"])
            tree = list(nexus_trees(directory / "mcc.tree"))[0][1]
            tree, result["projection"] = project_summary(tree, case["records"])
            window = run["scalar_windows"]["10"]
            result["diagnostics"] = {
                "convergence_established": False,
                "scalar_ess": {key: window["official_beast_statistics"][key]["ESS"]
                               for key in ("posterior", "likelihood", "TreeHeight")},
                "posterior_first_half_mean": window["posterior_first_half_mean"],
                "posterior_second_half_mean": window["posterior_second_half_mean"],
                "original_inference_resources": run["execution"],
            }
        else:
            payload = pilot.read_json(Path(case["input_path"]))
            raw_distance = pilot.read_json(pilot.safe_child(bank, case_record["distance_path"]))
            metadata = pilot.read_json(pilot.safe_child(bank, case_record["metadata_path"]))
            if (raw_distance.get("case_id") != case["case_id"]
                    or metadata.get("case_id") != case["case_id"]
                    or metadata.get("height") != case_record["height"]
                    or metadata.get("block_index") != case_record["block_index"]):
                raise ValueError("CTBF assets are not aligned to this case")
            distance = common.deserialize_distance(raw_distance)
            if set(distance.ids) != set(common.observed_labels(payload)):
                raise ValueError("CTBF distance labels are not exactly the observed states")
            tree, _, _, diagnostics = common.reconstruct_development_arm(
                common.resolve_arm_specs([CTBF_ARMS[method]])[0], payload, distance,
                reconstruction_seed=int(metadata["reconstruction_seed"]))
            result["reconstruction_metadata"] = diagnostics
        # No truth data have entered any operation above this boundary.
        pilot.write_json(directory / "tree.json", serialize_tree(tree))
        result["tree_summary"] = common.tree_summary(tree)
        stage = "evaluation"
        evaluation = _evaluate(bank, case_record, tree, sorted({r["state_label"] for r in case["records"]}))
        pilot.write_json(directory / "evaluation.json", evaluation)
        if evaluation["status"] != "success":
            result["failure"] = evaluation["failure"]
        elif evaluation["inputs"]["observation_label_coverage"]["fraction"] != 1.0:
            result["failure"] = {"stage": stage, "code": "incomplete_coverage"}
        else:
            result.update(status="success", grf=evaluation["metrics"]["grf"], failure=None)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        result["failure"] = {"stage": stage, "code": type(error).__name__, "message": str(error)}
    pilot.write_json(directory / "result.json", result)
    _seal(directory)
    return result


def report_markdown(report: dict) -> str:
    lines = ["# NestedBD-Long: fixed-budget pilot topology comparison", "",
             "One development history, two nested depths. No independent-replicate inference.",
             "Both nominal-clock seeds are retained separately. Convergence is not established.",
             "MCC uses 10% burn-in and original sampled heights, with sampled-ancestor-aware BEAST 2.6.7.",
             "No clock/seed/tree was selected using truth or CTBF agreement.", "",
             "M0: pooled partial CNP2CNP-NJ. M2: adaptive/deferred biopsy-guided partial anticentral.",
             "GRF is native rooted multiset-cluster similarity (higher is better), not percent topology recovered.",
             "Truth is the full original prefix. Recurrent state labels and inferred unlabeled nodes are retained.", "",
             "| Input | Method | Seed | Status | GRF | GRF minus M0 | GRF minus M2 | Sampled ancestors |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    def number(value):
        return "NA" if value is None else f"{value:.6f}"
    for row in report["records"]:
        others = {r["method"]: r for r in report["records"] if r["case_id"] == row["case_id"]}
        differences = [None if row["grf"] is None or others[m]["grf"] is None
                       else row["grf"] - others[m]["grf"] for m in CTBF_ARMS]
        lines.append(f"| H{row['height']} | {row['method']} | {row['seed'] or '—'} | {row['status']} | "
                     f"{number(row['grf'])} | {number(differences[0])} | {number(differences[1])} | "
                     f"{row.get('projection', {}).get('sampled_ancestor_count', '—')} |")
    lines += ["", "## Retained convergence diagnostics", "",
              "| Input | Seed | Posterior ESS | Likelihood ESS | Height ESS | Posterior half 1 | Posterior half 2 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["records"]:
        if "diagnostics" in row:
            d = row["diagnostics"]
            values = [d["scalar_ess"][key] for key in ("posterior", "likelihood", "TreeHeight")]
            values += [d["posterior_first_half_mean"], d["posterior_second_half_mean"]]
            lines.append(f"| H{row['height']} | {row['seed']} | " + " | ".join(map(number, values)) + " |")
    lines += ["", "## Scope and limitations", "",
              "- Fixed 2M-iteration output accuracy is assessable despite unresolved mixing. It does not validate posterior support.",
              "- H14/H24 share one source history. MCMC seeds are not independent biological replicates. No p-values or overall winner.",
              "- All 100 bins and 17/60 biopsy-indexed occurrences are unchanged. Cross-biopsy repeated states are not merged.",
              "- NestedBD-Long receives numerical generation times, whereas M2 uses biopsy ordering. This is not a matched time ablation.",
              "- Only the mean clock was calibrated approximately. Separate OrigTime=1, birth-death/sampling priors and CN9 cap remain unchanged.",
              "- CTBF uses cached exact D. No whole-pipeline runtime/RSS comparison or speedup is reported. Such a claim would require D computation.",
              "- The versioned evaluator audit files also contain AD quantities by schema. They are not endpoints for these partial trees.",
              "- Each failed summary, mapping or evaluation remains a failed row. No seed or input replacement.", "",
              f"Protected files unchanged: {report['protected_files_unchanged']}. "
              f"Failures: {report['failures']}/{len(report['records'])}.", ""]
    for row in report["records"]:
        if row["status"] != "success":
            lines.append(f"Failure {row['case_id']} {row['method']} {row['seed']}: {row.get('failure')}")
    return "\n".join(lines) + "\n"


def run(bank: Path, bio: Path, calibration_root: Path, output: Path, *, resume: bool) -> dict:
    bank, bio, calibration_root, output = (path.resolve() for path in (bank, bio, calibration_root, output))
    pilot.output_guard(output, bank, bio)
    for protected_root in (calibration_root, pilot.DEFAULT_OUTPUT.resolve()):
        if output.is_relative_to(protected_root) or protected_root.is_relative_to(output):
            raise ValueError("output overlaps a preceding pilot")
    cases, protected = pilot.load_cases(bank)
    cases = [case for case in cases if case["case_id"] in calibration.SELECTED_CASES]
    selected, source_protected = calibration_source(calibration_root, cases)
    protected.update(source_protected)
    backend = pilot.load_backend(bio)
    protected.update(pilot.backend_snapshot(backend))
    protected.update(source_snapshot())
    bank_cases = {record["case_id"]: record for record in pilot.read_json(bank / "bank_manifest.json")["cases"]}
    for case in cases:
        for field in ("input_path", "distance_path", "metadata_path", "truth_path"):
            path = pilot.safe_child(bank, bank_cases[case["case_id"]][field])
            protected[str(path)] = pilot.sha256(path)
    contract = {"schema_version": SCHEMA, "rules": RULES,
                "bank_root": str(bank), "bio_root": str(bio), "calibration_root": str(calibration_root),
                "protected_sha256": protected,
                "runtime": {"python": sys.version,
                            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "networkx")}}}
    if output.exists():
        if not resume:
            raise FileExistsError("output exists; use --resume to verify/reuse completed postprocessing")
        if pilot.read_json(output / "analysis_contract.json") != contract:
            raise ValueError("comparison contract, source or runtime changed")
        if (output / "_SEALED.json").is_file():
            _verify_seal(output)
            pilot.verify_hashes(protected)
            return pilot.read_json(output / "report.json")
    else:
        output.mkdir(parents=True)
        pilot.write_json(output / "analysis_contract.json", contract)
    records = []
    for case in cases:
        jobs = [(method, None) for method in CTBF_ARMS]
        jobs += [("NestedBD-Long", selected[(case["case_id"], seed)]) for seed in pilot.SEEDS]
        for method, source_run in jobs:
            suffix = method if source_run is None else f"NestedBD-Long_seed_{source_run['seed']}"
            directory, result = _attempt(output / "arms" / case["case_id"] / suffix)
            if result is None:
                result = execute_arm(backend, bank, case, bank_cases[case["case_id"]], method, source_run, directory)
            records.append(result)
            print(f"{case['case_id']} {suffix}: {result['status']}", flush=True)
    pilot.verify_hashes(protected)
    report = {"schema_version": SCHEMA, "status": "fixed_budget_comparison_complete_not_convergence",
              "records": records, "failures": sum(row["status"] != "success" for row in records),
              "protected_files_unchanged": True, "independent_histories": 1,
              "convergence_established": False, "mcmc_executed": False}
    # Preserve an interrupted report write by making a new numbered report directory.
    reports = output / "reports"
    attempts = sorted(reports.glob("report_*")) if reports.exists() else []
    report_dir = reports / f"report_{len(attempts) + 1:03d}"
    report_dir.mkdir(parents=True)
    pilot.write_json(report_dir / "report.json", report)
    (report_dir / "report.md").write_text(report_markdown(report), encoding="utf-8")
    # Only these two convenience paths are written at completion, never overwritten.
    if not (output / "report.json").exists():
        pilot.write_json(output / "report.json", report)
    if not (output / "report.md").exists():
        (output / "report.md").write_text(report_markdown(report), encoding="utf-8")
    if pilot.read_json(output / "report.json") != report:
        raise ValueError("an interrupted final report differs; use a new output root")
    _seal(output)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--bio-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, default=pilot.DEFAULT_BANK)
    parser.add_argument("--calibration-root", type=Path, default=calibration.DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.bank_root, args.bio_root, args.calibration_root, args.output_root, resume=args.resume)
    print(f"complete: fixed-budget topology comparison; {len(report['records'])} arm records; "
          f"{report['failures']} failures; protected results unchanged; no MCMC; convergence not established")
    print("report: " + str(args.output_root.resolve() / "report.md"))
    return 0 if report["failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
