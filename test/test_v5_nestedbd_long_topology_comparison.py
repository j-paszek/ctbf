from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from algorithm_evaluation import v5_nestedbd_long_topology_comparison as c
from algorithm_evaluation import v5_nestedbd_long_pilot as p


def records():
    return [{"taxon": taxon, "state_label": state, "biopsy_level": level,
             "generation": level, "cnp": [2, 2]}
            for taxon, state, level in (("A", "same", 0), ("B", "same", 1), ("C", "different", 1))]


def nexus(path, trees, *, names=None, translation=True):
    header = "#NEXUS\nBegin trees;\n"
    if translation:
        header += "Translate\n1 A,\n2 B,\n3 C\n;\n"
    names = names or [f"STATE_{i * 1000}" for i in range(len(trees))]
    path.write_text(header + "\n".join(f"tree {name} = [&R] {tree}" for name, tree in zip(names, trees)) + "\nEnd;\n")
    return path


def parse(text):
    return c.parse_newick(text, {"1": "A", "2": "B", "3": "C"})


def test_projection_promotes_ancestor_preserves_recurrence_and_root():
    original = parse("((1:0,2:1):2,3:3):0;")
    tree, audit = c.project_summary(original, records())
    assert tree.number_of_nodes() == 4 and nx.is_arborescence(tree)
    assert audit["sampled_ancestor_count"] == 1
    assert audit["other_edges_contracted"] == 0
    by_taxon = {data["observed_taxon"]: node for node, data in tree.nodes(data=True) if "observed_taxon" in data}
    assert tree.has_edge(by_taxon["A"], by_taxon["B"])
    assert tree.nodes[by_taxon["A"]]["cell_id"] == tree.nodes[by_taxon["B"]]["cell_id"] == "same"
    assert original == parse("((1:0,2:1):2,3:3):0;")


def test_observed_root_is_not_replaced_by_artificial_root():
    tree, audit = c.project_summary(parse("(1:0,(2:1,3:1):1):0;"), records())
    root = next(n for n, degree in tree.in_degree() if degree == 0)
    assert tree.nodes[root]["observed_taxon"] == "A"
    assert audit["root_is_observed"] and len(tree) == 4


def test_zero_internal_edges_and_positive_tiny_terminal_edges_are_not_contracted():
    tree, audit = c.project_summary(parse("((1:1e-20,2:1):0,3:1):0;"), records())
    assert len(tree) == 5 and audit["sampled_ancestor_count"] == 0
    assert any(data["weight"] == 0 for _, _, data in tree.edges(data=True))


def test_zero_tip_uses_sampled_length_not_posterior_height_annotation():
    tree, audit = c.project_summary(parse("((1[&height=1.23]:0,2:1)[&height=9.8]:2,3:3):0;"), records())
    assert len(tree) == 4 and audit["sampled_ancestor_count"] == 1


@pytest.mark.parametrize("text", [
    "((1:0,2:0):1,3:1):0;",  # ambiguous sampled ancestor identity
    "((1:0,1:1):1,3:2):0;",  # duplicated input occurrence
    "((1,2:1):1,3:2):0;",  # missing terminal length
])
def test_projection_fails_without_repair(text):
    with pytest.raises(ValueError):
        c.project_summary(parse(text), records())


@pytest.mark.parametrize("text", [
    "((1:-1,2:1):1,3:2);", "((1:NaN,2:1):1,3:2);",
    "((1:1,2:1)[&x=Infinity]:1,3:2);", "((1:1,2:1):1,3:2)",
    "((1:1,2:1)named:1,3:2);", "(1:1,2:1,3:1);",
    "((1:1,2:1):1,4:2);", "[&U] ((1:1,2:1):1,3:2);",
])
def test_parser_refuses_unsupported_or_invalid_trees(text):
    with pytest.raises(ValueError):
        parse(text)


def test_nexus_translation_and_named_native_output(tmp_path):
    numeric = nexus(tmp_path / "numeric.trees", ["((1:0,2:1):2,3:3):0;"])
    named = nexus(tmp_path / "named.tree", ["(('A':0,B:1):2,C:3):0;"], names=["TREE1"], translation=False)
    assert c.topology_key(list(c.nexus_trees(numeric))[0][1]) == c.topology_key(list(c.nexus_trees(named))[0][1])


def test_summary_must_be_in_retained_stream_not_only_burnin(tmp_path, monkeypatch):
    monkeypatch.setitem(c.RULES, "total_samples", 3)
    monkeypatch.setitem(c.RULES, "discarded_samples", 1)
    source = nexus(tmp_path / "source.trees", ["((1:0,2:1):2,3:3):0;",
                                               "((1:1,3:1):2,2:3):0;",
                                               "((1:1,3:1):3,2:4):0;"])
    burned = nexus(tmp_path / "burned.tree", ["((1:0,2:1):2,3:3):0;"], names=["TREE1"])
    retained = nexus(tmp_path / "retained.tree", ["((1:1,3:1):2,2:3):0;"], names=["TREE1"])
    with pytest.raises(ValueError, match="absent"):
        c.validate_sampled_summary(source, burned, records())
    audit = c.validate_sampled_summary(source, retained, records())
    assert audit["retained"] == 2
    assert audit["first_retained_state_with_summary_topology"] == "STATE_1000"


def test_summary_verification_rejects_incomplete_or_shifted_stream(tmp_path, monkeypatch):
    monkeypatch.setitem(c.RULES, "total_samples", 3)
    monkeypatch.setitem(c.RULES, "discarded_samples", 1)
    trees = ["((1:0,2:1):2,3:3):0;"] * 2
    source = nexus(tmp_path / "source.trees", trees)
    summary = nexus(tmp_path / "summary.tree", trees[:1], names=["TREE1"])
    with pytest.raises(ValueError, match="complete"):
        c.validate_sampled_summary(source, summary, records())
    nexus(source, trees, names=["STATE_0", "STATE_2000"])
    with pytest.raises(ValueError, match="misaligned"):
        c.validate_sampled_summary(source, summary, records())


def test_topology_key_distinguishes_sampled_ancestors_but_not_order_or_heights():
    assert c.topology_key(parse("((1:0,2:1):2,3:3):0;")) == c.topology_key(parse("(3:9,(2:7,1:0):2):0;"))
    assert c.topology_key(parse("((1:0,2:1):2,3:3):0;")) != c.topology_key(parse("((1:1,2:0):2,3:3):0;"))


def test_prescribed_summary_command_never_executes_inference(tmp_path, monkeypatch):
    calls = []
    backend = SimpleNamespace(runtime=SimpleNamespace(RUNTIME=tmp_path, java_prefix=lambda _: ["java", "-Xmx2g"]))
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_text("toy output")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(c.subprocess, "run", execute)
    c.native_summary(backend, tmp_path / "cold.trees", tmp_path)
    command, options = calls[0]
    assert command[2:7] == ["beast.app.treeannotator.TreeAnnotator", "-burnin", "10", "-heights", "keep"]
    assert not any(value in command for value in ("-noSA", "-target", "beast.app.BeastMCMC"))
    assert options["timeout"] == 180


def test_zero_exit_without_output_is_failure(tmp_path, monkeypatch):
    backend = SimpleNamespace(runtime=SimpleNamespace(RUNTIME=tmp_path, java_prefix=lambda _: ["java"]))
    monkeypatch.setattr(c.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0))
    with pytest.raises(ValueError, match="failed"):
        c.native_summary(backend, tmp_path / "cold.trees", tmp_path)


def test_attempt_reuses_sealed_result_preserves_interrupted_attempt_and_checks_hashes(tmp_path):
    unit = tmp_path / "unit"
    first, cached = c._attempt(unit)
    assert cached is None
    (first / "partial.txt").write_text("partial")
    second, _ = c._attempt(unit)
    p.write_json(second / "result.json", {"status": "success"})
    c._seal(second)
    assert c._attempt(unit) == (second, {"status": "success"})
    assert (first / "partial.txt").read_text() == "partial"
    (second / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        c._attempt(unit)


def test_typed_summary_failure_never_reads_truth(tmp_path, monkeypatch):
    case = {"case_id": "toy", "records": records()}
    run = {"seed": 42, "attempt": str(tmp_path)}
    case_record = {"height": 14, "block_index": 0}
    def fail(*_):
        raise ValueError("toy summary failure")
    monkeypatch.setattr(c, "native_summary", fail)
    monkeypatch.setattr(c, "_evaluate", lambda *_: pytest.fail("truth accessed after summary failure"))
    result = c.execute_arm(None, tmp_path, case, case_record, "NestedBD-Long", run, tmp_path)
    assert result["status"] == "failure" and result["grf"] is None
    assert result["failure"]["stage"] == "reconstruction_or_summary"
    c._verify_seal(tmp_path)


def test_metric_projection_keeps_recurrent_cluster_multiplicity():
    from evaluation_contract import evaluate_tree_pair_result
    tree, _ = c.project_summary(parse("((1:0,2:1):2,3:3):0;"), records())
    result = evaluate_tree_pair_result(tree, tree, ["same", "different"])
    assert result["status"] == "success" and result["metrics"]["grf"] == 1
    changed = tree.copy()
    node = next(n for n, a in changed.nodes(data=True) if a.get("observed_taxon") == "A")
    changed.nodes[node]["cell_id"] = None
    assert evaluate_tree_pair_result(tree, changed, ["same", "different"])["metrics"]["grf"] < 1


def test_markdown_retains_both_seeds_failures_and_limitations():
    rows = [{"case_id": "toy", "height": 14, "method": m, "seed": s,
             "status": "success" if value is not None else "failure", "grf": value,
             "failure": {"message": "test"} if value is None else None}
            for m, s, value in (("M0", None, .1), ("M2", None, .2),
                                ("NestedBD-Long", 42, .15), ("NestedBD-Long", 43, None))]
    text = c.report_markdown({"records": rows, "failures": 1, "protected_files_unchanged": True})
    assert "42" in text and "43" in text and "NA" in text
    assert "Convergence is not established" in text
    assert "not endpoints" in text and "One development history" in text
    assert "| 0.150000 | 0.050000 | -0.050000 |" in text


@pytest.fixture
def toy_workflow(tmp_path, monkeypatch):
    from test_v5_nestedbd_long_pilot import frozen_bank
    from algorithm_evaluation import v5_algorithm_development_common as common
    from ctbs import DistanceMatrix
    bank = frozen_bank(tmp_path / "bank")
    manifest = p.read_json(bank / "bank_manifest.json")
    for item in manifest["cases"]:
        height = int(item["case_id"].rsplit("H", 1)[1])
        item.update(height=height, distance_path=item["case_id"] + "-D.json",
                    metadata_path=item["case_id"] + "-metadata.json", truth_path="truth.json")
        payload = p.read_json(bank / item["input_path"])
        # The frozen development bank uses integer labels at the reconstruction boundary.
        for level in payload["levels"]:
            for state in level["states"]:
                state["state_label"] = 3 if state["state_label"] == "a" else int(state["state_label"])
        (bank / item["input_path"]).write_text(json.dumps(payload))
        labels = common.observed_labels(payload)
        matrix = [[abs(i - j) for i in range(len(labels))] for j in range(len(labels))]
        p.write_json(bank / item["distance_path"], common.serialize_distance(
            item["case_id"], DistanceMatrix(labels, matrix)))
        p.write_json(bank / item["metadata_path"], {
            "case_id": item["case_id"], "height": height, "block_index": 0,
            "schema_version": common.CASE_METADATA_SCHEMA_VERSION, "reconstruction_seed": 7})
    (bank / "bank_manifest.json").write_text(json.dumps(manifest))
    tree = nx.DiGraph()
    tree.add_node("root", cell_id=3, generation=0)
    for index in range(3):
        tree.add_node(str(index), cell_id=index, generation=index + 1)
        tree.add_edge("root", str(index))
    p.write_json(bank / "truth.json", common.serialize_truth(0, tree))
    cases, _ = p.load_cases(bank)
    cases = cases[:2]
    source = tmp_path / "calibration"
    source.mkdir()
    runs, newicks = [], {}
    monkeypatch.setitem(c.RULES, "total_samples", 11)
    monkeypatch.setitem(c.RULES, "discarded_samples", 1)
    monkeypatch.setitem(c.RULES, "retained_samples", 10)
    for case in cases:
        taxa = [row["taxon"] for row in case["records"]]
        newick = f"(({taxa[0]}:1,{taxa[1]}:1):1,(({taxa[2]}:1,{taxa[3]}:1):1,({taxa[4]}:1,{taxa[5]}:1):1):1):0;"
        for clock in ("lower", "nominal", "higher"):
            for seed in p.SEEDS:
                run = {"case_id": case["case_id"], "seed": seed, "status": "technical_complete",
                       "clock": {"label": clock, "mean_clock_rate": c.RULES["mean_clock_rate"]}}
                if clock == "nominal":
                    attempt = source / case["case_id"] / str(seed)
                    attempt.mkdir(parents=True)
                    path = nexus(attempt / "shared_pilot.trees", [newick] * 11, translation=False)
                    newicks[path] = newick
                    (attempt / "input.xml").write_text("toy input")
                    run.update(attempt=str(attempt), input_sha256=case["input_sha256"],
                               config=c.calibration.phase_config(case, "pilot"), execution={"scope": "toy"},
                               scalar_windows={"10": {
                                   "official_beast_statistics": {key: {"ESS": 3}
                                                                for key in ("posterior", "likelihood", "TreeHeight")},
                                   "posterior_first_half_mean": -10, "posterior_second_half_mean": -9}})
                    p.write_json(attempt / "report.json", run)
                    c._seal(attempt)
                runs.append(run)
    report = {"schema_version": c.calibration.SCHEMA, "bounded_check": False,
              "status": "clock_sensitivity_complete_pending_review", "accuracy_metrics_computed": False,
              "pilot_runs": runs}
    p.write_json(source / "report.json", report)
    (source / "report.md").write_text("toy source report")
    p.write_json(source / "analysis_contract.json", {
        "settings": c.calibration.SETTINGS, "protected_sha256": {}})
    p.write_json(source / "_COMPLETE.json", {
        "report": "report.json", "sha256": p.sha256(source / "report.json"),
        "markdown_sha256": p.sha256(source / "report.md")})
    bio = tmp_path / "bio"
    bio.mkdir()
    monkeypatch.setattr(p, "load_backend", lambda _: None)
    monkeypatch.setattr(p, "backend_snapshot", lambda _: {})
    calls = []
    def summary(_backend, path, directory):
        calls.append(str(path))
        nexus(directory / "mcc.tree", [newicks[path]], names=["TREE1"], translation=False)
        return {"native": "toy"}
    monkeypatch.setattr(c, "native_summary", summary)
    return SimpleNamespace(bank=bank, bio=bio, source=source, cases=cases,
                           output=tmp_path / "comparison", calls=calls)


def test_end_to_end_toy_real_ctbf_evaluator_contract_and_readonly_resume(toy_workflow, monkeypatch):
    s = toy_workflow
    original_evaluate = c._evaluate
    def evaluate(bank, case, tree, labels):
        assert p.read_json(s.output / "analysis_contract.json")["rules"] == c.RULES
        return original_evaluate(bank, case, tree, labels)
    monkeypatch.setattr(c, "_evaluate", evaluate)
    sources_before = c.source_snapshot()
    result = c.run(s.bank, s.bio, s.source, s.output, resume=True)
    assert result["failures"] == 0 and len(result["records"]) == 8
    assert len(s.calls) == 4
    assert c.source_snapshot() == sources_before
    before = {str(path): p.sha256(path) for path in s.output.rglob("*") if path.is_file()}
    assert c.run(s.bank, s.bio, s.source, s.output, resume=True) == result
    assert len(s.calls) == 4
    assert {str(path): p.sha256(path) for path in s.output.rglob("*") if path.is_file()} == before
    with pytest.raises(FileExistsError):
        c.run(s.bank, s.bio, s.source, s.output, resume=False)


def test_resume_refuses_changed_summary_contract(toy_workflow, monkeypatch):
    s = toy_workflow
    c.run(s.bank, s.bio, s.source, s.output, resume=True)
    monkeypatch.setitem(c.RULES, "burnin_percent", 50)
    with pytest.raises(ValueError, match="contract"):
        c.run(s.bank, s.bio, s.source, s.output, resume=True)


def test_completed_failure_is_retained_without_seed_replacement(toy_workflow, monkeypatch):
    s = toy_workflow
    native = c.native_summary
    count = 0
    def summary(*args):
        nonlocal count
        count += 1
        if count == 1:
            raise ValueError("one retained failure")
        return native(*args)
    monkeypatch.setattr(c, "native_summary", summary)
    result = c.run(s.bank, s.bio, s.source, s.output, resume=True)
    assert result["failures"] == 1 and len(result["records"]) == 8
    assert c.run(s.bank, s.bio, s.source, s.output, resume=True) == result
    assert count == 4


def test_source_tampering_and_output_overlap_are_refused(toy_workflow):
    s = toy_workflow
    with pytest.raises(ValueError, match="overlap"):
        c.run(s.bank, s.bio, s.source, s.source / "new", resume=True)
    selected, _ = c.calibration_source(s.source, s.cases)
    attempt = Path(next(iter(selected.values()))["attempt"])
    (attempt / "shared_pilot.trees").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        c.calibration_source(s.source, s.cases)


@pytest.mark.skipif(not os.environ.get("CTBF_NESTEDBD_BIO_ROOT"), reason="opt-in pinned native toy postprocessing")
def test_native_sampled_ancestor_mcc_on_toy_only(tmp_path):
    backend = p.load_backend(Path(os.environ["CTBF_NESTEDBD_BIO_ROOT"]))
    retained = [f"((1:0,2:1):{i},3:{i + 1}):0;" for i in range(1, 11)]
    source = nexus(tmp_path / "toy.trees", ["((1:1,3:1):1,2:2):0;"] + retained)
    directory = tmp_path / "native"
    directory.mkdir()
    c.native_summary(backend, source, directory)
    trees = list(c.nexus_trees(directory / "mcc.tree"))
    assert len(trees) == 1
    tree, audit = c.project_summary(trees[0][1], records())
    assert audit["sampled_ancestor_count"] == 1 and len(tree) == 4
    # First post-burn-in sample wins exact MCC ties, with its original heights.
    assert trees[0][1] == parse(retained[0])
    assert "sampled ancestor" in (directory / "stderr.txt").read_text().lower()
