"""Bounded tutorial checks; no external executable, browser, or network."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from ctbs import DistanceMatrix, unique_cells_by_cell_id
from examples import quickstart as demo
from trimmed_truth_evaluation import prepare_trimmed_truth


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    return json.loads(demo.DEFAULT_CONFIG.read_text())


def supplied_test_matrix(cells):
    # Explicit test fixture only. The public runner always uses cnp2cnp.
    cnps = np.stack([cell.genome for cell in cells])
    matrix = np.abs(cnps[:, None, :] - cnps[None, :, :]).sum(axis=2).astype(float)
    return DistanceMatrix(ids=[cell.cell_id for cell in cells], matrix=matrix)


def test_demo_uses_standard_event_settings(settings):
    base = json.loads((ROOT / "simulator_examples/default.json").read_text())
    base["NUMBER_OF_GENERATIONS"] = 14
    assert settings["simulator"] == base
    assert settings["biopsy_generations"] == [9, 12, 14]


@pytest.mark.parametrize("key,value", [
    ("biopsy_generations", [9, 9]),
    ("biopsy_generations", [12, 9]), ("biopsy_generations", [9, 15]),
    ("methods", ["M0", "M0"]), ("methods", ["M99"]),
    ("methods", []), ("seed", -1), ("cnp2cnp_timeout_seconds", 0),
    ("visualize", "false"), ("write_newick", 1), ("partial_ad_f1", None),
    ("distance_workers", 0), ("distance_workers", True), ("distance_workers", 1.5),
    ("output", None), ("cnp2cnp", ""), ("bedfile", False), ("simulator", []),
])
def test_bad_settings_fail_early(settings, key, value):
    settings[key] = value
    with pytest.raises(ValueError):
        demo.validate_settings(settings)


def test_compositions_match_paper_and_benchmark(settings, monkeypatch):
    from algorithm_evaluation import v5_algorithm_development_common as common
    from algorithm_evaluation.v5_algorithm_development_common import ARM_SPEC_BY_ID
    with (ROOT / "reproducibility/method-map.tsv").open() as stream:
        mapping = {row["paper_id"]: row for row in csv.DictReader(stream, delimiter="\t")}
    sim, biopsies, sampling = demo.simulate(settings)
    cells = unique_cells_by_cell_id([c for b in biopsies for c in b])
    distance = supplied_test_matrix(cells)
    calls, original = [], common.build_evolution_tree

    def checked(inputs, **kwargs):
        calls.append((inputs, kwargs))
        return original(inputs, **kwargs)

    monkeypatch.setattr(common, "build_evolution_tree", checked)
    before = [[(c.cell_id, c.node_id, c.genome.copy()) for c in b] for b in biopsies]
    matrix_before = distance.matrix.copy()
    trees = demo.reconstruct(biopsies, distance, settings["methods"], settings["seed"])
    for method, (inputs, kwargs) in zip(settings["methods"], calls):
        spec = ARM_SPEC_BY_ID[mapping[method]["arm_id"]]
        assert demo.METHODS[method][1] == mapping[method]["arm_id"]
        assert kwargs["neighbor_joining"].__name__ == spec.algorithm_name
        assert kwargs["only_nj"] == spec.only_nj
        if method in {"M2", "M4", "M6", "M7"}:
            assert kwargs["biopsy_guided_config"].radius_policy == spec.radius_policy
            assert kwargs["biopsy_guided_config"].candidate_tie_breaker.__name__ == "select_deferred_candidate"
            assert kwargs["r"] is None
        else:
            assert len(inputs) == 1
        assert nx.is_arborescence(trees[method])
        labels = [attrs.get("cell_id") for _, attrs in trees[method].nodes(data=True)]
        assert (None in labels) == (method in demo.PARTIAL_METHODS)
        assert {str(c.cell_id) for c in cells} <= {str(x) for x in labels if x is not None}
    for biopsy, saved in zip(biopsies, before):
        for c, (label, node, genome) in zip(biopsy, saved):
            assert (c.cell_id, c.node_id) == (label, node)
            np.testing.assert_array_equal(c.genome, genome)
    np.testing.assert_array_equal(distance.matrix, matrix_before)
    assert set(trees["M2"].edges) == set(trees["M6"].edges)
    cleared = []
    for node in trees["M6"]:
        partial, full = trees["M2"].nodes[node], trees["M6"].nodes[node]
        if partial["cell_id"] is None:
            cleared.append(node)
            assert full["cell_id"] is not None and partial["genome"] is None
        else:
            assert partial["cell_id"] == full["cell_id"]
    assert cleared
    truth = prepare_trimmed_truth(sim.canonicalized_tree_by_genome(), [c.cell_id for c in cells])
    assert truth.trimmed.metadata["node_count"] <= truth.original.metadata["node_count"]
    assert all(row["selected"] <= row["available"] for row in sampling)


def test_gallery_complete_without_bundling_external_software(settings, tmp_path, monkeypatch):
    external = tmp_path / "external.py"
    external.write_text("fixture path, never executed\n")
    monkeypatch.setattr(demo.Cnp2CnpFileDistanceProvider, "compute",
                        lambda self, cells: supplied_test_matrix(cells))
    output = tmp_path / "gallery"
    result = demo.run(settings, output, external)
    assert set(result["methods"]) == {f"M{i}" for i in range(8)}
    assert len(list(output.glob("*.html"))) == 11
    assert len(list(output.glob("*.json"))) == 4
    for name in ("true_tree", "trimmed_true_tree", *demo.METHODS):
        html = (output / f"{name}.html").read_text()
        assert "https://cdn.plot.ly/" in html
        assert len(html) < 200_000  # no embedded Plotly library
    inputs = json.loads((output / "input.json").read_text())
    assert all(set(cell) == {"cell_id", "genome"} for b in inputs["biopsies"] for cell in b["cells"])
    saved = json.loads((output / "trees.json").read_text())
    truth = nx.node_link_graph(saved["true_tree"], edges="edges")
    trimmed = nx.node_link_graph(saved["trimmed_true_tree"], edges="edges")
    labels = {cell["cell_id"] for b in inputs["biopsies"] for cell in b["cells"]}
    expected = demo.trim_true_tree(truth, labels)
    assert set(trimmed.nodes) == set(expected.nodes)
    assert set(trimmed.edges) == set(expected.edges)
    for n in trimmed:
        assert trimmed.nodes[n] == expected.nodes[n]
    with pytest.raises(FileExistsError):
        demo.run(settings, output, external)


@pytest.mark.parametrize("sampling", [
    {"mode": "fixed", "size": 0}, {"mode": "fixed", "size": True},
    {"mode": "fraction", "fraction": 0, "lower_bound": 6},
    {"mode": "fraction", "fraction": True, "lower_bound": 6},
    {"mode": "fraction", "fraction": float("nan"), "lower_bound": 6},
    {"mode": "fraction", "fraction": 1.1, "lower_bound": 6},
    {"mode": "fraction", "fraction": .5, "lower_bound": 0},
    {"mode": "fixed", "size": 6, "fraction": .5}, {}, None,
])
def test_invalid_sampling(settings, sampling):
    settings["biopsy_sampling"] = sampling
    with pytest.raises(ValueError, match="biopsy_sampling"):
        demo.validate_settings(settings)


def test_fixed_sampling_is_capped_and_deterministic(settings):
    settings["biopsy_sampling"] = {"mode": "fixed", "size": 3}
    first = demo.simulate(settings)[2]
    second = demo.simulate(settings)[2]
    assert first == second
    assert all(row["requested"] == min(3, row["available"]) for row in first)


def test_json_only_headless_newick_and_partial_diagnostic(settings, tmp_path, monkeypatch, capsys):
    external = tmp_path / "external.py"
    external.write_text("not executed\n")
    settings.update(cnp2cnp="external.py", output="result", methods=["M0"],
                    visualize=False, write_newick=True, partial_ad_f1=True)
    config = tmp_path / "run.json"
    config.write_text(json.dumps(settings))
    monkeypatch.setattr(demo.Cnp2CnpFileDistanceProvider, "compute",
                        lambda self, cells: supplied_test_matrix(cells))
    monkeypatch.setattr(demo, "write_gallery", lambda *args: pytest.fail("headless must not plot"))
    assert demo.main(["--config", str(config)]) == 0
    output = tmp_path / "result"
    assert not list(output.glob("*.html"))
    assert len(list(output.glob("*.nwk"))) == 3
    text = (output / "M0.nwk").read_text()
    assert text.strip().endswith(";") and "None" not in text and ":" not in text
    summary = json.loads((output / "summary.json").read_text())
    assert summary["methods"]["M0"]["ad_f1"] == 0
    assert summary["methods"]["M0"]["ad_f1_role"] == "partial_diagnostic"
    assert "partial diagnostic" in capsys.readouterr().out


def test_subset_gallery_and_cli_overrides(settings, tmp_path, monkeypatch):
    external = tmp_path / "external.py"
    external.write_text("not executed\n")
    settings.update(methods=["M1"], output="do-not-use", cnp2cnp="missing.py")
    config = tmp_path / "run.json"
    config.write_text(json.dumps(settings))
    monkeypatch.setattr(demo.Cnp2CnpFileDistanceProvider, "compute",
                        lambda self, cells: supplied_test_matrix(cells))
    output = tmp_path / "override"
    demo.main(["--config", str(config), "--cnp2cnp", str(external), "--output", str(output)])
    text = (output / "index.html").read_text()
    assert "1 reconstruction</h1>" in text
    assert "four reconstructions" not in text and 'href="M0.html"' not in text
    assert not (tmp_path / "do-not-use").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["methods"]["M1"]["ad_f1"] is not None


def test_pairwise_provider_dispatch(settings, tmp_path, monkeypatch):
    external = tmp_path / "external.py"
    external.write_text("not executed\n")
    settings.update(distance_workers=3, visualize=False, methods=["M3"])
    calls = []

    def compute(provider, cells):
        calls.append(provider.max_threads)
        return supplied_test_matrix(cells)

    monkeypatch.setattr(demo.Cnp2CnpPairwiseDistanceProvider, "compute", compute)
    monkeypatch.setattr(demo.Cnp2CnpFileDistanceProvider, "compute",
                        lambda *a: pytest.fail("must use requested pairwise provider"))
    result = demo.run(settings, tmp_path / "output", external)
    assert calls == [3] and result["distance"]["construction"] == "pairwise"


def test_bed_relative_to_json_and_chromosome_guard(settings, tmp_path):
    for key in ("GENOME_LENGTH", "NUMBER_OF_CHROMOSOMES", "INITIAL_COPY_NUMBER", "CRUCIAL_BIN_INDICES"):
        del settings["simulator"][key]
    settings["bedfile"] = "bins.csv"
    bed = tmp_path / "bins.csv"
    bed.write_text("ChromosomeNumber,Start,End,Parameters\n1,0,100,CN=2;CRUCIAL=1\n")
    sim = demo.validate_settings(settings, base_dir=tmp_path)
    assert sim.genome_length == 1 and sim.bed_sha256
    bed.write_text(bed.read_text() + "2,0,100,CN=2\n")
    with pytest.raises(ValueError, match="single-chromosome"):
        demo.validate_settings(settings, base_dir=tmp_path)


def test_multiple_chromosomes_without_bed_are_not_silently_concatenated(settings):
    settings["simulator"]["NUMBER_OF_CHROMOSOMES"] = 2
    with pytest.raises(ValueError, match="single-chromosome"):
        demo.validate_settings(settings)


def test_recurrent_single_state_works_for_every_method(settings):
    settings["simulator"]["CNA_EVENT_PROBABILITY"] = 0
    sim, biopsies, _ = demo.simulate(settings)
    cells = unique_cells_by_cell_id([c for b in biopsies for c in b])
    assert len(cells) == 1
    trees = demo.reconstruct(biopsies, supplied_test_matrix(cells), settings["methods"], settings["seed"])
    truth = sim.canonicalized_tree_by_genome()
    scores = demo.evaluate(truth, demo.trim_true_tree(truth, [cells[0].cell_id]),
                           trees, [cells[0].cell_id], partial_ad_f1=True)
    assert set(scores) == set(demo.METHODS)
    assert all(nx.is_arborescence(tree) for tree in trees.values())
    assert all(0 <= row["grf_full"] <= 1 and 0 <= row["ad_f1"] <= 1 for row in scores.values())


def test_missing_external_path_and_unknown_options_fail_before_simulation(settings, monkeypatch, tmp_path):
    settings["output"] = str(tmp_path / "not-created")
    monkeypatch.setattr(demo, "simulate", lambda *a, **k: pytest.fail("must fail before simulation"))
    with pytest.raises(ValueError, match="Set cnp2cnp"):
        demo.run(settings)
    settings["unexpected"] = 1
    with pytest.raises(ValueError, match="exactly"):
        demo.run(settings)
    assert not (tmp_path / "not-created").exists()


def test_profile_guard_does_not_compute_distance_or_create_output(settings, tmp_path, monkeypatch):
    external = tmp_path / "external.py"
    external.write_text("fixture\n")
    settings["max_distance_profiles"] = 2
    monkeypatch.setattr(demo.Cnp2CnpFileDistanceProvider, "compute",
                        lambda *args: pytest.fail("distance stage must not run"))
    with pytest.raises(ValueError, match="max_distance_profiles"):
        demo.run(settings, tmp_path / "output", external)
    assert not (tmp_path / "output").exists()


def test_sampled_truth_ids_are_not_used_as_reconstruction_matches(settings):
    sim, biopsies, _ = demo.simulate(settings)
    changed = deepcopy(biopsies)
    for biopsy in changed:
        for c in biopsy:
            c.node_id += 10000
    cells = unique_cells_by_cell_id([c for b in biopsies for c in b])
    distance = supplied_test_matrix(cells)
    first = demo.reconstruct(biopsies, distance, settings["methods"], settings["seed"])
    second = demo.reconstruct(changed, distance, settings["methods"], settings["seed"])
    for method in first:
        assert set(first[method].edges) == set(second[method].edges)
        assert nx.get_node_attributes(first[method], "cell_id") == nx.get_node_attributes(second[method], "cell_id")
