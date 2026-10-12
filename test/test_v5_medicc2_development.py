from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import sys

import networkx as nx

from algorithm_evaluation import v5_medicc2_development_run as medicc2_run
from algorithm_evaluation.v5_algorithm_development_common import (
    serialize_truth,
    write_json,
)


def _observation_input(case_id: str = "fake_medicc2") -> dict:
    return {
        "schema_version": "ctbf-v5-reconstruction-input-v1",
        "case_id": case_id,
        "condition_id": "fixture",
        "levels": [
            {
                "biopsy_level": 0,
                "generation": 1,
                "states": [
                    {"state_label": "A", "cnp": [3, 3, 2, 2]},
                    {"state_label": "B", "cnp": [2, 2, 3, 3]},
                ],
            },
            {
                "biopsy_level": 1,
                "generation": 2,
                "states": [
                    {"state_label": "A", "cnp": [3, 3, 2, 2]},
                ],
            },
        ],
    }


def _write_fake_medicc2(path: Path) -> None:
    path.write_text(
        r"""from pathlib import Path
import csv
import sys

input_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
prefix = sys.argv[sys.argv.index("--prefix") + 1]
with input_path.open(encoding="utf-8", newline="") as handle:
    taxa = list(dict.fromkeys(row["sample_id"] for row in csv.DictReader(handle, delimiter="\t")))
normal = "ctbf_artificial_diploid_normal"
labels = [normal, *taxa]
with (output_root / f"{prefix}_pairwise_distances.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(["", *labels])
    for label in labels:
        writer.writerow([label, *(0 for _ in labels)])
xml = f'''<phyloxml xmlns="http://www.phyloxml.org"><phylogeny rooted="true"><clade>
<clade><name>tumor</name><clade><name>{taxa[0]}</name></clade><clade><name>inner</name>
<clade><name>{taxa[1]}</name></clade><clade><name>{taxa[2]}</name></clade>
</clade></clade><clade><name>{normal}</name></clade></clade></phylogeny></phyloxml>'''
(output_root / f"{prefix}_final_tree.xml").write_text(xml, encoding="utf-8")
for suffix in ("branch_lengths.tsv", "final_tree.new", "summary.tsv"):
    (output_root / f"{prefix}_{suffix}").write_text("fixture\n", encoding="utf-8")
""",
        encoding="utf-8",
    )


def test_external_arm_is_contextual_topology_only_partial():
    assert medicc2_run.MEDICC2_ARM_SPEC == {
        "arm_id": "medicc2_total_cn_topology_partial",
        "algorithm_name": "external_medicc2_total_cn_topology_only",
        "family": "contextual_reference",
        "problem": "partial",
        "input_mode": "pooled",
        "only_nj": True,
        "radius": None,
        "primary_metric": "grf",
        "complementary_metrics": [],
        "role": "external_contextual_baseline",
        "external_distance": "medicc2_symmetric_minimum_event_distance",
        "ancestor_policy": "topology_only_internal_nodes_unlabeled",
        "rooting_policy": "ctbf-medicc2-diploid-outgroup-removal-v1",
    }


def test_reconstruct_case_preserves_occurrences_and_exact_inventory(tmp_path):
    executable = tmp_path / "fake_medicc2.py"
    _write_fake_medicc2(executable)

    tree, root, metadata, runtime = medicc2_run.reconstruct_medicc2_case(
        _observation_input(),
        tmp_path / "case",
        command_prefix=(sys.executable, executable),
        timeout_seconds=10,
        rss_limit_bytes=4 * 1024**3,
        capture_limit_bytes=10_000,
        runtime_cache_root=tmp_path / "cache",
    )

    assert nx.is_arborescence(tree)
    assert tree.in_degree(root) == 0
    assert Counter(
        attributes["cell_id"]
        for _node, attributes in tree.nodes(data=True)
        if attributes["cell_id"] is not None
    ) == Counter({"A": 2, "B": 1})
    assert all(
        attributes["cell_id"] is None
        for node, attributes in tree.nodes(data=True)
        if tree.out_degree(node) > 0
    )
    assert {row["name"] for row in metadata["output_inventory"]} == (
        medicc2_run.expected_output_names("fake_medicc2")
    )
    assert metadata["truth_identity_used"] is False
    assert "--topology-only" in metadata["external_process"]["command"]
    assert runtime["external_execution_status"] == "success"


def test_live_capture_limit_stops_external_process(tmp_path):
    script = tmp_path / "noisy.py"
    script.write_text(
        "import sys, time\nsys.stdout.write('x' * 10000)\nsys.stdout.flush()\ntime.sleep(2)\n",
        encoding="utf-8",
    )
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    record = medicc2_run.run_bounded_medicc2_process(
        (sys.executable, str(script)),
        cwd=tmp_path,
        environment=os.environ,
        stdout_path=stdout,
        stderr_path=stderr,
        timeout_seconds=5,
        rss_limit_bytes=4 * 1024**3,
        capture_limit_bytes=100,
    )

    assert record["status"] == "failure"
    assert record["failure"]["code"] == "capture_limit_exceeded"
    assert record["stdout"]["byte_count"] > 100
    assert record["wall_time_ns"] < 2_000_000_000


def test_plus2_selector_requires_observed_descendant_ancestry():
    tree = nx.DiGraph()
    tree.add_node("root", generation=0, cell_id="root")
    tree.add_node("plus2", generation=1, cell_id="X")
    tree.add_node("observed", generation=2, cell_id="A")
    tree.add_edge(
        "root",
        "plus2",
        events=[
            {
                "event_class": "interval_mode_cna",
                "direction": "gain",
                "operator": "additive",
                "magnitude": 2,
                "multi_position": True,
                "effective": True,
            }
        ],
    )
    tree.add_edge("plus2", "observed", events=[])

    assert medicc2_run._observed_plus2_ancestry_count(
        _observation_input(), tree, 2
    ) == 1
    assert medicc2_run._observed_plus2_ancestry_count(
        _observation_input(), tree, 0
    ) == 0


def test_resource_gate_row_selection_is_distinct_and_output_blind():
    rows = [
        {
            "case": {"case_id": "first", "block_index": 0, "height": 14},
            "occurrence_count": 5,
            "observed_plus2_ancestry_count": 0,
        },
        {
            "case": {"case_id": "largest", "block_index": 1, "height": 24},
            "occurrence_count": 20,
            "observed_plus2_ancestry_count": 0,
        },
        {
            "case": {"case_id": "plus2_small", "block_index": 2, "height": 24},
            "occurrence_count": 10,
            "observed_plus2_ancestry_count": 1,
        },
        {
            "case": {"case_id": "plus2_large", "block_index": 3, "height": 34},
            "occurrence_count": 15,
            "observed_plus2_ancestry_count": 2,
        },
    ]

    selected = medicc2_run._select_resource_gate_rows(rows)

    assert [row["case_id"] for row in selected] == [
        "first",
        "largest",
        "plus2_large",
    ]
    assert len({row["case_id"] for row in selected}) == 3
    assert all("metrics" not in row for row in selected)


def test_development_case_loads_truth_only_after_reconstruction(
    tmp_path, monkeypatch, controlled_stage_rss
):
    controlled_stage_rss()
    bank_root = tmp_path / "bank"
    bank_root.mkdir()
    input_payload = _observation_input("evaluated_case")
    write_json(bank_root / "input.json", input_payload)
    truth = nx.DiGraph()
    truth.add_node("truth_root", cell_id="A", generation=0)
    truth.add_node("truth_a", cell_id="A", generation=1)
    truth.add_node("truth_b", cell_id="B", generation=2)
    truth.add_edges_from(
        [("truth_root", "truth_a"), ("truth_a", "truth_b")]
    )
    write_json(bank_root / "truth.json", serialize_truth(0, truth))

    reconstructed = nx.DiGraph()
    reconstructed.add_node("root", cell_id=None)
    reconstructed.add_node("a1", cell_id="A")
    reconstructed.add_node("inner", cell_id=None)
    reconstructed.add_node("a2", cell_id="A")
    reconstructed.add_node("b", cell_id="B")
    reconstructed.add_edges_from(
        [("root", "a1"), ("root", "inner"), ("inner", "a2"), ("inner", "b")]
    )
    events = []

    def fake_reconstruction(*_args, **_kwargs):
        events.append("reconstruction")
        return (
            reconstructed,
            "root",
            {"occurrence_input": {"occurrence_count": 3}},
            {
                "wall_time_ns": 1,
                "memory": {
                    "method": "psutil_process_tree_rss_10ms_live_limit",
                    "peak_rss_bytes": 1,
                },
            },
        )

    original_read_json = medicc2_run.read_json

    def ordered_read_json(path):
        if Path(path).name == "truth.json":
            assert events == ["reconstruction"]
            events.append("truth_read")
        return original_read_json(path)

    monkeypatch.setattr(
        medicc2_run, "reconstruct_medicc2_case", fake_reconstruction
    )
    monkeypatch.setattr(medicc2_run, "read_json", ordered_read_json)
    record = medicc2_run._run_development_case(
        str(bank_root),
        {
            "case_id": "evaluated_case",
            "block_index": 0,
            "height": 2,
            "input_path": "input.json",
            "truth_path": "truth.json",
        },
        str(tmp_path / "case"),
        (sys.executable,),
        10,
        10,
        4 * 1024**3,
        10_000,
        str(tmp_path / "cache"),
    )

    assert events == ["reconstruction", "truth_read"]
    assert record["status"] == "success", record.get("failure")
    assert 0.0 <= record["metrics"]["grf"] <= 1.0
    assert record["tree_summary"]["repeated_observed_occurrence_count"] == 1


def test_output_inventory_rejects_extra_files(tmp_path):
    prefix = "case"
    for name in medicc2_run.expected_output_names(prefix):
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    (tmp_path / "unexpected.png").write_text("fixture", encoding="utf-8")

    try:
        medicc2_run.validate_output_inventory(tmp_path, prefix)
    except ValueError as error:
        assert "output inventory changed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("An extra MEDICC2 output must be rejected.")
