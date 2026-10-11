from collections import Counter
import csv

import networkx as nx
import pytest

from algorithm_evaluation import medicc2_reconstruction_preflight as medicc2
from algorithm_evaluation.paper_pipeline_runner import RECONSTRUCTION_INPUT_SCHEMA_VERSION


def _observation_input():
    return {
        "schema_version": RECONSTRUCTION_INPUT_SCHEMA_VERSION,
        "case_id": "toy",
        "condition_id": "two-level",
        "fraction": 1.0,
        "schedule_id": "toy",
        "levels": [
            {
                "biopsy_level": 0,
                "generation": 2,
                "states": [
                    {"state_label": "A", "cnp": [3, 3, 2, 2]},
                    {"state_label": "B", "cnp": [2, 2, 3, 3]},
                ],
            },
            {
                "biopsy_level": 1,
                "generation": 4,
                "states": [
                    {"state_label": "A", "cnp": [3, 3, 2, 2]},
                ],
            },
        ],
    }


def _phyloxml(normal_name=medicc2.ARTIFICIAL_NORMAL_TAXON):
    return f"""\
<phyloxml xmlns="http://www.phyloxml.org">
  <phylogeny rooted="true">
    <clade>
      <branch_length>0</branch_length>
      <clade>
        <name>internal2</name>
        <branch_length>0</branch_length>
        <clade>
          <name>external_A1</name>
          <branch_length>0</branch_length>
        </clade>
        <clade>
          <name>internal1</name>
          <branch_length>0.5</branch_length>
          <clade><name>external_A2</name><branch_length>0</branch_length></clade>
          <clade><name>external_B</name><branch_length>1</branch_length></clade>
        </clade>
      </clade>
      <clade><name>{normal_name}</name><branch_length>0</branch_length></clade>
    </clade>
  </phylogeny>
</phyloxml>
"""


def test_occurrence_export_preserves_recurrent_labels_as_distinct_taxa(tmp_path):
    output = tmp_path / "input.tsv"

    exported = medicc2.export_observation_occurrences(
        _observation_input(),
        output,
        chromosome_by_bin=("1", "1", "2", "2"),
    )

    assert exported.occurrence_count == 3
    assert exported.non_diploid_occurrence_count == 3
    assert exported.unique_state_label_count == 2
    assert exported.cnp_length == 4
    assert exported.chromosome_count == 2
    assert len(exported.taxon_to_state_label) == 3
    assert Counter(exported.taxon_to_state_label.values()) == Counter({"A": 2, "B": 1})
    assert len(set(exported.taxon_to_state_label)) == 3

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 12
    assert {row["sample_id"] for row in rows} == set(exported.taxon_to_state_label)
    first_taxon_rows = [row for row in rows if row["sample_id"] == "ctbf_o000000"]
    assert [(row["chrom"], row["start"], row["end"]) for row in first_taxon_rows] == [
        ("1", "0", "1"),
        ("1", "1", "2"),
        ("2", "0", "1"),
        ("2", "1", "2"),
    ]


def test_occurrence_export_rejects_silent_medicc2_copy_number_capping(tmp_path):
    payload = _observation_input()
    payload["levels"][0]["states"][1]["cnp"][0] = 9

    with pytest.raises(ValueError, match="rejected rather than capped"):
        medicc2.export_observation_occurrences(payload, tmp_path / "input.tsv")


def test_occurrence_export_rejects_reentered_chromosome_block(tmp_path):
    with pytest.raises(ValueError, match="one contiguous bin block"):
        medicc2.export_observation_occurrences(
            _observation_input(),
            tmp_path / "input.tsv",
            chromosome_by_bin=("1", "2", "1", "2"),
        )


def test_occurrence_export_preserves_an_all_diploid_zero_signal_case(tmp_path):
    payload = _observation_input()
    payload["levels"] = [
        {
            "biopsy_level": level,
            "generation": 2 * (level + 1),
            "states": [{"state_label": "A", "cnp": [2, 2, 2, 2]}],
        }
        for level in range(3)
    ]

    exported = medicc2.export_observation_occurrences(
        payload, tmp_path / "input.tsv"
    )

    assert exported.occurrence_count == 3
    assert exported.non_diploid_occurrence_count == 0


def test_occurrence_export_hides_biopsy_order_from_pooled_medicc2_input(tmp_path):
    first = _observation_input()
    second = _observation_input()
    second["levels"][0]["states"] = [second["levels"][0]["states"][0]]
    second["levels"][1]["states"].append(
        {"state_label": "B", "cnp": [2, 2, 3, 3]}
    )

    first_export = medicc2.export_observation_occurrences(
        first, tmp_path / "first.tsv"
    )
    second_export = medicc2.export_observation_occurrences(
        second, tmp_path / "second.tsv"
    )

    assert first_export.path.read_bytes() == second_export.path.read_bytes()
    assert all(
        taxon.startswith("ctbf_o") and "_b" not in taxon
        for taxon in first_export.taxon_to_state_label
    )


def test_bounded_scale_fixture_is_16_distinct_taxa_by_100_segments(tmp_path):
    case = medicc2._write_bounded_scale_fixture(tmp_path / "scale.tsv")

    with case.input_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    profiles = {}
    for taxon in case.taxon_to_state_label:
        profiles[taxon] = tuple(
            int(row["cn"]) for row in rows if row["sample_id"] == taxon
        )

    assert case.case_id == "scale100_n16"
    assert case.segment_count == 100
    assert len(case.taxon_to_state_label) == 16
    assert len(rows) == 1600
    assert all(len(profile) == 100 for profile in profiles.values())
    assert len(set(profiles.values())) == 16
    assert min(min(profile) for profile in profiles.values()) >= 0
    assert max(max(profile) for profile in profiles.values()) <= 8


def test_phyloxml_projection_removes_normal_and_preserves_repeated_labels(tmp_path):
    path = tmp_path / "tree.xml"
    path.write_text(_phyloxml(), encoding="utf-8")

    tree, root, metadata = medicc2.project_phyloxml_to_partial_tree(
        path,
        {"external_A1": "A", "external_A2": "A", "external_B": "B"},
    )

    assert nx.is_arborescence(tree)
    assert tree.in_degree(root) == 0
    assert tree.number_of_nodes() == 5
    assert tree.number_of_edges() == 4
    assert Counter(
        attributes["cell_id"]
        for _, attributes in tree.nodes(data=True)
        if attributes["cell_id"] is not None
    ) == Counter({"A": 2, "B": 1})
    assert all(
        attributes.get("medicc2_taxon") != medicc2.ARTIFICIAL_NORMAL_TAXON
        for _, attributes in tree.nodes(data=True)
    )
    assert metadata["normal_taxon_removed"] == medicc2.ARTIFICIAL_NORMAL_TAXON
    assert metadata["repeated_observed_label_occurrence_count"] == 1
    assert metadata["projected_tree"]["unlabeled_node_count"] == 2


def test_phyloxml_projection_requires_exact_outgroup_shape(tmp_path):
    path = tmp_path / "tree.xml"
    path.write_text(_phyloxml(normal_name="wrong_normal"), encoding="utf-8")

    with pytest.raises(ValueError, match="artificial normal leaf"):
        medicc2.project_phyloxml_to_partial_tree(
            path,
            {"external_A1": "A", "external_A2": "A", "external_B": "B"},
        )


def test_topology_command_freezes_partial_total_cn_headless_mode(tmp_path):
    command = medicc2.topology_only_command(
        ("/external/python", "/external/medicc2"),
        tmp_path / "input.tsv",
        tmp_path / "output",
        prefix="toy",
    )

    assert command[:2] == ["/external/python", "/external/medicc2"]
    assert "--topology-only" in command
    assert "--total-copy-numbers" in command
    assert command[command.index("--input-allele-columns") + 1] == "cn"
    assert command[command.index("--normal-name") + 1] == medicc2.ARTIFICIAL_NORMAL_TAXON
    assert command[command.index("--maxcn") + 1] == "8"
    assert command[command.index("--plot") + 1] == "none"
    assert "--no-plot-tree" in command
    assert "--no-wgd" in command


def test_pairwise_distance_reader_requires_symmetric_exact_taxa(tmp_path):
    path = tmp_path / "distances.tsv"
    path.write_text("\tA\tB\nA\t0\t2\nB\t2\t0\n", encoding="utf-8")

    assert medicc2.read_pairwise_distances(path)["A"]["B"] == 2.0

    path.write_text("\tA\tB\nA\t0\t2\nB\t3\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not symmetric"):
        medicc2.read_pairwise_distances(path)
