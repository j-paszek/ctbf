from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ctbs
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    cnp2cnp_provenance,
    combine_ordered_cnp2cnp_matrices,
    minimum_bidirectional_distance,
    parse_cnp2cnp_directional_distance,
    validate_distance_matrix,
)
from reconstructor import build_evolution_tree
from reconstructor_algorithms import neighbor_joining_baseline
from simulator import Genotype


@pytest.mark.parametrize(
    ("output", "match"),
    [
        ("", "exactly one"),
        ("1 2", "exactly one"),
        ("warning", "not numeric"),
        ("nan", "finite"),
        ("inf", "finite"),
        ("-1", "nonnegative"),
    ],
)
def test_directional_scalar_parser_rejects_ambiguous_or_invalid_output(output, match):
    with pytest.raises(ValueError, match=match):
        parse_cnp2cnp_directional_distance(output)


def test_minimum_bidirectional_distance_is_input_order_invariant():
    assert minimum_bidirectional_distance("7\n", "3\n") == 3.0
    assert minimum_bidirectional_distance("3\n", "7\n") == 3.0


def test_opposite_order_matrix_combination_realigns_ids_and_takes_minimum():
    forward_ids = ["A", "B", "C"]
    forward = np.array(
        [
            [0.0, 7.0, 2.0],
            [7.0, 0.0, 5.0],
            [2.0, 5.0, 0.0],
        ]
    )
    reverse_ids = ["C", "B", "A"]
    reverse = np.array(
        [
            [0.0, 1.0, 9.0],
            [1.0, 0.0, 3.0],
            [9.0, 3.0, 0.0],
        ]
    )

    ids, matrix = combine_ordered_cnp2cnp_matrices(
        forward_ids,
        forward,
        reverse_ids,
        reverse,
    )

    assert ids == forward_ids
    assert np.array_equal(
        matrix,
        np.array(
            [
                [0.0, 3.0, 2.0],
                [3.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
            ]
        ),
    )


def test_symmetric_pair_helper_calls_both_profile_orders(monkeypatch):
    calls = []

    def fake_directional(input_text, runfile=None, runtime_config=None):
        calls.append((input_text, runfile, runtime_config))
        return 7.0 if input_text.startswith(">1\n") else 3.0

    monkeypatch.setattr(
        ctbs,
        "use_cnp2cnp_to_compute_pairwise_distance",
        fake_directional,
    )
    left = Genotype([0, 1], node_id=100, cell_id=1)
    right = Genotype([1, 1], node_id=200, cell_id=2)

    assert ctbs.compute_symmetric_cnp2cnp_distance(left, right, runfile="tool.py") == 3.0
    assert ctbs.compute_symmetric_cnp2cnp_distance(right, left, runfile="tool.py") == 3.0
    assert [call[0].splitlines()[0] for call in calls] == [">1", ">2", ">2", ">1"]
    assert all(call[1] == "tool.py" for call in calls)


@pytest.mark.parametrize(
    ("ids", "matrix", "match"),
    [
        ([1, 1], np.zeros((2, 2)), "Duplicate"),
        ([1, 2], np.array([[0.0, -1.0], [-1.0, 0.0]]), "nonnegative"),
        ([1, 2], np.array([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
        ([1, 2], np.array([[0.0, 1.0], [1.0 + 1e-15, 0.0]]), "exactly symmetric"),
        ([1, 2], np.array([[1e-15, 1.0], [1.0, 0.0]]), "exactly zero"),
    ],
)
def test_distance_matrix_validation_enforces_the_publication_contract(ids, matrix, match):
    with pytest.raises(ValueError, match=match):
        validate_distance_matrix(ids, matrix)


def test_path_backed_reconstruction_uses_the_same_validation(tmp_path):
    matrix_path = tmp_path / "duplicate_ids.phy"
    matrix_path.write_text("2\n1 0 1\n1 1 0\n")
    cells = [[
        Genotype([2], node_id=10, cell_id=1),
        Genotype([3], node_id=20, cell_id=2),
    ]]

    with pytest.raises(ValueError, match="Duplicate"):
        build_evolution_tree(
            cells,
            dist_matrix_path=matrix_path,
            only_nj=True,
            neighbor_joining=neighbor_joining_baseline,
        )


def test_cnp2cnp_provenance_records_semantics_command_and_source_hashes(tmp_path):
    executable = tmp_path / "cnp2cnp.py"
    solver = tmp_path / "cnpsolver.py"
    executable.write_text("print('test')\n")
    solver.write_text("class CNPSolver: pass\n")

    provenance = cnp2cnp_provenance(
        executable,
        construction="bidirectional_pair_mode",
        python_executable="/python",
    )

    assert provenance["semantics_version"] == CNP2CNP_SEMANTICS_VERSION
    assert provenance["formula"] == "min(d_any(u,v),d_any(v,u))"
    assert provenance["directional_calls_per_unordered_pair"] == 2
    assert provenance["command_template"] == [
        "/python",
        str(executable.resolve()),
        "-m",
        "dist",
        "-d",
        "any",
        "-i",
        "<forward-or-reverse-pair.fa>",
    ]
    assert set(provenance["source_sha256"]) == {"cnp2cnp.py", "cnpsolver.py"}
