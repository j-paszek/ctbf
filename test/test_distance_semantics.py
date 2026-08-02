from pathlib import Path
import importlib
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ctbs
from distance_semantics import (
    CNP2CNP_ORDERED_TRIANGLE_FAST,
    CNP2CNP_ORDERED_TRIANGLE_FAST_SEMANTICS_VERSION,
    CNP2CNP_SEMANTICS_VERSION,
    DIRECTED_DISTANCE_BUNDLE_SCHEMA_VERSION,
    DirectedDistanceBundle,
    cnp2cnp_provenance,
    combine_ordered_cnp2cnp_matrices,
    distance_input_cache_key,
    directed_bundle_from_ordered_cnp2cnp_matrices,
    minimum_bidirectional_distance,
    parse_cnp2cnp_directional_distance,
    parse_labeled_distance_matrix,
    stable_row_order_digest,
    validate_directed_distance_matrix,
    validate_distance_label_coverage,
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


def test_opposite_order_matrices_recover_immutable_directed_bundle():
    provenance = {"schema_version": DIRECTED_DISTANCE_BUNDLE_SCHEMA_VERSION}
    bundle = directed_bundle_from_ordered_cnp2cnp_matrices(
        ["A", "B", "C"],
        np.array([
            [0.0, 7.0, 2.0],
            [7.0, 0.0, 5.0],
            [2.0, 5.0, 0.0],
        ]),
        ["C", "B", "A"],
        np.array([
            [0.0, 1.0, 9.0],
            [1.0, 0.0, 3.0],
            [9.0, 3.0, 0.0],
        ]),
        provenance=provenance,
    )

    assert bundle.ids == ("A", "B", "C")
    assert np.array_equal(
        bundle.directed_matrix,
        np.array([
            [0.0, 7.0, 2.0],
            [3.0, 0.0, 5.0],
            [9.0, 1.0, 0.0],
        ]),
    )
    assert np.array_equal(
        bundle.minimum_matrix,
        np.array([
            [0.0, 3.0, 2.0],
            [3.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]),
    )
    assert bundle.directed_matrix.flags.writeable is False
    assert bundle.minimum_matrix.flags.writeable is False
    returned_provenance = bundle.provenance
    returned_provenance["schema_version"] = "changed"
    assert bundle.provenance == provenance
    with pytest.raises(ValueError, match="read-only"):
        bundle.directed_matrix[0, 1] = 0.0


def test_directed_bundle_rejects_invalid_values_and_misaligned_minimum():
    with pytest.raises(ValueError, match="diagonal"):
        validate_directed_distance_matrix([1, 2], [[1, 2], [3, 0]])
    with pytest.raises(ValueError, match="nonnegative"):
        DirectedDistanceBundle([1, 2], [[0, -1], [2, 0]])
    with pytest.raises(ValueError, match=r"min\(C, C.T\)"):
        DirectedDistanceBundle(
            [1, 2],
            [[0, 7], [3, 0]],
            minimum_matrix=[[0, 7], [7, 0]],
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


def test_shared_matrix_parser_is_strict_and_preserves_text_labels(tmp_path):
    matrix_path = tmp_path / "valid.phy"
    matrix_path.write_text("2\nA 0 2\n2 2 0\n")

    ids, matrix = parse_labeled_distance_matrix(matrix_path)

    assert ids == ["A", 2]
    assert np.array_equal(matrix, [[0.0, 2.0], [2.0, 0.0]])

    matrix_path.write_text("1\nA 0\nB 0\n")
    with pytest.raises(ValueError, match="unexpected extra rows"):
        parse_labeled_distance_matrix(matrix_path)

    matrix_path.write_text("0\n")
    ids, matrix = parse_labeled_distance_matrix(matrix_path)
    assert ids == []
    assert matrix.shape == (0, 0)


def test_distance_label_coverage_reports_missing_and_optional_extra_labels():
    assert validate_distance_label_coverage([1, 2, 3], [1, 2]) == [1, 2, 3]
    with pytest.raises(ValueError, match=r"missing=\[4\]"):
        validate_distance_label_coverage([1, 2, 3], [1, 4])
    with pytest.raises(ValueError, match=r"extra=\[3\]"):
        validate_distance_label_coverage(
            [1, 2, 3],
            [1, 2],
            allow_extra=False,
        )


def test_distance_input_cache_key_is_deterministic_and_order_sensitive():
    provenance = cnp2cnp_provenance(
        None,
        construction="bidirectional_pair_mode",
        profile_count=2,
    )
    records = [(1, "2,2"), (2, "3,2")]

    key = distance_input_cache_key(records, provenance)

    assert key == distance_input_cache_key(list(records), dict(provenance))
    assert key != distance_input_cache_key(list(reversed(records)), provenance)
    assert key != distance_input_cache_key([(1, "2,2"), (2, "4,2")], provenance)


def test_reconstruction_rejects_missing_observed_distance_label(tmp_path):
    matrix_path = tmp_path / "missing_observation.phy"
    matrix_path.write_text("1\n1 0\n")
    cells = [[
        Genotype([2], node_id=10, cell_id=1),
        Genotype([3], node_id=20, cell_id=2),
    ]]

    with pytest.raises(ValueError, match=r"missing=\[2\]"):
        build_evolution_tree(
            cells,
            dist_matrix_path=matrix_path,
            only_nj=True,
            neighbor_joining=neighbor_joining_baseline,
        )


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
    assert provenance["tool_identity_policy"] == "source_sha256_plus_git_revision"


def test_checked_cnp2cnp_failure_exposes_captured_status(monkeypatch):
    def fake_run(args, cwd, capture_output, text, check):
        raise ctbs.subprocess.CalledProcessError(
            9,
            args,
            output="partial output",
            stderr="solver failed",
        )

    monkeypatch.setattr(ctbs.subprocess, "run", fake_run)

    with pytest.raises(ctbs.Cnp2CnpExecutionError) as error:
        ctbs.use_cnp2cnp_to_compute_pairwise_distance(
            ">1\n2,2\n",
            runfile="/tmp/cnp2cnp.py",
        )

    record = error.value.record
    assert record["status"] == "failed"
    assert record["returncode"] == 9
    assert record["stdout"]["preview"] == "partial output"
    assert record["stderr"]["preview"] == "solver failed"


def test_invalid_cnp2cnp_scalar_keeps_execution_record(monkeypatch):
    def fake_run(args, cwd, capture_output, text, check):
        return type(
            "Completed",
            (),
            {"stdout": "warning 7", "stderr": "", "returncode": 0},
        )()

    monkeypatch.setattr(ctbs.subprocess, "run", fake_run)

    with pytest.raises(ctbs.Cnp2CnpExecutionError) as error:
        ctbs.use_cnp2cnp_to_compute_pairwise_distance(
            ">1\n2,2\n",
            runfile="/tmp/cnp2cnp.py",
        )

    assert error.value.record["status"] == "invalid_output"
    assert "exactly one" in error.value.record["validation_error"]


def test_missing_cnp2cnp_matrix_output_is_typed_failure(monkeypatch):
    def fake_run(args, cwd, capture_output, text, check):
        return type(
            "Completed",
            (),
            {"stdout": "", "stderr": "", "returncode": 0},
        )()

    monkeypatch.setattr(ctbs.subprocess, "run", fake_run)

    with pytest.raises(ctbs.Cnp2CnpExecutionError) as error:
        ctbs._run_cnp2cnp_ordered_matrix(
            [(1, "2,2"), (2, "3,2")],
            "/tmp/cnp2cnp.py",
        )

    assert error.value.record["status"] == "missing_output"


def test_invalid_cnp2cnp_matrix_keeps_output_hash_and_execution_record(monkeypatch):
    def fake_run(args, cwd, capture_output, text, check):
        output_path = Path(args[args.index("-o") + 1])
        output_path.write_text("2\n1 0\n")
        return type(
            "Completed",
            (),
            {"stdout": "", "stderr": "", "returncode": 0},
        )()

    monkeypatch.setattr(ctbs.subprocess, "run", fake_run)

    with pytest.raises(ctbs.Cnp2CnpExecutionError) as error:
        ctbs._run_cnp2cnp_ordered_matrix(
            [(1, "2,2"), (2, "3,2")],
            "/tmp/cnp2cnp.py",
        )

    assert error.value.record["status"] == "invalid_output"
    assert len(error.value.record["output_sha256"]) == 64
    assert "expected 2" in error.value.record["validation_error"]


def test_distance_worker_count_is_machine_and_policy_bounded(monkeypatch):
    monkeypatch.setattr(ctbs.os, "cpu_count", lambda: 64)

    assert ctbs.resolve_distance_worker_count(None, 100) == 4
    assert ctbs.resolve_distance_worker_count(7, 3) == 3
    assert ctbs.resolve_distance_worker_count(None, 0) == 0
    with pytest.raises(ValueError, match="positive integer"):
        ctbs.resolve_distance_worker_count(2.0, 100)
    with pytest.raises(ValueError, match="may not exceed"):
        ctbs.resolve_distance_worker_count(33, 100)


def test_bounded_process_map_limits_pending_tasks_and_preserves_order(monkeypatch):
    state = {"outstanding": 0, "maximum": 0, "workers": None}

    class ImmediateFuture:
        def __init__(self, function, value):
            self.function = function
            self.value = value

        def result(self):
            try:
                return self.function(self.value)
            finally:
                state["outstanding"] -= 1

        def cancel(self):
            state["outstanding"] -= 1
            return True

    class FakeExecutor:
        def __init__(self, max_workers):
            state["workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, value):
            state["outstanding"] += 1
            state["maximum"] = max(state["maximum"], state["outstanding"])
            return ImmediateFuture(function, value)

    monkeypatch.setattr(ctbs, "ProcessPoolExecutor", FakeExecutor)

    result = list(
        ctbs.bounded_process_map(
            lambda value: value * 2,
            range(10),
            max_workers=2,
            task_count=10,
        )
    )

    assert result == list(range(0, 20, 2))
    assert state == {"outstanding": 0, "maximum": 4, "workers": 2}


def test_single_worker_distance_map_does_not_construct_nested_process_pool(monkeypatch):
    monkeypatch.setattr(
        ctbs,
        "ProcessPoolExecutor",
        lambda *args, **kwargs: pytest.fail("one worker must execute serially"),
    )

    result = list(
        ctbs.bounded_process_map(
            lambda value: value + 1,
            range(3),
            max_workers=1,
            task_count=3,
        )
    )

    assert result == [1, 2, 3]


def test_pairwise_provider_records_real_bounded_execution():
    cells = [
        Genotype([2, 2], node_id=10, cell_id=1),
        Genotype([3, 2], node_id=20, cell_id=2),
        Genotype([2, 3], node_id=30, cell_id=3),
    ]
    provider = ctbs.Cnp2CnpPairwiseDistanceProvider(
        ctbs.load_ctbs_runtime_config(),
        max_threads=2,
    )

    result = provider.compute(cells)

    assert result.ids == [1, 2, 3]
    assert result.matrix.shape == (3, 3)
    assert result.matrix[0, 1] == result.matrix[1, 0]
    execution = result.provenance["external_execution"]
    assert execution["command_count"] == 6
    assert execution["status_counts"] == {"success": 6}
    assert execution["effective_worker_count"] == 2
    assert execution["pending_task_limit"] == 4
    assert len(result.provenance["input_cache_key"]) == 64


def test_cnp2cnp_matrix_mode_rejects_non_roundtripping_label_type():
    with pytest.raises(ValueError, match="round-trip"):
        ctbs._run_cnp2cnp_ordered_matrix(
            [("2", "2,2"), ("A", "3,2")],
            "/tmp/cnp2cnp.py",
        )


def test_figure3_missing_direct_backend_falls_back_to_cnp2cnp_not_l1(
    monkeypatch,
):
    figure3_dir = PROJECT_ROOT / "measures_evaluation" / "supp_figure_3"
    monkeypatch.syspath_prepend(str(figure3_dir))
    module = importlib.import_module("simulation_utils_optimized")
    monkeypatch.setattr(module, "CNP2CNP_AVAILABLE", False)
    cells = [
        Genotype([2, 2], node_id=10, cell_id=1),
        Genotype([3, 2], node_id=20, cell_id=2),
    ]
    expected = ([1, 2], np.array([[0.0, 1.0], [1.0, 0.0]]))
    calls = []

    def fake_cnp2cnp(observations, max_threads=None):
        calls.append((observations, max_threads))
        return expected

    monkeypatch.setattr(ctbs, "distance_matrix_from_biopsy", fake_cnp2cnp)
    monkeypatch.setattr(
        module,
        "compute_naive_distance_matrix",
        lambda *args, **kwargs: pytest.fail("L1 must remain a separate comparator"),
    )

    ids, matrix = module.distance_matrix_from_biopsy_optimized(
        cells,
        max_threads=1,
        show_progress=False,
    )

    assert ids == expected[0]
    assert np.array_equal(matrix, expected[1])
    assert calls == [(cells, 1)]


def test_fast_and_directed_provenance_are_separately_versioned():
    row_order = [1, "2", 3]
    fast = cnp2cnp_provenance(
        None,
        construction="ordered_triangle_matrix_mode",
        semantic_mode=CNP2CNP_ORDERED_TRIANGLE_FAST,
        row_order=row_order,
        profile_count=3,
    )
    directed = cnp2cnp_provenance(
        None,
        construction="opposite_order_matrix_mode_directed_bundle",
        profile_count=3,
        retains_directed=True,
    )

    assert fast["semantics_version"] == (
        CNP2CNP_ORDERED_TRIANGLE_FAST_SEMANTICS_VERSION
    )
    assert fast["semantics_version"] != CNP2CNP_SEMANTICS_VERSION
    assert fast["row_order_sha256"] == stable_row_order_digest(row_order)
    assert fast["directional_transformation_count"] == 3
    assert fast["external_process_count"] == 1
    assert directed["semantics_version"] == CNP2CNP_SEMANTICS_VERSION
    assert directed["directional_transformation_count"] == 6
    assert directed["external_process_count"] == 2
    assert directed["directed_bundle_schema_version"] == (
        DIRECTED_DISTANCE_BUNDLE_SCHEMA_VERSION
    )


def test_explicit_matrix_providers_use_one_or_two_recorded_orders(monkeypatch, tmp_path):
    calls = []
    directional = {(1, 2): 7.0, (2, 1): 3.0}

    def fake_run(args, cwd, capture_output, text, check):
        input_path = Path(args[args.index("-i") + 1])
        output_path = Path(args[args.index("-o") + 1])
        assert Path(cwd) == input_path.parent == output_path.parent
        lines = [line.strip() for line in input_path.read_text().splitlines()]
        ids = [int(lines[index][1:]) for index in range(0, len(lines), 2)]
        calls.append(ids)
        matrix = np.zeros((len(ids), len(ids)), dtype=float)
        for left in range(len(ids)):
            for right in range(left + 1, len(ids)):
                matrix[left, right] = directional[(ids[left], ids[right])]
                matrix[right, left] = matrix[left, right]
        output_path.write_text(
            f"{len(ids)}\n"
            + "".join(
                f"{cell_id} " + " ".join(str(value) for value in row) + "\n"
                for cell_id, row in zip(ids, matrix)
            )
        )
        return type(
            "Completed",
            (),
            {"stdout": "", "stderr": "", "returncode": 0},
        )()

    monkeypatch.setattr(ctbs.subprocess, "run", fake_run)
    base = ctbs.default_ctbs_runtime_config()
    runtime = ctbs.CtbsRuntimeConfig(
        in_file_name=base.in_file_name,
        out_file_name=str(tmp_path / "compatibility.phy"),
        sim_dm=base.sim_dm,
        cnp2cnp_folder=base.cnp2cnp_folder,
        cnp2cnp_file="/tmp/cnp2cnp.py",
        true_tree_root_id=base.true_tree_root_id,
        run_single_test=base.run_single_test,
    )
    cells = [
        Genotype([3], node_id=20, cell_id=2),
        Genotype([2], node_id=10, cell_id=1),
    ]

    fast = ctbs.Cnp2CnpOrderedTriangleFastDistanceProvider(runtime).compute(cells)
    assert calls == [[1, 2]]
    assert fast.ids == [1, 2]
    assert fast.matrix[0, 1] == 7.0
    assert fast.provenance["row_order_sha256"] == stable_row_order_digest([1, 2])

    calls.clear()
    bundle = ctbs.Cnp2CnpDirectedFileDistanceProvider(runtime).compute(cells)
    assert calls == [[1, 2], [2, 1]]
    assert np.array_equal(bundle.directed_matrix, [[0.0, 7.0], [3.0, 0.0]])
    assert np.array_equal(bundle.minimum_matrix, [[0.0, 3.0], [3.0, 0.0]])

    calls.clear()
    minimum = ctbs.Cnp2CnpFileDistanceProvider(runtime).compute(cells)
    assert calls == [[2, 1], [1, 2]]
    assert np.array_equal(minimum.matrix, [[0.0, 3.0], [3.0, 0.0]])
    assert not Path(runtime.out_file_name).exists()
    assert minimum.provenance["external_execution"]["command_count"] == 2
    assert len(minimum.provenance["input_cache_key"]) == 64


def test_default_provider_keeps_minimum_and_rejects_unsupported_parallel_modes():
    base = ctbs.default_ctbs_runtime_config()
    runtime = ctbs.CtbsRuntimeConfig(
        in_file_name=base.in_file_name,
        out_file_name=base.out_file_name,
        sim_dm=base.sim_dm,
        cnp2cnp_folder=base.cnp2cnp_folder,
        cnp2cnp_file="/tmp/nonexistent-ctbf-cnp2cnp.py",
        true_tree_root_id=base.true_tree_root_id,
        run_single_test=base.run_single_test,
    )

    assert isinstance(
        ctbs.default_distance_provider(runtime_config=runtime),
        ctbs.Cnp2CnpFileDistanceProvider,
    )
    assert isinstance(
        ctbs.default_distance_provider(
            runtime_config=runtime,
            distance_construction=ctbs.CNP2CNP_DISTANCE_CONSTRUCTION_FAST,
        ),
        ctbs.Cnp2CnpOrderedTriangleFastDistanceProvider,
    )
    with pytest.raises(ValueError, match="parallel=False"):
        ctbs.default_distance_provider(
            parallel=True,
            runtime_config=runtime,
            distance_construction=ctbs.CNP2CNP_DISTANCE_CONSTRUCTION_DIRECTED,
        )


@pytest.mark.parametrize("cells", [[], [Genotype([2], node_id=1, cell_id=1)]])
def test_fast_and_directed_providers_handle_degenerate_inputs_without_processes(
    monkeypatch,
    cells,
):
    monkeypatch.setattr(
        ctbs.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("cnp2cnp process should not run"),
    )
    base = ctbs.default_ctbs_runtime_config()
    runtime = ctbs.CtbsRuntimeConfig(
        in_file_name=base.in_file_name,
        out_file_name=base.out_file_name,
        sim_dm=base.sim_dm,
        cnp2cnp_folder=base.cnp2cnp_folder,
        cnp2cnp_file="/tmp/nonexistent-ctbf-cnp2cnp.py",
        true_tree_root_id=base.true_tree_root_id,
        run_single_test=base.run_single_test,
    )

    fast = ctbs.Cnp2CnpOrderedTriangleFastDistanceProvider(runtime).compute(cells)
    directed = ctbs.Cnp2CnpDirectedFileDistanceProvider(runtime).compute(cells)

    assert fast.matrix.shape == (len(cells), len(cells))
    assert directed.minimum_matrix.shape == (len(cells), len(cells))
    assert fast.provenance["external_process_count"] == 0
    assert directed.provenance["external_process_count"] == 0


def test_runtime_distance_construction_selection_is_explicit(monkeypatch):
    cells = [
        Genotype([2], node_id=10, cell_id=1),
        Genotype([3], node_id=20, cell_id=2),
    ]
    calls = []
    supplied = ctbs.SuppliedDistanceProvider(
        [1, 2],
        [[0.0, 1.0], [1.0, 0.0]],
    )

    def fake_default_provider(**kwargs):
        calls.append(kwargs)
        return supplied

    monkeypatch.setattr(ctbs, "default_distance_provider", fake_default_provider)
    result = ctbs._compute_distance_matrix(
        [cells],
        parallel=False,
        time_collector=None,
        runtime_config=ctbs.default_ctbs_runtime_config(),
        distance_construction=ctbs.CNP2CNP_DISTANCE_CONSTRUCTION_FAST,
    )

    assert isinstance(result, ctbs.DistanceMatrix)
    assert calls[0]["distance_construction"] == (
        ctbs.CNP2CNP_DISTANCE_CONSTRUCTION_FAST
    )
    with pytest.raises(ValueError, match="either distance_provider"):
        ctbs._compute_distance_matrix(
            [cells],
            parallel=False,
            time_collector=None,
            runtime_config=ctbs.default_ctbs_runtime_config(),
            distance_provider=supplied,
            distance_construction=ctbs.CNP2CNP_DISTANCE_CONSTRUCTION_FAST,
        )
