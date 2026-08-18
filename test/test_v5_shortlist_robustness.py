from __future__ import annotations

import copy
import multiprocessing
import os

import networkx as nx
import numpy as np
import pytest

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.process_isolation import (
    CASE_ARM_WORKER_UNIT,
    TRUTH_BLOCK_SIMULATION_WORKER_UNIT,
    FreshSpawnPerTaskExecutor,
    fresh_process_contract,
)
from algorithm_evaluation.v5_shortlist_robustness_bank import generate_bank
from algorithm_evaluation.v5_shortlist_robustness_common import (
    ADAPTIVE_A_PRIME_ID,
    ADAPTIVE_B_PRIME_ID,
    ADAPTIVE_C_PRIME_ID,
    ADAPTIVE_D_PRIME_ID,
    ADAPTIVE_RADIUS_ARM_IDS,
    ARM_SET_BY_NAME,
    DECLARED_METRICS,
    DISTANCE_EXECUTION_SCHEMA_VERSION,
    FULL_DEVELOPMENT_ARM_IDS,
    ORDERED_A_ID,
    ORDERED_B_ID,
    ORDERED_C_ID,
    POOLED_D_ID,
    PREVIOUS_RUN_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    V2_COMPLETE_ARM_IDS,
    V2_EXTENSION_ARM_IDS,
    derived_seed,
    late_schedule,
    load_bank_manifest,
    random_schedule,
    spread_schedule,
    write_json,
)
from algorithm_evaluation.v5_shortlist_robustness_report import (
    FULL_PRINCIPAL_PAIRS,
    _bank_resource_execution,
    _depth_interactions,
    _placement_interactions,
    _record_index,
    write_report,
)
from algorithm_evaluation.v5_shortlist_robustness_run import run_shortlist
from algorithm_evaluation.v5_shortlist_resource_isolation_probe import run_probe
from ctbs import DistanceMatrix
from distance_semantics import cnp2cnp_provenance
from simulator import Genotype, SimulationResourceLimitExceeded


class _TinyH38Simulator:
    def __init__(self, _mapping, seed):
        self.seed = seed
        self.tree = nx.DiGraph()
        self.genotypes = {}
        for generation in range(39):
            genome = np.full(100, 2, dtype=int)
            genome[generation % 100] = 2 + generation
            genotype = Genotype(
                genome=genome,
                node_id=generation,
                generation=generation,
                cell_id=generation,
            )
            self.genotypes[generation] = genotype
            self.tree.add_node(
                generation,
                genome=genome.tolist(),
                generation=generation,
                cell_id=generation,
            )
            if generation:
                self.tree.add_edge(
                    generation - 1,
                    generation,
                    events=[],
                    events_text="",
                    event_count=0,
                )
        self.tree.graph["simulation_outcome"] = {
            "status": "completed",
            "configured_final_generation": 38,
            "last_retained_generation": 38,
            "extinction_generation": None,
            "failure_generation": None,
        }

    def run_simulation(self):
        return None

    def canonicalize_biopsy_genotypes(self, genotypes):
        return [
            Genotype(
                genome=genotype.genome.copy(),
                node_id=genotype.node_id,
                generation=genotype.generation,
                cell_id=genotype.cell_id,
            )
            for genotype in genotypes
        ]

    def canonicalized_tree_by_genome(self):
        return copy.deepcopy(self.tree)

    def diagnostics_snapshot(self):
        return {"totals": {}}


class _GuardedTinyH38Simulator(_TinyH38Simulator):
    def run_simulation(self):
        raise SimulationResourceLimitExceeded(
            generation=38,
            limit_name="MAX_REPRESENTATIVES_PER_GENERATION",
            configured_limit=2000,
            attempted_count=2001,
        )


def _injected_distance(cells) -> DistanceMatrix:
    ids = [cell.cell_id for cell in cells]
    profiles = np.asarray([cell.genome for cell in cells], dtype=float)
    matrix = np.abs(profiles[:, None, :] - profiles[None, :, :]).sum(axis=2)
    return DistanceMatrix(
        ids=ids,
        matrix=matrix,
        provenance=cnp2cnp_provenance(
            construction="opposite_order_matrix_mode",
            profile_count=len(ids),
        ),
    )


def _interrupting_distance(_cells) -> DistanceMatrix:
    raise KeyboardInterrupt("distance fixture interruption")


_ISOLATION_WORKER_COUNTER = 0


def _isolation_worker_identity():
    global _ISOLATION_WORKER_COUNTER
    _ISOLATION_WORKER_COUNTER += 1
    return os.getpid(), _ISOLATION_WORKER_COUNTER


def _success_record(block, height, policy, arm, value):
    return {
        "block_index": block,
        "height": height,
        "placement_policy": policy,
        "arm_id": arm,
        "status": "success",
        "metrics": {metric: float(value) for metric in DECLARED_METRICS},
    }


def test_shortlist_schedules_are_fixed_and_random_is_prospective():
    assert ARM_SET_BY_NAME["v2-complete"] == V2_COMPLETE_ARM_IDS
    assert ARM_SET_BY_NAME["v2-extensions"] == V2_EXTENSION_ARM_IDS
    assert ARM_SET_BY_NAME["adaptive-radius"] == ADAPTIVE_RADIUS_ARM_IDS
    assert ADAPTIVE_RADIUS_ARM_IDS == (
        ADAPTIVE_A_PRIME_ID,
        ADAPTIVE_B_PRIME_ID,
        ADAPTIVE_C_PRIME_ID,
        ADAPTIVE_D_PRIME_ID,
    )
    assert (ADAPTIVE_A_PRIME_ID, ADAPTIVE_B_PRIME_ID) in FULL_PRINCIPAL_PAIRS
    assert (ADAPTIVE_A_PRIME_ID, ADAPTIVE_C_PRIME_ID) in FULL_PRINCIPAL_PAIRS
    assert (ADAPTIVE_C_PRIME_ID, ADAPTIVE_D_PRIME_ID) in FULL_PRINCIPAL_PAIRS
    assert (ADAPTIVE_B_PRIME_ID, ADAPTIVE_D_PRIME_ID) in FULL_PRINCIPAL_PAIRS
    assert spread_schedule(14) == (9, 12, 14)
    assert spread_schedule(24) == (15, 20, 24)
    assert spread_schedule(34) == (21, 28, 34)
    assert spread_schedule(38) == (23, 31, 38)
    assert late_schedule(38) == (36, 37, 38)
    first, seed = random_schedule(height=38, base_seed=20260817, block_index=0)
    repeated, repeated_seed = random_schedule(
        height=38,
        base_seed=20260817,
        block_index=0,
    )
    assert first == repeated
    assert seed == repeated_seed
    assert first[-1] == 38
    assert len(set(first)) == 3
    assert 23 <= first[0] < first[1] <= 37
    assert seed != derived_seed("sampling", 20260817, 0, first[0])


def test_fresh_spawn_executor_reclaims_every_previous_task_process():
    with FreshSpawnPerTaskExecutor() as executor:
        first = executor.run(_isolation_worker_identity, timeout_seconds=30)
        second = executor.run(_isolation_worker_identity, timeout_seconds=30)
    assert first[0] != second[0]
    assert first[1] == second[1] == 1


def test_bank_resource_qualification_requires_fresh_simulation_and_distance():
    bank = {
        "schema_version": "fixture-bank",
        "block_count": 1,
        "resource_contract": {
            "simulation_execution": fresh_process_contract(
                TRUTH_BLOCK_SIMULATION_WORKER_UNIT
            )
        },
        "distance_execution_semantics": "fixture",
        "distance_execution_by_block": [
            {
                "schema_version": DISTANCE_EXECUTION_SCHEMA_VERSION,
                "worker_lifecycle": "fresh_spawn_process_per_condition",
            }
        ],
    }
    assert _bank_resource_execution(bank)[
        "all_bank_resources_fresh_process_qualified"
    ] is True
    bank["distance_execution_by_block"][0]["worker_lifecycle"] = (
        "inline_injected_test_double"
    )
    assert _bank_resource_execution(bank)[
        "all_bank_resources_fresh_process_qualified"
    ] is False


def test_shortlist_depth_and_placement_interactions_preserve_direction():
    arms = (ORDERED_A_ID, ORDERED_B_ID, ORDERED_C_ID, POOLED_D_ID)
    records = []
    for height, policy in ((34, "spread"), (38, "spread"), (34, "late"), (38, "late")):
        values = {arm: 0.5 for arm in arms}
        if (height, policy) == (34, "spread"):
            values[ORDERED_A_ID], values[POOLED_D_ID] = 0.6, 0.5
        elif (height, policy) == (38, "spread"):
            values[ORDERED_A_ID], values[POOLED_D_ID] = 0.5, 0.6
        elif (height, policy) == (34, "late"):
            values[ORDERED_A_ID], values[POOLED_D_ID] = 0.6, 0.5
        else:
            values[ORDERED_A_ID], values[POOLED_D_ID] = 0.8, 0.5
        records.extend(
            _success_record(0, height, policy, arm, values[arm]) for arm in arms
        )
    index = _record_index(records)
    bank = {
        "block_count": 1,
        "heights": [34, 38],
        "placement_policies": ["spread", "late"],
    }
    depth = _depth_interactions(bank=bank, index=index)
    target_depth = next(
        row
        for row in depth
        if row["arm_a"] == ORDERED_A_ID
        and row["arm_b"] == POOLED_D_ID
        and row["metric"] == "ad_f1"
        and row["placement_policy"] == "spread"
        and row["lower_height"] == 34
        and row["upper_height"] == 38
    )
    assert target_depth["effect"]["mean"] == pytest.approx(-0.2)
    placement = _placement_interactions(bank=bank, index=index)
    target_placement = next(
        row
        for row in placement
        if row["arm_a"] == ORDERED_A_ID
        and row["arm_b"] == POOLED_D_ID
        and row["metric"] == "ad_f1"
        and row["height"] == 38
        and row["policy_a"] == "late"
        and row["policy_b"] == "spread"
    )
    assert target_placement["effect"]["mean"] == pytest.approx(0.4)


def test_one_block_h38_late_bank_run_report_and_resume(tmp_path):
    bank_root = tmp_path / "bank"
    manifest = generate_bank(
        output_root=bank_root,
        block_count=1,
        heights=(38,),
        placement_policies=("late",),
        technical_preflight=True,
        distance_compute=_injected_distance,
        simulator_factory=lambda mapping, seed: _TinyH38Simulator(mapping, seed),
        created_at_utc="fixture",
    )
    assert manifest["status"] == "complete"
    assert manifest["declared_condition_count"] == 1
    assert manifest["available_condition_count"] == 1
    assert manifest["unavailable_condition_count"] == 0
    assert manifest["v2_reproduction_arm_ids"] == list(V2_COMPLETE_ARM_IDS)
    _root, loaded = load_bank_manifest(bank_root, expected_block_count=1)
    assert loaded["cases"][0]["generations"] == [36, 37, 38]

    run_root = tmp_path / "run"
    run = run_shortlist(
        bank_root=bank_root,
        output_root=run_root,
        run_id="fixture-shortlist",
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert run["status"] == "complete"
    assert run["completed_record_count"] == 4
    assert run["failure_count"] == 0
    assert run["resources"]["record_execution"] == fresh_process_contract(
        CASE_ARM_WORKER_UNIT
    )
    assert {record["placement_policy"] for record in run["records"]} == {"late"}

    interrupted = copy.deepcopy(run)
    interrupted["status"] = "failure"
    interrupted["records"] = interrupted["records"][:1]
    interrupted["completed_record_count"] = 1
    interrupted["completed_available_condition_count"] = 0
    interrupted["runner_failure"] = {
        "stage": "runner",
        "type": "FixtureInterruption",
        "message": "fixture",
    }
    interrupted.pop("completed_at_utc")
    interrupted.pop("success_count")
    interrupted.pop("failure_count")
    write_json(run_root / "result.json", interrupted)
    resumed = run_shortlist(
        bank_root=bank_root,
        output_root=run_root,
        run_id="fixture-shortlist",
        expected_block_count=1,
        resume=True,
    )
    assert resumed["completed_record_count"] == 4
    assert resumed["resume_history"][-1]["preserved_record_count"] == 1

    report_root = tmp_path / "report"
    report = write_report(
        result_root=run_root,
        output_root=report_root,
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["record_count"] == 4
    assert report["bank_resource_execution"][
        "all_bank_resources_fresh_process_qualified"
    ] is False
    assert report["interpretation_contract"][
        "pooled_1200_condition_ranking_generated"
    ] is False
    assert report["interpretation_contract"][
        "ad_f1_and_grf_combined_into_one_score"
    ] is False
    assert (report_root / "report.md").is_file()
    assert (report_root / "pairwise_by_cell.csv").is_file()

    extension_root = tmp_path / "extension-run"
    extension = run_shortlist(
        bank_root=bank_root,
        output_root=extension_root,
        run_id="fixture-shortlist-extensions",
        arm_set="v2-extensions",
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert extension["completed_record_count"] == len(V2_EXTENSION_ARM_IDS)
    assert extension["failure_count"] == 0
    assert extension["arm_ids"] == list(V2_EXTENSION_ARM_IDS)

    combined_root = tmp_path / "combined-report"
    combined = write_report(
        result_roots=(run_root, extension_root),
        output_root=combined_root,
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert combined["arm_count"] == len(V2_COMPLETE_ARM_IDS)
    assert combined["record_count"] == len(V2_COMPLETE_ARM_IDS)
    assert combined["comparison_groups"]["fully_labeled"]["arm_ids"] == list(
        V2_COMPLETE_ARM_IDS[:7]
    )
    assert combined["comparison_groups"]["partial"]["arm_ids"] == list(
        V2_COMPLETE_ARM_IDS[7:]
    )
    assert combined["interpretation_contract"][
        "cross_output_family_comparisons_generated"
    ] is False
    assert (combined_root / "partial_pairwise_by_cell.csv").is_file()

    adaptive_root = tmp_path / "adaptive-run"
    adaptive = run_shortlist(
        bank_root=bank_root,
        output_root=adaptive_root,
        run_id="fixture-shortlist-adaptive-radius",
        arm_set="adaptive-radius",
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert adaptive["completed_record_count"] == len(ADAPTIVE_RADIUS_ARM_IDS)
    assert adaptive["failure_count"] == 0
    assert adaptive["arm_ids"] == list(ADAPTIVE_RADIUS_ARM_IDS)
    d_prime = next(
        record
        for record in adaptive["records"]
        if record["arm_id"] == ADAPTIVE_D_PRIME_ID
    )
    assert d_prime["reconstruction_metadata"]["algorithm"] == "rooted_labeled_nj"
    assert d_prime["reconstruction_metadata"]["biopsy_preset"] == "default"
    assert all(
        len(
            record["reconstruction_metadata"]["biopsy_layer_decision_audit"][
                "transition_records"
            ]
        )
        == 2
        for record in adaptive["records"]
    )

    adaptive_report_root = tmp_path / "adaptive-combined-report"
    adaptive_report = write_report(
        result_roots=(run_root, extension_root, adaptive_root),
        output_root=adaptive_report_root,
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert adaptive_report["arm_count"] == (
        len(V2_COMPLETE_ARM_IDS) + len(ADAPTIVE_RADIUS_ARM_IDS)
    )
    assert adaptive_report["comparison_groups"]["fully_labeled"]["arm_ids"] == list(
        FULL_DEVELOPMENT_ARM_IDS
    )
    assert [
        row["arm_id"] for row in adaptive_report["adaptive_radius_diagnostics"]
    ] == list(ADAPTIVE_RADIUS_ARM_IDS)
    assert all(
        row["successful_case_count"] == 1
        for row in adaptive_report["adaptive_radius_diagnostics"]
    )
    assert (adaptive_report_root / "adaptive_radius_by_cell.csv").is_file()

    probe = run_probe(
        bank_root=bank_root,
        output_root=tmp_path / "isolation-probe",
        case_ids=("short-b001-H38-late",),
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert probe["completed_record_count"] == 4
    assert probe["failure_count"] == 0

    unqualified_root = tmp_path / "unqualified-run"
    unqualified_root.mkdir()
    unqualified = copy.deepcopy(resumed)
    unqualified["resources"].pop("record_execution")
    write_json(unqualified_root / "result.json", unqualified)
    with pytest.raises(ValueError, match="fresh-process"):
        write_report(
            result_root=unqualified_root,
            output_root=tmp_path / "forbidden-unqualified-report",
            expected_block_count=1,
        )

    previous_root = tmp_path / "previous-v2-run"
    previous_root.mkdir()
    previous = copy.deepcopy(resumed)
    previous["schema_version"] = PREVIOUS_RUN_SCHEMA_VERSION
    previous.pop("arm_set")
    previous.pop("arm_ids")
    write_json(previous_root / "result.json", previous)
    previous_report = write_report(
        result_root=previous_root,
        output_root=tmp_path / "previous-v2-report",
        expected_block_count=1,
    )
    assert previous_report["arm_count"] == 4

    legacy_root = tmp_path / "legacy-run"
    legacy_root.mkdir()
    legacy = copy.deepcopy(resumed)
    legacy["schema_version"] = "ctbf-v5-shortlist-robustness-run-v1"
    write_json(legacy_root / "result.json", legacy)
    with pytest.raises(ValueError, match="Unknown shortlist-robustness run schema"):
        write_report(
            result_root=legacy_root,
            output_root=tmp_path / "forbidden-legacy-report",
            expected_block_count=1,
        )


def test_one_block_full_factorial_reuses_generation_samples_and_reports_interactions(
    tmp_path,
):
    bank_root = tmp_path / "factorial-bank"
    manifest = generate_bank(
        output_root=bank_root,
        block_count=1,
        allow_nonproduction_size=True,
        distance_compute=_injected_distance,
        simulator_factory=lambda mapping, seed: _TinyH38Simulator(mapping, seed),
        created_at_utc="fixture",
    )
    assert manifest["declared_condition_count"] == 12
    assert manifest["available_condition_count"] == 12
    samples_by_generation = {}
    for case in manifest["cases"]:
        payload = read_json(bank_root / case["input_path"])
        for level in payload["levels"]:
            state_records = tuple(
                (state["state_label"], tuple(state["cnp"]))
                for state in level["states"]
            )
            generation = int(level["generation"])
            if generation in samples_by_generation:
                assert samples_by_generation[generation] == state_records
            else:
                samples_by_generation[generation] = state_records

    run_root = tmp_path / "factorial-run"
    run = run_shortlist(
        bank_root=bank_root,
        output_root=run_root,
        run_id="fixture-factorial",
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert run["completed_record_count"] == 48
    assert run["failure_count"] == 0
    report_root = tmp_path / "factorial-report"
    report = write_report(
        result_root=run_root,
        output_root=report_root,
        expected_block_count=1,
        created_at_utc="fixture",
    )
    assert len(report["algorithm_cell_summaries"]) == 48
    assert report["depth_interactions"]
    assert report["placement_interactions"]
    assert report["dependence_contract"]["independent_block_count"] == 1


def test_shortlist_bank_resume_preserves_completed_block_prefix(tmp_path):
    bank_root = tmp_path / "resume-bank"

    def interrupted_factory(_mapping, _seed):
        raise KeyboardInterrupt("fixture interruption")

    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        generate_bank(
            output_root=bank_root,
            block_count=1,
            heights=(38,),
            placement_policies=("late",),
            technical_preflight=True,
            distance_compute=_injected_distance,
            simulator_factory=interrupted_factory,
            created_at_utc="fixture",
        )
    failed = read_json(bank_root / "bank_manifest.json")
    assert failed["status"] == "failure"
    assert failed["completed_block_count"] == 0
    resumed = generate_bank(
        output_root=bank_root,
        block_count=1,
        heights=(38,),
        placement_policies=("late",),
        technical_preflight=True,
        distance_compute=_injected_distance,
        simulator_factory=lambda mapping, seed: _TinyH38Simulator(mapping, seed),
        resume=True,
    )
    assert resumed["status"] == "complete"
    assert resumed["completed_block_count"] == 1
    assert resumed["resume_history"][-1]["preserved_block_count"] == 0


def test_parallel_distance_workers_preserve_serial_inputs_and_distances(tmp_path):
    if multiprocessing.cpu_count() < 2:
        pytest.skip("Parallel shortlist smoke requires at least two CPUs.")
    common = {
        "block_count": 1,
        "allow_nonproduction_size": True,
        "distance_compute": _injected_distance,
        "simulator_factory": lambda mapping, seed: _TinyH38Simulator(mapping, seed),
        "created_at_utc": "fixture",
    }
    serial_root = tmp_path / "serial-bank"
    serial = generate_bank(
        output_root=serial_root,
        distance_workers=1,
        **common,
    )
    parallel_root = tmp_path / "parallel-bank"
    parallel = generate_bank(
        output_root=parallel_root,
        distance_workers=6,
        **common,
    )

    assert [case["case_id"] for case in parallel["cases"]] == [
        case["case_id"] for case in serial["cases"]
    ]
    for serial_case, parallel_case in zip(serial["cases"], parallel["cases"]):
        assert read_json(serial_root / serial_case["input_path"]) == read_json(
            parallel_root / parallel_case["input_path"]
        )
        assert read_json(serial_root / serial_case["distance_path"]) == read_json(
            parallel_root / parallel_case["distance_path"]
        )
    serial_execution = serial["distance_execution_by_block"][0]
    parallel_execution = parallel["distance_execution_by_block"][0]
    assert serial_execution["scheduler"] == (
        "serial_inline_injected_test_condition"
    )
    assert serial_execution["worker_lifecycle"] == "inline_injected_test_double"
    assert serial_execution["effective_worker_count"] == 1
    assert parallel_execution["scheduler"] == (
        "spawn_process_pool_fresh_conditions"
    )
    assert parallel_execution["worker_lifecycle"] == (
        "fresh_spawn_process_per_condition"
    )
    expected_parallel_workers = min(6, multiprocessing.cpu_count())
    assert parallel_execution["requested_worker_count"] == 6
    assert parallel_execution["effective_worker_count"] == expected_parallel_workers
    assert parallel_execution["distance_task_count"] == 12
    assert parallel_execution["task_submission_order"] == (
        "descending_unique_profile_count_then_declared_ordinal"
    )
    parallel_metadata = read_json(
        parallel_root / parallel["cases"][0]["metadata_path"]
    )
    assert parallel_metadata["distance_execution"][
        "effective_worker_count"
    ] == expected_parallel_workers
    assert parallel_metadata["distance_execution"][
        "within_condition_order_execution"
    ] == "sequential_forward_then_reverse"


def test_parallel_resume_migrates_and_preserves_serial_block_prefix(tmp_path):
    if multiprocessing.cpu_count() < 2:
        pytest.skip("Parallel shortlist resume smoke requires at least two CPUs.")
    bank_root = tmp_path / "mixed-bank"
    complete = generate_bank(
        output_root=bank_root,
        block_count=2,
        allow_nonproduction_size=True,
        distance_workers=1,
        distance_compute=_injected_distance,
        simulator_factory=lambda mapping, seed: _TinyH38Simulator(mapping, seed),
        created_at_utc="fixture",
    )
    first_block_cases = [
        case for case in complete["cases"] if int(case["block_index"]) == 0
    ]
    preserved_distances = {
        case["case_id"]: read_json(bank_root / case["distance_path"])
        for case in first_block_cases
    }
    interrupted = copy.deepcopy(complete)
    interrupted["status"] = "in_progress"
    interrupted["completed_block_count"] = 1
    interrupted["condition_inventory"] = [
        record
        for record in interrupted["condition_inventory"]
        if int(record["block_index"]) == 0
    ]
    interrupted["cases"] = first_block_cases
    interrupted["completed_condition_count"] = 12
    interrupted["available_condition_count"] = 12
    interrupted["unavailable_condition_count"] = 0
    interrupted.pop("completed_at_utc")
    interrupted.pop("distance_execution_semantics")
    interrupted.pop("distance_execution_by_block")
    write_json(bank_root / "bank_manifest.json", interrupted)

    resumed = generate_bank(
        output_root=bank_root,
        block_count=2,
        allow_nonproduction_size=True,
        distance_workers=2,
        distance_compute=_injected_distance,
        simulator_factory=lambda mapping, seed: _TinyH38Simulator(mapping, seed),
        resume=True,
    )
    assert resumed["status"] == "complete"
    assert resumed["completed_block_count"] == 2
    assert resumed["resume_history"][-1]["preserved_block_count"] == 1
    assert resumed["resume_history"][-1][
        "preserved_execution_prefix_was_inferred"
    ] is True
    assert resumed["resume_history"][-1]["requested_distance_worker_count"] == 2
    assert resumed["distance_execution_by_block"][0]["record_origin"] == (
        "inferred_pre_parallel_option_serial_prefix"
    )
    assert resumed["distance_execution_by_block"][0]["requested_worker_count"] == 1
    assert resumed["distance_execution_by_block"][1]["requested_worker_count"] == 2
    _root, loaded = load_bank_manifest(bank_root, expected_block_count=2)
    assert loaded["distance_execution_by_block"] == resumed[
        "distance_execution_by_block"
    ]
    for case in first_block_cases:
        assert read_json(bank_root / case["distance_path"]) == preserved_distances[
            case["case_id"]
        ]


def test_distance_keyboard_interrupt_is_not_scientific_unavailability(tmp_path):
    bank_root = tmp_path / "distance-interrupt-bank"
    with pytest.raises(KeyboardInterrupt, match="distance fixture interruption"):
        generate_bank(
            output_root=bank_root,
            block_count=1,
            heights=(38,),
            placement_policies=("late",),
            technical_preflight=True,
            distance_compute=_interrupting_distance,
            simulator_factory=lambda mapping, seed: _TinyH38Simulator(mapping, seed),
            created_at_utc="fixture",
        )
    failed = read_json(bank_root / "bank_manifest.json")
    assert failed["status"] == "failure"
    assert failed["completed_block_count"] == 0
    assert failed["completed_condition_count"] == 0
    assert failed["condition_inventory"] == []
    assert failed["runner_failure"]["type"] == "KeyboardInterrupt"


def test_guarded_h38_condition_is_unavailable_and_never_replaced(tmp_path):
    manifest = generate_bank(
        output_root=tmp_path / "guarded-bank",
        block_count=1,
        heights=(38,),
        placement_policies=("late",),
        technical_preflight=True,
        distance_compute=_injected_distance,
        simulator_factory=lambda mapping, seed: _GuardedTinyH38Simulator(
            mapping, seed
        ),
        created_at_utc="fixture",
    )
    assert manifest["status"] == "complete"
    assert manifest["available_condition_count"] == 0
    assert manifest["unavailable_condition_count"] == 1
    condition = manifest["condition_inventory"][0]
    assert condition["status"] == "unavailable"
    assert condition["failure"]["type"] == "SimulationResourceLimitExceeded"
    assert condition["failure"]["simulation_outcome"]["failure_generation"] == 38
