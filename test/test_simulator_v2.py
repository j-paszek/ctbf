import csv
import json
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from simulator import CancerCellEvolutionSimulator, Genotype
from simulator_config import (
    SIMULATOR_SEMANTIC_VERSION,
    choose_crucial_mask,
    load_simulator_inputs,
)
from simulator_events import (
    CNAEventProposal,
    apply_event_sequence,
    count_edge_events,
    propose_event_sequence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_EXAMPLES = PROJECT_ROOT / "simulator_examples"


def v2_config(**overrides):
    config = {
        "SIMULATOR_SEMANTIC_VERSION": SIMULATOR_SEMANTIC_VERSION,
        "GENOME_LENGTH": 4,
        "INITIAL_COPY_NUMBER": 1,
        "NUMBER_OF_GENERATIONS": 1,
        "OFFSPRING_MODEL": "constant",
        "OFFSPRING_PARAMETER": 0,
        "BASELINE_DESCENDANT_ATTEMPTS": 1,
        "CNA_EVENT_PROBABILITY": 0.0,
        "GAIN_GIVEN_CNA_PROBABILITY": 0.5,
        "INTERVAL_CNA_PROBABILITY": 0.0,
        "INTERVAL_GAIN_OPERATOR_PROBABILITIES": {
            "unit": 1.0,
            "additive": 0.0,
            "multiplicative": 0.0,
        },
        "ADDITIVE_GAIN_LAMBDA": 0.0,
        "MULTIPLICATIVE_FACTOR_PROBABILITIES": {"2": 1.0},
        "WGD_PROBABILITY": 0.0,
        "REPRESENTATION_TYPE": "representative",
        "TELOMERIC_INSTABILITY_ENABLED": False,
        "TELOMERIC_INSTABILITY_INCREMENT": 0.0,
        "TELOMERIC_FRACTION": 0.0,
        "CRUCIAL_SURVIVAL_ENABLED": False,
    }
    config.update(overrides)
    return config


def bed_config(**overrides):
    config = v2_config(**overrides)
    config.pop("GENOME_LENGTH")
    config.pop("INITIAL_COPY_NUMBER")
    return config


def write_bed(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ChromosomeNumber", "Start", "End", "Parameters"])
        writer.writerows(rows)
    return path


def proposal(
    proposal_id,
    time,
    start,
    end,
    *,
    direction,
    operator="unit",
    magnitude=1,
    event_class="interval_mode_cna",
    chromosome="1",
):
    return CNAEventProposal(
        proposal_id=proposal_id,
        within_generation_time=time,
        event_class=event_class,
        chromosome=chromosome,
        start_index=start,
        end_index=end,
        direction=direction,
        operator=operator,
        magnitude=magnitude,
    )


def test_v2_configuration_rejects_unknown_and_legacy_fields():
    legacy = v2_config(GENERAL_LOSS_PROB=0.5)
    with pytest.raises(ValueError, match="Unknown simulator configuration keys"):
        CancerCellEvolutionSimulator(legacy, seed=1)

    wrong_version = v2_config(SIMULATOR_SEMANTIC_VERSION="v1")
    with pytest.raises(ValueError, match="SIMULATOR_SEMANTIC_VERSION"):
        CancerCellEvolutionSimulator(wrong_version, seed=1)

    fitness = v2_config(OFFSPRING_MODEL="fitness")
    with pytest.raises(ValueError, match="fitness is deferred"):
        CancerCellEvolutionSimulator(fitness, seed=1)

    with pytest.raises(ValueError, match="fixes BASELINE_DESCENDANT_ATTEMPTS at 1"):
        CancerCellEvolutionSimulator(
            v2_config(BASELINE_DESCENDANT_ATTEMPTS=0), seed=1
        )


def test_total_descendant_attempts_are_baseline_one_plus_additional_count():
    simulator = CancerCellEvolutionSimulator(
        v2_config(
            OFFSPRING_MODEL="constant",
            OFFSPRING_PARAMETER=2,
            REPRESENTATION_TYPE="full",
        ),
        seed=1,
    )
    simulator.run_simulation()

    totals = simulator.diagnostics_snapshot()["totals"]
    assert len(simulator.tree) == 4
    assert totals["attempted_children"] == 3
    assert totals["viable_children"] == 3
    assert totals["retained_children"] == 3


@pytest.mark.parametrize(
    ("event_probability", "gain_probability", "expected", "event_count"),
    [
        (0.0, 1.0, [1, 1, 1, 1], 0),
        (1.0, 1.0, [2, 2, 2, 2], 4),
        (1.0, 0.0, [0, 0, 0, 0], 4),
    ],
)
def test_p8_two_stage_probability_extremes(
    event_probability,
    gain_probability,
    expected,
    event_count,
):
    simulator = CancerCellEvolutionSimulator(
        v2_config(
            CNA_EVENT_PROBABILITY=event_probability,
            GAIN_GIVEN_CNA_PROBABILITY=gain_probability,
        ),
        seed=4,
    )
    simulator.run_simulation()

    assert simulator.tree.nodes[1]["genome"] == expected
    assert len(simulator.tree.edges[0, 1]["events"]) == event_count


def test_point_unit_and_interval_gain_operators_are_distinct():
    bins = load_simulator_inputs(v2_config()).genome_bins

    point = apply_event_sequence(
        np.array([2, 2, 2, 2]),
        [
            proposal(
                0,
                0.1,
                1,
                1,
                direction="gain",
                event_class="point_unit_cna",
            )
        ],
        bins,
        crucial_survival_enabled=False,
    )
    additive = apply_event_sequence(
        np.array([2, 2, 2, 2]),
        [proposal(0, 0.1, 1, 3, direction="gain", operator="additive", magnitude=3)],
        bins,
        crucial_survival_enabled=False,
    )
    multiplicative = apply_event_sequence(
        np.array([2, 2, 2, 2]),
        [
            proposal(
                0,
                0.1,
                1,
                3,
                direction="gain",
                operator="multiplicative",
                magnitude=3,
            )
        ],
        bins,
        crucial_survival_enabled=False,
    )

    assert point.genome.tolist() == [2, 3, 2, 2]
    assert additive.genome.tolist() == [2, 5, 5, 5]
    assert multiplicative.genome.tolist() == [2, 6, 6, 6]
    assert point.edge_records[0].event_class == "point_unit_cna"
    assert additive.edge_records[0].multi_position is True


def test_interval_mode_may_realize_one_position():
    bins = load_simulator_inputs(v2_config()).genome_bins
    result = apply_event_sequence(
        np.array([1, 1, 1, 1]),
        [proposal(0, 0.1, 3, 3, direction="gain", operator="additive", magnitude=2)],
        bins,
        crucial_survival_enabled=False,
    )

    assert result.genome.tolist() == [1, 1, 1, 3]
    assert result.edge_records[0].event_class == "interval_mode_cna"
    assert result.edge_records[0].multi_position is False


def test_overlapping_events_use_recorded_time_not_genomic_coordinate():
    bins = load_simulator_inputs(v2_config(GENOME_LENGTH=1)).genome_bins
    loss_then_gain = apply_event_sequence(
        np.array([1]),
        [
            proposal(1, 0.8, 0, 0, direction="gain"),
            proposal(0, 0.2, 0, 0, direction="loss"),
        ],
        bins,
        crucial_survival_enabled=False,
    )
    gain_then_loss = apply_event_sequence(
        np.array([1]),
        [
            proposal(0, 0.2, 0, 0, direction="gain"),
            proposal(1, 0.8, 0, 0, direction="loss"),
        ],
        bins,
        crucial_survival_enabled=False,
    )

    assert loss_then_gain.genome.tolist() == [0]
    assert gain_then_loss.genome.tolist() == [1]
    assert gain_then_loss.net_zero_sequence is True
    assert gain_then_loss.edge_records == ()


def test_net_zero_child_is_retained_with_parent_state_identity(monkeypatch):
    simulator = CancerCellEvolutionSimulator(v2_config(GENOME_LENGTH=1), seed=5)
    result = apply_event_sequence(
        np.array([1]),
        [
            proposal(0, 0.1, 0, 0, direction="gain"),
            proposal(1, 0.2, 0, 0, direction="loss"),
        ],
        simulator.genome_bins,
        crucial_survival_enabled=False,
    )
    monkeypatch.setattr(simulator, "_apply_copy_number_events", lambda _genome: result)
    simulator.run_simulation()

    assert simulator.tree.nodes[1]["genome"] == [1]
    assert simulator.tree.nodes[1]["cell_id"] == 0
    assert simulator.tree.edges[0, 1]["events"] == []
    assert simulator.tree.edges[0, 1]["event_count"] == 0
    assert simulator.diagnostics_snapshot()["totals"]["net_zero_sequences"] == 1


def test_zero_is_absorbing_and_ineffective_events_leave_no_truth_event():
    bins = load_simulator_inputs(v2_config(GENOME_LENGTH=1, INITIAL_COPY_NUMBER=0)).genome_bins
    result = apply_event_sequence(
        np.array([0]),
        [proposal(0, 0.1, 0, 0, direction="gain", operator="additive", magnitude=5)],
        bins,
        crucial_survival_enabled=False,
    )

    assert result.genome.tolist() == [0]
    assert result.attempted_records[0].effective is False
    assert result.net_zero_sequence is True
    assert result.edge_records == ()


def test_explicit_wgd_is_one_typed_event_and_preserves_zero():
    bins = load_simulator_inputs(v2_config(GENOME_LENGTH=3)).genome_bins
    result = apply_event_sequence(
        np.array([1, 0, 3]),
        [
            proposal(
                0,
                0.1,
                0,
                2,
                direction="gain",
                operator="whole_genome_doubling",
                magnitude=2,
                event_class="whole_genome_doubling",
                chromosome=None,
            )
        ],
        bins,
        crucial_survival_enabled=False,
    )

    assert result.genome.tolist() == [2, 0, 6]
    assert len(result.edge_records) == 1
    assert result.edge_records[0].event_class == "whole_genome_doubling"
    assert result.edge_records[0].as_dict()["multiplication_factor"] == 2


def test_crucial_loss_rejects_child_but_control_survives():
    base_bin = load_simulator_inputs(v2_config(GENOME_LENGTH=1)).genome_bins[0]
    crucial_bins = (replace(base_bin, crucial=True),)
    loss = [proposal(0, 0.1, 0, 0, direction="loss")]

    rejected = apply_event_sequence(
        np.array([1]),
        loss,
        crucial_bins,
        crucial_survival_enabled=True,
    )
    control = apply_event_sequence(
        np.array([1]),
        loss,
        crucial_bins,
        crucial_survival_enabled=False,
    )

    assert rejected.genome is None
    assert rejected.rejected_by_crucial_survival is True
    assert control.genome.tolist() == [0]

    rejected_before_rescue = apply_event_sequence(
        np.array([1]),
        [
            proposal(0, 0.1, 0, 0, direction="loss"),
            proposal(1, 0.2, 0, 0, direction="gain"),
        ],
        crucial_bins,
        crucial_survival_enabled=True,
    )
    assert rejected_before_rescue.genome is None
    assert len(rejected_before_rescue.attempted_records) == 1


def test_bed_overrides_use_start_row_and_chromosome_bounds(tmp_path):
    bed = write_bed(
        tmp_path / "genome.csv",
        [
            ["1", 0, 10, "CN=1;CNA_EVENT_PROBABILITY=1;GAIN_GIVEN_CNA_PROBABILITY=1"],
            ["1", 10, 20, "CN=1;CNA_EVENT_PROBABILITY=1;GAIN_GIVEN_CNA_PROBABILITY=1"],
            ["2", 0, 10, "CN=1;CNA_EVENT_PROBABILITY=1;GAIN_GIVEN_CNA_PROBABILITY=0"],
            ["2", 10, 20, "CN=1;CNA_EVENT_PROBABILITY=1;GAIN_GIVEN_CNA_PROBABILITY=0"],
        ],
    )
    inputs = load_simulator_inputs(
        bed_config(
            CNA_EVENT_PROBABILITY=0.0,
            GAIN_GIVEN_CNA_PROBABILITY=0.5,
            INTERVAL_CNA_PROBABILITY=1.0,
        ),
        bed,
    )
    simulator = CancerCellEvolutionSimulator(
        bed_config(
            CNA_EVENT_PROBABILITY=0.0,
            GAIN_GIVEN_CNA_PROBABILITY=0.5,
            INTERVAL_CNA_PROBABILITY=1.0,
        ),
        bed,
        seed=8,
    )
    proposals = propose_event_sequence(
        inputs.genome_bins,
        wgd_probability=0.0,
        rng_streams=simulator._rng_streams,
    )

    assert all(
        inputs.genome_bins[event.start_index].chromosome
        == inputs.genome_bins[event.end_index].chromosome
        for event in proposals
    )
    assert {event.direction for event in proposals if event.start_index < 2} == {"gain"}
    assert {event.direction for event in proposals if event.start_index >= 2} == {"loss"}


def test_telomeric_initiation_is_resolved_per_chromosome_and_start_row(tmp_path):
    rows = []
    for chromosome in ("1", "2"):
        for position in range(5):
            params = "CN=1"
            if chromosome == "1" and position == 2:
                params += ";TELOMERIC_INSTABILITY=1"
            rows.append([chromosome, position * 10, (position + 1) * 10, params])
    bed = write_bed(tmp_path / "telomeric.csv", rows)
    config = bed_config(
        CNA_EVENT_PROBABILITY=0.0,
        GAIN_GIVEN_CNA_PROBABILITY=1.0,
        TELOMERIC_INSTABILITY_ENABLED=True,
        TELOMERIC_INSTABILITY_INCREMENT=1.0,
        TELOMERIC_FRACTION=0.2,
    )
    simulator = CancerCellEvolutionSimulator(config, bed, seed=3)
    proposals = propose_event_sequence(
        simulator.genome_bins,
        wgd_probability=0.0,
        rng_streams=simulator._rng_streams,
    )

    assert sorted(event.start_index for event in proposals) == [0, 2, 4, 5, 9]


def test_bed_validation_rejects_overlap_and_unknown_parameter(tmp_path):
    overlapping = write_bed(
        tmp_path / "overlap.csv",
        [
            ["1", 0, 10, "CN=1"],
            ["1", 9, 20, "CN=1"],
        ],
    )
    with pytest.raises(ValueError, match="unsorted or overlaps"):
        load_simulator_inputs(bed_config(), overlapping)

    unknown = write_bed(
        tmp_path / "unknown.csv",
        [["1", 0, 10, "CN=1;LOSS_PROBABILITY=0.5"]],
    )
    with pytest.raises(ValueError, match="Unknown BED-like parameter"):
        load_simulator_inputs(bed_config(), unknown)


def test_representative_collision_keeps_first_parent_and_reports_cross_parent(monkeypatch):
    simulator = CancerCellEvolutionSimulator(v2_config(GENOME_LENGTH=1), seed=3)
    simulator.genotypes = {
        0: Genotype([1], node_id=0, generation=0, cell_id=0),
        10: Genotype([3], node_id=10, generation=0, cell_id=10),
    }
    simulator.tree = nx.DiGraph()
    simulator.tree.add_node(0, genome=[1], generation=0, cell_id=0)
    simulator.tree.add_node(10, genome=[3], generation=0, cell_id=10)
    simulator.node_counter = 11
    bins = simulator.genome_bins

    def converge_to_two(genome):
        direction = "gain" if int(genome[0]) == 1 else "loss"
        return apply_event_sequence(
            genome,
            [proposal(0, 0.1, 0, 0, direction=direction)],
            bins,
            crucial_survival_enabled=False,
        )

    monkeypatch.setattr(simulator, "_apply_copy_number_events", converge_to_two)
    simulator._spawn_children(1)

    assert set(simulator.tree.nodes) == {0, 10, 11}
    assert list(simulator.tree.predecessors(11)) == [0]
    diagnostics = simulator.diagnostics_snapshot()["totals"]
    assert diagnostics["representative_collisions"] == 1
    assert diagnostics["cross_parent_representative_collisions"] == 1


def test_owned_rng_does_not_mutate_numpy_process_global_state():
    np.random.seed(123)
    first = np.random.random()
    CancerCellEvolutionSimulator(v2_config(), seed=9)
    second = np.random.random()

    np.random.seed(123)
    assert first == np.random.random()
    assert second == np.random.random()


def test_same_seed_reproduces_tree_events_and_diagnostics():
    config = v2_config(
        NUMBER_OF_GENERATIONS=3,
        OFFSPRING_MODEL="poisson",
        OFFSPRING_PARAMETER=1.0,
        CNA_EVENT_PROBABILITY=0.4,
        GAIN_GIVEN_CNA_PROBABILITY=0.6,
        INTERVAL_CNA_PROBABILITY=0.5,
        INTERVAL_GAIN_OPERATOR_PROBABILITIES={
            "unit": 0.4,
            "additive": 0.3,
            "multiplicative": 0.3,
        },
        WGD_PROBABILITY=0.1,
    )
    first = CancerCellEvolutionSimulator(config, seed=77)
    second = CancerCellEvolutionSimulator(config, seed=77)
    first.run_simulation()
    second.run_simulation()

    assert list(first.tree.nodes(data=True)) == list(second.tree.nodes(data=True))
    assert list(first.tree.edges(data=True)) == list(second.tree.edges(data=True))
    assert first.diagnostics_snapshot() == second.diagnostics_snapshot()


def test_typed_events_drive_truth_distance_and_newick_event_count():
    simulator = CancerCellEvolutionSimulator(
        v2_config(CNA_EVENT_PROBABILITY=1.0, GAIN_GIVEN_CNA_PROBABILITY=1.0),
        seed=5,
    )
    simulator.run_simulation()

    edge_events = simulator.tree.edges[0, 1]["events"]
    assert count_edge_events(edge_events) == 4
    assert {
        "event_class",
        "chromosome",
        "start_coordinate",
        "end_coordinate",
        "footprint_length",
        "direction",
        "operator",
        "magnitude",
        "before",
        "after",
        "changed_positions",
        "effective",
    } <= set(edge_events[0])
    labels, matrix = simulator.to_distance_matrix(node_list=[0, 1], labels=[0, 1])
    assert labels == [0, 1]
    assert matrix.tolist() == [[0.0, 4.0], [4.0, 0.0]]


def test_exact_random_ten_percent_crucial_mask_is_reproducible():
    first = choose_crucial_mask(100, 0.10, seed=91)
    second = choose_crucial_mask(100, 0.10, seed=91)

    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10
    with pytest.raises(ValueError, match="needs no rounding rule"):
        choose_crucial_mask(11, 0.10, seed=91)


def test_tracked_ten_percent_crucial_pair_uses_one_recorded_mask():
    manifest = json.loads(
        (SIMULATOR_EXAMPLES / "crucial_10pct_pair_manifest.json").read_text()
    )
    selected = choose_crucial_mask(
        manifest["mask_selection"]["eligible_row_count"],
        manifest["mask_selection"]["selected_fraction"],
        seed=manifest["mask_selection"]["mask_seed"],
    )
    assert list(selected) == manifest["mask_selection"]["zero_based_selected_rows"]

    crucial_mapping = json.loads(
        (SIMULATOR_EXAMPLES / manifest["crucial_config"]).read_text()
    )
    control_mapping = json.loads(
        (SIMULATOR_EXAMPLES / manifest["control_config"]).read_text()
    )
    assert {
        key
        for key in crucial_mapping
        if crucial_mapping[key] != control_mapping[key]
    } == {"CRUCIAL_SURVIVAL_ENABLED"}

    bed = SIMULATOR_EXAMPLES / manifest["shared_bed"]
    crucial = CancerCellEvolutionSimulator(
        SIMULATOR_EXAMPLES / manifest["crucial_config"], bed, seed=73
    )
    control = CancerCellEvolutionSimulator(
        SIMULATOR_EXAMPLES / manifest["control_config"], bed, seed=73
    )
    assert tuple(
        index for index, genome_bin in enumerate(crucial.genome_bins) if genome_bin.crucial
    ) == selected
    assert crucial.config.crucial_survival_enabled is True
    assert control.config.crucial_survival_enabled is False

    crucial.run_simulation()
    control.run_simulation()
    crucial_totals = crucial.diagnostics_snapshot()["totals"]
    control_totals = control.diagnostics_snapshot()["totals"]
    assert crucial_totals["crucial_rejections"] > 0
    assert (
        crucial_totals["attempted_children"]
        == crucial_totals["viable_children"] + crucial_totals["crucial_rejections"]
    )
    assert "crucial_rejections" not in control_totals
    assert control_totals["attempted_children"] == control_totals["viable_children"]


def test_tracked_default_v2_configuration_runs_without_bed():
    simulator = CancerCellEvolutionSimulator(
        SIMULATOR_EXAMPLES / "default.json", seed=9
    )
    simulator.run_simulation()
    assert simulator.semantic_version == SIMULATOR_SEMANTIC_VERSION
    assert len(simulator.tree) > 1
    provenance = simulator.provenance()
    assert provenance["seed"] == 9
    assert provenance["bit_generator"] == "PCG64"
    assert set(provenance["rng_stream_spawn_keys"]) == set(
        CancerCellEvolutionSimulator._RNG_STREAM_NAMES
    )
    assert len(provenance["config_sha256"]) == 64
    assert provenance["bed_sha256"] is None


def test_bed_export_round_trips_resolved_v2_parameters(tmp_path):
    simulator = CancerCellEvolutionSimulator(v2_config(), seed=2)
    bed = tmp_path / "roundtrip.csv"
    simulator.create_bed_csv(bed)

    reloaded = CancerCellEvolutionSimulator(bed_config(), bed, seed=2)
    assert reloaded.founder_genome.tolist() == simulator.founder_genome.tolist()
    assert reloaded.genome_bins == simulator.genome_bins


def test_from_tree_owns_a_copy_and_seeded_biopsy_is_deterministic():
    tree = nx.DiGraph()
    tree.add_node(0, genome=[1], generation=0, cell_id=0)
    tree.add_node(1, genome=[2], generation=1, cell_id=1)
    tree.add_node(2, genome=[3], generation=1, cell_id=2)
    tree.add_edge(0, 1, events=[])
    tree.add_edge(0, 2, events=[])
    simulator = CancerCellEvolutionSimulator.from_tree(tree)

    tree.nodes[1]["genome"][0] = 99
    first = simulator.perform_biopsy(generation=1, biopsy_size=1, seed=8)
    second = simulator.perform_biopsy(generation=1, biopsy_size=1, seed=8)

    assert simulator.tree.nodes[1]["genome"] == [2]
    assert simulator.provenance()["simulator_semantic_version"] == "external-tree"
    assert [(item.node_id, item.cell_id) for item in first] == [
        (item.node_id, item.cell_id) for item in second
    ]
