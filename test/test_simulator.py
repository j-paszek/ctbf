import csv
import json
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from simulator import (
    CancerCellEvolutionSimulator,
    Genotype,
    SimulationDiagnostics,
    SimulationResourceLimitExceeded,
    state_lineage_survival_probability,
)
from simulator_config import (
    SIMULATOR_SEMANTIC_VERSION,
    choose_crucial_mask,
    cna_initiation_schedule_at_generation,
    load_simulator_inputs,
)
from simulator_events import (
    CNAEventProposal,
    apply_event_sequence,
    count_edge_events,
    propose_event_sequence,
    segmental_initiation_probabilities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_EXAMPLES = PROJECT_ROOT / "simulator_examples"


def active_config(**overrides):
    config = {
        "SIMULATOR_SEMANTIC_VERSION": SIMULATOR_SEMANTIC_VERSION,
        "GENOME_LENGTH": 4,
        "NUMBER_OF_CHROMOSOMES": 1,
        "INITIAL_COPY_NUMBER": 1,
        "CRUCIAL_BIN_INDICES": [],
        "NUMBER_OF_GENERATIONS": 1,
        "OFFSPRING_MODEL": "constant",
        "OFFSPRING_PARAMETER": 0,
        "BASELINE_DESCENDANT_ATTEMPTS": 1,
        "CNA_EVENT_PROBABILITY": 0.0,
        "CNA_INITIATION_SCHEDULE": {"MODEL": "constant"},
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
        "STATE_LINEAGE_REGULATION": {"MODEL": "none"},
        "RESOURCE_GUARD": {
            "MAX_REPRESENTATIVES_PER_GENERATION": 2000,
            "MAX_TOTAL_NODES": 40000,
        },
        "TELOMERIC_INSTABILITY_ENABLED": False,
        "TELOMERIC_INSTABILITY_INCREMENT": 0.0,
        "TELOMERIC_FRACTION": 0.0,
        "CRUCIAL_SURVIVAL_ENABLED": False,
    }
    config.update(overrides)
    return config


def bed_config(**overrides):
    config = active_config(**overrides)
    config.pop("GENOME_LENGTH")
    config.pop("NUMBER_OF_CHROMOSOMES")
    config.pop("INITIAL_COPY_NUMBER")
    config.pop("CRUCIAL_BIN_INDICES")
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
    initiation_index=None,
    footprint_direction=None,
):
    if event_class == "whole_genome_doubling":
        resolved_initiation_index = None
        resolved_footprint_direction = "whole_genome"
    elif event_class == "point_unit_cna":
        resolved_initiation_index = start
        resolved_footprint_direction = "point"
    else:
        resolved_initiation_index = (
            start if initiation_index is None else initiation_index
        )
        resolved_footprint_direction = (
            "right" if footprint_direction is None else footprint_direction
        )
    return CNAEventProposal(
        proposal_id=proposal_id,
        within_generation_time=time,
        event_class=event_class,
        chromosome=chromosome,
        initiation_index=resolved_initiation_index,
        start_index=start,
        end_index=end,
        footprint_direction=resolved_footprint_direction,
        direction=direction,
        operator=operator,
        magnitude=magnitude,
    )


def test_active_configuration_rejects_unknown_and_obsolete_fields():
    unknown_fields = active_config(GENERAL_LOSS_PROB=0.5)
    with pytest.raises(ValueError, match="Unknown simulator configuration keys"):
        CancerCellEvolutionSimulator(unknown_fields, seed=1)

    wrong_version = active_config(
        SIMULATOR_SEMANTIC_VERSION="ctbf-cnp-state-simulator-unsupported"
    )
    with pytest.raises(ValueError, match="SIMULATOR_SEMANTIC_VERSION"):
        CancerCellEvolutionSimulator(wrong_version, seed=1)

    fitness = active_config(OFFSPRING_MODEL="fitness")
    with pytest.raises(ValueError, match="fitness is not part of CTBF v5"):
        CancerCellEvolutionSimulator(fitness, seed=1)

    with pytest.raises(ValueError, match="fixes BASELINE_DESCENDANT_ATTEMPTS at 1"):
        CancerCellEvolutionSimulator(
            active_config(BASELINE_DESCENDANT_ATTEMPTS=0), seed=1
        )


def test_no_bed_chromosome_layout_is_required_strict_and_deterministic():
    missing = active_config()
    missing.pop("NUMBER_OF_CHROMOSOMES")
    with pytest.raises(
        ValueError,
        match="Missing required simulator configuration keys",
    ):
        CancerCellEvolutionSimulator(missing, seed=1)

    with pytest.raises(ValueError, match="must not exceed GENOME_LENGTH"):
        CancerCellEvolutionSimulator(
            active_config(GENOME_LENGTH=3, NUMBER_OF_CHROMOSOMES=4),
            seed=1,
        )

    missing_crucial_indices = active_config()
    missing_crucial_indices.pop("CRUCIAL_BIN_INDICES")
    with pytest.raises(
        ValueError,
        match="Missing required simulator configuration keys",
    ):
        CancerCellEvolutionSimulator(missing_crucial_indices, seed=1)

    for invalid_indices in ([1, 1], [2, 1], [4]):
        with pytest.raises(ValueError, match="CRUCIAL_BIN_INDICES"):
            CancerCellEvolutionSimulator(
                active_config(CRUCIAL_BIN_INDICES=invalid_indices),
                seed=1,
            )

    crucial_inputs = load_simulator_inputs(
        active_config(CRUCIAL_BIN_INDICES=[0, 3])
    )
    assert [
        index
        for index, genome_bin in enumerate(crucial_inputs.genome_bins)
        if genome_bin.crucial
    ] == [0, 3]

    with pytest.raises(ValueError, match="requires at least one crucial bin"):
        CancerCellEvolutionSimulator(
            active_config(CRUCIAL_SURVIVAL_ENABLED=True),
            seed=1,
        )

    with pytest.raises(ValueError, match="founder genome cannot be entirely"):
        CancerCellEvolutionSimulator(
            active_config(INITIAL_COPY_NUMBER=0),
            seed=1,
        )

    inputs = load_simulator_inputs(
        active_config(GENOME_LENGTH=10, NUMBER_OF_CHROMOSOMES=3)
    )
    assert [genome_bin.chromosome for genome_bin in inputs.genome_bins] == [
        "1",
        "1",
        "1",
        "1",
        "2",
        "2",
        "2",
        "3",
        "3",
        "3",
    ]
    assert [
        (genome_bin.start, genome_bin.end) for genome_bin in inputs.genome_bins
    ] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 1),
        (1, 2),
        (2, 3),
    ]

    bed_mapping = bed_config()
    bed_mapping["NUMBER_OF_CHROMOSOMES"] = 2
    with pytest.raises(ValueError, match="must be omitted when BED-like input"):
        load_simulator_inputs(bed_mapping, PROJECT_ROOT / "simulator_examples" / "crucial_10pct_bed.csv")


def test_state_lineage_regulation_configuration_is_strict():
    missing = active_config()
    missing.pop("STATE_LINEAGE_REGULATION")
    with pytest.raises(ValueError, match="Missing required simulator configuration keys"):
        CancerCellEvolutionSimulator(missing, seed=1)

    none = CancerCellEvolutionSimulator(
        active_config(),
        seed=1,
    )
    assert none.config.state_lineage_regulation.model == "none"
    assert none.config.state_lineage_regulation.capacity is None

    with pytest.raises(ValueError, match="Unknown STATE_LINEAGE_REGULATION keys"):
        CancerCellEvolutionSimulator(
            active_config(
                STATE_LINEAGE_REGULATION={"MODEL": "none", "CAPACITY": 10}
            ),
            seed=1,
        )
    with pytest.raises(ValueError, match="CAPACITY must be >= 1"):
        CancerCellEvolutionSimulator(
            active_config(
                STATE_LINEAGE_REGULATION={
                    "MODEL": "soft_capacity",
                    "CAPACITY": 0,
                    "STEEPNESS": 2.0,
                }
            ),
            seed=1,
        )
    with pytest.raises(ValueError, match="STEEPNESS must be > 0"):
        CancerCellEvolutionSimulator(
            active_config(
                STATE_LINEAGE_REGULATION={
                    "MODEL": "soft_capacity",
                    "CAPACITY": 10,
                    "STEEPNESS": 0.0,
                }
            ),
            seed=1,
        )
    with pytest.raises(ValueError, match="requires REPRESENTATION_TYPE"):
        CancerCellEvolutionSimulator(
            active_config(
                REPRESENTATION_TYPE="full",
                STATE_LINEAGE_REGULATION={
                    "MODEL": "soft_capacity",
                    "CAPACITY": 10,
                    "STEEPNESS": 2.0,
                },
            ),
            seed=1,
        )


def test_cna_initiation_schedule_configuration_is_strict():
    missing = active_config()
    missing.pop("CNA_INITIATION_SCHEDULE")
    with pytest.raises(
        ValueError,
        match="Missing required simulator configuration keys",
    ):
        CancerCellEvolutionSimulator(missing, seed=1)

    constant = CancerCellEvolutionSimulator(active_config(), seed=1)
    assert constant.cna_initiation_schedule.model == "constant"
    assert constant.cna_initiation_schedule.initial_multiplier == 1.0
    assert constant.cna_initiation_schedule.final_multiplier == 1.0
    assert constant.cna_initiation_schedule.decay_exponent is None

    with pytest.raises(ValueError, match="Unknown CNA_INITIATION_SCHEDULE keys"):
        CancerCellEvolutionSimulator(
            active_config(
                CNA_INITIATION_SCHEDULE={
                    "MODEL": "constant",
                    "FINAL_MULTIPLIER": 1.0,
                }
            ),
            seed=1,
        )

    early_burst = {
        "MODEL": "early_burst_decay",
        "INITIAL_MULTIPLIER": 2.0,
        "FINAL_MULTIPLIER": 0.25,
        "DECAY_EXPONENT": 2.0,
    }
    parsed = CancerCellEvolutionSimulator(
        active_config(CNA_INITIATION_SCHEDULE=early_burst),
        seed=1,
    )
    assert parsed.cna_initiation_schedule.model == "early_burst_decay"

    for field, value, message in (
        ("FINAL_MULTIPLIER", 0.0, "FINAL_MULTIPLIER must be > 0"),
        ("INITIAL_MULTIPLIER", 0.25, "must be greater than FINAL_MULTIPLIER"),
        ("DECAY_EXPONENT", 0.0, "DECAY_EXPONENT must be > 0"),
    ):
        invalid = dict(early_burst)
        invalid[field] = value
        with pytest.raises(ValueError, match=message):
            CancerCellEvolutionSimulator(
                active_config(CNA_INITIATION_SCHEDULE=invalid),
                seed=1,
            )


def test_horizon_normalized_cna_schedule_has_exact_endpoints_and_late_floor():
    simulator = CancerCellEvolutionSimulator(
        active_config(
            NUMBER_OF_GENERATIONS=5,
            CNA_INITIATION_SCHEDULE={
                "MODEL": "early_burst_decay",
                "INITIAL_MULTIPLIER": 4.0,
                "FINAL_MULTIPLIER": 0.25,
                "DECAY_EXPONENT": 2.0,
            },
        ),
        seed=1,
    )
    points = [
        cna_initiation_schedule_at_generation(
            simulator.cna_initiation_schedule,
            generation=generation,
            number_of_generations=5,
        )
        for generation in range(1, 6)
    ]

    assert [point[0] for point in points] == pytest.approx(
        [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    multipliers = [point[1] for point in points]
    assert multipliers[0] == pytest.approx(4.0)
    assert multipliers[-1] == pytest.approx(0.25)
    assert all(
        first > second > 0.0
        for first, second in zip(multipliers, multipliers[1:])
    )
    assert cna_initiation_schedule_at_generation(
        simulator.cna_initiation_schedule,
        generation=1,
        number_of_generations=1,
    ) == pytest.approx((0.0, 4.0))


def test_schedule_scales_combined_base_and_telomeric_hazard_before_capping():
    genome_bin = replace(
        load_simulator_inputs(active_config(GENOME_LENGTH=1)).genome_bins[0],
        cna_event_probability=0.8,
        telomeric_instability=0.8,
    )
    probabilities = segmental_initiation_probabilities(
        [genome_bin],
        initiation_multiplier=0.5,
    )

    assert probabilities.tolist() == pytest.approx([0.8])
    assert segmental_initiation_probabilities(
        [genome_bin],
        initiation_multiplier=2.0,
    ).tolist() == pytest.approx([1.0])
    for invalid_multiplier in (0.0, True, "1.0", float("nan")):
        with pytest.raises(ValueError, match="finite and > 0"):
            segmental_initiation_probabilities(
                [genome_bin],
                initiation_multiplier=invalid_multiplier,
            )


def test_resource_guard_configuration_is_strict():
    missing = active_config()
    missing.pop("RESOURCE_GUARD")
    with pytest.raises(ValueError, match="Missing required simulator configuration keys"):
        CancerCellEvolutionSimulator(missing, seed=1)

    simulator = CancerCellEvolutionSimulator(active_config(), seed=1)
    assert simulator.resource_guard.max_representatives_per_generation == 2000
    assert simulator.resource_guard.max_total_nodes == 40000

    with pytest.raises(ValueError, match="Unknown RESOURCE_GUARD keys"):
        CancerCellEvolutionSimulator(
            active_config(
                RESOURCE_GUARD={
                    "MAX_REPRESENTATIVES_PER_GENERATION": 2000,
                    "MAX_TOTAL_NODES": 40000,
                    "MODEL": "none",
                }
            ),
            seed=1,
        )

    for key, value in (
        ("MAX_REPRESENTATIVES_PER_GENERATION", 0),
        ("MAX_REPRESENTATIVES_PER_GENERATION", True),
        ("MAX_TOTAL_NODES", 0),
        ("MAX_TOTAL_NODES", True),
    ):
        guard = {
            "MAX_REPRESENTATIVES_PER_GENERATION": 2000,
            "MAX_TOTAL_NODES": 40000,
        }
        guard[key] = value
        with pytest.raises(ValueError, match=key):
            CancerCellEvolutionSimulator(
                active_config(RESOURCE_GUARD=guard),
                seed=1,
            )


def test_resource_guard_allows_exact_generation_boundary_and_types_breach():
    exact = CancerCellEvolutionSimulator(
        active_config(
            REPRESENTATION_TYPE="full",
            OFFSPRING_PARAMETER=2,
            RESOURCE_GUARD={
                "MAX_REPRESENTATIVES_PER_GENERATION": 3,
                "MAX_TOTAL_NODES": 4,
            },
        ),
        seed=1,
    )
    exact.run_simulation()

    assert len(exact.tree) == 4
    assert exact.tree.graph["simulation_outcome"]["status"] == "completed"
    assert exact.provenance()["resource_guard"] == {
        "max_representatives_per_generation": 3,
        "max_total_nodes": 4,
    }

    limited = CancerCellEvolutionSimulator(
        active_config(
            REPRESENTATION_TYPE="full",
            OFFSPRING_PARAMETER=2,
            RESOURCE_GUARD={
                "MAX_REPRESENTATIVES_PER_GENERATION": 2,
                "MAX_TOTAL_NODES": 10,
            },
        ),
        seed=1,
    )
    with pytest.raises(SimulationResourceLimitExceeded) as caught:
        limited.run_simulation()

    error = caught.value
    assert error.limit_name == "MAX_REPRESENTATIVES_PER_GENERATION"
    assert error.configured_limit == 2
    assert error.attempted_count == 3
    assert error.generation == 1
    assert len(limited.tree) == 3
    assert set(limited.genotypes) == set(limited.tree)
    assert error.simulation_outcome == limited.tree.graph["simulation_outcome"]
    assert error.simulation_outcome == {
        "status": "resource_limit_exceeded",
        "configured_final_generation": 1,
        "last_retained_generation": 1,
        "extinction_generation": None,
        "failure_generation": 1,
        "resource_limit": {
            "limit_name": "MAX_REPRESENTATIVES_PER_GENERATION",
            "configured_limit": 2,
            "attempted_count": 3,
            "generation": 1,
        },
    }
    assert limited.diagnostics_snapshot()["totals"]["resource_guard_aborts"] == 1


def test_resource_guard_types_total_node_breach():
    simulator = CancerCellEvolutionSimulator(
        active_config(
            REPRESENTATION_TYPE="full",
            OFFSPRING_PARAMETER=2,
            RESOURCE_GUARD={
                "MAX_REPRESENTATIVES_PER_GENERATION": 3,
                "MAX_TOTAL_NODES": 3,
            },
        ),
        seed=1,
    )
    with pytest.raises(SimulationResourceLimitExceeded) as caught:
        simulator.run_simulation()

    assert caught.value.limit_name == "MAX_TOTAL_NODES"
    assert caught.value.configured_limit == 3
    assert caught.value.attempted_count == 4
    assert len(simulator.tree) == 3
    assert set(simulator.genotypes) == set(simulator.tree)


def test_nonbinding_resource_guard_preserves_seeded_soft_regulation_history():
    base = active_config(
        NUMBER_OF_GENERATIONS=3,
        OFFSPRING_PARAMETER=1,
        CNA_EVENT_PROBABILITY=0.4,
        STATE_LINEAGE_REGULATION={
            "MODEL": "soft_capacity",
            "CAPACITY": 100,
            "STEEPNESS": 2.0,
        },
    )
    first = CancerCellEvolutionSimulator(
        active_config(
            **{
                **base,
                "RESOURCE_GUARD": {
                    "MAX_REPRESENTATIVES_PER_GENERATION": 100,
                    "MAX_TOTAL_NODES": 1000,
                },
            }
        ),
        seed=77,
    )
    second = CancerCellEvolutionSimulator(
        active_config(
            **{
                **base,
                "RESOURCE_GUARD": {
                    "MAX_REPRESENTATIVES_PER_GENERATION": 200,
                    "MAX_TOTAL_NODES": 2000,
                },
            }
        ),
        seed=77,
    )
    first.run_simulation()
    second.run_simulation()

    assert list(first.tree.nodes(data=True)) == list(second.tree.nodes(data=True))
    assert list(first.tree.edges(data=True)) == list(second.tree.edges(data=True))
    assert first.diagnostics_snapshot() == second.diagnostics_snapshot()


def test_state_lineage_survival_response_is_bounded_and_capacity_balanced():
    at_capacity = state_lineage_survival_probability(
        state_count=100,
        unregulated_expected_descendant_attempts=2.0,
        capacity=100,
        steepness=2.0,
    )
    below_capacity = state_lineage_survival_probability(
        state_count=10,
        unregulated_expected_descendant_attempts=2.0,
        capacity=100,
        steepness=2.0,
    )
    above_capacity = state_lineage_survival_probability(
        state_count=1000,
        unregulated_expected_descendant_attempts=2.0,
        capacity=100,
        steepness=2.0,
    )
    three_attempts_at_capacity = state_lineage_survival_probability(
        state_count=100,
        unregulated_expected_descendant_attempts=3.0,
        capacity=100,
        steepness=2.0,
    )

    assert at_capacity == pytest.approx(0.5)
    assert three_attempts_at_capacity == pytest.approx(1.0 / 3.0)
    assert 0.0 < above_capacity < at_capacity < below_capacity < 1.0
    assert state_lineage_survival_probability(
        state_count=10**12,
        unregulated_expected_descendant_attempts=2.0,
        capacity=1,
        steepness=100.0,
    ) == pytest.approx(0.0)
    assert state_lineage_survival_probability(
        state_count=10,
        unregulated_expected_descendant_attempts=1.0,
        capacity=1,
        steepness=2.0,
    ) == 1.0


class _FixedLineageSurvivalRng:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def random(self, size):
        assert size == len(self.values)
        return self.values.copy()


def test_soft_state_lineage_regulation_precedes_child_generation_and_types_extinction(
    monkeypatch,
):
    simulator = CancerCellEvolutionSimulator(
        active_config(
            NUMBER_OF_GENERATIONS=3,
            OFFSPRING_PARAMETER=1,
            STATE_LINEAGE_REGULATION={
                "MODEL": "soft_capacity",
                "CAPACITY": 1,
                "STEEPNESS": 1.0,
            },
        ),
        seed=7,
    )
    simulator._rng_streams[simulator._STATE_LINEAGE_SURVIVAL_RNG_STREAM] = (
        _FixedLineageSurvivalRng([0.75])
    )

    def events_must_not_be_generated(_genome, _initiation_multiplier):
        raise AssertionError("Extinct state lineage reached child event generation.")

    monkeypatch.setattr(
        simulator,
        "_apply_copy_number_events",
        events_must_not_be_generated,
    )
    simulator.run_simulation()

    assert list(simulator.tree.nodes) == [0]
    assert simulator.tree.graph["simulation_outcome"] == {
        "status": "extinct",
        "configured_final_generation": 3,
        "last_retained_generation": 0,
        "extinction_generation": 1,
    }
    diagnostics = simulator.diagnostics_snapshot()
    assert diagnostics["totals"] == {
        "population_extinctions": 1,
        "state_lineage_extinctions": 1,
        "state_lineages_considered": 1,
        "state_lineages_continued": 0,
        "zero_burden_profiles": 1,
    }
    assert diagnostics["state_lineage_regulation_by_generation"]["1"] == {
        "model": "soft_capacity",
        "parent_state_count": 1,
        "unregulated_expected_descendant_attempts": 2.0,
        "capacity": 1,
        "steepness": 1.0,
        "survival_probability": 0.5,
        "continued_lineage_count": 0,
        "extinct_lineage_count": 1,
    }
    assert simulator.tree.graph["simulation_diagnostics"] == diagnostics


def test_soft_state_lineage_continuation_preserves_existing_child_semantics():
    simulator = CancerCellEvolutionSimulator(
        active_config(
            OFFSPRING_PARAMETER=1,
            STATE_LINEAGE_REGULATION={
                "MODEL": "soft_capacity",
                "CAPACITY": 1,
                "STEEPNESS": 1.0,
            },
        ),
        seed=7,
    )
    simulator._rng_streams[simulator._STATE_LINEAGE_SURVIVAL_RNG_STREAM] = (
        _FixedLineageSurvivalRng([0.25])
    )
    simulator.run_simulation()

    totals = simulator.diagnostics_snapshot()["totals"]
    assert len(simulator.tree) == 2
    assert totals["state_lineages_continued"] == 1
    assert totals["attempted_children"] == 2
    assert totals["representative_collisions"] == 1
    assert totals["retained_children"] == 1
    assert simulator.tree.graph["simulation_outcome"]["status"] == "completed"


def test_total_descendant_attempts_are_baseline_one_plus_additional_count():
    simulator = CancerCellEvolutionSimulator(
        active_config(
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
    (
        "event_probability",
        "gain_probability",
        "initial_copy_number",
        "expected",
        "event_count",
    ),
    [
        (0.0, 1.0, 1, [1, 1, 1, 1], 0),
        (1.0, 1.0, 1, [2, 2, 2, 2], 4),
        (1.0, 0.0, 2, [1, 1, 1, 1], 4),
    ],
)
def test_p8_two_stage_probability_extremes(
    event_probability,
    gain_probability,
    initial_copy_number,
    expected,
    event_count,
):
    simulator = CancerCellEvolutionSimulator(
        active_config(
            CNA_EVENT_PROBABILITY=event_probability,
            GAIN_GIVEN_CNA_PROBABILITY=gain_probability,
            INITIAL_COPY_NUMBER=initial_copy_number,
        ),
        seed=4,
    )
    simulator.run_simulation()

    assert simulator.tree.nodes[1]["genome"] == expected
    assert len(simulator.tree.edges[0, 1]["events"]) == event_count


def test_point_unit_and_interval_gain_operators_are_distinct():
    bins = load_simulator_inputs(active_config()).genome_bins

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
    bins = load_simulator_inputs(active_config()).genome_bins
    result = apply_event_sequence(
        np.array([1, 1, 1, 1]),
        [proposal(0, 0.1, 3, 3, direction="gain", operator="additive", magnitude=2)],
        bins,
        crucial_survival_enabled=False,
    )

    assert result.genome.tolist() == [1, 1, 1, 3]
    assert result.edge_records[0].event_class == "interval_mode_cna"
    assert result.edge_records[0].multi_position is False


class _FixedFootprintRng:
    def __init__(self, direction_draws, endpoint_draws):
        self.draw_batches = [
            np.asarray(direction_draws, dtype=float),
            np.asarray(endpoint_draws, dtype=float),
        ]

    def random(self, size):
        values = self.draw_batches.pop(0)
        assert len(values) == size
        return values.copy()


def test_interval_footprints_are_bidirectional_anchored_and_chromosome_bounded():
    simulator = CancerCellEvolutionSimulator(
        active_config(
            GENOME_LENGTH=4,
            NUMBER_OF_CHROMOSOMES=1,
            CNA_EVENT_PROBABILITY=1.0,
            GAIN_GIVEN_CNA_PROBABILITY=1.0,
            INTERVAL_CNA_PROBABILITY=1.0,
        ),
        seed=4,
    )
    simulator._rng_streams["footprint"] = _FixedFootprintRng(
        direction_draws=[0.6, 0.1, 0.6, 0.1],
        endpoint_draws=[0.99, 0.99, 0.99, 0.99],
    )
    proposals = propose_event_sequence(
        simulator.genome_bins,
        wgd_probability=0.0,
        initiation_multiplier=1.0,
        rng_streams=simulator._rng_streams,
    )

    by_anchor = {proposal.initiation_index: proposal for proposal in proposals}
    assert {
        anchor: (
            event.start_index,
            event.end_index,
            event.footprint_direction,
        )
        for anchor, event in by_anchor.items()
    } == {
        0: (0, 3, "right"),
        1: (0, 1, "left"),
        2: (2, 3, "right"),
        3: (0, 3, "left"),
    }
    left_result = apply_event_sequence(
        np.array([1, 1, 1, 1]),
        [by_anchor[3]],
        simulator.genome_bins,
        crucial_survival_enabled=False,
    )
    left_record = left_result.edge_records[0].as_dict()
    assert left_record["initiation_index"] == 3
    assert left_record["initiation_coordinate"] == 3
    assert (left_record["start_index"], left_record["end_index"]) == (0, 3)
    assert left_record["footprint_direction"] == "left"

    two_chromosomes = CancerCellEvolutionSimulator(
        active_config(
            GENOME_LENGTH=4,
            NUMBER_OF_CHROMOSOMES=2,
            CNA_EVENT_PROBABILITY=1.0,
            INTERVAL_CNA_PROBABILITY=1.0,
        ),
        seed=4,
    )
    two_chromosomes._rng_streams["footprint"] = _FixedFootprintRng(
        direction_draws=[0.6, 0.1, 0.6, 0.1],
        endpoint_draws=[0.99, 0.99, 0.99, 0.99],
    )
    bounded = propose_event_sequence(
        two_chromosomes.genome_bins,
        wgd_probability=0.0,
        initiation_multiplier=1.0,
        rng_streams=two_chromosomes._rng_streams,
    )
    assert all(
        two_chromosomes.genome_bins[event.start_index].chromosome
        == two_chromosomes.genome_bins[event.end_index].chromosome
        for event in bounded
    )


def test_interval_singleton_retains_sampled_direction_and_anchor():
    simulator = CancerCellEvolutionSimulator(
        active_config(
            CNA_EVENT_PROBABILITY=1.0,
            INTERVAL_CNA_PROBABILITY=1.0,
        ),
        seed=5,
    )
    simulator._rng_streams["footprint"] = _FixedFootprintRng(
        direction_draws=[0.1, 0.6, 0.1, 0.6],
        endpoint_draws=[0.0, 0.0, 0.0, 0.0],
    )
    proposals = propose_event_sequence(
        simulator.genome_bins,
        wgd_probability=0.0,
        initiation_multiplier=1.0,
        rng_streams=simulator._rng_streams,
    )

    assert all(event.start_index == event.end_index for event in proposals)
    assert all(event.start_index == event.initiation_index for event in proposals)
    assert [
        event.footprint_direction
        for event in sorted(proposals, key=lambda event: event.initiation_index)
    ] == [
        "left",
        "right",
        "left",
        "right",
    ]


def test_overlapping_events_use_recorded_time_not_genomic_coordinate():
    bins = load_simulator_inputs(active_config(GENOME_LENGTH=2)).genome_bins
    loss_then_gain = apply_event_sequence(
        np.array([1, 1]),
        [
            proposal(1, 0.8, 0, 0, direction="gain"),
            proposal(0, 0.2, 0, 0, direction="loss"),
        ],
        bins,
        crucial_survival_enabled=False,
    )
    gain_then_loss = apply_event_sequence(
        np.array([1, 1]),
        [
            proposal(0, 0.2, 0, 0, direction="gain"),
            proposal(1, 0.8, 0, 0, direction="loss"),
        ],
        bins,
        crucial_survival_enabled=False,
    )

    assert loss_then_gain.genome.tolist() == [0, 1]
    assert gain_then_loss.genome.tolist() == [1, 1]
    assert gain_then_loss.net_zero_sequence is True
    assert gain_then_loss.edge_records == ()


def test_net_zero_child_is_retained_with_parent_state_identity(monkeypatch):
    simulator = CancerCellEvolutionSimulator(active_config(GENOME_LENGTH=1), seed=5)
    result = apply_event_sequence(
        np.array([1]),
        [
            proposal(0, 0.1, 0, 0, direction="gain"),
            proposal(1, 0.2, 0, 0, direction="loss"),
        ],
        simulator.genome_bins,
        crucial_survival_enabled=False,
    )
    monkeypatch.setattr(
        simulator,
        "_apply_copy_number_events",
        lambda _genome, _initiation_multiplier: result,
    )
    simulator.run_simulation()

    assert simulator.tree.nodes[1]["genome"] == [1]
    assert simulator.tree.nodes[1]["cell_id"] == 0
    assert simulator.tree.edges[0, 1]["events"] == []
    assert simulator.tree.edges[0, 1]["event_count"] == 0
    assert simulator.diagnostics_snapshot()["totals"]["net_zero_sequences"] == 1


def test_zero_is_absorbing_and_ineffective_events_leave_no_truth_event():
    bins = load_simulator_inputs(active_config(GENOME_LENGTH=2)).genome_bins
    result = apply_event_sequence(
        np.array([0, 1]),
        [proposal(0, 0.1, 0, 0, direction="gain", operator="additive", magnitude=5)],
        bins,
        crucial_survival_enabled=False,
    )

    assert result.genome.tolist() == [0, 1]
    assert result.attempted_records[0].effective is False
    assert result.net_zero_sequence is True
    assert result.edge_records == ()


def test_explicit_wgd_is_one_typed_event_and_preserves_zero():
    bins = load_simulator_inputs(active_config(GENOME_LENGTH=3)).genome_bins
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
    base_bins = load_simulator_inputs(active_config(GENOME_LENGTH=2)).genome_bins
    crucial_bins = (replace(base_bins[0], crucial=True), base_bins[1])
    loss = [proposal(0, 0.1, 0, 0, direction="loss")]

    rejected = apply_event_sequence(
        np.array([1, 1]),
        loss,
        crucial_bins,
        crucial_survival_enabled=True,
    )
    control = apply_event_sequence(
        np.array([1, 1]),
        loss,
        crucial_bins,
        crucial_survival_enabled=False,
    )

    assert rejected.genome is None
    assert rejected.viability_rejection_reason == "crucial_bin_zero"
    assert control.genome.tolist() == [0, 1]

    rejected_before_rescue = apply_event_sequence(
        np.array([1, 1]),
        [
            proposal(0, 0.1, 0, 0, direction="loss"),
            proposal(
                1,
                0.2,
                0,
                0,
                direction="gain",
                event_class="point_unit_cna",
            ),
        ],
        crucial_bins,
        crucial_survival_enabled=True,
    )
    assert rejected_before_rescue.genome is None
    assert len(rejected_before_rescue.proposed_events) == 2
    assert len(rejected_before_rescue.attempted_records) == 1

    already_zero = apply_event_sequence(
        np.array([0, 1]),
        [],
        crucial_bins,
        crucial_survival_enabled=True,
    )
    assert already_zero.viability_rejection_reason == "crucial_bin_zero"
    assert already_zero.attempted_records == ()


def test_all_zero_genome_is_universally_nonviable():
    bins = load_simulator_inputs(active_config(GENOME_LENGTH=2)).genome_bins
    result = apply_event_sequence(
        np.array([1, 1]),
        [proposal(0, 0.1, 0, 1, direction="loss")],
        bins,
        crucial_survival_enabled=False,
    )

    assert result.genome is None
    assert result.viability_rejection_reason == "all_zero_genome"
    assert len(result.attempted_records) == 1

    already_zero = apply_event_sequence(
        np.array([0, 0]),
        [],
        bins,
        crucial_survival_enabled=False,
    )
    assert already_zero.genome is None
    assert already_zero.viability_rejection_reason == "all_zero_genome"
    assert already_zero.attempted_records == ()


def test_zero_burden_diagnostics_respect_chromosome_boundaries():
    diagnostics = SimulationDiagnostics()
    diagnostics.record_retained_profile(
        3,
        np.array([0, 0, 0, 0]),
        ("1", "1", "2", "2"),
    )
    diagnostics.record_retained_profile(
        3,
        np.array([2, 0, 0, 2]),
        ("1", "1", "2", "2"),
    )

    snapshot = diagnostics.snapshot()
    assert snapshot["zero_burden_by_generation"]["3"] == {
        "profile_count": 2,
        "zero_bin_count_histogram": {"2": 1, "4": 1},
        "longest_contiguous_zero_run_histogram": {"1": 1, "2": 1},
    }
    assert snapshot["totals"]["profiles_at_or_above_50pct_cn0"] == 2
    assert snapshot["totals"]["retained_all_zero_profiles"] == 1


def test_proposed_footprints_are_audited_before_viability_selection():
    bins = load_simulator_inputs(active_config(GENOME_LENGTH=3)).genome_bins
    result = apply_event_sequence(
        np.array([1, 1, 1]),
        [
            proposal(0, 0.1, 0, 2, direction="loss"),
            proposal(
                1,
                0.2,
                0,
                0,
                direction="gain",
                event_class="point_unit_cna",
            ),
        ],
        bins,
        crucial_survival_enabled=False,
    )
    diagnostics = SimulationDiagnostics()
    diagnostics.record_proposed_segmental_events(1, result.proposed_events)
    diagnostics.record_retained_segmental_events(1, result.edge_records)
    snapshot = diagnostics.snapshot()

    assert result.viability_rejection_reason == "all_zero_genome"
    assert len(result.proposed_events) == 2
    assert len(result.attempted_records) == 1
    assert result.edge_records == ()
    assert snapshot["totals"] == {
        "proposed_interval_mode_cna_loss_events": 1,
        "proposed_point_unit_cna_gain_events": 1,
        "proposed_segmental_gain_events": 1,
        "proposed_segmental_loss_events": 1,
    }
    footprints = snapshot["interval_footprint_length_histograms"]
    assert footprints["proposed"]["totals"]["loss"] == {"3": 1}
    assert footprints["retained"]["totals"]["loss"] == {}


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
        initiation_multiplier=1.0,
        rng_streams=simulator._rng_streams,
    )

    assert all(
        inputs.genome_bins[event.start_index].chromosome
        == inputs.genome_bins[event.end_index].chromosome
        for event in proposals
    )
    assert {
        event.direction for event in proposals if event.initiation_index < 2
    } == {"gain"}
    assert {
        event.direction for event in proposals if event.initiation_index >= 2
    } == {"loss"}


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
        initiation_multiplier=1.0,
        rng_streams=simulator._rng_streams,
    )

    assert sorted(event.initiation_index for event in proposals) == [0, 2, 4, 5, 9]


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
    simulator = CancerCellEvolutionSimulator(active_config(GENOME_LENGTH=1), seed=3)
    simulator.genotypes = {
        0: Genotype([1], node_id=0, generation=0, cell_id=0),
        10: Genotype([3], node_id=10, generation=0, cell_id=10),
    }
    simulator.tree = nx.DiGraph()
    simulator.tree.add_node(0, genome=[1], generation=0, cell_id=0)
    simulator.tree.add_node(10, genome=[3], generation=0, cell_id=10)
    simulator.node_counter = 11
    bins = simulator.genome_bins

    def converge_to_two(genome, _initiation_multiplier):
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
    CancerCellEvolutionSimulator(active_config(), seed=9)
    second = np.random.random()

    np.random.seed(123)
    assert first == np.random.random()
    assert second == np.random.random()


def test_same_seed_reproduces_tree_events_and_diagnostics():
    config = active_config(
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


def test_cna_schedule_records_exposure_and_does_not_add_an_rng_stream():
    simulator = CancerCellEvolutionSimulator(
        active_config(
            GENOME_LENGTH=2,
            NUMBER_OF_GENERATIONS=3,
            REPRESENTATION_TYPE="full",
            CNA_EVENT_PROBABILITY=0.4,
            GAIN_GIVEN_CNA_PROBABILITY=1.0,
            CNA_INITIATION_SCHEDULE={
                "MODEL": "early_burst_decay",
                "INITIAL_MULTIPLIER": 2.0,
                "FINAL_MULTIPLIER": 0.25,
                "DECAY_EXPONENT": 2.0,
            },
        ),
        seed=17,
    )
    simulator.run_simulation()

    provenance = simulator.provenance()
    assert provenance["cna_initiation_schedule"] == {
        "model": "early_burst_decay",
        "time_basis": "horizon_normalized",
        "initial_multiplier": 2.0,
        "final_multiplier": 0.25,
        "decay_exponent": 2.0,
    }
    assert set(provenance["rng_stream_spawn_keys"]) == set(
        simulator._RNG_STREAM_NAMES
    )
    assert "cna_initiation_schedule" not in provenance["rng_stream_spawn_keys"]

    audit = simulator.diagnostics_snapshot()[
        "cna_initiation_schedule_by_generation"
    ]
    assert [
        audit[str(generation)]["initiation_multiplier"]
        for generation in range(1, 4)
    ] == pytest.approx([2.0, 0.6875, 0.25])
    assert [
        audit[str(generation)]["expected_segmental_starts_per_attempt"]
        for generation in range(1, 4)
    ] == pytest.approx(
        [1.6, 0.55, 0.2]
    )
    for generation in range(1, 4):
        generation_audit = audit[str(generation)]
        assert generation_audit["time_basis"] == "horizon_normalized"
        assert generation_audit["attempted_children"] == 1
        assert generation_audit[
            "expected_segmental_starts_over_attempts"
        ] == pytest.approx(generation_audit["expected_segmental_starts_per_attempt"])


def test_cna_schedule_changes_only_initiation_threshold_when_all_starts_are_certain():
    shared = active_config(
        NUMBER_OF_GENERATIONS=1,
        REPRESENTATION_TYPE="full",
        CNA_EVENT_PROBABILITY=1.0,
        INTERVAL_CNA_PROBABILITY=1.0,
        WGD_PROBABILITY=1.0,
    )
    constant = CancerCellEvolutionSimulator(shared, seed=29)
    early_burst = CancerCellEvolutionSimulator(
        active_config(
            **{
                **shared,
                "CNA_INITIATION_SCHEDULE": {
                    "MODEL": "early_burst_decay",
                    "INITIAL_MULTIPLIER": 2.0,
                    "FINAL_MULTIPLIER": 0.25,
                    "DECAY_EXPONENT": 2.0,
                },
            }
        ),
        seed=29,
    )

    constant.run_simulation()
    early_burst.run_simulation()

    assert list(constant.tree.nodes(data=True)) == list(
        early_burst.tree.nodes(data=True)
    )
    assert list(constant.tree.edges(data=True)) == list(
        early_burst.tree.edges(data=True)
    )
    assert constant._rng_spawn_keys == early_burst._rng_spawn_keys


def test_typed_events_drive_truth_distance_and_newick_event_count():
    simulator = CancerCellEvolutionSimulator(
        active_config(CNA_EVENT_PROBABILITY=1.0, GAIN_GIVEN_CNA_PROBABILITY=1.0),
        seed=5,
    )
    simulator.run_simulation()

    edge_events = simulator.tree.edges[0, 1]["events"]
    assert count_edge_events(edge_events) == 4
    assert {
        "event_class",
        "chromosome",
        "initiation_index",
        "initiation_coordinate",
        "start_coordinate",
        "end_coordinate",
        "footprint_length",
        "footprint_direction",
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

    standard_mapping = json.loads(
        (SIMULATOR_EXAMPLES / manifest["standard_no_bed_config"]).read_text()
    )
    assert standard_mapping["CRUCIAL_BIN_INDICES"] == list(selected)
    assert standard_mapping["CRUCIAL_SURVIVAL_ENABLED"] is True

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
    stress = manifest["integration_test_stress_override"]
    crucial_stress_mapping = dict(crucial_mapping)
    control_stress_mapping = dict(control_mapping)
    crucial_stress_mapping["CNA_EVENT_PROBABILITY"] = stress[
        "CNA_EVENT_PROBABILITY"
    ]
    control_stress_mapping["CNA_EVENT_PROBABILITY"] = stress[
        "CNA_EVENT_PROBABILITY"
    ]
    crucial = CancerCellEvolutionSimulator(
        crucial_stress_mapping, bed, seed=stress["simulation_seed"]
    )
    control = CancerCellEvolutionSimulator(
        control_stress_mapping, bed, seed=stress["simulation_seed"]
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
    assert crucial_totals["crucial_bin_zero_rejections"] > 0
    assert (
        crucial_totals["attempted_children"]
        == crucial_totals["viable_children"]
        + crucial_totals["viability_rejections"]
    )
    assert "crucial_bin_zero_rejections" not in control_totals
    assert control_totals["attempted_children"] == control_totals["viable_children"]


def test_tracked_default_configuration_runs_without_bed():
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
    assert provenance["viability_model"] == {
        "all_zero_genome_nonviable": True,
        "crucial_survival_enabled": True,
        "crucial_bin_indices": [4, 7, 38, 40, 41, 49, 54, 61, 69, 72],
        "crucial_bin_count": 10,
        "crucial_bin_fraction": pytest.approx(0.1),
    }


def test_tracked_soft_state_lineage_example_runs_without_bed():
    simulator = CancerCellEvolutionSimulator(
        SIMULATOR_EXAMPLES / "state_lineage_soft_capacity.json",
        seed=9,
    )
    simulator.run_simulation()

    assert simulator.semantic_version == SIMULATOR_SEMANTIC_VERSION
    assert simulator.state_lineage_regulation.model == "soft_capacity"
    assert simulator.state_lineage_regulation.capacity == 1000
    assert simulator.state_lineage_regulation.steepness == 2.0
    assert "state_lineage_survival" in simulator.provenance()[
        "rng_stream_spawn_keys"
    ]
    assert simulator.tree.graph["simulation_outcome"]["status"] in {
        "completed",
        "extinct",
    }
    assert "state_lineage_regulation_by_generation" in (
        simulator.diagnostics_snapshot()
    )


def test_tracked_early_burst_example_runs_without_regulation():
    simulator = CancerCellEvolutionSimulator(
        SIMULATOR_EXAMPLES / "cna_early_burst_decay.json",
        seed=9,
    )
    simulator.run_simulation()

    assert simulator.cna_initiation_schedule.model == "early_burst_decay"
    assert simulator.state_lineage_regulation.model == "none"
    audit = simulator.diagnostics_snapshot()[
        "cna_initiation_schedule_by_generation"
    ]
    assert audit["1"]["initiation_multiplier"] == pytest.approx(2.0)
    assert audit["6"]["initiation_multiplier"] == pytest.approx(0.25)
    assert simulator.tree.graph["simulation_outcome"]["status"] == "completed"


@pytest.mark.parametrize(
    "relative_path",
    [
        "default.json",
        "state_lineage_soft_capacity.json",
        "cna_early_burst_decay.json",
        "crucial_10pct.json",
        "crucial_10pct_control.json",
        "paper_v5/clean_balanced.json",
        "paper_v5/clean_interval_gain.json",
        "paper_v5/clean_telomeric.json",
        "paper_v5/crucial_10pct_control.json",
        "paper_v5/crucial_10pct_survival.json",
        "paper_v5/wgd_1pct.json",
    ],
)
def test_tracked_v5_configs_use_the_100_row_crc_organoid_standard(relative_path):
    mapping = json.loads((SIMULATOR_EXAMPLES / relative_path).read_text())
    assert mapping["CNA_EVENT_PROBABILITY"] == pytest.approx(0.1 / 100)
    assert mapping["GAIN_GIVEN_CNA_PROBABILITY"] == pytest.approx(0.5)
    assert mapping["INTERVAL_GAIN_OPERATOR_PROBABILITIES"] == {
        "unit": 0.8,
        "additive": 0.2,
        "multiplicative": 0,
    }
    assert mapping["ADDITIVE_GAIN_LAMBDA"] == 0
    assert mapping["MULTIPLICATIVE_FACTOR_PROBABILITIES"] == {"2": 1}
    if "GENOME_LENGTH" in mapping:
        assert mapping["NUMBER_OF_CHROMOSOMES"] == 1
        assert mapping["CRUCIAL_BIN_INDICES"] == [
            4,
            7,
            38,
            40,
            41,
            49,
            54,
            61,
            69,
            72,
        ]
    else:
        assert "NUMBER_OF_CHROMOSOMES" not in mapping
        assert "CRUCIAL_BIN_INDICES" not in mapping


@pytest.mark.parametrize(
    "relative_path",
    [
        "default.json",
        "state_lineage_soft_capacity.json",
        "cna_early_burst_decay.json",
        "paper_v5/clean_balanced.json",
        "paper_v5/clean_interval_gain.json",
        "paper_v5/clean_telomeric.json",
        "paper_v5/crucial_10pct_survival.json",
        "paper_v5/wgd_1pct.json",
    ],
)
def test_tracked_v5_standard_configs_enable_the_fixed_viability_prior(
    relative_path,
):
    mapping = json.loads((SIMULATOR_EXAMPLES / relative_path).read_text())

    assert mapping["CRUCIAL_SURVIVAL_ENABLED"] is True
    assert len(mapping["CRUCIAL_BIN_INDICES"]) == 10


def test_tracked_paper_configs_are_strict_and_paired_by_one_field():
    paper_configs = SIMULATOR_EXAMPLES / "paper_v5"
    clean_paths = [
        paper_configs / "clean_balanced.json",
        paper_configs / "clean_interval_gain.json",
        paper_configs / "clean_telomeric.json",
        paper_configs / "wgd_1pct.json",
    ]
    for path in clean_paths:
        simulator = CancerCellEvolutionSimulator(path, seed=1)
        assert simulator.semantic_version == SIMULATOR_SEMANTIC_VERSION
        assert simulator.num_generations == 7
        assert simulator.config.cna_event_probability == pytest.approx(0.001)

    balanced = json.loads(clean_paths[0].read_text())
    wgd = json.loads(clean_paths[-1].read_text())
    assert {
        key for key in balanced if balanced[key] != wgd[key]
    } == {"WGD_PROBABILITY"}
    assert balanced["WGD_PROBABILITY"] == 0
    assert wgd["WGD_PROBABILITY"] == 0.01

    crucial_paths = [
        paper_configs / "crucial_10pct_control.json",
        paper_configs / "crucial_10pct_survival.json",
    ]
    crucial_configs = [json.loads(path.read_text()) for path in crucial_paths]
    assert {
        key
        for key in crucial_configs[0]
        if crucial_configs[0][key] != crucial_configs[1][key]
    } == {"CRUCIAL_SURVIVAL_ENABLED"}
    for path in crucial_paths:
        simulator = CancerCellEvolutionSimulator(path, seed=1)
        assert simulator.semantic_version == SIMULATOR_SEMANTIC_VERSION
        assert simulator.num_generations == 7
        assert simulator.config.cna_event_probability == pytest.approx(0.001)


def test_bed_export_round_trips_resolved_parameters(tmp_path):
    simulator = CancerCellEvolutionSimulator(active_config(), seed=2)
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
