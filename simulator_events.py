"""Pure typed event proposal and application mechanics for CTBF simulator v5."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from simulator_config import GenomeBin


@dataclass(frozen=True)
class CNAEventProposal:
    proposal_id: int
    within_generation_time: float
    event_class: str
    chromosome: Optional[str]
    initiation_index: Optional[int]
    start_index: int
    end_index: int
    footprint_direction: str
    direction: str
    operator: str
    magnitude: int


@dataclass(frozen=True)
class CNAEventRecord:
    proposal_id: int
    within_generation_order: int
    within_generation_time: float
    event_class: str
    chromosome: Optional[str]
    initiation_index: Optional[int]
    start_index: int
    end_index: int
    initiation_coordinate: Optional[int]
    start_coordinate: Optional[int]
    end_coordinate: Optional[int]
    footprint_length: int
    multi_position: bool
    footprint_direction: str
    direction: str
    operator: str
    magnitude: int
    before: Tuple[int, ...]
    after: Tuple[int, ...]
    changed_positions: Tuple[int, ...]
    effective: bool

    def as_dict(self) -> dict[str, Any]:
        magnitude_name = (
            "multiplication_factor"
            if self.operator in {"multiplicative", "whole_genome_doubling"}
            else "copies_added"
            if self.direction == "gain"
            else "copies_removed"
        )
        return {
            "proposal_id": self.proposal_id,
            "within_generation_order": self.within_generation_order,
            "within_generation_time": self.within_generation_time,
            "event_class": self.event_class,
            "chromosome": self.chromosome,
            "initiation_index": self.initiation_index,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "initiation_coordinate": self.initiation_coordinate,
            "start_coordinate": self.start_coordinate,
            "end_coordinate": self.end_coordinate,
            "footprint_length": self.footprint_length,
            "multi_position": self.multi_position,
            "footprint_direction": self.footprint_direction,
            "direction": self.direction,
            "operator": self.operator,
            "magnitude": self.magnitude,
            magnitude_name: self.magnitude,
            "before": list(self.before),
            "after": list(self.after),
            "changed_positions": list(self.changed_positions),
            "effective": self.effective,
        }


@dataclass(frozen=True)
class EventApplicationResult:
    genome: Optional[np.ndarray]
    proposed_events: Tuple[CNAEventProposal, ...]
    attempted_records: Tuple[CNAEventRecord, ...]
    edge_records: Tuple[CNAEventRecord, ...]
    viability_rejection_reason: Optional[str]
    net_zero_sequence: bool


def _chromosome_boundary_indices(
    genome_bins: Sequence[GenomeBin],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    starts = [0] * len(genome_bins)
    ends = [0] * len(genome_bins)
    start = 0
    while start < len(genome_bins):
        chromosome = genome_bins[start].chromosome
        end = start
        while end + 1 < len(genome_bins) and genome_bins[end + 1].chromosome == chromosome:
            end += 1
        for index in range(start, end + 1):
            starts[index] = start
            ends[index] = end
        start = end + 1
    return tuple(starts), tuple(ends)


def segmental_initiation_probabilities(
    genome_bins: Sequence[GenomeBin],
    *,
    initiation_multiplier: float,
) -> np.ndarray:
    """Resolve scheduled base-plus-telomeric initiation probabilities."""
    if isinstance(initiation_multiplier, bool) or not isinstance(
        initiation_multiplier,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("initiation_multiplier must be finite and > 0.")
    multiplier = float(initiation_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("initiation_multiplier must be finite and > 0.")

    unscheduled_hazards = np.fromiter(
        (
            genome_bin.cna_event_probability + genome_bin.telomeric_instability
            for genome_bin in genome_bins
        ),
        dtype=float,
        count=len(genome_bins),
    )
    with np.errstate(over="ignore"):
        return np.minimum(1.0, multiplier * unscheduled_hazards)


def propose_event_sequence(
    genome_bins: Sequence[GenomeBin],
    *,
    wgd_probability: float,
    initiation_multiplier: float,
    rng_streams: Mapping[str, np.random.Generator],
) -> Tuple[CNAEventProposal, ...]:
    """Sample all proposals first, then order independently of genomic position."""
    number_of_positions = len(genome_bins)
    chromosome_starts, chromosome_ends = _chromosome_boundary_indices(genome_bins)
    initiation_probabilities = segmental_initiation_probabilities(
        genome_bins,
        initiation_multiplier=initiation_multiplier,
    )

    initiation_draws = rng_streams["event_initiation"].random(number_of_positions)
    interval_draws = rng_streams["event_class"].random(number_of_positions)
    footprint_direction_draws = rng_streams["footprint"].random(number_of_positions)
    footprint_endpoint_draws = rng_streams["footprint"].random(number_of_positions)
    direction_draws = rng_streams["event_type"].random(number_of_positions)
    operator_draws = rng_streams["gain_operator"].random(number_of_positions)
    factor_draws = rng_streams["magnitude_factor"].random(number_of_positions)
    order_draws = rng_streams["event_order"].random(number_of_positions)
    additive_lambdas = np.asarray(
        [genome_bin.additive_gain_lambda for genome_bin in genome_bins],
        dtype=float,
    )
    additive_draws = rng_streams["magnitude_additive"].poisson(additive_lambdas)

    proposals = []
    proposal_id = 0
    for index, genome_bin in enumerate(genome_bins):
        if initiation_draws[index] >= initiation_probabilities[index]:
            continue

        is_interval = interval_draws[index] < genome_bin.interval_cna_probability
        if is_interval:
            extends_left = footprint_direction_draws[index] < 0.5
            if extends_left:
                available = index - chromosome_starts[index] + 1
                sampled_endpoint = index - min(
                    available - 1,
                    int(footprint_endpoint_draws[index] * available),
                )
                start_index = sampled_endpoint
                end_index = index
                footprint_direction = "left"
            else:
                available = chromosome_ends[index] - index + 1
                sampled_endpoint = index + min(
                    available - 1,
                    int(footprint_endpoint_draws[index] * available),
                )
                start_index = index
                end_index = sampled_endpoint
                footprint_direction = "right"
            event_class = "interval_mode_cna"
        else:
            start_index = index
            end_index = index
            footprint_direction = "point"
            event_class = "point_unit_cna"

        is_gain = direction_draws[index] < genome_bin.gain_given_cna_probability
        direction = "gain" if is_gain else "loss"
        if not is_interval or not is_gain:
            operator = "unit"
            magnitude = 1
        else:
            operator = str(genome_bin.interval_gain_operators.select(operator_draws[index]))
            if operator == "unit":
                magnitude = 1
            elif operator == "additive":
                magnitude = 2 + int(additive_draws[index])
            elif operator == "multiplicative":
                magnitude = int(genome_bin.multiplicative_factors.select(factor_draws[index]))
            else:  # pragma: no cover - strict configuration prevents this
                raise ValueError(f"Unsupported gain operator {operator!r}.")

        proposals.append(
            CNAEventProposal(
                proposal_id=proposal_id,
                within_generation_time=float(order_draws[index]),
                event_class=event_class,
                chromosome=genome_bin.chromosome,
                initiation_index=index,
                start_index=start_index,
                end_index=end_index,
                footprint_direction=footprint_direction,
                direction=direction,
                operator=operator,
                magnitude=magnitude,
            )
        )
        proposal_id += 1

    if rng_streams["wgd"].random() < wgd_probability:
        proposals.append(
            CNAEventProposal(
                proposal_id=proposal_id,
                within_generation_time=float(rng_streams["event_order"].random()),
                event_class="whole_genome_doubling",
                chromosome=None,
                initiation_index=None,
                start_index=0,
                end_index=number_of_positions - 1,
                footprint_direction="whole_genome",
                direction="gain",
                operator="whole_genome_doubling",
                magnitude=2,
            )
        )

    return tuple(
        sorted(
            proposals,
            key=lambda proposal: (
                proposal.within_generation_time,
                proposal.proposal_id,
            ),
        )
    )


def _apply_proposal(
    genome: np.ndarray,
    proposal: CNAEventProposal,
    genome_bins: Sequence[GenomeBin],
    order: int,
) -> CNAEventRecord:
    positions = tuple(range(proposal.start_index, proposal.end_index + 1))
    before = tuple(int(genome[position]) for position in positions)

    for position in positions:
        current = int(genome[position])
        if current == 0:
            continue
        if proposal.operator == "unit":
            if proposal.direction == "gain":
                genome[position] = current + 1
            else:
                genome[position] = max(0, current - 1)
        elif proposal.operator == "additive":
            genome[position] = current + proposal.magnitude
        elif proposal.operator in {"multiplicative", "whole_genome_doubling"}:
            genome[position] = current * proposal.magnitude
        else:  # pragma: no cover - proposals are internally typed
            raise ValueError(f"Unsupported event operator {proposal.operator!r}.")

    after = tuple(int(genome[position]) for position in positions)
    changed_positions = tuple(
        position
        for position, before_value, after_value in zip(positions, before, after)
        if before_value != after_value
    )
    return CNAEventRecord(
        proposal_id=proposal.proposal_id,
        within_generation_order=order,
        within_generation_time=proposal.within_generation_time,
        event_class=proposal.event_class,
        chromosome=proposal.chromosome,
        initiation_index=proposal.initiation_index,
        start_index=proposal.start_index,
        end_index=proposal.end_index,
        initiation_coordinate=(
            None
            if proposal.initiation_index is None
            else genome_bins[proposal.initiation_index].start
        ),
        start_coordinate=(
            None
            if proposal.event_class == "whole_genome_doubling"
            else genome_bins[proposal.start_index].start
        ),
        end_coordinate=(
            None
            if proposal.event_class == "whole_genome_doubling"
            else genome_bins[proposal.end_index].end
        ),
        footprint_length=len(positions),
        multi_position=len(positions) > 1,
        footprint_direction=proposal.footprint_direction,
        direction=proposal.direction,
        operator=proposal.operator,
        magnitude=proposal.magnitude,
        before=before,
        after=after,
        changed_positions=changed_positions,
        effective=bool(changed_positions),
    )


def apply_event_sequence(
    parent_genome: np.ndarray,
    proposals: Sequence[CNAEventProposal],
    genome_bins: Sequence[GenomeBin],
    *,
    crucial_survival_enabled: bool,
) -> EventApplicationResult:
    genome = np.asarray(parent_genome, dtype=int).copy()
    parent = genome.copy()
    records = []
    crucial_positions = tuple(
        index for index, genome_bin in enumerate(genome_bins) if genome_bin.crucial
    )

    ordered_proposals = tuple(
        sorted(
            proposals,
            key=lambda proposal: (
                proposal.within_generation_time,
                proposal.proposal_id,
            ),
        )
    )

    if not np.any(genome):
        return EventApplicationResult(
            genome=None,
            proposed_events=ordered_proposals,
            attempted_records=(),
            edge_records=(),
            viability_rejection_reason="all_zero_genome",
            net_zero_sequence=False,
        )
    if crucial_survival_enabled and any(
        genome[index] == 0 for index in crucial_positions
    ):
        return EventApplicationResult(
            genome=None,
            proposed_events=ordered_proposals,
            attempted_records=(),
            edge_records=(),
            viability_rejection_reason="crucial_bin_zero",
            net_zero_sequence=False,
        )

    for order, proposal in enumerate(ordered_proposals):
        record = _apply_proposal(genome, proposal, genome_bins, order)
        records.append(record)
        if not np.any(genome):
            return EventApplicationResult(
                genome=None,
                proposed_events=ordered_proposals,
                attempted_records=tuple(records),
                edge_records=(),
                viability_rejection_reason="all_zero_genome",
                net_zero_sequence=False,
            )
        if crucial_survival_enabled and any(genome[index] == 0 for index in crucial_positions):
            return EventApplicationResult(
                genome=None,
                proposed_events=ordered_proposals,
                attempted_records=tuple(records),
                edge_records=(),
                viability_rejection_reason="crucial_bin_zero",
                net_zero_sequence=False,
            )

    final_equals_parent = bool(np.array_equal(genome, parent))
    net_zero_sequence = bool(records) and final_equals_parent
    edge_records = (
        ()
        if final_equals_parent
        else tuple(record for record in records if record.effective)
    )
    return EventApplicationResult(
        genome=genome,
        proposed_events=ordered_proposals,
        attempted_records=tuple(records),
        edge_records=edge_records,
        viability_rejection_reason=None,
        net_zero_sequence=net_zero_sequence,
    )


def event_records_to_dicts(records: Sequence[CNAEventRecord]) -> list[dict[str, Any]]:
    return [record.as_dict() for record in records]


def event_records_to_text(records: Sequence[CNAEventRecord]) -> str:
    parts = []
    for record in records:
        if record.event_class == "whole_genome_doubling":
            parts.append("whole_genome_doubling(factor=2)")
            continue
        span = (
            str(record.start_index)
            if record.start_index == record.end_index
            else f"{record.start_index}-{record.end_index}"
        )
        if record.operator == "unit":
            signed_magnitude = record.magnitude if record.direction == "gain" else -record.magnitude
            parts.append(f"{record.direction}(pos={span}, copies={signed_magnitude})")
        elif record.operator == "additive":
            parts.append(f"additive_gain(pos={span}, copies={record.magnitude})")
        else:
            parts.append(f"multiplicative_gain(pos={span}, factor={record.magnitude})")
    return ";".join(parts)


def count_edge_events(events: Any) -> int:
    if events is None:
        return 0
    if isinstance(events, (list, tuple)):
        return len(events)
    if isinstance(events, str):
        return sum(1 for token in events.split(";") if token.strip())
    raise ValueError(f"Unsupported edge event representation {type(events).__name__}.")
