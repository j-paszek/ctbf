"""Occurrence-level temporal CNP arborescence reconstruction.

This module is deliberately separate from the NJ-like reconstruction engine.
Its input is the complete ordered biopsy record, not one pooled state list.
"""

from dataclasses import dataclass
import json
import random

import networkx as nx
import numpy as np

from reconstructor_plausibility import is_biologically_plausible_ancestor
from distance_semantics import validate_distance_matrix


ORDERED_OCCURRENCE_INPUT_MODE = "ordered_occurrences"


@dataclass(frozen=True)
class _TemporalOccurrence:
    node_id: int
    biopsy_level: int
    cell_id: object
    genome: np.ndarray


def _stable_label_key(value):
    """Return a deterministic ordering key for supported state labels."""
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        payload = repr(value)
    return type_name, payload


def _normalize_occurrences(cell_lists):
    if cell_lists is None:
        raise ValueError("Ordered biopsy levels are required.")

    records = []
    genome_by_state = {}
    for level, cells in enumerate(cell_lists):
        if cells is None:
            raise ValueError(f"Biopsy level {level} is None; expected an iterable of cells.")

        seen_at_level = set()
        for cell in cells:
            cell_id = getattr(cell, "cell_id", None)
            if cell_id is None:
                raise ValueError("Every temporal occurrence requires a non-null cell_id.")
            try:
                already_seen = cell_id in seen_at_level
            except TypeError as exc:
                raise ValueError("Temporal occurrence cell_id values must be hashable.") from exc

            genome = np.asarray(getattr(cell, "genome", None))
            if genome.ndim != 1 or genome.size == 0:
                raise ValueError(
                    f"State {cell_id!r} requires a nonempty one-dimensional genome."
                )

            previous_genome = genome_by_state.get(cell_id)
            if previous_genome is not None and not np.array_equal(previous_genome, genome):
                raise ValueError(
                    f"Repeated cell_id {cell_id!r} is associated with inconsistent genomes."
                )
            if previous_genome is None:
                genome_by_state[cell_id] = np.array(genome, copy=True)

            if already_seen:
                continue
            seen_at_level.add(cell_id)
            records.append((level, cell_id, np.array(genome, copy=True)))

    if not records:
        raise ValueError("Temporal reconstruction requires at least one observation occurrence.")

    genome_shapes = {record[2].shape for record in records}
    if len(genome_shapes) != 1:
        raise ValueError("All temporal occurrence genomes must have the same shape.")

    # State-first ordering keeps the no-time ablation independent of biopsy
    # order for distinct states. Repeated equal-state records remain distinct.
    records.sort(key=lambda record: (_stable_label_key(record[1]), record[0]))
    return [
        _TemporalOccurrence(
            node_id=node_id,
            biopsy_level=level,
            cell_id=cell_id,
            genome=genome,
        )
        for node_id, (level, cell_id, genome) in enumerate(records)
    ]


def _validate_distance_input(dist_matrix, ids, occurrences):
    if ids is None:
        raise ValueError("Distance-matrix ids are required for temporal reconstruction.")

    ids = list(ids)
    id_to_index = {}
    for index, cell_id in enumerate(ids):
        try:
            if cell_id in id_to_index:
                raise ValueError(f"Duplicate distance-matrix id {cell_id!r}.")
            id_to_index[cell_id] = index
        except TypeError as exc:
            raise ValueError("Distance-matrix ids must be hashable.") from exc

    ids, D = validate_distance_matrix(ids, dist_matrix)

    observed_states = {occurrence.cell_id for occurrence in occurrences}
    matrix_states = set(ids)
    if observed_states != matrix_states:
        missing = sorted(observed_states - matrix_states, key=_stable_label_key)
        extra = sorted(matrix_states - observed_states, key=_stable_label_key)
        raise ValueError(
            "Distance ids must match the observed CNP states exactly "
            f"(missing={missing!r}, extra={extra!r})."
        )

    return D, id_to_index


def _exact_distance_units(D):
    """Represent every binary float exactly under one integer denominator."""
    ratios = [float(value).as_integer_ratio() for value in D.flat]
    common_denominator = max(denominator for _, denominator in ratios)
    units = [
        numerator * (common_denominator // denominator)
        for numerator, denominator in ratios
    ]
    return np.asarray(units, dtype=object).reshape(D.shape)


def _candidate_biological_edges(occurrences, use_time):
    edges = []
    for parent in occurrences:
        for child in occurrences:
            if parent.node_id == child.node_id:
                continue
            if use_time and parent.biopsy_level > child.biopsy_level:
                continue
            edges.append((parent.node_id, child.node_id))
    return edges


def _seeded_tie_ranks(biological_edges, root_candidates, seed):
    keys = [
        ("edge", parent, child)
        for parent, child in biological_edges
    ]
    keys.extend(("root", root) for root in root_candidates)
    random.Random(seed).shuffle(keys)
    return {key: rank for rank, key in enumerate(keys)}


def _mixed_radix_coefficients(
    occurrence_count,
    max_distance_unit,
    max_tie_rank,
):
    biological_edge_count = max(occurrence_count - 1, 0)
    max_violation_total = biological_edge_count
    max_distance_total = biological_edge_count * max_distance_unit
    max_root_score_total = biological_edge_count * max_distance_unit
    # A spanning arborescence on n occurrences plus the virtual root selects
    # exactly n edges. This bound also covers invalid multi-root candidates.
    max_tie_total = occurrence_count * max_tie_rank

    root_score_coefficient = max_tie_total + 1
    distance_coefficient = (
        max_root_score_total * root_score_coefficient
        + max_tie_total
        + 1
    )
    violation_coefficient = (
        max_distance_total * distance_coefficient
        + max_root_score_total * root_score_coefficient
        + max_tie_total
        + 1
    )
    virtual_root_coefficient = (
        max_violation_total * violation_coefficient
        + max_distance_total * distance_coefficient
        + max_root_score_total * root_score_coefficient
        + max_tie_total
        + 1
    )
    return {
        "root_score": root_score_coefficient,
        "distance": distance_coefficient,
        "violation": violation_coefficient,
        "virtual_root": virtual_root_coefficient,
    }


def _solve_temporal_arborescence(D, id_to_index, occurrences, seed, use_time):
    occurrence_by_id = {occurrence.node_id: occurrence for occurrence in occurrences}
    biological_edges = _candidate_biological_edges(occurrences, use_time)
    if use_time:
        earliest_level = min(occurrence.biopsy_level for occurrence in occurrences)
        root_candidates = [
            occurrence.node_id
            for occurrence in occurrences
            if occurrence.biopsy_level == earliest_level
        ]
    else:
        root_candidates = [occurrence.node_id for occurrence in occurrences]

    # Rank the complete no-time edge/root universe in both modes, then filter
    # candidates. Shared choices therefore retain exactly the same seeded
    # final-tier priority in the ordered method and its information ablation.
    all_biological_edges = _candidate_biological_edges(occurrences, use_time=False)
    all_root_candidates = [occurrence.node_id for occurrence in occurrences]
    tie_ranks = _seeded_tie_ranks(
        all_biological_edges,
        all_root_candidates,
        seed,
    )
    distance_units = _exact_distance_units(D)
    max_distance_unit = max(int(value) for value in distance_units.flat)
    max_tie_rank = max(tie_ranks.values(), default=0)
    coefficients = _mixed_radix_coefficients(
        len(occurrences),
        max_distance_unit,
        max_tie_rank,
    )

    candidate_graph = nx.DiGraph()
    candidate_graph.add_nodes_from(occurrence_by_id)
    virtual_root = object()
    candidate_graph.add_node(virtual_root)

    for parent_id, child_id in biological_edges:
        parent = occurrence_by_id[parent_id]
        child = occurrence_by_id[child_id]
        distance_unit = int(
            distance_units[
                id_to_index[parent.cell_id],
                id_to_index[child.cell_id],
            ]
        )
        violation = int(not is_biologically_plausible_ancestor(parent, child))
        cost = (
            violation * coefficients["violation"]
            + distance_unit * coefficients["distance"]
            + tie_ranks[("edge", parent_id, child_id)]
        )
        candidate_graph.add_edge(parent_id, child_id, objective_cost=cost)

    for root_id in root_candidates:
        root = occurrence_by_id[root_id]
        root_state_index = id_to_index[root.cell_id]
        root_score_unit = sum(
            int(distance_units[root_state_index, id_to_index[other.cell_id]])
            for other in occurrences
            if other.node_id != root_id
        )
        cost = (
            coefficients["virtual_root"]
            + root_score_unit * coefficients["root_score"]
            + tie_ranks[("root", root_id)]
        )
        candidate_graph.add_edge(virtual_root, root_id, objective_cost=cost)

    solved = nx.minimum_spanning_arborescence(
        candidate_graph,
        attr="objective_cost",
        preserve_attrs=True,
    )
    selected_root_edges = [
        (parent, child)
        for parent, child in solved.edges()
        if parent is virtual_root
    ]
    if len(selected_root_edges) != 1:
        raise RuntimeError(
            "Temporal arborescence scalarization did not select exactly one root."
        )
    root_id = selected_root_edges[0][1]

    tree = nx.DiGraph()
    for occurrence in occurrences:
        tree.add_node(
            occurrence.node_id,
            genome=np.array(occurrence.genome, copy=True),
            cell_id=occurrence.cell_id,
            biopsy_level=occurrence.biopsy_level,
        )

    selected_biological_edges = sorted(
        (parent, child)
        for parent, child in solved.edges()
        if parent is not virtual_root and child is not virtual_root
    )
    for parent_id, child_id in selected_biological_edges:
        parent = occurrence_by_id[parent_id]
        child = occurrence_by_id[child_id]
        tree.add_edge(
            parent_id,
            child_id,
            weight=float(
                D[id_to_index[parent.cell_id], id_to_index[child.cell_id]]
            ),
        )

    if not nx.is_arborescence(tree):
        raise RuntimeError("Temporal reconstruction did not produce an arborescence.")
    if next(iter(nx.topological_sort(tree))) != root_id:
        raise RuntimeError("Temporal reconstruction root does not match the selected root.")
    return tree, {}, root_id


def temporal_cnp_arborescence(
    dist_matrix,
    cell_lists,
    ids,
    seed=7,
    *,
    use_time=True,
):
    """Build the global fully labeled occurrence arborescence.

    Parameters
    ----------
    dist_matrix
        Symmetric state-level CNP dissimilarity matrix.
    cell_lists
        Ordered biopsy levels, earliest first.
    ids
        State labels aligned to ``dist_matrix`` rows and columns.
    seed
        Seed used only for the last lexicographic tie tier.
    use_time
        If false, remove temporal edge and root restrictions while retaining
        exactly the same normalized occurrence vertices and all other costs.
    """
    occurrences = _normalize_occurrences(cell_lists)
    D, id_to_index = _validate_distance_input(dist_matrix, ids, occurrences)
    return _solve_temporal_arborescence(D, id_to_index, occurrences, seed, use_time)


def temporal_cnp_arborescence_no_time(dist_matrix, cell_lists, ids, seed=7):
    """Parameter-fixed exact information ablation for publication workflows."""
    return temporal_cnp_arborescence(
        dist_matrix,
        cell_lists,
        ids,
        seed=seed,
        use_time=False,
    )


def uses_ordered_occurrence_input(algorithm):
    return (
        getattr(algorithm, "ctbf_input_mode", None)
        == ORDERED_OCCURRENCE_INPUT_MODE
    )


temporal_cnp_arborescence.ctbf_input_mode = ORDERED_OCCURRENCE_INPUT_MODE
temporal_cnp_arborescence.ctbf_use_time = True
temporal_cnp_arborescence.ctbf_order_ablation = temporal_cnp_arborescence_no_time
temporal_cnp_arborescence_no_time.ctbf_input_mode = ORDERED_OCCURRENCE_INPUT_MODE
temporal_cnp_arborescence_no_time.ctbf_use_time = False
temporal_cnp_arborescence_no_time.ctbf_order_ablation = None


__all__ = [
    "ORDERED_OCCURRENCE_INPUT_MODE",
    "temporal_cnp_arborescence",
    "temporal_cnp_arborescence_no_time",
    "uses_ordered_occurrence_input",
]
