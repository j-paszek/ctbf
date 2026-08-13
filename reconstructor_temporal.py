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
from distance_semantics import DirectedDistanceBundle, validate_distance_matrix


ORDERED_OCCURRENCE_INPUT_MODE = "ordered_occurrences"
TEMPORAL_ARBORESCENCE_SOLVER_VERSION = "ctbf-compact-chu-liu-edmonds-v1"


@dataclass(frozen=True)
class _TemporalOccurrence:
    node_id: int
    biopsy_level: int
    cell_id: object
    genome: np.ndarray


@dataclass(slots=True)
class _WorkingEdge:
    """One compact Edmonds edge, retaining its original occurrence endpoints."""

    edge_id: int
    original_source: int
    original_target: int
    source: int
    target: int
    weight: int


@dataclass(frozen=True, slots=True)
class _Contraction:
    """Information sufficient to expand one contracted directed cycle."""

    component: int
    members: tuple[int, ...]
    selected_edges: tuple[tuple[int, int], ...]


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

    # Exact CNP state, not its opaque alignment label, owns the normalized
    # occurrence order and therefore the seeded tie universe.  The label is
    # only a final fallback for malformed/noninjective generic inputs with two
    # labels on one exact profile; paper inputs reject that situation before
    # reconstruction.  Repeated occurrences of one state remain level-ordered
    # and distinct.
    records.sort(
        key=lambda record: (
            tuple(record[2].tolist()),
            record[0],
            _stable_label_key(record[1]),
        )
    )
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


def _compact_seeded_tie_ranks(occurrence_count, seed):
    """Return the legacy complete-universe ranks without tuple/dict storage."""
    universe_size = occurrence_count * occurrence_count
    dtype = np.int32 if universe_size <= np.iinfo(np.int32).max else np.int64
    shuffled_keys = np.arange(universe_size, dtype=dtype)
    random.Random(seed).shuffle(shuffled_keys)
    ranks = np.empty(universe_size, dtype=dtype)
    ranks[shuffled_keys] = np.arange(universe_size, dtype=dtype)
    return ranks


def _edge_tie_key_index(parent_id, child_id, occurrence_count):
    """Index one edge in the legacy parent-major no-time tie universe."""
    return (
        parent_id * (occurrence_count - 1)
        + child_id
        - int(child_id > parent_id)
    )


def _mixed_radix_coefficients(
    occurrence_count,
    max_edge_distance_unit,
    max_root_distance_unit,
    max_tie_rank,
):
    biological_edge_count = max(occurrence_count - 1, 0)
    max_violation_total = biological_edge_count
    max_distance_total = biological_edge_count * max_edge_distance_unit
    max_root_score_total = biological_edge_count * max_root_distance_unit
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


def _validate_directed_input(directed_distance_bundle, ids, D):
    if not isinstance(directed_distance_bundle, DirectedDistanceBundle):
        raise ValueError(
            "Directed temporal reconstruction requires a DirectedDistanceBundle."
        )
    bundle_ids = list(directed_distance_bundle.ids)
    if set(bundle_ids) != set(ids):
        raise ValueError(
            "Directed-distance ids must match the symmetric distance ids exactly."
        )
    bundle_index = {cell_id: index for index, cell_id in enumerate(bundle_ids)}
    alignment = [bundle_index[cell_id] for cell_id in ids]
    directed = directed_distance_bundle.directed_matrix[np.ix_(alignment, alignment)]
    minimum = directed_distance_bundle.minimum_matrix[np.ix_(alignment, alignment)]
    if not np.array_equal(minimum, D):
        raise ValueError(
            "Directed-distance bundle minimum must equal the reconstruction matrix."
        )
    return np.array(directed, copy=True)


def _incoming_edge_buckets(nodes, active_edges):
    incoming = {node: [] for node in nodes}
    for edge in active_edges:
        incoming[edge.target].append(edge)
    return incoming


def _find_desired_edge(incoming_edges):
    """Find the first minimum incoming edge in current graph iteration order."""
    desired = None
    for edge in incoming_edges:
        if desired is None or edge.weight < desired.weight:
            desired = edge
    return desired


class _DisjointSet:
    """Small internal union-find used only for cycle detection."""

    def __init__(self, nodes):
        self._parent = {node: node for node in nodes}
        self._rank = {node: 0 for node in nodes}

    def find(self, node):
        parent = self._parent[node]
        while parent != self._parent[parent]:
            parent = self._parent[parent]
        while node != parent:
            next_node = self._parent[node]
            self._parent[node] = parent
            node = next_node
        return parent

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        first_rank = self._rank[first_root]
        second_rank = self._rank[second_root]
        if first_rank < second_rank:
            first_root, second_root = second_root, first_root
        self._parent[second_root] = first_root
        if first_rank == second_rank:
            self._rank[first_root] += 1


def _rebuild_disjoint_set(nodes, selected_edges, edge_by_id):
    components = _DisjointSet(nodes)
    for edge_id in selected_edges.values():
        edge = edge_by_id[edge_id]
        components.union(edge.source, edge.target)
    return components


def _cycle_members(target, source, selected_edges, edge_by_id):
    """Return the directed cycle closed by ``source -> target``."""
    members = [target]
    current = source
    while current != target:
        if current in members or current not in selected_edges:
            raise RuntimeError("Compact Edmonds cycle bookkeeping is inconsistent.")
        members.append(current)
        current = edge_by_id[selected_edges[current]].source
    return tuple(members)


def _contract_cycle_edges(
    active_edges,
    nodes,
    cycle,
    component,
    selected_edges,
    edge_by_id,
):
    """Contract one cycle while matching NetworkX's stable edge iteration."""
    cycle_set = set(cycle)
    incoming_weight = {
        member: edge_by_id[selected_edges[member]].weight
        for member in cycle
    }
    maximum_cycle_weight = max(incoming_weight.values())

    existing_by_source = {node: [] for node in nodes if node not in cycle_set}
    incoming_by_source = {node: [] for node in nodes if node not in cycle_set}
    outgoing_by_target = {}
    outgoing_target_order = []

    for edge in active_edges:
        source_in_cycle = edge.source in cycle_set
        target_in_cycle = edge.target in cycle_set
        if source_in_cycle and target_in_cycle:
            continue
        if not source_in_cycle and target_in_cycle:
            old_target = edge.target
            edge.target = component
            edge.weight += maximum_cycle_weight - incoming_weight[old_target]
            incoming_by_source[edge.source].append(edge)
            continue
        if source_in_cycle:
            edge.source = component
            if edge.target not in outgoing_by_target:
                outgoing_by_target[edge.target] = []
                outgoing_target_order.append(edge.target)
            outgoing_by_target[edge.target].append(edge)
            continue
        existing_by_source[edge.source].append(edge)

    remaining_nodes = [node for node in nodes if node not in cycle_set]
    reordered_edges = []
    for source in remaining_nodes:
        reordered_edges.extend(existing_by_source[source])
        reordered_edges.extend(incoming_by_source[source])
    for target in outgoing_target_order:
        reordered_edges.extend(outgoing_by_target[target])
    return reordered_edges, remaining_nodes + [component]


def _expand_contractions(
    selected_edge_ids,
    contractions,
    component_masks,
    edge_by_id,
):
    """Expand compact cycle history into original candidate-edge ids."""
    selected = set(selected_edge_ids)
    for contraction in reversed(contractions):
        mask = component_masks[contraction.component]
        entering = []
        for edge_id in selected:
            edge = edge_by_id[edge_id]
            source_inside = bool(mask & (1 << edge.original_source))
            target_inside = bool(mask & (1 << edge.original_target))
            if target_inside and not source_inside:
                entering.append(edge_id)
        if len(entering) != 1:
            raise RuntimeError(
                "Compact Edmonds expansion did not find exactly one cycle entry."
            )

        entering_target = edge_by_id[entering[0]].original_target
        entry_member = next(
            (
                member
                for member in contraction.members
                if component_masks[member] & (1 << entering_target)
            ),
            None,
        )
        if entry_member is None:
            raise RuntimeError("Compact Edmonds expansion lost the cycle entry target.")
        selected.update(
            edge_id
            for member, edge_id in contraction.selected_edges
            if member != entry_member
        )
    return selected


def _compact_minimum_spanning_arborescence(node_count, root, edges):
    """Return original edge ids for an exact minimum rooted arborescence.

    This is a compact Chu--Liu/Edmonds implementation specialized to CTBF's
    simple candidate graph. It follows the former NetworkX solver's node and
    edge iteration order, but retains only one mutable edge universe and
    O(V)-sized cycle-expansion records instead of full graph copies.
    """
    if node_count <= 0 or root < 0 or root >= node_count:
        raise ValueError("Compact Edmonds requires a valid nonempty rooted graph.")
    nodes = list(range(node_count))
    edge_by_id = list(edges)
    active_edges = list(edge_by_id)
    if any(edge.edge_id != edge_id for edge_id, edge in enumerate(edge_by_id)):
        raise ValueError("Compact Edmonds edge ids must be contiguous and ordered.")
    if any(
        edge.source == edge.target
        or edge.source not in range(node_count)
        or edge.target not in range(node_count)
        for edge in active_edges
    ):
        raise ValueError("Compact Edmonds edges require distinct valid endpoints.")

    selected_nodes = set()
    selected_edges = {}
    components = _DisjointSet(nodes)
    component_masks = {node: 1 << node for node in nodes}
    contractions = []
    next_component = node_count
    node_index = 0
    incoming_edges = _incoming_edge_buckets(nodes, active_edges)

    while node_index < len(nodes):
        target = nodes[node_index]
        node_index += 1
        if target in selected_nodes:
            continue
        selected_nodes.add(target)
        desired = _find_desired_edge(incoming_edges[target])
        if desired is None:
            if target != root:
                raise RuntimeError(
                    "Candidate graph has no spanning arborescence from its root."
                )
            continue

        closes_cycle = components.find(desired.source) == components.find(target)
        selected_edges[target] = desired.edge_id
        components.union(desired.source, target)
        if not closes_cycle:
            continue

        cycle = _cycle_members(
            target,
            desired.source,
            selected_edges,
            edge_by_id,
        )
        cycle_selected_edges = tuple(
            (member, selected_edges[member])
            for member in cycle
        )
        component = next_component
        next_component += 1
        component_masks[component] = 0
        for member in cycle:
            component_masks[component] |= component_masks[member]
        contractions.append(
            _Contraction(
                component=component,
                members=cycle,
                selected_edges=cycle_selected_edges,
            )
        )

        active_edges, nodes = _contract_cycle_edges(
            active_edges,
            nodes,
            cycle,
            component,
            selected_edges,
            edge_by_id,
        )
        for member in cycle:
            selected_nodes.discard(member)
            selected_edges.pop(member, None)
        selected_nodes.discard(component)
        components = _rebuild_disjoint_set(nodes, selected_edges, edge_by_id)
        incoming_edges = _incoming_edge_buckets(nodes, active_edges)
        node_index = 0

    if set(nodes) != selected_nodes or len(selected_edges) != len(nodes) - 1:
        raise RuntimeError("Compact Edmonds did not construct one spanning branching.")
    selected = _expand_contractions(
        selected_edges.values(),
        contractions,
        component_masks,
        edge_by_id,
    )
    if len(selected) != node_count - 1:
        raise RuntimeError("Compact Edmonds expansion returned the wrong edge count.")
    return selected


def _solve_temporal_arborescence(
    D,
    id_to_index,
    occurrences,
    seed,
    use_time,
    directed_distances=None,
):
    occurrence_by_id = {occurrence.node_id: occurrence for occurrence in occurrences}
    if use_time:
        earliest_level = min(occurrence.biopsy_level for occurrence in occurrences)
        root_candidates = [
            occurrence.node_id
            for occurrence in occurrences
            if occurrence.biopsy_level == earliest_level
        ]
    else:
        root_candidates = [occurrence.node_id for occurrence in occurrences]

    # Rank the complete no-time edge/root universe in both modes. Integer key
    # indexes reproduce the former tuple-list shuffle exactly while avoiding a
    # second O(n^2) edge list and a Python tuple-keyed rank dictionary.
    occurrence_count = len(occurrences)
    tie_ranks = _compact_seeded_tie_ranks(occurrence_count, seed)
    edge_distances = D if directed_distances is None else directed_distances
    edge_distance_units = _exact_distance_units(edge_distances)
    root_distance_units = _exact_distance_units(D)
    max_edge_distance_unit = max(
        int(value)
        for value in edge_distance_units.flat
    )
    max_root_distance_unit = max(
        int(value)
        for value in root_distance_units.flat
    )
    max_tie_rank = occurrence_count * occurrence_count - 1
    coefficients = _mixed_radix_coefficients(
        occurrence_count,
        max_edge_distance_unit,
        max_root_distance_unit,
        max_tie_rank,
    )

    edges = []
    for parent in occurrences:
        for child in occurrences:
            parent_id = parent.node_id
            child_id = child.node_id
            if parent_id == child_id:
                continue
            if use_time and parent.biopsy_level > child.biopsy_level:
                continue
            distance_unit = int(
                edge_distance_units[
                    id_to_index[parent.cell_id],
                    id_to_index[child.cell_id],
                ]
            )
            violation = int(not is_biologically_plausible_ancestor(parent, child))
            tie_rank = int(
                tie_ranks[
                    _edge_tie_key_index(
                        parent_id,
                        child_id,
                        occurrence_count,
                    )
                ]
            )
            cost = (
                violation * coefficients["violation"]
                + distance_unit * coefficients["distance"]
                + tie_rank
            )
            edge_id = len(edges)
            edges.append(
                _WorkingEdge(
                    edge_id=edge_id,
                    original_source=parent_id,
                    original_target=child_id,
                    source=parent_id,
                    target=child_id,
                    weight=cost,
                )
            )

    virtual_root = occurrence_count
    root_tie_offset = occurrence_count * (occurrence_count - 1)
    for root_id in root_candidates:
        root = occurrence_by_id[root_id]
        root_state_index = id_to_index[root.cell_id]
        root_score_unit = sum(
            int(root_distance_units[root_state_index, id_to_index[other.cell_id]])
            for other in occurrences
            if other.node_id != root_id
        )
        cost = (
            coefficients["virtual_root"]
            + root_score_unit * coefficients["root_score"]
            + int(tie_ranks[root_tie_offset + root_id])
        )
        edge_id = len(edges)
        edges.append(
            _WorkingEdge(
                edge_id=edge_id,
                original_source=virtual_root,
                original_target=root_id,
                source=virtual_root,
                target=root_id,
                weight=cost,
            )
        )

    selected_edge_ids = _compact_minimum_spanning_arborescence(
        occurrence_count + 1,
        virtual_root,
        edges,
    )
    selected_edges = [edges[edge_id] for edge_id in selected_edge_ids]
    selected_root_edges = [
        edge
        for edge in selected_edges
        if edge.original_source == virtual_root
    ]
    if len(selected_root_edges) != 1:
        raise RuntimeError(
            "Temporal arborescence scalarization did not select exactly one root."
        )
    root_id = selected_root_edges[0].original_target

    tree = nx.DiGraph()
    for occurrence in occurrences:
        tree.add_node(
            occurrence.node_id,
            genome=np.array(occurrence.genome, copy=True),
            cell_id=occurrence.cell_id,
            biopsy_level=occurrence.biopsy_level,
        )

    selected_biological_edges = sorted(
        (edge.original_source, edge.original_target)
        for edge in selected_edges
        if edge.original_source != virtual_root
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


def _temporal_cnp_arborescence_directed(
    dist_matrix,
    cell_lists,
    ids,
    seed,
    *,
    use_time,
    directed_distance_bundle,
):
    occurrences = _normalize_occurrences(cell_lists)
    D, id_to_index = _validate_distance_input(dist_matrix, ids, occurrences)
    directed = _validate_directed_input(directed_distance_bundle, ids, D)
    return _solve_temporal_arborescence(
        D,
        id_to_index,
        occurrences,
        seed,
        use_time,
        directed_distances=directed,
    )


def temporal_cnp_arborescence_directed(
    dist_matrix,
    cell_lists,
    ids,
    seed=7,
    *,
    directed_distance_bundle,
):
    """G0-03-A variant using C[parent,child] after time/plausibility tiers."""
    return _temporal_cnp_arborescence_directed(
        dist_matrix,
        cell_lists,
        ids,
        seed,
        use_time=True,
        directed_distance_bundle=directed_distance_bundle,
    )


def temporal_cnp_arborescence_directed_no_time(
    dist_matrix,
    cell_lists,
    ids,
    seed=7,
    *,
    directed_distance_bundle,
):
    """Exact no-time ablation of the G0-03-A directed edge-cost variant."""
    return _temporal_cnp_arborescence_directed(
        dist_matrix,
        cell_lists,
        ids,
        seed,
        use_time=False,
        directed_distance_bundle=directed_distance_bundle,
    )


def uses_ordered_occurrence_input(algorithm):
    return (
        getattr(algorithm, "ctbf_input_mode", None)
        == ORDERED_OCCURRENCE_INPUT_MODE
    )


def uses_directed_distance_input(algorithm):
    return bool(getattr(algorithm, "ctbf_requires_directed_distances", False))


temporal_cnp_arborescence.ctbf_input_mode = ORDERED_OCCURRENCE_INPUT_MODE
temporal_cnp_arborescence.ctbf_use_time = True
temporal_cnp_arborescence.ctbf_solver_version = (
    TEMPORAL_ARBORESCENCE_SOLVER_VERSION
)
temporal_cnp_arborescence.ctbf_order_ablation = temporal_cnp_arborescence_no_time
temporal_cnp_arborescence_no_time.ctbf_input_mode = ORDERED_OCCURRENCE_INPUT_MODE
temporal_cnp_arborescence_no_time.ctbf_use_time = False
temporal_cnp_arborescence_no_time.ctbf_solver_version = (
    TEMPORAL_ARBORESCENCE_SOLVER_VERSION
)
temporal_cnp_arborescence_no_time.ctbf_order_ablation = None
temporal_cnp_arborescence_directed.ctbf_input_mode = ORDERED_OCCURRENCE_INPUT_MODE
temporal_cnp_arborescence_directed.ctbf_use_time = True
temporal_cnp_arborescence_directed.ctbf_requires_directed_distances = True
temporal_cnp_arborescence_directed.ctbf_solver_version = (
    TEMPORAL_ARBORESCENCE_SOLVER_VERSION
)
temporal_cnp_arborescence_directed.ctbf_order_ablation = (
    temporal_cnp_arborescence_directed_no_time
)
temporal_cnp_arborescence_directed_no_time.ctbf_input_mode = (
    ORDERED_OCCURRENCE_INPUT_MODE
)
temporal_cnp_arborescence_directed_no_time.ctbf_use_time = False
temporal_cnp_arborescence_directed_no_time.ctbf_requires_directed_distances = True
temporal_cnp_arborescence_directed_no_time.ctbf_solver_version = (
    TEMPORAL_ARBORESCENCE_SOLVER_VERSION
)
temporal_cnp_arborescence_directed_no_time.ctbf_order_ablation = None


__all__ = [
    "ORDERED_OCCURRENCE_INPUT_MODE",
    "TEMPORAL_ARBORESCENCE_SOLVER_VERSION",
    "temporal_cnp_arborescence",
    "temporal_cnp_arborescence_directed",
    "temporal_cnp_arborescence_directed_no_time",
    "temporal_cnp_arborescence_no_time",
    "uses_directed_distance_input",
    "uses_ordered_occurrence_input",
]
