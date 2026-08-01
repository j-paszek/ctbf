import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np
from reconstructor_engine import Orientation, PairChoice, run_agglomerative_reconstruction
from reconstructor_pair_selection import _best_pair_from_score_matrix
from reconstructor_plausibility import is_biologically_plausible_ancestor
from simulator import Genotype


BiopsyCandidateSelector = Callable[[object, list, np.ndarray, dict, float], object | None]
BiopsyTieBreaker = Callable[[list, object, np.ndarray, dict], object]
BiopsyAttachmentStrategy = Callable[[nx.DiGraph, object, object, np.ndarray, dict], None]
BiopsyGroupAttachmentStrategy = Callable[
    [nx.DiGraph, object, list, np.ndarray, dict, object, dict, int],
    None,
]
BiopsyMissingParentStrategy = Callable[
    [object, list, nx.DiGraph, dict, object, int],
    object,
]
BiopsyLevelExtender = Callable[[list], list]
FinalCellSelector = Callable[[list], list]


@dataclass(frozen=True)
class BiopsySubtreeConfig:
    pair_selector: object = None
    ancestor_selector: object = None
    merge_strategy: object = None
    distance_update: object = None
    configure_state: object = None
    seed: int = 7


@dataclass(frozen=True)
class BiopsyGuidedConfig:
    level_extender: BiopsyLevelExtender = None
    candidate_selector: BiopsyCandidateSelector = None
    candidate_tie_breaker: BiopsyTieBreaker = None
    attachment_strategy: BiopsyAttachmentStrategy = None
    group_attachment_strategy: BiopsyGroupAttachmentStrategy = None
    missing_parent_strategy: BiopsyMissingParentStrategy = None
    only_nj_final_cell_selector: FinalCellSelector = None


def default_biopsy_guided_config():
    return BiopsyGuidedConfig(
        level_extender=extend_biopsy_levels,
        candidate_selector=select_biopsy_parent,
        candidate_tie_breaker=select_first_candidate,
        attachment_strategy=add_biopsy_attachment,
        group_attachment_strategy=make_direct_group_attachment_strategy(add_biopsy_attachment),
        missing_parent_strategy=copy_missing_parent_to_upper,
        only_nj_final_cell_selector=deduplicate_cells_by_cell_id,
    )


def normalize_biopsy_guided_config(config=None):
    if config is None:
        return default_biopsy_guided_config()

    defaults = default_biopsy_guided_config()
    candidate_tie_breaker = config.candidate_tie_breaker or defaults.candidate_tie_breaker
    candidate_selector = (
        config.candidate_selector
        if config.candidate_selector is not None
        else make_biopsy_parent_selector(candidate_tie_breaker)
    )
    attachment_strategy = config.attachment_strategy or defaults.attachment_strategy
    group_attachment_strategy = (
        config.group_attachment_strategy
        if config.group_attachment_strategy is not None
        else make_direct_group_attachment_strategy(attachment_strategy)
    )
    return BiopsyGuidedConfig(
        level_extender=config.level_extender or defaults.level_extender,
        candidate_selector=candidate_selector,
        candidate_tie_breaker=candidate_tie_breaker,
        attachment_strategy=attachment_strategy,
        group_attachment_strategy=group_attachment_strategy,
        missing_parent_strategy=config.missing_parent_strategy or defaults.missing_parent_strategy,
        only_nj_final_cell_selector=(
            config.only_nj_final_cell_selector or defaults.only_nj_final_cell_selector
        ),
    )


def clone_reconstruction_cell(cell):
    """Copy observable state while discarding simulator-local graph identity."""
    return Genotype(
        np.array(cell.genome, copy=True),
        cell.cell_id,
        generation=getattr(cell, "generation", None),
        cell_id=cell.cell_id,
    )


def copy_reconstruction_cell_lists(cell_lists):
    clones = {}

    def copy_cell(cell, level_index):
        key = (id(cell), level_index)
        if key not in clones:
            clones[key] = clone_reconstruction_cell(cell)
        return clones[key]

    return [
        [copy_cell(cell, level_index) for cell in cell_list]
        for level_index, cell_list in enumerate(cell_lists)
    ]


def extend_biopsy_levels(cell_lists):
    """
    Ensure that a cell observed in multiple biopsy levels also appears in
    intermediate levels. This preserves the previous in-place behavior.
    """
    cell_levels = defaultdict(list)
    for level, cell_list in enumerate(cell_lists):
        for cell in cell_list:
            cell_levels[cell.cell_id].append(level)

    for cell_id, levels in cell_levels.items():
        if len(levels) <= 1:
            continue

        min_level, max_level = min(levels), max(levels)
        for level in range(min_level, max_level + 1):
            if any(cell.cell_id == cell_id for cell in cell_lists[level]):
                continue

            nearest_level = min(levels, key=lambda existing_level: abs(existing_level - level))
            original = next(cell for cell in cell_lists[nearest_level] if cell.cell_id == cell_id)
            copied_cell = Genotype(list(original.genome), original.cell_id)
            copied_cell.node_id = original.node_id
            cell_lists[level].append(copied_cell)

    return cell_lists


def make_id_to_index(ids):
    return {cell_id: index for index, cell_id in enumerate(ids)}


def make_unique_node_counter(ids):
    return itertools.count(start=max(ids) + 1)


def assign_compatible_node_ids(cell_lists):
    """Assign reconstruction-local graph ids without using simulator node_id values."""
    for cell_list in cell_lists:
        for cell in cell_list:
            cell.node_id = cell.cell_id


def initialize_biopsy_tree(cell_lists, unique_node_counter, only_nj=False):
    tree = nx.DiGraph()
    node_levels = defaultdict(lambda: None)

    for level, cell_list in enumerate(cell_lists[::-1]):
        for cell in cell_list:
            if cell.node_id in tree.nodes:
                if not only_nj:
                    cell.node_id = next(unique_node_counter)
                    tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)
            else:
                tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)
            node_levels[cell.node_id] = level

    return tree, node_levels


def distance_between(full_dist_matrix, id_to_index, a, b):
    return full_dist_matrix[id_to_index[a.cell_id], id_to_index[b.cell_id]]


def _cell_indices(cells, id_to_index):
    return np.fromiter(
        (id_to_index[cell.cell_id] for cell in cells),
        dtype=int,
        count=len(cells),
    )


def _find_radius_candidates_from_indices(y, upper_cells, upper_indices, full_dist_matrix, id_to_index, radius):
    y_idx = id_to_index[y.cell_id]
    in_radius = full_dist_matrix[y_idx, upper_indices] <= radius
    return [
        x
        for x, is_candidate in zip(upper_cells, in_radius)
        if is_candidate
    ]


def find_radius_candidates(y, upper_cells, full_dist_matrix, id_to_index, radius):
    if not upper_cells:
        return []

    upper_indices = _cell_indices(upper_cells, id_to_index)
    return _find_radius_candidates_from_indices(
        y,
        upper_cells,
        upper_indices,
        full_dist_matrix,
        id_to_index,
        radius,
    )


def select_same_id_candidate(candidates, y):
    same_id_matches = [x for x in candidates if x.cell_id == y.cell_id]
    if same_id_matches:
        return same_id_matches[0]
    return None


def filter_plausible_ancestor_candidates(candidates, y):
    return [x for x in candidates if is_biologically_plausible_ancestor(x, y)]


def select_first_candidate(candidates, y, full_dist_matrix, id_to_index):
    return candidates[0]


def select_anticentral_candidate(candidates, y, full_dist_matrix, id_to_index):
    row_sums = full_dist_matrix.sum(axis=1)
    candidate_indices = _cell_indices(candidates, id_to_index)
    return _select_anticentral_candidate_from_indices(candidates, candidate_indices, row_sums)


def _select_anticentral_candidate_from_indices(candidates, candidate_indices, row_sums):
    return candidates[int(np.argmax(row_sums[candidate_indices]))]


def _select_closest_candidate_from_indices(
    candidates,
    candidate_indices,
    y,
    full_dist_matrix,
    id_to_index,
    tie_breaker=None,
    row_sums=None,
):
    tie_breaker = tie_breaker or select_first_candidate
    y_idx = id_to_index[y.cell_id]
    distances = full_dist_matrix[y_idx, candidate_indices]
    if np.issubdtype(distances.dtype, np.inexact) and np.isnan(distances[0]):
        tied_candidates = []
        tied_indices = candidate_indices[:0]
    else:
        comparable_distances = distances
        if np.issubdtype(distances.dtype, np.inexact):
            nan_mask = np.isnan(distances)
            if np.any(nan_mask):
                comparable_distances = distances.copy()
                comparable_distances[nan_mask] = np.inf
        min_distance = np.min(comparable_distances)
        tied_mask = distances == min_distance
        tied_candidates = [
            x
            for x, is_tied in zip(candidates, tied_mask)
            if is_tied
        ]
        tied_indices = candidate_indices[tied_mask]

    if len(tied_candidates) == 1:
        return tied_candidates[0]
    if tie_breaker is select_anticentral_candidate and row_sums is not None:
        return _select_anticentral_candidate_from_indices(tied_candidates, tied_indices, row_sums)
    return tie_breaker(tied_candidates, y, full_dist_matrix, id_to_index)


def select_closest_candidate(candidates, y, full_dist_matrix, id_to_index, tie_breaker=None):
    candidate_indices = _cell_indices(candidates, id_to_index)
    return _select_closest_candidate_from_indices(
        candidates,
        candidate_indices,
        y,
        full_dist_matrix,
        id_to_index,
        tie_breaker=tie_breaker,
    )


def add_biopsy_attachment(tree, parent, child, full_dist_matrix, id_to_index):
    tree.add_edge(
        parent.node_id,
        child.node_id,
        weight=distance_between(full_dist_matrix, id_to_index, child, parent),
    )


def make_direct_group_attachment_strategy(attachment_strategy=None):
    attachment_strategy = attachment_strategy or add_biopsy_attachment

    def attach_group(
        tree,
        parent,
        children,
        full_dist_matrix,
        id_to_index,
        unique_node_counter,
        node_levels,
        child_level,
    ):
        for child in children:
            attachment_strategy(tree, parent, child, full_dist_matrix, id_to_index)

    return attach_group


def copy_missing_parent_to_upper(
    y,
    upper_cells,
    tree,
    node_levels,
    unique_node_counter,
    copied_level,
):
    copied_cell = Genotype(list(y.genome), y.cell_id)
    copied_cell.node_id = next(unique_node_counter)
    upper_cells.append(copied_cell)
    node_levels[copied_cell.node_id] = copied_level
    tree.add_node(copied_cell.node_id, genome=copied_cell.genome, cell_id=copied_cell.cell_id)
    tree.add_edge(copied_cell.node_id, y.node_id, weight=0)
    return copied_cell


def select_biopsy_parent(
    y,
    upper_cells,
    full_dist_matrix,
    id_to_index,
    radius,
    tie_breaker=None,
):
    candidates = find_radius_candidates(y, upper_cells, full_dist_matrix, id_to_index, radius)

    same_id_candidate = select_same_id_candidate(candidates, y)
    if same_id_candidate is not None:
        return same_id_candidate

    plausible_candidates = filter_plausible_ancestor_candidates(candidates, y)
    if len(plausible_candidates) == 1:
        return plausible_candidates[0]
    if len(plausible_candidates) > 1:
        return select_closest_candidate(
            plausible_candidates,
            y,
            full_dist_matrix,
            id_to_index,
            tie_breaker=tie_breaker,
        )

    return None


def _select_biopsy_parent_from_indices(
    y,
    upper_cells,
    upper_indices,
    full_dist_matrix,
    id_to_index,
    radius,
    tie_breaker=None,
    row_sums=None,
):
    candidates = _find_radius_candidates_from_indices(
        y,
        upper_cells,
        upper_indices,
        full_dist_matrix,
        id_to_index,
        radius,
    )

    same_id_candidate = select_same_id_candidate(candidates, y)
    if same_id_candidate is not None:
        return same_id_candidate

    plausible_candidates = filter_plausible_ancestor_candidates(candidates, y)
    if len(plausible_candidates) == 1:
        return plausible_candidates[0]
    if len(plausible_candidates) > 1:
        plausible_indices = _cell_indices(plausible_candidates, id_to_index)
        return _select_closest_candidate_from_indices(
            plausible_candidates,
            plausible_indices,
            y,
            full_dist_matrix,
            id_to_index,
            tie_breaker=tie_breaker,
            row_sums=row_sums,
        )

    return None


def make_biopsy_parent_selector(tie_breaker=None):
    def select_parent(y, upper_cells, full_dist_matrix, id_to_index, radius):
        return select_biopsy_parent(
            y,
            upper_cells,
            full_dist_matrix,
            id_to_index,
            radius,
            tie_breaker=tie_breaker,
        )

    select_parent._ctbf_default_biopsy_parent_selector = True
    select_parent._ctbf_tie_breaker = tie_breaker
    return select_parent


def _default_biopsy_parent_tie_breaker(candidate_selector):
    if candidate_selector is select_biopsy_parent:
        return True, None
    if getattr(candidate_selector, "_ctbf_default_biopsy_parent_selector", False):
        return True, getattr(candidate_selector, "_ctbf_tie_breaker", None)
    return False, None


def _minimum_distance_pair_selector(state):
    i, j, distance = _best_pair_from_score_matrix(state.D)
    return PairChoice(i, j, score=distance)


def _pair_order_ancestor_selector(state, pair):
    return Orientation(pair.i, pair.j)


def _node_root_strategy(state):
    return state.tree, state.new_nodes, state.node_list[0]


def _consume_used_node_ids(unique_node_counter, used_count):
    for _ in range(max(used_count - 1, 0)):
        next(unique_node_counter)


def normalize_biopsy_subtree_config(
    subtree_config=None,
    *,
    pair_selector=None,
    ancestor_selector=None,
    merge_strategy=None,
    distance_update=None,
    configure_state=None,
    seed=None,
):
    subtree_config = subtree_config or BiopsySubtreeConfig()
    return BiopsySubtreeConfig(
        pair_selector=pair_selector or subtree_config.pair_selector or _minimum_distance_pair_selector,
        ancestor_selector=ancestor_selector or subtree_config.ancestor_selector or _pair_order_ancestor_selector,
        merge_strategy=merge_strategy or subtree_config.merge_strategy,
        distance_update=distance_update or subtree_config.distance_update,
        configure_state=configure_state or subtree_config.configure_state,
        seed=seed if seed is not None else subtree_config.seed,
    )


def make_binarized_group_attachment_strategy(
    subtree_config=None,
    *,
    pair_selector=None,
    ancestor_selector=None,
    merge_strategy=None,
    distance_update=None,
    configure_state=None,
    seed=None,
):
    subtree_config = normalize_biopsy_subtree_config(
        subtree_config,
        pair_selector=pair_selector,
        ancestor_selector=ancestor_selector,
        merge_strategy=merge_strategy,
        distance_update=distance_update,
        configure_state=configure_state,
        seed=seed,
    )

    def attach_group(
        tree,
        parent,
        children,
        full_dist_matrix,
        id_to_index,
        unique_node_counter,
        node_levels,
        child_level,
    ):
        if len(children) == 1:
            add_biopsy_attachment(tree, parent, children[0], full_dist_matrix, id_to_index)
            return

        child_dist_matrix = build_final_distance_matrix(children, full_dist_matrix, id_to_index)
        first_internal_id = next(unique_node_counter)
        kwargs = {
            "seed": subtree_config.seed,
            "existing_tree": tree,
            "pair_selector": subtree_config.pair_selector,
            "ancestor_selector": subtree_config.ancestor_selector,
            "root_strategy": _node_root_strategy,
        }
        if subtree_config.merge_strategy is not None:
            kwargs["merge_strategy"] = subtree_config.merge_strategy
        if subtree_config.distance_update is not None:
            kwargs["distance_update"] = subtree_config.distance_update
        if subtree_config.configure_state is not None:
            kwargs["configure_state"] = subtree_config.configure_state

        tree, new_nodes, group_root = run_agglomerative_reconstruction(
            child_dist_matrix,
            children,
            first_internal_id - 1,
            **kwargs,
        )
        _consume_used_node_ids(unique_node_counter, len(new_nodes))

        for node in new_nodes:
            node_levels[node.node_id] = child_level

        tree.add_edge(
            parent.node_id,
            group_root.node_id,
            weight=distance_between(full_dist_matrix, id_to_index, group_root, parent),
        )

    return attach_group


def reconstruct_biopsy_layers(
    cell_lists,
    tree,
    node_levels,
    full_dist_matrix,
    id_to_index,
    radius,
    unique_node_counter,
    config=None,
):
    config = normalize_biopsy_guided_config(config)
    row_sums = full_dist_matrix.sum(axis=1)
    use_default_candidate_selector, default_tie_breaker = _default_biopsy_parent_tie_breaker(
        config.candidate_selector
    )

    for level_index in reversed(range(1, len(cell_lists))):
        upper_cells = cell_lists[level_index - 1]
        bottom_cells = cell_lists[level_index]
        children_by_parent = defaultdict(list)
        upper_indices = _cell_indices(upper_cells, id_to_index)
        for y in bottom_cells:
            if use_default_candidate_selector:
                parent = _select_biopsy_parent_from_indices(
                    y,
                    upper_cells,
                    upper_indices,
                    full_dist_matrix,
                    id_to_index,
                    radius,
                    tie_breaker=default_tie_breaker,
                    row_sums=row_sums,
                )
            else:
                parent = config.candidate_selector(y, upper_cells, full_dist_matrix, id_to_index, radius)

            if parent is not None:
                children_by_parent[parent].append(y)
                continue

            upper_len_before = len(upper_cells)
            config.missing_parent_strategy(
                y,
                upper_cells,
                tree,
                node_levels,
                unique_node_counter,
                copied_level=len(cell_lists) - level_index,
            )
            if use_default_candidate_selector and len(upper_cells) != upper_len_before:
                upper_indices = _cell_indices(upper_cells, id_to_index)

        child_level = len(cell_lists) - level_index - 1
        for parent, children in children_by_parent.items():
            config.group_attachment_strategy(
                tree,
                parent,
                children,
                full_dist_matrix,
                id_to_index,
                unique_node_counter,
                node_levels,
                child_level,
            )


def deduplicate_cells_by_cell_id(cells):
    unique = {}
    for cell in cells:
        if cell.cell_id not in unique:
            unique[cell.cell_id] = cell
    return list(unique.values())


def build_final_distance_matrix(final_cells, full_dist_matrix, id_to_index):
    indices = [id_to_index[cell.cell_id] for cell in final_cells]
    return np.asarray(full_dist_matrix[np.ix_(indices, indices)], dtype=float)


def assign_new_node_levels(new_nodes, node_levels):
    max_level = max((level for level in node_levels.values() if level is not None), default=-1)
    for node in new_nodes:
        max_level += 1
        node_levels[node.node_id] = max_level


__all__ = [
    "BiopsyGuidedConfig",
    "BiopsySubtreeConfig",
    "add_biopsy_attachment",
    "assign_compatible_node_ids",
    "assign_new_node_levels",
    "build_final_distance_matrix",
    "clone_reconstruction_cell",
    "copy_missing_parent_to_upper",
    "copy_reconstruction_cell_lists",
    "deduplicate_cells_by_cell_id",
    "distance_between",
    "extend_biopsy_levels",
    "filter_plausible_ancestor_candidates",
    "find_radius_candidates",
    "initialize_biopsy_tree",
    "make_id_to_index",
    "make_unique_node_counter",
    "make_binarized_group_attachment_strategy",
    "make_biopsy_parent_selector",
    "make_direct_group_attachment_strategy",
    "default_biopsy_guided_config",
    "normalize_biopsy_guided_config",
    "normalize_biopsy_subtree_config",
    "reconstruct_biopsy_layers",
    "select_biopsy_parent",
    "select_anticentral_candidate",
    "select_closest_candidate",
    "select_first_candidate",
    "select_same_id_candidate",
]
