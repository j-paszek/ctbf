import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np

from reconstructor_ancestor_selection import _is_biologically_plausible_ancestor
from simulator import Genotype


BiopsyCandidateSelector = Callable[[object, list, np.ndarray, dict, float], object | None]
BiopsyTieBreaker = Callable[[list, object, np.ndarray, dict], object]
BiopsyAttachmentStrategy = Callable[[nx.DiGraph, object, object, np.ndarray, dict], None]
BiopsyMissingParentStrategy = Callable[
    [object, list, nx.DiGraph, dict, object, int],
    object,
]
BiopsyLevelExtender = Callable[[list], list]
FinalCellSelector = Callable[[list], list]


@dataclass(frozen=True)
class BiopsyGuidedConfig:
    level_extender: BiopsyLevelExtender = None
    candidate_selector: BiopsyCandidateSelector = None
    candidate_tie_breaker: BiopsyTieBreaker = None
    attachment_strategy: BiopsyAttachmentStrategy = None
    missing_parent_strategy: BiopsyMissingParentStrategy = None
    only_nj_final_cell_selector: FinalCellSelector = None


def default_biopsy_guided_config():
    return BiopsyGuidedConfig(
        level_extender=extend_biopsy_levels,
        candidate_selector=select_biopsy_parent,
        candidate_tie_breaker=select_first_candidate,
        attachment_strategy=add_biopsy_attachment,
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
    return BiopsyGuidedConfig(
        level_extender=config.level_extender or defaults.level_extender,
        candidate_selector=candidate_selector,
        candidate_tie_breaker=candidate_tie_breaker,
        attachment_strategy=config.attachment_strategy or defaults.attachment_strategy,
        missing_parent_strategy=config.missing_parent_strategy or defaults.missing_parent_strategy,
        only_nj_final_cell_selector=(
            config.only_nj_final_cell_selector or defaults.only_nj_final_cell_selector
        ),
    )


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
    for cell_list in cell_lists:
        for cell in cell_list:
            cell.node_id = cell.cell_id


def initialize_biopsy_tree(cell_lists, unique_node_counter, only_nj=False):
    tree = nx.DiGraph()
    node_levels = defaultdict(lambda: None)

    for level, cell_list in enumerate(cell_lists[::-1]):
        for cell in cell_list:
            node_levels[cell] = level
            if cell.node_id in tree.nodes:
                if not only_nj:
                    cell.node_id = next(unique_node_counter)
                    tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)
            else:
                tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)

    return tree, node_levels


def distance_between(full_dist_matrix, id_to_index, a, b):
    return full_dist_matrix[id_to_index[a.cell_id], id_to_index[b.cell_id]]


def find_radius_candidates(y, upper_cells, full_dist_matrix, id_to_index, radius):
    y_idx = id_to_index[y.cell_id]
    return [
        x
        for x in upper_cells
        if full_dist_matrix[y_idx, id_to_index[x.cell_id]] <= radius
    ]


def select_same_id_candidate(candidates, y):
    same_id_matches = [x for x in candidates if x.cell_id == y.cell_id]
    if same_id_matches:
        return same_id_matches[0]
    return None


def filter_plausible_ancestor_candidates(candidates, y):
    return [x for x in candidates if _is_biologically_plausible_ancestor(x, y)]


def select_first_candidate(candidates, y, full_dist_matrix, id_to_index):
    return candidates[0]


def select_anticentral_candidate(candidates, y, full_dist_matrix, id_to_index):
    return max(
        candidates,
        key=lambda x: full_dist_matrix[id_to_index[x.cell_id]].sum(),
    )


def select_closest_candidate(candidates, y, full_dist_matrix, id_to_index, tie_breaker=None):
    tie_breaker = tie_breaker or select_first_candidate
    y_idx = id_to_index[y.cell_id]
    min_distance = min(full_dist_matrix[y_idx, id_to_index[x.cell_id]] for x in candidates)
    tied_candidates = [
        x
        for x in candidates
        if full_dist_matrix[y_idx, id_to_index[x.cell_id]] == min_distance
    ]

    if len(tied_candidates) == 1:
        return tied_candidates[0]
    return tie_breaker(tied_candidates, y, full_dist_matrix, id_to_index)


def add_biopsy_attachment(tree, parent, child, full_dist_matrix, id_to_index):
    tree.add_edge(
        parent.node_id,
        child.node_id,
        weight=distance_between(full_dist_matrix, id_to_index, child, parent),
    )


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
    node_levels[copied_cell] = copied_level
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

    return select_parent


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
    for level_index in reversed(range(1, len(cell_lists))):
        upper_cells = cell_lists[level_index - 1]
        bottom_cells = cell_lists[level_index]
        for y in bottom_cells:
            parent = config.candidate_selector(y, upper_cells, full_dist_matrix, id_to_index, radius)
            if parent is not None:
                config.attachment_strategy(tree, parent, y, full_dist_matrix, id_to_index)
                continue

            config.missing_parent_strategy(
                y,
                upper_cells,
                tree,
                node_levels,
                unique_node_counter,
                copied_level=len(cell_lists) - level_index,
            )


def deduplicate_cells_by_cell_id(cells):
    unique = {}
    for cell in cells:
        if cell.cell_id not in unique:
            unique[cell.cell_id] = cell
    return list(unique.values())


def build_final_distance_matrix(final_cells, full_dist_matrix, id_to_index):
    final_ids = [cell.cell_id for cell in final_cells]
    dist_matrix = np.zeros((len(final_ids), len(final_ids)))
    for i, cell1 in enumerate(final_cells):
        for j, cell2 in enumerate(final_cells):
            idx1 = id_to_index[cell1.cell_id]
            idx2 = id_to_index[cell2.cell_id]
            dist_matrix[i, j] = full_dist_matrix[idx1, idx2]
    return dist_matrix


def assign_new_node_levels(new_nodes, node_levels):
    for node in new_nodes:
        node_levels[node] = max(node_levels.values()) + 1


__all__ = [
    "BiopsyGuidedConfig",
    "add_biopsy_attachment",
    "assign_compatible_node_ids",
    "assign_new_node_levels",
    "build_final_distance_matrix",
    "copy_missing_parent_to_upper",
    "deduplicate_cells_by_cell_id",
    "distance_between",
    "extend_biopsy_levels",
    "filter_plausible_ancestor_candidates",
    "find_radius_candidates",
    "initialize_biopsy_tree",
    "make_id_to_index",
    "make_unique_node_counter",
    "make_biopsy_parent_selector",
    "default_biopsy_guided_config",
    "normalize_biopsy_guided_config",
    "reconstruct_biopsy_layers",
    "select_biopsy_parent",
    "select_anticentral_candidate",
    "select_closest_candidate",
    "select_first_candidate",
    "select_same_id_candidate",
]
