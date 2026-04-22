import itertools
from collections import defaultdict

import networkx as nx
import numpy as np

from reconstructor_ancestor_selection import _is_biologically_plausible_ancestor
from reconstructor_utils import parse_distance_matrix
from simulator import Genotype


def _extend_biopsies(cell_lists):
    """
    Ensures that if a cell_id appears in multiple biopsy levels (e.g., 1 and 3),
    it also appears in all intermediate levels (e.g., 2).
    Returns a modified copy of cell_lists.
    """
    # Map cell_id -> all levels where it appears
    cell_levels = defaultdict(list)
    for level, lst in enumerate(cell_lists):
        for cell in lst:
            cell_levels[cell.cell_id].append(level)

    # Extend intermediate levels
    for cell_id, levels in cell_levels.items():
        if len(levels) > 1:
            min_l, max_l = min(levels), max(levels)
            for l in range(min_l, max_l + 1):
                # If missing at level l, copy from nearest existing one
                if all(c.cell_id != cell_id for c in cell_lists[l]):
                    # copy genome from nearest level
                    nearest_level = min(levels, key=lambda x: abs(x - l))
                    orig = next(c for c in cell_lists[nearest_level] if c.cell_id == cell_id)
                    copied_cell = Genotype(list(orig.genome), orig.cell_id)
                    copied_cell.node_id = orig.node_id  # keep same ID for consistency
                    cell_lists[l].append(copied_cell)
    return cell_lists


def build_evolution_tree_impl(
    cell_lists,
    seed=7,
    dist_matrix_path=None,
    r=2,
    only_nj=False,
    inids=None,
    indm=None,
    neighbor_joining=None,
):
    if dist_matrix_path:
        ids, full_dist_matrix = parse_distance_matrix(dist_matrix_path)
    elif inids is not None and indm is not None:
        ids, full_dist_matrix = inids, indm
    else:
        print("Please provide either dist_matrix_path or inids and indm")
        return None

    # extend biopsies so no cell skips levels ---
    cell_lists = _extend_biopsies(cell_lists)

    id_to_index = {cid: i for i, cid in enumerate(ids)}
    unique_node_counter = itertools.count(start=max(ids) + 1)

    for lst in cell_lists:
        for cell in lst:
            cell.node_id = cell.cell_id  # Retained for compatibility

    tree = nx.DiGraph()
    node_levels = defaultdict(lambda: None)
    for level, lst in enumerate(cell_lists[::-1]):
        for cell in lst:
            node_levels[cell] = level
            if cell.node_id in tree.nodes:
                if not only_nj: # for simple NJ we ignore cell copies from different biopsies
                    new_node_id = next(unique_node_counter)
                    cell.node_id = new_node_id
                    tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)
            else:
                tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)

    if not only_nj: # reconstruction logic
        for i in reversed(range(1, len(cell_lists))):
            upper, bottom = cell_lists[i - 1], cell_lists[i]
            for y in bottom:
                y_idx = id_to_index[y.cell_id]
                x_ks = []
                for x in upper:
                    x_idx = id_to_index[x.cell_id]
                    if full_dist_matrix[y_idx, x_idx] <= r:
                        x_ks.append(x)

                same_id_match = [x for x in x_ks if x.cell_id == y.cell_id]
                if same_id_match:
                    x = same_id_match[0]
                    tree.add_edge(x.node_id, y.node_id, weight=full_dist_matrix[y_idx, id_to_index[x.cell_id]])
                    continue

                # appearance constraint
                x_ks = [x for x in x_ks if _is_biologically_plausible_ancestor(x, y)]

                if len(x_ks) == 1:
                    x = x_ks[0]
                    tree.add_edge(x.node_id, y.node_id, weight=full_dist_matrix[y_idx, id_to_index[x.cell_id]])
                    continue

                if len(x_ks) > 1:
                    closest = min(x_ks, key=lambda x: full_dist_matrix[y_idx, id_to_index[x.cell_id]])
                    tree.add_edge(closest.node_id, y.node_id, weight=full_dist_matrix[y_idx, id_to_index[closest.cell_id]])
                    continue

                # the case when x_ks empty there is no neighbour near
                new_node_id = next(unique_node_counter)
                copied_cell = Genotype(list(y.genome), y.cell_id)
                copied_cell.node_id = new_node_id
                cell_lists[i - 1].append(copied_cell)
                node_levels[copied_cell] = len(cell_lists) - i
                tree.add_node(copied_cell.node_id, genome=copied_cell.genome, cell_id=copied_cell.cell_id)
                tree.add_edge(copied_cell.node_id, y.node_id, weight=0)

    final_cells = cell_lists[0]  # may contain same genotype 0 distance cells not merged due to appearance constraint
    if only_nj: # in NJ triggered by argument we assume no duplicate cell genotypes
        unique = {}
        for g in final_cells:
            if g.cell_id not in unique:
                unique[g.cell_id] = g
        final_cells = list(unique.values())

    final_ids = [cell.cell_id for cell in final_cells]
    dist_matrix = np.zeros((len(final_ids), len(final_ids)))
    for i, cell1 in enumerate(final_cells):
        for j, cell2 in enumerate(final_cells):
            idx1, idx2 = id_to_index[cell1.cell_id], id_to_index[cell2.cell_id]
            dist_matrix[i, j] = full_dist_matrix[idx1, idx2]

    max_id = next(unique_node_counter)
    tree, new_nodes, final_root = neighbor_joining(dist_matrix, final_cells, max_id, existing_tree=tree, seed=seed)

    for node in new_nodes:
        node_levels[node] = max(node_levels.values()) + 1

    return tree, node_levels, final_root
