import random
from functools import partial

import networkx as nx
import numpy as np

from reconstructor_ancestor_selection import (
    _choose_parent_full_nj,
    _choose_parent_hybrid_inv_centrality,
    _choose_parent_with_plausibility_fallback,
    _final_parent_choice_full_matrix,
)
from reconstructor_pair_selection import (
    _select_pair_cps,
    _select_pair_full,
    _select_pair_hybrid,
    _select_pair_hybrid_inv_centrality,
)
from simulator import Genotype


NJ_REC_TR_ROOT_ID = -1


def neighbor_joining_standard(dist_matrix, cells, max_id, seed=7, existing_tree=None):
    D = dist_matrix.copy()
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}
    id_map = {i: cells[i] for i in range(len(cells))}
    next_id = max_id + 1

    for cell in cells:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)

    while len(D) > 2:
        n = len(D)
        total_dist = D.sum(axis=1)
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i][j] = (n - 2) * D[i][j] - total_dist[i] - total_dist[j]

        i, j = divmod(np.argmin(Q), n)
        if j < i:
            i, j = j, i

        delta = (total_dist[i] - total_dist[j]) / (n - 2)
        limb_len_i = 0.5 * (D[i][j] + delta)
        limb_len_j = 0.5 * (D[i][j] - delta)

        id_i, id_j = id_map[i], id_map[j]
        new_cell = Genotype(None, next_id)
        next_id += 1

        tree.add_node(new_cell.node_id, genome=new_cell.genome, cell_id=None)
        tree.add_edge(new_cell.node_id, id_i.node_id, weight=limb_len_i)
        tree.add_edge(new_cell.node_id, id_j.node_id, weight=limb_len_j)
        new_nodes[new_cell] = (id_i, id_j)

        new_row = [(D[i][k] + D[j][k] - D[i][j]) / 2 for k in range(n) if k != i and k != j]
        D = np.delete(D, [i, j], axis=0)
        D = np.delete(D, [i, j], axis=1)
        D = np.vstack([D, new_row])
        new_col = np.append(new_row, [0])[:, None]
        D = np.hstack([D, new_col])

        keys = [id_map[k] for k in range(n) if k != i and k != j]
        id_map = {k: v for k, v in enumerate(keys)}
        id_map[len(id_map)] = new_cell

    if len(id_map) == 2:
        id1, id2 = id_map[0], id_map[1]
        root_cell = Genotype(None, NJ_REC_TR_ROOT_ID)
        tree.add_node(root_cell.node_id, genome=root_cell.genome, cell_id=None)
        tree.add_edge(root_cell.node_id, id1.node_id, weight=D[0][1] / 2)
        tree.add_edge(root_cell.node_id, id2.node_id, weight=D[0][1] / 2)
        new_nodes[root_cell] = (id1, id2)
    elif len(id_map) == 1:
        return tree, new_nodes, id_map[0].cell_id

    return tree, new_nodes, NJ_REC_TR_ROOT_ID


def neighbor_joining_baseline(dist_matrix, cells, max_id, seed=7, existing_tree=None):
    """
    Simplest NJ-like reconstruction.
    - Always pick the pair with minimal D[i,j]
    - Parent = node with smaller sum of distances
    - No biological plausibility
    - No full-information root correction
    - No CPS/Hybrid scoring
    - No global heuristics
    """
    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}
    node_list = [cells[i] for i in range(len(cells))]

    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    while len(node_list) > 1:
        n = len(D)
        min_val = None
        best_pair = (0, 1)

        tri_i, tri_j = np.triu_indices(n, k=1)
        for i, j in zip(tri_i, tri_j):
            d = D[i, j]
            if min_val is None or d < min_val:
                min_val = d
                best_pair = (i, j)

        i, j = best_pair

        others = [k for k in range(n) if k != i and k != j]
        sum_i = D[i, others].sum() if others else 0
        sum_j = D[j, others].sum() if others else 0

        if sum_i < sum_j:
            parent_idx, child_idx = i, j
        elif sum_j < sum_i:
            parent_idx, child_idx = j, i
        else:
            parent_idx, child_idx = (i, j) if rng.random() < 0.5 else (j, i)

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        internal = type(parent_leaf)(
            genome=parent_leaf.genome,
            node_id=next_id,
            cell_id=parent_leaf.cell_id,
        )
        next_id += 1

        tree.add_node(internal.node_id, genome=internal.genome, cell_id=internal.cell_id)
        tree.add_edge(internal.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(internal.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))
        new_nodes[internal] = (parent_leaf, child_leaf)

        keep = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep, keep)]
        node_list[parent_idx] = internal
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbour_joining_core(
    dist_matrix,
    cells,
    max_id,
    seed=7,
    existing_tree=None,
    select_pair_func=_select_pair_full,
    select_ancestor_func=_choose_parent_full_nj,
    full_information=False,
):
    rng = random.Random(seed)
    D_full = dist_matrix.copy().astype(float)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}
    node_list = [cells[i] for i in range(len(cells))]
    origin_index = {node: idx for idx, node in enumerate(node_list)}

    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    while len(node_list) > 1:
        n = len(D)
        i, j = select_pair_func(D, node_list, rng, minimize=True)

        if len(node_list) == 2:
            x = node_list[0]
            y = node_list[1]
            parent_leaf, child_leaf = _final_parent_choice_full_matrix(
                x, y, D_full, origin_index, rng, select_ancestor_func
            )
            parent_idx = 0 if node_list[0] is parent_leaf else 1
            child_idx = 1 - parent_idx
        else:
            parent_idx, child_idx = _choose_parent_with_plausibility_fallback(
                D,
                D_full,
                origin_index,
                node_list,
                i,
                j,
                rng,
                select_ancestor_func,
                full_information=full_information,
            )

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        internal_node = type(parent_leaf)(
            genome=parent_leaf.genome,
            node_id=next_id,
            cell_id=parent_leaf.cell_id,
        )
        next_id += 1
        origin_index[internal_node] = origin_index[parent_leaf]
        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))
        new_nodes[internal_node] = (parent_leaf, child_leaf)

        keep_indices = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep_indices, keep_indices)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def make_nj_full_variant(full_information):
    def _variant(dist_matrix, cells, max_id, seed=7, existing_tree=None):
        return neighbour_joining_core(
            dist_matrix=dist_matrix,
            cells=cells,
            max_id=max_id,
            seed=seed,
            existing_tree=existing_tree,
            select_pair_func=_select_pair_full,
            select_ancestor_func=_choose_parent_full_nj,
            full_information=full_information,
        )

    _variant.__name__ = f"neighbor_joining_full_{'full' if full_information else 'partial'}"
    return _variant


def make_nj_full_cps_variant(full_information: bool):
    def _variant(dist_matrix, cells, max_id, seed=7, existing_tree=None):
        return neighbour_joining_core(
            dist_matrix=dist_matrix,
            cells=cells,
            max_id=max_id,
            seed=seed,
            existing_tree=existing_tree,
            select_pair_func=_select_pair_cps,
            select_ancestor_func=_choose_parent_full_nj,
            full_information=full_information,
        )

    _variant.__name__ = f"neighbor_joining_full_cps_{'full' if full_information else 'partial'}"
    return _variant


def make_nj_hybrid_variant(full_information: bool, alpha=1.0, beta=0.5):
    def _variant(dist_matrix, cells, max_id, seed=7, existing_tree=None):
        select_pair_func = partial(_select_pair_hybrid, alpha=alpha, beta=beta)

        return neighbour_joining_core(
            dist_matrix=dist_matrix,
            cells=cells,
            max_id=max_id,
            seed=seed,
            existing_tree=existing_tree,
            select_pair_func=select_pair_func,
            select_ancestor_func=_choose_parent_full_nj,
            full_information=full_information,
        )

    _variant.__name__ = f"neighbor_joining_hybrid_{'full' if full_information else 'partial'}"
    return _variant


def make_nj_hybrid_inv_cent_variant(
    full_information: bool,
    alpha=1.0,
    beta=0.5,
    epsilon=1e-6,
):
    def _variant(dist_matrix, cells, max_id, seed=7, existing_tree=None):
        select_pair_func = partial(
            _select_pair_hybrid_inv_centrality,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
        )
        select_ancestor_func = partial(
            _choose_parent_hybrid_inv_centrality,
            epsilon=epsilon,
        )

        return neighbour_joining_core(
            dist_matrix=dist_matrix,
            cells=cells,
            max_id=max_id,
            seed=seed,
            existing_tree=existing_tree,
            select_pair_func=select_pair_func,
            select_ancestor_func=select_ancestor_func,
            full_information=full_information,
        )

    _variant.__name__ = (
        f"neighbor_joining_hybrid_inverse_centrality_"
        f"{'full' if full_information else 'partial'}"
    )
    return _variant
