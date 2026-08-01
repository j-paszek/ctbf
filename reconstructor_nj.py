import random
from functools import partial
from inspect import signature

import networkx as nx
import numpy as np

from reconstructor_ancestor_selection import (
    _choose_parent_full_nj,
    _choose_parent_hybrid_inv_centrality,
    _choose_parent_with_plausibility_fallback,
    _final_parent_choice_full_matrix,
)
from reconstructor_pair_selection import (
    _best_pair_from_score_matrix,
    _select_pair_cps,
    _select_pair_full,
    _select_pair_hybrid,
    _select_pair_hybrid_inv_centrality,
)
from reconstructor_engine import (
    Orientation,
    PairChoice,
    run_agglomerative_reconstruction,
)
from reconstructor_metrics import nj_q_matrix, sum_distance_centrality, upper_triangle_indices
from simulator import Genotype


NJ_REC_TR_ROOT_ID = -1
ROOTED_LABELED_GLOBAL_ROW_SUM_CONTEXT_KEY = "rooted_labeled_global_row_sum"


def _validated_distance_copy(dist_matrix, cells):
    D = np.asarray(dist_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if D.shape[0] != len(cells):
        raise ValueError(
            "Distance matrix size must match the number of reconstruction cells."
        )
    if len(cells) == 0:
        raise ValueError("Reconstruction requires at least one cell.")
    if len({cell.node_id for cell in cells}) != len(cells):
        raise ValueError("Reconstruction cells must have unique node_id values.")
    return D.copy()


def _next_available_node_id(tree, preferred_id, fallback_id):
    if preferred_id not in tree:
        return preferred_id

    candidate = fallback_id
    while candidate in tree:
        candidate += 1
    return candidate


def _select_min_distance_pair(state):
    i, j, distance = _best_pair_from_score_matrix(state.D)
    return PairChoice(i, j, score=distance)


def _select_sum_distance_parent(state, pair):
    i, j = pair.i, pair.j
    sum_i = state.D[i].sum() - state.D[i, i] - state.D[i, j]
    sum_j = state.D[j].sum() - state.D[j, j] - state.D[j, i]

    if sum_i < sum_j:
        parent_idx, child_idx = i, j
    elif sum_j < sum_i:
        parent_idx, child_idx = j, i
    else:
        parent_idx, child_idx = (i, j) if state.rng.random() < 0.5 else (j, i)

    return Orientation(parent_idx, child_idx)


def _legacy_pair_selector(select_pair_func):
    try:
        supports_metadata = "return_metadata" in signature(select_pair_func).parameters
    except (TypeError, ValueError):
        supports_metadata = False

    def select_pair(state):
        if supports_metadata:
            result = select_pair_func(
                state.D,
                state.node_list,
                state.rng,
                minimize=True,
                return_metadata=True,
            )
        else:
            result = select_pair_func(state.D, state.node_list, state.rng, minimize=True)

        if len(result) == 3 and isinstance(result[2], dict):
            i, j, metadata = result
        else:
            i, j = result
            metadata = {}
        return PairChoice(i, j, metadata=metadata)

    return select_pair


def _legacy_parent_selector(select_ancestor_func, full_information):
    def select_parent(state, pair):
        if len(state.node_list) == 2:
            x = state.node_list[0]
            y = state.node_list[1]
            parent_leaf, child_leaf = _final_parent_choice_full_matrix(
                x,
                y,
                state.D_full,
                state.origin_index,
                state.rng,
                select_ancestor_func,
            )
            parent_idx = 0 if state.node_list[0] is parent_leaf else 1
            child_idx = 1 - parent_idx
        else:
            parent_idx, child_idx = _choose_parent_with_plausibility_fallback(
                state.D,
                state.D_full,
                state.origin_index,
                state.node_list,
                pair.i,
                pair.j,
                state.rng,
                select_ancestor_func,
                full_information=full_information,
                pair_metadata=pair.metadata,
            )

        return Orientation(parent_idx, child_idx)

    return select_parent


def neighbor_joining_standard(dist_matrix, cells, max_id, seed=7, existing_tree=None):
    """Classical NJ with unlabeled latent nodes and a synthetic final-join root.

    The topology and limb-length recurrence are classical NJ. The returned
    directed tree is a rooted rendering used by CTBF's rooted evaluators; NJ
    itself does not infer a biological root.
    """
    D = _validated_distance_copy(dist_matrix, cells)
    tree = existing_tree if existing_tree is not None else nx.DiGraph()
    new_nodes = {}
    id_map = {i: cells[i] for i in range(len(cells))}
    next_id = max_id + 1

    for cell in cells:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)

    while len(D) > 2:
        n = len(D)
        total_dist = sum_distance_centrality(D)
        Q = nj_q_matrix(D)

        i, j = divmod(np.argmin(Q), n)
        if j < i:
            i, j = j, i

        delta = (total_dist[i] - total_dist[j]) / (n - 2)
        limb_len_i = 0.5 * (D[i][j] + delta)
        limb_len_j = 0.5 * (D[i][j] - delta)

        id_i, id_j = id_map[i], id_map[j]
        next_id = _next_available_node_id(tree, next_id, next_id + 1)
        new_cell = Genotype(None, next_id)
        next_id += 1

        tree.add_node(new_cell.node_id, genome=new_cell.genome, cell_id=None)
        tree.add_edge(new_cell.node_id, id_i.node_id, weight=limb_len_i)
        tree.add_edge(new_cell.node_id, id_j.node_id, weight=limb_len_j)
        new_nodes[new_cell] = (id_i, id_j)

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[[i, j]] = False
        new_row = (D[i, keep_mask] + D[j, keep_mask] - D[i, j]) / 2
        next_D = np.empty((n - 1, n - 1), dtype=np.result_type(D, float))
        next_D[:-1, :-1] = D[keep_mask][:, keep_mask]
        next_D[-1, :-1] = new_row
        next_D[:-1, -1] = new_row
        next_D[-1, -1] = 0.0
        D = next_D

        keys = [id_map[k] for k in np.flatnonzero(keep_mask)]
        id_map = {k: v for k, v in enumerate(keys)}
        id_map[len(id_map)] = new_cell

    if len(id_map) == 2:
        id1, id2 = id_map[0], id_map[1]
        root_id = _next_available_node_id(tree, NJ_REC_TR_ROOT_ID, next_id)
        root_cell = Genotype(None, root_id)
        tree.add_node(root_cell.node_id, genome=root_cell.genome, cell_id=None)
        tree.add_edge(root_cell.node_id, id1.node_id, weight=D[0][1] / 2)
        tree.add_edge(root_cell.node_id, id2.node_id, weight=D[0][1] / 2)
        new_nodes[root_cell] = (id1, id2)
    elif len(id_map) == 1:
        return tree, new_nodes, id_map[0].node_id

    return tree, new_nodes, root_cell.node_id


def neighbor_joining_classical(dist_matrix, cells, max_id, seed=7, existing_tree=None):
    """Publication-facing name for the partial-output classical NJ routine."""
    return neighbor_joining_standard(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
    )


def _configure_rooted_labeled_state(state):
    state.context[ROOTED_LABELED_GLOBAL_ROW_SUM_CONTEXT_KEY] = (
        sum_distance_centrality(state.D_full)
    )


def _select_rooted_labeled_q_pair(state):
    n = len(state.D)
    tri_i, tri_j = upper_triangle_indices(n)
    if len(tri_i) == 0:
        raise ValueError("No pair is available for rooted labeled NJ.")

    if n == 2:
        return PairChoice(int(tri_i[0]), int(tri_j[0]), score=state.D[0, 1])

    Q = nj_q_matrix(state.D)
    values = Q[tri_i, tri_j]
    best_value = np.min(values)
    best_offsets = np.flatnonzero(values == best_value)
    selected_offset = int(best_offsets[state.rng.randrange(len(best_offsets))])
    return PairChoice(
        int(tri_i[selected_offset]),
        int(tri_j[selected_offset]),
        score=float(best_value),
    )


def _select_rooted_labeled_parent(state, pair):
    global_row_sum = state.context[ROOTED_LABELED_GLOBAL_ROW_SUM_CONTEXT_KEY]
    node_i = state.node_list[pair.i]
    node_j = state.node_list[pair.j]
    score_i = global_row_sum[state.origin_index[node_i]]
    score_j = global_row_sum[state.origin_index[node_j]]

    if score_i < score_j:
        return Orientation(pair.i, pair.j)
    if score_j < score_i:
        return Orientation(pair.j, pair.i)
    if state.rng.random() < 0.5:
        return Orientation(pair.i, pair.j)
    return Orientation(pair.j, pair.i)


def _connect_rooted_labeled_components(state, orientation):
    parent = state.node_list[orientation.parent_idx]
    child = state.node_list[orientation.child_idx]
    state.tree.add_edge(
        parent.node_id,
        child.node_id,
        weight=float(state.D[orientation.parent_idx, orientation.child_idx]),
    )
    return parent


def _remaining_component_node_root(state):
    root = state.node_list[0]
    return state.tree, state.new_nodes, root.node_id


def rooted_labeled_nj(dist_matrix, cells, max_id, seed=7, existing_tree=None):
    """Q-guided rooted baseline whose vertices are the labeled input occurrences.

    Q selects an active component pair. Original-matrix row-sum centrality
    orients the pair, the child component root is attached to the parent root,
    and the parent's representative distance row is retained. No latent or
    copied internal occurrence is created.
    """
    D = _validated_distance_copy(dist_matrix, cells)
    return run_agglomerative_reconstruction(
        D,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=_select_rooted_labeled_q_pair,
        ancestor_selector=_select_rooted_labeled_parent,
        merge_strategy=_connect_rooted_labeled_components,
        root_strategy=_remaining_component_node_root,
        configure_state=_configure_rooted_labeled_state,
    )


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
    return run_agglomerative_reconstruction(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=_select_min_distance_pair,
        ancestor_selector=_select_sum_distance_parent,
    )


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
    return run_agglomerative_reconstruction(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=_legacy_pair_selector(select_pair_func),
        ancestor_selector=_legacy_parent_selector(select_ancestor_func, full_information),
    )


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
