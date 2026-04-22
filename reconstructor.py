import random
from functools import partial
import networkx as nx
import numpy as np
from simulator import Genotype
from reconstructor_ancestor_selection import (
    _is_biologically_plausible_ancestor,
    _final_parent_choice_full_matrix,
    _choose_parent_full_nj,
    _choose_parent_hybrid_inv_centrality,
    _choose_parent_with_plausibility_fallback,
)
from reconstructor_pair_selection import (
    _resolve_pair_ties,
    _select_pair_core,
    _select_pair_full,
    _select_pair_cps,
    _select_pair_hybrid,
    _select_pair_hybrid_inv_centrality,
)
from reconstructor_anticentral import (
    _initial_anticentral_v3_centrality,
    _anticentral_adaptive_v3_score_matrix,
    _best_pair_from_score,
    _ordered_pairs_by_score,
    _copy_parent_internal_node,
    _merge_parent_child,
    _total_deviation_from_baseline,
)
from reconstructor_utils import parse_distance_matrix, visualize_tree_plotly
from reconstructor_biopsy_guided import _extend_biopsies, build_evolution_tree_impl


NJ_REC_TR_ROOT_ID = -1


def neighbor_joining_standard(dist_matrix, cells, max_id, seed=7, existing_tree=None):
    D = dist_matrix.copy()
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}  # Store new nodes for visualization
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
    elif len(id_map) == 1:        # Only one cell in the highest biopsy level, and all connected, ready tree here
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

    This is the best "baseline" NJ reconstruction to compare improvements against.
    """
    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    # node list aligned with D
    node_list = [cells[i] for i in range(len(cells))]

    # add original leaves
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    while len(node_list) > 1:
        n = len(D)

        # ----------------------------
        # 1. Find pair with minimal distance
        # ----------------------------
        min_val = None
        best_pair = (0, 1)

        tri_i, tri_j = np.triu_indices(n, k=1)
        for i, j in zip(tri_i, tri_j):
            d = D[i, j]
            if min_val is None or d < min_val:
                min_val = d
                best_pair = (i, j)

        i, j = best_pair

        # ----------------------------
        # 2. Choose parent by minimal sum to others
        # ----------------------------
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

        # ----------------------------
        # 3. Create internal node = copy of parent
        # ----------------------------
        internal = type(parent_leaf)(
            genome=parent_leaf.genome,
            node_id=next_id,
            cell_id=parent_leaf.cell_id
        )
        next_id += 1

        tree.add_node(internal.node_id,
                      genome=internal.genome,
                      cell_id=internal.cell_id)

        tree.add_edge(internal.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(internal.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))

        new_nodes[internal] = (parent_leaf, child_leaf)

        # ----------------------------
        # 4. Update D and node list
        # ----------------------------
        keep = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep, keep)]

        node_list[parent_idx] = internal
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


# ============================================================
#  UNIVERSAL CORE NJ ENGINE
# ============================================================
def neighbour_joining_core(dist_matrix, cells, max_id, seed=7, existing_tree=None,
                           select_pair_func=_select_pair_full,
                           select_ancestor_func=_choose_parent_full_nj,
                           full_information=False,
                           ):
    rng = random.Random(seed)
    # full (original) matrix for final-step decision
    D_full = dist_matrix.copy().astype(float)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    # keep ordered list aligned with D
    node_list = [cells[i] for i in range(len(cells))]

    # map each initial node to its original index in D_full
    origin_index = {node: idx for idx, node in enumerate(node_list)}

    # add leaf nodes
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    while len(node_list) > 1:
        n = len(D)

        # ---- 1. Select pair (i, j)
        i, j = select_pair_func(D, node_list, rng, minimize=True)

        # special case - finding the root
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

        # reconstructing tree
        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        # create a new internal node (copy of parent) with empty genome
        internal_node = type(parent_leaf)(genome=parent_leaf.genome, node_id=next_id, cell_id=parent_leaf.cell_id)
        next_id += 1
        origin_index[internal_node] = origin_index[parent_leaf]
        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)

        # attach original leaves under the internal node
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))

        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # replace parent_leaf in node_list with internal_node, remove child_leaf
        keep_indices = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep_indices, keep_indices)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    # final remaining node is root
    root = node_list[0]
    return tree, new_nodes, root.cell_id


# ============================================================
#  WRAPPER FUNCTIONS USING PARTIALS
# ============================================================
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
            full_information=full_information
        )
    # give the function a name for analyzer display
    _variant.__name__ = (
        f"neighbor_joining_full_{'full' if full_information else 'partial'}"
    )
    return _variant


def make_nj_full_cps_variant(full_information: bool):
    """
    Factory generating variants of neighbor_joining_full_cps
    with different full_information settings.
    """
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

    # Give the function a unique readable name
    _variant.__name__ = (
        f"neighbor_joining_full_cps_{'full' if full_information else 'partial'}"
    )
    return _variant


def make_nj_hybrid_variant(full_information: bool, alpha=1.0, beta=0.5):
    """
    Factory producing hybrid-NJ variants differing only in their
    full_information setting.
    """
    def _variant(dist_matrix, cells, max_id, seed=7, existing_tree=None):
        select_pair_func = partial(
            _select_pair_hybrid,
            alpha=alpha,
            beta=beta
        )

        return neighbour_joining_core(
            dist_matrix=dist_matrix,
            cells=cells,
            max_id=max_id,
            seed=seed,
            existing_tree=existing_tree,
            select_pair_func=select_pair_func,
            select_ancestor_func=_choose_parent_full_nj,
            full_information=full_information
        )

    # Assign a clear and unique name
    _variant.__name__ = (
        f"neighbor_joining_hybrid_{'full' if full_information else 'partial'}"
    )
    return _variant


def make_nj_hybrid_inv_cent_variant(full_information: bool,
                                    alpha=1.0, beta=0.5, epsilon=1e-6):
    """
    Factory producing hybrid-inverse-centrality NJ variants differing only
    in their full_information setting.
    """
    def _variant(dist_matrix, cells, max_id, seed=7, existing_tree=None):
        select_pair_func = partial(
            _select_pair_hybrid_inv_centrality,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon
        )

        select_ancestor_func = partial(
            _choose_parent_hybrid_inv_centrality,
            epsilon=epsilon
        )

        return neighbour_joining_core(
            dist_matrix=dist_matrix,
            cells=cells,
            max_id=max_id,
            seed=seed,
            existing_tree=existing_tree,
            select_pair_func=select_pair_func,
            select_ancestor_func=select_ancestor_func,
            full_information=full_information
        )

    # Assign stable, analyzer-friendly unique names
    _variant.__name__ = (
        f"neighbor_joining_hybrid_inverse_centrality_"
        f"{'full' if full_information else 'partial'}"
    )
    return _variant


def neighbor_joining_adaptive_centrality(dist_matrix, cells, max_id, alpha=1.0, beta=0.5, epsilon=1e-6, seed=7, existing_tree=None):
    """
    Adaptive neighbor joining with centrality blending.
    Starts with global-distance centrality, transitions to
    local inverse-distance centrality as nodes merge.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise distance matrix.
    cells : list
        List of Genotype-like objects with .node_id, .cell_id, .genome.
    max_id : int
        Maximum node ID so far.
    alpha : float
        Weight for distance term (default 1.0).
    beta : float
        Weight for centrality asymmetry term (default 0.5).
    epsilon : float
        Small constant to prevent division by zero.
    seed : int
        Random seed for tie-breaking.
    existing_tree : nx.DiGraph, optional

    Returns
    -------
    tree : nx.DiGraph
    new_nodes : dict
    root_cell_id
    """
    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    node_list = [cells[i] for i in range(len(cells))]
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1
    N0 = len(D)

    while len(node_list) > 1:
        n = len(D)
        weight = (n - 2) / max(N0 - 2, 1)  # adaptive blending factor

        # --- Compute both centralities ---
        sum_dist = D.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            invD = 1.0 / (D + epsilon)
            np.fill_diagonal(invD, 0.0)
            inv_sum = invD.sum(axis=1)

        # Combine them adaptively: higher C(x) = more central
        centrality = weight * (1.0 / (sum_dist + epsilon)) + (1 - weight) * inv_sum

        # --- Hybrid score ---
        best_score = np.inf
        best_pair = None

        for i in range(n):
            for j in range(i + 1, n):
                score = alpha * D[i, j] - beta * abs(centrality[i] - centrality[j])
                if score < best_score:
                    best_score = score
                    best_pair = (i, j)

        i, j = best_pair
        c_i, c_j = centrality[i], centrality[j]

        # More central node becomes parent
        if c_i > c_j:
            parent_idx, child_idx = i, j
        elif c_i < c_j:
            parent_idx, child_idx = j, i
        else:
            parent_idx, child_idx = (i, j) if rng.random() < 0.5 else (j, i)

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        # Create new internal node (copy of parent)
        internal_node = type(parent_leaf)(genome=parent_leaf.genome, node_id=next_id, cell_id=parent_leaf.cell_id)
        next_id += 1

        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=float(D[parent_idx, child_idx]))
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))

        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # Update distance matrix
        keep_indices = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep_indices, keep_indices)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbor_joining_adaptive_centrality_nonlinear(
    dist_matrix, cells, max_id,
    alpha=1.0, beta=0.5, epsilon=1e-6,
    k=10.0, tau=0.5, seed=7, existing_tree=None
):
    """
    Adaptive neighbor joining with nonlinear sigmoid blending between
    global and inverse-distance centralities.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise distance matrix.
    cells : list
        List of Genotype-like objects with .node_id, .cell_id, .genome.
    max_id : int
        Maximum node ID so far.
    alpha : float
        Weight for distance term.
    beta : float
        Weight for centrality asymmetry term.
    epsilon : float
        Small constant to avoid division by zero.
    k : float
        Sigmoid steepness parameter.
    tau : float
        Sigmoid midpoint parameter (0–1).
    seed : int
        Random seed for reproducibility.
    existing_tree : nx.DiGraph, optional
        Tree to extend.

    Returns
    -------
    tree : nx.DiGraph
    new_nodes : dict
    root_cell_id
    """
    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    node_list = [cells[i] for i in range(len(cells))]
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1
    N0 = len(D)

    while len(node_list) > 1:
        n = len(D)
        # nonlinear adaptive blending
        frac = n / max(N0, 1)
        weight = 1.0 / (1.0 + np.exp(-k * (frac - tau)))  # sigmoid in [0,1]

        # compute global & inverse centralities
        sum_dist = D.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            invD = 1.0 / (D + epsilon)
            np.fill_diagonal(invD, 0.0)
            inv_sum = invD.sum(axis=1)

        # adaptive centrality blending
        centrality = weight * (1.0 / (sum_dist + epsilon)) + (1 - weight) * inv_sum

        # choose best pair: small D + asymmetric centrality
        best_score = np.inf
        best_pair = None
        for i in range(n):
            for j in range(i + 1, n):
                score = alpha * D[i, j] - beta * abs(centrality[i] - centrality[j])
                if score < best_score:
                    best_score = score
                    best_pair = (i, j)

        i, j = best_pair
        c_i, c_j = centrality[i], centrality[j]

        # choose parent-child
        if c_i > c_j:
            parent_idx, child_idx = i, j
        elif c_i < c_j:
            parent_idx, child_idx = j, i
        else:
            parent_idx, child_idx = (i, j) if rng.random() < 0.5 else (j, i)

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        internal_node = type(parent_leaf)(
            genome=parent_leaf.genome, node_id=next_id, cell_id=parent_leaf.cell_id
        )
        next_id += 1

        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=float(D[parent_idx, child_idx]))
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))
        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # update distance matrix
        keep_indices = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep_indices, keep_indices)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbor_joining_adaptive_centrality_reversed(dist_matrix, cells, max_id, seed=7, existing_tree=None):
    """
    Adaptive neighbor-joining with REVERSED weighting between global and local (inverse) centrality.

    Early merges emphasize local cluster cohesion (inverse centrality),
    while later merges emphasize global centrality structure.

    This aims to combine the stability of global centrality and
    the biological realism of local clustering in early reconstruction.
    """

    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}
    node_list = [cells[i] for i in range(len(cells))]
    rng = random.Random(seed)

    # add leaf nodes
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1
    N0 = len(D)
    epsilon = 1e-9

    while len(node_list) > 1:
        n = len(D)
        # progress-based weight: 0 early → 1 late
        weight = 1 - (n / N0)

        # compute global centrality (sum of distances)
        sum_dist = D.sum(axis=1)
        # compute local (inverse-distance) centrality
        inv_sum = np.sum(1.0 / (D + epsilon), axis=1)

        # reversed adaptive weighting:
        # early: local dominates, later: global dominates
        centrality = (1 - weight) * inv_sum + weight * (1.0 / (sum_dist + epsilon))

        # composite score for each pair (low distance, high centrality asymmetry)
        best_score = np.inf
        best_pair = None
        for i in range(n):
            for j in range(i + 1, n):
                dist_ij = D[i, j]
                asym = abs(centrality[i] - centrality[j])
                # heuristic combination: balance closeness and asymmetry
                score = dist_ij * (1.0 - 0.25 * asym)  # asym helps prioritize central vs peripheral
                if score < best_score:
                    best_score = score
                    best_pair = (i, j)

        i, j = best_pair

        # decide parent-child based on centrality (higher = more central)
        if centrality[i] >= centrality[j]:
            parent_idx, child_idx = i, j
        else:
            parent_idx, child_idx = j, i

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        # create internal node (copy of parent)
        internal_node = type(parent_leaf)(
            genome=parent_leaf.genome, node_id=next_id, cell_id=parent_leaf.cell_id
        )
        next_id += 1

        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))
        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # update distance matrix (replace parent, remove child)
        keep_indices = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep_indices, keep_indices)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbor_joining_hybrid_opt(
    dist_matrix,
    cells,
    max_id,
    alpha=1.0,
    beta=0.7,
    gamma=0.2,
    epsilon=1e-9,
    k=10.0,
    tau=0.5,
    reverse_centrality=False,
    use_q_as_secondary=True,
    seed=7,
    existing_tree=None
):
    """
    Hybrid-OPT neighbor joining heuristic.
    - Primary score: alpha * D[i,j] - beta * |C[i] - C[j]|
      where C is adaptive blended centrality (nonlinear sigmoid).
    - Optional secondary NJ-style Q influence: - gamma * Q[i,j] (lower Q is better).
    - Parent chosen as the more-central node (higher C).
    - Deterministic randomness via seed for tie-breaking.

    Parameters
    ----------
    dist_matrix : np.ndarray (n x n)
    cells : list of objects with .node_id, .cell_id, .genome
    max_id : int
    alpha : float   (weight for distance; keep ~1.0)
    beta : float    (weight for centrality asymmetry; keep ~0.5-1.0)
    gamma : float   (weight for Q-matrix tie-break; 0 disables)
    epsilon : float small constant to avoid /0
    k : float       sigmoid steepness (>=0, larger -> sharper transition)
    tau : float     sigmoid midpoint in fraction n/N0 (0..1)
    reverse_centrality : bool
        If False: early => inverse-distance, late => global (like previous nonlinear)
        If True: reversed behavior (early global, late inverse)
    use_q_as_secondary : bool  include Q term if True
    seed : int
    existing_tree : nx.DiGraph or None

    Returns
    -------
    tree : nx.DiGraph
    new_nodes : dict (mapping internal node object -> (child_a, child_b))
    root_cell_id : ID or cell.cell_id
    """

    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    node_list = [cells[i] for i in range(len(cells))]
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1
    N0 = len(D)

    def compute_Q_from_D(Dmat):
        """Standard NJ Q-matrix for current D matrix."""
        n = Dmat.shape[0]
        total = Dmat.sum(axis=1)
        Q = np.full((n, n), np.inf, dtype=float)
        if n <= 2:
            return Q
        factor = (n - 2)
        for i in range(n):
            for j in range(i + 1, n):
                Qval = factor * Dmat[i, j] - total[i] - total[j]
                Q[i, j] = Qval
                Q[j, i] = Qval
        return Q

    while len(node_list) > 1:
        n = len(D)
        frac = n / max(N0, 1)
        # sigmoid blending weight in [0,1]
        w = 1.0 / (1.0 + np.exp(-k * (frac - tau))) if k > 0 else frac

        # compute global centrality (smaller sum = more central) -> convert to "bigger is more central"
        sum_dist = D.sum(axis=1)
        global_c = 1.0 / (sum_dist + epsilon)   # larger => more central

        # inverse-dist centrality: local dense clusters => larger is more central
        with np.errstate(divide='ignore', invalid='ignore'):
            invD = 1.0 / (D + epsilon)
            np.fill_diagonal(invD, 0.0)
            inv_c = invD.sum(axis=1)

        # choose blending direction
        if reverse_centrality:
            # early: global, late: inverse
            C = (1 - w) * global_c + w * inv_c
        else:
            # early: inverse (local), late: global
            C = w * global_c + (1 - w) * inv_c

        # precompute Q if needed
        Q = compute_Q_from_D(D) if use_q_as_secondary and D.shape[0] > 2 else None

        # find best pair by composite score
        best_score = np.inf
        best_candidates = []  # store candidates with equal best_score for tie-breaking
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = D[i, j]
                asym = abs(C[i] - C[j])
                # primary hybrid score
                score = alpha * d_ij - beta * asym
                # secondary NJ-like influence (smaller Q is better) -> subtract gamma*Q so that smaller Q reduces score
                if Q is not None:
                    qij = Q[i, j]
                    # normalize Q to similar scale? optional - for now treat raw with gamma small
                    score -= gamma * qij
                if score < best_score - 1e-12:
                    best_score = score
                    best_candidates = [(i, j, d_ij, asym, (Q[i,j] if Q is not None else None))]
                elif abs(score - best_score) <= 1e-12:
                    best_candidates.append((i, j, d_ij, asym, (Q[i,j] if Q is not None else None)))

        # if multiple candidates, tie-break deterministically:
        if len(best_candidates) > 1:
            # prefer smallest raw distance
            best_candidates.sort(key=lambda x: (x[2], -x[3], x[4] if x[4] is not None else 0.0))
            # (d_ij asc, asym desc, Q asc)
        i, j, _, _, _ = best_candidates[0]

        # parent is the more-central (larger C)
        if C[i] > C[j]:
            parent_idx, child_idx = i, j
        elif C[j] > C[i]:
            parent_idx, child_idx = j, i
        else:
            # if equal centrality, use Q if available
            if Q is not None:
                if Q[i, j] < Q[j, i]:  # symmetric but keep structure
                    parent_idx, child_idx = i, j
                elif Q[j, i] < Q[i, j]:
                    parent_idx, child_idx = j, i
                else:
                    # final random tie-break but seeded
                    parent_idx, child_idx = (i, j) if rng.random() < 0.5 else (j, i)
            else:
                parent_idx, child_idx = (i, j) if rng.random() < 0.5 else (j, i)

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        # create new internal node (copy of parent)
        internal_node = type(parent_leaf)(
            genome=parent_leaf.genome,
            node_id=next_id,
            cell_id=parent_leaf.cell_id
        )
        next_id += 1
        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)

        # attach original leaves under this internal node
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))

        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # update distance matrix and node list: keep parent, remove child
        keep_indices = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep_indices, keep_indices)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbor_joining_hybrid_opt_adaptive(dist_matrix, cells, max_id, seed=7, existing_tree=None,
                                         alpha=1.0, beta=0.8, gamma=0.1, epsilon=1e-6):
    """
    Optimized hybrid neighbor-joining with adaptive alpha/beta weighting.

    Heuristic:
      - Select pairs that are both close (small D[x,y]) and asymmetric in centrality (|C[x]-C[y]| large).
      - Centrality C[x] = sum_i 1/(D[x,i] + ε)
      - Alpha/beta are dynamically adapted depending on heterogeneity of D:
            more heterogeneous -> emphasize distance (alpha↑)
            more uniform -> emphasize centrality (beta↑)
    """
    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    # keep ordered list aligned with D
    node_list = [cells[i] for i in range(len(cells))]

    # add leaf nodes
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    while len(node_list) > 1:
        n = len(D)

        # --- adaptive weighting based on matrix heterogeneity ---
        dist_range = D.max() - D.min()
        if dist_range > 1e-9:
            heterogeneity = np.std(D) / dist_range
        else:
            heterogeneity = 0.0
        alpha_eff = alpha * (1 + 0.5 * heterogeneity)
        beta_eff = beta  * (1 - 0.5 * heterogeneity)

        # --- compute inverse-distance centralities ---
        inv_D = 1.0 / (D + epsilon)
        np.fill_diagonal(inv_D, 0)
        centrality = inv_D.sum(axis=1)

        # --- hybrid score matrix ---
        score = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = D[i, j]
                asym = abs(centrality[i] - centrality[j])
                score[i, j] = alpha_eff * d_ij - beta_eff * asym

        # --- choose best pair (lowest score) ---
        i, j = divmod(np.argmin(score), n)
        if j < i:
            i, j = j, i

        # decide parent/child by centrality
        if centrality[i] >= centrality[j]:
            parent_idx, child_idx = i, j
        else:
            parent_idx, child_idx = j, i

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        # create internal node (copy of parent)
        internal_node = type(parent_leaf)(
            genome=parent_leaf.genome, node_id=next_id, cell_id=parent_leaf.cell_id
        )
        next_id += 1
        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)

        # connect
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))
        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # --- update matrix ---
        keep_indices = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep_indices, keep_indices)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    # final remaining node is root
    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbor_joining_hybrid_opt_v2(dist_matrix, cells, max_id, seed=7, existing_tree=None,
                                   alpha=1.0, beta=1.0, lam=1.0, epsilon=1e-6):
    """
    Enhanced hybrid NJ with combined direct + inverse centrality.
    Builds on neighbor_joining_hybrid_opt.

    c_dir(x) = sum(D[x,i])
    c_inv(x) = sum(1/(D[x,i]+eps))
    c_mix(x) = c_inv(x) / (c_dir(x)+eps)^lam
    Score = alpha*D[x,y] - beta*|c_mix(x) - c_mix(y)|
    """

    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    node_list = [cells[i] for i in range(len(cells))]

    # add leaf nodes
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    while len(node_list) > 1:
        n = len(D)

        # --- centrality computations ---
        c_dir = D.sum(axis=1)
        inv_D = 1.0 / (D + epsilon)
        np.fill_diagonal(inv_D, 0.0)
        c_inv = inv_D.sum(axis=1)

        # --- mixed centrality ---
        c_mix = c_inv / np.power(c_dir + epsilon, lam)

        # --- hybrid score matrix ---
        score = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = D[i, j]
                asym = abs(c_mix[i] - c_mix[j])
                score[i, j] = alpha * d_ij - beta * asym

        # --- choose lowest score pair ---
        i, j = divmod(np.argmin(score), n)
        if j < i:
            i, j = j, i

        # parent = node with higher c_mix
        if c_mix[i] >= c_mix[j]:
            parent_idx, child_idx = i, j
        else:
            parent_idx, child_idx = j, i

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        internal_node = type(parent_leaf)(
            genome=parent_leaf.genome, node_id=next_id, cell_id=parent_leaf.cell_id
        )
        next_id += 1
        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))

        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # --- update matrix ---
        keep = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep, keep)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbor_joining_hybrid_opt_refined(dist_matrix, cells, max_id, seed=7, existing_tree=None,
                                        alpha=1.0, beta=1.0, gamma=1.0):
    """
    Refined hybrid NJ variant:
    - keeps hybrid rule
    - normalizes distances and centralities
    - uses non-linear asymmetry penalty (|Δc|**gamma)
    """
    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    node_list = [cells[i] for i in range(len(cells))]
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    while len(node_list) > 1:
        n = len(D)

        # --- centrality ---
        c = D.sum(axis=1)
        mean_D = D[np.triu_indices(n, 1)].mean()
        mean_c = np.mean(c)

        score = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = D[i, j] / mean_D
                asym = abs(c[i] - c[j]) / mean_c
                score[i, j] = alpha * d_ij - beta * (asym ** gamma)

        i, j = divmod(np.argmin(score), n)
        if j < i: i, j = j, i

        # pick parent as more central node
        parent_idx, child_idx = (i, j) if c[i] <= c[j] else (j, i)

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        internal_node = type(parent_leaf)(
            genome=parent_leaf.genome,
            node_id=next_id,
            cell_id=parent_leaf.cell_id,
        )
        next_id += 1
        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))

        # update matrix
        keep = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep, keep)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, {}, root.cell_id


def neighbor_joining_hybrid_anticentral_opt(
    dist_matrix,
    cells,
    max_id,
    seed=7,
    existing_tree=None,
    alpha=1.0,
    beta=1.0,
    lam=1.0,
    epsilon=1e-6
):
    """
    Anticentral hybrid neighbor joining.
    Opposite of the hybrid_opt heuristic — favors merging anti-central pairs
    (large |c_mix(x) - c_mix(y)|, but also high distance).

    Score = alpha*D[x,y] + beta*|c_mix(x) - c_mix(y)|
    (note the + instead of - in hybrid_opt)
    """

    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    node_list = [cells[i] for i in range(len(cells))]

    # add all leaf nodes
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    while len(node_list) > 1:
        n = len(D)

        # --- compute centralities ---
        c_dir = D.sum(axis=1)
        inv_D = 1.0 / (D + epsilon)
        np.fill_diagonal(inv_D, 0.0)
        c_inv = inv_D.sum(axis=1)
        c_mix = c_inv / np.power(c_dir + epsilon, lam)

        # --- anti-central hybrid score matrix ---
        score = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = D[i, j]
                if not np.isfinite(d_ij):  # skip invalid pairs
                    continue
                asym = abs(c_mix[i] - c_mix[j])
                score[i, j] = alpha * d_ij + beta * asym

        # --- choose the *highest* score pair (anti-central) ---
        i, j = divmod(np.argmax(score), n)
        if j < i:
            i, j = j, i

        # guard: break if score invalid
        if not np.isfinite(score[i, j]) or i == j:
            break

        # --- parent = less central node (lower c_mix) ---
        if c_mix[i] <= c_mix[j]:
            parent_idx, child_idx = i, j
        else:
            parent_idx, child_idx = j, i

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        internal_node = type(parent_leaf)(
            genome=parent_leaf.genome,
            node_id=next_id,
            cell_id=parent_leaf.cell_id,
        )
        next_id += 1

        tree.add_node(
            internal_node.node_id,
            genome=internal_node.genome,
            cell_id=internal_node.cell_id,
        )
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(
            internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx])
        )

        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # --- update matrix ---
        keep = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep, keep)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    # --- unify if multiple roots remain ---
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    if len(roots) > 1:
        super_root = next_id
        tree.add_node(super_root)
        for r in roots:
            tree.add_edge(super_root, r, weight=0.0)
        root_node_id = super_root
    else:
        root_node_id = roots[0]

    root_cell_id = tree.nodes[root_node_id].get("cell_id", None)
    return tree, new_nodes, root_cell_id


def neighbor_joining_hybrid_anticentral_adaptive_v2(
    dist_matrix,
    cells,
    max_id,
    seed=7,
    existing_tree=None,
    alpha=1.0,
    beta=1.0,
    gamma=0.2,
    lam=1.0,
    epsilon=1e-6
):
    """
    Hybrid-Anticentral Adaptive NJ v2
    Combines distance, centrality asymmetry, and inverse-distance penalty
    to improve 2-F1 reconstruction accuracy.

    Score = α*D[i,j] + β*|c_mix(i)-c_mix(j)| - γ/(D[i,j]+ε)
    Adaptive scaling adjusts α,β,γ with iteration depth.
    """

    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    node_list = [cells[i] for i in range(len(cells))]

    # add all leaves
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1
    iteration = 0

    while len(node_list) > 1:
        iteration += 1
        n = len(D)
        meanD = np.nanmean(D[np.isfinite(D)])
        # adaptive scaling (start exploratory, end focused)
        a = alpha * (1 + 0.2 * np.tanh(3 * (1 - len(node_list) / len(cells))))
        b = beta * (1 + 0.3 * np.tanh(2 * (len(node_list) / len(cells))))
        g = gamma * (0.5 + 0.5 * np.tanh(3 * len(node_list) / len(cells)))

        # --- compute centralities ---
        c_dir = D.sum(axis=1)
        inv_D = 1.0 / (D + epsilon)
        np.fill_diagonal(inv_D, 0.0)
        c_inv = inv_D.sum(axis=1)
        c_mix = c_inv / np.power(c_dir + epsilon, lam)

        # --- hybrid adaptive score ---
        score = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = D[i, j]
                if not np.isfinite(d_ij):
                    continue
                asym = abs(c_mix[i] - c_mix[j])
                score[i, j] = a * d_ij + b * asym - g / (d_ij + epsilon)

        # choose highest score (anticentral-like)
        i, j = divmod(np.argmax(score), n)
        if j < i:
            i, j = j, i
        if not np.isfinite(score[i, j]) or i == j:
            break

        # choose parent: less central node (lower c_mix)
        if c_mix[i] <= c_mix[j]:
            parent_idx, child_idx = i, j
        else:
            parent_idx, child_idx = j, i

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        internal_node = type(parent_leaf)(
            genome=parent_leaf.genome,
            node_id=next_id,
            cell_id=parent_leaf.cell_id,
        )
        next_id += 1

        tree.add_node(
            internal_node.node_id,
            genome=internal_node.genome,
            cell_id=internal_node.cell_id,
        )
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=0.0)
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))

        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # update matrix
        keep = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep, keep)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    # unify multiple roots
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    if len(roots) > 1:
        super_root = next_id
        tree.add_node(super_root)
        for r in roots:
            tree.add_edge(super_root, r, weight=0.0)
        root_id = super_root
    else:
        root_id = roots[0]

    root_cell_id = tree.nodes[root_id].get("cell_id", None)
    return tree, new_nodes, root_cell_id


def neighbor_joining_hybrid_anticentral_adaptive_v3(dist_matrix, cells, max_id,
                                                    seed=None, existing_tree=None,
                                                    alpha=1.0, beta=1.0, gamma=0.5):
    """
    Hybrid anticentral neighbor joining with adaptive centrality weighting (v3).
    Fully compatible with the standard NJ call pattern.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Symmetric distance matrix.
    cells : list
        List of cell/genotype objects (must have node_id, genome, cell_id).
    max_id : int
        Current max node ID.
    seed : int, optional
        Random seed for tie-breaking.
    existing_tree : nx.DiGraph, optional
        Existing tree to append nodes to.
    alpha, beta, gamma : float
        Weight parameters controlling distance, anticentrality, and adaptivity.

    Returns
    -------
    tree : nx.DiGraph
    new_nodes : dict
        Mapping of internal nodes to their children.
    root_cell_id : any
    """

    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    # Initialize leaves
    node_list = [cells[i] for i in range(len(cells))]
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    c = _initial_anticentral_v3_centrality(D)

    while len(node_list) > 1:
        score = _anticentral_adaptive_v3_score_matrix(D, c, rng, alpha, beta, gamma)
        i_best, j_best = _best_pair_from_score(score)

        a = node_list[i_best]
        b = node_list[j_best]

        internal_node, next_id = _copy_parent_internal_node(
            tree,
            new_nodes,
            a,
            b,
            next_id,
            D[i_best, j_best],
        )

        # Update distance and centrality
        D, c = _merge_parent_child(D, node_list, c, i_best, j_best, internal_node)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbor_joining_hybrid_anticentral_adaptive_v3_plausible(
    dist_matrix,
    cells,
    max_id,
    seed=None,
    existing_tree=None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    enforce_plausibility: bool = True,
):
    """
    Hybrid anticentral neighbor joining with adaptive centrality weighting (v3),
    extended with *directional biological plausibility*.

    Differences vs neighbor_joining_hybrid_anticentral_adaptive_v3:
    - The same pair (a, b) is always chosen (same score logic).
    - BUT we decide whether the ancestor template should be 'a' or 'b'
      using _is_biologically_plausible_ancestor, when possible.

    Logic within a chosen pair (a, b):
      can_a = ancestor(a -> b) plausible?
      can_b = ancestor(b -> a) plausible?

      - if can_a and not can_b: internal node copies 'a' (original behaviour)
      - if can_b and not can_a: internal node copies 'b' (we swap a <-> b)
      - if can_a and can_b: keep original anticentral design (no swap)
      - if neither: keep original (no swap, plausibility ignored for this pair)

    This way plausibility refines ancestor direction, but never spoils
    the anticentral adaptive scoring or pair ordering.
    """

    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    # Initialize leaves
    node_list = [cells[i] for i in range(len(cells))]
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    c = _initial_anticentral_v3_centrality(D)

    while len(node_list) > 1:
        score = _anticentral_adaptive_v3_score_matrix(D, c, rng, alpha, beta, gamma)
        i_best, j_best = _best_pair_from_score(score)

        a = node_list[i_best]
        b = node_list[j_best]

        # --------------------------------------------------
        #  BIOLOGICAL PLAUSIBILITY: choose ancestor template
        # --------------------------------------------------
        if enforce_plausibility:
            can_a_parent_b = _is_biologically_plausible_ancestor(a, b)
            can_b_parent_a = _is_biologically_plausible_ancestor(b, a)

            if can_b_parent_a and not can_a_parent_b:
                # biologically only b can be ancestor → swap roles
                a, b = b, a
                i_best, j_best = j_best, i_best
                # we do NOT change which pair is merged, only which one
                # is treated as ancestor template

            # if both directions plausible: keep anticentral choice (no swap)
            # if neither plausible: keep original (no swap, soft failure)

        # Now 'a' is the chosen ancestor template for this pair
        parent_leaf = a
        child_leaf = b
        parent_idx = i_best
        child_idx = j_best

        internal_node, next_id = _copy_parent_internal_node(
            tree,
            new_nodes,
            parent_leaf,
            child_leaf,
            next_id,
            D[parent_idx, child_idx],
        )

        # Update distance matrix and centrality
        D, c = _merge_parent_child(D, node_list, c, parent_idx, child_idx, internal_node)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible(
    dist_matrix,
    cells,
    max_id,
    seed=None,
    existing_tree=None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
):
    """
    Anticentral Adaptive NJ (v3) with biological plausibility *pair skipping*.

    CORE IDEA:
    After computing scores for all pairs (i, j):
        -> iterate pairs in increasing score order
        -> pick the first pair where at least one direction
           is biologically plausible (a->b or b->a).
        -> if both directions plausible: keep anticentral orientation
        -> if exactly one direction plausible: force that parent
        -> if none are plausible for ANY pair: fallback to best pair

    This enforces biological realism without destroying the NJ-like ordering.
    """

    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    # Init leaves
    node_list = list(cells)
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    c = _initial_anticentral_v3_centrality(D)

    while len(node_list) > 1:
        score = _anticentral_adaptive_v3_score_matrix(D, c, rng, alpha, beta, gamma)

        # --- sorted list of pairs by score ---
        pair_order = _ordered_pairs_by_score(score)

        picked_pair = None
        picked_parent_is_i = True  # orientation

        # --- try pairs in order until one is plausible ---
        for (i, j) in pair_order:
            a = node_list[i]
            b = node_list[j]
            can_ai_bj = _is_biologically_plausible_ancestor(a, b)
            can_bj_ai = _is_biologically_plausible_ancestor(b, a)

            if can_ai_bj or can_bj_ai:        # we accept this pair
                picked_pair = (i, j)
                if can_ai_bj and not can_bj_ai:
                    picked_parent_is_i = True
                elif can_bj_ai and not can_ai_bj:
                    picked_parent_is_i = False
                else:
                    # both plausible → keep anticentral's default “more anticentral parent”
                    picked_parent_is_i = (c[i] > c[j])
                break

        # If no plausible pair found → fallback to best pair
        if picked_pair is None:
            i_best, j_best = pair_order[0]
            picked_parent_is_i = (c[i_best] > c[j_best])
            picked_pair = (i_best, j_best)

        i_best, j_best = picked_pair

        # assign actual leaves
        parent_idx = i_best if picked_parent_is_i else j_best
        child_idx  = j_best if picked_parent_is_i else i_best

        parent_leaf = node_list[parent_idx]
        child_leaf  = node_list[child_idx]

        internal_node, next_id = _copy_parent_internal_node(
            tree,
            new_nodes,
            parent_leaf,
            child_leaf,
            next_id,
            D[parent_idx, child_idx],
        )

        # ---- Update D and centrality ----
        D, c = _merge_parent_child(D, node_list, c, parent_idx, child_idx, internal_node)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony(
    dist_matrix,
    cells,
    max_id,
    seed=None,
    existing_tree=None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    baseline_cn: int = 2,
):
    """
    Anticentral Adaptive NJ (v3) with biological plausibility + parsimony bias.

    Differences vs plain v3:
    -------------------------
    - Pair (i, j) is still chosen purely by v3 anticentral score.
    - Biological plausibility is enforced on direction:
        * if only x->y plausible: x is parent
        * if only y->x plausible: y is parent
        * if both directions plausible:
             - parent is genome closer to baseline CN (parsimony)
             - if still tied, fall back to anticentral rule
        * if neither direction plausible:
             - fall back to anticentral direction (very rare / late merges)

    This keeps the strong v3 pair selection but orients edges in a more
    biologically realistic way (ancestor is less aberrant).
    """

    rng = random.Random(seed)
    D = dist_matrix.copy().astype(float)
    tree = existing_tree or nx.DiGraph()
    new_nodes = {}

    # Initialize leaves
    node_list = [cells[i] for i in range(len(cells))]
    for node in node_list:
        tree.add_node(node.node_id, genome=node.genome, cell_id=node.cell_id)

    next_id = max_id + 1

    c = _initial_anticentral_v3_centrality(D)

    while len(node_list) > 1:
        score = _anticentral_adaptive_v3_score_matrix(D, c, rng, alpha, beta, gamma)
        i_best, j_best = _best_pair_from_score(score)

        a = node_list[i_best]
        b = node_list[j_best]

        # --- biological plausibility of directions ---
        can_a_parent_b = _is_biologically_plausible_ancestor(a, b)
        can_b_parent_a = _is_biologically_plausible_ancestor(b, a)

        if can_a_parent_b and not can_b_parent_a:
            # only a -> b allowed
            parent_idx, child_idx = i_best, j_best

        elif can_b_parent_a and not can_a_parent_b:
            # only b -> a allowed
            parent_idx, child_idx = j_best, i_best

        elif can_a_parent_b and can_b_parent_a:
            # both directions allowed → apply parsimony rule
            dev_a = _total_deviation_from_baseline(a.genome, baseline_cn)
            dev_b = _total_deviation_from_baseline(b.genome, baseline_cn)

            if dev_a < dev_b:
                parent_idx, child_idx = i_best, j_best
            elif dev_b < dev_a:
                parent_idx, child_idx = j_best, i_best
            else:
                # same deviation → fall back to anticentral preference
                if c[i_best] > c[j_best]:
                    parent_idx, child_idx = i_best, j_best
                elif c[j_best] > c[i_best]:
                    parent_idx, child_idx = j_best, i_best
                else:
                    # total tie → random but seeded
                    parent_idx, child_idx = (
                        (i_best, j_best)
                        if rng.random() < 0.5
                        else (j_best, i_best)
                    )

        else:
            # Neither direction is plausible (very rare).
            # To avoid breaking good v3 behavior, just keep anticentral rule.
            if c[i_best] > c[j_best]:
                parent_idx, child_idx = i_best, j_best
            elif c[j_best] > c[i_best]:
                parent_idx, child_idx = j_best, i_best
            else:
                parent_idx, child_idx = (
                    (i_best, j_best)
                    if rng.random() < 0.5
                    else (j_best, i_best)
                )

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        internal_node, next_id = _copy_parent_internal_node(
            tree,
            new_nodes,
            parent_leaf,
            child_leaf,
            next_id,
            D[parent_idx, child_idx],
        )

        # update D and centrality
        D, c = _merge_parent_child(D, node_list, c, parent_idx, child_idx, internal_node)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def build_evolution_tree(cell_lists, seed=7, dist_matrix_path=None, r=2, only_nj=False, inids=None, indm=None,
                         neighbor_joining=neighbor_joining_standard):
    return build_evolution_tree_impl(
        cell_lists,
        seed=seed,
        dist_matrix_path=dist_matrix_path,
        r=r,
        only_nj=only_nj,
        inids=inids,
        indm=indm,
        neighbor_joining=neighbor_joining,
    )


if __name__ == '__main__':
    # CNPs here do not influence distances only for checking if descendant has x>0 where ancestor has x=0
    cell_lists = [
        [Genotype([2, 0, 1], 1), Genotype([1, 1, 1], 2)],
        [Genotype([2, 1, 1], 3), Genotype([1, 2, 0], 4)]
    ]
    cell_lists1 = [
        [Genotype([2, 2, 1], 1), Genotype([1, 1, 1], 2)],
        [Genotype([2, 1, 1], 3), Genotype([1, 2, 0], 4)]
    ]

    # # 3->2
    # tree, a, b = build_evolution_tree(cell_lists, "test/data/dm/distance_matrix.txt", r=2)
    # visualize_tree_plotly(tree, a)
    # # 3->1
    # tree, a, b = build_evolution_tree(cell_lists1, "test/data/dm/distance_matrix.txt", r=2)
    # visualize_tree_plotly(tree, a)
    # 3->2, 4->2
    tree, a, _ = build_evolution_tree(cell_lists, "test/data/dm/distance_matrix.txt", r=4)
    visualize_tree_plotly(tree, a)
