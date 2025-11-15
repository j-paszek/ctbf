import itertools
import random

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from simulator import Genotype


NJ_REC_TR_ROOT_ID = -1


def parse_distance_matrix(path):
    with open(path) as f:
        n = int(f.readline())
        ids = []
        matrix = []
        for _ in range(n):
            parts = f.readline().strip().split()
            ids.append(int(parts[0]))
            matrix.append([float(x) for x in parts[1:]])
    return ids, np.array(matrix)


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


def _is_biologically_plausible_pair(x, y):
    """
    Biological plausibility of merging two cells x and y.

    Currently uses an "appearance constraint" similar to the
    biopsy-guided reconstruction, i.e. disallow merges where one
    genome has copy-number 0 and the other has a positive copy
    number at the same position.
    """
    return not (np.any((x.genome == 0) & (y.genome > 0)) or np.any((y.genome == 0) & (x.genome > 0)))


def _resolve_pair_ties(candidate_pairs, node_list, rng, return_first=False):
    """
    Resolve ties between equally scoring candidate pairs.

    Parameters
    ----------
    candidate_pairs : list of (i, j) index tuples
        All pairs that have the same (best) score according to a heuristic.
    node_list : list
        Current ordered list of nodes corresponding to rows/cols of D.
    rng : random.Random
        Random generator for deterministic tie-breaking.

    Returns
    -------
    (i, j) : tuple of int
        Selected pair of indices.
    """
    if not candidate_pairs:
        raise ValueError("No candidate pairs provided to _resolve_pair_ties.")

    if return_first:
        return candidate_pairs[0]

    # Optional biological plausibility filter
    plausible = [
        (i, j) for (i, j) in candidate_pairs
        if _is_biologically_plausible_pair(node_list[i], node_list[j])
    ]
    if plausible:
        candidate_pairs = plausible

    # If only one candidate remains, return it deterministically
    if len(candidate_pairs) == 1:
        return candidate_pairs[0]

    # Deterministic random tie-breaker
    return rng.choice(candidate_pairs)


def _select_pair_core(D, node_list, rng, pair_score_func, minimize=True, apply_plausibility=True):
    """
    Universal pair-selection core used by all NJ variants.
    Applies biological plausibility and tie-breaking.
    """
    n = len(D)
    best_score = None
    candidate_pairs = []

    tri_i, tri_j = np.triu_indices(n, k=1)
    for i, j in zip(tri_i, tri_j):
        score = pair_score_func(i, j)

        if best_score is None:
            best_score = score
            candidate_pairs = [(i, j)]
        else:
            if minimize:
                if score < best_score:
                    best_score = score
                    candidate_pairs = [(i, j)]
                elif score == best_score:
                    candidate_pairs.append((i, j))
            else:
                if score > best_score:
                    best_score = score
                    candidate_pairs = [(i, j)]
                elif score == best_score:
                    candidate_pairs.append((i, j))

    # APPLY BIOLOGICAL PLAUSIBILITY HERE FOR ALL VARIANTS
    if apply_plausibility:
        i, j = _resolve_pair_ties(candidate_pairs, node_list, rng)
        return i, j
    else:
        # EXACT BEHAVIOR OF ORIGINAL NJ-FULL:
        # pick the FIRST candidate pair (NumPy order)
        return candidate_pairs[0]


def _choose_parent_by_centrality(centrality, i, j, rng, larger_is_more_central):
    """
    Decide which index becomes the parent based on centrality values.

    Parameters
    ----------
    centrality : array-like
        Centrality score per node; semantics depend on the caller.
    i, j : int
        Indices of the two candidate nodes.
    rng : random.Random
        Used for deterministic tie-breaking.
    larger_is_more_central : bool
        If True, node with larger centrality is considered more central.
        If False, node with smaller centrality is considered more central.

    Returns
    -------
    (parent_idx, child_idx) : tuple of int
    """
    c_i = centrality[i]
    c_j = centrality[j]

    if larger_is_more_central:
        if c_i > c_j:
            return i, j
        if c_j > c_i:
            return j, i
    else:
        if c_i < c_j:
            return i, j
        if c_j < c_i:
            return j, i

    # Tie: random but seeded
    return (i, j) if rng.random() < 0.5 else (j, i)


def _choose_parent_full_nj(D, i, j, rng, larger_is_more_central):
    centrality = D.sum(axis=1)
    return _choose_parent_by_centrality(
        centrality, i, j, rng, larger_is_more_central=larger_is_more_central
    )


def neighbour_joining_core(dist_matrix, cells, max_id, seed=7, existing_tree=None,
                           select_pair_func=_select_pair_core,
                           select_ancestor_func=_choose_parent_full_nj
                           ):
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

        # core differences
        i, j = select_pair_func(D, node_list, rng, minimize=True)
        parent_idx, child_idx = select_ancestor_func(D, i, j, rng, larger_is_more_central=False)


        # reconstructing tree
        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        # create a new internal node (copy of parent) with empty genome
        internal_node = type(parent_leaf)(genome=parent_leaf.genome, node_id=next_id, cell_id=parent_leaf.cell_id)
        next_id += 1
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


def _select_pair_full(D, node_list, rng, minimize=True):
    return _select_pair_core(
        D, node_list, rng,
        pair_score_func=lambda i, j: D[i, j],   # score = distance
        minimize=True,
        apply_plausibility=False
    )


def neighbor_joining_full(dist_matrix, cells, max_id, seed=7, existing_tree=None):
    """
    Deterministic, parent-retaining neighbor-joining replacement.
    Each merge creates a new internal node (copy of parent) with empty genome.
    Original leaves are preserved.
    Returns: tree, new_nodes dict, root cell_id
    """
    return neighbour_joining_core(dist_matrix, cells, max_id, seed=7, existing_tree=None,
                           select_pair_func=_select_pair_full,
                           select_ancestor_func=_choose_parent_full_nj)


def _select_pair_cps(D, node_list, rng, minimize=True):
    centrality = D.sum(axis=1)

    def pair_score_func(i, j):
        c_i, c_j = centrality[i], centrality[j]
        return (D[i, j], min(c_i, c_j), -max(c_i, c_j))  # CPS tuple

    return _select_pair_core(
        D, node_list, rng,
        pair_score_func=pair_score_func,
        minimize=True,
        apply_plausibility=True
    )


def neighbor_joining_full_cps(dist_matrix, cells, max_id, seed=7, existing_tree=None):
    """
    Centrality-guided neighbor joining heuristic.
    1. Find all pairs with minimal distance.
    2. For each, compute centrality = sum of distances to all other cells.
    3. Pick pair (x, y) such that:
         - the smaller centrality is minimal (most central ancestor)
         - if tie, the larger centrality is maximal (most peripheral child)
    4. Parent = the node with smaller centrality (more central one).
    """
    return neighbour_joining_core(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        select_pair_func=_select_pair_cps,
        select_ancestor_func=_choose_parent_full_nj,  # same parent rule as full NJ
    )


def neighbor_joining_hybrid(dist_matrix, cells, max_id, alpha=1.0, beta=0.5, seed=7, existing_tree=None):
    """
    Hybrid neighbor joining: prefers pairs that are both close (small D[x,y])
    and asymmetric in centrality (|c(x) - c(y)| large).

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise distance matrix between cells.
    cells : list
        List of Genotype-like cell objects (must have .node_id, .cell_id, .genome).
    max_id : int
        Max node ID so far; new nodes start from max_id + 1.
    alpha : float
        Weight for distance term (default 1.0).
    beta : float
        Weight for asymmetry term (default 0.5).
    seed : int
        Random seed for tie-breaking.
    existing_tree : nx.DiGraph, optional
        If provided, new nodes will be added to this tree.

    Returns
    -------
    tree : nx.DiGraph
    new_nodes : dict
    root_cell_id : any
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
        centrality = D.sum(axis=1)

        # Compute hybrid score for each pair
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

        # Determine parent-child relationship
        if c_i < c_j:
            parent_idx, child_idx = i, j
        elif c_i > c_j:
            parent_idx, child_idx = j, i
        else:
            parent_idx, child_idx = (i, j) if rng.random() < 0.5 else (j, i)

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        internal_node = type(parent_leaf)(genome=parent_leaf.genome, node_id=next_id, cell_id=parent_leaf.cell_id)
        next_id += 1

        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=D[parent_idx, child_idx])
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=D[parent_idx, child_idx])

        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # Update matrix
        keep_indices = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep_indices, keep_indices)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


def neighbor_joining_hybrid_inverse_centrality(dist_matrix, cells, max_id, alpha=1.0, beta=0.5, epsilon=1e-6, seed=7, existing_tree=None):
    """
    Hybrid neighbor joining with inverse-distance centrality.
    Prefers pairs that are both close (small D[x,y]) and asymmetric
    in weighted centrality c'(x) = sum(1 / (D[x,i] + eps)).

    Parameters
    ----------
    dist_matrix : np.ndarray
        Pairwise distance matrix between cells.
    cells : list
        List of Genotype-like objects with .node_id, .cell_id, .genome.
    max_id : int
        Maximum node ID so far; new nodes start from max_id + 1.
    alpha : float
        Weight for distance term (default 1.0).
    beta : float
        Weight for centrality asymmetry term (default 0.5).
    epsilon : float
        Small constant to prevent division by zero (default 1e-6).
    seed : int
        Random seed for tie-breaking.
    existing_tree : nx.DiGraph, optional
        If provided, new nodes will be added to this tree.

    Returns
    -------
    tree : nx.DiGraph
    new_nodes : dict
    root_cell_id : any
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

        # --- Inverse-distance weighted centrality ---
        with np.errstate(divide='ignore', invalid='ignore'):
            invD = 1.0 / (D + epsilon)
            np.fill_diagonal(invD, 0.0)
            centrality = invD.sum(axis=1)

        # --- Hybrid score: prefer close + asymmetric ---
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

        # More central node (higher c') becomes parent
        if c_i > c_j:
            parent_idx, child_idx = i, j
        elif c_i < c_j:
            parent_idx, child_idx = j, i
        else:
            parent_idx, child_idx = (i, j) if rng.random() < 0.5 else (j, i)

        parent_leaf = node_list[parent_idx]
        child_leaf = node_list[child_idx]

        internal_node = type(parent_leaf)(genome=parent_leaf.genome, node_id=next_id, cell_id=parent_leaf.cell_id)
        next_id += 1

        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)
        tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=float(D[parent_idx, child_idx]))
        tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(D[parent_idx, child_idx]))

        new_nodes[internal_node] = (parent_leaf, child_leaf)

        # Update matrix after merge
        keep_indices = [k for k in range(n) if k != child_idx]
        D = D[np.ix_(keep_indices, keep_indices)]
        node_list[parent_idx] = internal_node
        node_list.pop(child_idx)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


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

    # Compute base "centrality" (inverse mean distance)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = 1.0 / (np.mean(D, axis=1) + 1e-9)
    c = (c - np.min(c)) / (np.ptp(c) + 1e-12)

    while len(node_list) > 1:
        n = len(D)
        score = np.full((n, n), np.inf)

        # Compute adaptive anticentral scores
        for i in range(n):
            for j in range(i + 1, n):
                c_diff = abs(c[i] - c[j])
                adaptive = np.exp(-gamma * c_diff / (np.std(c) + 1e-9))
                anticentral_term = (2 - (c[i] + c[j]))  # penalize high-centrality nodes
                score[i, j] = alpha * D[i, j] * (1 + gamma * adaptive) - beta * anticentral_term
                score[i, j] += rng.random() * 1e-6  # break ties

        i_best, j_best = divmod(np.argmin(score), n)
        if j_best < i_best:
            i_best, j_best = j_best, i_best

        a = node_list[i_best]
        b = node_list[j_best]

        # Create internal node
        internal_node = type(a)(
            genome=a.genome, node_id=next_id, cell_id=a.cell_id
        )
        next_id += 1
        tree.add_node(internal_node.node_id, genome=internal_node.genome, cell_id=internal_node.cell_id)
        tree.add_edge(internal_node.node_id, a.node_id, weight=float(D[i_best, j_best]))
        tree.add_edge(internal_node.node_id, b.node_id, weight=float(D[i_best, j_best]))

        new_nodes[internal_node] = (a, b)

        # Update distance and centrality
        keep = [k for k in range(n) if k != j_best]
        D = D[np.ix_(keep, keep)]
        node_list[i_best] = internal_node
        node_list.pop(j_best)
        c[i_best] = np.mean([c[i_best], c[j_best]])
        c = np.delete(c, j_best)

    root = node_list[0]
    return tree, new_nodes, root.cell_id


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


def build_evolution_tree(cell_lists, seed=7, dist_matrix_path=None, r=2, only_nj=False, inids=None, indm=None,
                         neighbor_joining=neighbor_joining_standard):
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
                x_ks = [x for x in x_ks if not np.any((x.genome == 0) & (y.genome > 0))]

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


def visualize_tree_plotly(tree, node_levels=None, output_file="reconstructed.html", level_node_ordering=None):
    pos = {}
    level_to_nodes = defaultdict(list)

    # Group nodes by level and sort them
    for node, level in node_levels.items():
        level_to_nodes[level].append(node)

    for level in level_to_nodes:
        nodes_in_level = level_to_nodes[level]
        if level_node_ordering and level in level_node_ordering:
            # Map from cell_id to node
            cell_id_to_node = {n.cell_id: n for n in nodes_in_level}
            specified_ids = level_node_ordering[level]
            specified_nodes = [cell_id_to_node[cid] for cid in specified_ids if cid in cell_id_to_node]

            # Get remaining nodes not specified
            remaining_nodes = [n for n in nodes_in_level if n.cell_id not in specified_ids]
            remaining_nodes.sort(key=lambda n: n.cell_id)  # optional sort of unspecified nodes

            # Combine specified + remaining
            level_to_nodes[level] = specified_nodes + remaining_nodes
        else:
            # Default: sort by cell_id
            level_to_nodes[level].sort(key=lambda n: n.cell_id)

    # Assign x/y positions
    offset = 0.25
    max_level = len(level_to_nodes)
    z = 1
    for level, nodes in level_to_nodes.items():
        for i, node in enumerate(nodes):
            if node.genome.size == 1 and node.genome.flatten()[0] is None:
                if z % 2:
                    pos[node.node_id] = (offset, level)
                else:
                    pos[node.node_id] = (max_level - offset, level)
                offset += 0.5
                z += 1
            else:
                pos[node.node_id] = (i, level)

    edge_x = []
    edge_y = []
    edge_label_pos_x, edge_label_pos_y = [], []
    edge_hover_labels = []
    edge_labels = []
    edge_marker_colors = []
    for (u, v), w in nx.get_edge_attributes(tree, 'weight').items():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])  # Add None to break the line
        edge_y.extend([y0, y1, None])
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        edge_label_pos_x.append(mid_x)
        edge_label_pos_y.append(mid_y)
        edge_labels.append("")  # Hide label by default
        # hover_edge_labels.append(str(event))  # Show label only on hover
        edge_marker_colors.append("green")
        edge_hover_labels.append(f"Distance: {w:.2f}")

        # Add markers for edge labels
    edge_l = go.Scatter(
            x=edge_label_pos_x, y=edge_label_pos_y, mode='markers+text',
            marker=dict(size=8, color=edge_marker_colors, opacity=0.5),  # Change color based on label presence
            text=edge_labels,
            hovertext=edge_hover_labels,  # Show edge label on hover
            textposition='middle center',
            hoverinfo='text'
        )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    text = []

    for node, data in tree.nodes(data=True):
        gen_str = data.get("genome", "N/A")
        if gen_str.size == 1 and gen_str.flatten()[0] is None:
            gen_str = "N/A"
        cell_id = data.get("cell_id", "N/A")
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label = f"cell_id={cell_id}<br>CN={gen_str}"
        text.append(label)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='lightblue',
            size=35,
            line_width=4),
        text=[data.get("cell_id", "N/A") for node, data in tree.nodes(data=True)],
        hovertext=text,
        textfont=dict(size=24)
    )

    pic=[]
    if level_node_ordering is not None:
        pic = [edge_trace, node_trace]
    else:
        pic = [edge_trace, node_trace, edge_l]

    fig = go.Figure(data=pic,
                   layout=go.Layout(
                       title=dict(
                           text='Reconstructed Tree',
                           font=dict(size=16)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white',
                        paper_bgcolor='white')
                   )
    fig.write_html(output_file)
    fig.show()


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