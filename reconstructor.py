import random
import networkx as nx
import numpy as np
from simulator import Genotype
from reconstructor_ancestor_selection import _is_biologically_plausible_ancestor
from reconstructor_anticentral import (
    _initial_anticentral_v3_centrality,
    _anticentral_adaptive_v3_score_matrix,
    _best_pair_from_score,
    _ordered_pairs_by_score,
    _copy_parent_internal_node,
    _merge_parent_child,
    _total_deviation_from_baseline,
)
from reconstructor_biopsy_guided import build_evolution_tree_impl
from reconstructor_nj import (
    make_nj_full_cps_variant,
    make_nj_full_variant,
    make_nj_hybrid_inv_cent_variant,
    make_nj_hybrid_variant,
    neighbour_joining_core,
    neighbor_joining_baseline,
    neighbor_joining_standard,
)
from reconstructor_utils import visualize_tree_plotly

from reconstructor_adaptive import (
    neighbor_joining_adaptive_centrality,
    neighbor_joining_adaptive_centrality_nonlinear,
    neighbor_joining_adaptive_centrality_reversed,
    neighbor_joining_hybrid_opt,
    neighbor_joining_hybrid_opt_adaptive,
    neighbor_joining_hybrid_opt_refined,
    neighbor_joining_hybrid_opt_v2,
)


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
