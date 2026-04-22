from simulator import Genotype
from reconstructor_anticentral import (
    anticentral_v3_distance_update,
    anticentral_weighted_copy_parent_node,
    configure_anticentral_v3_state,
    keep_pair_order_parent_selector,
    less_mixed_centrality_parent_selector,
    make_anticentral_adaptive_v2_pair_selector,
    make_anticentral_adaptive_v3_pair_selector,
    make_anticentral_adaptive_v3_skip_unplausible_pair_selector,
    make_anticentral_hybrid_opt_pair_selector,
    make_plausible_pair_order_parent_selector,
    make_plausible_parsimony_parent_selector,
    pair_choice_orientation_selector,
)
from reconstructor_biopsy_guided import build_evolution_tree_impl
from reconstructor_engine import run_agglomerative_reconstruction
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

    return run_agglomerative_reconstruction(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_hybrid_opt_pair_selector(alpha, beta, lam, epsilon),
        ancestor_selector=less_mixed_centrality_parent_selector,
    )


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

    return run_agglomerative_reconstruction(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v2_pair_selector(alpha, beta, gamma, lam, epsilon),
        ancestor_selector=less_mixed_centrality_parent_selector,
    )


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

    return run_agglomerative_reconstruction(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v3_pair_selector(alpha, beta, gamma),
        ancestor_selector=keep_pair_order_parent_selector,
        merge_strategy=anticentral_weighted_copy_parent_node,
        distance_update=anticentral_v3_distance_update,
        configure_state=configure_anticentral_v3_state,
    )


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

    return run_agglomerative_reconstruction(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v3_pair_selector(alpha, beta, gamma),
        ancestor_selector=make_plausible_pair_order_parent_selector(enforce_plausibility),
        merge_strategy=anticentral_weighted_copy_parent_node,
        distance_update=anticentral_v3_distance_update,
        configure_state=configure_anticentral_v3_state,
    )


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

    return run_agglomerative_reconstruction(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v3_skip_unplausible_pair_selector(alpha, beta, gamma),
        ancestor_selector=pair_choice_orientation_selector,
        merge_strategy=anticentral_weighted_copy_parent_node,
        distance_update=anticentral_v3_distance_update,
        configure_state=configure_anticentral_v3_state,
    )


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

    return run_agglomerative_reconstruction(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v3_pair_selector(alpha, beta, gamma),
        ancestor_selector=make_plausible_parsimony_parent_selector(baseline_cn),
        merge_strategy=anticentral_weighted_copy_parent_node,
        distance_update=anticentral_v3_distance_update,
        configure_state=configure_anticentral_v3_state,
    )


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
