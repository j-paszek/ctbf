from reconstructor_ancestor_selection import (
    keep_pair_order_parent_selector,
    less_mixed_centrality_parent_selector,
    lower_sum_distance_parent_selector,
    make_plausible_pair_order_parent_selector,
    make_plausible_parsimony_parent_selector,
    more_central_parent_selector,
    more_central_parent_selector_left_tie,
    pair_choice_orientation_selector,
    plausible_then_centrality_parent_selector,
)
from reconstructor_anticentral import (
    configure_anticentral_v3_state,
)
from reconstructor_distance_update import anticentral_v3_distance_update
from reconstructor_engine import run_agglomerative_reconstruction
from reconstructor_merge import (
    anticentral_weighted_copy_parent_node,
    copy_parent_equal_weight_internal_node,
    copy_parent_without_new_node_record,
)
from reconstructor_nj import (
    make_nj_full_cps_variant,
    make_nj_full_variant,
    make_nj_hybrid_inv_cent_variant,
    make_nj_hybrid_variant,
    neighbour_joining_core,
    neighbor_joining_baseline,
    neighbor_joining_classical,
    neighbor_joining_standard,
    rooted_labeled_nj,
)
from reconstructor_pair_selection import (
    make_anticentral_adaptive_v2_pair_selector,
    make_anticentral_adaptive_v3_pair_selector,
    make_anticentral_adaptive_v3_skip_unplausible_pair_selector,
    make_anticentral_hybrid_opt_pair_selector,
    make_adaptive_centrality_nonlinear_pair_selector,
    make_adaptive_centrality_pair_selector,
    make_adaptive_centrality_reversed_pair_selector,
    make_hybrid_opt_adaptive_pair_selector,
    make_hybrid_opt_pair_selector,
    make_hybrid_opt_refined_pair_selector,
    make_hybrid_opt_v2_pair_selector,
)
from reconstructor_temporal import (
    temporal_cnp_arborescence,
    temporal_cnp_arborescence_no_time,
)


def _run_configured_algorithm(
    dist_matrix,
    cells,
    max_id,
    *,
    seed,
    existing_tree,
    pair_selector,
    ancestor_selector,
    merge_strategy=None,
    distance_update=None,
    configure_state=None,
):
    kwargs = {
        "seed": seed,
        "existing_tree": existing_tree,
        "pair_selector": pair_selector,
        "ancestor_selector": ancestor_selector,
    }
    if merge_strategy is not None:
        kwargs["merge_strategy"] = merge_strategy
    if distance_update is not None:
        kwargs["distance_update"] = distance_update
    if configure_state is not None:
        kwargs["configure_state"] = configure_state

    return run_agglomerative_reconstruction(dist_matrix, cells, max_id, **kwargs)


def _run_anticentral_v3_algorithm(
    dist_matrix,
    cells,
    max_id,
    *,
    seed,
    existing_tree,
    pair_selector,
    ancestor_selector,
):
    return _run_configured_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=pair_selector,
        ancestor_selector=ancestor_selector,
        merge_strategy=anticentral_weighted_copy_parent_node,
        distance_update=anticentral_v3_distance_update,
        configure_state=configure_anticentral_v3_state,
    )


def neighbor_joining_adaptive_centrality(
    dist_matrix,
    cells,
    max_id,
    alpha=1.0,
    beta=0.5,
    epsilon=1e-6,
    seed=7,
    existing_tree=None,
):
    return _run_configured_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_adaptive_centrality_pair_selector(alpha, beta, epsilon),
        ancestor_selector=more_central_parent_selector,
        merge_strategy=copy_parent_equal_weight_internal_node,
    )


def neighbor_joining_adaptive_centrality_nonlinear(
    dist_matrix,
    cells,
    max_id,
    alpha=1.0,
    beta=0.5,
    epsilon=1e-6,
    k=10.0,
    tau=0.5,
    seed=7,
    existing_tree=None,
):
    return _run_configured_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_adaptive_centrality_nonlinear_pair_selector(alpha, beta, epsilon, k, tau),
        ancestor_selector=more_central_parent_selector,
        merge_strategy=copy_parent_equal_weight_internal_node,
    )


def neighbor_joining_adaptive_centrality_reversed(
    dist_matrix,
    cells,
    max_id,
    seed=7,
    existing_tree=None,
):
    return _run_configured_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_adaptive_centrality_reversed_pair_selector(),
        ancestor_selector=more_central_parent_selector_left_tie,
    )


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
    existing_tree=None,
):
    return _run_configured_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_hybrid_opt_pair_selector(
            alpha,
            beta,
            gamma,
            epsilon,
            k,
            tau,
            reverse_centrality,
            use_q_as_secondary,
        ),
        ancestor_selector=more_central_parent_selector,
    )


def neighbor_joining_hybrid_opt_adaptive(
    dist_matrix,
    cells,
    max_id,
    seed=7,
    existing_tree=None,
    alpha=1.0,
    beta=0.8,
    gamma=0.1,
    epsilon=1e-6,
):
    return _run_configured_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_hybrid_opt_adaptive_pair_selector(alpha, beta, epsilon),
        ancestor_selector=more_central_parent_selector_left_tie,
    )


def neighbor_joining_hybrid_opt_v2(
    dist_matrix,
    cells,
    max_id,
    seed=7,
    existing_tree=None,
    alpha=1.0,
    beta=1.0,
    lam=1.0,
    epsilon=1e-6,
):
    return _run_configured_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_hybrid_opt_v2_pair_selector(alpha, beta, lam, epsilon),
        ancestor_selector=more_central_parent_selector_left_tie,
    )


def neighbor_joining_hybrid_opt_refined(
    dist_matrix,
    cells,
    max_id,
    seed=7,
    existing_tree=None,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    return _run_configured_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_hybrid_opt_refined_pair_selector(alpha, beta, gamma),
        ancestor_selector=lower_sum_distance_parent_selector,
        merge_strategy=copy_parent_without_new_node_record,
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
    Opposite of the hybrid_opt heuristic - favors merging anti-central pairs
    (large |c_mix(x) - c_mix(y)|, but also high distance).

    Score = alpha*D[x,y] + beta*|c_mix(x) - c_mix(y)|
    (note the + instead of - in hybrid_opt)
    """

    return _run_configured_algorithm(
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

    Score = alpha*D[i,j] + beta*|c_mix(i)-c_mix(j)| - gamma/(D[i,j]+epsilon)
    Adaptive scaling adjusts alpha,beta,gamma with iteration depth.
    """

    return _run_configured_algorithm(
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

    return _run_anticentral_v3_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v3_pair_selector(alpha, beta, gamma),
        ancestor_selector=keep_pair_order_parent_selector,
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
    extended with directional biological plausibility.

    Differences vs neighbor_joining_hybrid_anticentral_adaptive_v3:
    - The same pair (a, b) is always chosen (same score logic).
    - BUT we decide whether the ancestor template should be 'a' or 'b'
      using is_biologically_plausible_ancestor, when possible.

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

    return _run_anticentral_v3_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v3_pair_selector(alpha, beta, gamma),
        ancestor_selector=make_plausible_pair_order_parent_selector(enforce_plausibility),
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
    Anticentral Adaptive NJ (v3) with biological plausibility pair skipping.

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

    return _run_anticentral_v3_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v3_skip_unplausible_pair_selector(alpha, beta, gamma),
        ancestor_selector=pair_choice_orientation_selector,
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

    return _run_anticentral_v3_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v3_pair_selector(alpha, beta, gamma),
        ancestor_selector=make_plausible_parsimony_parent_selector(baseline_cn),
    )


def new_alg(
    dist_matrix,
    cells,
    max_id,
    seed=None,
    existing_tree=None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
):
    return _run_anticentral_v3_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=seed,
        existing_tree=existing_tree,
        pair_selector=make_anticentral_adaptive_v3_pair_selector(alpha, beta, gamma),
        ancestor_selector=plausible_then_centrality_parent_selector,
    )


__all__ = [
    "make_nj_full_cps_variant",
    "make_nj_full_variant",
    "make_nj_hybrid_inv_cent_variant",
    "make_nj_hybrid_variant",
    "neighbour_joining_core",
    "new_alg",
    "neighbor_joining_adaptive_centrality",
    "neighbor_joining_adaptive_centrality_nonlinear",
    "neighbor_joining_adaptive_centrality_reversed",
    "neighbor_joining_baseline",
    "neighbor_joining_classical",
    "neighbor_joining_hybrid_anticentral_adaptive_v2",
    "neighbor_joining_hybrid_anticentral_adaptive_v3",
    "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible",
    "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
    "neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible",
    "neighbor_joining_hybrid_anticentral_opt",
    "neighbor_joining_hybrid_opt",
    "neighbor_joining_hybrid_opt_adaptive",
    "neighbor_joining_hybrid_opt_refined",
    "neighbor_joining_hybrid_opt_v2",
    "neighbor_joining_standard",
    "rooted_labeled_nj",
    "temporal_cnp_arborescence",
    "temporal_cnp_arborescence_no_time",
]
