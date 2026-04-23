from reconstructor_algorithms import (
    make_nj_full_cps_variant,
    make_nj_full_variant,
    make_nj_hybrid_inv_cent_variant,
    make_nj_hybrid_variant,
    neighbor_joining_adaptive_centrality,
    neighbor_joining_adaptive_centrality_nonlinear,
    neighbor_joining_adaptive_centrality_reversed,
    neighbor_joining_baseline,
    neighbor_joining_hybrid_anticentral_adaptive_v3,
    neighbor_joining_hybrid_anticentral_adaptive_v3_plausible,
    neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony,
    neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible,
    neighbor_joining_hybrid_anticentral_opt,
    neighbor_joining_hybrid_opt,
    neighbor_joining_hybrid_opt_adaptive,
    neighbor_joining_hybrid_opt_refined,
    neighbor_joining_hybrid_opt_v2,
)


LEGACY_ALGORITHM_NAMES = [
    "neighbor_joining_baseline",
    "neighbor_joining_full_full",
    "neighbor_joining_full_partial",
    "neighbor_joining_full_cps_full",
    "neighbor_joining_full_cps_partial",
    "neighbor_joining_hybrid_full",
    "neighbor_joining_hybrid_partial",
    "neighbor_joining_hybrid_inverse_centrality_full",
    "neighbor_joining_hybrid_inverse_centrality_partial",
    "neighbor_joining_adaptive_centrality",
    "neighbor_joining_adaptive_centrality_nonlinear",
    "neighbor_joining_adaptive_centrality_reversed",
    "neighbor_joining_hybrid_opt",
    "neighbor_joining_hybrid_opt_adaptive",
    "neighbor_joining_hybrid_opt_v2",
    "neighbor_joining_hybrid_opt_refined",
    "neighbor_joining_hybrid_anticentral_opt",
    "neighbor_joining_hybrid_anticentral_adaptive_v3",
    "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible",
    "neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible",
    "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
]


def get_legacy_algorithms():
    neighbor_joining_full_full = make_nj_full_variant(True)
    neighbor_joining_full_partial = make_nj_full_variant(False)
    neighbor_joining_full_cps_full = make_nj_full_cps_variant(True)
    neighbor_joining_full_cps_partial = make_nj_full_cps_variant(False)
    neighbor_joining_hybrid_full = make_nj_hybrid_variant(True)
    neighbor_joining_hybrid_partial = make_nj_hybrid_variant(False)
    neighbor_joining_hybrid_inverse_centrality_full = make_nj_hybrid_inv_cent_variant(True)
    neighbor_joining_hybrid_inverse_centrality_partial = make_nj_hybrid_inv_cent_variant(False)
    return [
        neighbor_joining_baseline,
        neighbor_joining_full_full,
        neighbor_joining_full_partial,
        neighbor_joining_full_cps_full,
        neighbor_joining_full_cps_partial,
        neighbor_joining_hybrid_full,
        neighbor_joining_hybrid_partial,
        neighbor_joining_hybrid_inverse_centrality_full,
        neighbor_joining_hybrid_inverse_centrality_partial,
        neighbor_joining_adaptive_centrality,
        neighbor_joining_adaptive_centrality_nonlinear,
        neighbor_joining_adaptive_centrality_reversed,
        neighbor_joining_hybrid_opt,
        neighbor_joining_hybrid_opt_adaptive,
        neighbor_joining_hybrid_opt_v2,
        neighbor_joining_hybrid_opt_refined,
        neighbor_joining_hybrid_anticentral_opt,
        neighbor_joining_hybrid_anticentral_adaptive_v3,
        neighbor_joining_hybrid_anticentral_adaptive_v3_plausible,
        neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible,
        neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony,
    ]


def get_experimental_algorithms():
    return []


def get_algorithms():
    return get_legacy_algorithms() + get_experimental_algorithms()


def get_algorithm_map(algorithms=None):
    selected_algorithms = get_algorithms() if algorithms is None else algorithms
    return {
        getattr(algorithm, "__name__", str(algorithm)): algorithm
        for algorithm in selected_algorithms
    }


def resolve_reconstruction_algorithm(algorithm_name):
    if algorithm_name is None:
        return None
    if isinstance(algorithm_name, str) and algorithm_name.strip().lower() in {"", "none"}:
        return None

    algorithms_by_name = get_algorithm_map()
    if algorithm_name not in algorithms_by_name:
        available = ", ".join(sorted(algorithms_by_name))
        raise ValueError(
            f"Unknown reconstruction algorithm '{algorithm_name}'. Available options: {available}"
        )
    return algorithms_by_name[algorithm_name]


get_legacy_algorithms_to_test = get_legacy_algorithms
get_experimental_algorithms_to_test = get_experimental_algorithms
get_algorithms_to_test = get_algorithms


__all__ = [
    "LEGACY_ALGORITHM_NAMES",
    "get_algorithm_map",
    "get_algorithms",
    "get_algorithms_to_test",
    "get_experimental_algorithms",
    "get_experimental_algorithms_to_test",
    "get_legacy_algorithms",
    "get_legacy_algorithms_to_test",
    "resolve_reconstruction_algorithm",
]
