import numpy as np

from reconstructor_plausibility import (
    is_biologically_plausible_ancestor,
    is_biologically_plausible_pair,
)
from reconstructor_distance_update import ANTICENTRAL_V3_CONTEXT_KEY
from reconstructor_engine import Orientation, PairChoice
from reconstructor_metrics import (
    hybrid_opt_centrality,
    inverse_distance_centrality,
    linear_blended_centrality,
    mixed_direct_inverse_centrality,
    nj_q_matrix,
    reversed_adaptive_centrality,
    score_distance_minus_asymmetry,
    sigmoid_blended_centrality,
    sum_distance_centrality,
    upper_triangle_pairs,
)


def _upper_triangle_pairs(n):
    return upper_triangle_pairs(n)


def _best_pair_from_score_matrix(score, minimize=True):
    best_pair = None
    best_score = None

    for i, j in _upper_triangle_pairs(score.shape[0]):
        current_score = score[i, j]
        if best_score is None:
            best_pair = (i, j)
            best_score = current_score
        elif minimize and current_score < best_score:
            best_pair = (i, j)
            best_score = current_score
        elif not minimize and current_score > best_score:
            best_pair = (i, j)
            best_score = current_score

    if best_pair is None:
        raise ValueError("No upper-triangle pairs available in score matrix.")

    return best_pair[0], best_pair[1], best_score


def _ordered_pairs_by_score_matrix(score, minimize=True):
    n = score.shape[0]
    return sorted(
        [(i, j) for i in range(n) for j in range(i + 1, n)],
        key=lambda pair: score[pair[0], pair[1]],
        reverse=not minimize,
    )


def make_adaptive_centrality_pair_selector(alpha=1.0, beta=0.5, epsilon=1e-6):
    def select_pair(state):
        centrality = linear_blended_centrality(state, epsilon)
        score = score_distance_minus_asymmetry(state.D, centrality, alpha, beta)
        i, j, _ = _best_pair_from_score_matrix(score)
        return PairChoice(i, j, score=score[i, j], metadata={"centrality": centrality})

    return select_pair


def make_adaptive_centrality_nonlinear_pair_selector(
    alpha=1.0,
    beta=0.5,
    epsilon=1e-6,
    k=10.0,
    tau=0.5,
):
    def select_pair(state):
        centrality = sigmoid_blended_centrality(state, epsilon, k, tau)
        score = score_distance_minus_asymmetry(state.D, centrality, alpha, beta)
        i, j, _ = _best_pair_from_score_matrix(score)
        return PairChoice(i, j, score=score[i, j], metadata={"centrality": centrality})

    return select_pair


def make_adaptive_centrality_reversed_pair_selector(epsilon=1e-9):
    def select_pair(state):
        centrality = reversed_adaptive_centrality(state, epsilon)
        n = len(state.D)
        score = np.full((n, n), np.inf)
        for i, j in _upper_triangle_pairs(n):
            asym = abs(centrality[i] - centrality[j])
            score[i, j] = state.D[i, j] * (1.0 - 0.25 * asym)

        i, j, _ = _best_pair_from_score_matrix(score)
        return PairChoice(i, j, score=score[i, j], metadata={"centrality": centrality})

    return select_pair


def make_hybrid_opt_pair_selector(
    alpha=1.0,
    beta=0.7,
    gamma=0.2,
    epsilon=1e-9,
    k=10.0,
    tau=0.5,
    reverse_centrality=False,
    use_q_as_secondary=True,
):
    def select_pair(state):
        D = state.D
        n = len(D)
        centrality = hybrid_opt_centrality(state, epsilon, k, tau, reverse_centrality)
        Q = nj_q_matrix(D) if use_q_as_secondary and n > 2 else None

        best_score = np.inf
        best_candidates = []
        for i, j in _upper_triangle_pairs(n):
            d_ij = D[i, j]
            asym = abs(centrality[i] - centrality[j])
            score = alpha * d_ij - beta * asym
            if Q is not None:
                score -= gamma * Q[i, j]

            candidate = (i, j, d_ij, asym, Q[i, j] if Q is not None else None)
            if score < best_score - 1e-12:
                best_score = score
                best_candidates = [candidate]
            elif abs(score - best_score) <= 1e-12:
                best_candidates.append(candidate)

        if len(best_candidates) > 1:
            best_candidates.sort(key=lambda x: (x[2], -x[3], x[4] if x[4] is not None else 0.0))

        i, j, _, _, _ = best_candidates[0]
        return PairChoice(i, j, score=best_score, metadata={"centrality": centrality, "Q": Q})

    return select_pair


def make_hybrid_opt_adaptive_pair_selector(alpha=1.0, beta=0.8, epsilon=1e-6):
    def select_pair(state):
        D = state.D
        dist_range = D.max() - D.min()
        heterogeneity = np.std(D) / dist_range if dist_range > 1e-9 else 0.0
        alpha_eff = alpha * (1 + 0.5 * heterogeneity)
        beta_eff = beta * (1 - 0.5 * heterogeneity)

        centrality = inverse_distance_centrality(D, epsilon)
        score = score_distance_minus_asymmetry(D, centrality, alpha_eff, beta_eff)
        i, j, _ = _best_pair_from_score_matrix(score)
        return PairChoice(
            i,
            j,
            score=score[i, j],
            metadata={"centrality": centrality, "alpha_eff": alpha_eff, "beta_eff": beta_eff},
        )

    return select_pair


def make_hybrid_opt_v2_pair_selector(alpha=1.0, beta=1.0, lam=1.0, epsilon=1e-6):
    def select_pair(state):
        centrality = mixed_direct_inverse_centrality(state.D, epsilon, lam)
        score = score_distance_minus_asymmetry(state.D, centrality, alpha, beta)
        i, j, _ = _best_pair_from_score_matrix(score)
        return PairChoice(i, j, score=score[i, j], metadata={"centrality": centrality})

    return select_pair


def make_hybrid_opt_refined_pair_selector(alpha=1.0, beta=1.0, gamma=1.0):
    def select_pair(state):
        D = state.D
        n = len(D)
        centrality = sum_distance_centrality(D)
        mean_D = D[np.triu_indices(n, 1)].mean()
        mean_c = np.mean(centrality)

        score = np.full((n, n), np.inf)
        for i, j in _upper_triangle_pairs(n):
            d_ij = D[i, j] / mean_D
            asym = abs(centrality[i] - centrality[j]) / mean_c
            score[i, j] = alpha * d_ij - beta * (asym ** gamma)

        i, j, _ = _best_pair_from_score_matrix(score)
        return PairChoice(i, j, score=score[i, j], metadata={"centrality": centrality})

    return select_pair


def _anticentral_adaptive_v3_score_matrix(D, c, rng, alpha=1.0, beta=1.0, gamma=0.5):
    n = len(D)
    score = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            c_diff = abs(c[i] - c[j])
            adaptive = np.exp(-gamma * c_diff / (np.std(c) + 1e-9))
            anticentral_term = 2 - (c[i] + c[j])
            score[i, j] = (
                alpha * D[i, j] * (1 + gamma * adaptive)
                - beta * anticentral_term
            )
            score[i, j] += rng.random() * 1e-6
    return score


def make_anticentral_adaptive_v3_pair_selector(alpha=1.0, beta=1.0, gamma=0.5):
    def select_pair(state):
        c = state.context[ANTICENTRAL_V3_CONTEXT_KEY]
        score = _anticentral_adaptive_v3_score_matrix(state.D, c, state.rng, alpha, beta, gamma)
        i_best, j_best, _ = _best_pair_from_score_matrix(score)
        return PairChoice(i_best, j_best, score=score[i_best, j_best])

    return select_pair


def make_anticentral_adaptive_v3_skip_unplausible_pair_selector(alpha=1.0, beta=1.0, gamma=0.5):
    def select_pair(state):
        c = state.context[ANTICENTRAL_V3_CONTEXT_KEY]
        score = _anticentral_adaptive_v3_score_matrix(state.D, c, state.rng, alpha, beta, gamma)
        pair_order = _ordered_pairs_by_score_matrix(score)

        for i, j in pair_order:
            a = state.node_list[i]
            b = state.node_list[j]
            can_i_parent_j = is_biologically_plausible_ancestor(a, b)
            can_j_parent_i = is_biologically_plausible_ancestor(b, a)

            if can_i_parent_j or can_j_parent_i:
                if can_i_parent_j and not can_j_parent_i:
                    parent_idx, child_idx = i, j
                elif can_j_parent_i and not can_i_parent_j:
                    parent_idx, child_idx = j, i
                else:
                    parent_idx, child_idx = (i, j) if c[i] > c[j] else (j, i)
                return PairChoice(
                    i,
                    j,
                    score=score[i, j],
                    metadata={"orientation": Orientation(parent_idx, child_idx)},
                )

        i, j = pair_order[0]
        parent_idx, child_idx = (i, j) if c[i] > c[j] else (j, i)
        return PairChoice(
            i,
            j,
            score=score[i, j],
            metadata={"orientation": Orientation(parent_idx, child_idx)},
        )

    return select_pair


def make_anticentral_hybrid_opt_pair_selector(alpha=1.0, beta=1.0, lam=1.0, epsilon=1e-6):
    def select_pair(state):
        n = len(state.D)
        c_mix = mixed_direct_inverse_centrality(state.D, epsilon, lam)

        score = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = state.D[i, j]
                if not np.isfinite(d_ij):
                    continue
                asym = abs(c_mix[i] - c_mix[j])
                score[i, j] = alpha * d_ij + beta * asym

        i, j, _ = _best_pair_from_score_matrix(score, minimize=False)

        return PairChoice(i, j, score=score[i, j], metadata={"c_mix": c_mix})

    return select_pair


def make_anticentral_adaptive_v2_pair_selector(
    alpha=1.0,
    beta=1.0,
    gamma=0.2,
    lam=1.0,
    epsilon=1e-6,
):
    def select_pair(state):
        n = len(state.D)
        original_n = len(state.D_full)
        a = alpha * (1 + 0.2 * np.tanh(3 * (1 - len(state.node_list) / original_n)))
        b = beta * (1 + 0.3 * np.tanh(2 * (len(state.node_list) / original_n)))
        g = gamma * (0.5 + 0.5 * np.tanh(3 * len(state.node_list) / original_n))

        c_mix = mixed_direct_inverse_centrality(state.D, epsilon, lam)

        score = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = state.D[i, j]
                if not np.isfinite(d_ij):
                    continue
                asym = abs(c_mix[i] - c_mix[j])
                score[i, j] = a * d_ij + b * asym - g / (d_ij + epsilon)

        i, j, _ = _best_pair_from_score_matrix(score, minimize=False)

        return PairChoice(i, j, score=score[i, j], metadata={"c_mix": c_mix})

    return select_pair


# ============================================================
#  PAIR TIE-BREAKER
# ============================================================
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

    # Deterministic random tie-breaker
    return rng.choice(candidate_pairs)


# ============================================================
#  UNIVERSAL PAIR SELECTION CORE
# ============================================================
def _select_pair_core(D, node_list, rng, pair_score_func, minimize=True, apply_plausibility=True):
    """
    Universal pair-selection core used by all NJ variants.

    If apply_plausibility is True:
      - among all PLAUSIBLE pairs (according to is_biologically_plausible_pair),
        choose the one(s) with best score;
      - if NO plausible pair exists at all, fall back to best score among ALL pairs.

    If apply_plausibility is False:
      - ignore biological plausibility, just choose best score among ALL pairs.
    """
    n = len(D)

    best_score_all = None
    best_pairs_all = []

    best_score_plaus = None
    best_pairs_plaus = []

    for i, j in _upper_triangle_pairs(n):
        score = pair_score_func(i, j)

        # --- track best over ALL pairs ---
        if best_score_all is None:
            best_score_all = score
            best_pairs_all = [(i, j)]
        else:
            if minimize:
                if score < best_score_all:
                    best_score_all = score
                    best_pairs_all = [(i, j)]
                elif score == best_score_all:
                    best_pairs_all.append((i, j))
            else:
                if score > best_score_all:
                    best_score_all = score
                    best_pairs_all = [(i, j)]
                elif score == best_score_all:
                    best_pairs_all.append((i, j))

        # --- track best among PLAUSIBLE pairs (if enabled) ---
        if apply_plausibility:
            if is_biologically_plausible_pair(node_list[i], node_list[j]):
                if best_score_plaus is None:
                    best_score_plaus = score
                    best_pairs_plaus = [(i, j)]
                else:
                    if minimize:
                        if score < best_score_plaus:
                            best_score_plaus = score
                            best_pairs_plaus = [(i, j)]
                        elif score == best_score_plaus:
                            best_pairs_plaus.append((i, j))
                    else:
                        if score > best_score_plaus:
                            best_score_plaus = score
                            best_pairs_plaus = [(i, j)]
                        elif score == best_score_plaus:
                            best_pairs_plaus.append((i, j))

    # ---- DECISION ----
    if apply_plausibility and best_pairs_plaus:
        # we found at least one plausible pair -> use only those
        return _resolve_pair_ties(best_pairs_plaus, node_list, rng)

    # no plausible pair (or plausibility disabled) -> fall back to best over ALL pairs
    if apply_plausibility and not best_pairs_plaus:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@ not plausible @@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    return _resolve_pair_ties(best_pairs_all, node_list, rng)


# ============================================================
#  PAIR SELECTORS FOR SPECIFIC ALGORITHMS
# ============================================================

# --- Full NJ: score(i,j) = D[i,j]
def _select_pair_full(D, node_list, rng, minimize=True):
    return _select_pair_core(
        D=D,
        node_list=node_list,
        rng=rng,
        pair_score_func=lambda i, j: D[i, j],
        minimize=minimize,
        apply_plausibility=True
    )


# --- CPS NJ selector
def _select_pair_cps(D, node_list, rng, minimize=True):
    centrality = sum_distance_centrality(D)

    def cps_score(i, j):
        c_i, c_j = centrality[i], centrality[j]
        return (D[i, j], min(c_i, c_j), -max(c_i, c_j))

    return _select_pair_core(
        D=D,
        node_list=node_list,
        rng=rng,
        pair_score_func=cps_score,
        minimize=minimize,
        apply_plausibility=True
    )


# --- Hybrid NJ selector
def _select_pair_hybrid(D, node_list, rng, minimize=True, alpha=1.0, beta=0.5):
    centrality = sum_distance_centrality(D)

    def hybrid_score(i, j):
        return alpha * D[i, j] - beta * abs(centrality[i] - centrality[j])

    return _select_pair_core(
        D=D,
        node_list=node_list,
        rng=rng,
        pair_score_func=hybrid_score,
        minimize=minimize,
        apply_plausibility=True
    )


def _select_pair_hybrid_inv_centrality(D, node_list, rng,
                                       minimize=True,
                                       alpha=1.0, beta=0.5,
                                       epsilon=1e-6):
    """
    Hybrid NJ with inverse-distance weighted centrality:
        score = alpha * D[i,j] - beta * abs(c'[i] - c'[j])
    where    c'[i] = sum_k 1/(D[i,k] + eps)
    """
    centrality = inverse_distance_centrality(D, epsilon)

    def hybrid_inv_score(i, j):
        return alpha * D[i, j] - beta * abs(centrality[i] - centrality[j])

    return _select_pair_core(
        D=D,
        node_list=node_list,
        rng=rng,
        pair_score_func=hybrid_inv_score,
        minimize=True,
        apply_plausibility=True
    )


__all__ = [
    "make_anticentral_adaptive_v2_pair_selector",
    "make_anticentral_adaptive_v3_pair_selector",
    "make_anticentral_adaptive_v3_skip_unplausible_pair_selector",
    "make_anticentral_hybrid_opt_pair_selector",
    "make_adaptive_centrality_nonlinear_pair_selector",
    "make_adaptive_centrality_pair_selector",
    "make_adaptive_centrality_reversed_pair_selector",
    "make_hybrid_opt_adaptive_pair_selector",
    "make_hybrid_opt_pair_selector",
    "make_hybrid_opt_refined_pair_selector",
    "make_hybrid_opt_v2_pair_selector",
]
