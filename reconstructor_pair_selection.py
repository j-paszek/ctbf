import numpy as np

from reconstructor_ancestor_selection import _is_biologically_plausible_pair
from reconstructor_engine import PairChoice


def _upper_triangle_pairs(n):
    return zip(*np.triu_indices(n, k=1))


def _best_pair_from_score_matrix(score, minimize=True):
    selected_score = np.min(score) if minimize else np.max(score)
    i, j = divmod(np.argmin(score) if minimize else np.argmax(score), score.shape[0])
    if j < i:
        i, j = j, i
    return i, j, selected_score


def _ordered_pairs_by_score_matrix(score, minimize=True):
    n = score.shape[0]
    return sorted(
        [(i, j) for i in range(n) for j in range(i + 1, n)],
        key=lambda pair: score[pair[0], pair[1]],
        reverse=not minimize,
    )


def _inverse_distance_centrality(D, epsilon, include_diagonal=False):
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_D = 1.0 / (D + epsilon)
    if not include_diagonal:
        np.fill_diagonal(inv_D, 0.0)
    return inv_D.sum(axis=1)


def _mixed_direct_inverse_centrality(D, epsilon, lam):
    c_dir = D.sum(axis=1)
    c_inv = _inverse_distance_centrality(D, epsilon)
    return c_inv / np.power(c_dir + epsilon, lam)


def _nj_q_matrix(D):
    n = D.shape[0]
    Q = np.full((n, n), np.inf, dtype=float)
    if n <= 2:
        return Q

    total = D.sum(axis=1)
    factor = n - 2
    for i in range(n):
        for j in range(i + 1, n):
            q_val = factor * D[i, j] - total[i] - total[j]
            Q[i, j] = q_val
            Q[j, i] = q_val
    return Q


def _score_distance_minus_asymmetry(D, centrality, alpha, beta):
    n = len(D)
    score = np.full((n, n), np.inf)
    for i, j in _upper_triangle_pairs(n):
        score[i, j] = alpha * D[i, j] - beta * abs(centrality[i] - centrality[j])
    return score


def _linear_blended_centrality(state, epsilon):
    n = len(state.D)
    original_n = len(state.D_full)
    weight = (n - 2) / max(original_n - 2, 1)
    sum_dist = state.D.sum(axis=1)
    global_c = 1.0 / (sum_dist + epsilon)
    inv_c = _inverse_distance_centrality(state.D, epsilon)
    return weight * global_c + (1 - weight) * inv_c


def _sigmoid_blended_centrality(state, epsilon, k, tau):
    frac = len(state.D) / max(len(state.D_full), 1)
    weight = 1.0 / (1.0 + np.exp(-k * (frac - tau)))
    sum_dist = state.D.sum(axis=1)
    global_c = 1.0 / (sum_dist + epsilon)
    inv_c = _inverse_distance_centrality(state.D, epsilon)
    return weight * global_c + (1 - weight) * inv_c


def _reversed_adaptive_centrality(state, epsilon):
    weight = 1 - (len(state.D) / len(state.D_full))
    sum_dist = state.D.sum(axis=1)
    inv_sum = _inverse_distance_centrality(state.D, epsilon, include_diagonal=True)
    return (1 - weight) * inv_sum + weight * (1.0 / (sum_dist + epsilon))


def _hybrid_opt_centrality(state, epsilon, k, tau, reverse_centrality):
    frac = len(state.D) / max(len(state.D_full), 1)
    weight = 1.0 / (1.0 + np.exp(-k * (frac - tau))) if k > 0 else frac

    sum_dist = state.D.sum(axis=1)
    global_c = 1.0 / (sum_dist + epsilon)
    inv_c = _inverse_distance_centrality(state.D, epsilon)

    if reverse_centrality:
        return (1 - weight) * global_c + weight * inv_c
    return weight * global_c + (1 - weight) * inv_c


def make_adaptive_centrality_pair_selector(alpha=1.0, beta=0.5, epsilon=1e-6):
    def select_pair(state):
        centrality = _linear_blended_centrality(state, epsilon)
        score = _score_distance_minus_asymmetry(state.D, centrality, alpha, beta)
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
        centrality = _sigmoid_blended_centrality(state, epsilon, k, tau)
        score = _score_distance_minus_asymmetry(state.D, centrality, alpha, beta)
        i, j, _ = _best_pair_from_score_matrix(score)
        return PairChoice(i, j, score=score[i, j], metadata={"centrality": centrality})

    return select_pair


def make_adaptive_centrality_reversed_pair_selector(epsilon=1e-9):
    def select_pair(state):
        centrality = _reversed_adaptive_centrality(state, epsilon)
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
        centrality = _hybrid_opt_centrality(state, epsilon, k, tau, reverse_centrality)
        Q = _nj_q_matrix(D) if use_q_as_secondary and n > 2 else None

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

        centrality = _inverse_distance_centrality(D, epsilon)
        score = _score_distance_minus_asymmetry(D, centrality, alpha_eff, beta_eff)
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
        centrality = _mixed_direct_inverse_centrality(state.D, epsilon, lam)
        score = _score_distance_minus_asymmetry(state.D, centrality, alpha, beta)
        i, j, _ = _best_pair_from_score_matrix(score)
        return PairChoice(i, j, score=score[i, j], metadata={"centrality": centrality})

    return select_pair


def make_hybrid_opt_refined_pair_selector(alpha=1.0, beta=1.0, gamma=1.0):
    def select_pair(state):
        D = state.D
        n = len(D)
        centrality = D.sum(axis=1)
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
      - among all PLAUSIBLE pairs (according to _is_biologically_plausible_pair),
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
            if _is_biologically_plausible_pair(node_list[i], node_list[j]):
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
    centrality = D.sum(axis=1)

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
    centrality = D.sum(axis=1)

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
    centrality = _inverse_distance_centrality(D, epsilon)

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
