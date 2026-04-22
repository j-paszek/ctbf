import numpy as np


# ============================================================
#  BIOLOGICAL PLAUSIBILITY USED BY PAIR SELECTION
# ============================================================
def _is_biologically_plausible_ancestor(ancestor, descendant):
    """
    Returns True if 'ancestor' could biologically be the parent of 'descendant'.

    Constraint:
    - ancestor cannot generate descendant if ancestor has CN=0 at a locus
      where descendant has CN>0 (i.e. a gain from 0 -> positive is disallowed).
    """
    return not np.any((ancestor.genome == 0) & (descendant.genome > 0))


def _is_biologically_plausible_pair(x, y):
    """
    A pair (x, y) is biologically plausible if at least one direction
    (x->y or y->x) is biologically possible.

    This means:
    - keep this pair if x can be parent of y OR y can be parent of x.
    """
    return (_is_biologically_plausible_ancestor(x, y) or
            _is_biologically_plausible_ancestor(y, x))


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

    tri_i, tri_j = np.triu_indices(n, k=1)

    for i, j in zip(tri_i, tri_j):
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
    with np.errstate(divide='ignore', invalid='ignore'):
        invD = 1.0 / (D + epsilon)
        np.fill_diagonal(invD, 0.0)
        centrality = invD.sum(axis=1)

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
