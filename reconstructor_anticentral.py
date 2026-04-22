import numpy as np

from reconstructor_ancestor_selection import _is_biologically_plausible_ancestor
from reconstructor_engine import Orientation, PairChoice


ANTICENTRAL_V3_CONTEXT_KEY = "anticentral_v3_centrality"


def _initial_anticentral_v3_centrality(D):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = 1.0 / (np.mean(D, axis=1) + 1e-9)
    return (c - np.min(c)) / (np.ptp(c) + 1e-12)


def _anticentral_adaptive_v3_score_matrix(D, c, rng, alpha=1.0, beta=1.0, gamma=0.5):
    n = len(D)
    score = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            c_diff = abs(c[i] - c[j])
            adaptive = np.exp(-gamma * c_diff / (np.std(c) + 1e-9))
            anticentral_term = (2 - (c[i] + c[j]))
            score[i, j] = (
                alpha * D[i, j] * (1 + gamma * adaptive)
                - beta * anticentral_term
            )
            score[i, j] += rng.random() * 1e-6
    return score


def _best_pair_from_score(score):
    n = score.shape[0]
    i_best, j_best = divmod(np.argmin(score), n)
    if j_best < i_best:
        i_best, j_best = j_best, i_best
    return i_best, j_best


def _ordered_pairs_by_score(score):
    n = score.shape[0]
    return [
        (i, j)
        for (i, j) in sorted(
            [(i, j) for i in range(n) for j in range(i + 1, n)],
            key=lambda p: score[p[0], p[1]],
        )
    ]


def _copy_parent_internal_node(tree, new_nodes, parent_leaf, child_leaf, next_id, weight):
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
    tree.add_edge(internal_node.node_id, parent_leaf.node_id, weight=float(weight))
    tree.add_edge(internal_node.node_id, child_leaf.node_id, weight=float(weight))

    new_nodes[internal_node] = (parent_leaf, child_leaf)
    return internal_node, next_id


def _merge_parent_child(D, node_list, c, parent_idx, child_idx, internal_node):
    n = len(D)
    keep = [k for k in range(n) if k != child_idx]
    D = D[np.ix_(keep, keep)]
    node_list[parent_idx] = internal_node
    node_list.pop(child_idx)
    c[parent_idx] = np.mean([c[parent_idx], c[child_idx]])
    c = np.delete(c, child_idx)
    return D, c


def _total_deviation_from_baseline(genome, baseline_cn):
    g = np.asarray(genome, dtype=float)
    return float(np.sum(np.abs(g - baseline_cn)))


def configure_anticentral_v3_state(state):
    state.context[ANTICENTRAL_V3_CONTEXT_KEY] = _initial_anticentral_v3_centrality(state.D)


def make_anticentral_adaptive_v3_pair_selector(alpha=1.0, beta=1.0, gamma=0.5):
    def select_pair(state):
        c = state.context[ANTICENTRAL_V3_CONTEXT_KEY]
        score = _anticentral_adaptive_v3_score_matrix(state.D, c, state.rng, alpha, beta, gamma)
        i_best, j_best = _best_pair_from_score(score)
        return PairChoice(i_best, j_best, score=score[i_best, j_best])

    return select_pair


def make_anticentral_adaptive_v3_skip_unplausible_pair_selector(alpha=1.0, beta=1.0, gamma=0.5):
    def select_pair(state):
        c = state.context[ANTICENTRAL_V3_CONTEXT_KEY]
        score = _anticentral_adaptive_v3_score_matrix(state.D, c, state.rng, alpha, beta, gamma)
        pair_order = _ordered_pairs_by_score(score)

        for i, j in pair_order:
            a = state.node_list[i]
            b = state.node_list[j]
            can_i_parent_j = _is_biologically_plausible_ancestor(a, b)
            can_j_parent_i = _is_biologically_plausible_ancestor(b, a)

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
        c_dir = state.D.sum(axis=1)
        inv_D = 1.0 / (state.D + epsilon)
        np.fill_diagonal(inv_D, 0.0)
        c_inv = inv_D.sum(axis=1)
        c_mix = c_inv / np.power(c_dir + epsilon, lam)

        score = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = state.D[i, j]
                if not np.isfinite(d_ij):
                    continue
                asym = abs(c_mix[i] - c_mix[j])
                score[i, j] = alpha * d_ij + beta * asym

        i, j = divmod(np.argmax(score), n)
        if j < i:
            i, j = j, i

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

        c_dir = state.D.sum(axis=1)
        inv_D = 1.0 / (state.D + epsilon)
        np.fill_diagonal(inv_D, 0.0)
        c_inv = inv_D.sum(axis=1)
        c_mix = c_inv / np.power(c_dir + epsilon, lam)

        score = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = state.D[i, j]
                if not np.isfinite(d_ij):
                    continue
                asym = abs(c_mix[i] - c_mix[j])
                score[i, j] = a * d_ij + b * asym - g / (d_ij + epsilon)

        i, j = divmod(np.argmax(score), n)
        if j < i:
            i, j = j, i

        return PairChoice(i, j, score=score[i, j], metadata={"c_mix": c_mix})

    return select_pair


def keep_pair_order_parent_selector(state, pair):
    return Orientation(pair.i, pair.j)


def pair_choice_orientation_selector(state, pair):
    return pair.metadata["orientation"]


def less_mixed_centrality_parent_selector(state, pair):
    c_mix = pair.metadata["c_mix"]
    if c_mix[pair.i] <= c_mix[pair.j]:
        return Orientation(pair.i, pair.j)
    return Orientation(pair.j, pair.i)


def _anticentral_centrality_orientation(state, i, j):
    c = state.context[ANTICENTRAL_V3_CONTEXT_KEY]
    if c[i] > c[j]:
        return Orientation(i, j)
    if c[j] > c[i]:
        return Orientation(j, i)
    if state.rng.random() < 0.5:
        return Orientation(i, j)
    return Orientation(j, i)


def make_plausible_pair_order_parent_selector(enforce_plausibility=True):
    def select_parent(state, pair):
        parent_idx = pair.i
        child_idx = pair.j

        if enforce_plausibility:
            parent = state.node_list[parent_idx]
            child = state.node_list[child_idx]
            can_parent_child = _is_biologically_plausible_ancestor(parent, child)
            can_child_parent = _is_biologically_plausible_ancestor(child, parent)

            if can_child_parent and not can_parent_child:
                parent_idx, child_idx = child_idx, parent_idx

        return Orientation(parent_idx, child_idx)

    return select_parent


def make_plausible_parsimony_parent_selector(baseline_cn=2):
    def select_parent(state, pair):
        i = pair.i
        j = pair.j
        a = state.node_list[i]
        b = state.node_list[j]

        can_a_parent_b = _is_biologically_plausible_ancestor(a, b)
        can_b_parent_a = _is_biologically_plausible_ancestor(b, a)

        if can_a_parent_b and not can_b_parent_a:
            return Orientation(i, j)

        if can_b_parent_a and not can_a_parent_b:
            return Orientation(j, i)

        if can_a_parent_b and can_b_parent_a:
            dev_a = _total_deviation_from_baseline(a.genome, baseline_cn)
            dev_b = _total_deviation_from_baseline(b.genome, baseline_cn)

            if dev_a < dev_b:
                return Orientation(i, j)

            if dev_b < dev_a:
                return Orientation(j, i)

        return _anticentral_centrality_orientation(state, i, j)

    return select_parent


def anticentral_weighted_copy_parent_node(state, orientation):
    parent_leaf = state.node_list[orientation.parent_idx]
    child_leaf = state.node_list[orientation.child_idx]
    weight = state.D[orientation.parent_idx, orientation.child_idx]
    internal_node, state.next_id = _copy_parent_internal_node(
        state.tree,
        state.new_nodes,
        parent_leaf,
        child_leaf,
        state.next_id,
        weight,
    )
    state.origin_index[internal_node] = state.origin_index[parent_leaf]
    return internal_node


def anticentral_v3_distance_update(state, orientation, internal_node):
    c = state.context[ANTICENTRAL_V3_CONTEXT_KEY]
    state.D, state.context[ANTICENTRAL_V3_CONTEXT_KEY] = _merge_parent_child(
        state.D,
        state.node_list,
        c,
        orientation.parent_idx,
        orientation.child_idx,
        internal_node,
    )
