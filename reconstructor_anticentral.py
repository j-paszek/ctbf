import numpy as np


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
