import numpy as np


def upper_triangle_pairs(n):
    return zip(*np.triu_indices(n, k=1))


def sum_distance_centrality(D):
    return D.sum(axis=1)


def inverse_distance_centrality(D, epsilon, include_diagonal=False):
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_D = 1.0 / (D + epsilon)
    if not include_diagonal:
        np.fill_diagonal(inv_D, 0.0)
    return inv_D.sum(axis=1)


def mixed_direct_inverse_centrality(D, epsilon, lam):
    c_dir = sum_distance_centrality(D)
    c_inv = inverse_distance_centrality(D, epsilon)
    return c_inv / np.power(c_dir + epsilon, lam)


def nj_q_matrix(D):
    n = D.shape[0]
    Q = np.full((n, n), np.inf, dtype=float)
    if n <= 2:
        return Q

    total = sum_distance_centrality(D)
    factor = n - 2
    tri_i, tri_j = np.triu_indices(n, k=1)
    q_values = factor * D[tri_i, tri_j] - total[tri_i] - total[tri_j]
    Q[tri_i, tri_j] = q_values
    Q[tri_j, tri_i] = q_values
    return Q


def score_distance_minus_asymmetry(D, centrality, alpha, beta):
    n = len(D)
    score = np.full((n, n), np.inf)
    for i, j in upper_triangle_pairs(n):
        score[i, j] = alpha * D[i, j] - beta * abs(centrality[i] - centrality[j])
    return score


def linear_blended_centrality(state, epsilon):
    n = len(state.D)
    original_n = len(state.D_full)
    weight = (n - 2) / max(original_n - 2, 1)
    sum_dist = sum_distance_centrality(state.D)
    global_c = 1.0 / (sum_dist + epsilon)
    inv_c = inverse_distance_centrality(state.D, epsilon)
    return weight * global_c + (1 - weight) * inv_c


def sigmoid_blended_centrality(state, epsilon, k, tau):
    frac = len(state.D) / max(len(state.D_full), 1)
    weight = 1.0 / (1.0 + np.exp(-k * (frac - tau)))
    sum_dist = sum_distance_centrality(state.D)
    global_c = 1.0 / (sum_dist + epsilon)
    inv_c = inverse_distance_centrality(state.D, epsilon)
    return weight * global_c + (1 - weight) * inv_c


def reversed_adaptive_centrality(state, epsilon):
    weight = 1 - (len(state.D) / len(state.D_full))
    sum_dist = sum_distance_centrality(state.D)
    inv_sum = inverse_distance_centrality(state.D, epsilon, include_diagonal=True)
    return (1 - weight) * inv_sum + weight * (1.0 / (sum_dist + epsilon))


def hybrid_opt_centrality(state, epsilon, k, tau, reverse_centrality):
    frac = len(state.D) / max(len(state.D_full), 1)
    weight = 1.0 / (1.0 + np.exp(-k * (frac - tau))) if k > 0 else frac

    sum_dist = sum_distance_centrality(state.D)
    global_c = 1.0 / (sum_dist + epsilon)
    inv_c = inverse_distance_centrality(state.D, epsilon)

    if reverse_centrality:
        return (1 - weight) * global_c + weight * inv_c
    return weight * global_c + (1 - weight) * inv_c


def normalized_inverse_mean_centrality(D):
    with np.errstate(divide='ignore', invalid='ignore'):
        centrality = 1.0 / (np.mean(D, axis=1) + 1e-9)
    return (centrality - np.min(centrality)) / (np.ptp(centrality) + 1e-12)


def row_sum_anticentrality(D, index):
    return sum_distance_centrality(D)[index]


__all__ = [
    "hybrid_opt_centrality",
    "inverse_distance_centrality",
    "linear_blended_centrality",
    "mixed_direct_inverse_centrality",
    "nj_q_matrix",
    "normalized_inverse_mean_centrality",
    "reversed_adaptive_centrality",
    "row_sum_anticentrality",
    "score_distance_minus_asymmetry",
    "sigmoid_blended_centrality",
    "sum_distance_centrality",
    "upper_triangle_pairs",
]
