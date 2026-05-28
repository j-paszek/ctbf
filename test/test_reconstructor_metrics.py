import numpy as np
import pytest

from reconstructor_engine import initialize_reconstruction_state
from reconstructor_metrics import (
    inverse_distance_centrality,
    linear_blended_centrality,
    mixed_direct_inverse_centrality,
    nj_q_matrix,
    normalized_inverse_mean_centrality,
    row_sum_anticentrality,
    score_distance_minus_asymmetry,
    sum_distance_centrality,
    upper_triangle_pairs,
)
from simulator import Genotype


def _matrix():
    return np.array(
        [
            [0.0, 2.0, 4.0],
            [2.0, 0.0, 3.0],
            [4.0, 3.0, 0.0],
        ],
        dtype=float,
    )


def _nj_q_matrix_loop_reference(D):
    n = D.shape[0]
    Q = np.full((n, n), np.inf, dtype=float)
    if n <= 2:
        return Q

    total = sum_distance_centrality(D)
    factor = n - 2
    for i in range(n):
        for j in range(i + 1, n):
            q_val = factor * D[i, j] - total[i] - total[j]
            Q[i, j] = q_val
            Q[j, i] = q_val
    return Q


def test_upper_triangle_pairs_excludes_diagonal():
    assert list(upper_triangle_pairs(3)) == [(0, 1), (0, 2), (1, 2)]


def test_distance_centrality_metrics_are_stable():
    D = _matrix()

    assert np.array_equal(sum_distance_centrality(D), np.array([6.0, 5.0, 7.0]))
    assert np.allclose(
        inverse_distance_centrality(D, epsilon=1e-6),
        np.array([0.7499998125000547, 0.8333334722221714, 0.5833331736111597]),
    )
    assert np.allclose(
        mixed_direct_inverse_centrality(D, epsilon=1e-6, lam=1.0),
        inverse_distance_centrality(D, epsilon=1e-6) / (sum_distance_centrality(D) + 1e-6),
    )


def test_nj_q_matrix_and_distance_asymmetry_score():
    D = _matrix()
    centrality = np.array([1.0, 3.0, 2.0])

    assert np.array_equal(
        nj_q_matrix(D),
        np.array(
            [
                [np.inf, -9.0, -9.0],
                [-9.0, np.inf, -9.0],
                [-9.0, -9.0, np.inf],
            ]
        ),
    )
    assert np.array_equal(
        score_distance_minus_asymmetry(D, centrality, alpha=1.0, beta=0.5),
        np.array(
            [
                [np.inf, 1.0, 3.5],
                [np.inf, np.inf, 2.5],
                [np.inf, np.inf, np.inf],
            ]
        ),
    )


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_nj_q_matrix_matches_loop_reference(n):
    rng = np.random.default_rng(123)
    D = rng.random((n, n))
    np.fill_diagonal(D, 0.0)

    np.testing.assert_allclose(nj_q_matrix(D), _nj_q_matrix_loop_reference(D), rtol=0.0, atol=0.0)


def test_adaptive_and_anticentral_metric_helpers():
    D = _matrix()
    cells = [Genotype([2], i + 1) for i in range(3)]
    state = initialize_reconstruction_state(D, cells, max_id=3, seed=7)

    assert linear_blended_centrality(state, epsilon=1e-6).shape == (3,)
    assert normalized_inverse_mean_centrality(D).shape == (3,)
    assert row_sum_anticentrality(D, 2) == 7.0
