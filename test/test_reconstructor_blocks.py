import random

import numpy as np
import pytest

from reconstructor_ancestor_selection import (
    keep_pair_order_parent_selector,
    lower_sum_distance_parent_selector,
    make_plausible_pair_order_parent_selector,
    more_central_parent_selector,
)
from reconstructor_distance_update import (
    ANTICENTRAL_V3_CONTEXT_KEY,
    anticentral_v3_distance_update,
    drop_child_keep_parent_update,
)
from reconstructor_engine import Orientation, PairChoice, initialize_reconstruction_state
from reconstructor_merge import (
    copy_parent_equal_weight_internal_node,
    copy_parent_internal_node,
    copy_parent_without_new_node_record,
)
from reconstructor_pair_selection import (
    _anticentral_adaptive_v3_score_matrix,
    _best_pair_from_score_matrix,
    _ordered_pairs_by_score_matrix,
    make_hybrid_opt_pair_selector,
)
from reconstructor_metrics import hybrid_opt_centrality, nj_q_matrix
from simulator import Genotype


def _cells(n):
    return [Genotype([2, 2, i], i + 1) for i in range(n)]


def _state(n=3):
    D = np.array(
        [
            [0.0, 2.0, 5.0],
            [2.0, 0.0, 3.0],
            [5.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    return initialize_reconstruction_state(D[:n, :n], _cells(n), max_id=n, seed=7)


def _anticentral_adaptive_v3_score_matrix_loop_reference(D, c, rng, alpha=1.0, beta=1.0, gamma=0.5):
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


def _hybrid_opt_pair_selector_loop_reference(
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
        for i, j in [(i, j) for i in range(n) for j in range(i + 1, n)]:
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


def _best_pair_from_score_matrix_loop_reference(score, minimize=True):
    best_pair = None
    best_score = None

    for i, j in [(i, j) for i in range(score.shape[0]) for j in range(i + 1, score.shape[0])]:
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


def _assert_pair_choice_matches(actual, expected):
    assert actual[:2] == expected[:2]
    if np.isnan(expected[2]):
        assert np.isnan(actual[2])
    else:
        assert actual[2] == expected[2]


def test_score_matrix_pair_helpers_choose_and_order_pairs():
    score = np.array(
        [
            [np.inf, 4.0, 1.0],
            [np.inf, np.inf, 2.0],
            [np.inf, np.inf, np.inf],
        ]
    )

    assert _best_pair_from_score_matrix(score) == (0, 2, 1.0)
    assert _best_pair_from_score_matrix(score, minimize=False) == (0, 1, 4.0)
    assert _ordered_pairs_by_score_matrix(score) == [(0, 2), (1, 2), (0, 1)]


@pytest.mark.parametrize(
    "score,minimize",
    [
        (
            np.array(
                [
                    [np.inf, 2.0, 1.0],
                    [0.0, np.inf, 1.0],
                    [np.inf, np.inf, np.inf],
                ]
            ),
            True,
        ),
        (
            np.array(
                [
                    [np.inf, 2.0, 4.0],
                    [10.0, np.inf, 4.0],
                    [np.inf, np.inf, np.inf],
                ]
            ),
            False,
        ),
        (np.full((3, 3), np.inf), True),
        (np.full((3, 3), np.inf), False),
        (np.full((3, 3), -np.inf), True),
        (np.full((3, 3), -np.inf), False),
        (
            np.array(
                [
                    [np.inf, np.nan, 1.0],
                    [np.inf, np.inf, 2.0],
                    [np.inf, np.inf, np.inf],
                ]
            ),
            True,
        ),
        (
            np.array(
                [
                    [np.inf, np.nan, 1.0],
                    [np.inf, np.inf, 2.0],
                    [np.inf, np.inf, np.inf],
                ]
            ),
            False,
        ),
        (
            np.array(
                [
                    [np.inf, 2.0, np.nan],
                    [np.inf, np.inf, 1.0],
                    [np.inf, np.inf, np.inf],
                ]
            ),
            True,
        ),
        (
            np.array(
                [
                    [np.inf, 2.0, np.nan],
                    [np.inf, np.inf, 3.0],
                    [np.inf, np.inf, np.inf],
                ]
            ),
            False,
        ),
    ],
)
def test_best_pair_from_score_matrix_matches_loop_reference_for_edge_values(score, minimize):
    expected = _best_pair_from_score_matrix_loop_reference(score, minimize=minimize)
    actual = _best_pair_from_score_matrix(score, minimize=minimize)

    _assert_pair_choice_matches(actual, expected)


def test_best_pair_from_score_matrix_raises_without_upper_triangle_pairs():
    with pytest.raises(ValueError):
        _best_pair_from_score_matrix(np.full((1, 1), np.inf))


@pytest.mark.parametrize("n", [1, 2, 8, 30])
def test_anticentral_adaptive_v3_score_matrix_matches_loop_reference(n):
    rng = np.random.default_rng(123)
    D = rng.random((n, n))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    c = rng.random(n)
    expected_rng = random.Random(99)
    actual_rng = random.Random(99)

    expected = _anticentral_adaptive_v3_score_matrix_loop_reference(
        D,
        c,
        expected_rng,
        alpha=1.3,
        beta=0.7,
        gamma=0.4,
    )
    actual = _anticentral_adaptive_v3_score_matrix(
        D,
        c,
        actual_rng,
        alpha=1.3,
        beta=0.7,
        gamma=0.4,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert actual_rng.random() == expected_rng.random()


def test_anticentral_adaptive_v3_score_matrix_matches_loop_reference_with_zero_std_centrality():
    D = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    c = np.ones(3)
    expected_rng = random.Random(7)
    actual_rng = random.Random(7)

    expected = _anticentral_adaptive_v3_score_matrix_loop_reference(D, c, expected_rng)
    actual = _anticentral_adaptive_v3_score_matrix(D, c, actual_rng)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert actual_rng.random() == expected_rng.random()


@pytest.mark.parametrize(
    "use_q_as_secondary, reverse_centrality",
    [
        (True, False),
        (False, False),
        (True, True),
    ],
)
def test_hybrid_opt_pair_selector_matches_loop_reference(use_q_as_secondary, reverse_centrality):
    D = np.array(
        [
            [0.0, 4.0, 4.5, 7.0],
            [4.0, 0.0, 2.0, 6.0],
            [4.5, 2.0, 0.0, 3.0],
            [7.0, 6.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    state = initialize_reconstruction_state(D, _cells(4), max_id=4, seed=7)
    kwargs = {
        "alpha": 1.1,
        "beta": 0.6,
        "gamma": 0.25,
        "epsilon": 1e-8,
        "k": 8.0,
        "tau": 0.45,
        "reverse_centrality": reverse_centrality,
        "use_q_as_secondary": use_q_as_secondary,
    }

    expected = _hybrid_opt_pair_selector_loop_reference(**kwargs)(state)
    actual = make_hybrid_opt_pair_selector(**kwargs)(state)

    assert (actual.i, actual.j) == (expected.i, expected.j)
    assert actual.score == expected.score
    np.testing.assert_allclose(actual.metadata["centrality"], expected.metadata["centrality"])
    if expected.metadata["Q"] is None:
        assert actual.metadata["Q"] is None
    else:
        np.testing.assert_allclose(actual.metadata["Q"], expected.metadata["Q"])


def test_parent_selectors_use_pair_metadata_metrics():
    state = _state()
    pair = PairChoice(0, 1, metadata={"centrality": np.array([2.0, 1.0, 3.0])})

    assert more_central_parent_selector(state, pair) == Orientation(0, 1)
    assert lower_sum_distance_parent_selector(state, pair) == Orientation(1, 0)
    assert keep_pair_order_parent_selector(state, pair) == Orientation(0, 1)


def test_plausible_pair_order_parent_selector_flips_impossible_direction():
    state = initialize_reconstruction_state(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        [Genotype([0], 1), Genotype([1], 2)],
        max_id=2,
        seed=7,
    )
    pair = PairChoice(0, 1)

    selector = make_plausible_pair_order_parent_selector(enforce_plausibility=True)

    assert selector(state, pair) == Orientation(1, 0)


@pytest.mark.parametrize(
    "merge_strategy, expected_parent_weight, expected_child_weight, expected_new_nodes",
    [
        (copy_parent_internal_node, 0.0, 2.0, 1),
        (copy_parent_equal_weight_internal_node, 2.0, 2.0, 1),
        (copy_parent_without_new_node_record, 0.0, 2.0, 0),
    ],
)
def test_copy_parent_merge_strategies_create_expected_edges(
    merge_strategy,
    expected_parent_weight,
    expected_child_weight,
    expected_new_nodes,
):
    state = _state(n=2)
    orientation = Orientation(0, 1)

    internal_node = merge_strategy(state, orientation)

    assert internal_node.node_id == 3
    assert internal_node.cell_id == state.node_list[0].cell_id
    assert state.tree.edges[internal_node.node_id, state.node_list[0].node_id]["weight"] == expected_parent_weight
    assert state.tree.edges[internal_node.node_id, state.node_list[1].node_id]["weight"] == expected_child_weight
    assert len(state.new_nodes) == expected_new_nodes


def test_drop_child_keep_parent_update_replaces_parent_and_removes_child():
    state = _state()
    orientation = Orientation(0, 1)
    internal_node = copy_parent_internal_node(state, orientation)

    drop_child_keep_parent_update(state, orientation, internal_node)

    assert state.D.shape == (2, 2)
    assert state.node_list[0] is internal_node
    assert [node.node_id for node in state.node_list] == [4, 3]


def test_anticentral_v3_distance_update_keeps_context_aligned():
    state = _state()
    state.context[ANTICENTRAL_V3_CONTEXT_KEY] = np.array([0.2, 0.8, 0.4])
    orientation = Orientation(0, 1)
    internal_node = copy_parent_equal_weight_internal_node(state, orientation)

    anticentral_v3_distance_update(state, orientation, internal_node)

    assert state.D.shape == (2, 2)
    assert state.node_list[0] is internal_node
    assert np.allclose(state.context[ANTICENTRAL_V3_CONTEXT_KEY], np.array([0.5, 0.4]))
