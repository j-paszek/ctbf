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
    _best_pair_from_score_matrix,
    _ordered_pairs_by_score_matrix,
)
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
