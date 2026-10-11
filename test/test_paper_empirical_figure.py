"""Small provenance regressions, without patient data or reconstruction runs."""
import pytest

from paper.figures.build_ktn152_empirical_figure_v2 import (
    high_copy_counts, observed_node_biopsies, spaced_positions,
)


def test_recurrent_state_colours_follow_occurrences_not_sorted_state_queues():
    payload = {'levels': [
        {'generation': 0, 'states': [{'state_label': 'a', 'cnp': [2, 2]}]},
        {'generation': 1, 'states': [{'state_label': 'b', 'cnp': [3, 2]}]},
        {'generation': 2, 'states': [{'state_label': 'b', 'cnp': [3, 2]}]},
    ]}
    labels = {'entries': [
        {'external_state_label': 'a', 'internal_state_label': 0},
        {'external_state_label': 'b', 'internal_state_label': 1},
    ]}
    tree = {'nodes': [
        {'id': 0, 'cell_id': 'a', 'genome': [2, 2]},
        {'id': 1, 'cell_id': 'b', 'genome': [3, 2]},
        {'id': 2, 'cell_id': 'b', 'genome': [3, 2]},
    ], 'links': [{'source': 0, 'target': 2}, {'source': 2, 'target': 1}]}
    assert observed_node_biopsies(payload, labels, tree) == {0: 0, 2: 1, 1: 2}
    tree['links'] = [{'source': 0, 'target': 1}, {'source': 1, 'target': 2}]
    with pytest.raises(ValueError, match='violates biopsy order'):
        observed_node_biopsies(payload, labels, tree)


def test_short_chromosome_labels_do_not_overlap():
    actual = [20, 80, 112, 120, 129, 138, 180]
    labels = spaced_positions(actual, 0, 200, 24)
    assert all(b-a >= 24 for a,b in zip(labels, labels[1:]))
    assert labels[0] >= 0 and labels[-1] <= 200
    assert actual == [20, 80, 112, 120, 129, 138, 180]


def test_eight_is_distinguished_from_above_the_pinned_medicc_cap():
    counts = high_copy_counts([[2,8,9], [4,7,40]])
    assert counts['cn_ge_8'] == {'entries': 3, 'profiles': 2, 'bins': 2}
    assert counts['cn_ge_9'] == {'entries': 2, 'profiles': 2, 'bins': 1}
    assert counts['entry_count'] == 6
    assert counts['maximum_copy_number'] == 40
