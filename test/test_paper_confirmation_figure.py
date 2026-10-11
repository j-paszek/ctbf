"""Small checks for the read-only confirmation-figure derivative."""
import copy

import numpy as np
import pytest

from paper.figures.build_confirmation_figure_v3 import (
    audit_identity, biopsy_value, bootstrap_means,
)


def test_identity_includes_bottom_not_only_shared_top_callable():
    spec = {'arm_id': 'guided', 'problem': 'closed',
            'algorithm_name': 'shared_top', 'input_mode': 'ordered',
            'only_nj': False, 'biopsy_preset': 'deferred_tie',
            'radius_policy': 'median', 'top_output_projection': 'none'}
    record = {'arm_id': 'guided', 'problem': 'closed', 'case_id': 'case',
              'reconstruction_metadata': {
                  'algorithm': 'shared_top', 'input_mode': 'ordered',
                  'only_nj': False, 'biopsy_preset': 'deferred_tie',
                  'radius_policy': 'median', 'top_output_projection': 'none'}}
    audit_identity(record, spec)
    changed = copy.deepcopy(record)
    changed['reconstruction_metadata']['input_mode'] = 'pooled'
    with pytest.raises(ValueError, match='input_mode'):
        audit_identity(changed, spec)


def test_native_partial_metric_is_not_called_fixed_label():
    assert biopsy_value({'metrics': {'grf': .2}, 'fixed_label_evaluation': None}, 'M2') == (.2, None)
    with pytest.raises(ValueError, match='Unexpected fixed-label'):
        biopsy_value({'fixed_label_evaluation': {'metrics': {'ad_f1': .9}}}, 'M2')


def test_closed_state_uses_bilateral_fixed_target_not_native_f1():
    record = {'metrics': {'ad_f1': .9}, 'fixed_label_evaluation': {
        'status': 'success', 'fixed_labels': ['a','b'],
        'semantics': {'endpoint_filter': 'bilateral_both_endpoints_in_fixed_labels'},
        'metrics': {'ad_f1': .6, 'counts': {'true_unique_pair_count': 2}}}}
    assert biopsy_value(record, 'M6') == (.6, (('a','b'), 2))


def test_bootstrap_preserves_block_pairing_and_is_repeatable():
    values = np.array([[.1,.3],[.5,.7],[.6,.8]])
    draws = bootstrap_means(values, draws=127)
    assert np.array_equal(draws, bootstrap_means(values, draws=127))
    assert np.allclose(draws[:,1]-draws[:,0], .2)
