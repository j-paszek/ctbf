"""Small, data-free regressions for the matched partial-comparator figure."""
from copy import deepcopy
import json

import pytest

from paper.figures.build_confirmation_figure_v5 import (
    HEIGHTS, METHODS, METRICS, case_scores, digest, numeric_records,
    partial_points, protected_hashes,
)


def rows():
    return [{'block_index': b, 'condition_id': f'H{h}_rule_y',
             'method': m, 'status': 'success', 'grf_full': .1 + b * .03,
             'grf_trim': .2 + b * .03}
            for b in range(3) for h in HEIGHTS for m in METHODS
            if h != 34 or m != 'NestedBD-Long']


def test_matching_drops_failed_case_for_all_methods_and_both_targets():
    source = rows()
    failure = next(r for r in source if r['block_index'] == 1 and
                   r['condition_id'] == 'H24_rule_y' and r['method'] == 'NestedBD-Long')
    failure.update(status='incomplete', grf_full=None, grf_trim=None)
    before = deepcopy(source)
    points = partial_points(source, draws=97)
    assert source == before
    assert len(points) == 28
    for r in points:
        assert r['count'] == (2 if r['height'] == 24 else 3)
        assert r['planned'] == 3
        assert r['block_indices'] == ([0, 2] if r['height'] == 24 else [0, 1, 2])
    assert not any(r['paper_id'] == 'NestedBD-Long' and r['height'] == 34 for r in points)


def test_shared_history_resamples_for_methods_and_truth_targets():
    points = partial_points(rows(), draws=197)
    for h in HEIGHTS:
        for metric in METRICS:
            subset = [r for r in points if r['height'] == h and r['metric'] == metric]
            for key in ('mean', 'lower', 'upper'):
                assert len({r[key] for r in subset}) == 1
        full = next(r for r in points if r['height'] == h and r['metric'] == 'grf_full')
        trim = next(r for r in points if r['height'] == h and r['metric'] == 'grf_trim')
        for key in ('mean', 'lower', 'upper'):
            assert trim[key] - full[key] == pytest.approx(.1)


def test_seed_failure_is_not_replaced_by_successful_mate():
    base = {'case_id': 'a', 'condition_id': 'H14_rule_y', 'block_index': 0,
            'height': 14, 'problem': 'partial', 'method': 'NestedBD-Long'}
    pair = [{**base, 'replicate': 1, 'status': 'success', 'grf_full': .7, 'grf_trim': .8},
            {**base, 'replicate': 2, 'status': 'failure', 'grf_full': None, 'grf_trim': None}]
    result = case_scores(pair)[0]
    assert result['status'] == 'incomplete' and result['grf_full'] is None
    pair[1].update(status='success', grf_full=.3, grf_trim=.4)
    assert case_scores(pair)[0]['grf_full'] == pytest.approx(.5)


def test_duplicate_rows_rejected():
    source = rows()
    with pytest.raises(ValueError, match='Duplicate'):
        partial_points(source + [source[0]], draws=5)


@pytest.mark.parametrize('value', ['nan', 'inf', '-0.01', '1.01'])
def test_invalid_successful_score_rejected(value):
    with pytest.raises(ValueError, match='finite GRF'):
        numeric_records([{'status': 'success', 'grf_full': value, 'grf_trim': '.2'}])


def test_failed_score_cannot_be_imputed():
    with pytest.raises(ValueError, match='imputed'):
        numeric_records([{'status': 'failure', 'grf_full': '0', 'grf_trim': '0'}])


def test_only_sealed_declared_operational_repair_is_accepted(tmp_path):
    source = '/repo/algorithm_evaluation/v5_partial_comparator_extension.py'
    contract = {'protected_sha256': {source: 'old', '/repo/science.py': 'science'}}
    repair = {
        'schema_version': 'ctbf-v5-partial-comparator-resume-repair-v1',
        'original_contract_sha256': 'contract',
        'scope': 'terminal_outcome_reuse_and_selected_budget_reporting',
        'scientific_rules_and_results_unchanged': True,
        'source_replacements': {source: {'before': 'old', 'after': 'new'}},
    }
    path = tmp_path / 'resume_repair_v1.json'
    path.write_text(json.dumps(repair))
    marker = {'contract_sha256': 'contract',
              'artifacts': {'resume_repair_v1.json': digest(path)}}
    assert protected_hashes(tmp_path, contract, marker) == {
        source: 'new', '/repo/science.py': 'science'}
    assert contract['protected_sha256'][source] == 'old'

    path.write_text(json.dumps({**repair, 'scope': 'different'}))
    with pytest.raises(ValueError, match='Sealed source changed'):
        protected_hashes(tmp_path, contract, marker)
    marker['artifacts']['resume_repair_v1.json'] = digest(path)
    with pytest.raises(ValueError, match='Unrecognized operational'):
        protected_hashes(tmp_path, contract, marker)

    repair['source_replacements'] = {'/repo/science.py': {'before': 'science', 'after': 'new'}}
    path.write_text(json.dumps(repair))
    marker['artifacts']['resume_repair_v1.json'] = digest(path)
    with pytest.raises(ValueError, match='original source declaration'):
        protected_hashes(tmp_path, contract, marker)

    repair['source_replacements'] = {source: {'before': 'other', 'after': 'new'}}
    path.write_text(json.dumps(repair))
    marker['artifacts']['resume_repair_v1.json'] = digest(path)
    with pytest.raises(ValueError, match='original source declaration'):
        protected_hashes(tmp_path, contract, marker)
