import numpy as np
import pytest

from paper.figures.build_confirmation_figure_v6 import display_id, summarize


def fixture_rows():
    return [dict(block_index=b, method=m, condition_id=c, status='success',
                 ad_f1=.2 + .1*b + .2*j, grf_trim=.1 + .1*b)
            for b in range(3) for m in ('M1', 'Pooled-Q')
            for j, c in enumerate(('two', 'five'))]


def test_native_endpoint_and_pairing():
    points, gains = summarize(fixture_rows(), ('M1', 'Pooled-Q'), ('two', 'five'), 'ad_f1', draws=101)
    assert [p['paper_id'] for p in points] == ['M1', 'M1', 'M3', 'M3']
    assert all(p['count'] == 3 for p in points)
    assert all(np.isclose(g['mean'], .2) and np.isclose(g['lower'], .2)
               and np.isclose(g['upper'], .2) for g in gains)
    assert points[0]['mean'] == points[2]['mean']
    assert points[0]['lower'] == points[2]['lower']


def test_alias_does_not_mutate_source_identity():
    rows = fixture_rows()
    summarize(rows, ('Pooled-Q',), ('two', 'five'), 'ad_f1', draws=31)
    assert rows[-1]['method'] == 'Pooled-Q'
    assert display_id('M4') == 'M4'


@pytest.mark.parametrize('damage', ['duplicate', 'missing', 'failed', 'nan', 'range'])
def test_reject_invalid_data(damage):
    rows = fixture_rows()
    if damage == 'duplicate':
        rows.append(dict(rows[0]))
    elif damage == 'missing':
        rows.pop()
    elif damage == 'failed':
        rows[-1]['status'] = 'failure'
    else:
        rows[-1]['ad_f1'] = float('nan') if damage == 'nan' else 1.1
    with pytest.raises(ValueError):
        summarize(rows, ('M1', 'Pooled-Q'), ('two', 'five'), 'ad_f1', draws=11)


def test_trimmed_scores_are_selected_not_shifted():
    points, gains = summarize(fixture_rows(), ('M1',), ('two', 'five'), 'grf_trim', draws=101)
    assert points[0]['mean'] == points[1]['mean']
    assert gains[0]['mean'] == gains[0]['lower'] == gains[0]['upper'] == 0
