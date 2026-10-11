"""Focused checks for complementary GRF summaries, without frozen patient/run data."""
import numpy as np
import pytest

from paper.figures.build_confirmation_figure_v4 import (
    COLUMNS, method_label, summarize_closed_grf,
)


def test_pooled_nj_label_is_explicit():
    assert method_label('M0') == r'$M_0$ (cnp2cnp–NJ)'
    assert method_label('M6') == r'$M_6$'
    assert method_label('Pooled-Q') == 'Pooled-Q'


def test_closed_grf_keeps_history_pairing_across_heights():
    matrix = np.zeros((3, len(COLUMNS)))
    for j, (name, height) in enumerate(COLUMNS):
        matrix[:, j] = np.array([.1, .2, .3]) + height / 1000
        if name in ('M6', 'M7'):
            matrix[:, j] += .1
    before = matrix.copy()
    points, contrasts = summarize_closed_grf(matrix, draws=127)
    assert len(points) == 15 and len(contrasts) == 8
    assert np.array_equal(matrix, before)
    for row in contrasts:
        assert row['count'] == 3  # not three heights x three independent histories
        assert row['mean'] == pytest.approx(.1)
        assert row['lower'] == pytest.approx(.1)
        assert row['upper'] == pytest.approx(.1)
        assert (row['wins'], row['ties'], row['losses']) == (3, 0, 0)
    assert all(row['metric'] == 'grf' for row in points)


@pytest.mark.parametrize('bad', [float('nan'), -0.1, 1.1])
def test_closed_grf_rejects_invalid_scores(bad):
    values = np.zeros((3, len(COLUMNS)))
    values[0, 0] = bad
    with pytest.raises(ValueError, match='GRF scores'):
        summarize_closed_grf(values, draws=5)
