from __future__ import annotations

from pathlib import Path

import pytest

from algorithm_evaluation.v5_shortlist_robustness_common import HEIGHTS
from poster2026.generate_v5_development_figure import (
    ALL_CONTRASTS,
    ContrastResult,
    _format_delta,
    render_figure,
    summarize_block_deltas,
    write_results_csv,
)


def test_block_summary_averages_placements_before_wtl() -> None:
    policies = ("spread", "late", "random")
    condition_deltas = {
        (0, "spread"): 1.0,
        (0, "late"): -1.0,
        (0, "random"): 3.0,
        (1, "spread"): -1.0,
        (1, "late"): 1.0,
        (1, "random"): -3.0,
        (2, "spread"): 1.0,
        (2, "late"): -1.0,
        (2, "random"): 0.0,
    }

    summary = summarize_block_deltas(
        condition_deltas,
        block_indices=(0, 1, 2),
        placement_policies=policies,
    )

    assert summary["block_deltas"] == pytest.approx((1.0, -1.0, 0.0))
    assert summary["mean_delta"] == pytest.approx(0.0)
    assert summary["wins"] == 1
    assert summary["ties"] == 1
    assert summary["losses"] == 1


def test_block_summary_rejects_missing_placement() -> None:
    with pytest.raises(ValueError, match="Missing paired condition delta"):
        summarize_block_deltas(
            {(0, "spread"): 1.0},
            block_indices=(0,),
            placement_policies=("spread", "late"),
        )


def test_delta_formatter_preserves_small_component_effects() -> None:
    assert _format_delta(0.12331) == "+0.123"
    assert _format_delta(-0.01143) == "-0.011"
    assert _format_delta(0.000689) == "+0.0007"


def test_vector_raster_and_data_outputs_render(tmp_path: Path) -> None:
    results = tuple(
        ContrastResult(
            spec=spec,
            height=height,
            block_count=100,
            mean_delta=(row_index + 1) * 0.004 * (1 if height < 34 else -1),
            median_delta=0.0,
            minimum_delta=-0.1,
            maximum_delta=0.1,
            wins=60,
            ties=5,
            losses=35,
        )
        for row_index, spec in enumerate(ALL_CONTRASTS)
        for height in HEIGHTS
    )
    output_pdf = tmp_path / "figure.pdf"
    output_png = tmp_path / "figure.png"
    output_csv = tmp_path / "figure.csv"

    render_figure(results, output_pdf=output_pdf, output_png=output_png)
    write_results_csv(results, output_csv)

    assert output_pdf.stat().st_size > 10_000
    assert output_png.stat().st_size > 10_000
    assert output_csv.read_text(encoding="utf-8").count("\n") == 33
