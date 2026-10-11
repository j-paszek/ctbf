from pathlib import Path

import pytest

from algorithm_evaluation.v5_confirmation_common import (
    M0_ID,
    M2_ID,
    M5_ID,
    M6_ID,
    M7_ID,
    POOLED_Q_ID,
)
from algorithm_evaluation.v5_confirmation_publication import (
    MEDICC2_DEVELOPMENT_ARM_ID,
    conservative_m2_resource_accounting,
    render_main_figure,
)


def _resource(seconds, rss):
    return {
        "wall_time_ns": int(seconds * 1e9),
        "memory": {"peak_rss_bytes": rss},
    }


def test_conservative_m2_resources_include_shared_distance_once_per_block():
    result = conservative_m2_resource_accounting(
        distance_resources={0: _resource(9, 90), 1: _resource(3, 30)},
        m2_resources={
            0: [_resource(1, 20), _resource(1, 100), _resource(1, 40)],
            1: [_resource(1, 20), _resource(1, 40), _resource(1, 50)],
        },
        medicc2_resources={
            0: [_resource(8, 200), _resource(10, 220), _resource(12, 210)],
            1: [_resource(8, 180), _resource(10, 190), _resource(12, 200)],
        },
    )

    block = result["m2_three_standard_cases_plus_full_shared_distance_per_block"]
    amortized = result["m2_conservative_amortized_standard_case"]
    external = result["medicc2_block_mean_standard_case"]

    assert block["wall_seconds"]["mean"] == pytest.approx(9.0)
    assert amortized["wall_seconds"]["mean"] == pytest.approx(3.0)
    assert amortized["peak_rss_bytes"]["maximum"] == 100
    assert external["wall_seconds"]["mean"] == pytest.approx(10.0)
    assert external["peak_rss_bytes"]["maximum"] == 220


def test_confirmation_publication_figure_is_headless(tmp_path: Path):
    partial = (M0_ID, MEDICC2_DEVELOPMENT_ARM_ID, M2_ID)
    closed = (POOLED_Q_ID, M5_ID, M6_ID, M7_ID)
    rows = []
    for family, metric, arms in (
        ("partial", "grf", partial),
        ("closed_state", "ad_f1", closed),
    ):
        for arm_offset, arm_id in enumerate(arms):
            for height in (14, 24, 34):
                mean = 0.2 + arm_offset * 0.05 - height * 0.001
                rows.append(
                    {
                        "family": family,
                        "metric": metric,
                        "arm_id": arm_id,
                        "height": height,
                        "mean": mean,
                        "bootstrap_lower": mean - 0.01,
                        "bootstrap_upper": mean + 0.01,
                    }
                )
    primary_rows = [
        {"complete": "True", "block_effect": str(0.02 + index / 10_000)}
        for index in range(100)
    ]
    report = {
        "primary": {
            "block_effect_summary": {"mean": 0.025},
            "bootstrap_interval": {"lower": 0.02, "upper": 0.03},
        }
    }

    pdf = tmp_path / "figure.pdf"
    png = tmp_path / "figure.png"
    render_main_figure(
        figure_rows=rows,
        primary_rows=primary_rows,
        report=report,
        output_pdf=pdf,
        output_png=png,
    )

    assert pdf.stat().st_size > 1_000
    assert png.stat().st_size > 1_000
