from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from algorithm_evaluation import v5_frozen_transition_fixed_radius_report
from algorithm_evaluation.v5_frozen_transition_development_report import (
    REGIME_IDS,
    ResultBundle,
)
from algorithm_evaluation.v5_frozen_transition_fixed_radius_report import (
    CURRENT_NINE_ARM_IDS,
    _contrast_rows,
    _score_change_rows,
    write_report,
)
from algorithm_evaluation.v5_shortlist_robustness_common import (
    ADAPTIVE_A_PRIME_ID,
    ADAPTIVE_B_PRIME_ID,
    BASELINE_SIMULATOR_REGIME,
    CURRENT_PAPER_DEVELOPMENT_ARM_IDS,
    FIXED_RADIUS_CORRECTION_ARM_IDS,
    ORDERED_A_ID,
    ORDERED_B_ID,
    PARTIAL_ADAPTIVE_Y_PRIME_ID,
    PARTIAL_X_ID,
    PARTIAL_Y_ID,
    PLACEMENT_POLICIES,
)


def _metrics(value: float) -> dict[str, float]:
    return {
        "ad_f1": value,
        "grf": value,
        "ad_precision": value,
        "ad_recall": value,
    }


def _bundle(
    *,
    current: bool,
    arm_ids: tuple[str, ...] = CURRENT_NINE_ARM_IDS,
) -> ResultBundle:
    historical_values = {arm_id: 0.30 for arm_id in CURRENT_NINE_ARM_IDS}
    historical_values.update(
        {
            ORDERED_A_ID: 0.40,
            ADAPTIVE_A_PRIME_ID: 0.55,
            ORDERED_B_ID: 0.45,
            ADAPTIVE_B_PRIME_ID: 0.50,
            PARTIAL_X_ID: 0.20,
            PARTIAL_Y_ID: 0.40,
            PARTIAL_ADAPTIVE_Y_PRIME_ID: 0.50,
        }
    )
    current_values = {
        **historical_values,
        ORDERED_A_ID: 0.48,
        ADAPTIVE_A_PRIME_ID: 0.52,
        ORDERED_B_ID: 0.51,
        ADAPTIVE_B_PRIME_ID: 0.49,
        PARTIAL_Y_ID: 0.46,
        PARTIAL_ADAPTIVE_Y_PRIME_ID: 0.47,
    }
    values = current_values if current else historical_values
    records = {}
    for block_index in range(2):
        for placement_policy in PLACEMENT_POLICIES:
            for arm_id in arm_ids:
                records[(block_index, 14, placement_policy, arm_id)] = {
                    "metrics": _metrics(values[arm_id]),
                }
    return ResultBundle(
        regime_id=BASELINE_SIMULATOR_REGIME,
        result_roots=(),
        run_ids=(),
        bank_root=Path("/fixture/bank"),
        bank={
            "heights": [14],
            "cases": [{} for _ in range(2 * len(PLACEMENT_POLICIES))],
        },
        records=records,
        historical=not current,
    )


def test_fixed_radius_rows_pair_all_three_adaptive_contrasts() -> None:
    current = _bundle(current=True)
    historical = _bundle(current=False)

    blocks, placements = _contrast_rows(current, historical)
    by_key = {
        (row["contrast_id"], row["metric"]): row for row in blocks
    }

    assert len(blocks) == 5
    assert by_key[("A_prime_minus_A", "ad_f1")]["current"][
        "mean"
    ] == pytest.approx(0.04)
    assert by_key[("B_prime_minus_B", "ad_f1")][
        "retention_status"
    ] == "historical_positive_contrast_not_retained"
    y_row = by_key[("Y_prime_minus_Y", "grf")]
    assert y_row["historical"]["mean"] == pytest.approx(0.10)
    assert y_row["current"]["mean"] == pytest.approx(0.01)
    assert y_row["correction_shift"]["mean"] == pytest.approx(-0.09)
    assert (
        y_row["current"]["wins"],
        y_row["current"]["ties"],
        y_row["current"]["losses"],
    ) == (2, 0, 0)
    assert len(placements) == len(blocks) * len(PLACEMENT_POLICIES)

    changes = {
        (row["arm_id"], row["metric"]): row[
            "current_minus_historical"
        ]["mean"]
        for row in _score_change_rows(current, historical)
    }
    assert changes[(ORDERED_A_ID, "ad_f1")] == pytest.approx(0.08)
    assert changes[(ADAPTIVE_A_PRIME_ID, "ad_f1")] == pytest.approx(-0.03)
    assert changes[(PARTIAL_Y_ID, "grf")] == pytest.approx(0.06)
    assert changes[(PARTIAL_ADAPTIVE_Y_PRIME_ID, "grf")] == pytest.approx(
        -0.03
    )


def test_fixed_radius_report_separates_families_and_writes_all_tables(
    tmp_path,
    monkeypatch,
) -> None:
    base = {
        regime_id: replace(
            _bundle(
                current=True,
                arm_ids=CURRENT_PAPER_DEVELOPMENT_ARM_IDS,
            ),
            regime_id=regime_id,
        )
        for regime_id in REGIME_IDS
    }
    fixed_radius = {
        regime_id: replace(
            _bundle(
                current=True,
                arm_ids=FIXED_RADIUS_CORRECTION_ARM_IDS,
            ),
            regime_id=regime_id,
        )
        for regime_id in REGIME_IDS
    }
    historical = {
        regime_id: replace(
            _bundle(current=False),
            regime_id=regime_id,
        )
        for regime_id in REGIME_IDS
    }
    monkeypatch.setattr(
        v5_frozen_transition_fixed_radius_report,
        "_load_current_bundle",
        lambda regime_id, _root: base[regime_id],
    )
    monkeypatch.setattr(
        v5_frozen_transition_fixed_radius_report,
        "_load_fixed_radius_bundle",
        lambda regime_id, _root, *, base_current: fixed_radius[regime_id],
    )
    monkeypatch.setattr(
        v5_frozen_transition_fixed_radius_report,
        "_load_historical_bundle",
        lambda regime_id, _roots, **_kwargs: historical[regime_id],
    )

    report = write_report(
        base_current_result_roots={
            regime_id: "base" for regime_id in REGIME_IDS
        },
        fixed_radius_result_roots={
            regime_id: "fixed-radius" for regime_id in REGIME_IDS
        },
        historical_result_roots={
            regime_id: ("historical",) for regime_id in REGIME_IDS
        },
        output_root=tmp_path / "verification",
        created_at_utc="fixture",
    )

    inventory = report["verification_inventory"]
    assert inventory["primary_contrast_cell_count"] == 3 * len(REGIME_IDS)
    assert inventory["all_metric_contrast_cell_count"] == 5 * len(
        REGIME_IDS
    )
    assert inventory["historical_positive_primary_retained_count"] == (
        2 * len(REGIME_IDS)
    )
    assert inventory["current_positive_primary_cell_count"] == (
        2 * len(REGIME_IDS)
    )
    assert {
        row["family"] for row in report["adaptive_minus_fixed_by_height"]
    } == {"fully_labeled", "partial"}
    assert (tmp_path / "verification/report.md").is_file()
    assert (
        tmp_path / "verification/adaptive_minus_fixed_by_height.csv"
    ).is_file()
    assert (
        tmp_path / "verification/adaptive_minus_fixed_by_placement.csv"
    ).is_file()
    assert (tmp_path / "verification/method_score_changes.csv").is_file()
