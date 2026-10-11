from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from algorithm_evaluation import v5_frozen_transition_development_run
from algorithm_evaluation import v5_frozen_transition_development_report
from algorithm_evaluation.v5_frozen_transition_development_run import (
    _assert_existing_report_values_preserved,
)
from algorithm_evaluation.v5_frozen_transition_development_report import (
    REGIME_IDS,
    ResultBundle,
    _comparator_rows,
    _complete_block_means,
    _contrast_rows,
    _effect_summary,
    _score_change_rows,
    _validate_current_audits,
    write_report,
)
from algorithm_evaluation.v5_shortlist_robustness_common import (
    ADAPTIVE_A_PRIME_ID,
    ADAPTIVE_B_PRIME_ID,
    BASELINE_SIMULATOR_REGIME,
    CURRENT_PAPER_DEVELOPMENT_ARM_IDS,
    PARTIAL_ADAPTIVE_Y_PRIME_ID,
    PARTIAL_X_ID,
    PLACEMENT_POLICIES,
    POOLED_D_ID,
    POOLED_E_ID,
)
from reconstructor_biopsy_blocks import (
    BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
    FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY,
)


def _metrics(value: float) -> dict[str, float]:
    return {
        "ad_f1": value,
        "grf": value,
        "ad_precision": value,
        "ad_recall": value,
    }


def _bundle(*, current: bool) -> ResultBundle:
    historical_values = {
        POOLED_E_ID: 0.30,
        POOLED_D_ID: 0.40,
        ADAPTIVE_A_PRIME_ID: 0.60,
        ADAPTIVE_B_PRIME_ID: 0.55,
        PARTIAL_X_ID: 0.20,
        PARTIAL_ADAPTIVE_Y_PRIME_ID: 0.50,
    }
    current_values = {
        **historical_values,
        ADAPTIVE_A_PRIME_ID: 0.55,
        ADAPTIVE_B_PRIME_ID: 0.35,
        PARTIAL_ADAPTIVE_Y_PRIME_ID: 0.45,
    }
    values = current_values if current else historical_values
    records = {}
    for block_index in range(2):
        for placement_policy in PLACEMENT_POLICIES:
            for arm_id in CURRENT_PAPER_DEVELOPMENT_ARM_IDS:
                records[(block_index, 14, placement_policy, arm_id)] = {
                    "metrics": _metrics(values[arm_id]),
                }
    return ResultBundle(
        regime_id=BASELINE_SIMULATOR_REGIME,
        result_roots=(),
        run_ids=(),
        bank_root=Path("."),
        bank={"heights": [14]},
        records=records,
        historical=not current,
    )


def test_effect_summary_reports_wins_ties_and_losses() -> None:
    summary = _effect_summary((1.0, 0.0, -2.0, 3.0))

    assert summary["mean"] == pytest.approx(0.5)
    assert (summary["wins"], summary["ties"], summary["losses"]) == (2, 1, 1)


def test_expanded_report_must_preserve_every_existing_value() -> None:
    cell = {"arm_id": "E", "metrics": {"ad_f1": {"mean": 0.4}}}
    pair = {"arm_a": "E", "arm_b": "D", "effect": {"mean": -0.1}}
    depth = {"arm_a": "E", "arm_b": "D", "effect": {"mean": 0.02}}
    placement = {
        "arm_a": "E",
        "arm_b": "D",
        "effect": {"mean": -0.03},
    }
    base = {
        "comparison_groups": {
            "fully_labeled": {
                "arm_ids": ["E", "D"],
                "algorithm_cell_summaries": [cell],
                "pairwise_by_height_and_placement": [pair],
                "depth_interactions": [depth],
                "placement_interactions": [placement],
            }
        },
        "condition_diagnostics": [{"height": 14}],
        "adaptive_radius_diagnostics": [{"arm_id": "A-prime"}],
    }
    expanded = {
        **base,
        "comparison_groups": {
            "fully_labeled": {
                "arm_ids": ["Pooled-Q", "E", "D"],
                "algorithm_cell_summaries": [
                    {"arm_id": "Pooled-Q", "metrics": {}},
                    cell,
                ],
                "pairwise_by_height_and_placement": [
                    {"arm_a": "Pooled-Q", "arm_b": "E", "effect": {}},
                    pair,
                ],
                "depth_interactions": [
                    {"arm_a": "Pooled-Q", "arm_b": "E", "effect": {}},
                    depth,
                ],
                "placement_interactions": [
                    {"arm_a": "Pooled-Q", "arm_b": "E", "effect": {}},
                    placement,
                ],
            }
        },
    }
    _assert_existing_report_values_preserved(base, expanded)

    changed = {
        **expanded,
        "comparison_groups": {
            "fully_labeled": {
                **expanded["comparison_groups"]["fully_labeled"],
                "algorithm_cell_summaries": [
                    {"arm_id": "Pooled-Q", "metrics": {}},
                    {"arm_id": "E", "metrics": {"ad_f1": {"mean": 0.5}}},
                ],
            }
        },
    }
    with pytest.raises(ValueError, match="changed existing report values"):
        _assert_existing_report_values_preserved(base, changed)


def test_add_pooled_q_flag_selects_only_additive_pipeline(
    monkeypatch,
    capsys,
) -> None:
    calls = []

    def fake_extension(**kwargs):
        calls.append(kwargs)
        return {"record_count": 17, "failure_count": 0}

    monkeypatch.setattr(
        v5_frozen_transition_development_run,
        "run_pooled_q_extension",
        fake_extension,
    )
    monkeypatch.setattr(
        v5_frozen_transition_development_run,
        "run_pipeline",
        lambda **_kwargs: pytest.fail("six-arm pipeline must not run"),
    )

    assert v5_frozen_transition_development_run.main(
        ["--add-pooled-q", "--record-workers", "3"]
    ) == 0
    assert len(calls) == 1
    assert calls[0]["record_workers"] == 3
    assert "17 arm records" in capsys.readouterr().out


def test_complete_block_means_require_every_placement() -> None:
    values = {
        (0, 14, "spread"): 1.0,
        (0, 14, "late"): 2.0,
        (0, 14, "random"): 3.0,
        (1, 14, "spread"): 100.0,
    }

    assert _complete_block_means(values, height=14) == {0: pytest.approx(2.0)}


def test_paired_report_rows_separate_retained_and_lost_contrasts() -> None:
    current = _bundle(current=True)
    historical = _bundle(current=False)

    contrasts, placements = _contrast_rows(current, historical)
    by_id = {
        (row["contrast_id"], row["metric"]): row
        for row in contrasts
    }
    assert by_id[("Y_prime_minus_X", "grf")]["retention_status"] == (
        "positive_contrast_retained"
    )
    assert by_id[("B_prime_minus_D", "ad_f1")]["retention_status"] == (
        "historical_positive_contrast_not_retained"
    )
    assert by_id[("A_prime_minus_D", "ad_f1")]["correction_shift"][
        "mean"
    ] == pytest.approx(-0.05)
    assert len(placements) == len(contrasts) * len(PLACEMENT_POLICIES)

    score_rows = _score_change_rows(current, historical)
    y_change = next(
        row
        for row in score_rows
        if row["arm_id"] == PARTIAL_ADAPTIVE_Y_PRIME_ID
        and row["metric"] == "grf"
    )
    assert y_change["current_minus_historical"]["mean"] == pytest.approx(-0.05)

    comparator_rows = _comparator_rows(current, historical)
    assert comparator_rows
    assert all(row["exact_within_tolerance"] for row in comparator_rows)


def test_current_audit_validation_requires_frozen_transition_policy() -> None:
    key = (0, 14, "spread", ADAPTIVE_A_PRIME_ID)
    record = {
        "reconstruction_metadata": {
            "biopsy_layer_decision_audit": {
                "schema_version": BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
                "parent_eligibility_policy": (
                    FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
                ),
            }
        }
    }
    _validate_current_audits({key: record})

    broken = {
        **record,
        "reconstruction_metadata": {
            "biopsy_layer_decision_audit": {
                "schema_version": BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
                "parent_eligibility_policy": "online",
            }
        },
    }
    with pytest.raises(ValueError, match="wrong parent policy"):
        _validate_current_audits({key: broken})


def test_complete_verification_report_keeps_regimes_separate(
    tmp_path,
    monkeypatch,
) -> None:
    current = {
        regime_id: replace(_bundle(current=True), regime_id=regime_id)
        for regime_id in REGIME_IDS
    }
    historical = {
        regime_id: replace(_bundle(current=False), regime_id=regime_id)
        for regime_id in REGIME_IDS
    }
    monkeypatch.setattr(
        v5_frozen_transition_development_report,
        "_load_current_bundle",
        lambda regime_id, _root: current[regime_id],
    )
    monkeypatch.setattr(
        v5_frozen_transition_development_report,
        "_load_historical_bundle",
        lambda regime_id, _roots, *, current: historical[regime_id],
    )

    report = write_report(
        current_result_roots={regime_id: "current" for regime_id in REGIME_IDS},
        historical_result_roots={
            regime_id: ("historical",) for regime_id in REGIME_IDS
        },
        output_root=tmp_path / "verification",
        created_at_utc="fixture",
    )

    assert report["verification_inventory"]["primary_contrast_cell_count"] == 20
    assert {row["regime_id"] for row in report["block_contrasts"]} == set(
        REGIME_IDS
    )
    assert (tmp_path / "verification/report.md").is_file()
    assert (tmp_path / "verification/block_contrasts.csv").is_file()
