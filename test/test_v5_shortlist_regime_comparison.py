from __future__ import annotations

import pytest

from algorithm_evaluation.v5_shortlist_regime_comparison import (
    _adaptive_features,
    _complete_blocks_from_inventory,
    _summarize_paired_values,
)
from reconstructor_biopsy_blocks import (
    ADAPTIVE_BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
    FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY,
)


def test_complete_blocks_require_every_placement_available() -> None:
    inventory = [
        {
            "block_index": block_index,
            "height": 24,
            "placement_policy": policy,
            "status": status,
        }
        for block_index, declarations in (
            (0, (("spread", "available"), ("late", "available"), ("random", "available"))),
            (1, (("spread", "available"), ("late", "available"))),
            (2, (("spread", "available"), ("late", "unavailable"), ("random", "available"))),
        )
        for policy, status in declarations
    ]

    assert _complete_blocks_from_inventory(inventory, height=24) == (0,)


def test_paired_summary_distinguishes_conditions_from_blocks() -> None:
    policies = ("spread", "late")
    baseline = {
        (0, 14, "spread"): 10.0,
        (0, 14, "late"): 20.0,
        (1, 14, "spread"): 30.0,
        (1, 14, "late"): 40.0,
    }
    sensitivity = {
        (0, 14, "spread"): 11.0,
        (0, 14, "late"): 23.0,
        (1, 14, "spread"): 29.0,
        (1, 14, "late"): 37.0,
    }

    summary = _summarize_paired_values(
        baseline,
        sensitivity,
        blocks=(0, 1),
        height=14,
        policies=policies,
        require_complete=True,
    )

    assert summary["paired_condition_count"] == 4
    assert summary["paired_block_count"] == 2
    assert summary["condition_delta"]["wins"] == 2
    assert summary["condition_delta"]["losses"] == 2
    assert summary["block_delta"]["minimum"] == pytest.approx(-2.0)
    assert summary["block_delta"]["maximum"] == pytest.approx(2.0)
    assert summary["block_delta"]["mean"] == pytest.approx(0.0)


def test_paired_summary_rejects_incomplete_declared_block() -> None:
    baseline = {
        (0, 14, "spread"): 1.0,
        (0, 14, "late"): 2.0,
    }
    sensitivity = {(0, 14, "spread"): 1.0}

    with pytest.raises(ValueError, match="missing coordinates"):
        _summarize_paired_values(
            baseline,
            sensitivity,
            blocks=(0,),
            height=14,
            policies=("spread", "late"),
            require_complete=True,
        )


def test_adaptive_features_aggregate_transition_and_decision_counts() -> None:
    record = {
        "reconstruction_metadata": {
            "biopsy_layer_decision_audit": {
                "schema_version": ADAPTIVE_BIOPSY_GUIDED_AUDIT_SCHEMA_VERSION,
                "parent_eligibility_policy": (
                    FROZEN_TRANSITION_PARENT_ELIGIBILITY_POLICY
                ),
                "child_decision_count": 10,
                "copy_up_count": 3,
                "transition_records": [
                    {
                        "effective_radius": 1,
                        "child_snapshot_count": 4,
                        "frozen_radius_candidate_covered_child_count": 3,
                    },
                    {
                        "effective_radius": 3,
                        "child_snapshot_count": 6,
                        "frozen_radius_candidate_covered_child_count": 5,
                    },
                ],
            }
        }
    }

    features = _adaptive_features(record)

    assert features["mean_effective_radius"] == pytest.approx(2.0)
    assert features["radius_gt1_transition_fraction"] == pytest.approx(0.5)
    assert features["copy_up_fraction"] == pytest.approx(0.3)
    assert features["frozen_candidate_coverage_fraction"] == pytest.approx(0.8)
