"""Shared contract for the reusable CTBF v5 algorithm-development bank.

This module deliberately declares the v5 candidate roster directly.  It does
not obtain candidate identities or ordering from the legacy benchmark
registry.  The bank is development evidence only; paper execution uses a
later disjoint protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import wraps
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from algorithm_evaluation.paper_pipeline_contract import (
    json_safe,
    read_json,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import (
    RECONSTRUCTION_INPUT_SCHEMA_VERSION,
    deserialize_tree,
    serialize_tree,
    validate_reconstruction_input,
)
from ctbs import DistanceMatrix
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    stable_distance_label_key,
    validate_distance_label_coverage,
)
from reconstructor import build_evolution_tree
from reconstructor_algorithms import (
    make_nj_full_cps_variant,
    make_nj_full_variant,
    make_nj_hybrid_inv_cent_variant,
    make_nj_hybrid_variant,
    neighbor_joining_adaptive_centrality,
    neighbor_joining_adaptive_centrality_nonlinear,
    neighbor_joining_adaptive_centrality_reversed,
    neighbor_joining_anticentral_parent_reuse,
    neighbor_joining_baseline,
    neighbor_joining_classical,
    neighbor_joining_hybrid_anticentral_adaptive_v3,
    neighbor_joining_hybrid_anticentral_adaptive_v3_plausible,
    neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony,
    neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible,
    neighbor_joining_hybrid_anticentral_opt,
    neighbor_joining_hybrid_opt,
    neighbor_joining_hybrid_opt_adaptive,
    neighbor_joining_hybrid_opt_refined,
    neighbor_joining_hybrid_opt_v2,
    new_alg,
    rooted_labeled_nj,
    temporal_cnp_arborescence,
    temporal_cnp_arborescence_no_time,
)
from reconstructor_biopsy_presets import resolve_biopsy_guided_config
from reconstructor_biopsy_blocks import (
    ADAPTIVE_MEDIAN_PRIOR_NN_RADIUS_POLICY,
    BiopsyGuidedDecisionAudit,
)
from simulator import Genotype


BANK_SCHEMA_VERSION = "ctbf-v5-algorithm-development-bank-v3"
LEGACY_BANK_SCHEMA_VERSION = "ctbf-v5-algorithm-development-bank-v2"
BANK_MANIFEST_NAME = "bank_manifest.json"
BANK_CONFIG_NAME = "simulator_config.json"
CASE_INPUT_SCHEMA_VERSION = RECONSTRUCTION_INPUT_SCHEMA_VERSION
CASE_DISTANCE_SCHEMA_VERSION = "ctbf-v5-algorithm-development-distance-v1"
CASE_METADATA_SCHEMA_VERSION = "ctbf-v5-algorithm-development-case-v1"
RUN_SCHEMA_VERSION = "ctbf-v5-algorithm-development-run-v3"
LEGACY_RUN_SCHEMA_VERSION = "ctbf-v5-algorithm-development-run-v2"
REPORT_SCHEMA_VERSION = "ctbf-v5-algorithm-development-report-v8"

# Retain the seed namespace deliberately: v2 extends the established 50-block
# stream with block indices 50--99 while its bank/run schemas prevent artifact
# mixing. This is scientific pairing, not a compatibility mode.
DEVELOPMENT_NAMESPACE = "ctbf-v5-g1-06-existing-screen-v1"
DEFAULT_BANK_ID = "ctbf-v5-g1-06-development-bank-v2"
DEFAULT_BLOCK_COUNT = 100
DEFAULT_BASE_SEED = 20260813
HEIGHT_SCHEDULES = {
    14: (9, 12, 14),
    24: (15, 20, 24),
    34: (21, 28, 34),
}
TARGET_FRACTION = 0.5
BIOPSY_LOWER_BOUND = 6
SAMPLING_RULE = "min(N,max(6,ceil(0.5*N)))"

PARTIAL_FAMILY = "partial"
INFERRED_COPY_FAMILY = "inferred_copy"
BIOPSY_GUIDED_FULL_FAMILY = "biopsy_guided_full"
CONTEXT_FAMILY = "contextual_reference"
COMPARISON_FAMILIES = (
    PARTIAL_FAMILY,
    INFERRED_COPY_FAMILY,
    BIOPSY_GUIDED_FULL_FAMILY,
)

TOP_OUTPUT_PROJECTION_NONE = "none"
TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED = (
    "created_top_nodes_unlabeled"
)
TOP_OUTPUT_PROJECTIONS = {
    TOP_OUTPUT_PROJECTION_NONE,
    TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
}

PARTIAL_BOTTOM_CONTROL_ID = "biopsy_guided_top_anticentral_binary_r2"
PARTIAL_BOTTOM_CANDIDATE_ROLE = "bottom_reconstruction_candidate"
PARTIAL_BOTTOM_TOP_INTERACTION_ARM_ID = (
    "biopsy_guided_classical_r2_bottom_deferred_tie"
)
PARTIAL_BOTTOM_TOP_INTERACTION_ROLE = "bottom_top_interaction_candidate"

PARTIAL_INCUMBENT_ID = "biopsy_guided_classical_r4"
INFERRED_COPY_INCUMBENT_ID = (
    "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony"
)
BIOPSY_GUIDED_FULL_BASELINE_ID = (
    "biopsy_guided_full_rooted_labeled_q_r2"
)
BIOPSY_GUIDED_FULL_DEFAULT_ID = (
    "biopsy_guided_full_anticentral_binary_r2"
)
BIOPSY_GUIDED_FULL_INCUMBENT_ID = (
    "biopsy_guided_full_anticentral_binary_r2_bottom_deferred_tie"
)
BIOPSY_GUIDED_FULL_ROOTED_Q_DEFERRED_ID = (
    "biopsy_guided_full_rooted_labeled_q_r2_bottom_deferred_tie"
)
BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_DEFERRED_ID = (
    "biopsy_guided_full_anticentral_binary_adaptive_median_prior_nn_"
    "bottom_deferred_tie"
)
BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_ROOTED_Q_DEFERRED_ID = (
    "biopsy_guided_full_rooted_labeled_q_adaptive_median_prior_nn_"
    "bottom_deferred_tie"
)
BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_DEFAULT_ID = (
    "biopsy_guided_full_anticentral_binary_adaptive_median_prior_nn_"
    "bottom_default"
)
BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_ROOTED_Q_DEFAULT_ID = (
    "biopsy_guided_full_rooted_labeled_q_adaptive_median_prior_nn_"
    "bottom_default"
)
BIOPSY_GUIDED_FULL_BOTTOM_TOP_INTERACTION_ROLE = (
    "rooted_q_bottom_top_interaction_candidate"
)
INCUMBENT_BY_FAMILY = {
    PARTIAL_FAMILY: PARTIAL_INCUMBENT_ID,
    INFERRED_COPY_FAMILY: INFERRED_COPY_INCUMBENT_ID,
    BIOPSY_GUIDED_FULL_FAMILY: BIOPSY_GUIDED_FULL_INCUMBENT_ID,
}
BASELINE_BY_FAMILY = {
    PARTIAL_FAMILY: "classical_partial",
    INFERRED_COPY_FAMILY: "neighbor_joining_baseline",
    BIOPSY_GUIDED_FULL_FAMILY: BIOPSY_GUIDED_FULL_BASELINE_ID,
}


@dataclass(frozen=True)
class DevelopmentArmSpec:
    arm_id: str
    algorithm_name: str
    family: str
    problem: str
    input_mode: str
    only_nj: bool
    radius: float | None
    primary_metric: str
    complementary_metrics: tuple[str, ...]
    role: str
    biopsy_preset: str | None = None
    top_output_projection: str = TOP_OUTPUT_PROJECTION_NONE
    radius_policy: str | None = None

    def as_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["complementary_metrics"] = list(self.complementary_metrics)
        if self.radius_policy is None:
            value.pop("radius_policy")
        return value


def _partial_arm(
    arm_id: str,
    *,
    radius: float,
    role: str,
    biopsy_preset: str | None = None,
    only_nj: bool = False,
    algorithm_name: str = "neighbor_joining_classical",
    top_output_projection: str = TOP_OUTPUT_PROJECTION_NONE,
) -> DevelopmentArmSpec:
    if top_output_projection not in TOP_OUTPUT_PROJECTIONS:
        raise ValueError(
            f"Unknown top-output projection {top_output_projection!r}."
        )
    return DevelopmentArmSpec(
        arm_id=arm_id,
        algorithm_name=algorithm_name,
        family=PARTIAL_FAMILY,
        problem="partial",
        input_mode="pooled" if only_nj else "ordered",
        only_nj=only_nj,
        radius=float(radius),
        primary_metric="grf",
        complementary_metrics=(),
        role=role,
        biopsy_preset=biopsy_preset,
        top_output_projection=top_output_projection,
    )


def _inferred_arm(algorithm_name: str, role: str = "candidate") -> DevelopmentArmSpec:
    return DevelopmentArmSpec(
        arm_id=algorithm_name,
        algorithm_name=algorithm_name,
        family=INFERRED_COPY_FAMILY,
        problem="inferred_copy_fully_labeled_closed_state",
        input_mode="pooled",
        only_nj=True,
        radius=4.0,
        primary_metric="ad_f1",
        complementary_metrics=("grf", "ad_precision", "ad_recall"),
        role=role,
    )


def _biopsy_guided_full_arm(
    arm_id: str,
    *,
    radius: float | None,
    role: str,
    biopsy_preset: str,
    algorithm_name: str,
    radius_policy: str | None = None,
) -> DevelopmentArmSpec:
    """Declare an ordered, fully labeled closed-state reconstruction arm."""
    return DevelopmentArmSpec(
        arm_id=arm_id,
        algorithm_name=algorithm_name,
        family=BIOPSY_GUIDED_FULL_FAMILY,
        problem="biopsy_guided_occurrence_aware_fully_labeled_closed_state",
        input_mode="ordered",
        only_nj=False,
        radius=None if radius is None else float(radius),
        primary_metric="ad_f1",
        complementary_metrics=("grf", "ad_precision", "ad_recall"),
        role=role,
        biopsy_preset=biopsy_preset,
        top_output_projection=TOP_OUTPUT_PROJECTION_NONE,
        radius_policy=radius_policy,
    )


def _context_arm(
    arm_id: str,
    algorithm_name: str,
    *,
    problem: str,
    input_mode: str,
    only_nj: bool,
    role: str,
) -> DevelopmentArmSpec:
    return DevelopmentArmSpec(
        arm_id=arm_id,
        algorithm_name=algorithm_name,
        family=CONTEXT_FAMILY,
        problem=problem,
        input_mode=input_mode,
        only_nj=only_nj,
        radius=4.0,
        primary_metric="ad_f1",
        complementary_metrics=("grf", "ad_precision", "ad_recall"),
        role=role,
    )


INITIAL_ARM_SPECS: tuple[DevelopmentArmSpec, ...] = (
    _partial_arm("classical_partial", radius=4, role="baseline", only_nj=True),
    _partial_arm(PARTIAL_INCUMBENT_ID, radius=4, role="incumbent", biopsy_preset="default"),
    _partial_arm(
        "biopsy_guided_classical_r2",
        radius=2,
        role="radius_sensitivity",
        biopsy_preset="default",
    ),
    _partial_arm(
        "biopsy_guided_classical_r8",
        radius=8,
        role="radius_sensitivity",
        biopsy_preset="default",
    ),
    _partial_arm(
        "biopsy_preset_anticentral_tie_r4",
        radius=4,
        role="preset_candidate",
        biopsy_preset="anticentral_tie",
    ),
    _partial_arm(
        "biopsy_preset_binarized_r4",
        radius=4,
        role="preset_candidate",
        biopsy_preset="binarized",
    ),
    _partial_arm(
        "biopsy_preset_anticentral_binarized_r4",
        radius=4,
        role="preset_candidate",
        biopsy_preset="anticentral_binarized",
    ),
    _inferred_arm("neighbor_joining_baseline", role="baseline"),
    _inferred_arm("neighbor_joining_full_full"),
    _inferred_arm("neighbor_joining_full_partial"),
    _inferred_arm("neighbor_joining_full_cps_full"),
    _inferred_arm("neighbor_joining_full_cps_partial"),
    _inferred_arm("neighbor_joining_hybrid_full"),
    _inferred_arm("neighbor_joining_hybrid_partial"),
    _inferred_arm("neighbor_joining_hybrid_inverse_centrality_full"),
    _inferred_arm("neighbor_joining_hybrid_inverse_centrality_partial"),
    _inferred_arm("neighbor_joining_adaptive_centrality"),
    _inferred_arm("neighbor_joining_adaptive_centrality_nonlinear"),
    _inferred_arm("neighbor_joining_adaptive_centrality_reversed"),
    _inferred_arm("neighbor_joining_hybrid_opt"),
    _inferred_arm("neighbor_joining_hybrid_opt_adaptive"),
    _inferred_arm("neighbor_joining_hybrid_opt_v2"),
    _inferred_arm("neighbor_joining_hybrid_opt_refined"),
    _inferred_arm("neighbor_joining_hybrid_anticentral_opt"),
    _inferred_arm("neighbor_joining_hybrid_anticentral_adaptive_v3"),
    _inferred_arm("neighbor_joining_hybrid_anticentral_adaptive_v3_plausible"),
    _inferred_arm("neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible"),
    _inferred_arm(INFERRED_COPY_INCUMBENT_ID, role="incumbent"),
    _inferred_arm("new_alg"),
    _context_arm(
        "rooted_labeled_nj",
        "rooted_labeled_nj",
        problem="fully_labeled_closed_state",
        input_mode="pooled",
        only_nj=True,
        role="closed_state_baseline",
    ),
    _context_arm(
        "temporal_minimum",
        "temporal_cnp_arborescence",
        problem="occurrence_aware_fully_labeled_closed_state",
        input_mode="ordered",
        only_nj=False,
        role="temporal_reference",
    ),
    _context_arm(
        "temporal_minimum_no_time",
        "temporal_cnp_arborescence_no_time",
        problem="occurrence_aware_fully_labeled_closed_state",
        input_mode="ordered",
        only_nj=False,
        role="temporal_ablation",
    ),
)

# Owner-directed X-to-Y refinements use new stable ids.  The initial 32-arm
# meaning of ``--arms all`` remains immutable.
PARTIAL_TOP_EXTENSION_ARM_SPECS: tuple[DevelopmentArmSpec, ...] = (
    _partial_arm(
        "biopsy_guided_top_rooted_labeled_q_r2",
        radius=2,
        role="top_reconstruction_candidate",
        biopsy_preset="default",
        algorithm_name="rooted_labeled_nj",
        top_output_projection=TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    ),
    _partial_arm(
        "biopsy_guided_top_anticentral_binary_r2",
        radius=2,
        role="top_reconstruction_candidate",
        biopsy_preset="default",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
        top_output_projection=TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    ),
    _partial_arm(
        "biopsy_guided_top_anticentral_parent_reuse_r2",
        radius=2,
        role="top_reconstruction_candidate",
        biopsy_preset="default",
        algorithm_name="neighbor_joining_anticentral_parent_reuse",
        top_output_projection=TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    ),
    _partial_arm(
        "biopsy_guided_top_anticentral_binary_r4",
        radius=4,
        role="top_radius_interaction_candidate",
        biopsy_preset="default",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
        top_output_projection=TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    ),
)

PARTIAL_BOTTOM_EXTENSION_ARM_SPECS: tuple[DevelopmentArmSpec, ...] = (
    _partial_arm(
        "biopsy_guided_top_anticentral_binary_r2_bottom_anticentral_tie",
        radius=2,
        role=PARTIAL_BOTTOM_CANDIDATE_ROLE,
        biopsy_preset="anticentral_tie",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
        top_output_projection=TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    ),
    _partial_arm(
        "biopsy_guided_top_anticentral_binary_r2_bottom_binarized",
        radius=2,
        role=PARTIAL_BOTTOM_CANDIDATE_ROLE,
        biopsy_preset="binarized",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
        top_output_projection=TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    ),
    _partial_arm(
        "biopsy_guided_top_anticentral_binary_r2_bottom_anticentral_tie_binarized",
        radius=2,
        role=PARTIAL_BOTTOM_CANDIDATE_ROLE,
        biopsy_preset="anticentral_tie_binarized",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
        top_output_projection=TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    ),
    _partial_arm(
        "biopsy_guided_top_anticentral_binary_r2_bottom_deferred_tie",
        radius=2,
        role=PARTIAL_BOTTOM_CANDIDATE_ROLE,
        biopsy_preset="deferred_tie",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
        top_output_projection=TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    ),
    _partial_arm(
        "biopsy_guided_top_anticentral_binary_r2_bottom_central_tie",
        radius=2,
        role=PARTIAL_BOTTOM_CANDIDATE_ROLE,
        biopsy_preset="central_tie",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
        top_output_projection=TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    ),
    _partial_arm(
        "biopsy_guided_top_anticentral_binary_r2_bottom_diploid_parsimony_tie",
        radius=2,
        role=PARTIAL_BOTTOM_CANDIDATE_ROLE,
        biopsy_preset="diploid_parsimony_tie",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
        top_output_projection=TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED,
    ),
)

PARTIAL_BOTTOM_TOP_EXTENSION_ARM_SPECS: tuple[DevelopmentArmSpec, ...] = (
    _partial_arm(
        PARTIAL_BOTTOM_TOP_INTERACTION_ARM_ID,
        radius=2,
        role=PARTIAL_BOTTOM_TOP_INTERACTION_ROLE,
        biopsy_preset="deferred_tie",
    ),
)

BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS: tuple[DevelopmentArmSpec, ...] = (
    _biopsy_guided_full_arm(
        BIOPSY_GUIDED_FULL_BASELINE_ID,
        radius=2,
        role="baseline",
        biopsy_preset="default",
        algorithm_name="rooted_labeled_nj",
    ),
    _biopsy_guided_full_arm(
        BIOPSY_GUIDED_FULL_DEFAULT_ID,
        radius=2,
        role="default_bottom_control",
        biopsy_preset="default",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
    ),
    _biopsy_guided_full_arm(
        "biopsy_guided_full_anticentral_parent_reuse_r2",
        radius=2,
        role="top_reconstruction_candidate",
        biopsy_preset="default",
        algorithm_name="neighbor_joining_anticentral_parent_reuse",
    ),
    _biopsy_guided_full_arm(
        "biopsy_guided_full_anticentral_binary_r4",
        radius=4,
        role="radius_candidate",
        biopsy_preset="default",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
    ),
    _biopsy_guided_full_arm(
        "biopsy_guided_full_anticentral_binary_r2_bottom_anticentral_tie",
        radius=2,
        role="bottom_reconstruction_candidate",
        biopsy_preset="anticentral_tie",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
    ),
    _biopsy_guided_full_arm(
        "biopsy_guided_full_anticentral_binary_r2_bottom_binarized",
        radius=2,
        role="bottom_reconstruction_candidate",
        biopsy_preset="binarized",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
    ),
    _biopsy_guided_full_arm(
        "biopsy_guided_full_anticentral_binary_r2_bottom_anticentral_tie_binarized",
        radius=2,
        role="bottom_reconstruction_candidate",
        biopsy_preset="anticentral_tie_binarized",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
    ),
    _biopsy_guided_full_arm(
        BIOPSY_GUIDED_FULL_INCUMBENT_ID,
        radius=2,
        role="incumbent",
        biopsy_preset="deferred_tie",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
    ),
    _biopsy_guided_full_arm(
        "biopsy_guided_full_anticentral_binary_r2_bottom_central_tie",
        radius=2,
        role="bottom_reconstruction_candidate",
        biopsy_preset="central_tie",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
    ),
    _biopsy_guided_full_arm(
        "biopsy_guided_full_anticentral_binary_r2_bottom_diploid_parsimony_tie",
        radius=2,
        role="bottom_reconstruction_candidate",
        biopsy_preset="diploid_parsimony_tie",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
    ),
)

# This owner-approved cell completes the fully labeled
# {default,deferred-bottom} x {rooted-labeled-Q,binary-anticentral-top}
# attribution design.  Keep it outside the completed ten-arm exact-counterpart
# roster so the historical ``--arms biopsy_guided_full`` alias stays immutable.
BIOPSY_GUIDED_FULL_BOTTOM_TOP_EXTENSION_ARM_SPECS: tuple[
    DevelopmentArmSpec, ...
] = (
    _biopsy_guided_full_arm(
        BIOPSY_GUIDED_FULL_ROOTED_Q_DEFERRED_ID,
        radius=2,
        role=BIOPSY_GUIDED_FULL_BOTTOM_TOP_INTERACTION_ROLE,
        biopsy_preset="deferred_tie",
        algorithm_name="rooted_labeled_nj",
    ),
)

ADAPTIVE_RADIUS_EXTENSION_ARM_SPECS: tuple[DevelopmentArmSpec, ...] = (
    _biopsy_guided_full_arm(
        BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_DEFERRED_ID,
        radius=None,
        role="adaptive_radius_candidate",
        biopsy_preset="deferred_tie",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
        radius_policy=ADAPTIVE_MEDIAN_PRIOR_NN_RADIUS_POLICY,
    ),
    _biopsy_guided_full_arm(
        BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_ROOTED_Q_DEFERRED_ID,
        radius=None,
        role="adaptive_radius_top_interaction_candidate",
        biopsy_preset="deferred_tie",
        algorithm_name="rooted_labeled_nj",
        radius_policy=ADAPTIVE_MEDIAN_PRIOR_NN_RADIUS_POLICY,
    ),
    _biopsy_guided_full_arm(
        BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_DEFAULT_ID,
        radius=None,
        role="adaptive_radius_bottom_interaction_candidate",
        biopsy_preset="default",
        algorithm_name=INFERRED_COPY_INCUMBENT_ID,
        radius_policy=ADAPTIVE_MEDIAN_PRIOR_NN_RADIUS_POLICY,
    ),
    _biopsy_guided_full_arm(
        BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_ROOTED_Q_DEFAULT_ID,
        radius=None,
        role="adaptive_radius_factorial_candidate",
        biopsy_preset="default",
        algorithm_name="rooted_labeled_nj",
        radius_policy=ADAPTIVE_MEDIAN_PRIOR_NN_RADIUS_POLICY,
    ),
)

BIOPSY_GUIDED_FULL_COUNTERPART_BY_PARTIAL_ID = {
    "biopsy_guided_top_rooted_labeled_q_r2": BIOPSY_GUIDED_FULL_BASELINE_ID,
    "biopsy_guided_top_anticentral_binary_r2": BIOPSY_GUIDED_FULL_DEFAULT_ID,
    "biopsy_guided_top_anticentral_parent_reuse_r2": (
        "biopsy_guided_full_anticentral_parent_reuse_r2"
    ),
    "biopsy_guided_top_anticentral_binary_r4": (
        "biopsy_guided_full_anticentral_binary_r4"
    ),
    "biopsy_guided_top_anticentral_binary_r2_bottom_anticentral_tie": (
        "biopsy_guided_full_anticentral_binary_r2_bottom_anticentral_tie"
    ),
    "biopsy_guided_top_anticentral_binary_r2_bottom_binarized": (
        "biopsy_guided_full_anticentral_binary_r2_bottom_binarized"
    ),
    "biopsy_guided_top_anticentral_binary_r2_bottom_anticentral_tie_binarized": (
        "biopsy_guided_full_anticentral_binary_r2_bottom_anticentral_tie_binarized"
    ),
    "biopsy_guided_top_anticentral_binary_r2_bottom_deferred_tie": (
        BIOPSY_GUIDED_FULL_INCUMBENT_ID
    ),
    "biopsy_guided_top_anticentral_binary_r2_bottom_central_tie": (
        "biopsy_guided_full_anticentral_binary_r2_bottom_central_tie"
    ),
    "biopsy_guided_top_anticentral_binary_r2_bottom_diploid_parsimony_tie": (
        "biopsy_guided_full_anticentral_binary_r2_bottom_diploid_parsimony_tie"
    ),
}

DEVELOPMENT_EXTENSION_ARM_SPECS: tuple[DevelopmentArmSpec, ...] = (
    PARTIAL_TOP_EXTENSION_ARM_SPECS
    + PARTIAL_BOTTOM_EXTENSION_ARM_SPECS
    + PARTIAL_BOTTOM_TOP_EXTENSION_ARM_SPECS
    + BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS
    + BIOPSY_GUIDED_FULL_BOTTOM_TOP_EXTENSION_ARM_SPECS
)
ALL_ARM_SPECS = (
    INITIAL_ARM_SPECS
    + DEVELOPMENT_EXTENSION_ARM_SPECS
    + ADAPTIVE_RADIUS_EXTENSION_ARM_SPECS
)
ARM_SPEC_BY_ID = {spec.arm_id: spec for spec in ALL_ARM_SPECS}


def validate_initial_roster() -> None:
    if len(INITIAL_ARM_SPECS) != 32:
        raise RuntimeError("The v5 initial development roster must contain 32 arms.")
    if len({spec.arm_id for spec in INITIAL_ARM_SPECS}) != len(INITIAL_ARM_SPECS):
        raise RuntimeError("The v5 initial development roster contains duplicate ids.")
    if len(ARM_SPEC_BY_ID) != len(ALL_ARM_SPECS):
        raise RuntimeError("The v5 development roster contains duplicate ids.")
    projected_partial_ids = {
        spec.arm_id
        for spec in ALL_ARM_SPECS
        if spec.top_output_projection
        == TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED
    }
    if set(BIOPSY_GUIDED_FULL_COUNTERPART_BY_PARTIAL_ID) != projected_partial_ids:
        raise RuntimeError(
            "Every projected biopsy-guided arm must have exactly one declared "
            "fully labeled counterpart."
        )
    if set(BIOPSY_GUIDED_FULL_COUNTERPART_BY_PARTIAL_ID.values()) != {
        spec.arm_id for spec in BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS
    }:
        raise RuntimeError(
            "The fully labeled biopsy-guided roster contains an undeclared or "
            "missing projected-arm counterpart."
        )
    for spec in ALL_ARM_SPECS:
        if spec.top_output_projection not in TOP_OUTPUT_PROJECTIONS:
            raise RuntimeError(
                f"Arm {spec.arm_id!r} has an unknown top-output projection."
            )
        if (
            spec.top_output_projection
            == TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED
            and (
                spec.family != PARTIAL_FAMILY
                or spec.input_mode != "ordered"
                or spec.only_nj
            )
        ):
            raise RuntimeError(
                f"Arm {spec.arm_id!r} applies top projection outside the "
                "ordered partial-output contract."
            )
        if spec.family == BIOPSY_GUIDED_FULL_FAMILY and (
            spec.problem
            != "biopsy_guided_occurrence_aware_fully_labeled_closed_state"
            or spec.input_mode != "ordered"
            or spec.only_nj
            or spec.primary_metric != "ad_f1"
            or spec.top_output_projection != TOP_OUTPUT_PROJECTION_NONE
        ):
            raise RuntimeError(
                f"Arm {spec.arm_id!r} violates the fully labeled "
                "biopsy-guided output contract."
            )
    family_counts = {
        family: sum(spec.family == family for spec in INITIAL_ARM_SPECS)
        for family in (PARTIAL_FAMILY, INFERRED_COPY_FAMILY, CONTEXT_FAMILY)
    }
    if family_counts != {PARTIAL_FAMILY: 7, INFERRED_COPY_FAMILY: 22, CONTEXT_FAMILY: 3}:
        raise RuntimeError(f"Unexpected v5 development family counts: {family_counts!r}.")
    for family, incumbent in INCUMBENT_BY_FAMILY.items():
        if ARM_SPEC_BY_ID[incumbent].family != family:
            raise RuntimeError(f"Incumbent {incumbent!r} is in the wrong family.")
    for family, baseline in BASELINE_BY_FAMILY.items():
        if ARM_SPEC_BY_ID[baseline].family != family:
            raise RuntimeError(f"Baseline {baseline!r} is in the wrong family.")


validate_initial_roster()


def _current_algorithm_map() -> dict[str, Any]:
    """Build current callables without consulting the legacy spec registry.

    Owner-directed refinements are added here and to
    ``DEVELOPMENT_EXTENSION_ARM_SPECS`` under new stable ids; they do not alter
    the fixed meaning of the initial 32-arm roster.
    """
    algorithms = {
        "neighbor_joining_classical": neighbor_joining_classical,
        "neighbor_joining_baseline": neighbor_joining_baseline,
        "neighbor_joining_full_full": make_nj_full_variant(True),
        "neighbor_joining_full_partial": make_nj_full_variant(False),
        "neighbor_joining_full_cps_full": make_nj_full_cps_variant(True),
        "neighbor_joining_full_cps_partial": make_nj_full_cps_variant(False),
        "neighbor_joining_hybrid_full": make_nj_hybrid_variant(True),
        "neighbor_joining_hybrid_partial": make_nj_hybrid_variant(False),
        "neighbor_joining_hybrid_inverse_centrality_full": make_nj_hybrid_inv_cent_variant(True),
        "neighbor_joining_hybrid_inverse_centrality_partial": (
            make_nj_hybrid_inv_cent_variant(False)
        ),
        "neighbor_joining_adaptive_centrality": neighbor_joining_adaptive_centrality,
        "neighbor_joining_adaptive_centrality_nonlinear": (
            neighbor_joining_adaptive_centrality_nonlinear
        ),
        "neighbor_joining_adaptive_centrality_reversed": (
            neighbor_joining_adaptive_centrality_reversed
        ),
        "neighbor_joining_anticentral_parent_reuse": (
            neighbor_joining_anticentral_parent_reuse
        ),
        "neighbor_joining_hybrid_opt": neighbor_joining_hybrid_opt,
        "neighbor_joining_hybrid_opt_adaptive": neighbor_joining_hybrid_opt_adaptive,
        "neighbor_joining_hybrid_opt_v2": neighbor_joining_hybrid_opt_v2,
        "neighbor_joining_hybrid_opt_refined": neighbor_joining_hybrid_opt_refined,
        "neighbor_joining_hybrid_anticentral_opt": neighbor_joining_hybrid_anticentral_opt,
        "neighbor_joining_hybrid_anticentral_adaptive_v3": (
            neighbor_joining_hybrid_anticentral_adaptive_v3
        ),
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible": (
            neighbor_joining_hybrid_anticentral_adaptive_v3_plausible
        ),
        "neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible": (
            neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible
        ),
        INFERRED_COPY_INCUMBENT_ID: (
            neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony
        ),
        "new_alg": new_alg,
        "rooted_labeled_nj": rooted_labeled_nj,
        "temporal_cnp_arborescence": temporal_cnp_arborescence,
        "temporal_cnp_arborescence_no_time": temporal_cnp_arborescence_no_time,
    }
    for name, algorithm in algorithms.items():
        algorithm.__name__ = name
    return algorithms


CURRENT_ALGORITHM_BY_NAME = _current_algorithm_map()


def _project_created_top_nodes_to_unlabeled(
    algorithm,
    projection_audit: dict[str, Any],
):
    """Wrap a final-layer solver with the partial-output projection contract."""

    return _audit_created_top_nodes(
        algorithm,
        projection_audit,
        clear_created_labels=True,
    )


def _audit_created_top_nodes(
    algorithm,
    projection_audit: dict[str, Any],
    *,
    clear_created_labels: bool,
):
    """Record top-created nodes and optionally apply the partial projection."""

    @wraps(algorithm)
    def audited_algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=7,
        existing_tree=None,
    ):
        if existing_tree is None:
            raise ValueError(
                "Top-node output auditing requires the initialized biopsy tree."
            )
        nodes_before_top = set(existing_tree.nodes)
        tree, new_nodes, root = algorithm(
            dist_matrix,
            cells,
            max_id,
            seed=seed,
            existing_tree=existing_tree,
        )
        created_node_ids = set(tree.nodes) - nodes_before_top
        labels_cleared = 0
        genomes_cleared = 0
        labels_retained = 0
        genomes_retained = 0
        for node_id in created_node_ids:
            attributes = tree.nodes[node_id]
            has_label = attributes.get("cell_id") is not None
            has_genome = attributes.get("genome") is not None
            if clear_created_labels and has_label:
                labels_cleared += 1
            elif has_label:
                labels_retained += 1
            if clear_created_labels and has_genome:
                genomes_cleared += 1
            elif has_genome:
                genomes_retained += 1
            if clear_created_labels:
                attributes["cell_id"] = None
                attributes["genome"] = None
        projection_audit.update(
            {
                "top_created_node_count": len(created_node_ids),
                "top_labels_cleared_count": labels_cleared,
                "top_genomes_cleared_count": genomes_cleared,
                "top_labels_retained_count": labels_retained,
                "top_genomes_retained_count": genomes_retained,
            }
        )
        return tree, new_nodes, root

    return audited_algorithm


def resolve_arm_specs(arm_ids: Sequence[str] | None) -> tuple[DevelopmentArmSpec, ...]:
    if not arm_ids or tuple(arm_ids) == ("all",):
        return INITIAL_ARM_SPECS
    if tuple(arm_ids) == (BIOPSY_GUIDED_FULL_FAMILY,):
        return BIOPSY_GUIDED_FULL_EXTENSION_ARM_SPECS
    values = []
    seen = set()
    for arm_id in arm_ids:
        if arm_id in seen:
            raise ValueError(f"Duplicate requested arm id {arm_id!r}.")
        if arm_id not in ARM_SPEC_BY_ID:
            available = ", ".join(sorted(ARM_SPEC_BY_ID))
            raise ValueError(f"Unknown development arm {arm_id!r}; available: {available}.")
        seen.add(arm_id)
        values.append(ARM_SPEC_BY_ID[arm_id])
    return tuple(values)


def derived_seed(
    stream: str,
    base_seed: int,
    block_index: int,
    *coordinates: int,
) -> int:
    if not stream:
        raise ValueError("Seed stream must be nonempty.")
    values = (base_seed, block_index, *coordinates)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("Seed coordinates must be nonnegative integers.")
    material = "\0".join(
        [DEVELOPMENT_NAMESPACE, stream, *(str(value) for value in values)]
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def case_id(block_index: int, height: int) -> str:
    return f"dev-b{block_index + 1:03d}-H{height}"


def condition_paths(block_index: int, height: int) -> dict[str, str]:
    base = Path("cases") / case_id(block_index, height)
    return {
        "metadata": (base / "case.json").as_posix(),
        "input": (base / "input.json").as_posix(),
        "distance": (base / "distance.json").as_posix(),
    }


def truth_path(block_index: int) -> str:
    return (Path("truths") / f"block_{block_index + 1:03d}.json").as_posix()


def serialize_distance(case_id_value: str, distance: DistanceMatrix) -> dict[str, Any]:
    return {
        "schema_version": CASE_DISTANCE_SCHEMA_VERSION,
        "case_id": case_id_value,
        "semantics_version": CNP2CNP_SEMANTICS_VERSION,
        "ids": json_safe(distance.ids),
        "matrix": json_safe(np.asarray(distance.matrix, dtype=float)),
        "provenance": json_safe(distance.provenance),
    }


def deserialize_distance(payload: Mapping[str, Any]) -> DistanceMatrix:
    if payload.get("schema_version") != CASE_DISTANCE_SCHEMA_VERSION:
        raise ValueError("Unknown development distance schema.")
    if payload.get("semantics_version") != CNP2CNP_SEMANTICS_VERSION:
        raise ValueError("Development distance semantics changed.")
    return DistanceMatrix(
        ids=list(payload["ids"]),
        matrix=payload["matrix"],
        provenance=payload.get("provenance"),
    )


def serialize_truth(block_index: int, tree: nx.DiGraph) -> dict[str, Any]:
    if not nx.is_arborescence(tree):
        raise ValueError("Development truth must be one directed arborescence.")
    return {
        "schema_version": "ctbf-v5-algorithm-development-truth-v1",
        "block_index": block_index,
        "tree": serialize_tree(tree),
    }


def deserialize_truth(payload: Mapping[str, Any]) -> nx.DiGraph:
    if payload.get("schema_version") != "ctbf-v5-algorithm-development-truth-v1":
        raise ValueError("Unknown development truth schema.")
    tree = deserialize_tree(payload["tree"])
    if not nx.is_arborescence(tree):
        raise ValueError("Stored development truth is not an arborescence.")
    return tree


def truth_prefix(tree: nx.DiGraph, height: int) -> nx.DiGraph:
    selected = [
        node
        for node, attributes in tree.nodes(data=True)
        if int(attributes.get("generation", -1)) <= int(height)
    ]
    prefix = tree.subgraph(selected).copy()
    if not nx.is_arborescence(prefix):
        raise ValueError(f"Truth prefix through H{height} is not an arborescence.")
    return prefix


def observed_labels(payload: Mapping[str, Any]) -> list[Any]:
    validate_reconstruction_input(payload)
    values = {
        state["state_label"]
        for level in payload["levels"]
        for state in level["states"]
    }
    return sorted(values, key=stable_distance_label_key)


def _cells_from_input(payload: Mapping[str, Any]) -> list[list[Genotype]]:
    validate_reconstruction_input(payload)
    return [
        [
            Genotype(
                state["cnp"],
                state["state_label"],
                generation=level["generation"],
                cell_id=state["state_label"],
            )
            for state in level["states"]
        ]
        for level in payload["levels"]
    ]


def actual_root(tree: nx.DiGraph) -> Any:
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one directed root, found {len(roots)}.")
    return roots[0]


def reconstruct_development_arm(
    spec: DevelopmentArmSpec,
    reconstruction_input: Mapping[str, Any],
    distance: DistanceMatrix,
    *,
    reconstruction_seed: int,
    algorithm_override: Any | None = None,
) -> tuple[nx.DiGraph, dict[Any, Any], Any, dict[str, Any]]:
    if (
        algorithm_override is None
        and spec.algorithm_name not in CURRENT_ALGORITHM_BY_NAME
    ):
        raise ValueError(f"No current callable is declared for {spec.algorithm_name!r}.")
    algorithm = (
        CURRENT_ALGORITHM_BY_NAME[spec.algorithm_name]
        if algorithm_override is None
        else algorithm_override
    )
    if not callable(algorithm):
        raise TypeError("algorithm_override must be callable when provided.")
    if spec.top_output_projection not in TOP_OUTPUT_PROJECTIONS:
        raise ValueError(
            f"Unknown top-output projection {spec.top_output_projection!r}."
        )
    projection_audit = {
        "top_created_node_count": 0,
        "top_labels_cleared_count": 0,
        "top_genomes_cleared_count": 0,
        "top_labels_retained_count": 0,
        "top_genomes_retained_count": 0,
    }
    if (
        spec.top_output_projection
        == TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED
    ):
        algorithm = _project_created_top_nodes_to_unlabeled(
            algorithm,
            projection_audit,
        )
    elif spec.family == BIOPSY_GUIDED_FULL_FAMILY:
        algorithm = _audit_created_top_nodes(
            algorithm,
            projection_audit,
            clear_created_labels=False,
        )
    cell_lists = _cells_from_input(reconstruction_input)
    build_input = (
        [[cell for level in cell_lists for cell in level]]
        if spec.input_mode == "pooled"
        else cell_lists
    )
    biopsy_decision_audit = (
        None
        if spec.biopsy_preset is None
        else BiopsyGuidedDecisionAudit()
    )
    biopsy_config = (
        None
        if spec.biopsy_preset is None
        else resolve_biopsy_guided_config(
            spec.biopsy_preset,
            decision_audit=biopsy_decision_audit,
        )
    )
    if spec.radius_policy is not None:
        if biopsy_config is None or spec.input_mode != "ordered" or spec.only_nj:
            raise ValueError(
                "An adaptive biopsy radius requires ordered biopsy-guided input."
            )
        if biopsy_config.radius_policy not in {None, spec.radius_policy}:
            raise ValueError("The arm and biopsy preset declare different radius policies.")
        biopsy_config = replace(
            biopsy_config,
            radius_policy=spec.radius_policy,
        )
    elif spec.radius is None:
        raise ValueError("A fixed-radius reconstruction arm lacks its radius.")
    before_input = canonical_json_digest(reconstruction_input)
    before_ids = tuple(distance.ids)
    before_matrix = np.array(distance.matrix, copy=True)
    tree, levels, returned_root = build_evolution_tree(
        build_input,
        seed=int(reconstruction_seed),
        r=spec.radius,
        only_nj=bool(spec.only_nj),
        distance_matrix=distance,
        neighbor_joining=algorithm,
        biopsy_guided_config=biopsy_config,
    )
    if not nx.is_arborescence(tree):
        raise ValueError("Reconstruction did not return one directed arborescence.")
    root = actual_root(tree)
    if returned_root != root:
        raise ValueError("Reconstruction returned a root different from the graph root.")
    if canonical_json_digest(reconstruction_input) != before_input:
        raise ValueError("Reconstruction mutated its stored observable input.")
    if tuple(distance.ids) != before_ids or not np.array_equal(distance.matrix, before_matrix):
        raise ValueError("Reconstruction mutated its stored distance input.")
    diagnostics = {
        "arm_id": spec.arm_id,
        "algorithm": spec.algorithm_name,
        "family": spec.family,
        "problem": spec.problem,
        "input_mode": spec.input_mode,
        "only_nj": spec.only_nj,
        "radius": spec.radius,
        "biopsy_preset": spec.biopsy_preset,
        "top_output_projection": spec.top_output_projection,
        **projection_audit,
        "biopsy_layer_decision_audit": (
            None
            if biopsy_decision_audit is None
            else biopsy_decision_audit.as_record()
        ),
        "reconstruction_seed": int(reconstruction_seed),
        "returned_root": json_safe(returned_root),
        "actual_root": json_safe(root),
    }
    if spec.radius_policy is not None:
        diagnostics["radius_policy"] = spec.radius_policy
    return tree, dict(levels), returned_root, diagnostics


def tree_summary(tree: nx.DiGraph) -> dict[str, Any]:
    root = actual_root(tree)
    depths = nx.single_source_shortest_path_length(tree, root)
    labels = [
        attributes.get("cell_id")
        for _node, attributes in tree.nodes(data=True)
        if attributes.get("cell_id") is not None
    ]
    repeated = len(labels) - len(set(labels))
    return {
        "node_count": tree.number_of_nodes(),
        "edge_count": tree.number_of_edges(),
        "leaf_count": sum(tree.out_degree(node) == 0 for node in tree),
        "maximum_out_degree": max(
            (tree.out_degree(node) for node in tree),
            default=0,
        ),
        "maximum_depth": max(depths.values(), default=0),
        "labeled_occurrence_count": len(labels),
        "unlabeled_node_count": tree.number_of_nodes() - len(labels),
        "unique_state_label_count": len(set(labels)),
        "repeated_state_label_occurrence_count": repeated,
        "inferred_copy_occurrence_count": repeated,
        "canonical_topology_digest": canonical_topology_digest(tree),
    }


def canonical_topology_digest(tree: nx.DiGraph) -> str:
    """Hash rooted labeled topology without recursive whole-tree conversion.

    The two bottom-up encodings preserve the original canonical JSON bytes:
    child ordering used JSON's default ASCII escaping, while the final digest
    used UTF-8 JSON. Child encodings are released as soon as their sole parent
    is encoded, bounding live memory by the reconstructed tree rather than by
    an additional recursively duplicated object graph.
    """
    root = actual_root(tree)
    if not nx.is_arborescence(tree):
        raise ValueError("Topology digest requires a rooted directed arborescence.")

    ascii_encodings: dict[Any, bytes] = {}
    utf8_encodings: dict[Any, bytes] = {}
    for node in reversed(list(nx.topological_sort(tree))):
        children = [
            (ascii_encodings.pop(child), utf8_encodings.pop(child))
            for child in tree.successors(node)
        ]
        children.sort(key=lambda pair: pair[0])
        label = json_safe(tree.nodes[node].get("cell_id"))
        ascii_label = json.dumps(
            label,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        utf8_label = json.dumps(
            label,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        ascii_encodings[node] = (
            b'{"children":['
            + b",".join(pair[0] for pair in children)
            + b'],"label":'
            + ascii_label
            + b"}"
        )
        utf8_encodings[node] = (
            b'{"children":['
            + b",".join(pair[1] for pair in children)
            + b'],"label":'
            + utf8_label
            + b"}"
        )

    if set(ascii_encodings) != {root} or set(utf8_encodings) != {root}:
        raise ValueError("Topology digest did not reduce to the declared root.")
    return hashlib.sha256(utf8_encodings[root]).hexdigest()


def canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_bank_manifest(
    bank_root: Path | str,
    *,
    expected_block_count: int = DEFAULT_BLOCK_COUNT,
) -> tuple[Path, dict[str, Any]]:
    root = Path(bank_root).expanduser().resolve()
    manifest = read_json(root / BANK_MANIFEST_NAME)
    if manifest.get("schema_version") not in {
        BANK_SCHEMA_VERSION,
        LEGACY_BANK_SCHEMA_VERSION,
    }:
        raise ValueError("Unknown algorithm-development bank schema.")
    if manifest.get("bank_id") != DEFAULT_BANK_ID:
        raise ValueError("Unknown algorithm-development bank id.")
    if manifest.get("seed_namespace") != DEVELOPMENT_NAMESPACE:
        raise ValueError("Unknown algorithm-development seed namespace.")
    if manifest.get("status") != "complete":
        raise ValueError("Algorithm-development bank is not complete.")
    if manifest.get("block_count") != expected_block_count:
        raise ValueError(
            "Development bank has the wrong truth-block count: "
            f"expected {expected_block_count}, observed {manifest.get('block_count')}."
        )
    if manifest.get("condition_count") != expected_block_count * len(HEIGHT_SCHEDULES):
        raise ValueError("Development bank condition count is inconsistent.")
    expected_schedules = {
        str(height): list(generations)
        for height, generations in HEIGHT_SCHEDULES.items()
    }
    if manifest.get("height_schedules") != expected_schedules:
        raise ValueError("Development bank height schedules are inconsistent.")
    if manifest.get("simulation_height") != max(HEIGHT_SCHEDULES):
        raise ValueError("Development bank simulation height is inconsistent.")
    if manifest.get("paired_condition_heights") != sorted(HEIGHT_SCHEDULES):
        raise ValueError("Development bank paired-height inventory is inconsistent.")
    if manifest.get("completed_condition_count") != manifest["condition_count"]:
        raise ValueError("Development bank completion count is inconsistent.")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != manifest["condition_count"]:
        raise ValueError("Development bank case inventory is incomplete.")
    expected_ids = {
        case_id(block_index, height)
        for block_index in range(expected_block_count)
        for height in HEIGHT_SCHEDULES
    }
    if {case.get("case_id") for case in cases} != expected_ids:
        raise ValueError("Development bank case ids are incomplete or duplicated.")
    for case in cases:
        for field in ("metadata_path", "input_path", "distance_path", "truth_path"):
            path = root / case[field]
            if not path.is_file():
                raise ValueError(f"Development bank is missing {path}.")
    return root, manifest


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_json_atomic(path, value)


def read_case_assets(
    bank_root: Path,
    case_record: Mapping[str, Any],
    *,
    truth_cache: dict[str, nx.DiGraph] | None = None,
) -> tuple[dict[str, Any], DistanceMatrix, nx.DiGraph, dict[str, Any]]:
    input_payload = read_json(bank_root / case_record["input_path"])
    validate_reconstruction_input(input_payload)
    distance_payload = read_json(bank_root / case_record["distance_path"])
    distance = deserialize_distance(distance_payload)
    metadata = read_json(bank_root / case_record["metadata_path"])
    truth_relative = str(case_record["truth_path"])
    cache = truth_cache if truth_cache is not None else {}
    if truth_relative not in cache:
        truth_payload = read_json(bank_root / truth_relative)
        if truth_payload.get("block_index") != int(case_record["block_index"]):
            raise ValueError("Stored truth block index changed.")
        cache[truth_relative] = deserialize_truth(truth_payload)
    truth = truth_prefix(cache[truth_relative], int(case_record["height"]))
    if input_payload.get("case_id") != case_record["case_id"]:
        raise ValueError("Stored reconstruction input case id changed.")
    if distance_payload.get("case_id") != case_record["case_id"]:
        raise ValueError("Stored distance case id changed.")
    if metadata.get("schema_version") != CASE_METADATA_SCHEMA_VERSION:
        raise ValueError("Unknown stored case-metadata schema.")
    if metadata.get("case_id") != case_record["case_id"]:
        raise ValueError("Stored case metadata id changed.")
    if metadata.get("block_index") != int(case_record["block_index"]):
        raise ValueError("Stored case metadata block index changed.")
    if metadata.get("height") != int(case_record["height"]):
        raise ValueError("Stored case metadata height changed.")
    validate_distance_label_coverage(
        distance.ids,
        observed_labels(input_payload),
        allow_extra=False,
    )
    return input_payload, distance, truth, metadata


def ensure_new_output_root(path: Path | str) -> Path:
    root = Path(path).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Output root already exists: {root}.")
    root.mkdir(parents=True)
    return root


def numeric_summary(values: Iterable[float]) -> dict[str, Any] | None:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return None
    if not np.all(np.isfinite(array)):
        raise ValueError("Numeric summary values must be finite.")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


__all__ = [
    "ADAPTIVE_RADIUS_EXTENSION_ARM_SPECS",
    "ARM_SPEC_BY_ID",
    "ALL_ARM_SPECS",
    "BANK_CONFIG_NAME",
    "BANK_MANIFEST_NAME",
    "BANK_SCHEMA_VERSION",
    "LEGACY_BANK_SCHEMA_VERSION",
    "BASELINE_BY_FAMILY",
    "BIOPSY_GUIDED_FULL_BOTTOM_TOP_EXTENSION_ARM_SPECS",
    "BIOPSY_GUIDED_FULL_BOTTOM_TOP_INTERACTION_ROLE",
    "BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_DEFERRED_ID",
    "BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_DEFAULT_ID",
    "BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_ROOTED_Q_DEFERRED_ID",
    "BIOPSY_GUIDED_FULL_ADAPTIVE_MEDIAN_ROOTED_Q_DEFAULT_ID",
    "BIOPSY_GUIDED_FULL_ROOTED_Q_DEFERRED_ID",
    "BIOPSY_LOWER_BOUND",
    "CASE_DISTANCE_SCHEMA_VERSION",
    "CASE_INPUT_SCHEMA_VERSION",
    "CASE_METADATA_SCHEMA_VERSION",
    "COMPARISON_FAMILIES",
    "CONTEXT_FAMILY",
    "DEFAULT_BANK_ID",
    "DEFAULT_BASE_SEED",
    "DEFAULT_BLOCK_COUNT",
    "DEVELOPMENT_NAMESPACE",
    "DEVELOPMENT_EXTENSION_ARM_SPECS",
    "DevelopmentArmSpec",
    "HEIGHT_SCHEDULES",
    "INCUMBENT_BY_FAMILY",
    "INFERRED_COPY_FAMILY",
    "INITIAL_ARM_SPECS",
    "PARTIAL_FAMILY",
    "PARTIAL_BOTTOM_CANDIDATE_ROLE",
    "PARTIAL_BOTTOM_CONTROL_ID",
    "PARTIAL_BOTTOM_EXTENSION_ARM_SPECS",
    "PARTIAL_BOTTOM_TOP_EXTENSION_ARM_SPECS",
    "PARTIAL_BOTTOM_TOP_INTERACTION_ARM_ID",
    "PARTIAL_BOTTOM_TOP_INTERACTION_ROLE",
    "PARTIAL_TOP_EXTENSION_ARM_SPECS",
    "REPORT_SCHEMA_VERSION",
    "LEGACY_RUN_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "SAMPLING_RULE",
    "TARGET_FRACTION",
    "TOP_OUTPUT_PROJECTION_CREATED_NODES_UNLABELED",
    "TOP_OUTPUT_PROJECTION_NONE",
    "TOP_OUTPUT_PROJECTIONS",
    "actual_root",
    "canonical_json_digest",
    "case_id",
    "condition_paths",
    "derived_seed",
    "deserialize_distance",
    "ensure_new_output_root",
    "load_bank_manifest",
    "numeric_summary",
    "observed_labels",
    "read_case_assets",
    "reconstruct_development_arm",
    "resolve_arm_specs",
    "serialize_distance",
    "serialize_truth",
    "tree_summary",
    "truth_path",
    "truth_prefix",
    "validate_initial_roster",
    "write_json",
]
