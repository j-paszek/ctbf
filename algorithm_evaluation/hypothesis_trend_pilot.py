"""Prepare the bounded CTBF v5 hypothesis-trend pilot.

The first tranche is intentionally dependency-free: it evaluates the exact
interior point-event metric-ball formula and writes the provisional length and
height trend plan.  It does not simulate, call cnp2cnp, reconstruct a tree, or
read an existing result corpus.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from algorithm_evaluation.paper_pipeline_contract import write_json_atomic


SCHEMA_VERSION = "ctbf-hypothesis-trend-pilot-v1"
ANALYSIS_ROLE = "discovery_hypothesis_trend"
THEORY_MODEL_VERSION = "interior-integer-l1-point-event-ball-v1"
PLAN_VERSION = "ctbf-length-height-one-factor-endpoint-plan-v1"

DEFAULT_LENGTHS = (10, 50, 100)
DEFAULT_HEIGHTS = (8, 12, 16)
DEFAULT_MAX_RADIUS = 2
DEFAULT_EXPECTED_CNA_STARTS = 0.1
DEFAULT_LENGTH_TREND_HEIGHT = 8
DEFAULT_HEIGHT_TREND_LENGTH = 50
DEFAULT_RELATIVE_BIOPSY_POSITIONS = (3 / 7, 5 / 7, 1.0)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HYPOTHESIS_DOCUMENT = (
    PROJECT_ROOT / "experimental_description" / "hypothesis_driven_trend_program.md"
)
SENSITIVITY_DOCUMENT = (
    PROJECT_ROOT / "experimental_description" / "cnp_length_tree_height_sensitivity.md"
)


def _positive_unique_integers(values: Iterable[int], field: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if not normalized:
        raise ValueError(f"{field} must contain at least one value.")
    if any(value <= 0 for value in normalized):
        raise ValueError(f"{field} must contain only positive integers.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must not contain duplicates.")
    return tuple(sorted(normalized))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer_l1_shell_size(dimension: int, radius: int) -> int:
    """Return the number of integer lattice points at exact L1 radius.

    The formula assumes an interior profile where every selected coordinate
    may change in either direction.  It is therefore a simplified point-event
    model, not the cnp2cnp state graph near copy-number boundaries.
    """
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer.")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a nonnegative integer.")
    if radius == 0:
        return 1
    return sum(
        math.comb(dimension, nonzero_coordinates)
        * math.comb(radius - 1, nonzero_coordinates - 1)
        * (2 ** nonzero_coordinates)
        for nonzero_coordinates in range(1, min(dimension, radius) + 1)
    )


def integer_l1_ball_size(dimension: int, radius: int) -> int:
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a nonnegative integer.")
    return sum(integer_l1_shell_size(dimension, value) for value in range(radius + 1))


def interval_event_description_count(length: int) -> int:
    """Count signed contiguous unit-interval events at an interior profile."""
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer.")
    return length * (length + 1)


def _relative_biopsy_generations(
    height: int,
    positions: Sequence[float] = DEFAULT_RELATIVE_BIOPSY_POSITIONS,
) -> list[int]:
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < float(value) <= 1
        for value in positions
    ):
        raise ValueError("relative biopsy positions must be numeric values in (0,1].")
    generations = [
        min(height, max(1, math.floor(float(position) * height + 0.5)))
        for position in positions
    ]
    if generations != sorted(generations) or len(set(generations)) != len(generations):
        raise ValueError(
            f"Relative biopsy positions collapse at height {height}: {generations}"
        )
    return generations


def _case_row(
    *,
    trend: str,
    length: int,
    height: int,
    expected_cna_starts: float,
) -> dict[str, Any]:
    probability = expected_cna_starts / length
    if probability > 1:
        raise ValueError(
            f"Expected CNA starts {expected_cna_starts} are impossible at length {length}."
        )
    return {
        "case_key": f"{trend}-L{length}-H{height}",
        "trend": trend,
        "genome_length": length,
        "number_of_generations": height,
        "unreduced_binary_generation_occurrence_bound": 2 ** height,
        "unreduced_binary_total_node_bound": (2 ** (height + 1)) - 1,
        "expected_cna_starts_per_attempted_child": expected_cna_starts,
        "cna_event_probability": probability,
        "relative_biopsy_generations": _relative_biopsy_generations(height),
    }


def build_pilot_report(
    *,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    heights: Sequence[int] = DEFAULT_HEIGHTS,
    max_radius: int = DEFAULT_MAX_RADIUS,
    expected_cna_starts: float = DEFAULT_EXPECTED_CNA_STARTS,
    length_trend_height: int = DEFAULT_LENGTH_TREND_HEIGHT,
    height_trend_length: int = DEFAULT_HEIGHT_TREND_LENGTH,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    lengths = _positive_unique_integers(lengths, "lengths")
    heights = _positive_unique_integers(heights, "heights")
    if isinstance(max_radius, bool) or not isinstance(max_radius, int) or max_radius < 0:
        raise ValueError("max_radius must be a nonnegative integer.")
    if max_radius > 12:
        raise ValueError("max_radius above 12 is outside this bounded theory pilot.")
    if (
        isinstance(expected_cna_starts, bool)
        or not isinstance(expected_cna_starts, (int, float))
        or not math.isfinite(float(expected_cna_starts))
        or expected_cna_starts <= 0
    ):
        raise ValueError("expected_cna_starts must be a finite positive number.")
    if length_trend_height <= 0 or height_trend_length <= 0:
        raise ValueError("fixed trend height and length must be positive.")

    expected_cna_starts = float(expected_cna_starts)
    theory_rows = []
    for length in lengths:
        shells = {
            str(radius): integer_l1_shell_size(length, radius)
            for radius in range(max_radius + 1)
        }
        balls = {
            str(radius): integer_l1_ball_size(length, radius)
            for radius in range(max_radius + 1)
        }
        theory_rows.append(
            {
                "genome_length": length,
                "exact_shell_size_by_radius": shells,
                "closed_ball_size_by_radius": balls,
                "signed_contiguous_unit_interval_event_count": (
                    interval_event_description_count(length)
                ),
            }
        )

    length_rows = [
        _case_row(
            trend="length",
            length=length,
            height=int(length_trend_height),
            expected_cna_starts=expected_cna_starts,
        )
        for length in lengths
    ]
    height_rows = [
        _case_row(
            trend="height",
            length=int(height_trend_length),
            height=height,
            expected_cna_starts=expected_cna_starts,
        )
        for height in heights
    ]
    endpoint_rows = [
        _case_row(
            trend="endpoint_interaction",
            length=length,
            height=height,
            expected_cna_starts=expected_cna_starts,
        )
        for length in (lengths[0], lengths[-1])
        for height in (heights[0], heights[-1])
    ]

    created_at_utc = created_at_utc or datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "created_at_utc": created_at_utc,
        "prediction_audit": {
            "hypothesis_document": HYPOTHESIS_DOCUMENT.relative_to(PROJECT_ROOT).as_posix(),
            "hypothesis_document_sha256": _file_sha256(HYPOTHESIS_DOCUMENT),
            "sensitivity_document": SENSITIVITY_DOCUMENT.relative_to(PROJECT_ROOT).as_posix(),
            "sensitivity_document_sha256": _file_sha256(SENSITIVITY_DOCUMENT),
            "prediction_ids": [
                "H1a",
                "H1b",
                "H1c",
                "H2",
                "H3",
                "H4",
                "H5",
                "H6",
            ],
            "result_status": {
                "H1a": "exact_theory_calculated",
                "H1b": "not_tested_requires_simulation",
                "H1c": "not_tested_requires_candidate_graph_instrumentation",
                "H2": "not_tested_requires_instrumented_reconstruction",
                "H3": "not_tested_requires_simulation",
                "H4": "not_tested_requires_evaluator_fixture",
                "H5": "not_tested_requires_simulation",
                "H6": "not_tested_requires_resolution_projection",
            },
        },
        "theory": {
            "model_version": THEORY_MODEL_VERSION,
            "scope": "unbounded interior integer lattice with signed unit point events",
            "paper_evidence_allowed": False,
            "formula": {
                "exact_shell_radius_k": (
                    "sum_{j=1}^{min(L,k)} choose(L,j)*choose(k-1,j-1)*2^j"
                ),
                "radius_one": "2*L",
                "radius_two": "2*L^2",
                "closed_radius_two": "1+2*L+2*L^2",
                "signed_contiguous_unit_intervals": "L*(L+1)",
            },
            "limitations": [
                "copy-number nonnegativity can reduce loss directions near zero",
                "event equivalences can merge descriptions into one endpoint",
                "cnp2cnp radius values require provider-specific validation",
                "state-space cardinality does not imply observed-sample occupancy",
            ],
            "rows": theory_rows,
        },
        "prospective_plan": {
            "plan_version": PLAN_VERSION,
            "status": "discovery_factor_plan_not_execution_manifest",
            "paper_factor_levels_frozen": False,
            "seed_count_frozen": False,
            "distance_provider": "ctbf-cnp2cnp-any-min-bidirectional-v1",
            "base_regime": "clean_balanced",
            "base_branching_bound": {
                "descendant_attempts_per_retained_parent": 2,
                "generation_H_occurrence_bound": "2^H",
                "total_truth_node_bound": "2^(H+1)-1",
                "representative_collisions_or_viability_loss_may_reduce_realized_counts": True,
            },
            "event_burden_control": "CNA_EVENT_PROBABILITY=expected_cna_starts/GENOME_LENGTH",
            "relative_biopsy_position_source": "clean_L3_positions_3_over_7_5_over_7_1",
            "observation_arms": {
                "fixed_fraction": {
                    "provisional_fraction": 0.5,
                    "status": "blocked_until_population_and_profile_cap_preflight",
                },
                "fixed_budget": {
                    "status": "budget_requires_one_case_population_preflight",
                },
            },
            "length_trend": length_rows,
            "height_trend": height_rows,
            "endpoint_interaction": endpoint_rows,
            "execution_order": [
                "population_only_height_preflight",
                "one_case_fixed_budget_cnp2cnp_runtime_preflight",
                "length_trend",
                "fixed_budget_height_trend",
                "fixed_fraction_height_cells_only_if_profile_cap_passes",
                "endpoint_interaction_if_prior_stages_pass",
                "tie_policy_tranche_selected_by_ambiguity_not_accuracy",
                "evaluator_tranche_after_metric_policy_freeze",
            ],
        },
        "provenance": {
            "source_module": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "source_module_sha256": _file_sha256(Path(__file__)),
            "reads_existing_result_corpus": False,
            "runs_simulation": False,
            "runs_cnp2cnp": False,
            "runs_reconstruction": False,
        },
    }
    validate_pilot_report(report)
    return report


def validate_pilot_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown hypothesis-trend pilot schema.")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("Pilot report must remain discovery evidence.")
    theory = report.get("theory")
    if not isinstance(theory, dict) or theory.get("paper_evidence_allowed") is not False:
        raise ValueError("Theory block must be explicitly ineligible as paper evidence.")
    rows = theory.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Theory block must contain rows.")
    lengths = [row.get("genome_length") for row in rows]
    if len(lengths) != len(set(lengths)):
        raise ValueError("Theory rows contain duplicate genome lengths.")
    plan = report.get("prospective_plan")
    if not isinstance(plan, dict) or plan.get("paper_factor_levels_frozen") is not False:
        raise ValueError("Pilot plan must not freeze paper factor levels.")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict) or any(
        provenance.get(field) is not False
        for field in (
            "reads_existing_result_corpus",
            "runs_simulation",
            "runs_cnp2cnp",
            "runs_reconstruction",
        )
    ):
        raise ValueError("T0 provenance must remain dependency-free and corpus-free.")
    json.dumps(report, sort_keys=True, allow_nan=False)


def compact_summary(report: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = report["theory"]["rows"]
    return {
        "analysis_role": report["analysis_role"],
        "schema_version": report["schema_version"],
        "output": str(output.resolve()),
        "H1a_status": report["prediction_audit"]["result_status"]["H1a"],
        "point_event_counts": {
            str(row["genome_length"]): {
                "radius_one_shell": row["exact_shell_size_by_radius"].get("1"),
                "radius_two_shell": row["exact_shell_size_by_radius"].get("2"),
                "radius_two_ball": row["closed_ball_size_by_radius"].get("2"),
                "signed_interval_events": row[
                    "signed_contiguous_unit_interval_event_count"
                ],
            }
            for row in rows
        },
        "planned_case_counts": {
            key: len(report["prospective_plan"][key])
            for key in ("length_trend", "height_trend", "endpoint_interaction")
        },
        "height_unreduced_total_node_bounds": {
            str(row["number_of_generations"]): row[
                "unreduced_binary_total_node_bound"
            ]
            for row in report["prospective_plan"]["height_trend"]
        },
        "simulation_status": "not_run_by_T0",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write the dependency-free T0 hypothesis-trend theory report and "
            "provisional length/height case plan."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--heights", type=int, nargs="+", default=list(DEFAULT_HEIGHTS))
    parser.add_argument("--max-radius", type=int, default=DEFAULT_MAX_RADIUS)
    parser.add_argument(
        "--expected-cna-starts",
        type=float,
        default=DEFAULT_EXPECTED_CNA_STARTS,
    )
    parser.add_argument(
        "--length-trend-height",
        type=int,
        default=DEFAULT_LENGTH_TREND_HEIGHT,
    )
    parser.add_argument(
        "--height-trend-length",
        type=int,
        default=DEFAULT_HEIGHT_TREND_LENGTH,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output. Never use this for a closed evidence artifact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new versioned path or pass --overwrite."
        )
    report = build_pilot_report(
        lengths=arguments.lengths,
        heights=arguments.heights,
        max_radius=arguments.max_radius,
        expected_cna_starts=arguments.expected_cna_starts,
        length_trend_height=arguments.length_trend_height,
        height_trend_length=arguments.height_trend_length,
    )
    write_json_atomic(output, report)
    print(json.dumps(compact_summary(report, output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
