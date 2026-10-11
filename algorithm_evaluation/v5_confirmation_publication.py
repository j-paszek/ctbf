"""Create publication-only derivatives of the closed CTBF v5 confirmation.

This module reads checksum-closed evidence and writes a new, non-overwriting
artifact root.  It never changes the registered report or any reconstruction
record.  All accuracy summaries remain separated by output class.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Sequence


_MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "ctbf-v5-confirmation-mpl-cache"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from algorithm_evaluation.paper_pipeline_contract import (  # noqa: E402
    file_sha256,
    read_json,
)
from algorithm_evaluation.v5_confirmation_bank import load_confirmation_bank  # noqa: E402
from algorithm_evaluation.v5_confirmation_common import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    M0_ID,
    M2_ID,
    M5_ID,
    M6_ID,
    M7_ID,
    PAPER_ID_BY_ARM,
    POOLED_Q_ID,
    STANDARD_CONDITION_IDS,
    derive_analysis_seed,
    validate_checksum_closure,
    validate_manifest,
    write_checksum_closure,
    write_json,
)
from algorithm_evaluation.v5_confirmation_medicc2 import (  # noqa: E402
    RUN_CHECKSUM_NAME as MEDICC2_CHECKSUM_NAME,
    load_medicc2_confirmation,
)
from algorithm_evaluation.v5_confirmation_report import (  # noqa: E402
    REPORT_CHECKSUM_NAME,
    _bootstrap_interval,
    _load_medicc2_index,
    _load_record_index,
    _metric,
    _numeric_summary,
    _wins_ties_losses,
)
from algorithm_evaluation.v5_confirmation_run import load_confirmation_run  # noqa: E402
from algorithm_evaluation.v5_medicc2_development_run import (  # noqa: E402
    MEDICC2_DEVELOPMENT_ARM_ID,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIRMATION_ROOT = (
    PROJECT_ROOT
    / "algorithm_evaluation"
    / "results"
    / "v5_confirmation"
    / "confirmation_v1"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "paper" / "figures" / "v5-confirmation-v1"

PUBLICATION_SCHEMA_VERSION = "ctbf-v5-confirmation-publication-v1"
PUBLICATION_CHECKSUM_NAME = "checksums.sha256"
BOOTSTRAP_REPETITIONS = 100_000

HEIGHT_LABEL = {
    "H14_rule_y": "H14",
    "H24_rule_y": "H24",
    "H34_rule_y": "H34",
}

METHOD_LABEL = {
    M0_ID: r"$M_0$",
    M2_ID: r"$M_2$",
    M5_ID: r"$M_5$",
    M6_ID: r"$M_6$",
    M7_ID: r"$M_7$",
    POOLED_Q_ID: "Pooled-Q",
    MEDICC2_DEVELOPMENT_ARM_ID: "MEDICC2",
}

PARTIAL_FIGURE_METHODS = (
    M0_ID,
    MEDICC2_DEVELOPMENT_ARM_ID,
    M2_ID,
)
FULL_FIGURE_METHODS = (
    POOLED_Q_ID,
    M5_ID,
    M6_ID,
    M7_ID,
)


@dataclass(frozen=True)
class PublicationInputs:
    validation: Mapping[str, Any]
    bank_root: Path
    bank: Mapping[str, Any]
    run_root: Path
    run: Mapping[str, Any]
    medicc2_root: Path
    medicc2_run: Mapping[str, Any]
    report_root: Path
    report: Mapping[str, Any]
    internal_index: Mapping[tuple[int, str, str], Mapping[str, Any]]
    medicc2_index: Mapping[tuple[int, str, str], Mapping[str, Any]]


def _require_finite(value: Any, field: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite publication value for {field}.")
    return numeric


def _resource_pair(resource: Mapping[str, Any], field: str) -> tuple[float, int]:
    wall_time_ns = resource.get("wall_time_ns")
    memory = resource.get("memory")
    if not isinstance(wall_time_ns, int) or not isinstance(memory, Mapping):
        raise ValueError(f"Missing auditable resources for {field}.")
    peak_rss_bytes = memory.get("peak_rss_bytes")
    if not isinstance(peak_rss_bytes, int):
        raise ValueError(f"Missing peak RSS for {field}.")
    return wall_time_ns / 1e9, peak_rss_bytes


def _resource_summary(wall: Sequence[float], rss: Sequence[int]) -> dict[str, Any]:
    if not wall or len(wall) != len(rss):
        raise ValueError("A resource summary requires aligned nonempty values.")
    return {
        "count": len(wall),
        "wall_seconds": _numeric_summary(wall),
        "peak_rss_bytes": _numeric_summary(rss),
    }


def conservative_m2_resource_accounting(
    *,
    distance_resources: Mapping[int, Mapping[str, Any]],
    m2_resources: Mapping[int, Sequence[Mapping[str, Any]]],
    medicc2_resources: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize stored stages without pretending shared D is method-local.

    The conservative per-standard-case time charges the complete block-wide D
    stage only across the three standard M2 cases, although that matrix served
    all eight conditions and all internal methods.  Sequential peak RSS is the
    maximum stage RSS rather than their sum.
    """

    block_ids = sorted(distance_resources)
    if not block_ids:
        raise ValueError("No distance-resource blocks were supplied.")
    if set(block_ids) != set(m2_resources) or set(block_ids) != set(medicc2_resources):
        raise ValueError("Resource block inventories are not aligned.")

    distance_wall: list[float] = []
    distance_rss: list[int] = []
    m2_wall: list[float] = []
    m2_rss: list[int] = []
    medicc2_wall: list[float] = []
    medicc2_rss: list[int] = []
    m2_block_wall: list[float] = []
    m2_amortized_wall: list[float] = []
    m2_pipeline_rss: list[int] = []
    medicc2_block_mean_wall: list[float] = []
    medicc2_block_rss: list[int] = []

    standard_count = len(STANDARD_CONDITION_IDS)
    for block_index in block_ids:
        d_wall, d_rss = _resource_pair(
            distance_resources[block_index],
            f"distance block {block_index}",
        )
        internal = list(m2_resources[block_index])
        external = list(medicc2_resources[block_index])
        if len(internal) != standard_count or len(external) != standard_count:
            raise ValueError(
                f"Block {block_index} does not contain all standard resource records."
            )
        internal_pairs = [
            _resource_pair(resource, f"M2 block {block_index}")
            for resource in internal
        ]
        external_pairs = [
            _resource_pair(resource, f"MEDICC2 block {block_index}")
            for resource in external
        ]

        distance_wall.append(d_wall)
        distance_rss.append(d_rss)
        m2_wall.extend(wall for wall, _ in internal_pairs)
        m2_rss.extend(rss for _, rss in internal_pairs)
        medicc2_wall.extend(wall for wall, _ in external_pairs)
        medicc2_rss.extend(rss for _, rss in external_pairs)

        total_wall = d_wall + sum(wall for wall, _ in internal_pairs)
        m2_block_wall.append(total_wall)
        m2_amortized_wall.append(total_wall / standard_count)
        m2_pipeline_rss.append(max([d_rss, *(rss for _, rss in internal_pairs)]))
        medicc2_block_mean_wall.append(
            statistics.fmean(wall for wall, _ in external_pairs)
        )
        medicc2_block_rss.append(max(rss for _, rss in external_pairs))

    return {
        "accounting_rule": (
            "The cnp2cnp stage built one maximal matrix per truth block for all "
            "eight conditions. The conservative M2 value charges its complete "
            "wall time only across the three standard M2 cases. Sequential peak "
            "RSS is the maximum of the distance and reconstruction stages."
        ),
        "distance_block_wide_shared_stage": _resource_summary(
            distance_wall, distance_rss
        ),
        "m2_reconstruction_standard_case": _resource_summary(m2_wall, m2_rss),
        "m2_three_standard_cases_plus_full_shared_distance_per_block": (
            _resource_summary(m2_block_wall, m2_pipeline_rss)
        ),
        "m2_conservative_amortized_standard_case": _resource_summary(
            m2_amortized_wall, m2_pipeline_rss
        ),
        "medicc2_standard_case": _resource_summary(medicc2_wall, medicc2_rss),
        "medicc2_block_mean_standard_case": _resource_summary(
            medicc2_block_mean_wall, medicc2_block_rss
        ),
    }


def load_publication_inputs(
    *,
    bank_root: Path | str,
    run_root: Path | str,
    medicc2_root: Path | str,
    report_root: Path | str,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> PublicationInputs:
    validation = validate_manifest(manifest_path, require_approved=True)
    resolved_bank_root, bank = load_confirmation_bank(
        bank_root, manifest_path=manifest_path
    )
    resolved_run_root, run = load_confirmation_run(
        run_root, manifest_path=manifest_path
    )
    resolved_medicc2_root, medicc2_run = load_medicc2_confirmation(
        medicc2_root, validation=validation
    )
    medicc2_run = dict(medicc2_run)
    medicc2_run["checksum_closure"] = validate_checksum_closure(
        resolved_medicc2_root, MEDICC2_CHECKSUM_NAME
    )
    resolved_report_root = Path(report_root).expanduser().resolve()
    report_closure = validate_checksum_closure(
        resolved_report_root, REPORT_CHECKSUM_NAME
    )
    report = read_json(resolved_report_root / "summary.json")

    expected = {
        "status": "complete",
        "scientific_role": "prospectively_locked_independent_confirmation",
        "design_manifest_sha256": validation["manifest_sha256"],
        "protocol_sha256": validation["protocol_sha256"],
        "source_lock_sha256": validation["source_lock"]["sha256"],
        "run_checksum_sha256": run["checksum_closure"]["sha256"],
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"Confirmation report {field} changed.")
    if report.get("block_count") != 100:
        raise ValueError("Publication evidence requires all 100 truth blocks.")
    reported_medicc2_checksum = report.get("secondary_medicc2", {}).get(
        "run_checksum_sha256"
    )
    if reported_medicc2_checksum not in (
        None,
        medicc2_run["checksum_closure"]["sha256"],
    ):
        raise ValueError("MEDICC2 report/run closure changed.")
    report = dict(report)
    report["checksum_closure"] = report_closure

    return PublicationInputs(
        validation=validation,
        bank_root=resolved_bank_root,
        bank=bank,
        run_root=resolved_run_root,
        run=run,
        medicc2_root=resolved_medicc2_root,
        medicc2_run=medicc2_run,
        report_root=resolved_report_root,
        report=report,
        internal_index=_load_record_index(resolved_run_root, run),
        medicc2_index=_load_medicc2_index(
            resolved_medicc2_root, medicc2_run
        ),
    )


def _method_values(
    inputs: PublicationInputs,
    *,
    arm_id: str,
    condition_id: str,
    metric: str,
) -> list[float]:
    index = (
        inputs.medicc2_index
        if arm_id == MEDICC2_DEVELOPMENT_ARM_ID
        else inputs.internal_index
    )
    values = [
        _metric(index, block_index, condition_id, arm_id, metric)
        for block_index in range(100)
    ]
    if any(value is None for value in values):
        raise ValueError(
            f"Publication cell is incomplete: {arm_id}/{condition_id}/{metric}."
        )
    return [_require_finite(value, f"{arm_id}/{condition_id}/{metric}") for value in values]


def build_figure_rows(inputs: PublicationInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specifications = [
        *(('partial', 'grf', arm_id) for arm_id in PARTIAL_FIGURE_METHODS),
        *(('closed_state', 'ad_f1', arm_id) for arm_id in FULL_FIGURE_METHODS),
    ]
    for family, metric, arm_id in specifications:
        for condition_id in STANDARD_CONDITION_IDS:
            values = _method_values(
                inputs,
                arm_id=arm_id,
                condition_id=condition_id,
                metric=metric,
            )
            interval = _bootstrap_interval(
                values,
                repetitions=BOOTSTRAP_REPETITIONS,
                seed=derive_analysis_seed(
                    f"publication_{family}_{metric}_{arm_id}_{condition_id}"
                ),
            )
            rows.append(
                {
                    "schema_version": PUBLICATION_SCHEMA_VERSION,
                    "family": family,
                    "metric": metric,
                    "arm_id": arm_id,
                    "paper_id": (
                        "MEDICC2"
                        if arm_id == MEDICC2_DEVELOPMENT_ARM_ID
                        else PAPER_ID_BY_ARM[arm_id]
                    ),
                    "condition_id": condition_id,
                    "height": int(HEIGHT_LABEL[condition_id][1:]),
                    "count": len(values),
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "bootstrap_lower": interval["lower"],
                    "bootstrap_upper": interval["upper"],
                    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                }
            )
    return rows


def _contrast_by_height(
    inputs: PublicationInputs,
    *,
    contrast_id: str,
    left_arm: str,
    right_arm: str,
    metric: str,
) -> list[dict[str, Any]]:
    rows = []
    for offset, condition_id in enumerate(STANDARD_CONDITION_IDS):
        left = _method_values(
            inputs, arm_id=left_arm, condition_id=condition_id, metric=metric
        )
        right = _method_values(
            inputs, arm_id=right_arm, condition_id=condition_id, metric=metric
        )
        effects = [a - b for a, b in zip(left, right, strict=True)]
        interval = _bootstrap_interval(
            effects,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=derive_analysis_seed(f"publication_{contrast_id}_{offset}"),
        )
        rows.append(
            {
                "schema_version": PUBLICATION_SCHEMA_VERSION,
                "contrast_id": contrast_id,
                "metric": metric,
                "condition_id": condition_id,
                "height": int(HEIGHT_LABEL[condition_id][1:]),
                **_numeric_summary(effects),
                **_wins_ties_losses(effects, 1e-12),
                "bootstrap_lower": interval["lower"],
                "bootstrap_upper": interval["upper"],
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            }
        )
    return rows


def build_contrast_rows(inputs: PublicationInputs) -> list[dict[str, Any]]:
    rows = []
    for arguments in (
        ("M2_minus_M0", M2_ID, M0_ID, "grf"),
        (
            "M2_minus_MEDICC2",
            M2_ID,
            MEDICC2_DEVELOPMENT_ARM_ID,
            "grf",
        ),
        ("M6_minus_M5", M6_ID, M5_ID, "ad_f1"),
        ("M7_minus_Pooled-Q", M7_ID, POOLED_Q_ID, "ad_f1"),
    ):
        rows.extend(
            _contrast_by_height(
                inputs,
                contrast_id=arguments[0],
                left_arm=arguments[1],
                right_arm=arguments[2],
                metric=arguments[3],
            )
        )
    return rows


def build_resource_summary(inputs: PublicationInputs) -> dict[str, Any]:
    distance_resources = {
        int(record["block_index"]): record["resources"]
        for record in inputs.bank["distance_blocks"]
        if record.get("status") == "success"
    }
    m2_resources: dict[int, list[Mapping[str, Any]]] = {
        block_index: [] for block_index in range(100)
    }
    medicc2_resources: dict[int, list[Mapping[str, Any]]] = {
        block_index: [] for block_index in range(100)
    }
    for block_index in range(100):
        for condition_id in STANDARD_CONDITION_IDS:
            internal = inputs.internal_index[(block_index, condition_id, M2_ID)]
            external = inputs.medicc2_index[
                (block_index, condition_id, MEDICC2_DEVELOPMENT_ARM_ID)
            ]
            if internal.get("status") != "success" or external.get("status") != "success":
                raise ValueError("Resource publication requires complete standard records.")
            m2_resources[block_index].append(internal["resources"]["reconstruction"])
            medicc2_resources[block_index].append(
                external["resources"]["reconstruction"]
            )
    return conservative_m2_resource_accounting(
        distance_resources=distance_resources,
        m2_resources=m2_resources,
        medicc2_resources=medicc2_resources,
    )


def _read_primary_rows(inputs: PublicationInputs) -> list[dict[str, Any]]:
    path = inputs.report_root / "primary_block_effects.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100 or any(row.get("complete") != "True" for row in rows):
        raise ValueError("Primary block-effect table is incomplete.")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty publication table: {path.name}.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_resource_csv(path: Path, resources: Mapping[str, Any]) -> None:
    rows = []
    for stage, summary in resources.items():
        if stage == "accounting_rule":
            continue
        rows.append(
            {
                "schema_version": PUBLICATION_SCHEMA_VERSION,
                "stage": stage,
                "count": summary["count"],
                "wall_mean_seconds": summary["wall_seconds"]["mean"],
                "wall_median_seconds": summary["wall_seconds"]["median"],
                "wall_maximum_seconds": summary["wall_seconds"]["maximum"],
                "rss_mean_bytes": summary["peak_rss_bytes"]["mean"],
                "rss_median_bytes": summary["peak_rss_bytes"]["median"],
                "rss_maximum_bytes": summary["peak_rss_bytes"]["maximum"],
            }
        )
    _write_csv(path, rows)


def render_main_figure(
    *,
    figure_rows: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    output_pdf: Path,
    output_png: Path,
) -> None:
    colors = {
        M0_ID: "#6B7280",
        MEDICC2_DEVELOPMENT_ARM_ID: "#D97706",
        M2_ID: "#087F8C",
        POOLED_Q_ID: "#6B7280",
        M5_ID: "#D97706",
        M6_ID: "#087F8C",
        M7_ID: "#263B70",
    }
    markers = {
        M0_ID: "o",
        MEDICC2_DEVELOPMENT_ARM_ID: "s",
        M2_ID: "D",
        POOLED_Q_ID: "o",
        M5_ID: "s",
        M6_ID: "D",
        M7_ID: "^",
    }
    by_cell = {
        (str(row["arm_id"]), int(row["height"])): row for row in figure_rows
    }
    heights = [14, 24, 34]

    figure = plt.figure(figsize=(8.2, 6.7), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.08))
    partial_axis = figure.add_subplot(grid[0, 0])
    effect_axis = figure.add_subplot(grid[0, 1])
    full_axis = figure.add_subplot(grid[1, :])

    for arm_id in PARTIAL_FIGURE_METHODS:
        cells = [by_cell[(arm_id, height)] for height in heights]
        means = [float(cell["mean"]) for cell in cells]
        lower = [
            mean - float(cell["bootstrap_lower"])
            for mean, cell in zip(means, cells, strict=True)
        ]
        upper = [
            float(cell["bootstrap_upper"]) - mean
            for mean, cell in zip(means, cells, strict=True)
        ]
        partial_axis.errorbar(
            heights,
            means,
            yerr=[lower, upper],
            color=colors[arm_id],
            marker=markers[arm_id],
            linewidth=1.8,
            markersize=5.5,
            capsize=3,
            label=METHOD_LABEL[arm_id],
        )
    partial_axis.set_title("A  Partial-tree recovery", loc="left", fontweight="bold")
    partial_axis.set_xlabel("True-tree height")
    partial_axis.set_ylabel("Rooted GRF similarity")
    partial_axis.set_xticks(heights)
    partial_axis.grid(axis="y", color="#D8DEE5", linewidth=0.7)
    partial_axis.legend(frameon=False, fontsize=8)

    effects = sorted(float(row["block_effect"]) for row in primary_rows)
    effect_axis.scatter(
        range(1, len(effects) + 1),
        effects,
        s=12,
        color="#263B70",
        alpha=0.85,
        linewidths=0,
    )
    primary = report["primary"]
    mean = float(primary["block_effect_summary"]["mean"])
    lower = float(primary["bootstrap_interval"]["lower"])
    upper = float(primary["bootstrap_interval"]["upper"])
    effect_axis.axhline(0, color="#7A1F1F", linewidth=1, linestyle="--")
    effect_axis.axhspan(lower, upper, color="#B8DED9", alpha=0.8)
    effect_axis.axhline(mean, color="#087F8C", linewidth=2)
    effect_axis.set_title("B  Registered block effect", loc="left", fontweight="bold")
    effect_axis.set_xlabel("Truth blocks, ordered by effect")
    effect_axis.set_ylabel(r"Mean $M_2-M_0$ rooted-GRF effect")
    effect_axis.set_xlim(0, 101)
    effect_axis.grid(axis="y", color="#D8DEE5", linewidth=0.7)
    effect_axis.text(
        0.03,
        0.97,
        f"mean {mean:.3f}  (95% CI {lower:.3f}–{upper:.3f})\n"
        f"block W/T/L 100/0/0",
        transform=effect_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#17202A",
    )

    for arm_id in FULL_FIGURE_METHODS:
        cells = [by_cell[(arm_id, height)] for height in heights]
        means = [float(cell["mean"]) for cell in cells]
        lower = [
            mean - float(cell["bootstrap_lower"])
            for mean, cell in zip(means, cells, strict=True)
        ]
        upper = [
            float(cell["bootstrap_upper"]) - mean
            for mean, cell in zip(means, cells, strict=True)
        ]
        full_axis.errorbar(
            heights,
            means,
            yerr=[lower, upper],
            color=colors[arm_id],
            marker=markers[arm_id],
            linewidth=1.8,
            markersize=5.5,
            capsize=3,
            label=METHOD_LABEL[arm_id],
        )
    full_axis.set_title("C  Closed-state recovery", loc="left", fontweight="bold")
    full_axis.set_xlabel("True-tree height")
    full_axis.set_ylabel("AD-F1")
    full_axis.set_xticks(heights)
    full_axis.grid(axis="y", color="#D8DEE5", linewidth=0.7)
    full_axis.legend(frameon=False, fontsize=8, ncol=4, loc="best")

    for axis in (partial_axis, effect_axis, full_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.title.set_fontsize(10)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)

    metadata = {
        "Title": "CTBF v5 independent-confirmation results",
        "Author": "CTBF",
        "Subject": (
            "Prospectively locked partial rooted-GRF and closed-state AD-F1 "
            "results over 100 independent truth blocks"
        ),
        "Keywords": "CTBF; copy number; longitudinal biopsy; confirmation",
        "Creator": "algorithm_evaluation.v5_confirmation_publication",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(
        output_pdf,
        format="pdf",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata=metadata,
    )
    figure.savefig(
        output_png,
        format="png",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)


def _write_readme(path: Path, evidence: Mapping[str, Any]) -> None:
    primary = evidence["registered_primary"]
    resources = evidence["resource_accounting"]
    m2 = resources["m2_conservative_amortized_standard_case"]
    medicc2 = resources["medicc2_block_mean_standard_case"]
    path.write_text(
        "\n".join(
            (
                "# CTBF v5 confirmation publication package",
                "",
                "This directory contains read-only publication derivatives of the ",
                "checksum-closed `confirmation_v1` evidence. It does not replace or ",
                "modify the registered report.",
                "",
                "- `main-figure.pdf`: vector manuscript figure.",
                "- `main-figure.png`: raster preview.",
                "- `main-figure-data.csv`: absolute score means and block-bootstrap intervals.",
                "- `contrast-by-height.csv`: paired height-specific descriptive contrasts.",
                "- `primary-block-effects.csv`: exact registered block effects copied from the report.",
                "- `resource-summary.csv`: stored M2, shared-D, and MEDICC2 resources.",
                "- `evidence-summary.json`: evidence identities, registered result, secondary results, and accounting rules.",
                "- `checksums.sha256`: closure over every preceding artifact.",
                "",
                f"Registered mean M2-minus-M0 block effect: {primary['block_effect_summary']['mean']:.6f}.",
                f"Conservative M2 time per standard case: {m2['wall_seconds']['mean']:.3f} s.",
                f"MEDICC2 mean time per standard case: {medicc2['wall_seconds']['mean']:.3f} s.",
                "",
                "The M2 time charges the complete block-wide cnp2cnp stage across only the three standard cases, although that matrix also served five sensitivity conditions and every internal method. Peak RSS is the maximum across sequential stages, not their sum.",
                "",
            )
        ),
        encoding="utf-8",
    )


def generate_publication_package(
    *,
    bank_root: Path | str,
    run_root: Path | str,
    medicc2_root: Path | str,
    report_root: Path | str,
    output_root: Path | str,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Publication output already exists: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)

    inputs = load_publication_inputs(
        bank_root=bank_root,
        run_root=run_root,
        medicc2_root=medicc2_root,
        report_root=report_root,
        manifest_path=manifest_path,
    )
    figure_rows = build_figure_rows(inputs)
    contrast_rows = build_contrast_rows(inputs)
    resource_summary = build_resource_summary(inputs)
    primary_rows = _read_primary_rows(inputs)

    evidence = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "scientific_role": "read_only_publication_derivative",
        "independent_unit": "truth_block",
        "input_evidence": {
            "manifest_sha256": inputs.validation["manifest_sha256"],
            "protocol_sha256": inputs.validation["protocol_sha256"],
            "source_lock_sha256": inputs.validation["source_lock"]["sha256"],
            "bank_checksum_sha256": inputs.bank["checksum_closure"]["sha256"],
            "internal_run_checksum_sha256": inputs.run["checksum_closure"]["sha256"],
            "medicc2_run_checksum_sha256": inputs.medicc2_run["checksum_closure"]["sha256"],
            "source_report_medicc2_run_checksum_sha256": inputs.report[
                "secondary_medicc2"
            ].get("run_checksum_sha256"),
            "report_checksum_sha256": inputs.report["checksum_closure"]["sha256"],
            "report_summary_sha256": file_sha256(inputs.report_root / "summary.json"),
        },
        "registered_primary": inputs.report["primary"],
        "prespecified_descriptive_results": {
            "standard_contrasts": inputs.report["secondary_standard_contrasts"],
            "biopsy_count": inputs.report["secondary_biopsy_count"],
            "placement": inputs.report["secondary_placement"],
            "medicc2": inputs.report["secondary_medicc2"],
            "absolute_standard_scores": inputs.report["absolute_standard_scores"],
        },
        "figure_interval_rule": (
            "Two-sided percentile 95% bootstrap interval from 100000 "
            "truth-block resamples; descriptive for absolute-score cells."
        ),
        "resource_accounting": resource_summary,
        "interpretation_contract": inputs.report["interpretation_contract"],
        "provenance_note": (
            "The source report serialized secondary_medicc2.run_checksum_sha256 "
            "as null although report generation validated the MEDICC2 closure. "
            "This derivative independently revalidated that closure and records "
            "its actual digest in input_evidence."
        ),
    }

    with tempfile.TemporaryDirectory(
        prefix="ctbf-v5-confirmation-publication-", dir=output.parent
    ) as temporary:
        temporary_root = Path(temporary) / output.name
        temporary_root.mkdir()
        _write_csv(temporary_root / "main-figure-data.csv", figure_rows)
        _write_csv(temporary_root / "contrast-by-height.csv", contrast_rows)
        _write_csv(temporary_root / "primary-block-effects.csv", primary_rows)
        _write_resource_csv(temporary_root / "resource-summary.csv", resource_summary)
        write_json(temporary_root / "evidence-summary.json", evidence)
        render_main_figure(
            figure_rows=figure_rows,
            primary_rows=primary_rows,
            report=inputs.report,
            output_pdf=temporary_root / "main-figure.pdf",
            output_png=temporary_root / "main-figure.png",
        )
        _write_readme(temporary_root / "README.md", evidence)
        write_checksum_closure(temporary_root, PUBLICATION_CHECKSUM_NAME)
        temporary_root.replace(output)

    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank-root", type=Path, default=DEFAULT_CONFIRMATION_ROOT / "bank"
    )
    parser.add_argument(
        "--run-root", type=Path, default=DEFAULT_CONFIRMATION_ROOT / "internal"
    )
    parser.add_argument(
        "--medicc2-root", type=Path, default=DEFAULT_CONFIRMATION_ROOT / "medicc2"
    )
    parser.add_argument(
        "--report-root", type=Path, default=DEFAULT_CONFIRMATION_ROOT / "report"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    evidence = generate_publication_package(
        bank_root=arguments.bank_root,
        run_root=arguments.run_root,
        medicc2_root=arguments.medicc2_root,
        report_root=arguments.report_root,
        output_root=arguments.output_root,
        manifest_path=arguments.manifest,
    )
    primary = evidence["registered_primary"]
    print(
        "complete: publication package; "
        f"{primary['complete_block_count']} primary blocks; "
        f"registered support={primary['registered_support']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
