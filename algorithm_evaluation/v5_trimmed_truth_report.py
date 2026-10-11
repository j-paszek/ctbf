"""Descriptive paired summaries of unchanged full and new trimmed truth scores.

No registered primary decision is recalculated. Bootstrap units are complete
histories. NestedBD's two fixed seeds are averaged within an input only when
both are successful, never selected, pooled as trees, or treated as histories.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
from pathlib import Path
import statistics

from algorithm_evaluation.paper_pipeline_contract import write_json_atomic
from algorithm_evaluation.v5_confirmation_common import STANDARD_CONDITION_IDS
from algorithm_evaluation.v5_confirmation_report import _bootstrap_interval, _wins_ties_losses


REPORT_SCHEMA = "ctbf-v5-trimmed-truth-paired-report-v1"
TIE_TOLERANCE = 1e-12


def case_scores(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["case_id"], record["method"])].append(record)
    result = []
    for (case_id, method), rows in sorted(grouped.items()):
        first = rows[0]
        if method == "NestedBD-Long":
            if len(rows) != 2 or {row["replicate"] for row in rows} != {1, 2}:
                raise ValueError("Missing or duplicate NestedBD seed record.")
        elif len(rows) != 1:
            raise ValueError("Duplicate deterministic case/method result.")
        complete = all(row["status"] == "success" for row in rows)
        result.append({
            **{key: first[key] for key in ("case_id", "condition_id", "block_index", "height", "problem")},
            "method": method, "status": "success" if complete else "incomplete",
            "replicate_count": len(rows),
            **{metric: statistics.fmean(row[metric] for row in rows) if complete else None
               for metric in ("grf_full", "grf_trim")},
        })
    return result


def _effect(values: list[float], key: str, draws: int) -> dict:
    seed = int(hashlib.sha256(("ctbf-trimmed-truth-v1:" + key).encode()).hexdigest()[:15], 16)
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "wtl": _wins_ties_losses(values, TIE_TOLERANCE),
            "bootstrap": _bootstrap_interval(values, repetitions=draws, seed=seed)}


def _contrast(index: dict, blocks: list, conditions: tuple, left: str, right: str,
              draws: int) -> dict:
    values = {metric: [] for metric in ("grf_full", "grf_trim")}
    combined = {metric: [] for metric in values}
    block_rows = []
    for block in blocks:
        row = {"block_index": block, "conditions": {}}
        for condition in conditions:
            a, b = index.get((block, condition, left)), index.get((block, condition, right))
            complete = a is not None and b is not None and a["status"] == b["status"] == "success"
            row["conditions"][condition] = {
                metric: a[metric] - b[metric] if complete else None for metric in values}
            if complete:
                for metric in values:
                    combined[metric].append(a[metric] - b[metric])
        row["complete"] = all(item["grf_full"] is not None for item in row["conditions"].values())
        for metric in values:
            row[metric] = statistics.fmean(item[metric] for item in row["conditions"].values()) if row["complete"] else None
            if row["complete"]:
                values[metric].append(row[metric])
        block_rows.append(row)
    name = f"{left}-minus-{right}:" + ",".join(conditions)
    return {"name": name, "left": left, "right": right, "conditions": list(conditions),
            "declared_histories": len(blocks), "complete_histories": len(values["grf_full"]),
            "missing_histories": len(blocks) - len(values["grf_full"]),
            "effects": {metric: _effect(data, name, draws) for metric, data in values.items()},
            "condition_wtl": {metric: _wins_ties_losses(data, TIE_TOLERANCE)
                              for metric, data in combined.items()},
            "condition_pair_count": len(combined["grf_full"]), "blocks": block_rows}


def build_report(records: list[dict], *, bootstrap_draws: int = 100000) -> dict:
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    rows = case_scores(records)
    index = {(row["block_index"], row["condition_id"], row["method"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate block/condition/method.")
    absolute = []
    for method, condition in sorted({(row["method"], row["condition_id"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method and row["condition_id"] == condition]
        complete = [row for row in selected if row["status"] == "success"]
        absolute.append({"method": method, "condition": condition, "planned": len(selected),
                         "complete": len(complete), "missing": len(selected) - len(complete),
                         **{metric: statistics.fmean(row[metric] for row in complete) if complete else None
                            for metric in ("grf_full", "grf_trim")}})
    contrasts = []
    for left, right in (("M2", "M0"), ("M2", "MEDICC2"), ("M2", "NestedBD-Long"),
                        ("M0", "MEDICC2"), ("M0", "NestedBD-Long"), ("MEDICC2", "NestedBD-Long"),
                        ("M6", "M5"), ("M7", "Pooled-Q")):
        left_declared = {(row["block_index"], row["condition_id"]) for row in rows if row["method"] == left}
        right_declared = {(row["block_index"], row["condition_id"]) for row in rows if row["method"] == right}
        shared = left_declared & right_declared  # failures remain in the declared universe
        for condition in sorted({condition for _, condition in shared}):
            blocks = sorted(block for block, cid in shared if cid == condition)
            contrasts.append(_contrast(index, blocks, (condition,), left, right, bootstrap_draws))
        standards = tuple(condition for condition in STANDARD_CONDITION_IDS
                          if any(cid == condition for _, cid in shared))
        if len(standards) > 1:
            blocks = sorted({block for block, condition in shared if condition in standards})
            contrasts.append(_contrast(index, blocks, standards, left, right, bootstrap_draws))
    sampling = []
    count_conditions = ("H24_L2", "H24_rule_y", "H24_L4", "H24_L5")
    for method in sorted({row["method"] for row in rows}):
        comparisons = list(combinations(count_conditions, 2))
        comparisons += [("H24_late", "H24_rule_y"), ("H24_random", "H24_rule_y")]
        for low, high in comparisons:
            blocks = sorted({block for block, condition, name in index if name == method and condition in {low, high}})
            if not any((block, low, method) in index and (block, high, method) in index for block in blocks):
                continue
            effects = {metric: [] for metric in ("grf_full", "grf_trim")}
            for block in blocks:
                a, b = index.get((block, high, method)), index.get((block, low, method))
                if a and b and a["status"] == b["status"] == "success":
                    for metric in effects:
                        effects[metric].append(a[metric] - b[metric])
            name = f"{method}:{high}-minus-{low}"
            sampling.append({"name": name, "method": method, "left": high, "right": low,
                             "declared_histories": len(blocks),
                             "effects": {metric: _effect(data, name, bootstrap_draws)
                                         for metric, data in effects.items()}})
    return {"schema_version": REPORT_SCHEMA, "scientific_role": "complementary_truth_target_sensitivity",
            "registered_primary_decision_recomputed": False, "p_values": False,
            "independent_unit": "simulated_history", "bootstrap_draws": bootstrap_draws,
            "nestedbd_seed_aggregation": "mean_of_both_successful_seed_scores_within_input",
            "nestedbd_convergence_claim": False, "raw_record_count": len(records),
            "status_counts": dict(Counter(row["status"] for row in records)),
            "case_scores": rows, "absolute_scores": absolute, "paired_contrasts": contrasts,
            "sampling_contrasts": sampling,
            "sampling_caveat": "Trimmed truth is condition-specific because the observed-state union changes; sampling contrasts do not hold the target fixed.",
            "unavailable_methods": ["M4 is not in this frozen source portfolio; no M4 inference was run."]}


def _fmt(value):
    return "NA" if value is None else f"{value:.6f}"


def _effect_text(effect):
    interval = effect["bootstrap"]
    return f"{_fmt(effect['mean'])} [{_fmt(interval['lower'])}, {_fmt(interval['upper'])}]"


def _wtl(effect):
    value = effect["wtl"]
    return f"{value['wins']}/{value['ties']}/{value['losses']}"


def write_report(output: Path, records: list[dict], report: dict) -> None:
    write_json_atomic(output / "summary.json", report)
    fields = ["case_id", "block_index", "condition_id", "height", "method", "problem", "replicate",
              "seed", "source_status", "status", "grf_full", "grf_trim", "ad_f1"]
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (output / "case_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["case_scores"][0]))
        writer.writeheader()
        writer.writerows(report["case_scores"])
    lines = ["# Independent trimmed-truth GRF audit", "",
             "Original full-truth scores are preserved verbatim. Only truth is trimmed; reconstructed trees are unchanged.",
             "This is a complementary target-sensitivity analysis, not a replacement of the registered primary test.",
             "Only matched successful records enter both columns. Missing/failed outcomes remain explicit in JSON/CSV.",
             "NestedBD scores are averaged across both successful fixed seeds within an input. This makes no convergence claim.",
             "Intervals resample whole histories. Condition W/T/L counts do not imply independent conditions.",
             "", f"Raw records: {len(records)}. Status counts: {report['status_counts']}.",
             "", "## Absolute scores", "",
             "| Method | Condition | Complete/planned | Full truth | Trimmed truth |",
             "|---|---|---:|---:|---:|"]
    for row in report["absolute_scores"]:
        lines.append(f"| {row['method']} | {row['condition']} | {row['complete']}/{row['planned']} | {_fmt(row['grf_full'])} | {_fmt(row['grf_trim'])} |")
    lines += ["", "## Paired method effects (left minus right)", "",
              "| Contrast | Conditions | Complete histories | Full mean [95% CI] | Trimmed mean [95% CI] | Full block W/T/L | Trimmed block W/T/L |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["paired_contrasts"]:
        a, b = row["effects"]["grf_full"], row["effects"]["grf_trim"]
        lines.append(f"| {row['left']} − {row['right']} | {', '.join(row['conditions'])} | {row['complete_histories']}/{row['declared_histories']} | {_effect_text(a)} | {_effect_text(b)} | {_wtl(a)} | {_wtl(b)} |")
    lines += ["", "## Paired sampling effects", "", report["sampling_caveat"], "",
              "| Method | Contrast | Complete histories | Full mean [95% CI] | Trimmed mean [95% CI] |",
              "|---|---|---:|---:|---:|"]
    for row in report["sampling_contrasts"]:
        a, b = row["effects"]["grf_full"], row["effects"]["grf_trim"]
        lines.append(f"| {row['method']} | {row['left']} − {row['right']} | {a['n']}/{row['declared_histories']} | {_effect_text(a)} | {_effect_text(b)} |")
    lines += ["", "No new inference, cnp2cnp calculations, M4 runs, figures or primary p-values were produced.",
              "AD-F1 checks and source digests are retained in the per-case audit files. Condition-level W/T/L is in summary.json.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
