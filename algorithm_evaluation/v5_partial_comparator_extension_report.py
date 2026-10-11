"""Matched, history-paired reporting for five partial reconstruction methods."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import statistics

from algorithm_evaluation import v5_trimmed_truth_report as paired
from algorithm_evaluation import v5_partial_comparator_extension_common as common
from algorithm_evaluation.paper_pipeline_contract import write_json_atomic


METHODS = ("M0", "M2", "M4", "MEDICC2", "NestedBD-Long")


def build_report(records: list[dict], *, bootstrap_draws=100000) -> dict:
    common.validate_roster(records, include_new=True)
    report = paired.build_report(records, bootstrap_draws=bootstrap_draws)
    report.update(schema_version=common.SCHEMA, scientific_role="additive_fixed_budget_partial_comparator_extension",
                  unavailable_methods=["NestedBD-Long: no H34 or biopsy-sensitivity conditions.",
                                       "MEDICC2: no biopsy-sensitivity conditions."],
                  original_scores_preserved=True, reused_comparator_scores_preserved=True,
                  nested_inference_budget_iterations=common.nested10m.ITERATIONS,
                  nested_2m_scores_included=common._PROFILE == "2m", convergence_established=False,
                  whole_pipeline_resource_comparison=False)
    index = {(r["block_index"], r["condition_id"], r["method"]): r for r in report["case_scores"]}
    for left, right in (("M2", "M4"), ("M4", "M0"), ("M4", "MEDICC2"), ("M4", "NestedBD-Long")):
        shared = {(block, condition) for block, condition, method in index
                  if method == left and (block, condition, right) in index}
        for condition in sorted({condition for _, condition in shared}):
            blocks = sorted({block for block, cid in shared if cid == condition})
            report["paired_contrasts"].append(paired._contrast(index, blocks, (condition,), left, right, bootstrap_draws))
        standards = tuple(c for c in paired.STANDARD_CONDITION_IDS if any(cid == c for _, cid in shared))
        if len(standards) > 1:
            blocks = sorted({block for block, cid in shared if cid in standards})
            report["paired_contrasts"].append(paired._contrast(index, blocks, standards, left, right, bootstrap_draws))
    # A five-method panel must not mix unmatched successful subsets.
    matched_common = []
    blocks = sorted({row["block_index"] for row in records})
    for condition in common.NESTED_CONDITIONS:
        complete = [b for b in blocks if all(index.get((b, condition, m), {}).get("status") == "success" for m in METHODS)]
        matched_common.append({"condition": condition, "planned_histories": len(blocks),
                       "complete_histories": len(complete), "block_indices": complete,
                       "scores": {m: {k: statistics.fmean(index[b, condition, m][k] for b in complete) if complete else None
                                      for k in ("grf_full", "grf_trim")} for m in METHODS}})
    report["five_method_matched_scores"] = matched_common
    report["record_counts_by_method"] = {
        m: dict(Counter(row["status"] for row in records if row["method"] == m)) for m in METHODS}
    return report


def write_report(directory: Path, records: list[dict], report: dict) -> None:
    write_json_atomic(directory / "report.json", report)
    fields = ["case_id", "block_index", "condition_id", "height", "method", "problem", "replicate",
              "seed", "source_status", "status", "grf_full", "grf_trim", "ad_f1",
              "inference_budget_iterations", "provenance"]
    for filename, data, columns in (("scores.csv", records, fields),
                                   ("case_scores.csv", report["case_scores"], list(report["case_scores"][0]))):
        with (directory / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
    f, e, w = paired._fmt, paired._effect_text, paired._wtl
    lines = ["# Five-method partial-tree comparison: 100 frozen histories", "",
        "M0: pooled classical NJ. M2: adaptive/deferred bottom + anticentral binary top.",
        "M4: the same adaptive/deferred bottom + classical NJ top.",
        "MEDICC2 and NestedBD-Long are topology-only contextual comparators.", "",
        "Existing M0/M2/MEDICC2 scores are reused unchanged. M4 is new on all 800 conditions.",
        (f"All 100 histories use the same {common.nested10m.LABEL} budget for both NestedBD seeds. "
         + ("All 400 seed runs are new."
            if common._PROFILE == "2m" else
            "Eight completed 10M diagnostic runs are reused, and 392 further runs are declared.")),
        (f"All 400 summary trees and scores use the {common.nested10m.LABEL} traces. "
         + ("Earlier ten-history 2M results remain separate and are not mixed into this series."
            if common._PROFILE == "2m" else
            "The old 2M results remain untouched in their separate source roots.")),
        (f"The summary uses {common.nested10m.SUMMARY['total_samples']:,} samples, discards "
         f"{common.nested10m.SUMMARY['discarded_samples']:,} (10%), and retains "
         f"{common.nested10m.SUMMARY['retained_samples']:,} per successful run."),
        "Both seed scores are averaged within each input only",
        "when both succeed. No better-seed choice, posterior pooling, convergence claim or primary-test rewrite.",
        "H34 is not available for NestedBD. Biopsy sensitivities compare M0/M2/M4 only.",
        "Each interval is a descriptive 95% paired bootstrap interval resampling whole histories.",
        "Condition W/T/L is an additional count, not a set of independent histories or a significance test.",
        "Positive differences favor the left method. GRF is similarity, not percent topology recovered.", "",
        f"Records: {len(records)}. Outcomes: {report['record_counts_by_method']}.", "",
        "## Absolute scores by declared condition", "",
        "These means use each method's successful inputs. Use the matched panel below if coverage differs.", "",
        "| Method | Condition | Complete/planned | Full truth | Trimmed truth |",
        "|---|---|---:|---:|---:|"]
    for row in report["absolute_scores"]:
        lines.append(f"| {row['method']} | {row['condition']} | {row['complete']}/{row['planned']} | {f(row['grf_full'])} | {f(row['grf_trim'])} |")
    lines += ["", "## Matched five-method panel", "",
        "Each row uses exactly the same histories for all five methods, with both NestedBD seeds.", "",
        "| Condition | Method | Shared histories | Full truth | Trimmed truth |", "|---|---|---:|---:|---:|"]
    for row in report["five_method_matched_scores"]:
        for method in METHODS:
            value = row["scores"][method]
            lines.append(f"| {row['condition']} | {method} | {row['complete_histories']}/{row['planned_histories']} | {f(value['grf_full'])} | {f(value['grf_trim'])} |")
    lines += ["", "## Paired method effects", "",
        "Combined rows average the indicated conditions within a history before bootstrapping.",
        "For NestedBD, H14/H24 provide at most 200 condition pairs but only 100 independent histories.",
        "Three-height comparisons provide at most 300 condition pairs, still only 100 histories.", "",
        "| Contrast | Conditions | Histories | Full mean [95% CI] | Trimmed mean [95% CI] | Full history W/T/L | Trim history W/T/L | Full condition W/T/L | Trim condition W/T/L |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["paired_contrasts"]:
        a, b = row["effects"]["grf_full"], row["effects"]["grf_trim"]
        cw = ["/".join(str(row["condition_wtl"][metric][key]) for key in ("wins", "ties", "losses"))
              for metric in ("grf_full", "grf_trim")]
        lines.append(f"| {row['left']} − {row['right']} | {', '.join(row['conditions'])} | {row['complete_histories']}/{row['declared_histories']} | {e(a)} | {e(b)} | {w(a)} | {w(b)} | {cw[0]} | {cw[1]} |")
    lines += ["", "## Biopsy sensitivities", "", report["sampling_caveat"], "",
        "| Method | Contrast | Histories | Full mean [95% CI] | Trimmed mean [95% CI] |", "|---|---|---:|---:|---:|"]
    for row in report["sampling_contrasts"]:
        a, b = row["effects"]["grf_full"], row["effects"]["grf_trim"]
        lines.append(f"| {row['method']} | {row['left']} − {row['right']} | {a['n']}/{row['declared_histories']} | {e(a)} | {e(b)} |")
    lines += ["", "## Boundaries", "",
        "MCMC scalar windows and between-seed clade disagreements for all 200 inputs are in diagnostics.json.",
        (f"Technical success is not convergence. This separate {common.nested10m.LABEL} series "
         "does not overwrite protected comparator scores."),
        "Failures and unsupported CN values are retained without clipping, replacement or favorable-seed selection.",
        "Resource records describe the executed stages. Cached input D excludes distance-computation cost.",
        "Do not infer whole-pipeline speed or memory superiority from these stage records. Use the separate",
        "condition-specific distance audit for any later end-to-end resource comparison.", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
