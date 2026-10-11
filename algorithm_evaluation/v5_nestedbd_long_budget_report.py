"""Descriptive sampler diagnostics only: no truth, GRF or convergence verdict."""

from __future__ import annotations

import math
from pathlib import Path

from algorithm_evaluation import v5_nestedbd_long_pilot as pilot


def successful(run: dict) -> bool:
    return run["status"] == "technical_complete" and run["phase"] in {"pilot", "long"}


def finite(value):
    return float(value) if isinstance(value, (float, int)) and math.isfinite(value) else None


def scalar_row(run: dict, budget: int) -> dict:
    window = run.get("scalar_windows", {}).get("10", {})
    stats = window.get("official_beast_statistics", {})
    resource = run.get("execution", {})
    seconds = finite(resource.get("wall_seconds"))
    posterior = finite(stats.get("posterior", {}).get("ESS"))
    return {"case_id": run["case_id"], "block_index": run["block_index"], "height": run["height"],
            "replicate": run["replicate"], "seed": run["seed"], "budget_iterations": budget,
            "executed_phase": run["phase"], "status": run["status"], "success": successful(run),
            "posterior_ess": posterior, "likelihood_ess": finite(stats.get("likelihood", {}).get("ESS")),
            "height_ess": finite(stats.get("TreeHeight", {}).get("ESS")),
            "posterior_ess_per_million": posterior / (budget / 1e6) if posterior is not None else None,
            "posterior_ess_per_minute": posterior * 60 / seconds if posterior is not None and seconds and seconds > 0 else None,
            "posterior_first_half_mean": finite(window.get("posterior_first_half_mean")),
            "posterior_second_half_mean": finite(window.get("posterior_second_half_mean")),
            "wall_seconds": seconds, "peak_rss_bytes": resource.get("peak_rss_bytes"),
            "attempt": run.get("attempt"), "error": run.get("error")}


def _windows(run: dict):
    values = pilot.read_json(Path(run["attempt"]) / "clade_windows.json")
    if set(values) != {"10", "25", "50"}:
        raise ValueError("Missing declared clade diagnostic window.")
    return {int(burn): window for burn, window in values.items()}


def make_report(backend, cases: list[dict], old: list[dict], new: list[dict]) -> dict:
    expected = {(case["case_id"], seed["seed"]) for case in cases for seed in case["seeds"]}
    indices = []
    rows, pairs, between_seeds, across_budgets = [], [], [], []
    for budget, runs in ((2000000, old), (10000000, new)):
        index = {(run["case_id"], run["seed"]): run for run in runs}
        if len(index) != len(runs) or set(index) != expected:
            raise ValueError("Missing or duplicate declared seed; do not select or replace it.")
        indices.append(index)
        for case in cases:
            ordered = [index[(case["case_id"], seed["seed"])] for seed in case["seeds"]]
            if [row["replicate"] for row in ordered] != [1, 2]:
                raise ValueError("Replicate identity differs from declared seed order.")
            rows.extend(scalar_row(run, budget) for run in ordered)
            complete = all(successful(run) for run in ordered)
            between_seeds.append({"case_id": case["case_id"], "budget_iterations": budget,
                                  "status": "success" if complete else "incomplete",
                                  "windows": backend.audit.review.compare_windows(*map(_windows, ordered)) if complete else None})
    for case in cases:
        for seed in case["seeds"]:
            key = (case["case_id"], seed["seed"])
            a, b = indices[0][key], indices[1][key]
            complete = successful(a) and successful(b)
            identity = {"case_id": key[0], "seed": key[1], "replicate": seed["replicate"],
                        "block_index": case["block_index"], "height": case["height"]}
            left, right = scalar_row(a, 2000000), scalar_row(b, 10000000)
            changes = {}
            for field in ("posterior_ess", "likelihood_ess", "height_ess", "posterior_ess_per_million", "posterior_ess_per_minute"):
                changes[field] = (right[field] - left[field] if complete and left[field] is not None
                                  and right[field] is not None else None)
            pairs.append({**identity, "complete": complete, "ten_million_minus_two_million": changes})
            across_budgets.append({**identity, "status": "success" if complete else "incomplete",
                                  "windows": backend.audit.review.compare_windows(_windows(a), _windows(b)) if complete else None})
    return {"status": "bounded_budget_diagnostic_complete_not_convergence",
            "independent_histories": len({case["block_index"] for case in cases}),
            "input_count": len(cases), "planned_new_runs": len(new),
            "new_success_count": sum(successful(run) for run in new),
            "old_runs": old, "new_runs": sorted(new, key=lambda run: (run["case_id"], run["replicate"])),
            "scalar_table": rows, "paired_scalar_changes": pairs,
            "between_seed_clades": between_seeds, "within_seed_across_budget_clades": across_budgets,
            "convergence_established": False, "p_values": False, "accuracy_scoring": False,
            "same_seed_prefix_identity_assumed": False,
            "interpretation": "Two selected histories, not a representative convergence certification. All seeds and failures retained. No automated change to the main 2M budget."}


def _num(value, digits=2):
    return "NA" if value is None else f"{value:.{digits}f}"


def report_text(report: dict) -> str:
    lines = ["# NestedBD-Long: bounded computation-budget diagnostic", "",
             "Original histories 001–002, H14/H24, both original seeds. The old 2M runs are read-only; new 10M runs restart from zero.",
             "Only chainLength changes in XML. Inputs, clock, priors, heating, log interval and primary 10% burn-in are unchanged.",
             "The main benchmark remains at 2M. No GRF, tree-accuracy selection, posterior pooling, favored seed or automatic convergence verdict.",
             f"New 10M runs successful: {report['new_success_count']}/{report['planned_new_runs']}. Independent histories: {report['independent_histories']}.",
             "Identical random seeds do not guarantee identical prefixes in parallel coupled-chain executions.", "",
             "## Scalar diagnostics at 10% burn-in", "",
             "ESS is reported with ESS per million iterations and per minute. Larger raw ESS alone need not mean better mixing efficiency.",
             "Phase identifies failed technical checks, for which no 10M run took place. Resource columns describe each JVM, not CTBF pipeline costs.", "",
             "| Case | Replicate | Budget | Phase/status | Posterior ESS | Likelihood ESS | Height ESS | ESS/M iterations | ESS/min | Minutes | Peak MiB |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["scalar_table"]:
        minutes = row["wall_seconds"] / 60 if row["wall_seconds"] is not None else None
        memory = row["peak_rss_bytes"] / 1024**2 if row["peak_rss_bytes"] is not None else None
        numbers = [row[key] for key in ("posterior_ess", "likelihood_ess", "height_ess", "posterior_ess_per_million", "posterior_ess_per_minute")]
        lines.append(f"| {row['case_id']} | {row['replicate']} | {row['budget_iterations'] // 1000000}M | "
                     f"{row['executed_phase']}/{row['status']} | " + " | ".join(map(_num, [*numbers, minutes, memory])) + " |")
    lines += ["", "## Within-run posterior drift (10% burn-in)", "",
              "Compare the first and second halves within a run, not posterior levels across different inputs.", "",
              "| Case | Replicate | Budget | First half mean | Second half mean |",
              "|---|---:|---:|---:|---:|"]
    for row in report["scalar_table"]:
        lines.append(f"| {row['case_id']} | {row['replicate']} | {row['budget_iterations'] // 1000000}M | "
                     f"{_num(row['posterior_first_half_mean'])} | {_num(row['posterior_second_half_mean'])} |")
    for heading, key, columns in (
        ("Between-seed descendant-tip clade frequencies", "between_seed_clades", "Budget"),
        ("Within-seed 2M versus 10M clade frequencies", "within_seed_across_budget_clades", "Replicate"),
    ):
        lines += ["", "## " + heading, "", "10% is primary; 25%/50% are diagnostic windows, not a choice of the best burn-in.", "",
                  f"| Case | {columns} | Burn-in | Shared majority clades | Majority counts | Maximum frequency difference |",
                  "|---|---:|---:|---:|---|---:|"]
        for row in report[key]:
            label = str(row["budget_iterations"] // 1000000) + "M" if "budget_iterations" in row else str(row["replicate"])
            if row["windows"] is None:
                lines.append(f"| {row['case_id']} | {label} | NA | incomplete | NA | NA |")
                continue
            for burn in ("10", "25", "50"):
                window = row["windows"][burn]
                lines.append(f"| {row['case_id']} | {label} | {burn}% | {window['shared_majority_clades']} | "
                             f"{window['majority_clades_per_seed']} | {_num(window['maximum_clade_frequency_difference'], 3)} |")
    lines += ["", "## Interpretation boundary", "",
              report["interpretation"],
              "Improved ESS, reduced within-run drift and closer between-run clade frequencies together support better sampling on these inputs, not proof of convergence or accuracy.",
              "Exact summary-tree equality is neither required nor tested. More iterations can leave sampling difficulties unresolved.",
              "The 10M runs use 10% of 10M as burn-in; the 2M runs use 10% of 2M. They therefore retain different absolute sampling windows.",
              "The released model's root/prior/sampling assumptions remain unchanged. This is not a model-repair experiment.",
              "Review this report before extending coverage to the remaining 90 histories. Do not silently replace the original 2M results with the longer runs.", ""]
    return "\n".join(lines)
