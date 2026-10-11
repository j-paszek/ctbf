"""Export current paper table values from stored evidence, never from TeX."""
from __future__ import annotations

import csv
import json
from pathlib import Path

CLOSED = ("M1", "Pooled-Q", "M5", "M6", "M7")
HEIGHTS = (14, 24, 34)


def alias(value):
    return str(value).replace("Pooled-Q", "M3")


def one(rows, **where):
    found = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in where.items())]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one row for {where}, got {len(found)}")
    return found[0]


def flat_effect(row):
    """Normalize stored schemas without recalculating any interval or score."""
    point = row.get("effect_summary", row)
    interval = row.get("bootstrap_interval", row.get("bootstrap", row))
    counts = row.get("wins_ties_losses", row.get("wtl", row))
    return dict(count=point.get("count", point.get("n")), mean=point["mean"], median=point.get("median"),
                lower=interval.get("lower", interval.get("bootstrap_lower")),
                upper=interval.get("upper", interval.get("bootstrap_upper")),
                **{key: counts.get(key) for key in ("wins", "ties", "losses")})


def synthetic_tables(bundle):
    v1, v3, v5, v6 = [bundle.json(k) for k in ("v1_summary", "v3_audit", "v5_audit", "v6_audit")]
    desc = v1["prespecified_descriptive_results"]
    means, contrasts = bundle.csv("v1_means"), bundle.csv("v1_contrasts")
    counts = bundle.csv("v3_counts") + bundle.csv("m4_counts")
    extension = bundle.json("extension_report")["absolute_scores"]
    tables = {}

    def add(table, panel, method, condition, metric, values, source, digits=6):
        tables.setdefault(table, []).append(dict(panel=panel, method=alias(method),
            condition=str(condition), metric=metric, **values,
            source_key=source, paper_decimal_places=digits))

    def effect(table, panel, method, condition, metric, row, source, digits=6):
        add(table, panel, method, condition, metric, flat_effect(row), source, digits)

    def sampling(method, left, right, metric):
        entry = one(v5["sampling_contrasts"], method=method, left=left, right=right)
        return entry["effects"][metric]

    for method in ("M0", "M2", "M4", "MEDICC2"):
        for height in HEIGHTS:
            condition = f"H{height}_rule_y"
            if method == "M4":
                row = one(extension, method=method, condition=condition)
                if row["complete"] != 100:
                    raise ValueError("All-history M4 table requires 100 histories")
                values, source = dict(count=row["complete"], mean=row["grf_full"]), "extension_report"
            else:
                row = one(means, paper_id=method, condition_id=condition, metric="grf")
                values, source = dict(count=row["count"], mean=row["mean"]), "v1_means"
            add("confirmatory-partial", "means", method, height, "grf_full", values, source)
    for name in ("M2_minus_M0", "M2_minus_MEDICC2"):
        for height in HEIGHTS:
            effect("confirmatory-partial", "contrasts", name, height, "grf_full",
                   one(contrasts, contrast_id=name, height=height), "v1_contrasts")
        if name == "M2_minus_M0":
            primary = v1["registered_primary"]
            block = dict(effect_summary=primary["block_effect_summary"],
                         bootstrap_interval=primary["bootstrap_interval"],
                         wins_ties_losses=primary["block_wins_ties_losses"])
        else:
            block = desc["medicc2"]["block"]
        effect("confirmatory-partial", "contrasts", name, "block_mean", "grf_full", block, "v1_summary")

    for method in CLOSED:
        for height in HEIGHTS:
            # The original plot omits M1, but the supplementary table includes it.
            row = desc["absolute_standard_scores"][method][f"H{height}_rule_y"]["ad_f1"]
            add("confirmatory-full", "means", method, height, "ad_f1",
                dict(count=row["count"], mean=row["mean"]), "v1_summary")
    for name in ("M6_minus_M5", "M7_minus_Pooled-Q"):
        for height in HEIGHTS:
            effect("confirmatory-full", "contrasts", name, height, "ad_f1",
                   one(contrasts, contrast_id=name, height=height), "v1_contrasts")
        effect("confirmatory-full", "contrasts", name, "block_mean", "ad_f1",
               desc["standard_contrasts"][name]["block"], "v1_summary")

    for source, panel, metric in (("v4_means", "means", "grf_full"),
                                  ("closed_trimmed", "trimmed_means", "grf_trim")):
        for row in bundle.csv(source):
            height = row.get("height") or row["condition_id"].split("_")[0][1:]
            add("confirmatory-full-grf", panel, row["paper_id"], height, metric,
                dict(count=row["count"], mean=row["mean"]), source)
    for row in bundle.csv("v4_contrasts"):
        effect("confirmatory-full-grf", "contrasts", row["contrast"], row["height"],
               "grf_full", row, "v4_contrasts")
    for row in bundle.csv("native_counts"):
        add("native-biopsy-count", "means", row["paper_id"], row["condition_id"], "ad_f1",
            dict(count=row["count"], mean=row["mean"]), "native_counts")
    for row in v6["native_ad_f1_five_minus_two"]:
        effect("native-biopsy-count", "five_minus_two", row["paper_id"], "5_minus_2",
               "ad_f1", row, "v6_audit")
    for row in counts:
        add("confirmatory-biopsy-count", "means", row["paper_id"], row["biopsies"],
            row["metric"], dict(count=row["count"], mean=row["mean"]),
            "m4_counts" if row["paper_id"] == "M4" else "v3_counts")
    for method in ("M0", "M2", "M4", *CLOSED):
        row = sampling(method, "H24_L5", "H24_L2", "grf_full") if method == "M4" else v3["endpoints_five_minus_two"][method]
        metric = "grf_full" if method in ("M0", "M2", "M4") else "fixed_U2_ad_f1"
        effect("confirmatory-biopsy-gains", "five_minus_two", method, "5_minus_2", metric,
               row, "v5_audit" if method == "M4" else "v3_audit")
    for name, row in v3["gain_contrasts"].items():
        effect("confirmatory-biopsy-gains", "difference_of_changes", name, "5_minus_2",
               "fixed_U2_ad_f1", row, "v3_audit")
    for method in ("M0", "M2", "M4", *CLOSED):
        for placement in ("late", "random"):
            condition = "H24_rule_y_minus_H24_" + placement
            if method == "M4":
                # Added with the original placement-family seed stream, not
                # the extension report's independently bootstrapped intervals.
                row, metric, source = bundle.json("m4_placement")["M4"][condition], "grf_full", "m4_placement"
            else:
                item = desc["placement"][method]
                row, metric, source = item["comparisons"][condition], item["metric"], "v1_summary"
            effect("confirmatory-placement", "contrasts", method, condition, metric, row, source, 5)

    resources = bundle.json("resources")
    if resources["case_count"] != 300 or not resources["publication_eligible"]:
        raise ValueError("Condition-specific resource audit is incomplete")
    for stage in ("m2_condition_specific_distance", "m2_reconstruction", "m2_condition_specific_pipeline", "medicc2_pipeline"):
        row = resources[stage]
        values = dict(count=row["count"], seconds_mean=row["wall_seconds"]["mean"],
            seconds_median=row["wall_seconds"]["median"], seconds_maximum=row["wall_seconds"]["maximum"],
            rss_MB_mean=row["peak_rss_bytes"]["mean"] / 1e6, rss_MB_maximum=row["peak_rss_bytes"]["maximum"] / 1e6)
        add("confirmatory-resources", "resources", stage, "300_standard_inputs", "time_and_memory", values, "resources", "seconds:3;MB:1")
    for row in bundle.csv("v5_means"):
        add("partial-extension-scores", "matched_means", row["paper_id"], row["height"],
            row["metric"], dict(count=row["count"], mean=row["mean"]), "v5_means", 5)
    wanted = [("M2", "M4", HEIGHTS), ("M4", "M0", HEIGHTS), ("M4", "MEDICC2", HEIGHTS),
              ("M2", "NestedBD-Long", (14,)), ("M2", "NestedBD-Long", (24,)),
              ("M2", "NestedBD-Long", (14, 24)), ("M4", "NestedBD-Long", (14, 24))]
    for left, right, heights in wanted:
        conditions = [f"H{h}_rule_y" for h in heights]
        selected = [r for r in v5["paired_contrasts"] if r["left"] == left and r["right"] == right and r["conditions"] == conditions]
        if len(selected) != 1:
            raise ValueError(f"Missing/duplicate paired effect: {left}, {right}, {conditions}")
        for metric, row in selected[0]["effects"].items():
            effect("partial-extension-effects", "contrasts", f"{left}_minus_{right}",
                   "/".join(map(str, heights)), metric, row, "v5_audit", 5)
    for method in ("M0", "M2", "M4"):
        full = sampling(method, "H24_L5", "H24_L2", "grf_full") if method == "M4" else v3["endpoints_five_minus_two"][method]
        effect("partial-trimmed-sampling", "five_minus_two", method, "5_minus_2", "grf_full",
               full, "v5_audit" if method == "M4" else "v3_audit", 5)
        # This table retains the v5 paired intervals, not the later v6 plot audit.
        trimmed = sampling(method, "H24_L5", "H24_L2", "grf_trim")
        effect("partial-trimmed-sampling", "five_minus_two", method, "5_minus_2", "grf_trim", trimmed, "v5_audit", 5)
    for height in (14, 24):
        for burnin in (10, 25, 50):
            row = v5["campaign_audit"]["heights"][str(height)]["windows"][str(burnin)]
            post, like = row["scalar_ess"]["posterior"], row["scalar_ess"]["likelihood"]
            values = dict(count=post["n"], burnin_percent=burnin, posterior_ESS_min=post["minimum"],
                posterior_ESS_median=post["median"], posterior_ESS_max=post["maximum"],
                posterior_ESS_at_least_200=post["at_least_200"], likelihood_ESS_median=like["median"],
                clade_difference_median=row["clade_max_difference_median"], clade_difference_max=row["clade_max_difference_maximum"])
            add("nestedbd-100-diagnostics", "diagnostics", "NestedBD-Long", height,
                "ESS_and_clade_frequency", values, "v5_audit", "ESS:1;clades:3")
    validate_tables(tables)
    return tables


def empirical_tables(bundle):
    report, diagnostics = bundle.json("empirical_report"), bundle.json("empirical_diagnostics")
    rows = []
    entries = [("D", report["distance"], {})] + [
        ("M0" if r["arm_id"] == "classical_partial" else "M2", r, r["tree_summary"])
        for r in report["reconstructions"]]
    if len(entries) != 3:
        raise ValueError("Unexpected empirical method roster")
    for method, item, tree in entries:
        resource = item["resources"]
        rows.append(dict(method=method, states=report["input_summary"]["state_count"],
            vertices=tree.get("node_count"), unlabeled_vertices=tree.get("unlabeled_node_count"),
            seconds=resource["wall_time_ns"] / 1e9,
            rss_MiB=resource["memory"]["peak_rss_bytes"] / (1024 ** 2),
            source_key="empirical_report", paper_decimal_places="seconds:3;MiB:1"))
    convergence = []
    for burnin in (10, 25, 50):
        pair = diagnostics["tree_agreement"][str(burnin)]
        scalars = [r["scalar_windows"][str(burnin)]["official_beast_statistics"] for r in diagnostics["runs"]]
        if len(scalars) != 2:
            raise ValueError("Expected two empirical chains")
        convergence.append(dict(burnin_percent=burnin,
            posterior_ESS_run1=scalars[0]["posterior"]["ESS"], posterior_ESS_run2=scalars[1]["posterior"]["ESS"],
            height_ESS_run1=scalars[0]["TreeHeight"]["ESS"], height_ESS_run2=scalars[1]["TreeHeight"]["ESS"],
            majority_clades_run1=pair["majority_clades_per_seed"][0], majority_clades_run2=pair["majority_clades_per_seed"][1],
            shared_majority_clades=pair["shared_majority_clades"], source_key="empirical_diagnostics",
            paper_decimal_places="ESS:2;counts:0"))
    return {"ktn152-feasibility": rows, "nestedbd-long-convergence": convergence}


def validate_tables(tables):
    """Coverage guards protect against the earlier omitted-M4 class of error."""
    expected = {"confirmatory-partial": 20, "confirmatory-full": 23,
        "confirmatory-full-grf": 38, "native-biopsy-count": 25,
        "confirmatory-biopsy-count": 32, "confirmatory-biopsy-gains": 10,
        "confirmatory-placement": 16, "confirmatory-resources": 4,
        "partial-extension-scores": 28, "partial-extension-effects": 14,
        "partial-trimmed-sampling": 6, "nestedbd-100-diagnostics": 6}
    if {key: len(rows) for key, rows in tables.items()} != expected:
        raise ValueError("Unexpected numerical table coverage: " + str({k: len(v) for k, v in tables.items()}))
    for name, rows in tables.items():
        for row in rows:
            if row.get("wins") is not None:
                if sum(int(row[k]) for k in ("wins", "ties", "losses")) != int(row["count"]):
                    raise ValueError(f"W/T/L denominator mismatch in {name}")


def export_tables(output, tables):
    output = Path(output)
    output.mkdir()  # never append to or overwrite a prior export
    for name, rows in tables.items():
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"tab-supp-{name}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (output / "tables.json").open("x", encoding="utf-8") as stream:
        json.dump(tables, stream, indent=2, allow_nan=False)
        stream.write("\n")
