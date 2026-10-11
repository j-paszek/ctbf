"""Closed inputs for the additive M4 / 100-history NestedBD comparison.

Old source-pinned runners and completed scores are read-only. Selection is by
declared history/condition, never by input dimensions, accuracy or diagnostics.
"""

from __future__ import annotations

import csv
import copy
import io
import math
from pathlib import Path

from algorithm_evaluation import v5_confirmation_common as confirmation
from algorithm_evaluation import v5_confirmation_run as frozen
from algorithm_evaluation import v5_nestedbd_long_benchmark as benchmark
from algorithm_evaluation import v5_nestedbd_long_benchmark_report as recovery
from algorithm_evaluation import v5_partial_comparator_extension_nested as nested10m
from algorithm_evaluation import v5_partial_comparator_extension_2m_nested as nested2m
from algorithm_evaluation.paper_pipeline_contract import write_json_atomic
from algorithm_evaluation.v5_algorithm_development_common import (
    ARM_SPEC_BY_ID, PARTIAL_ADAPTIVE_MEDIAN_Z_ID,
)
from algorithm_evaluation.v5_trimmed_truth_sources import (
    ClosedSource, checked_json, object_digest, safe_child,
)
from trimmed_truth_evaluation import truth_target_descriptor


pilot, calibration, topology = benchmark.pilot, benchmark.calibration, benchmark.topology
SCHEMA = "ctbf-v5-partial-comparator-extension-10m-v1"
DEFAULT_BANK = benchmark.DEFAULT_BANK
DEFAULT_TRIM = DEFAULT_BANK.parent.parent / "trimmed_truth_v1"
DEFAULT_NESTED = recovery.DEFAULT_OUTPUT
DEFAULT_BUDGET = nested10m.budget.DEFAULT_OUTPUT
OLD_OUTPUT = benchmark.DEFAULT_OUTPUT.with_name("partial_comparators_100_v1")
DEFAULT_OUTPUT = benchmark.DEFAULT_OUTPUT.with_name("partial_comparators_100_10m_v1")
M4_ARM = PARTIAL_ADAPTIVE_MEDIAN_Z_ID
ALL_BLOCKS = tuple(range(100))
REUSED_BLOCKS = nested10m.budget.BLOCKS
NEW_BLOCKS = tuple(range(2, 100))
CONDITIONS = (
    "H14_rule_y", "H24_rule_y", "H34_rule_y", "H24_L2",
    "H24_L4", "H24_L5", "H24_late", "H24_random",
)
NESTED_CONDITIONS = benchmark.CONDITIONS
LIMITS = {"reconstruction_seconds": 300, "evaluation_seconds": 300,
          "trim_seconds": 300, "rss_bytes": 4 * 1024**3}
SUMMARY_KEYS = nested10m.SUMMARY_KEYS
_BASE_RULES = {
    "selection": "all_100_original_histories_in_original_order_no_replacements",
    "m4_arm_id": M4_ARM, "m4_conditions": list(CONDITIONS),
    "nested_conditions": list(NESTED_CONDITIONS),
    "nested_reused_blocks": list(REUSED_BLOCKS), "nested_new_blocks": list(NEW_BLOCKS),
    "nested_seed_namespace": benchmark.SEED_NAMESPACE,
    "nested_inference": nested10m.SETTINGS, "nested_clock": benchmark.CLOCK,
    # Reuse the exact summary semantics, not the pilot's obsolete case/seed
    # inventory or its one-history descriptive metadata.
    "nested_summary": nested10m.SUMMARY, "m4_limits": LIMITS,
    "nested_reuse": "eight_sealed_10M_runs_recompute_all_summaries_and_scores",
    "nested_2m_scores_included": False,
    "truth_target": truth_target_descriptor(),
    "distance": "reuse_exact_aligned_frozen_submatrix_no_new_cnp2cnp",
    "seed_aggregation": "mean_of_both_successful_seed_scores_per_input",
    "terminal_failures": "retain_without_retry_replacement_or_imputation",
    "interruptions": "preserve_attempt_restart_from_zero_never_restore_native_state",
    "independent_unit": "simulated_history", "bootstrap_draws": 100000,
    "original_primary_decision_recomputed": False, "p_values": False,
    "convergence_claim": False, "whole_pipeline_resource_comparison": False,
}
RULES = copy.deepcopy(_BASE_RULES)
_PROFILE = "10m"


def configure_profile(profile: str) -> None:
    """Select one immutable budget profile before preparing an output root."""
    global nested10m, SCHEMA, DEFAULT_BUDGET, DEFAULT_OUTPUT
    global REUSED_BLOCKS, NEW_BLOCKS, SUMMARY_KEYS, RULES, _PROFILE
    if profile not in {"2m", "10m"}:
        raise ValueError("nested budget must be 2m or 10m")
    _PROFILE = profile
    if profile == "2m":
        nested10m = nested2m
        SCHEMA = "ctbf-v5-partial-comparator-extension-2m-v1"
        DEFAULT_OUTPUT = benchmark.DEFAULT_OUTPUT.with_name("partial_comparators_100_2m_v1")
        REUSED_BLOCKS = ()
        NEW_BLOCKS = ALL_BLOCKS
    else:
        nested10m = __import__(
            "algorithm_evaluation.v5_partial_comparator_extension_nested",
            fromlist=["*"])
        SCHEMA = "ctbf-v5-partial-comparator-extension-10m-v1"
        DEFAULT_OUTPUT = benchmark.DEFAULT_OUTPUT.with_name("partial_comparators_100_10m_v1")
        REUSED_BLOCKS = nested10m.budget.BLOCKS
        NEW_BLOCKS = tuple(range(2, 100))
    SUMMARY_KEYS = nested10m.SUMMARY_KEYS
    RULES = copy.deepcopy(_BASE_RULES)
    RULES.update(
        nested_inference=nested10m.SETTINGS,
        nested_summary=nested10m.SUMMARY,
        nested_reuse=("no_reuse_all_2m_runs" if profile == "2m"
                       else "eight_sealed_10M_runs_recompute_all_summaries_and_scores"),
        nested_2m_scores_included=(profile == "2m"),
    )


def case_ids(blocks, conditions) -> list[str]:
    return [confirmation.confirmation_case_id(b, c) for b in blocks for c in conditions]


def seeds_for(case_id: str) -> list[dict]:
    return [{"replicate": r, "seed": benchmark.seed_for(case_id, r)} for r in (1, 2)]


def bind(path: Path, protected: dict, expected: str | None = None) -> str:
    digest = pilot.sha256(path)
    if expected is not None and digest != expected:
        raise ValueError("Changed source: " + str(path))
    if str(path) in protected and protected[str(path)] != digest:
        raise ValueError("Conflicting source identity: " + str(path))
    protected[str(path)] = digest
    return digest


def validate_bank(bank: dict, source: ClosedSource) -> None:
    approval = confirmation.read_owner_approval_lock()
    bind(Path(approval["path"]), source.protected, approval["sha256"])
    if (bank.get("schema_version") != confirmation.BANK_SCHEMA_VERSION
            or bank.get("status") != "complete" or bank.get("block_count") != 100
            or bank.get("scientific_role") != "prospectively_locked_independent_confirmation"
            or approval.get("status") != "approved"):
        raise ValueError("The completed owner-approved 100-history bank is required.")
    for field, key in (("design_manifest_sha256", "APPROVED_MANIFEST_SHA256"),
                       ("protocol_sha256", "APPROVED_PROTOCOL_SHA256"),
                       ("source_lock_sha256", "APPROVED_SOURCE_LOCK_SHA256")):
        if bank.get(field) != approval[key]:
            raise ValueError("Frozen design changed: " + field)
    ref = source.ref("owner_approval.snapshot.py")
    if ref["sha256"] != approval["sha256"] or bank.get("owner_approval_sha256") != ref["sha256"]:
        raise ValueError("Owner approval snapshot changed.")
    config = source.read("simulator_config.json")
    bind(calibration.REFERENCE_CONFIG, source.protected)
    if (config != pilot.read_json(calibration.REFERENCE_CONFIG)
            or calibration.calibration_moments(config)["nominal_mean_clock"] != benchmark.CLOCK["mean_clock_rate"]):
        raise ValueError("Standard simulator / fixed nominal clock changed.")


def select_cases(bank: dict, *, check: bool = False) -> tuple[list[dict], list[dict]]:
    """Declare all inputs first. The technical check is not a production subset."""
    declarations = frozen._declared_cases(bank)
    index = {case["case_id"]: case for case in declarations}
    if len(index) != len(declarations) or set(index) != set(case_ids(ALL_BLOCKS, CONDITIONS)):
        raise ValueError("The frozen eight-condition inventory differs.")
    for case in declarations:
        if case["case_id"] != confirmation.confirmation_case_id(case["block_index"], case["condition_id"]):
            raise ValueError("Case coordinates differ.")
    m4_ids = case_ids(ALL_BLOCKS, CONDITIONS)
    nested_ids = case_ids(ALL_BLOCKS, NESTED_CONDITIONS)
    if check:
        m4_ids = nested_ids = case_ids(NEW_BLOCKS[:1], NESTED_CONDITIONS[:1])
    seeds = [s["seed"] for cid in case_ids(ALL_BLOCKS, NESTED_CONDITIONS) for s in seeds_for(cid)]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Declared MCMC seeds collide.")
    return [index[cid] for cid in m4_ids], [index[cid] for cid in nested_ids]


def nested_case(source: ClosedSource, case: dict) -> dict:
    result = {key: case[key] for key in ("case_id", "block_index", "condition_id", "height", "generations")}
    expected = [9, 12, 14] if case["condition_id"] == "H14_rule_y" else [15, 20, 24]
    if case["generations"] != expected or case["height"] != expected[-1]:
        raise ValueError("NestedBD biopsy schedule differs.")
    result.update(seeds=seeds_for(case["case_id"]), source_case=None,
                  eligibility="bank_unavailable", input_summary=None, records=[],
                  bank_failure=case.get("failure"))
    if case["status"] == "available":
        ref = source.ref(case["input_path"])
        payload = checked_json(ref["path"], ref["sha256"])
        if (payload.get("case_id") != case["case_id"] or payload.get("condition_id") != case["condition_id"]
                or [level["generation"] for level in payload["levels"]] != expected):
            raise ValueError("NestedBD observable input identity differs.")
        records = pilot.input_records(payload)
        summary = pilot.compatibility(records)
        result.update(source_case=case, records=records, input_summary=summary,
                      input_path=ref["path"], input_sha256=ref["sha256"], eligibility=summary["status"])
    elif case["status"] != "unavailable":
        raise ValueError("Unknown bank availability status.")
    return result


def load_past(trim: Path, nested: Path, bank: ClosedSource, manifest: dict, bio: Path):
    """Consume compact sealed summaries, not old traces or evaluator replays."""
    protected = bank.protected
    marker_path = trim / "_COMPLETE.json"
    bind(marker_path, protected)
    marker = pilot.read_json(marker_path)
    cpath = trim / "analysis_contract.json"
    bind(cpath, protected, marker["contract_sha256"])
    contract = pilot.read_json(cpath)
    if (marker.get("schema_version") != "ctbf-v5-trimmed-truth-run-v1"
            or Path(contract["sources"]["bank_root"]).resolve() != bank.root
            or contract["sources"]["design_manifest_sha256"] != manifest["design_manifest_sha256"]):
        raise ValueError("Trimmed audit does not describe this frozen bank.")
    trusted = contract["sources"]["protected_sha256"]
    for path, digest in contract["code_sha256"].items():
        bind(Path(path), protected, digest)
    for name in (confirmation.BANK_CHECKSUM_NAME, confirmation.BANK_MANIFEST_NAME):
        path = str(bank.root / name)
        if trusted.get(path) != protected[path]:
            raise ValueError("Trimmed audit / bank identity differs.")
    scores_path = trim / "scores.csv"
    bind(scores_path, protected, marker["artifacts"]["scores.csv"])
    rows = []
    for raw in csv.DictReader(io.StringIO(scores_path.read_text(encoding="utf-8"))):
        if raw["method"] not in {"M0", "M2", "MEDICC2"}:
            continue
        row = dict(raw)
        for key in ("block_index", "height", "replicate", "seed"):
            row[key] = int(row[key]) if row[key] else None
        for key in ("grf_full", "grf_trim", "ad_f1"):
            row[key] = float(row[key]) if row[key] else None
        if row["problem"] != "partial":
            raise ValueError("Mixed output problems in the partial comparison.")
        if row["status"] == "success" and not all(
                row[key] is not None and math.isfinite(row[key]) for key in ("grf_full", "grf_trim")):
            raise ValueError("Successful old scores must be finite.")
        row["provenance"] = "reused_completed_trimmed_truth_audit"
        rows.append(row)
    validate_roster(rows, include_new=False)

    nmarker_path = nested / "_COMPLETE.json"
    bind(nmarker_path, protected, trusted[str(nmarker_path)])
    nmarker = pilot.read_json(nmarker_path)
    report_path = safe_child(nested, nmarker["report_directory"]) / "report.json"
    bind(report_path, protected, nmarker["sha256"])
    previous = pilot.read_json(report_path)
    if previous.get("schema_version") != "ctbf-v5-nestedbd-long-benchmark-report-recovery-v1":
        raise ValueError("Expected the completed reporting-only recovery.")
    old_root = Path(previous["source_root"]).resolve()
    recovery_path = nested / "analysis_contract.json"
    bind(recovery_path, protected, trusted[str(recovery_path)])
    recovered = pilot.read_json(recovery_path)
    old_path = old_root / "analysis_contract.json"
    bind(old_path, protected, recovered["protected_sha256"][str(old_path)])
    original = pilot.read_json(old_path)
    if (original.get("schema_version") != benchmark.SCHEMA or original.get("rules") != benchmark.RULES
            or Path(original["bank_root"]).resolve() != bank.root or original["bio_root"] != str(bio)
            or original["runtime"] != recovery.runtime_identity()):
        raise ValueError("Original 2M model, runtime or input bank differs.")
    backend = pilot.load_backend(bio)
    for path, digest in pilot.backend_snapshot(backend).items():
        if original["protected_sha256"].get(path) != digest:
            raise ValueError("Pinned external implementation changed: " + path)
        bind(Path(path), protected, digest)
    for path, digest in original["protected_sha256"].items():
        if Path(path).suffix == ".py":
            bind(Path(path), protected, digest)
    return rows, previous, backend, [trim, nested, old_root]


def record_key(row: dict) -> tuple:
    return row["case_id"], row["method"], row.get("replicate"), row.get("seed")


def validate_roster(rows: list[dict], *, include_new: bool) -> None:
    methods = ("M0", "M2", "M4") if include_new else ("M0", "M2")
    expected = {(cid, m, None, None) for cid in case_ids(ALL_BLOCKS, CONDITIONS) for m in methods}
    expected.update((cid, "MEDICC2", None, None) for cid in case_ids(ALL_BLOCKS, confirmation.STANDARD_CONDITION_IDS))
    blocks = ALL_BLOCKS if include_new else ()
    expected.update((cid, "NestedBD-Long", seed["replicate"], seed["seed"])
                    for cid in case_ids(blocks, NESTED_CONDITIONS) for seed in seeds_for(cid))
    actual = [record_key(row) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing, duplicate or undeclared partial-comparator record.")
    for row in rows:
        if row["case_id"] != confirmation.confirmation_case_id(row["block_index"], row["condition_id"]):
            raise ValueError("Score coordinates differ.")
        if row["height"] != int(row["condition_id"].split("_", 1)[0][1:]) or row["problem"] != "partial":
            raise ValueError("Score height or output problem differs.")
        if row["method"] == "NestedBD-Long" and row.get("inference_budget_iterations") != nested10m.ITERATIONS:
            raise ValueError("All NestedBD score records must use the uniform 10M budget.")
        if row["status"] == "success" and not all(
                row.get(key) is not None and math.isfinite(row[key]) and 0 <= row[key] <= 1
                for key in ("grf_full", "grf_trim")):
            raise ValueError("Successful scores must be finite similarities.")


def bundle_write(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError("Completed bundle is immutable: " + str(path))
    write_json_atomic(path, {"schema_version": SCHEMA, "sha256": object_digest(payload), "payload": payload})


def bundle_read(path: Path) -> dict:
    record = pilot.read_json(path)
    if record.get("schema_version") != SCHEMA or object_digest(record["payload"]) != record.get("sha256"):
        raise ValueError("Changed extension bundle: " + str(path))
    return record["payload"]


def output_guard(output: Path, bank: Path, bio: Path, roots) -> None:
    pilot.output_guard(output, bank, bio)
    for root in roots:
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output overlaps completed evidence: " + str(root))


def prepare_sources(bank_root: Path, trim: Path, nested: Path, bio: Path, budget_root: Path, *, check=False):
    protected = {}
    source = ClosedSource(bank_root, confirmation.BANK_CHECKSUM_NAME, protected)
    manifest = source.read(confirmation.BANK_MANIFEST_NAME)
    validate_bank(manifest, source)
    m4_cases, new_cases = select_cases(manifest, check=check)
    # Register required immutable members once. Matrices/truths are hashed, not
    # deserialized or used to select MCMC inputs.
    for case in m4_cases:
        if case["status"] == "available":
            for key in ("input_path", "metadata_path", "truth_path", "distance_path"):
                source.ref(case[key])
    nested_cases = [nested_case(source, case) for case in new_cases]
    old_rows, previous, backend, old_roots = load_past(trim, nested, source, manifest, bio)
    if _PROFILE == "10m":
        declarations = {c["case_id"]: c for c in frozen._declared_cases(manifest)}
        reuse_cases = [nested_case(source, declarations[cid])
                       for cid in nested10m.budget.selected_ids()]
        reused = nested10m.load_reused(
            budget_root, nested, bio, reuse_cases, backend, protected)
        previous = {"reused_outcomes": reused}
        old_roots += [budget_root, OLD_OUTPUT]
    else:
        # The 2M series is a fresh, uniform campaign. It deliberately does
        # not mix the earlier ten-history report into the 100-history output.
        previous = {"reused_outcomes": []}
        old_roots += [OLD_OUTPUT]
    spec = ARM_SPEC_BY_ID[M4_ARM]
    if spec.problem != "partial" or spec.algorithm_name != "neighbor_joining_classical":
        raise ValueError("Existing M4 algorithm identity differs.")
    protected.update(topology.source_snapshot())
    pilot.verify_hashes(protected)
    return m4_cases, nested_cases, old_rows, previous, backend, protected, old_roots
