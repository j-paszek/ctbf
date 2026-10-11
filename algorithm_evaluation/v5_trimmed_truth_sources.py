"""Read-only, checksum-bound inputs for the independent truth-target audit.

Read inventories, not directory trees. Verify every consumed artifact against
its closed source inventory. No external runtime or reconstruction is invoked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from algorithm_evaluation.paper_pipeline_contract import file_sha256
from algorithm_evaluation.v5_confirmation_common import PAPER_ARM_IDS, PAPER_ID_BY_ARM, STANDARD_CONDITION_IDS


PROJECT = Path(__file__).resolve().parents[1]
CONFIRMATION = PROJECT / "algorithm_evaluation/results/v5_confirmation/confirmation_v1"
DEFAULT_NESTED = PROJECT / "algorithm_evaluation/results/v5_nestedbd_long/benchmark_v1_report_v2"
MEDICC_ARM = "medicc2_total_cn_topology_partial"
NESTED_METHOD = "NestedBD-Long"


def object_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def safe_child(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe relative source path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Source path escapes its root.")
    return path


def checked_json(path: Path | str, digest: str):
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Source checksum changed: {path}")
    return json.loads(data)


class ClosedSource:
    def __init__(self, root: Path, checksum_name: str, protected: dict):
        self.root = root.resolve()
        self.protected = protected
        checksum = safe_child(self.root, checksum_name)
        data = checksum.read_bytes()
        protected[str(checksum)] = hashlib.sha256(data).hexdigest()
        self.inventory = {}
        for line in data.decode().splitlines():
            digest, separator, relative = line.partition("  ")
            if (not separator or len(digest) != 64 or
                    any(c not in "0123456789abcdef" for c in digest) or relative in self.inventory):
                raise ValueError(f"Malformed or duplicate checksum row in {checksum}")
            safe_child(self.root, relative)
            self.inventory[relative] = digest

    def ref(self, relative: str) -> dict:
        if relative not in self.inventory:
            raise ValueError(f"Source member is not checksum-closed: {relative}")
        path = safe_child(self.root, relative)
        digest = self.inventory[relative]
        self.protected[str(path)] = digest
        return {"path": str(path), "sha256": digest}

    def read(self, relative: str):
        ref = self.ref(relative)
        return checked_json(ref["path"], ref["sha256"])


def read_ref(ref: dict):
    return checked_json(ref["path"], ref["sha256"])


def _identity(source: dict, bank: dict, bank_root: Path, schema: str) -> None:
    if source.get("schema_version") != schema or source.get("status") != "complete":
        raise ValueError("Source run is not complete under its declared schema.")
    if (Path(source["bank_root"]).resolve() != bank_root or
            source.get("design_manifest_sha256") != bank.get("design_manifest_sha256")):
        raise ValueError("Source run uses a different frozen bank/design.")
    inventory = source.get("record_inventory", [])
    if len(inventory) != source.get("expected_record_count"):
        raise ValueError("Source run record inventory is incomplete.")


def _nested_records(root: Path, bank: ClosedSource, cases: dict, selected: set,
                    protected: dict, protected_roots: list) -> None:
    root = root.resolve()
    marker_path = root / "_COMPLETE.json"
    marker_digest = file_sha256(marker_path)
    marker = checked_json(marker_path, marker_digest)
    protected[str(marker_path)] = marker_digest
    report_dir = safe_child(root, marker["report_directory"])
    report_path = report_dir / "report.json"
    report = checked_json(report_path, marker["sha256"])
    protected[str(report_path)] = marker["sha256"]
    if (report.get("schema_version") != "ctbf-v5-nestedbd-long-benchmark-report-recovery-v1" or
            report.get("status") != "fixed_budget_benchmark_complete_not_convergence"):
        raise ValueError("Expected the completed reporting-only NestedBD benchmark.")
    contract_path = root / "analysis_contract.json"
    contract_digest = file_sha256(contract_path)
    contract = checked_json(contract_path, contract_digest)
    protected[str(contract_path)] = contract_digest
    trusted = {**contract["protected_sha256"], **report["artifact_sha256"]}
    original = Path(report["source_root"]).resolve()
    if Path(contract["source_root"]).resolve() != original:
        raise ValueError("NestedBD original-source identity differs.")
    protected_roots.append(str(original))
    declarations = {}
    for case in report["cases"]:
        case_id = case["case_id"]
        if case_id in declarations or case_id not in cases:
            raise ValueError("Duplicate or unknown NestedBD case.")
        native_case = cases[case_id]["case"]
        if any(case[key] != native_case[key] for key in ("block_index", "condition_id", "height")):
            raise ValueError("NestedBD input identity differs from the bank.")
        if native_case.get("input_path"):
            if case["input_sha256"] != bank.inventory[native_case["input_path"]]:
                raise ValueError("NestedBD did not use the same frozen input.")
            truth_path = str(safe_child(bank.root, native_case["truth_path"]))
            if trusted.get(truth_path) != bank.inventory[native_case["truth_path"]]:
                raise ValueError("NestedBD truth is not bound to the same frozen bank.")
        seeds = {(seed["replicate"], seed["seed"]) for seed in case["seeds"]}
        if len(seeds) != 2 or {pair[0] for pair in seeds} != {1, 2}:
            raise ValueError("NestedBD must retain both declared replicates.")
        declarations[case_id] = seeds
    seen = set()
    for row in report["records"]:
        if row["method"] != NESTED_METHOD:
            continue  # CTBF comparators already enter through their source inventory.
        key = (row["case_id"], row["replicate"], row["seed"])
        if key in seen or key[0] not in declarations or key[1:] not in declarations[key[0]]:
            raise ValueError("Duplicate or undeclared NestedBD replicate.")
        seen.add(key)
        if key[0] not in selected:
            continue
        ref = {"source_kind": "nested", "method": NESTED_METHOD, "problem": "partial",
               "replicate": row["replicate"], "seed": row["seed"], "status": row["status"],
               "failure": row.get("failure"), "grf": row.get("grf"),
               "report_record_digest": object_digest(row)}
        if row["status"] == "success":
            attempt = Path(row["attempt"]).resolve()
            if not any(attempt.is_relative_to(parent) for parent in (root, original)):
                raise ValueError("NestedBD summary escapes its declared source roots.")
            seal_path = attempt / "_SEALED.json"
            if str(seal_path) not in trusted:
                raise ValueError("NestedBD summary seal is not bound to the completed report.")
            seal = checked_json(seal_path, trusted[str(seal_path)])
            protected[str(seal_path)] = trusted[str(seal_path)]
            for field, filename in (("tree_ref", "tree.json"), ("evaluation_ref", "evaluation.json"),
                                    ("record_ref", "result.json")):
                path = safe_child(attempt, filename)
                if filename not in seal:
                    raise ValueError("NestedBD summary lacks a sealed tree/evaluation/result.")
                protected[str(path)] = seal[filename]
                ref[field] = {"path": str(path), "sha256": seal[filename]}
        cases[key[0]]["records"].append(ref)
    expected = {(case_id, replicate, seed) for case_id, seeds in declarations.items()
                for replicate, seed in seeds}
    if seen != expected or len(seen) != report["planned_external_runs"]:
        raise ValueError("NestedBD report omits a planned outcome.")


def collect_sources(bank_root: Path, internal_root: Path, medicc_root: Path | None,
                    nested_root: Path | None, case_ids: list[str] | None = None) -> dict:
    """Collect exact declarations without reading all stored reconstruction trees."""
    protected = {}
    bank_root, internal_root = bank_root.resolve(), internal_root.resolve()
    roots = [str(bank_root), str(internal_root)]
    bank = ClosedSource(bank_root, "bank_checksums.sha256", protected)
    bank_manifest = bank.read("bank_manifest.json")
    if (bank_manifest.get("schema_version") != "ctbf-v5-confirmatory-bank-v1" or
            bank_manifest.get("status") != "complete"):
        raise ValueError("Expected a complete confirmation bank.")
    if bank_manifest.get("scientific_role") == "technical_smoke_not_accuracy_evidence":
        raise ValueError("Technical smoke is not confirmation evidence.")
    available = {row["case_id"]: row for row in bank_manifest["cases"]}
    if len(available) != len(bank_manifest["cases"]):
        raise ValueError("Duplicate available bank case.")
    cases = {}
    for declaration in bank_manifest["condition_inventory"]:
        case_id = declaration["case_id"]
        if case_id in cases:
            raise ValueError("Duplicate declared case.")
        case = dict(declaration)
        if case["status"] == "available":
            case.update(available[case_id])
        cases[case_id] = {"case": case, "records": []}
    if len(cases) != bank_manifest["declared_condition_count"]:
        raise ValueError("Incomplete condition inventory.")
    selected = set(cases if case_ids is None else case_ids)
    if not selected or selected - cases.keys():
        raise ValueError("Empty or unknown case selection.")
    internal = ClosedSource(internal_root, "run_checksums.sha256", protected)
    run = internal.read("run_manifest.json")
    _identity(run, bank_manifest, bank_root, "ctbf-v5-confirmatory-run-v1")
    seen = set()
    for row in run["record_inventory"]:
        key = (row["case_id"], row["arm_id"])
        if key in seen or key[0] not in cases or key[1] not in PAPER_ARM_IDS:
            raise ValueError("Duplicate or undeclared internal result.")
        seen.add(key)
        if key[0] in selected:
            ref = internal.ref(row["path"])
            if row["sha256"] != ref["sha256"]:
                raise ValueError("Internal manifest and closure digests disagree.")
            cases[key[0]]["records"].append({
                "source_kind": "ctbf", "method": PAPER_ID_BY_ARM[key[1]],
                "problem": "partial" if PAPER_ID_BY_ARM[key[1]] in {"M0", "M2"} else "closed_state",
                "arm_id": key[1], "status": row["status"], "replicate": None, "seed": None,
                "record_ref": ref})
    if seen != {(case_id, arm) for case_id in cases for arm in PAPER_ARM_IDS}:
        raise ValueError("Internal run omits declared case/arm records.")
    if medicc_root is not None:
        roots.append(str(medicc_root.resolve()))
        medicc = ClosedSource(medicc_root, "run_checksums.sha256", protected)
        med_run = medicc.read("run_manifest.json")
        _identity(med_run, bank_manifest, bank_root, "ctbf-v5-confirmatory-medicc2-run-v1")
        seen = set()
        for row in med_run["record_inventory"]:
            case_id = row["case_id"]
            if (case_id in seen or case_id not in cases or
                    cases[case_id]["case"]["condition_id"] not in STANDARD_CONDITION_IDS):
                raise ValueError("Duplicate or undeclared MEDICC2 case.")
            seen.add(case_id)
            if case_id in selected:
                ref = medicc.ref(row["path"])
                if ref["sha256"] != row["sha256"]:
                    raise ValueError("MEDICC2 manifest and closure digests disagree.")
                cases[case_id]["records"].append({
                    "source_kind": "medicc2", "method": "MEDICC2", "problem": "partial",
                    "arm_id": MEDICC_ARM, "status": row["status"], "replicate": None,
                    "seed": None, "record_ref": ref})
        expected = {key for key, row in cases.items() if row["case"]["condition_id"] in STANDARD_CONDITION_IDS}
        if seen != expected:
            raise ValueError("MEDICC2 run omits a declared standard condition.")
    if nested_root is not None:
        roots.append(str(nested_root.resolve()))
        _nested_records(nested_root, bank, cases, selected, protected, roots)
    tasks = []
    for case_id, task in cases.items():
        if case_id not in selected:
            continue
        case = task["case"]
        if case["status"] == "available":
            task["input_ref"] = bank.ref(case["input_path"])
            task["truth_ref"] = bank.ref(case["truth_path"])
        tasks.append(task)
    tasks.sort(key=lambda task: (task["case"]["block_index"], task["case"]["condition_ordinal"]))
    return {"bank_root": str(bank_root), "design_manifest_sha256": bank_manifest["design_manifest_sha256"],
            "protected_roots": sorted(set(roots)), "protected_sha256": protected,
            "tasks": tasks, "selected_subset": case_ids is not None,
            "expected_record_count": sum(len(task["records"]) for task in tasks)}
