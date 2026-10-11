"""Run the frozen exploratory KTN152 empirical-feasibility reconstruction.

This runner deliberately has no truth-tree or accuracy-evaluation input.  It
constructs a chromosome-aware whole-genome cnp2cnp dissimilarity, then runs the
owner-frozen partial baseline and principal biopsy-guided partial method.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from algorithm_evaluation.paper_pipeline_contract import (
    json_safe,
    read_json,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import (
    PeakProcessTreeRss,
    measured_stage,
    serialize_tree,
    validate_reconstruction_input,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    ARM_SPEC_BY_ID,
    observed_labels,
    reconstruct_development_arm,
    tree_summary,
)
from ctbs import (
    Cnp2CnpDirectedFileDistanceProvider,
    CtbsRuntimeConfig,
    DistanceMatrix,
    load_ctbs_runtime_config,
)
from distance_semantics import (
    DISTANCE_PROVENANCE_SCHEMA_VERSION,
    DirectedDistanceBundle,
    stable_distance_label_key,
    validate_distance_matrix,
)
from simulator import Genotype


INPUT_SUCCESS_SCHEMA_VERSION = "ctbfbio-ktn152-feasibility-success-v1"
RUN_SCHEMA_VERSION = "ctbf-v5-empirical-feasibility-run-v1"
CHROMOSOME_RESULT_SCHEMA_VERSION = (
    "ctbf-v5-empirical-chromosome-directed-distance-v1"
)
DISTANCE_SCHEMA_VERSION = "ctbf-v5-empirical-distance-v1"
RECONSTRUCTION_SCHEMA_VERSION = "ctbf-v5-empirical-reconstruction-v1"
REPORT_SCHEMA_VERSION = "ctbf-v5-empirical-feasibility-report-v1"
CHROMOSOME_DISTANCE_SEMANTICS = (
    "ctbf-cnp2cnp-any-chromosome-additive-min-bidirectional-v1"
)

PARTIAL_BASELINE_ID = "classical_partial"
PARTIAL_PRINCIPAL_ID = (
    "biopsy_guided_top_anticentral_binary_adaptive_median_prior_nn_"
    "bottom_deferred_tie"
)
ARM_IDS = (PARTIAL_BASELINE_ID, PARTIAL_PRINCIPAL_ID)
DEFAULT_RECONSTRUCTION_SEED = 20260908
DEFAULT_WORKERS = 6
DEFAULT_CNP2CNP_TIMEOUT_SECONDS = 1800.0
DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS = 300
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1024**2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _verify_input_root(input_root: Path) -> dict[str, Any]:
    success_path = input_root / "_SUCCESS.json"
    manifest_path = input_root / "data_manifest.tsv"
    if not success_path.is_file() or not manifest_path.is_file():
        raise ValueError("Empirical input lacks its success marker or data manifest.")
    success = read_json(success_path)
    if success.get("schema_version") != INPUT_SUCCESS_SCHEMA_VERSION:
        raise ValueError("Unknown empirical-input success schema.")
    if success.get("data_manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("Empirical-input manifest digest differs from _SUCCESS.json.")

    rows = _read_tsv(manifest_path)
    if not rows or set(rows[0]) != {"path", "bytes", "sha256", "media_type"}:
        raise ValueError("Empirical-input data manifest has an invalid schema.")
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Empirical-input manifest contains an unsafe path.")
        path = input_root / relative
        if not path.is_file():
            raise ValueError(f"Empirical-input artifact is missing: {relative}.")
        if int(row["bytes"]) != path.stat().st_size:
            raise ValueError(f"Empirical-input size changed: {relative}.")
        if row["sha256"] != _sha256_file(path):
            raise ValueError(f"Empirical-input digest changed: {relative}.")

    contract = read_json(input_root / "analysis_contract.json")
    if contract.get("scientific_role") != "exploratory_empirical_feasibility":
        raise ValueError("Input is not frozen for exploratory empirical feasibility.")
    if contract.get("absolute_cnp_gate_status") != "rejected_unresolved_scale":
        raise ValueError("Input does not preserve the rejected absolute-CNP gate.")
    if contract.get("truth_tree_available") is not False:
        raise ValueError("Empirical feasibility input must not declare a truth tree.")
    return {
        "success": success,
        "contract": contract,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def chromosome_slices(
    bin_rows: Sequence[Mapping[str, str]],
) -> tuple[tuple[str, int, int], ...]:
    """Return ordered half-open chromosome blocks and reject chromosome re-entry."""

    if not bin_rows:
        raise ValueError("At least one genomic bin is required.")
    blocks: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    start = 0
    chromosome = str(bin_rows[0].get("chromosome", "")).strip()
    if not chromosome:
        raise ValueError("Every genomic bin requires a chromosome.")
    for index, row in enumerate(bin_rows):
        current = str(row.get("chromosome", "")).strip()
        if not current:
            raise ValueError("Every genomic bin requires a chromosome.")
        if current != chromosome:
            blocks.append((chromosome, start, index))
            seen.add(chromosome)
            if current in seen:
                raise ValueError(f"Chromosome {current!r} re-enters the bin order.")
            chromosome = current
            start = index
    blocks.append((chromosome, start, len(bin_rows)))
    if len({row.get("bin_id") for row in bin_rows}) != len(bin_rows):
        raise ValueError("Empirical bin identifiers must be unique.")
    return tuple(blocks)


def _subset_payload(payload: Mapping[str, Any], *, preflight: bool) -> dict[str, Any]:
    selected = json.loads(json.dumps(payload))
    if preflight:
        for level in selected["levels"]:
            level["states"] = sorted(
                level["states"],
                key=lambda state: stable_distance_label_key(state["state_label"]),
            )[:2]
        selected["case_id"] = f"{payload.get('case_id', 'empirical')}_technical_preflight"
        selected["condition_id"] = (
            f"{payload.get('condition_id', 'empirical')}_technical_preflight"
        )
    validate_reconstruction_input(selected)
    return selected


def _integerize_state_labels(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, Any], dict[str, Any]]:
    """Use numeric algorithm labels while preserving the external CNP labels."""

    external_labels = observed_labels(payload)
    internal_by_external = {
        external: index for index, external in enumerate(external_labels)
    }
    external_by_internal = {
        internal: external for external, internal in internal_by_external.items()
    }
    internal_payload = json.loads(json.dumps(payload))
    for level in internal_payload["levels"]:
        for state in level["states"]:
            state["state_label"] = internal_by_external[state["state_label"]]
    validate_reconstruction_input(internal_payload)
    mapping = {
        "schema_version": "ctbf-v5-empirical-state-label-map-v1",
        "purpose": "numeric_algorithm_labels_with_external_labels_restored_in_output",
        "entries": [
            {
                "internal_state_label": internal,
                "external_state_label": json_safe(external),
            }
            for internal, external in sorted(external_by_internal.items())
        ],
    }
    return internal_payload, external_by_internal, mapping


def _profiles_by_label(payload: Mapping[str, Any]) -> dict[Any, tuple[int, ...]]:
    profiles: dict[Any, tuple[int, ...]] = {}
    for level in payload["levels"]:
        for state in level["states"]:
            label = state["state_label"]
            profile = tuple(int(value) for value in state["cnp"])
            if label in profiles and profiles[label] != profile:
                raise ValueError("One empirical state label maps to multiple CNPs.")
            profiles[label] = profile
    return profiles


def _distance_task(
    chromosome: str,
    start: int,
    end: int,
    labels: Sequence[Any],
    profiles: Mapping[Any, Sequence[int]],
    runtime_config: CtbsRuntimeConfig,
) -> dict[str, Any]:
    cells = [
        Genotype(
            profiles[label][start:end],
            label,
            generation=None,
            cell_id=label,
        )
        for label in labels
    ]
    started = time.perf_counter_ns()
    with PeakProcessTreeRss() as memory:
        bundle = Cnp2CnpDirectedFileDistanceProvider(runtime_config).compute(cells)
    if not isinstance(bundle, DirectedDistanceBundle):
        raise TypeError("Chromosome distance provider did not retain directed costs.")
    return {
        "schema_version": CHROMOSOME_RESULT_SCHEMA_VERSION,
        "chromosome": chromosome,
        "start_index": start,
        "end_index_exclusive": end,
        "bin_count": end - start,
        "ids": json_safe(list(bundle.ids)),
        "directed_matrix": json_safe(bundle.directed_matrix),
        "provider_provenance": json_safe(bundle.provenance),
        "resources": {
            "wall_time_ns": time.perf_counter_ns() - started,
            "memory": memory.report(),
        },
    }


def _validate_chromosome_result(
    value: Mapping[str, Any],
    *,
    chromosome: str,
    start: int,
    end: int,
    labels: Sequence[Any],
) -> None:
    exact = {
        "schema_version": CHROMOSOME_RESULT_SCHEMA_VERSION,
        "chromosome": chromosome,
        "start_index": start,
        "end_index_exclusive": end,
        "bin_count": end - start,
        "ids": list(labels),
    }
    for field, expected in exact.items():
        if value.get(field) != expected:
            raise ValueError(f"Stored {chromosome} result changed field {field}.")
    directed = np.asarray(value.get("directed_matrix"), dtype=float)
    if directed.shape != (len(labels), len(labels)):
        raise ValueError(f"Stored {chromosome} directed matrix has the wrong shape.")
    if not np.all(np.isfinite(directed)) or np.any(directed < 0):
        raise ValueError(f"Stored {chromosome} directed matrix is invalid.")
    if not np.all(np.diag(directed) == 0):
        raise ValueError(f"Stored {chromosome} directed diagonal is nonzero.")


def aggregate_chromosome_distances(
    results: Sequence[Mapping[str, Any]],
    labels: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Sum directed chromosome costs, then take one whole-genome minimum."""

    directed = np.zeros((len(labels), len(labels)), dtype=float)
    for value in results:
        ids = list(value["ids"])
        matrix = np.asarray(value["directed_matrix"], dtype=float)
        index = {label: position for position, label in enumerate(ids)}
        if len(index) != len(ids) or set(ids) != set(labels):
            raise ValueError("Chromosome result labels differ from empirical states.")
        order = [index[label] for label in labels]
        directed += matrix[np.ix_(order, order)]
    minimum = np.minimum(directed, directed.T)
    _, minimum = validate_distance_matrix(labels, minimum)
    return directed, minimum


def _git_identity(root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "worktree_dirty": dirty}


def _run_contract(
    *,
    input_root: Path,
    input_audit: Mapping[str, Any],
    payload: Mapping[str, Any],
    blocks: Sequence[tuple[str, int, int]],
    workers: int,
    runtime_config: CtbsRuntimeConfig,
    preflight: bool,
    state_label_map: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "in_progress",
        "run_mode": "technical_preflight" if preflight else "empirical_feasibility",
        "paper_accuracy_evidence_allowed": False,
        "truth_tree_supplied": False,
        "input_root": str(input_root),
        "input_manifest_sha256": input_audit["manifest_sha256"],
        "case_id": payload.get("case_id"),
        "condition_id": payload.get("condition_id"),
        "state_count": len(observed_labels(payload)),
        "occurrence_count": sum(len(level["states"]) for level in payload["levels"]),
        "bin_count": len(payload["levels"][0]["states"][0]["cnp"]),
        "chromosome_blocks": [
            {
                "chromosome": chromosome,
                "start_index": start,
                "end_index_exclusive": end,
                "bin_count": end - start,
            }
            for chromosome, start, end in blocks
        ],
        "distance_semantics": CHROMOSOME_DISTANCE_SEMANTICS,
        "distance_formula": (
            "min(sum_c d_any(x_c,y_c), sum_c d_any(y_c,x_c))"
        ),
        "chromosome_events_cross_boundaries": False,
        "state_label_encoding": (
            "deterministic_zero_based_integers_for_algorithm_execution_"
            "with_external_labels_restored_in_serialized_trees"
        ),
        "state_label_map_sha256": hashlib.sha256(
            json.dumps(
                state_label_map,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "workers": workers,
        "cnp2cnp": {
            "executable": runtime_config.cnp2cnp_file,
            "source_sha256": _sha256_file(Path(runtime_config.cnp2cnp_file)),
            "per_process_timeout_seconds": runtime_config.cnp2cnp_timeout_seconds,
            "capture_limit_bytes": runtime_config.cnp2cnp_capture_limit_bytes,
        },
        "ctbf": _git_identity(root),
        "arm_ids": list(ARM_IDS),
        "reconstruction_seed": DEFAULT_RECONSTRUCTION_SEED,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _load_or_create_contract(
    output_root: Path,
    expected: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    path = output_root / "run_contract.json"
    if output_root.exists():
        if not resume or not path.is_file():
            raise ValueError(
                "Output root already exists. Use --resume only for this exact run."
            )
        observed = read_json(path)
        for field, value in expected.items():
            if field in {"status", "created_at_utc"}:
                continue
            if observed.get(field) != value:
                raise ValueError(f"Cannot resume because {field} changed.")
        if observed.get("status") == "complete":
            raise ValueError("The empirical feasibility run is already complete.")
        return observed
    if resume:
        raise ValueError("--resume requires an existing empirical output root.")
    output_root.mkdir(parents=True)
    (output_root / "distances" / "by_chromosome").mkdir(parents=True)
    (output_root / "reconstructions").mkdir(parents=True)
    write_json_atomic(path, dict(expected))
    return dict(expected)


def _chromosome_filename(chromosome: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in chromosome)
    return f"{safe}.json"


def _compute_or_resume_chromosomes(
    *,
    output_root: Path,
    blocks: Sequence[tuple[str, int, int]],
    labels: Sequence[Any],
    profiles: Mapping[Any, Sequence[int]],
    runtime_config: CtbsRuntimeConfig,
    workers: int,
) -> list[dict[str, Any]]:
    directory = output_root / "distances" / "by_chromosome"
    completed: dict[str, dict[str, Any]] = {}
    pending = []
    for chromosome, start, end in blocks:
        path = directory / _chromosome_filename(chromosome)
        if path.is_file():
            value = read_json(path)
            _validate_chromosome_result(
                value,
                chromosome=chromosome,
                start=start,
                end=end,
                labels=labels,
            )
            completed[chromosome] = value
        else:
            pending.append((chromosome, start, end))

    if completed:
        print(f"resumed: {len(completed)}/{len(blocks)} chromosome distances available")
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {
                executor.submit(
                    _distance_task,
                    chromosome,
                    start,
                    end,
                    labels,
                    profiles,
                    runtime_config,
                ): (chromosome, start, end)
                for chromosome, start, end in pending
            }
            try:
                for future in as_completed(futures):
                    chromosome, start, end = futures[future]
                    value = future.result()
                    _validate_chromosome_result(
                        value,
                        chromosome=chromosome,
                        start=start,
                        end=end,
                        labels=labels,
                    )
                    write_json_atomic(
                        directory / _chromosome_filename(chromosome),
                        value,
                    )
                    completed[chromosome] = value
                    print(
                        f"distance: {len(completed)}/{len(blocks)} chromosomes "
                        f"complete ({chromosome}, {end - start} bins)",
                        flush=True,
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    return [completed[chromosome] for chromosome, _start, _end in blocks]


def _temporal_tree_diagnostics(tree: nx.DiGraph) -> dict[str, Any]:
    comparable = 0
    reversed_edges = 0
    forward_edges = 0
    same_time_edges = 0
    for source, target in tree.edges:
        source_generation = tree.nodes[source].get("generation")
        target_generation = tree.nodes[target].get("generation")
        if source_generation is None or target_generation is None:
            continue
        comparable += 1
        if source_generation > target_generation:
            reversed_edges += 1
        elif source_generation < target_generation:
            forward_edges += 1
        else:
            same_time_edges += 1
    return {
        "generation_comparable_edge_count": comparable,
        "generation_forward_edge_count": forward_edges,
        "generation_same_time_edge_count": same_time_edges,
        "generation_reversed_edge_count": reversed_edges,
        "descriptive_only_not_accuracy": True,
    }


def _run_reconstruction(
    *,
    output_root: Path,
    arm_id: str,
    payload: Mapping[str, Any],
    distance: DistanceMatrix,
    timeout_seconds: int,
    rss_limit_bytes: int,
    external_label_by_internal: Mapping[int, Any],
) -> dict[str, Any]:
    spec = ARM_SPEC_BY_ID[arm_id]
    value, resources, error = measured_stage(
        lambda: reconstruct_development_arm(
            spec,
            payload,
            distance,
            reconstruction_seed=DEFAULT_RECONSTRUCTION_SEED,
        ),
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if error is not None:
        raise RuntimeError(f"Empirical reconstruction {arm_id} failed: {error}") from error
    if value is None:
        raise RuntimeError(f"Empirical reconstruction {arm_id} returned no result.")
    tree, _levels, returned_root, diagnostics = value
    if not nx.is_arborescence(tree):
        raise ValueError(f"Empirical reconstruction {arm_id} is not an arborescence.")
    required = set(observed_labels(payload))
    reconstructed = {
        attributes.get("cell_id")
        for _node, attributes in tree.nodes(data=True)
        if attributes.get("cell_id") is not None
    }
    if reconstructed != required:
        raise ValueError(
            f"Empirical reconstruction {arm_id} changed observation-label coverage."
        )
    external_tree = tree.copy()
    for _node, attributes in external_tree.nodes(data=True):
        label = attributes.get("cell_id")
        if label is not None:
            if label not in external_label_by_internal:
                raise ValueError(
                    f"Empirical reconstruction {arm_id} produced an unknown state label."
                )
            attributes["cell_id"] = external_label_by_internal[label]
    result = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "status": "success",
        "arm_id": arm_id,
        "problem": spec.problem,
        "truth_tree_supplied": False,
        "accuracy_metrics_computed": False,
        "observed_label_coverage": {
            "required_unique_label_count": len(required),
            "reconstructed_unique_label_count": len(reconstructed),
            "fraction": 1.0,
        },
        "tree_summary": tree_summary(external_tree),
        "temporal_diagnostics": _temporal_tree_diagnostics(external_tree),
        "reconstruction_metadata": json_safe(diagnostics),
        "resources": json_safe(resources),
        "returned_root": json_safe(returned_root),
        "tree": serialize_tree(external_tree),
    }
    write_json_atomic(output_root / "reconstructions" / f"{arm_id}.json", result)
    return result


def _report_markdown(report: Mapping[str, Any]) -> str:
    input_summary = report["input_summary"]
    biopsy_counts = ", ".join(
        f"B{index}={count}"
        for index, count in sorted(
            input_summary["cells_by_biopsy"].items(),
            key=lambda item: int(item[0]),
        )
    )
    lines = [
        "# KTN152 empirical CTBF feasibility",
        "",
        "> Exploratory feasibility analysis. Absolute CNP scale is unresolved, no true tree is available, and no accuracy metric was computed.",
        "",
        "## Frozen input",
        "",
        f"- Source CopyKit strict-QC cells: {input_summary['source_cell_count']} "
        f"({biopsy_counts}).",
        f"- Analyzed biopsy-indexed CNP occurrences after within-biopsy collapse: "
        f"{input_summary['occurrence_count']}.",
        f"- Analyzed distinct CNP states: {input_summary['state_count']}.",
        f"- Analyzed genomic bins: {input_summary['analyzed_bin_count']} of "
        f"{input_summary['source_bin_count']} across "
        f"{input_summary['chromosome_count']} chromosomes.",
        "",
        "## Reconstruction",
        "",
        "| Method | Nodes | Edges | Leaves | Unlabeled nodes | Max depth | Time reversals among generation-comparable edges | Wall time (s) | Peak RSS (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for value in report["reconstructions"]:
        summary = value["tree_summary"]
        temporal = value["temporal_diagnostics"]
        resources = value["resources"]
        peak = resources["memory"]["peak_rss_bytes"]
        lines.append(
            f"| `{value['arm_id']}` | {summary['node_count']} | "
            f"{summary['edge_count']} | {summary['leaf_count']} | "
            f"{summary['unlabeled_node_count']} | {summary['maximum_depth']} | "
            f"{temporal['generation_reversed_edge_count']} | "
            f"{resources['wall_time_ns'] / 1e9:.3f} | "
            f"{'NA' if peak is None else f'{peak / 1024**2:.1f}'} |"
        )
    lines.extend(
        [
            "",
            "The two outputs demonstrate execution on the same frozen empirical distance matrix. Their topology is not ranked because KTN152 has no known true evolutionary tree.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_output_manifest(output_root: Path) -> str:
    excluded = {"data_manifest.tsv", "_SUCCESS.json"}
    paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    rows = []
    for path in paths:
        relative = path.relative_to(output_root).as_posix()
        suffix = path.suffix.lower()
        media_type = {
            ".json": "application/json",
            ".md": "text/markdown",
        }.get(suffix, "application/octet-stream")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "media_type": media_type,
            }
        )
    manifest = output_root / "data_manifest.tsv"
    temporary = manifest.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "bytes", "sha256", "media_type"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    return _sha256_file(manifest)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    input_root = arguments.input_root.expanduser().resolve()
    output_root = arguments.output_root.expanduser().resolve()
    input_audit = _verify_input_root(input_root)
    source_payload = read_json(input_root / "ctbf_input.json")
    external_payload = _subset_payload(source_payload, preflight=arguments.preflight)
    payload, external_label_by_internal, state_label_map = _integerize_state_labels(
        external_payload
    )
    profiles = _profiles_by_label(payload)
    profile_lengths = {len(profile) for profile in profiles.values()}
    if len(profile_lengths) != 1:
        raise ValueError("Empirical CNPs do not share one bin count.")
    bin_rows = _read_tsv(input_root / "bins.tsv")
    if profile_lengths != {len(bin_rows)}:
        raise ValueError("Empirical CNP length differs from bins.tsv.")
    all_blocks = chromosome_slices(bin_rows)
    blocks = (
        tuple(sorted(all_blocks, key=lambda block: (block[2] - block[1], block[0]))[:2])
        if arguments.preflight
        else all_blocks
    )
    labels = sorted(profiles, key=stable_distance_label_key)

    runtime_config = replace(
        load_ctbs_runtime_config(arguments.cnp2cnp_config),
        cnp2cnp_timeout_seconds=float(arguments.cnp2cnp_timeout_seconds),
        cnp2cnp_capture_limit_bytes=int(arguments.capture_limit_bytes),
    )
    expected_contract = _run_contract(
        input_root=input_root,
        input_audit=input_audit,
        payload=payload,
        blocks=blocks,
        workers=arguments.workers,
        runtime_config=runtime_config,
        preflight=arguments.preflight,
        state_label_map=state_label_map,
    )
    contract = _load_or_create_contract(
        output_root,
        expected_contract,
        resume=arguments.resume,
    )
    state_label_map_path = output_root / "state_label_map.json"
    if state_label_map_path.is_file():
        if read_json(state_label_map_path) != state_label_map:
            raise ValueError("Stored empirical state-label mapping changed.")
    else:
        write_json_atomic(state_label_map_path, state_label_map)

    started = time.perf_counter_ns()
    with PeakProcessTreeRss() as distance_memory:
        chromosome_results = _compute_or_resume_chromosomes(
            output_root=output_root,
            blocks=blocks,
            labels=labels,
            profiles=profiles,
            runtime_config=runtime_config,
            workers=arguments.workers,
        )
    directed, minimum = aggregate_chromosome_distances(chromosome_results, labels)
    distance_resources = {
        "wall_time_ns": time.perf_counter_ns() - started,
        "memory": distance_memory.report(),
        "worker_count": min(arguments.workers, len(blocks)),
        "external_process_count": 2 * len(blocks),
    }
    provenance = {
        "schema_version": DISTANCE_PROVENANCE_SCHEMA_VERSION,
        "semantics_version": CHROMOSOME_DISTANCE_SEMANTICS,
        "metric": "cnp2cnp",
        "distance_mode": "any",
        "chromosome_aggregation": "sum_directed_then_minimum_bidirectional",
        "symmetrization": "minimum_bidirectional_after_chromosome_sum",
        "formula": "min(sum_c d_any(x_c,y_c),sum_c d_any(y_c,x_c))",
        "construction": "parallel_opposite_order_matrix_mode_per_chromosome",
        "chromosome_events_cross_boundaries": False,
        "chromosomes": [chromosome for chromosome, _start, _end in blocks],
        "cnp2cnp_source_sha256": contract["cnp2cnp"]["source_sha256"],
        "truth_or_reconstruction_used": False,
    }
    distance_payload = {
        "schema_version": DISTANCE_SCHEMA_VERSION,
        "ids": json_safe(labels),
        "matrix": json_safe(minimum),
        "directed_matrix": json_safe(directed),
        "provenance": provenance,
        "resources": distance_resources,
    }
    write_json_atomic(output_root / "distances" / "whole_genome.json", distance_payload)
    distance = DistanceMatrix(ids=list(labels), matrix=minimum, provenance=provenance)

    reconstructions = []
    for arm_id in ARM_IDS:
        path = output_root / "reconstructions" / f"{arm_id}.json"
        if path.is_file():
            value = read_json(path)
            if value.get("schema_version") != RECONSTRUCTION_SCHEMA_VERSION:
                raise ValueError(f"Stored reconstruction {arm_id} has the wrong schema.")
            if value.get("arm_id") != arm_id or value.get("status") != "success":
                raise ValueError(f"Stored reconstruction {arm_id} is not resumable.")
        else:
            value = _run_reconstruction(
                output_root=output_root,
                arm_id=arm_id,
                payload=payload,
                distance=distance,
                timeout_seconds=arguments.reconstruction_timeout_seconds,
                rss_limit_bytes=arguments.rss_limit_bytes,
                external_label_by_internal=external_label_by_internal,
            )
        reconstructions.append(value)
        print(f"reconstruction: {arm_id} complete", flush=True)

    export_summary = read_json(input_root / "export_summary.json")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "scientific_role": (
            "technical_preflight"
            if arguments.preflight
            else "exploratory_empirical_feasibility"
        ),
        "absolute_cnp_gate_status": "rejected_unresolved_scale",
        "truth_tree_available": False,
        "accuracy_metrics_computed": False,
        "input_summary": {
            "source_cell_count": export_summary["included_cell_count"],
            "cells_by_biopsy": export_summary["included_cells_by_biopsy"],
            "occurrence_count": sum(len(level["states"]) for level in payload["levels"]),
            "state_count": len(labels),
            "source_bin_count": len(bin_rows),
            "analyzed_bin_count": sum(end - start for _chromosome, start, end in blocks),
            "chromosome_count": len(blocks),
        },
        "distance": {
            "semantics_version": CHROMOSOME_DISTANCE_SEMANTICS,
            "resources": distance_resources,
        },
        "reconstructions": reconstructions,
        "interpretation_limit": (
            "Execution demonstrates feasibility only. The calls are caller-derived, "
            "their absolute scale is unresolved, and the inferred trees have no "
            "empirical ground truth."
        ),
    }
    write_json_atomic(output_root / "report.json", report)
    (output_root / "report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    contract["status"] = "complete"
    contract["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(output_root / "run_contract.json", contract)
    manifest_sha256 = _write_output_manifest(output_root)
    write_json_atomic(
        output_root / "_SUCCESS.json",
        {
            "schema_version": "ctbf-v5-empirical-feasibility-success-v1",
            "status": "complete",
            "run_mode": contract["run_mode"],
            "data_manifest_sha256": manifest_sha256,
        },
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cnp2cnp-config", type=Path, default=Path("ctbs_config.json"))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--cnp2cnp-timeout-seconds",
        type=float,
        default=DEFAULT_CNP2CNP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--reconstruction-timeout-seconds",
        type=int,
        default=DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument(
        "--capture-limit-bytes",
        type=int,
        default=DEFAULT_CAPTURE_LIMIT_BYTES,
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.workers <= 8:
        parser.error("--workers must lie between 1 and 8")
    if arguments.cnp2cnp_timeout_seconds <= 0:
        parser.error("--cnp2cnp-timeout-seconds must be positive")
    if arguments.reconstruction_timeout_seconds <= 0:
        parser.error("--reconstruction-timeout-seconds must be positive")
    if arguments.rss_limit_bytes <= 0 or arguments.capture_limit_bytes <= 0:
        parser.error("resource limits must be positive")
    report = run(arguments)
    print(
        "complete: "
        f"{report['input_summary']['state_count']} states; "
        f"{report['input_summary']['analyzed_bin_count']} bins; "
        f"{len(report['reconstructions'])} reconstructions; "
        "0 accuracy claims"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
