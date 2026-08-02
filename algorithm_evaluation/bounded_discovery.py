#!/usr/bin/env python3
"""Manifest-driven bounded discovery for G0-02-C and G0-03-B.

Truth is created and serialized separately from reconstruction inputs.  Every
reconstruction receives only biopsy records, declared distance inputs, radius,
and reconstruction seed.  Truth ancestry enters only the evaluator and the
post-reconstruction direction audit.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import threading
import time
import traceback

import networkx as nx
import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - recorded explicitly at runtime
    psutil = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnp2cnp_direction_ablation import (  # noqa: E402
    audit_directed_distances,
    audit_fast_reconstruction_sensitivity,
    audit_fast_row_order_sensitivity,
    canonical_fast_row_order,
    ordered_triangle_fast_view,
)
from ctbf_constraints import MIN_TOTAL_BIOPSY_CELLS  # noqa: E402
from ctbs import (  # noqa: E402
    Cnp2CnpDirectedFileDistanceProvider,
    Cnp2CnpFileDistanceProvider,
    Cnp2CnpOrderedTriangleFastDistanceProvider,
    CtbsRuntimeConfig,
    DistanceMatrix,
    load_ctbs_runtime_config,
    unique_cells_by_cell_id,
)
from distance_semantics import DirectedDistanceBundle, stable_distance_label_key  # noqa: E402
from evaluator import grf_tree  # noqa: E402
from evaluator_full import evaluate_4  # noqa: E402
from reconstructor import build_evolution_tree  # noqa: E402
from reconstructor_registry import resolve_reconstruction_algorithm  # noqa: E402
from simulator import CancerCellEvolutionSimulator, Genotype  # noqa: E402


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "experimental_description"
    / "g0_02_c_bounded_discovery_manifest.json"
)
RESULT_SCHEMA_VERSION = "ctbf-bounded-reconstruction-record-v1"
SUMMARY_SCHEMA_VERSION = "ctbf-bounded-reconstruction-summary-v1"
SOURCE_FILES = (
    "algorithm_evaluation/bounded_discovery.py",
    "cnp2cnp_direction_ablation.py",
    "ctbf_constraints.py",
    "ctbs.py",
    "ctbs_utils.py",
    "distance_semantics.py",
    "evaluator.py",
    "evaluator_full.py",
    "reconstructor.py",
    "reconstructor_algorithms.py",
    "reconstructor_algorithm_specs.py",
    "reconstructor_biopsy_blocks.py",
    "reconstructor_biopsy_guided.py",
    "reconstructor_engine.py",
    "reconstructor_temporal.py",
    "simulator.py",
    "test/data/config_for_pic.json",
    "test/data/config_high.json",
    "test/data/config_high_dm.json",
)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite floating-point value cannot be serialized.")
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(
            _json_safe(value),
            destination,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        destination.write("\n")
    os.replace(temporary, path)


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_contract_sha256(manifest):
    normalized = deepcopy(manifest)
    normalized.setdefault("source_contract", {})[
        "manifest_sha256_at_execution"
    ] = None
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def derive_seed(namespace, stream, replicate):
    payload = f"{namespace}:{stream}:{int(replicate)}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "big")


def collect_source_hashes(project_root=PROJECT_ROOT):
    return {
        relative: file_sha256(Path(project_root) / relative)
        for relative in SOURCE_FILES
    }


def validate_manifest(manifest, *, project_root=PROJECT_ROOT, require_sources=True):
    if manifest.get("schema_version") != "ctbf-bounded-reconstruction-discovery-manifest-v1":
        raise ValueError("Unsupported bounded-discovery manifest schema.")
    if manifest.get("held_out_evidence_allowed") is not False:
        raise ValueError("Bounded discovery must explicitly prohibit held-out evidence.")
    if manifest.get("data_dependent_stopping_allowed") is not False:
        raise ValueError("Data-dependent stopping must be disabled.")

    cases = manifest.get("cases") or []
    replicates = manifest.get("replicates") or []
    arms = manifest.get("portfolio_arms") or []
    for label, records, key in (
        ("case", cases, "id"),
        ("replicate", replicates, "replicate"),
        ("arm", arms, "id"),
    ):
        values = [record[key] for record in records]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Manifest {label} identifiers must be nonempty and unique.")

    expected_count = len(cases) * len(replicates)
    if manifest["failure_policy"]["required_record_count"] != expected_count:
        raise ValueError("required_record_count does not equal cases times replicates.")

    derivation = manifest["seed_derivation"]
    for record in replicates:
        for stream in derivation["streams"]:
            field = f"{stream}_seed"
            expected = derive_seed(
                derivation["namespace"],
                stream,
                record["replicate"],
            )
            if record[field] != expected:
                raise ValueError(
                    f"Replicate {record['replicate']} has an invalid {field}."
                )

    for case in cases:
        config_path = Path(project_root) / case["config"]
        if file_sha256(config_path) != case["config_sha256"]:
            raise ValueError(f"Config hash mismatch for case {case['id']}.")
        generations = case["biopsy_generations"]
        if generations != sorted(set(generations)) or len(generations) < 2:
            raise ValueError(f"Case {case['id']} needs ordered unique biopsy generations.")
        if not 0 < float(case["biopsy_size_scalable"]) <= 1:
            raise ValueError(f"Case {case['id']} has an invalid biopsy fraction.")

    if require_sources:
        source_contract = manifest.get("source_contract") or {}
        expected_hashes = source_contract.get("required_file_sha256")
        if not isinstance(expected_hashes, dict) or not expected_hashes:
            raise ValueError("Manifest source hashes are not frozen.")
        actual_hashes = collect_source_hashes(project_root)
        if expected_hashes != actual_hashes:
            changed = sorted(
                set(expected_hashes) | set(actual_hashes),
                key=str,
            )
            changed = [
                path
                for path in changed
                if expected_hashes.get(path) != actual_hashes.get(path)
            ]
            raise ValueError("Frozen source hash mismatch: " + ", ".join(changed))
        expected_manifest_hash = source_contract.get("manifest_sha256_at_execution")
        actual_manifest_hash = manifest_contract_sha256(manifest)
        if expected_manifest_hash != actual_manifest_hash:
            raise ValueError("Manifest execution hash is missing or stale.")
    return manifest


class PeakProcessTreeRss:
    """Sample aggregate RSS for this process and recursive children."""

    def __init__(self, interval_seconds=0.01):
        self.interval_seconds = interval_seconds
        self.peak_bytes = None
        self.sample_count = 0
        self.child_sampling_available = psutil is not None
        self.sampling_available = psutil is not None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        if psutil is None:
            return
        try:
            root = psutil.Process(os.getpid())
            try:
                children = root.children(recursive=True)
            except Exception:
                self.child_sampling_available = False
                children = []
            processes = [root] + children
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.sample_count += 1
            self.peak_bytes = rss if self.peak_bytes is None else max(self.peak_bytes, rss)
        except Exception:
            self.sampling_available = False
            return

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self):
        self._sample()
        if psutil is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._sample()

    def report(self):
        return {
            "method": (
                "unavailable_psutil_not_installed"
                if psutil is None
                else (
                    "unavailable_psutil_permission_denied"
                    if not self.sampling_available
                    else (
                        "psutil_process_tree_rss_10ms"
                        if self.child_sampling_available
                        else "psutil_runner_process_rss_10ms_child_access_denied"
                    )
                )
            ),
            "peak_rss_bytes": self.peak_bytes,
            "sample_count": self.sample_count,
        }


def measured_call(function, *args, **kwargs):
    start = time.perf_counter_ns()
    with PeakProcessTreeRss() as memory:
        value = function(*args, **kwargs)
    return value, {
        "wall_time_ns": time.perf_counter_ns() - start,
        "memory": memory.report(),
    }


def _actual_root(tree):
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one root, found {len(roots)}.")
    return roots[0]


def serialize_tree(tree):
    return {
        "directed": bool(tree.is_directed()),
        "nodes": [
            {
                "node_id": _json_safe(node),
                "attributes": _json_safe(data),
            }
            for node, data in sorted(tree.nodes(data=True), key=lambda item: str(item[0]))
        ],
        "edges": [
            {
                "source": _json_safe(source),
                "target": _json_safe(target),
                "attributes": _json_safe(data),
            }
            for source, target, data in sorted(
                tree.edges(data=True),
                key=lambda item: (str(item[0]), str(item[1])),
            )
        ],
    }


def deserialize_tree(serialized):
    tree = nx.DiGraph() if serialized.get("directed", True) else nx.Graph()
    for node in serialized.get("nodes", []):
        tree.add_node(node["node_id"], **node.get("attributes", {}))
    for edge in serialized.get("edges", []):
        tree.add_edge(
            edge["source"],
            edge["target"],
            **edge.get("attributes", {}),
        )
    return tree


def serialize_biopsies(cell_lists, generations):
    return [
        {
            "level": level,
            "generation": int(generation),
            "observations": [
                {
                    "cell_id": _json_safe(cell.cell_id),
                    "genome": _json_safe(np.asarray(cell.genome)),
                }
                for cell in sorted(
                    cells,
                    key=lambda cell: stable_distance_label_key(cell.cell_id),
                )
            ],
        }
        for level, (generation, cells) in enumerate(zip(generations, cell_lists))
    ]


def deserialize_biopsies(serialized):
    return [
        [
            Genotype(
                observation["genome"],
                observation["cell_id"],
                generation=level["generation"],
                cell_id=observation["cell_id"],
            )
            for observation in level["observations"]
        ]
        for level in serialized
    ]


def _observed_state_levels(cell_lists):
    levels = defaultdict(set)
    for level, cells in enumerate(cell_lists):
        for cell in cells:
            levels[cell.cell_id].add(level)
    return dict(levels)


def derive_truth_directions(true_tree, observed_ids, state_levels):
    """Return unambiguous state-level truth directions for evaluator use."""
    observed_ids = sorted(set(observed_ids), key=stable_distance_label_key)
    nodes_by_state = defaultdict(list)
    for node, data in true_tree.nodes(data=True):
        cell_id = data.get("cell_id")
        if cell_id in observed_ids:
            nodes_by_state[cell_id].append(node)

    descendants = {node: nx.descendants(true_tree, node) for node in true_tree.nodes}
    directions = []
    time_decided = []
    counts = Counter()
    for left_index, left in enumerate(observed_ids):
        if not nodes_by_state[left]:
            raise ValueError(f"Observed state {left!r} is absent from canonical truth.")
        for right in observed_ids[left_index + 1:]:
            if not nodes_by_state[right]:
                raise ValueError(f"Observed state {right!r} is absent from canonical truth.")
            left_to_right = any(
                right_node in descendants[left_node]
                for left_node in nodes_by_state[left]
                for right_node in nodes_by_state[right]
            )
            right_to_left = any(
                left_node in descendants[right_node]
                for right_node in nodes_by_state[right]
                for left_node in nodes_by_state[left]
            )
            counts["unordered_observed_state_pairs"] += 1
            if left_to_right and right_to_left:
                counts["recurrently_ambiguous"] += 1
                continue
            if not left_to_right and not right_to_left:
                counts["truth_incomparable"] += 1
                continue
            ancestor, descendant = (
                (left, right) if left_to_right else (right, left)
            )
            directions.append((ancestor, descendant))
            counts["truth_unambiguous"] += 1

            left_levels = state_levels[left]
            right_levels = state_levels[right]
            if max(left_levels) < min(right_levels) or max(right_levels) < min(left_levels):
                time_decided.append((left, right))
                counts["strictly_time_decided"] += 1

    return directions, time_decided, dict(counts)


def _runtime_config_for_work(base_config, work_dir, stem):
    return replace(
        base_config,
        in_file_name=str(Path(work_dir) / f"{stem}_input.txt"),
        out_file_name=str(Path(work_dir) / f"{stem}_matrix.txt"),
        sim_dm=str(Path(work_dir) / f"{stem}_truth_matrix.txt"),
    )


def _distance_record(distance_input, measurement):
    provenance = distance_input.provenance
    return {
        "wall_time_ns": measurement["wall_time_ns"],
        "memory": measurement["memory"],
        "external_process_count": (
            None if provenance is None else provenance.get("external_process_count")
        ),
        "directional_transformation_count": (
            None
            if provenance is None
            else provenance.get("directional_transformation_count")
        ),
        "provenance": provenance,
    }


def _align_matrix(ids, matrix, requested_ids):
    positions = {cell_id: index for index, cell_id in enumerate(ids)}
    if set(positions) != set(requested_ids):
        raise ValueError("Distance inputs contain different state ids.")
    alignment = [positions[cell_id] for cell_id in requested_ids]
    return np.asarray(matrix)[np.ix_(alignment, alignment)]


def compute_distance_inputs(unique_cells, base_runtime_config, work_dir):
    fast_provider = Cnp2CnpOrderedTriangleFastDistanceProvider(
        _runtime_config_for_work(base_runtime_config, work_dir, "fast")
    )
    minimum_provider = Cnp2CnpFileDistanceProvider(
        _runtime_config_for_work(base_runtime_config, work_dir, "minimum")
    )
    directed_provider = Cnp2CnpDirectedFileDistanceProvider(
        _runtime_config_for_work(base_runtime_config, work_dir, "directed")
    )

    fast, fast_measurement = measured_call(fast_provider.compute, unique_cells)
    minimum, minimum_measurement = measured_call(minimum_provider.compute, unique_cells)
    directed, directed_measurement = measured_call(directed_provider.compute, unique_cells)
    if not isinstance(fast, DistanceMatrix) or not isinstance(minimum, DistanceMatrix):
        raise TypeError("Fast and minimum providers must return DistanceMatrix.")
    if not isinstance(directed, DirectedDistanceBundle):
        raise TypeError("Directed provider must return DirectedDistanceBundle.")

    reference_ids = list(directed.ids)
    aligned_minimum = _align_matrix(minimum.ids, minimum.matrix, reference_ids)
    if not np.array_equal(aligned_minimum, directed.minimum_matrix):
        raise ValueError("Default minimum and directed-bundle minimum disagree.")
    derived_fast_ids, derived_fast_matrix = ordered_triangle_fast_view(directed)
    aligned_fast = _align_matrix(fast.ids, fast.matrix, derived_fast_ids)
    if not np.array_equal(aligned_fast, derived_fast_matrix):
        raise ValueError("Fast provider and canonical directed-bundle projection disagree.")

    return {
        "fast": fast,
        "minimum": minimum,
        "directed": directed,
        "records": {
            "ordered_triangle_fast": _distance_record(fast, fast_measurement),
            "minimum_bidirectional": _distance_record(minimum, minimum_measurement),
            "minimum_with_directed": _distance_record(directed, directed_measurement),
        },
    }


def _arm_build_spec(arm_id):
    specifications = {
        "classical_partial": ("neighbor_joining_classical", "pooled", "minimum"),
        "biopsy_guided_classical": (
            "neighbor_joining_classical",
            "biopsy_guided",
            "minimum",
        ),
        "rooted_labeled_nj": ("rooted_labeled_nj", "pooled", "minimum"),
        "temporal_minimum": (
            "temporal_cnp_arborescence",
            "ordered",
            "minimum",
        ),
        "temporal_minimum_no_time": (
            "temporal_cnp_arborescence_no_time",
            "ordered",
            "minimum",
        ),
        "legacy_closest_pair": ("neighbor_joining_baseline", "pooled", "minimum"),
        "anticentral_parsimony": (
            "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
            "pooled",
            "minimum",
        ),
        "temporal_fast": (
            "temporal_cnp_arborescence",
            "ordered",
            "fast",
        ),
        "temporal_directed": (
            "temporal_cnp_arborescence_directed",
            "ordered",
            "directed",
        ),
        "temporal_directed_no_time": (
            "temporal_cnp_arborescence_directed_no_time",
            "ordered",
            "directed",
        ),
    }
    return specifications[arm_id]


def _tree_output_summary(tree, actual_root, cell_lists):
    if not nx.is_arborescence(tree):
        raise ValueError("Reconstruction output is not one directed arborescence.")
    labels = [data.get("cell_id") for _, data in tree.nodes(data=True)]
    observed_ids = {
        cell.cell_id
        for cells in cell_lists
        for cell in cells
    }
    output_labels = {label for label in labels if label is not None}
    occurrence_expected = Counter(
        (level, cell.cell_id)
        for level, cells in enumerate(cell_lists)
        for cell in cells
    )
    occurrence_output = Counter(
        (data.get("biopsy_level"), data.get("cell_id"))
        for _, data in tree.nodes(data=True)
        if "biopsy_level" in data
    )
    return {
        "root": _json_safe(actual_root),
        "node_count": tree.number_of_nodes(),
        "edge_count": tree.number_of_edges(),
        "unlabeled_node_count": sum(label is None for label in labels),
        "outside_observed_label_count": len(output_labels - observed_ids),
        "observed_state_coverage": (
            len(output_labels & observed_ids) / len(observed_ids)
            if observed_ids
            else 1.0
        ),
        "occurrence_signature_exact": (
            occurrence_output == occurrence_expected
            if occurrence_output
            else None
        ),
    }


def run_arm(
    arm,
    cell_lists,
    distance_inputs,
    *,
    reconstruction_seed,
    r_dist,
    true_tree,
    true_root,
    observed_ids,
):
    arm_id = arm["id"]
    algorithm_name, input_mode, distance_name = _arm_build_spec(arm_id)
    if arm["algorithm"] != algorithm_name:
        raise ValueError(f"Manifest/code algorithm mismatch for {arm_id}.")
    algorithm = resolve_reconstruction_algorithm(algorithm_name)
    distance_input = distance_inputs[distance_name]
    pooled = [[cell for cells in cell_lists for cell in cells]]
    build_input = pooled if input_mode == "pooled" else cell_lists
    only_nj = input_mode == "pooled"

    def build():
        return build_evolution_tree(
            build_input,
            seed=reconstruction_seed,
            r=r_dist,
            only_nj=only_nj,
            distance_matrix=distance_input,
            neighbor_joining=algorithm,
        )

    try:
        (tree, _levels, returned_root), measurement = measured_call(build)
        actual_root = _actual_root(tree)
        normalized_observed_labels = {
            str(cell_id).strip()
            for cell_id in observed_ids
            if cell_id is not None and str(cell_id).strip()
        }
        ancestry = evaluate_4(
            true_tree,
            tree,
            restrict_labels=normalized_observed_labels,
        )
        metrics = {
            "ad_f1": ancestry["ancestors_unique_restricted"]["F1"],
            "grf": grf_tree(true_tree, true_root, tree, actual_root),
            "ad_f1_counts": ancestry["ancestors_unique_restricted"],
        }
        return {
            "status": "success",
            "algorithm": algorithm_name,
            "input_mode": input_mode,
            "distance_input": distance_name,
            "returned_root": _json_safe(returned_root),
            "runtime": measurement,
            "output": _tree_output_summary(tree, actual_root, cell_lists),
            "metrics": metrics,
            "tree": serialize_tree(tree),
        }
    except Exception as exc:  # explicit per-arm failure is part of the protocol
        return {
            "status": "failure",
            "algorithm": algorithm_name,
            "input_mode": input_mode,
            "distance_input": distance_name,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }


def _fast_row_orders(ids):
    canonical = list(canonical_fast_row_order(ids))
    if len(canonical) <= 1:
        rotated = list(canonical)
    else:
        rotated = canonical[1:] + canonical[:1]
    return [canonical, list(reversed(canonical)), rotated]


def run_case_record(case, replicate, manifest, *, output_root, base_runtime_config=None):
    simulation_start = time.perf_counter_ns()
    config_path = PROJECT_ROOT / case["config"]
    simulator = CancerCellEvolutionSimulator(
        str(config_path),
        seed=replicate["simulation_seed"],
    )
    simulator.run_simulation()
    simulation_wall_time_ns = time.perf_counter_ns() - simulation_start

    cell_lists = []
    biopsy_counts = []
    for generation in case["biopsy_generations"]:
        cells = simulator.perform_biopsy(
            generation=generation,
            biopsy_size_scalable=case["biopsy_size_scalable"],
            seed=replicate["biopsy_seed"],
        )
        biopsy_counts.append(len(cells))
        if not cells:
            raise ValueError(
                f"Frozen biopsy generation {generation} is empty; no case replacement is allowed."
            )
        cell_lists.append(cells)

    all_occurrences = [cell for cells in cell_lists for cell in cells]
    if len(all_occurrences) < manifest["failure_policy"]["minimum_total_observed_occurrences"]:
        raise ValueError(
            f"Observed only {len(all_occurrences)} biopsy occurrences; minimum is "
            f"{manifest['failure_policy']['minimum_total_observed_occurrences']}."
        )
    if len(all_occurrences) < MIN_TOTAL_BIOPSY_CELLS:
        raise ValueError("Record does not meet CTBF's reconstruction minimum biopsy size.")

    unique_cells = unique_cells_by_cell_id(all_occurrences)
    observed_ids = [cell.cell_id for cell in unique_cells]
    true_tree = simulator.canonicalized_tree_by_genome()
    true_root = _actual_root(true_tree)
    serialized_biopsies = serialize_biopsies(cell_lists, case["biopsy_generations"])

    base_runtime_config = base_runtime_config or load_ctbs_runtime_config()
    work_parent = Path(output_root) / "work"
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{case['id']}_r{replicate['replicate']}_",
        dir=work_parent,
    ) as work_dir:
        distance_inputs = compute_distance_inputs(
            unique_cells,
            base_runtime_config,
            work_dir,
        )

    state_levels = _observed_state_levels(cell_lists)
    truth_directions, time_decided, truth_selection = derive_truth_directions(
        true_tree,
        observed_ids,
        state_levels,
    )
    genomes_by_id = {cell.cell_id: np.asarray(cell.genome) for cell in unique_cells}
    directed_bundle = distance_inputs["directed"]
    direction_audit = audit_directed_distances(
        directed_bundle,
        genomes_by_id,
        truth_directions=truth_directions,
        time_decided_pairs=time_decided,
    )
    row_orders = _fast_row_orders(directed_bundle.ids)
    fast_matrix_audit = audit_fast_row_order_sensitivity(directed_bundle, row_orders)
    fast_tree_audit = {
        "ordered": audit_fast_reconstruction_sensitivity(
            cell_lists,
            directed_bundle,
            row_orders,
            seed=replicate["reconstruction_seed"],
            use_time=True,
        ),
        "no_time": audit_fast_reconstruction_sensitivity(
            cell_lists,
            directed_bundle,
            row_orders,
            seed=replicate["reconstruction_seed"],
            use_time=False,
        ),
    }

    arms = {}
    for arm in manifest["portfolio_arms"]:
        arms[arm["id"]] = run_arm(
            arm,
            cell_lists,
            distance_inputs,
            reconstruction_seed=replicate["reconstruction_seed"],
            r_dist=case["r_dist"],
            true_tree=true_tree,
            true_root=true_root,
            observed_ids=observed_ids,
        )

    fast = distance_inputs["fast"]
    minimum = distance_inputs["minimum"]
    directed = distance_inputs["directed"]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_contract_sha256(manifest),
        "status": "complete",
        "case": deepcopy(case),
        "replicate": deepcopy(replicate),
        "input_summary": {
            "simulation_wall_time_ns": simulation_wall_time_ns,
            "truth_node_count": true_tree.number_of_nodes(),
            "truth_edge_count": true_tree.number_of_edges(),
            "biopsy_counts": biopsy_counts,
            "observed_occurrence_count": len(all_occurrences),
            "observed_unique_state_count": len(unique_cells),
            "recurrent_observation_count": len(all_occurrences) - len(unique_cells),
        },
        "replay_input": {
            "truth_root": _json_safe(true_root),
            "truth_tree": serialize_tree(true_tree),
            "biopsies": serialized_biopsies,
            "distance_ids": _json_safe(directed.ids),
            "ordered_triangle_fast_matrix": _json_safe(fast.matrix),
            "minimum_bidirectional_matrix": _json_safe(minimum.matrix),
            "directed_matrix": _json_safe(directed.directed_matrix),
        },
        "distance": distance_inputs["records"],
        "direction_truth_selection": truth_selection,
        "direction_audit": direction_audit,
        "fast_order_audit": {
            "matrix": fast_matrix_audit,
            "tree": fast_tree_audit,
        },
        "arms": arms,
        "runtime_environment": runtime_environment(base_runtime_config),
    }


def failure_record(case, replicate, manifest, exc):
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_contract_sha256(manifest),
        "status": "record_failure",
        "case": deepcopy(case),
        "replicate": deepcopy(replicate),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def runtime_environment(runtime_config):
    cnp_path = Path(runtime_config.cnp2cnp_file)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "networkx": nx.__version__,
        "numpy": np.__version__,
        "psutil": None if psutil is None else psutil.__version__,
        "cnp2cnp_file": str(cnp_path),
        "cnp2cnp_sha256": file_sha256(cnp_path),
    }


def record_path(output_root, case_id, replicate):
    return (
        Path(output_root)
        / "records"
        / case_id
        / f"replicate_{int(replicate):02d}.json"
    )


def _numeric_summary(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _bootstrap_block_mean(block_values, repetitions, seed):
    if not block_values:
        return {"lower": None, "upper": None, "block_count": 0}
    ordered = [block_values[key] for key in sorted(block_values)]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ordered), size=(repetitions, len(ordered)))
    samples = np.asarray(ordered, dtype=float)[draws].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "block_count": len(ordered),
    }


def paired_metric_summary(records, left_arm, right_arm, metric, analysis_contract):
    rows = []
    for record in records:
        if record.get("status") != "complete":
            continue
        left = record["arms"].get(left_arm, {})
        right = record["arms"].get(right_arm, {})
        if left.get("status") != "success" or right.get("status") != "success":
            continue
        delta = float(left["metrics"][metric]) - float(right["metrics"][metric])
        rows.append({
            "case": record["case"]["id"],
            "replicate": record["replicate"]["replicate"],
            "delta": delta,
        })

    tolerance = analysis_contract["tie_tolerance"]
    deltas = [row["delta"] for row in rows]
    by_replicate = defaultdict(list)
    by_case = defaultdict(list)
    for row in rows:
        by_replicate[row["replicate"]].append(row["delta"])
        by_case[row["case"]].append(row["delta"])
    block_values = {
        replicate: statistics.fmean(values)
        for replicate, values in by_replicate.items()
    }
    return {
        "left_minus_right": f"{left_arm} - {right_arm}",
        "metric": metric,
        "complete_pair_count": len(rows),
        "summary": _numeric_summary(deltas),
        "wins": sum(delta > tolerance for delta in deltas),
        "ties": sum(abs(delta) <= tolerance for delta in deltas),
        "losses": sum(delta < -tolerance for delta in deltas),
        "mean_by_case": {
            case: statistics.fmean(values)
            for case, values in sorted(by_case.items())
        },
        "replicate_block_means": {
            str(replicate): value
            for replicate, value in sorted(block_values.items())
        },
        "block_bootstrap_mean_95": _bootstrap_block_mean(
            block_values,
            analysis_contract["bootstrap_repetitions"],
            analysis_contract["bootstrap_seed"],
        ),
    }


def _arm_failures(records, arm_ids):
    result = {}
    total = len(records)
    for arm_id in arm_ids:
        failures = 0
        for record in records:
            if record.get("status") != "complete":
                failures += 1
            elif record.get("arms", {}).get(arm_id, {}).get("status") != "success":
                failures += 1
        result[arm_id] = {
            "failures": failures,
            "total_records": total,
            "failure_rate": failures / total if total else None,
        }
    return result


def _wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return {"lower": None, "upper": None}
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return {"lower": center - radius, "upper": center + radius}


def aggregate_direction_audit(records):
    totals = Counter()
    strata = defaultdict(Counter)
    profile_strata = Counter()
    margins = []
    calibration = defaultdict(Counter)
    for record in records:
        if record.get("status") != "complete":
            continue
        audit = record["direction_audit"]
        totals["record_count"] += 1
        totals["unordered_pair_count"] += audit["unordered_pair_count"]
        totals["asymmetric_pair_count"] += audit["asymmetric_pair_count"]
        truth = audit["truth_direction"]
        for key in (
            "provided",
            "excluded_time_decided",
            "excluded_plausibility_decided",
            "excluded_neither_plausible",
            "eligible_both_plausible",
            "ties",
            "sign_informative",
            "correct",
            "incorrect",
        ):
            totals[key] += truth.get(key, 0)
        median = audit["asymmetry_magnitude"]["median"]
        if median is not None:
            margins.append(median)
        for name, values in audit["plausibility_strata"].items():
            strata[name].update(values)
        profile_strata.update(audit["profile_strata"])
        for difference, values in truth["accuracy_by_absolute_difference"].items():
            calibration[difference]["pairs"] += values["pairs"]
            calibration[difference]["correct"] += values["correct"]

    informative = totals["sign_informative"]
    correct = totals["correct"]
    eligible = totals["eligible_both_plausible"]
    unordered = totals["unordered_pair_count"]
    return {
        **dict(totals),
        "asymmetric_fraction": (
            totals["asymmetric_pair_count"] / unordered if unordered else None
        ),
        "median_of_record_asymmetry_medians": (
            statistics.median(margins) if margins else None
        ),
        "sign_accuracy": correct / informative if informative else None,
        "false_direction_rate": (
            totals["incorrect"] / informative if informative else None
        ),
        "sign_coverage": informative / eligible if eligible else None,
        "tie_fraction": totals["ties"] / eligible if eligible else None,
        "sign_accuracy_wilson_95": _wilson_interval(correct, informative),
        "plausibility_strata": {
            name: dict(values) for name, values in sorted(strata.items())
        },
        "profile_strata": dict(profile_strata),
        "accuracy_by_absolute_difference": {
            difference: {
                **dict(values),
                "accuracy": values["correct"] / values["pairs"],
            }
            for difference, values in sorted(
                calibration.items(),
                key=lambda item: float(item[0]),
            )
        },
    }


def aggregate_distance(records):
    modes = (
        "ordered_triangle_fast",
        "minimum_bidirectional",
        "minimum_with_directed",
    )
    result = {}
    for mode in modes:
        selected = [
            record["distance"][mode]
            for record in records
            if record.get("status") == "complete"
        ]
        result[mode] = {
            "wall_time_ns": _numeric_summary(
                [record["wall_time_ns"] for record in selected]
            ),
            "peak_rss_bytes": _numeric_summary(
                [
                    record["memory"]["peak_rss_bytes"]
                    for record in selected
                    if record["memory"]["peak_rss_bytes"] is not None
                ]
            ),
            "external_process_count_total": sum(
                record["external_process_count"] or 0 for record in selected
            ),
            "directional_transformation_count_total": sum(
                record["directional_transformation_count"] or 0
                for record in selected
            ),
        }
    speedups = [
        record["distance"]["minimum_bidirectional"]["wall_time_ns"]
        / record["distance"]["ordered_triangle_fast"]["wall_time_ns"]
        for record in records
        if record.get("status") == "complete"
        and record["distance"]["ordered_triangle_fast"]["wall_time_ns"] > 0
    ]
    result["minimum_over_fast_speedup"] = _numeric_summary(speedups)
    return result


def apply_promotion_gates(manifest, comparisons, failures, direction, distance):
    gates = manifest["promotion_gates"]
    ceiling = manifest["failure_policy"]["promotion_failure_ceiling"]

    anticentral_ad = comparisons["anticentral_vs_rooted_ad_f1"]
    anticentral_grf = comparisons["anticentral_vs_rooted_grf"]
    anticentral_spec = gates["anticentral_to_secondary_confirmation"]
    positive_regimes = sum(
        value > 0 for value in anticentral_ad["mean_by_case"].values()
    )
    anticentral_checks = {
        "ad_f1_mean": (
            anticentral_ad["summary"]["mean"] is not None
            and anticentral_ad["summary"]["mean"]
            >= anticentral_spec["minimum_ad_f1_mean_gain"]
        ),
        "ad_f1_bootstrap_lower": (
            anticentral_ad["block_bootstrap_mean_95"]["lower"] is not None
            and anticentral_ad["block_bootstrap_mean_95"]["lower"]
            >= anticentral_spec["minimum_ad_f1_block_bootstrap_lower"]
        ),
        "grf_mean": (
            anticentral_grf["summary"]["mean"] is not None
            and anticentral_grf["summary"]["mean"]
            >= anticentral_spec["minimum_grf_mean_gain"]
        ),
        "positive_regimes": positive_regimes
        >= anticentral_spec["minimum_positive_regime_count"],
        "failures": (
            failures["anticentral_parsimony"]["failures"]
            <= failures["rooted_labeled_nj"]["failures"]
            and failures["anticentral_parsimony"]["failure_rate"] <= ceiling
            and failures["rooted_labeled_nj"]["failure_rate"] <= ceiling
        ),
    }

    directed_ad = comparisons["directed_vs_minimum_ad_f1"]
    directed_grf = comparisons["directed_vs_minimum_grf"]
    directed_spec = gates["directed_to_secondary_confirmation"]
    directed_checks = {
        "sign_pair_count": direction["sign_informative"]
        >= directed_spec["minimum_truth_sign_informative_pairs"],
        "sign_accuracy": (
            direction["sign_accuracy"] is not None
            and direction["sign_accuracy"]
            > directed_spec["minimum_truth_sign_accuracy"]
        ),
        "sign_wilson_lower": (
            direction["sign_accuracy_wilson_95"]["lower"] is not None
            and direction["sign_accuracy_wilson_95"]["lower"]
            > directed_spec["minimum_wilson_95_lower_bound"]
        ),
        "ad_f1_mean": (
            directed_ad["summary"]["mean"] is not None
            and directed_ad["summary"]["mean"]
            >= directed_spec["minimum_ad_f1_mean_gain"]
        ),
        "ad_f1_bootstrap_lower": (
            directed_ad["block_bootstrap_mean_95"]["lower"] is not None
            and directed_ad["block_bootstrap_mean_95"]["lower"]
            >= directed_spec["minimum_ad_f1_block_bootstrap_lower"]
        ),
        "grf_mean": (
            directed_grf["summary"]["mean"] is not None
            and directed_grf["summary"]["mean"]
            >= directed_spec["minimum_grf_mean_gain"]
        ),
        "failures": (
            failures["temporal_directed"]["failures"]
            <= failures["temporal_minimum"]["failures"]
            and failures["temporal_directed"]["failure_rate"] <= ceiling
            and failures["temporal_minimum"]["failure_rate"] <= ceiling
        ),
    }

    fast_ad = comparisons["fast_vs_minimum_ad_f1"]
    fast_grf = comparisons["fast_vs_minimum_grf"]
    fast_spec = gates["fast_user_option"]
    fast_checks = {
        "speedup": (
            distance["minimum_over_fast_speedup"]["median"] is not None
            and distance["minimum_over_fast_speedup"]["median"]
            >= fast_spec["minimum_median_distance_speedup"]
        ),
        "ad_f1_mean": (
            fast_ad["summary"]["mean"] is not None
            and fast_ad["summary"]["mean"] >= fast_spec["minimum_ad_f1_mean_gain"]
        ),
        "grf_mean": (
            fast_grf["summary"]["mean"] is not None
            and fast_grf["summary"]["mean"] >= fast_spec["minimum_grf_mean_gain"]
        ),
    }

    temporal_blocked = (
        failures["temporal_minimum"]["failure_rate"] > ceiling
        or failures["temporal_minimum_no_time"]["failure_rate"] > ceiling
    )
    return {
        "temporal_primary_confirmation_blocked": temporal_blocked,
        "anticentral_to_secondary_confirmation": {
            "checks": anticentral_checks,
            "passed": all(anticentral_checks.values()),
            "positive_regime_count": positive_regimes,
        },
        "directed_to_secondary_confirmation": {
            "checks": directed_checks,
            "passed": all(directed_checks.values()),
        },
        "fast_useful_tradeoff": {
            "checks": fast_checks,
            "passed": all(fast_checks.values()),
            "availability_note": fast_spec["interpretation"],
        },
    }


def summarize_records(records, manifest):
    analysis = manifest["analysis_contract"]
    comparison_specs = {
        "biopsy_guided_vs_classical_grf": (
            "biopsy_guided_classical",
            "classical_partial",
            "grf",
        ),
        "temporal_vs_no_time_ad_f1": (
            "temporal_minimum",
            "temporal_minimum_no_time",
            "ad_f1",
        ),
        "temporal_vs_no_time_grf": (
            "temporal_minimum",
            "temporal_minimum_no_time",
            "grf",
        ),
        "temporal_vs_rooted_ad_f1": (
            "temporal_minimum",
            "rooted_labeled_nj",
            "ad_f1",
        ),
        "temporal_vs_rooted_grf": (
            "temporal_minimum",
            "rooted_labeled_nj",
            "grf",
        ),
        "anticentral_vs_rooted_ad_f1": (
            "anticentral_parsimony",
            "rooted_labeled_nj",
            "ad_f1",
        ),
        "anticentral_vs_rooted_grf": (
            "anticentral_parsimony",
            "rooted_labeled_nj",
            "grf",
        ),
        "directed_vs_minimum_ad_f1": (
            "temporal_directed",
            "temporal_minimum",
            "ad_f1",
        ),
        "directed_vs_minimum_grf": (
            "temporal_directed",
            "temporal_minimum",
            "grf",
        ),
        "directed_no_time_vs_minimum_no_time_ad_f1": (
            "temporal_directed_no_time",
            "temporal_minimum_no_time",
            "ad_f1",
        ),
        "fast_vs_minimum_ad_f1": (
            "temporal_fast",
            "temporal_minimum",
            "ad_f1",
        ),
        "fast_vs_minimum_grf": (
            "temporal_fast",
            "temporal_minimum",
            "grf",
        ),
    }
    comparisons = {
        name: paired_metric_summary(records, *spec, analysis)
        for name, spec in comparison_specs.items()
    }
    arm_ids = [arm["id"] for arm in manifest["portfolio_arms"]]
    failures = _arm_failures(records, arm_ids)
    direction = aggregate_direction_audit(records)
    distance = aggregate_distance(records)
    gates = apply_promotion_gates(
        manifest,
        comparisons,
        failures,
        direction,
        distance,
    )
    expected = manifest["failure_policy"]["required_record_count"]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_contract_sha256(manifest),
        "status": "complete" if len(records) == expected else "partial",
        "record_count": len(records),
        "expected_record_count": expected,
        "top_level_record_failures": sum(
            record.get("status") != "complete" for record in records
        ),
        "arm_failures": failures,
        "comparisons": comparisons,
        "direction_audit": direction,
        "distance": distance,
        "promotion_gates": gates,
        "interpretation_boundary": (
            "Bounded discovery only. Gate passage permits inclusion in a future "
            "held-out protocol; it is not a superiority claim."
        ),
    }


def load_all_records(output_root, manifest):
    records = []
    missing = []
    for case in manifest["cases"]:
        for replicate in manifest["replicates"]:
            path = record_path(output_root, case["id"], replicate["replicate"])
            if not path.exists():
                missing.append(str(path))
                continue
            record = _read_json(path)
            if record.get("manifest_sha256") != manifest_contract_sha256(manifest):
                raise ValueError(f"Record manifest hash mismatch: {path}")
            records.append(record)
    return records, missing


def write_checksums(output_root):
    output_root = Path(output_root)
    files = sorted(
        path
        for path in output_root.rglob("*.json")
        if path.name != "checksums.json"
        and "work" not in path.relative_to(output_root).parts
    )
    checksums = {
        str(path.relative_to(output_root)): file_sha256(path)
        for path in files
    }
    _write_json_atomic(output_root / "checksums.json", checksums)
    return checksums


def ensure_manifest_snapshot(output_root, manifest, manifest_path):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_root / "manifest.snapshot.json"
    if snapshot_path.exists():
        existing = _read_json(snapshot_path)
        if existing != manifest:
            raise ValueError("Output root contains a different manifest snapshot.")
    else:
        _write_json_atomic(snapshot_path, manifest)
    source_snapshot = {
        "required_file_sha256": collect_source_hashes(),
        "manifest_source_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_contract_sha256(manifest),
    }
    source_path = output_root / "source.snapshot.json"
    if source_path.exists() and _read_json(source_path) != source_snapshot:
        raise ValueError("Output root contains a different source snapshot.")
    if not source_path.exists():
        _write_json_atomic(source_path, source_snapshot)


def select_records(manifest, case_ids=(), replicate_ids=()):
    case_ids = set(case_ids)
    replicate_ids = {int(value) for value in replicate_ids}
    unknown_cases = case_ids - {case["id"] for case in manifest["cases"]}
    unknown_replicates = replicate_ids - {
        record["replicate"] for record in manifest["replicates"]
    }
    if unknown_cases:
        raise ValueError("Unknown case ids: " + ", ".join(sorted(unknown_cases)))
    if unknown_replicates:
        raise ValueError(
            "Unknown replicate ids: "
            + ", ".join(str(value) for value in sorted(unknown_replicates))
        )
    return [
        (case, replicate)
        for case in manifest["cases"]
        if not case_ids or case["id"] in case_ids
        for replicate in manifest["replicates"]
        if not replicate_ids or replicate["replicate"] in replicate_ids
    ]


def run_manifest(
    manifest,
    manifest_path,
    output_root,
    *,
    case_ids=(),
    replicate_ids=(),
    overwrite=False,
):
    validate_manifest(manifest)
    ensure_manifest_snapshot(output_root, manifest, manifest_path)
    selected = select_records(manifest, case_ids, replicate_ids)
    runtime_config = load_ctbs_runtime_config()
    for index, (case, replicate) in enumerate(selected, start=1):
        destination = record_path(
            output_root,
            case["id"],
            replicate["replicate"],
        )
        if destination.exists() and not overwrite:
            print(f"[{index}/{len(selected)}] existing {destination}", flush=True)
            continue
        print(
            f"[{index}/{len(selected)}] {case['id']} replicate "
            f"{replicate['replicate']}",
            flush=True,
        )
        try:
            record = run_case_record(
                case,
                replicate,
                manifest,
                output_root=output_root,
                base_runtime_config=runtime_config,
            )
        except Exception as exc:
            record = failure_record(case, replicate, manifest, exc)
        _write_json_atomic(destination, record)

    records, missing = load_all_records(output_root, manifest)
    if not missing:
        summary = summarize_records(records, manifest)
        _write_json_atomic(Path(output_root) / "summary.json", summary)
        write_checksums(output_root)
        print(f"Complete: {len(records)} records; summary written.", flush=True)
    else:
        write_checksums(output_root)
        print(
            f"Partial run: {len(records)} records present, {len(missing)} missing.",
            flush=True,
        )


def summarize_output(manifest, output_root):
    validate_manifest(manifest)
    records, missing = load_all_records(output_root, manifest)
    if missing:
        raise ValueError(
            f"Cannot produce the frozen summary while {len(missing)} records are missing."
        )
    summary = summarize_records(records, manifest)
    _write_json_atomic(Path(output_root) / "summary.json", summary)
    write_checksums(output_root)
    return summary


def format_dry_run(manifest, manifest_path, output_root, selected):
    lines = [
        f"Manifest: {Path(manifest_path).resolve()}",
        f"Manifest id: {manifest['manifest_id']}",
        f"Manifest contract SHA-256: {manifest_contract_sha256(manifest)}",
        f"Output root: {Path(output_root).resolve()}",
        f"Cases: {len(manifest['cases'])}",
        f"Replicates: {len(manifest['replicates'])}",
        f"Frozen records: {manifest['failure_policy']['required_record_count']}",
        f"Selected records: {len(selected)}",
        f"Arms per record: {len(manifest['portfolio_arms'])}",
        "Selected case/replicate keys:",
    ]
    lines.extend(
        f"  {case['id']} / {replicate['replicate']}"
        for case, replicate in selected
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen G0-02-C/G0-03-B bounded discovery."
    )
    parser.add_argument(
        "command",
        choices=("source-hashes", "dry-run", "run", "summarize"),
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--replicate", type=int, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = _read_json(manifest_path)
    output_root = (
        Path(args.output_root)
        if args.output_root
        else PROJECT_ROOT / manifest["result_root"]
    )

    if args.command == "source-hashes":
        validate_manifest(manifest, require_sources=False)
        print(json.dumps({
            "required_file_sha256": collect_source_hashes(),
            "manifest_sha256_at_execution": manifest_contract_sha256(manifest),
            "runtime": runtime_environment(load_ctbs_runtime_config()),
        }, indent=2, sort_keys=True))
        return 0

    validate_manifest(manifest)
    selected = select_records(manifest, args.case, args.replicate)
    if args.command == "dry-run":
        print(format_dry_run(manifest, manifest_path, output_root, selected))
        return 0
    if args.command == "run":
        run_manifest(
            manifest,
            manifest_path,
            output_root,
            case_ids=args.case,
            replicate_ids=args.replicate,
            overwrite=args.overwrite,
        )
        return 0
    summary = summarize_output(manifest, output_root)
    print(json.dumps(summary["promotion_gates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST",
    "RESULT_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "aggregate_direction_audit",
    "apply_promotion_gates",
    "collect_source_hashes",
    "derive_seed",
    "derive_truth_directions",
    "deserialize_biopsies",
    "deserialize_tree",
    "file_sha256",
    "manifest_contract_sha256",
    "paired_metric_summary",
    "run_case_record",
    "serialize_biopsies",
    "serialize_tree",
    "summarize_records",
    "validate_manifest",
]
