import subprocess
import sys
import tempfile
import time
import json
import os
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from copy import deepcopy
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor

from ctbf_constraints import MIN_TOTAL_BIOPSY_CELLS
from simulator import CancerCellEvolutionSimulator, Genotype
from reconstructor import build_evolution_tree, resolve_biopsy_guided_config, visualize_tree_plotly
from reconstructor_registry import get_algorithm_map, resolve_reconstruction_algorithm
from reconstructor_temporal import uses_ordered_occurrence_input
from distance_semantics import (
    CNP2CNP_DISTANCE,
    CNP2CNP_ORDERED_TRIANGLE_FAST,
    CNP2CNP_SYMMETRIZATION,
    DirectedDistanceBundle,
    cnp2cnp_provenance,
    combine_ordered_cnp2cnp_matrices,
    distance_input_cache_key,
    directed_bundle_from_ordered_cnp2cnp_matrices,
    minimum_bidirectional_distance,
    parse_cnp2cnp_directional_distance,
    parse_distance_label,
    parse_labeled_distance_matrix,
    stable_distance_label_key,
    validate_distance_label_coverage,
    validate_distance_matrix as _validate_distance_matrix,
)
from evaluator import grf_tree
from evaluator_full import evaluate_4, named_label
from ctbs_utils import to_newick, vizualize_nx_tree, get_biopsy_nodes_ids

DEFAULT_SIMULATOR_CONFIG_PATH = (
    Path(__file__).resolve().parent / "simulator_examples" / "default.json"
)

DEFAULT_CTBS_CONFIG = {
    "IN_FILE_NAME": "biopsy.txt",
    "OUT_FILE_NAME": "cnp_distance_matrix.txt",
    "SIM_DM": "sim_dm.txt",
    "cnp2cnp_FOLDER": "/Users/voronwe/Work/PyCharmProjects/cnp2cnp/examples",
    "cnp2cnp_FILE": "/Users/voronwe/Work/PyCharmProjects/cnp2cnp/cnp2cnp.py",
    "TRUE_TREE_ROOT_ID": 0,
    "RUN_SINGLE_TEST": {
        "seed": 2,
        "config": str(DEFAULT_SIMULATOR_CONFIG_PATH),
        "bedfile": None,
        "biopsy_size_scalable": 0.5,
        "biopsy_generations": [3, 5],
        "r_dist": 4,
        "write_newick": True,
        "visualize": False,
        "reconstruction_algorithm": "neighbor_joining_hybrid_anticentral_adaptive_v3",
        "biopsy_guided_strategy": None,
    },
}
CTBS_CONFIG_PATH = Path(__file__).with_name("ctbs_config.json")

RECONSTRUCTION_ALGORITHMS = get_algorithm_map()

CNP2CNP_DISTANCE_CONSTRUCTION_DEFAULT = CNP2CNP_SYMMETRIZATION
CNP2CNP_DISTANCE_CONSTRUCTION_FAST = CNP2CNP_ORDERED_TRIANGLE_FAST
CNP2CNP_DISTANCE_CONSTRUCTION_DIRECTED = "minimum_with_directed"
CNP2CNP_DISTANCE_CONSTRUCTIONS = frozenset({
    CNP2CNP_DISTANCE_CONSTRUCTION_DEFAULT,
    CNP2CNP_DISTANCE_CONSTRUCTION_FAST,
    CNP2CNP_DISTANCE_CONSTRUCTION_DIRECTED,
})
DEFAULT_DISTANCE_MAX_WORKERS = 4
MAX_DISTANCE_MAX_WORKERS = 32
CNP2CNP_EXECUTION_RECORD_SCHEMA_VERSION = "ctbf-cnp2cnp-execution-record-v1"
CNP2CNP_EXECUTION_SUMMARY_SCHEMA_VERSION = "ctbf-cnp2cnp-execution-summary-v1"
_CAPTURE_PREVIEW_CHARACTERS = 4096


class Cnp2CnpExecutionError(RuntimeError):
    """Checked cnp2cnp failure carrying a JSON-safe execution record."""

    def __init__(self, message, record=None):
        super().__init__(message, record)
        self.record = record

    def __str__(self):
        return str(self.args[0])


def _sha256_file(path):
    digest = sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _captured_stream(value):
    if value is None:
        text = ""
        encoded = b""
    elif isinstance(value, bytes):
        encoded = value
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
        encoded = text.encode("utf-8")
    return {
        "character_count": len(text),
        "byte_count": len(encoded),
        "sha256": sha256(encoded).hexdigest(),
        "preview": text[:_CAPTURE_PREVIEW_CHARACTERS],
        "preview_truncated": len(text) > _CAPTURE_PREVIEW_CHARACTERS,
    }


def _execution_record(command, workdir, completed, *, status, output_path=None):
    raw_workdir = Path(workdir)
    workdir = raw_workdir.resolve()
    workdir_prefixes = tuple(dict.fromkeys((str(raw_workdir), str(workdir))))
    normalized_command = []
    for value in command:
        text = str(value)
        for workdir_prefix in workdir_prefixes:
            if text == workdir_prefix:
                text = "<temporary-workdir>"
                break
            if text.startswith(workdir_prefix + os.sep):
                text = "<temporary-workdir>/" + Path(text).name
                break
        normalized_command.append(text)

    record = {
        "schema_version": CNP2CNP_EXECUTION_RECORD_SCHEMA_VERSION,
        "status": status,
        "returncode": getattr(completed, "returncode", None),
        "command": normalized_command,
        "working_directory": "isolated_temporary_directory",
        "stdout": _captured_stream(getattr(completed, "stdout", "")),
        "stderr": _captured_stream(getattr(completed, "stderr", "")),
    }
    if output_path is not None and Path(output_path).is_file():
        record["output_sha256"] = _sha256_file(output_path)
    return record


def _stream_byte_count(value):
    if value is None:
        return 0
    if isinstance(value, bytes):
        return len(value)
    return len(str(value).encode("utf-8"))


def _run_checked_cnp2cnp(
    command,
    workdir,
    *,
    output_path=None,
    timeout_seconds=None,
    capture_limit_bytes=None,
):
    run_kwargs = {
        "cwd": str(workdir),
        "capture_output": True,
        "text": True,
        "check": True,
    }
    if timeout_seconds is not None:
        run_kwargs["timeout"] = float(timeout_seconds)
    try:
        completed = subprocess.run(command, **run_kwargs)
    except subprocess.TimeoutExpired as exc:
        completed = type(
            "TimeoutFailure",
            (),
            {
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            },
        )()
        record = _execution_record(
            command,
            workdir,
            completed,
            status="timeout",
            output_path=output_path,
        )
        record["timeout_seconds"] = float(timeout_seconds)
        raise Cnp2CnpExecutionError(
            f"cnp2cnp exceeded its {timeout_seconds}-second timeout.",
            record,
        ) from exc
    except subprocess.CalledProcessError as exc:
        record = _execution_record(
            command,
            workdir,
            exc,
            status="failed",
            output_path=output_path,
        )
        raise Cnp2CnpExecutionError(
            f"cnp2cnp exited with status {exc.returncode}.",
            record,
        ) from exc
    except OSError as exc:
        completed = type(
            "LaunchFailure",
            (),
            {"returncode": None, "stdout": "", "stderr": str(exc)},
        )()
        record = _execution_record(
            command,
            workdir,
            completed,
            status="launch_failed",
            output_path=output_path,
        )
        raise Cnp2CnpExecutionError("cnp2cnp could not be started.", record) from exc

    if capture_limit_bytes is not None:
        stdout_bytes = _stream_byte_count(completed.stdout)
        stderr_bytes = _stream_byte_count(completed.stderr)
        if stdout_bytes > capture_limit_bytes or stderr_bytes > capture_limit_bytes:
            record = _execution_record(
                command,
                workdir,
                completed,
                status="capture_limit_exceeded",
                output_path=output_path,
            )
            record["capture_limit_bytes"] = int(capture_limit_bytes)
            raise Cnp2CnpExecutionError(
                "cnp2cnp stdout or stderr exceeded the configured capture limit.",
                record,
            )

    if output_path is not None and not Path(output_path).is_file():
        record = _execution_record(
            command,
            workdir,
            completed,
            status="missing_output",
        )
        raise Cnp2CnpExecutionError(
            "cnp2cnp completed without creating its matrix output.",
            record,
        )
    return completed, _execution_record(
        command,
        workdir,
        completed,
        status="success",
        output_path=output_path,
    )


def _invalid_output_record(record, message):
    invalid = deepcopy(record)
    invalid["status"] = "invalid_output"
    invalid["validation_error"] = str(message)
    return invalid


class _ExecutionSummaryAccumulator:
    def __init__(self):
        self.command_count = 0
        self.status_counts = {}
        self.nonempty_stderr_count = 0
        self._digest = sha256()
        self._first_two = []
        self._last = None

    def add(self, record):
        serialized = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if self.command_count:
            self._digest.update(b"\n")
        self._digest.update(serialized)
        self.command_count += 1
        status = record["status"]
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        self.nonempty_stderr_count += (
            record["stderr"]["character_count"] > 0
        )
        if len(self._first_two) < 2:
            self._first_two.append(deepcopy(record))
        self._last = deepcopy(record)

    def finish(self):
        summary = {
            "schema_version": CNP2CNP_EXECUTION_SUMMARY_SCHEMA_VERSION,
            "command_count": self.command_count,
            "status_counts": dict(self.status_counts),
            "records_sha256": self._digest.hexdigest(),
            "nonempty_stderr_count": self.nonempty_stderr_count,
        }
        if self.command_count <= 2:
            summary["records"] = self._first_two
        elif self.command_count:
            summary["record_samples"] = [self._first_two[0], self._last]
            summary["omitted_record_count"] = self.command_count - 2
        return summary


def _execution_summary(records):
    accumulator = _ExecutionSummaryAccumulator()
    for record in records:
        accumulator.add(record)
    return accumulator.finish()


def _provenance_for_records(
    runfile,
    records,
    *,
    construction,
    execution_records=(),
    execution_summary=None,
    **kwargs,
):
    provenance = cnp2cnp_provenance(
        runfile,
        construction=construction,
        profile_count=len(records),
        **kwargs,
    )
    provenance["input_cache_key"] = distance_input_cache_key(records, provenance)
    provenance["external_execution"] = (
        _execution_summary(execution_records)
        if execution_summary is None
        else deepcopy(execution_summary)
    )
    return provenance


def resolve_distance_worker_count(max_threads, task_count):
    """Resolve an explicit, machine-bounded worker count for distance tasks."""
    if isinstance(task_count, bool) or not isinstance(task_count, Integral):
        raise ValueError("task_count must be a nonnegative integer.")
    task_count = int(task_count)
    if task_count < 0:
        raise ValueError("task_count must be nonnegative.")
    if task_count == 0:
        return 0
    if max_threads is None:
        requested = DEFAULT_DISTANCE_MAX_WORKERS
    else:
        if isinstance(max_threads, bool) or not isinstance(max_threads, Integral):
            raise ValueError("max_threads must be a positive integer.")
        requested = int(max_threads)
        if requested <= 0:
            raise ValueError("max_threads must be a positive integer.")
        if requested > MAX_DISTANCE_MAX_WORKERS:
            raise ValueError(
                f"max_threads may not exceed {MAX_DISTANCE_MAX_WORKERS}."
            )
    return min(requested, os.cpu_count() or 1, task_count)


def bounded_process_map(function, tasks, *, max_workers, task_count):
    """Yield process results in input order with a bounded pending queue."""
    worker_count = resolve_distance_worker_count(max_workers, task_count)
    if worker_count == 0:
        return
    task_iterator = iter(tasks)
    if worker_count == 1:
        for task in task_iterator:
            yield function(task)
        return
    pending = deque()
    pending_limit = worker_count * 2

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for _ in range(pending_limit):
            try:
                pending.append(executor.submit(function, next(task_iterator)))
            except StopIteration:
                break
        try:
            while pending:
                future = pending.popleft()
                yield future.result()
                try:
                    pending.append(executor.submit(function, next(task_iterator)))
                except StopIteration:
                    pass
        except BaseException:
            for future in pending:
                future.cancel()
            raise


@dataclass(frozen=True)
class CtbsRuntimeConfig:
    in_file_name: str
    out_file_name: str
    sim_dm: str
    cnp2cnp_folder: str
    cnp2cnp_file: str
    true_tree_root_id: int
    run_single_test: dict
    cnp2cnp_timeout_seconds: float | None = None
    cnp2cnp_capture_limit_bytes: int | None = None

    @classmethod
    def from_mapping(cls, config):
        return cls(
            in_file_name=config["IN_FILE_NAME"],
            out_file_name=config["OUT_FILE_NAME"],
            sim_dm=config["SIM_DM"],
            cnp2cnp_folder=config["cnp2cnp_FOLDER"],
            cnp2cnp_file=config["cnp2cnp_FILE"],
            true_tree_root_id=config["TRUE_TREE_ROOT_ID"],
            run_single_test=deepcopy(config["RUN_SINGLE_TEST"]),
            cnp2cnp_timeout_seconds=config.get("CNP2CNP_TIMEOUT_SECONDS"),
            cnp2cnp_capture_limit_bytes=config.get("CNP2CNP_CAPTURE_LIMIT_BYTES"),
        )

    def as_legacy_dict(self):
        result = {
            "IN_FILE_NAME": self.in_file_name,
            "OUT_FILE_NAME": self.out_file_name,
            "SIM_DM": self.sim_dm,
            "cnp2cnp_FOLDER": self.cnp2cnp_folder,
            "cnp2cnp_FILE": self.cnp2cnp_file,
            "TRUE_TREE_ROOT_ID": self.true_tree_root_id,
            "RUN_SINGLE_TEST": deepcopy(self.run_single_test),
        }
        if self.cnp2cnp_timeout_seconds is not None:
            result["CNP2CNP_TIMEOUT_SECONDS"] = self.cnp2cnp_timeout_seconds
        if self.cnp2cnp_capture_limit_bytes is not None:
            result["CNP2CNP_CAPTURE_LIMIT_BYTES"] = self.cnp2cnp_capture_limit_bytes
        return result


def load_ctbs_config(config_path=CTBS_CONFIG_PATH):
    with open(config_path, "r") as f:
        loaded_config = json.load(f)

    config = deepcopy(DEFAULT_CTBS_CONFIG)
    config.update(loaded_config)
    return config


def load_ctbs_runtime_config(config_path=CTBS_CONFIG_PATH):
    return CtbsRuntimeConfig.from_mapping(load_ctbs_config(config_path))


def default_ctbs_runtime_config():
    return CtbsRuntimeConfig.from_mapping(DEFAULT_CTBS_CONFIG)


def _coerce_runtime_config(runtime_config=None):
    if runtime_config is None:
        return load_ctbs_runtime_config()
    if isinstance(runtime_config, CtbsRuntimeConfig):
        return runtime_config
    return CtbsRuntimeConfig.from_mapping(runtime_config)


def validate_distance_matrix(ids, matrix):
    return _validate_distance_matrix(ids, matrix)


@dataclass(frozen=True)
class DistanceMatrix:
    ids: list | None = None
    matrix: object | None = None
    path: str | None = None
    provenance: dict | None = None

    def __post_init__(self):
        if self.path is None and self.matrix is None:
            raise ValueError("DistanceMatrix requires either an in-memory matrix or a path.")
        if self.matrix is not None:
            ids, matrix = validate_distance_matrix(self.ids, self.matrix)
            object.__setattr__(self, "ids", ids)
            object.__setattr__(self, "matrix", matrix)
        if self.provenance is not None:
            object.__setattr__(self, "provenance", deepcopy(self.provenance))

    def build_tree_kwargs(self):
        if self.matrix is not None:
            return {
                "dist_matrix_path": None,
                "inids": self.ids,
                "indm": self.matrix,
            }
        return {"dist_matrix_path": self.path}


def unique_cells_by_cell_id(cells):
    unique = {}
    for cell in cells:
        if cell.cell_id not in unique:
            unique[cell.cell_id] = cell
    return list(unique.values())


def _distance_records(cells):
    return [(cell.get_id(), cell.get_cnp()) for cell in cells]


def _trivial_distance_matrix(cells, provenance=None):
    ids = [cell.get_id() for cell in cells]
    return DistanceMatrix(
        ids=ids,
        matrix=np.zeros((len(ids), len(ids)), dtype=float),
        provenance=provenance,
    )


class DistanceProvider:
    def compute(self, cells):
        raise NotImplementedError


@dataclass(frozen=True)
class SuppliedDistanceProvider(DistanceProvider):
    ids: list
    matrix: object

    def compute(self, cells):
        distance_matrix = DistanceMatrix(ids=self.ids, matrix=self.matrix)
        observed_ids = [cell.get_id() for cell in unique_cells_by_cell_id(cells)]
        validate_distance_label_coverage(
            distance_matrix.ids,
            observed_ids,
            allow_extra=True,
        )
        return distance_matrix


@dataclass(frozen=True)
class Cnp2CnpPairwiseDistanceProvider(DistanceProvider):
    runtime_config: CtbsRuntimeConfig
    max_threads: int | None = None

    def compute(self, cells):
        records = _distance_records(cells)
        if len(cells) <= 1:
            provenance = _provenance_for_records(
                self.runtime_config.cnp2cnp_file,
                records,
                construction="trivial_singleton",
            )
            return _trivial_distance_matrix(cells, provenance=provenance)
        ids, matrix, execution_summary = distance_matrix_from_biopsy(
            cells,
            max_threads=self.max_threads,
            runtime_config=self.runtime_config,
            return_execution_summary=True,
        )
        provenance = _provenance_for_records(
            self.runtime_config.cnp2cnp_file,
            records,
            construction="bidirectional_pair_mode",
            execution_summary=execution_summary,
        )
        return DistanceMatrix(ids=ids, matrix=matrix, provenance=provenance)


@dataclass(frozen=True)
class Cnp2CnpFileDistanceProvider(DistanceProvider):
    runtime_config: CtbsRuntimeConfig

    def compute(self, cells):
        if len(cells) <= 1:
            records = _distance_records(cells)
            return _trivial_distance_matrix(
                cells,
                provenance=_provenance_for_records(
                    self.runtime_config.cnp2cnp_file,
                    records,
                    construction="trivial_singleton",
                ),
            )
        return distance_matrix_from_cnp2cnp_matrix_mode(
            cells,
            runtime_config=self.runtime_config,
        )


def _ordered_cells(cells, requested_order=None):
    by_id = {}
    for cell in cells:
        cell_id = cell.get_id()
        if cell_id in by_id:
            raise ValueError(f"Duplicate cnp2cnp input id {cell_id!r}.")
        by_id[cell_id] = cell

    if requested_order is None:
        ordered_ids = sorted(by_id, key=stable_distance_label_key)
    else:
        ordered_ids = list(requested_order)
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError("Explicit cnp2cnp row order contains duplicate ids.")
        if set(ordered_ids) != set(by_id):
            raise ValueError(
                "Explicit cnp2cnp row order must contain every input id exactly once."
            )
    return [by_id[cell_id] for cell_id in ordered_ids]


@dataclass(frozen=True)
class Cnp2CnpOrderedTriangleFastDistanceProvider(DistanceProvider):
    """Explicit one-process, order-conditioned cnp2cnp compatibility mode."""

    runtime_config: CtbsRuntimeConfig
    row_order: tuple | None = None

    def compute(self, cells):
        ordered_cells = _ordered_cells(cells, self.row_order)
        if len(ordered_cells) <= 1:
            row_order = [cell.get_id() for cell in ordered_cells]
            records = _distance_records(ordered_cells)
            return _trivial_distance_matrix(
                ordered_cells,
                provenance=_provenance_for_records(
                    self.runtime_config.cnp2cnp_file,
                    records,
                    construction="trivial_singleton",
                    semantic_mode=CNP2CNP_ORDERED_TRIANGLE_FAST,
                    row_order=row_order,
                ),
            )
        return distance_matrix_from_cnp2cnp_ordered_triangle(
            ordered_cells,
            runtime_config=self.runtime_config,
        )


@dataclass(frozen=True)
class Cnp2CnpDirectedFileDistanceProvider(DistanceProvider):
    """Two-process provider retaining C[u,v] beside its symmetric minimum."""

    runtime_config: CtbsRuntimeConfig

    def compute(self, cells):
        ordered_cells = _ordered_cells(cells)
        if len(ordered_cells) <= 1:
            ids = [cell.get_id() for cell in ordered_cells]
            records = _distance_records(ordered_cells)
            return DirectedDistanceBundle(
                ids,
                np.zeros((len(ids), len(ids)), dtype=float),
                provenance=_provenance_for_records(
                    self.runtime_config.cnp2cnp_file,
                    records,
                    construction="trivial_singleton",
                    retains_directed=True,
                ),
            )
        return directed_distance_bundle_from_cnp2cnp_matrix_mode(
            ordered_cells,
            runtime_config=self.runtime_config,
        )


def default_distance_provider(
    parallel=False,
    runtime_config=None,
    max_threads=None,
    distance_construction=CNP2CNP_DISTANCE_CONSTRUCTION_DEFAULT,
):
    runtime_config = _coerce_runtime_config(runtime_config)
    if distance_construction not in CNP2CNP_DISTANCE_CONSTRUCTIONS:
        available = ", ".join(sorted(CNP2CNP_DISTANCE_CONSTRUCTIONS))
        raise ValueError(
            f"Unknown cnp2cnp distance construction {distance_construction!r}; "
            f"choose one of: {available}."
        )
    if distance_construction == CNP2CNP_DISTANCE_CONSTRUCTION_FAST:
        if parallel:
            raise ValueError(
                "ordered_triangle_fast is a one-process matrix construction; "
                "set parallel=False."
            )
        return Cnp2CnpOrderedTriangleFastDistanceProvider(runtime_config)
    if distance_construction == CNP2CNP_DISTANCE_CONSTRUCTION_DIRECTED:
        if parallel:
            raise ValueError(
                "minimum_with_directed currently uses two matrix processes; "
                "set parallel=False."
            )
        return Cnp2CnpDirectedFileDistanceProvider(runtime_config)
    if parallel:
        return Cnp2CnpPairwiseDistanceProvider(runtime_config, max_threads=max_threads)
    return Cnp2CnpFileDistanceProvider(runtime_config)


class Timer:
    def __init__(self, label, collector=None, verbose=False):
        self.label = label
        self.collector = collector  # collector is a dict for the current run
        self.verbose = verbose

    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter_ns() - self.start
        if self.collector is not None:
            self.collector[self.label] = duration
        if self.verbose:
            print(f"{self.label}: {duration/1e6:.3f} ms")

#  print to file in a format that is compatible with cnp2cnp tool
def to_file(file, cells):
    with open(file, 'w') as f:
        for c in cells:
            f.write(">" + str(c.get_id()) + "\n")
            f.write(c.get_cnp() + "\n")


def _compute_pair(args):
    c, d, i, j, runfile, timeout_seconds, capture_limit_bytes = args
    dist, execution_records = compute_symmetric_cnp2cnp_distance(
        c,
        d,
        runfile=runfile,
        timeout_seconds=timeout_seconds,
        capture_limit_bytes=capture_limit_bytes,
        return_execution_records=True,
    )
    return i, j, dist, execution_records


def distance_matrix_from_biopsy(
    cells,
    max_threads=None,
    runtime_config=None,
    *,
    return_execution_summary=False,
):
    """
    Build a distance matrix for a list of cells using cnp2cnp.
    """
    n = len(cells)
    ids = [c.get_id() for c in cells]
    dist_matrix = np.zeros((n, n), dtype=float)
    ids, dist_matrix = validate_distance_matrix(ids, dist_matrix)
    if n <= 1:
        result = (ids, dist_matrix)
        if return_execution_summary:
            summary = _execution_summary([])
            summary.update({"pair_count": 0, "effective_worker_count": 0})
            return (*result, summary)
        return result
    runtime_config = _coerce_runtime_config(runtime_config)

    pair_count = n * (n - 1) // 2
    worker_count = resolve_distance_worker_count(max_threads, pair_count)
    pairs = (
        (
            cells[i],
            cells[j],
            i,
            j,
            runtime_config.cnp2cnp_file,
            runtime_config.cnp2cnp_timeout_seconds,
            runtime_config.cnp2cnp_capture_limit_bytes,
        )
        for i in range(n)
        for j in range(i + 1, n)
    )
    execution = _ExecutionSummaryAccumulator()
    for i, j, dist, execution_records in bounded_process_map(
        _compute_pair,
        pairs,
        max_workers=worker_count,
        task_count=pair_count,
    ):
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist
        for record in execution_records:
            execution.add(record)

    ids, dist_matrix = validate_distance_matrix(ids, dist_matrix)
    if return_execution_summary:
        summary = execution.finish()
        summary.update(
            {
                "pair_count": pair_count,
                "effective_worker_count": worker_count,
                "pending_task_limit": 1 if worker_count == 1 else worker_count * 2,
            }
        )
        return ids, dist_matrix, summary
    return ids, dist_matrix


def use_cnp2cnp_to_compute_pairwise_distance(
    str_in,
    runfile=None,
    runtime_config=None,
    *,
    return_execution_record=False,
    timeout_seconds=None,
    capture_limit_bytes=None,
):
    if runtime_config is not None or runfile is None:
        resolved_runtime = _coerce_runtime_config(runtime_config)
        if runfile is None:
            runfile = resolved_runtime.cnp2cnp_file
        if timeout_seconds is None:
            timeout_seconds = resolved_runtime.cnp2cnp_timeout_seconds
        if capture_limit_bytes is None:
            capture_limit_bytes = resolved_runtime.cnp2cnp_capture_limit_bytes
    runfile = str(Path(runfile).expanduser().resolve())

    with tempfile.TemporaryDirectory(prefix="ctbf-cnp2cnp-pair-") as tmpdir:
        infile_path = Path(tmpdir) / "pair.fa"
        infile_path.write_text(str_in)
        command = [
            str(sys.executable),
            runfile,
            "-m",
            "dist",
            "-d",
            CNP2CNP_DISTANCE,
            "-i",
            str(infile_path),
        ]
        out, execution_record = _run_checked_cnp2cnp(
            command,
            tmpdir,
            timeout_seconds=timeout_seconds,
            capture_limit_bytes=capture_limit_bytes,
        )
        try:
            distance = parse_cnp2cnp_directional_distance(out.stdout)
        except ValueError as exc:
            invalid_record = _invalid_output_record(execution_record, exc)
            raise Cnp2CnpExecutionError(
                "cnp2cnp returned an invalid directional scalar.",
                invalid_record,
            ) from exc
    if return_execution_record:
        return distance, execution_record
    return distance


def compute_symmetric_cnp2cnp_distance(
    left,
    right,
    runfile=None,
    runtime_config=None,
    *,
    return_execution_records=False,
    timeout_seconds=None,
    capture_limit_bytes=None,
):
    """Compute min(d(left,right), d(right,left)) with two explicit calls."""
    forward_input = (
        f">{left.get_id()}\n{left.get_cnp()}\n"
        f">{right.get_id()}\n{right.get_cnp()}\n"
    )
    reverse_input = (
        f">{right.get_id()}\n{right.get_cnp()}\n"
        f">{left.get_id()}\n{left.get_cnp()}\n"
    )
    limit_kwargs = {}
    if timeout_seconds is not None:
        limit_kwargs["timeout_seconds"] = timeout_seconds
    if capture_limit_bytes is not None:
        limit_kwargs["capture_limit_bytes"] = capture_limit_bytes
    if return_execution_records:
        forward, forward_record = use_cnp2cnp_to_compute_pairwise_distance(
            forward_input,
            runfile=runfile,
            runtime_config=runtime_config,
            return_execution_record=True,
            **limit_kwargs,
        )
        reverse, reverse_record = use_cnp2cnp_to_compute_pairwise_distance(
            reverse_input,
            runfile=runfile,
            runtime_config=runtime_config,
            return_execution_record=True,
            **limit_kwargs,
        )
        return minimum_bidirectional_distance(forward, reverse), [
            forward_record,
            reverse_record,
        ]

    forward = use_cnp2cnp_to_compute_pairwise_distance(
        forward_input,
        runfile=runfile,
        runtime_config=runtime_config,
        **limit_kwargs,
    )
    reverse = use_cnp2cnp_to_compute_pairwise_distance(
        reverse_input,
        runfile=runfile,
        runtime_config=runtime_config,
        **limit_kwargs,
    )
    return minimum_bidirectional_distance(forward, reverse)


def _write_cnp2cnp_records(path, records):
    with open(path, "w") as destination:
        for cell_id, cnp in records:
            destination.write(f">{cell_id}\n{cnp}\n")


def _write_labeled_distance_matrix(path, ids, matrix):
    ids, matrix = validate_distance_matrix(ids, matrix)
    with open(path, "w") as destination:
        destination.write(f"{len(ids)}\n")
        for cell_id, row in zip(ids, matrix):
            values = " ".join(str(value) for value in row)
            destination.write(f"{str(cell_id).ljust(10)} {values}\n")


def _run_cnp2cnp_ordered_matrix(
    records,
    runfile,
    *,
    timeout_seconds=None,
    capture_limit_bytes=None,
):
    if not records:
        raise ValueError("cnp2cnp matrix construction requires at least one profile.")
    raw_record_ids = [record[0] for record in records]
    record_ids = [parse_distance_label(value) for value in raw_record_ids]
    for raw_id, serialized_id in zip(raw_record_ids, record_ids):
        if stable_distance_label_key(raw_id) != stable_distance_label_key(
            serialized_id
        ):
            raise ValueError(
                "cnp2cnp input ids must round-trip through their text label "
                f"without changing type/value; {raw_id!r} becomes "
                f"{serialized_id!r}."
            )
    record_ids, _ = validate_distance_matrix(
        record_ids,
        np.zeros((len(records), len(records)), dtype=float),
    )
    if len(records) == 1:
        return record_ids, np.zeros((1, 1), dtype=float), None

    runfile = str(Path(runfile).expanduser().resolve())
    with tempfile.TemporaryDirectory(prefix="ctbf-cnp2cnp-matrix-") as tmpdir:
        ordered_input = Path(tmpdir) / "ordered.fa"
        ordered_output = Path(tmpdir) / "ordered.phy"
        _write_cnp2cnp_records(ordered_input, records)
        command = [
            str(sys.executable),
            runfile,
            "-m",
            "matrix",
            "-d",
            CNP2CNP_DISTANCE,
            "-i",
            str(ordered_input),
            "-o",
            str(ordered_output),
        ]
        _completed, execution_record = _run_checked_cnp2cnp(
            command,
            tmpdir,
            output_path=ordered_output,
            timeout_seconds=timeout_seconds,
            capture_limit_bytes=capture_limit_bytes,
        )
        try:
            ids, matrix = parse_labeled_distance_matrix(ordered_output)
        except ValueError as exc:
            invalid_record = _invalid_output_record(execution_record, exc)
            raise Cnp2CnpExecutionError(
                "cnp2cnp returned an invalid labeled matrix.",
                invalid_record,
            ) from exc
    if ids != record_ids:
        message = (
            "cnp2cnp output labels/order do not match its ordered input "
            f"(input={record_ids!r}, output={ids!r})."
        )
        raise Cnp2CnpExecutionError(
            message,
            _invalid_output_record(execution_record, message),
        )
    return ids, matrix, execution_record


def _cnp2cnp_matrix_from_records(
    records,
    runfile,
    *,
    timeout_seconds=None,
    capture_limit_bytes=None,
):
    if not records:
        raise ValueError("cnp2cnp matrix construction requires at least one profile.")

    if len(records) <= 1:
        forward_ids, forward_matrix, _record = _run_cnp2cnp_ordered_matrix(
            records,
            runfile,
            timeout_seconds=timeout_seconds,
            capture_limit_bytes=capture_limit_bytes,
        )
        return DistanceMatrix(
            ids=forward_ids,
            matrix=forward_matrix,
            provenance=_provenance_for_records(
                runfile,
                records,
                construction="trivial_singleton",
            ),
        )

    forward_ids, forward_matrix, forward_record = _run_cnp2cnp_ordered_matrix(
        records,
        runfile,
        timeout_seconds=timeout_seconds,
        capture_limit_bytes=capture_limit_bytes,
    )
    reverse_ids, reverse_matrix, reverse_record = _run_cnp2cnp_ordered_matrix(
        list(reversed(records)),
        runfile,
        timeout_seconds=timeout_seconds,
        capture_limit_bytes=capture_limit_bytes,
    )

    ids, matrix = combine_ordered_cnp2cnp_matrices(
        forward_ids,
        forward_matrix,
        reverse_ids,
        reverse_matrix,
    )
    return DistanceMatrix(
        ids=ids,
        matrix=matrix,
        provenance=_provenance_for_records(
            runfile,
            records,
            construction="opposite_order_matrix_mode",
            execution_records=[forward_record, reverse_record],
        ),
    )


def _cnp2cnp_ordered_triangle_from_records(
    records,
    runfile,
    *,
    timeout_seconds=None,
    capture_limit_bytes=None,
):
    if not records:
        raise ValueError("cnp2cnp matrix construction requires at least one profile.")
    ids, matrix, execution_record = _run_cnp2cnp_ordered_matrix(
        records,
        runfile,
        timeout_seconds=timeout_seconds,
        capture_limit_bytes=capture_limit_bytes,
    )
    row_order = list(ids)
    construction = (
        "trivial_singleton" if len(records) <= 1 else "ordered_triangle_matrix_mode"
    )
    return DistanceMatrix(
        ids=ids,
        matrix=matrix,
        provenance=_provenance_for_records(
            runfile,
            records,
            construction=construction,
            semantic_mode=CNP2CNP_ORDERED_TRIANGLE_FAST,
            row_order=row_order,
            execution_records=(
                [] if execution_record is None else [execution_record]
            ),
        ),
    )


def _cnp2cnp_directed_bundle_from_records(
    records,
    runfile,
    *,
    timeout_seconds=None,
    capture_limit_bytes=None,
):
    if not records:
        raise ValueError("cnp2cnp matrix construction requires at least one profile.")
    if len(records) <= 1:
        forward_ids, forward_matrix, _record = _run_cnp2cnp_ordered_matrix(
            records,
            runfile,
            timeout_seconds=timeout_seconds,
            capture_limit_bytes=capture_limit_bytes,
        )
        return DirectedDistanceBundle(
            forward_ids,
            forward_matrix,
            provenance=_provenance_for_records(
                runfile,
                records,
                construction="trivial_singleton",
                retains_directed=True,
            ),
        )

    forward_ids, forward_matrix, forward_record = _run_cnp2cnp_ordered_matrix(
        records,
        runfile,
        timeout_seconds=timeout_seconds,
        capture_limit_bytes=capture_limit_bytes,
    )
    reverse_ids, reverse_matrix, reverse_record = _run_cnp2cnp_ordered_matrix(
        list(reversed(records)),
        runfile,
        timeout_seconds=timeout_seconds,
        capture_limit_bytes=capture_limit_bytes,
    )
    provenance = _provenance_for_records(
        runfile,
        records,
        construction="opposite_order_matrix_mode_directed_bundle",
        retains_directed=True,
        execution_records=[forward_record, reverse_record],
    )
    return directed_bundle_from_ordered_cnp2cnp_matrices(
        forward_ids,
        forward_matrix,
        reverse_ids,
        reverse_matrix,
        provenance=provenance,
    )


def distance_matrix_from_cnp2cnp_matrix_mode(cells, runtime_config=None):
    """Run cnp2cnp matrix mode in both global orders and take pairwise minima."""
    runtime_config = _coerce_runtime_config(runtime_config)
    records = [(cell.get_id(), cell.get_cnp()) for cell in cells]
    return _cnp2cnp_matrix_from_records(
        records,
        runtime_config.cnp2cnp_file,
        timeout_seconds=runtime_config.cnp2cnp_timeout_seconds,
        capture_limit_bytes=runtime_config.cnp2cnp_capture_limit_bytes,
    )


def distance_matrix_from_cnp2cnp_ordered_triangle(cells, runtime_config=None):
    """Run one recorded cnp2cnp triangle and mirror its ordered values."""
    runtime_config = _coerce_runtime_config(runtime_config)
    records = [(cell.get_id(), cell.get_cnp()) for cell in cells]
    return _cnp2cnp_ordered_triangle_from_records(
        records,
        runtime_config.cnp2cnp_file,
        timeout_seconds=runtime_config.cnp2cnp_timeout_seconds,
        capture_limit_bytes=runtime_config.cnp2cnp_capture_limit_bytes,
    )


def directed_distance_bundle_from_cnp2cnp_matrix_mode(cells, runtime_config=None):
    """Run opposite cnp2cnp triangles and retain the complete ordered matrix."""
    runtime_config = _coerce_runtime_config(runtime_config)
    records = [(cell.get_id(), cell.get_cnp()) for cell in cells]
    return _cnp2cnp_directed_bundle_from_records(
        records,
        runtime_config.cnp2cnp_file,
        timeout_seconds=runtime_config.cnp2cnp_timeout_seconds,
        capture_limit_bytes=runtime_config.cnp2cnp_capture_limit_bytes,
    )


def use_cnp2cnp_to_compute_dist_matrix(sample=None, folder=None, runfile=None,
                                       output=None, runtime_config=None):
    """
    Execute cnp2cnp. Input CNPs of cells and obtain evolutionary distance matrix of given cells.

    Parameters
    ----------
    sample : string
        name of the file that contains information about the cell sample; every cell is described in two lines:
        first describing id of the cell, begins with ">" (in example ">cell_0")
        second contains CNP profile as a list of values in a form "value1,value2,...,valueN",
        where N is the length of CNP
    folder : string
        name of the folder that contains cnp2cnp tool described in
        https://doi.org/10.1186/s12864-020-6611-3
        to manually set path where input file will be copied; output file generated before copied back;
    runfile : string
        name of the file to execute (cnp2cnp.py)
        to manually set path to the file
    output : string
        name of the output file that will be generated in location set in argument 'folder',
        and copied to current location.
        first line of the file contains the number N of cells
        the following N lines represent distance matrix, each line consist of the id of the cell
        and N values that are evolutionary distances to the corresponding cells;
        evolutionary distance is the minimal number of events to transform on cell (CNP) to another cell (CNP)

    Returns
    -------
        In the current directory generates a file which name is set in argument 'output'.
        The file will contain distance matrix of cells, which CNPs are given in file described by argument 'input'.
        The file is generated by an external tool cnp2cnp.
    """
    runtime_config = _coerce_runtime_config(runtime_config)
    sample = runtime_config.in_file_name if sample is None else sample
    runfile = runtime_config.cnp2cnp_file if runfile is None else runfile
    output = runtime_config.out_file_name if output is None else output
    _ = folder  # Retained only for call compatibility with the historical API.

    records = []
    with open(sample) as source:
        current_id = None
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:]
            elif current_id is None:
                raise ValueError("cnp2cnp input contains a profile before its id.")
            else:
                records.append((current_id, line))
                current_id = None
    if current_id is not None:
        raise ValueError("cnp2cnp input ends before the final profile line.")

    distance_matrix = _cnp2cnp_matrix_from_records(records, runfile)
    _write_labeled_distance_matrix(output, distance_matrix.ids, distance_matrix.matrix)
    return distance_matrix.provenance


def get_cell_manualy(cell_list, value):
    for cell in cell_list:
        if cell.get_id() == value:
            return cell
    return None

def show_cells(cell_list):
    for cell_l in cell_list:
        print("Biopsy: ", [cell.cell_id for cell in cell_l])

def _actual_root(tree):
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root, found {len(roots)}")
    return roots[0]

def _run_simulation(config, bedfile, seed, simulator_with_loaded_tree, time_collector):
    if simulator_with_loaded_tree:
        return simulator_with_loaded_tree

    if bedfile is not None:
        sim = CancerCellEvolutionSimulator(config, bedfile, seed=seed)
    else:
        sim = CancerCellEvolutionSimulator(config, seed=seed)

    if time_collector is not None:
        with Timer("Core simulation: ", time_collector):
            sim.run_simulation()
    else:
        sim.run_simulation()

    print("Simulation finished. Generated cell evolution tree total nodes:", len(sim.tree.nodes()))
    return sim


def _write_observed_truth_distance_matrix(sim, observations, output_path):
    """Write a truth diagnostic only when state-to-occurrence mapping is unique."""
    node_ids = []
    labels = []
    node_by_label = {}
    for observation in observations:
        cell_id = observation.cell_id
        if cell_id in node_by_label:
            raise ValueError(
                "compare_dm is undefined when the same canonical CNP label is "
                "sampled at multiple occurrences; truth event distance is an "
                f"occurrence-level quantity (repeated label {cell_id!r})."
            )
        node_by_label[cell_id] = observation.node_id
        node_ids.append(observation.node_id)
        labels.append(cell_id)
    return sim.to_distance_matrix(
        output_path,
        node_list=node_ids,
        labels=labels,
    )

def _perform_biopsies(sim, biopsy_generations, biopsy_size, biopsy_size_scalable, seed,
                      compare_dm, runtime_config):
    cell_lists, all_in_one_sample = [], [[]]

    for b_gen in biopsy_generations:
        biopsy = sim.perform_biopsy(
            biopsy_size=biopsy_size,
            biopsy_size_scalable=biopsy_size_scalable,
            generation=b_gen,
            seed=seed,
        )
        if biopsy: # we assume biopsy has at least one cell
            cell_lists.append(biopsy)
            all_in_one_sample[0] += biopsy
        else:
            print(f"Biopsy sample from generation {b_gen} has no cells. Skipping.")

    if compare_dm:
        _write_observed_truth_distance_matrix(
            sim,
            all_in_one_sample[0],
            runtime_config.sim_dm,
        )

    print("Number of biopsy cells:", len(all_in_one_sample[0]))
    return cell_lists, all_in_one_sample

def _handle_small_biopsy(time_collector, min_total_cells=MIN_TOTAL_BIOPSY_CELLS):
    print(f"Total number of cells in biopsy less than {min_total_cells}.")
    if time_collector is not None:
        for key in ["Computing cnp2cnp distance matrix: ", "Clear CNPs: ", "GRF our: ", "GRF NJ: "]:
            time_collector[key] = 0

def _compute_distance_matrix(
    all_in_one_sample,
    parallel,
    time_collector,
    runtime_config,
    distance_provider=None,
    distance_construction=None,
):
    # for parallel case single distances are being computed
    # for not parallel we write biopsy to cnp2cnp format file, and proces that
    unique_cells = unique_cells_by_cell_id(all_in_one_sample[0])
    if distance_provider is None:
        distance_provider = default_distance_provider(
            parallel=parallel,
            runtime_config=runtime_config,
            distance_construction=(
                distance_construction or CNP2CNP_DISTANCE_CONSTRUCTION_DEFAULT
            ),
        )
    elif distance_construction is not None:
        raise ValueError(
            "Pass either distance_provider or distance_construction, not both."
        )

    if time_collector is not None:
        with Timer("Computing cnp2cnp distance matrix: ", time_collector):
            distance_matrix = distance_provider.compute(unique_cells)
    else:
        distance_matrix = distance_provider.compute(unique_cells)
    return distance_matrix


def _symmetric_distance_view(distance_input):
    if isinstance(distance_input, DirectedDistanceBundle):
        return list(distance_input.ids), np.array(
            distance_input.minimum_matrix,
            copy=True,
        )
    return distance_input.ids, distance_input.matrix

def _reconstruct_and_evaluate(sim, seed, cell_lists, all_in_one_sample, r_dist, visualize,
                              clear_cnps, parallel, write_newick, reconstruction_algorithm,
                              biopsy_guided_config, inid, indm, time_collector,
                              runtime_config=None, distance_matrix=None):
    runtime_config = _coerce_runtime_config(runtime_config)
    if distance_matrix is None:
        if parallel:
            distance_matrix = DistanceMatrix(ids=inid, matrix=indm)
        else:
            distance_matrix = DistanceMatrix(path=runtime_config.out_file_name)
    cl, osl = deepcopy(cell_lists), deepcopy(all_in_one_sample)
    ordered_occurrence_algorithm = uses_ordered_occurrence_input(reconstruction_algorithm)
    if ordered_occurrence_algorithm and clear_cnps:
        raise ValueError(
            "Temporal CNP arborescence requires biopsy genomes for plausibility; "
            "clear_cnps is not supported."
        )
    if ordered_occurrence_algorithm and biopsy_guided_config is not None:
        raise ValueError(
            "Temporal CNP arborescence cannot be combined with a biopsy-guided preset."
        )
    show_cells(cell_lists)

    # Optional visualization
    if visualize:
        sim.plot_tree(biopsy_lists=cell_lists, highlight_nodes=all_in_one_sample[0],
                      legend_y_offset=-170, output_file="simulated_tree")

        # true tree
        only_nodes = [c.cell_id for c in all_in_one_sample[0]]
        sim.plot_tree(biopsy_lists=cell_lists, legend_y_offset=-170,
                      highlight_nodes=all_in_one_sample[0],extended=False,
                      only_nodes=only_nodes,node_numbers=True,output_file="true_tree")

    # # Options for True tree pic
    # only_nodes = [0, 1, 3, 5, 4, 7, 13, 12, 19]
    # if visualize:
    #     sim.plot_tree(biopsy_lists=cell_lists, legend_y_offset=-170,
    #                   highlight_nodes=all_in_one_sample[0], output_file="simulated_tree")
    #     sim.plot_tree(biopsy_lists=cell_lists,legend_y_offset=-170,
    #                   highlight_nodes=all_in_one_sample[0],extended=False,
    #                   only_nodes=only_nodes,node_numbers=True,output_file="true_tree")
    # # if visualize:
    # #     sim.plot_tree(biopsy_lists=cell_lists,legend_y_offset=-170,
    # #                   highlight_nodes=all_in_one_sample[0])

    # # Clear CNPs if requested
    # if clear_cnps:
    #     with Timer("Clear CNPs: ", time_collector) if time_collector else contextlib.nullcontext():
    #         true_tree_simplified = sim.tree_without_CNPs()
    #         for lst in (cl + [osl[0]]):
    #             for cell in lst:
    #                 cell.genome = np.array([], dtype=int)
    # else:
    #     true_tree_simplified = sim.tree

    if clear_cnps:    # clear CNPs
        if time_collector is not None:
            with Timer("Clear CNPs: ", time_collector):
                true_tree_simplified = sim.tree_without_CNPs()  # clears simulated tree
                for cell_list in cl:                            # clears biopsy
                    for cell in cell_list:
                        cell.genome = np.array([], dtype=int)
                for cell in osl[0]:
                    cell.genome = np.array([], dtype=int)
    else:
        true_tree_simplified = sim.tree

    # --- unified build config ---
    build_kwargs = {"r": r_dist}
    build_kwargs.update({"seed": seed})
    build_kwargs.update(distance_matrix.build_tree_kwargs())
    if reconstruction_algorithm:
        build_kwargs["neighbor_joining"] = reconstruction_algorithm

    # --- build trees ---
    if ordered_occurrence_algorithm:
        order_ablation = getattr(
            reconstruction_algorithm,
            "ctbf_order_ablation",
            None,
        ) or reconstruction_algorithm
        ablation_build_kwargs = build_kwargs.copy()
        ablation_build_kwargs["neighbor_joining"] = order_ablation
        njtree, nj_info, _returned_root_nj = build_evolution_tree(
            cl,
            **ablation_build_kwargs,
        )
        tree, rt_info, _returned_root_rt = build_evolution_tree(cl, **build_kwargs)
    else:
        njtree, nj_info, _returned_root_nj = build_evolution_tree(
            osl,
            only_nj=True,
            **build_kwargs,
        )
        rec_build_kwargs = build_kwargs.copy()
        if biopsy_guided_config is not None:
            rec_build_kwargs["biopsy_guided_config"] = biopsy_guided_config
        tree, rt_info, _returned_root_rt = build_evolution_tree(cl, **rec_build_kwargs)
    actual_root_nj = _actual_root(njtree)
    actual_root_rt = _actual_root(tree)

    if write_newick:
        print("Newick simulated:", to_newick(sim.tree))
        print("Reconstructed:", to_newick(tree))
        print("NJ tree:", to_newick(njtree))

    # Visualization (optional)
    if visualize:
        visualize_tree_plotly(tree, rt_info, output_file="reconstructed.html")
        visualize_tree_plotly(njtree, nj_info, output_file="nj.html")

    # if visualize:
    #     lno = {2:[8,6], 1:[20,21,22,25,16,30], 0:[50,32,54,34,56,57,21,38,65,43,71,48]}
    #     visualize_tree_plotly(tree, rt_node_info_for_plots, level_node_ordering=lno, output_file="reconstructed.html")
    #     lno1 = {0:[50,32,54,34,20,56,57,21,22,38,8,65,25,43,16,6,71,30,48]}
    #     visualize_tree_plotly(njtree, nj_node_info_for_plots, level_node_ordering=lno1, output_file="nj.html")


    # --- Evaluate GRF distances ---
    if time_collector is not None:
        with Timer("GRF our: ", time_collector):
            ret1 = grf_tree(true_tree_simplified, runtime_config.true_tree_root_id, tree, actual_root_rt)
        with Timer("GRF NJ: ", time_collector):
            ret2 = grf_tree(true_tree_simplified, runtime_config.true_tree_root_id, njtree, actual_root_nj)
    else:
        ret1 = grf_tree(true_tree_simplified, runtime_config.true_tree_root_id, tree, actual_root_rt)
        ret2 = grf_tree(true_tree_simplified, runtime_config.true_tree_root_id, njtree, actual_root_nj)

    print("GRF - reconstructed:", ret1)
    print("GRF - NJ:", ret2)

    return true_tree_simplified, tree, njtree


def run_single_test(config=DEFAULT_SIMULATOR_CONFIG_PATH, bedfile=None, seed=777,
                    biopsy_size=2, biopsy_size_scalable=None, biopsy_generations=(3, 5), r_dist=4,
                    visualize=False, time_collector=None, clear_cnps=False, compare_dm=False,
                    write_newick=False, simulator_with_loaded_tree=None, parallel=False,
                    reconstruction_algorithm=None, biopsy_guided_strategy=None,
                    biopsy_guided_config=None, runtime_config=None, distance_provider=None,
                    distance_construction=None):
    """
    Runs one test that consists of simulation, biopsy, tree reconstruction and tree evaluation.

    Parameters
    ----------
    config  - input configuration file for global parameters for the simulator
    bedfile - optional additional configuration file for positional parameters for the simulator
    seed    - seed for random number generator
    biopsy_size             - size of biopsy (number of cells sampled from given level)
    biopsy_size_scalable    - size of biopsy (percentage of cells sampled from given level)
    biopsy_generatons       - list of levels of the tree (generations numbers) used for biopsy
    r_dist  - the proximity radius for tree reconstruction
    visualize       - whether to visualize the simulation results
    time_collector  - whether to time the simulation runs
    clear_cnps  - whether to clear cnps (potential optimization - makes the simulation tree light)
    compare_dm      - whether to output distance matrix of simulated tree cells
    write_newick       - whether to output simulated tree and reconstructed tree in newick format
    simlulator_with_loaded_tree - for testing and repeatability, simlulator with loaded tree
    distance_provider - optional explicit provider object
    distance_construction - optional provider mode: minimum_bidirectional
        (default), ordered_triangle_fast, or minimum_with_directed

    Returns
    -------
    The similarity values between simulated tree and reconstructed tree,
    and between simulated tree and NJ-reconstructed tree.

    """
    runtime_config = _coerce_runtime_config(runtime_config)

    if biopsy_guided_config is None:
        biopsy_guided_config = resolve_biopsy_guided_config(biopsy_guided_strategy)

    # 1. Simulation phase
    sim = _run_simulation(config, bedfile, seed, simulator_with_loaded_tree, time_collector)

    # 2. Biopsy phase
    cell_lists, all_in_one_sample = _perform_biopsies(sim, biopsy_generations, biopsy_size,
                                                      biopsy_size_scalable, seed, compare_dm,
                                                      runtime_config)

    if len(all_in_one_sample[0]) < MIN_TOTAL_BIOPSY_CELLS:
        _handle_small_biopsy(time_collector)
        return

    # 3. Distance matrix computation
    distance_matrix = _compute_distance_matrix(
        all_in_one_sample,
        parallel,
        time_collector,
        runtime_config,
        distance_provider=distance_provider,
        distance_construction=distance_construction,
    )
    distance_ids, symmetric_distance_matrix = _symmetric_distance_view(distance_matrix)

    # 4. Tree reconstruction and evaluation
    return _reconstruct_and_evaluate(
        sim,
        seed,
        cell_lists,
        all_in_one_sample,
        r_dist,
        visualize,
        clear_cnps,
        parallel,
        write_newick,
        reconstruction_algorithm,
        biopsy_guided_config,
        distance_ids,
        symmetric_distance_matrix,
        time_collector,
        runtime_config,
        distance_matrix,
    )


def run_single_test_timed(seed, both=True, **kwargs):
    """
    Wrapper for run_single_test to measure the time for the cases with clear_cnps optimization on and off.

    Parameters
    ----------
    seed        passed to run_single_test
    both        if true executes run_single_test two times with clear_cnps optimization on and off;
                if false executes run_single_test without clear_cnps optimization;
    kwargs      parameters passed to run_single_test

    Returns
    -------
    Dictionaries with times of executions of parts of test (computing distance matrix, GRF distances ...)
    """
    run_timings_no_opt = {}
    with Timer("Total", run_timings_no_opt):
        run_single_test(seed=seed, time_collector=run_timings_no_opt, clear_cnps=False, **kwargs)

    run_timings_with_opt = {}
    if both:
        with Timer("Total", run_timings_with_opt):
            run_single_test(seed=seed, time_collector=run_timings_with_opt, clear_cnps=True, **kwargs)

    return run_timings_no_opt, run_timings_with_opt


def check_clearcnp_optimizaton(how_many=100, both=True, seeds=None, **kwargs):
    """
    Runner

    Parameters
    ----------
    how_many    number of tests to run
    both        if true runs pair of test with clear_cnps optimization on and off;
                otherwise runs one test without optimization
    seeds       seeds for executions of run_single_test; if not given, randomly selected here
    kwargs      parameters passed to run_single_test

    Returns
    -------
    Prints summary of the tests.

    """
    if not seeds:
        seeds = [random.randint(0, 1000) for _ in range(how_many)]
        seeds = [696]

    all_runs_no_opt = []
    all_runs_with_opt = []

    for s in seeds:
        print(f"\nTesting seed: {s}")
        run_no_opt, run_with_opt = run_single_test_timed(seed=s, biopsy_size_scalable=0.5, both=both,
                                                         biopsy_generatons=[4, 6, 8], r_dist=4, **kwargs)
        all_runs_no_opt.append(run_no_opt)
        if both:
            all_runs_with_opt.append(run_with_opt)

    def average_timings(runs):
        all_keys = {k for run in runs for k in run.keys()}
        avg_dict = {}
        for key in sorted(all_keys):
            avg_dict[key] = sum(run.get(key, 0) for run in runs) / len(runs) / 1e6
        without_cnp_avg = sum(
            run["Total"] - run.get("Computing cnp2cnp distance matrix: ", 0) for run in runs
        ) / len(runs) / 1e6
        return avg_dict, without_cnp_avg

    avg_no_opt_dict, avg_no_opt_total = average_timings(all_runs_no_opt)
    if both:
        avg_with_opt_dict, avg_with_opt_total = average_timings(all_runs_with_opt)

    print("\n--- Average durations WITHOUT optimization (ms) ---")
    for k, v in avg_no_opt_dict.items():
        print(f"{k:<35}: {v:.3f}")
    print(f"Total without cnp call{' ' * 10}: {avg_no_opt_total:.3f}")

    if both:
        print("\n--- Average durations WITH optimization (ms) ---")
        for k, v in avg_with_opt_dict.items():
            print(f"{k:<35}: {v:.3f}")
        print(f"Total without cnp call{' ' * 10}: {avg_with_opt_total:.3f}")


if __name__ == "__main__":
    # seeds = [56, 777, 727, 7, 77, 22, 32]
    # for s in seeds:
    #     run_single_test(config="config_telomeric.json", bedfile=None, seed=s,
    #                     biopsy_size=0.5, biopsy_generatons=[5, 7, 9], r_dist=4, visualize=False)

    # r_list = [20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    # for r in r_list:
    #     print("Running simulation for r=", r)
    #     run_single_test(config="config_telomeric.json", bedfile=None, seed=22,
    #                 biopsy_size=0.5, biopsy_generatons=[5, 7, 9], r_dist=r, visualize=False)

    # timing_data = defaultdict(list)
    # seeds = [56, 777, 7, 77, 22, 32, 727, 0, 100, 1000]
    # for s in seeds:
    #     with Timer("Total", timing_data):
    #         run_single_test(config="config_telomeric.json", bedfile=None, seed=s, biopsy_size=2,
    #                         biopsy_generatons=[4, 7], r_dist=4, visualize=False,
    #                         time_collector=timing_data, clear_cnps=True)
    #
    # print("\nAverage durations (ms):")
    # for key, times in timing_data.items():
    #     avg_ms = sum(times) / len(times) / 1e6
    #     print(f"{key:<15}: {avg_ms:.3f} ms")
    # run_single_test(config="config_for_pic.json", bedfile="pic.csv", seed=727,
    #                 biopsy_size_scalable=0.5, biopsy_generatons=[4, 7], r_dist=4,
    #                 visualize=True)

    # run_single_test(config="config_for_pic.json", bedfile="pic.csv", seed=727,
    #                 biopsy_size_scalable=0.5, biopsy_generatons=[4, 6, 8], r_dist=4,
    #                 visualize=True)

    # run_single_test(seed=727, config="test/data/config_for_pic.json", bedfile="test/data/pic.csv",
    #                       biopsy_size_scalable=0.5, biopsy_generatons=[4, 6, 8], r_dist=4, visualize=True)

    # check_clearcnp_optimizaton(how_many=1, seeds=[773], config="test/data/config_for_pic.json",
    #                            bedfile="test/data/pic.csv", parallel=True, both=False)

    # run_single_test(seed=773, config="test/data/config_for_pic.json",
    #                            bedfile="test/data/pic.csv") #, parallel=True)

    # run_single_test(seed=773, config="test/data/config_for_pic.json", bedfile="test/data/pic.csv",
    #                 biopsy_size_scalable=0.5, biopsy_generations=[4, 6, 8], r_dist=4, write_newick=True,
    #                 reconstruction_algorithm=neighbor_joining_full)

    # seed 35 !!!
    # seed 632
    runtime_config = load_ctbs_runtime_config()
    run_config = runtime_config.run_single_test.copy()
    run_config["reconstruction_algorithm"] = resolve_reconstruction_algorithm(
        run_config.get("reconstruction_algorithm")
    )
    run_config["biopsy_guided_config"] = resolve_biopsy_guided_config(
        run_config.pop("biopsy_guided_strategy", None)
    )
    a, b, c = run_single_test(**run_config, runtime_config=runtime_config)

    biopsy_nodes_ids = get_biopsy_nodes_ids(b, c)

    out = evaluate_4(a, b, restrict_labels=biopsy_nodes_ids, print_debug=True)
    print(out)

    out = evaluate_4(a, c, restrict_labels=biopsy_nodes_ids, print_debug=True)
    print(out)
    # print(c.edges)
    # print(len(c.nodes))
    # l = [(named_label(c, x), named_label(c, y)) for x, y in c.edges]
    # print(l)
    # roots = [n for n, indeg in c.in_degree() if indeg == 0]
    # if len(roots) != 1:
    #     raise ValueError(f"Tree must have exactly one root (found {len(roots)})")
    # root = roots[0]
    # vizualize_nx_tree(c)

    # run_single_test(seed=773, config="test/data/config_for_pic.json", bedfile="test/data/pic.csv",
    #                 biopsy_size_scalable=0.5, biopsy_generatons=[3, 5, 7, 9], r_dist=4)

    # check_clearcnp_optimizaton(how_many=10, config="test/data/config100.json", bedfile=None)
    # check_clearcnp_optimizaton(how_many=1, config="test/data/config_for_pic.json",
    #                            bedfile=None, both=False, parallel=True)
