"""Run the external MEDICC2 topology-only arm on the CTBF v5 development bank.

MEDICC2 is deliberately kept outside the in-process reconstruction registry.
Each bank case is exported from the observable reconstruction payload, run in
one fresh worker plus one fresh MEDICC2 subprocess, projected to a rooted
partial occurrence tree, and only then evaluated against stored truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import networkx as nx

try:
    import psutil
except ImportError:  # pragma: no cover - exercised by the explicit gate check
    psutil = None

from algorithm_evaluation.medicc2_reconstruction_preflight import (
    ARTIFICIAL_NORMAL_TAXON,
    INPUT_ENCODING_VERSION,
    PINNED_MEDICC2_VERSION,
    PROJECTION_VERSION,
    export_observation_occurrences,
    inspect_source_checkout,
    project_phyloxml_to_partial_tree,
    read_pairwise_distances,
    run_fixture_preflight,
    topology_only_command,
)
from algorithm_evaluation.paper_pipeline_contract import file_sha256, read_json
from algorithm_evaluation.paper_pipeline_runner import (
    measured_stage,
    serialize_tree,
    validate_reconstruction_input,
)
from algorithm_evaluation.process_isolation import (
    CASE_ARM_WORKER_UNIT,
    FreshProcessTimeoutError,
    FreshSpawnPerTaskExecutor,
    fresh_process_contract,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    BANK_MANIFEST_NAME,
    CASE_METADATA_SCHEMA_VERSION,
    CONTEXT_FAMILY,
    DEFAULT_BLOCK_COUNT,
    RUN_SCHEMA_VERSION,
    canonical_json_digest,
    deserialize_truth,
    ensure_new_output_root,
    load_bank_manifest,
    observed_labels,
    tree_summary,
    truth_prefix,
    write_json,
)
from evaluation_contract import evaluate_tree_pair_result, validate_evaluation_result


MEDICC2_DEVELOPMENT_ARM_ID = "medicc2_total_cn_topology_partial"
MEDICC2_ALGORITHM_NAME = "external_medicc2_total_cn_topology_only"
MEDICC2_RESOURCE_GATE_SCHEMA_VERSION = "ctbf-v5-medicc2-resource-gate-v3"
MEDICC2_RUNTIME_SCHEMA_VERSION = "ctbf-v5-medicc2-runtime-v1"
MEDICC2_EXECUTION_SCHEMA_VERSION = "ctbf-v5-medicc2-execution-v1"
MEDICC2_RECONSTRUCTION_SCHEMA_VERSION = "ctbf-v5-medicc2-reconstruction-v1"
MEDICC2_RESOURCE_GATE_NAME = "resource_gate.json"
RESULT_NAME = "result.json"

DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS = 300
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 300
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576
OUTER_WORKER_GRACE_SECONDS = 120
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.01
TAIL_LIMIT_BYTES = 4096

MEDICC2_ARM_SPEC = {
    "arm_id": MEDICC2_DEVELOPMENT_ARM_ID,
    "algorithm_name": MEDICC2_ALGORITHM_NAME,
    "family": CONTEXT_FAMILY,
    "problem": "partial",
    "input_mode": "pooled",
    "only_nj": True,
    "radius": None,
    "primary_metric": "grf",
    "complementary_metrics": [],
    "role": "external_contextual_baseline",
    "external_distance": "medicc2_symmetric_minimum_event_distance",
    "ancestor_policy": "topology_only_internal_nodes_unlabeled",
    "rooting_policy": PROJECTION_VERSION,
}

EXPECTED_OUTPUT_SUFFIXES = (
    "branch_lengths.tsv",
    "final_tree.new",
    "final_tree.xml",
    "pairwise_distances.tsv",
    "summary.tsv",
)


class Medicc2ExecutionError(RuntimeError):
    """A typed MEDICC2 process or output-boundary failure."""

    def __init__(self, message: str, record: Mapping[str, Any]):
        super().__init__(message)
        self.record = dict(record)


def _validate_positive_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")


def _bounded_tail(path: Path, maximum_bytes: int = TAIL_LIMIT_BYTES) -> dict[str, Any]:
    size = path.stat().st_size if path.is_file() else 0
    if not size:
        return {"byte_count": 0, "truncated": False, "tail": ""}
    with path.open("rb") as handle:
        if size > maximum_bytes:
            handle.seek(size - maximum_bytes)
        payload = handle.read(maximum_bytes)
    return {
        "byte_count": int(size),
        "truncated": size > maximum_bytes,
        "tail": payload.decode("utf-8", errors="replace"),
    }


def _runtime_environment(cache_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(environment["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    return environment


def _normalized_command_prefix(
    command_prefix: Sequence[Path | str],
) -> tuple[tuple[str, ...], Path]:
    if len(command_prefix) not in {1, 2}:
        raise ValueError("MEDICC2 command prefix must be an executable or Python plus script.")
    resolved = tuple(str(Path(value).expanduser().resolve()) for value in command_prefix)
    for value in resolved:
        if not Path(value).is_file():
            raise ValueError(f"MEDICC2 command component is not a file: {value}")
    if len(resolved) == 2:
        interpreter = Path(resolved[0])
    else:
        first_line = Path(resolved[0]).read_bytes().splitlines()[0].decode(
            "utf-8", errors="strict"
        )
        if not first_line.startswith("#!"):
            raise ValueError("Installed MEDICC2 executable has no auditable shebang.")
        shebang = shlex.split(first_line[2:])
        if len(shebang) != 1 or not Path(shebang[0]).is_absolute():
            raise ValueError("MEDICC2 executable must pin one absolute Python interpreter.")
        interpreter = Path(shebang[0]).resolve()
        if not interpreter.is_file():
            raise ValueError("MEDICC2 shebang interpreter does not exist.")
    return resolved, interpreter


def _run_small_probe(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"MEDICC2 runtime probe failed with status {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    if len(completed.stdout.encode("utf-8")) > DEFAULT_CAPTURE_LIMIT_BYTES or len(
        completed.stderr.encode("utf-8")
    ) > DEFAULT_CAPTURE_LIMIT_BYTES:
        raise ValueError("MEDICC2 runtime probe exceeded the capture limit.")
    return completed


def _conda_package_lock(prefix: Path) -> list[dict[str, Any]]:
    metadata_root = prefix / "conda-meta"
    if not metadata_root.is_dir():
        raise ValueError("The publication MEDICC2 runtime must be a Conda environment.")
    packages = []
    for path in sorted(metadata_root.glob("*.json"), key=lambda item: item.name):
        value = json.loads(path.read_text(encoding="utf-8"))
        packages.append(
            {
                field: value.get(field)
                for field in (
                    "name",
                    "version",
                    "build",
                    "build_number",
                    "channel",
                    "subdir",
                    "url",
                    "sha256",
                    "md5",
                )
            }
        )
    if not packages:
        raise ValueError("The MEDICC2 Conda package lock is empty.")
    return packages


def inspect_medicc2_runtime(
    command_prefix: Sequence[Path | str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate installed code against the pinned checkout and freeze dependencies."""
    command, interpreter = _normalized_command_prefix(command_prefix)
    probe_code = r'''
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import medicc
import fstlib

medicc_root = Path(medicc.__file__).resolve().parent
fstlib_root = Path(fstlib.__file__).resolve().parent
distributions = sorted(
    {
        ((item.metadata.get("Name") or "").lower(), item.version)
        for item in importlib.metadata.distributions()
        if item.metadata.get("Name")
    }
)
value = {
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "runtime_prefix": sys.prefix,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "medicc2_version": medicc.__version__,
    "medicc_package_root": str(medicc_root),
    "fstlib_package_root": str(fstlib_root),
    "fstlib_extensions": [
        str(path) for path in sorted((fstlib_root / "cext").glob("*.so"))
    ],
    "python_distributions": [
        {"name": name, "version": version} for name, version in distributions
    ],
}
print("CTBF_MEDICC2_RUNTIME_JSON=" + json.dumps(value, sort_keys=True))
'''
    with tempfile.TemporaryDirectory(prefix="ctbf-medicc2-runtime-probe-") as raw:
        probe_root = Path(raw)
        environment = _runtime_environment(probe_root / "cache")
        completed = _run_small_probe(
            (str(interpreter), "-c", probe_code),
            environment=environment,
            cwd=probe_root,
        )
        marker = "CTBF_MEDICC2_RUNTIME_JSON="
        rows = [row for row in completed.stdout.splitlines() if row.startswith(marker)]
        if len(rows) != 1:
            raise ValueError("MEDICC2 runtime probe did not emit one identity record.")
        identity = json.loads(rows[0][len(marker) :])
        version = _run_small_probe(
            (*command, "--version"),
            environment=environment,
            cwd=probe_root,
        )
        help_result = _run_small_probe(
            (*command, "--help"),
            environment=environment,
            cwd=probe_root,
        )

    if identity.get("medicc2_version") != PINNED_MEDICC2_VERSION:
        raise ValueError("Installed MEDICC2 version differs from the source pin.")
    if "--topology-only" not in help_result.stdout:
        raise ValueError("Installed MEDICC2 lacks --topology-only.")
    medicc_root = Path(identity["medicc_package_root"])
    fstlib_extensions = [Path(path) for path in identity["fstlib_extensions"]]
    required_installed = (medicc_root / "core.py", medicc_root / "io.py")
    if not all(path.is_file() for path in (*required_installed, *fstlib_extensions)):
        raise ValueError("Installed MEDICC2 runtime files are incomplete.")
    installed_hashes = {
        "medicc/core.py": file_sha256(required_installed[0]),
        "medicc/io.py": file_sha256(required_installed[1]),
        **{
            f"fstlib/cext/{path.name}": file_sha256(path)
            for path in sorted(fstlib_extensions, key=lambda item: item.name)
        },
    }
    source_hashes = source.get("sha256", {})
    for relative in ("medicc/core.py", "medicc/io.py"):
        if installed_hashes[relative] != source_hashes.get(relative):
            raise ValueError(f"Installed {relative} differs from the pinned checkout.")
    conda_packages = _conda_package_lock(Path(identity["runtime_prefix"]))
    record = {
        "schema_version": MEDICC2_RUNTIME_SCHEMA_VERSION,
        "command_prefix": list(command),
        "command_component_sha256": {
            value: file_sha256(Path(value)) for value in command
        },
        "interpreter": str(interpreter),
        "interpreter_sha256": file_sha256(interpreter),
        "identity": identity,
        "installed_sha256": installed_hashes,
        "source_revision": source["revision"],
        "source_submodule_revision": source["submodule_revision"],
        "version_output": version.stdout.strip(),
        "topology_only_flag_present": True,
        "conda_package_lock": conda_packages,
        "conda_package_lock_sha256": canonical_json_digest(conda_packages),
    }
    record["runtime_sha256"] = canonical_json_digest(record)
    return record


def _sample_process_tree_rss() -> tuple[int | None, bool]:
    if psutil is None:
        return None, False
    try:
        root = psutil.Process(os.getpid())
        children = root.children(recursive=True)
    except Exception:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss), False
        except Exception:
            return None, False
    total = 0
    child_sampling_available = True
    for process in (root, *children):
        try:
            total += int(process.memory_info().rss)
        except (psutil.AccessDenied, OSError):
            child_sampling_available = False
            continue
        except psutil.NoSuchProcess:
            continue
    return total, child_sampling_available


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5.0)


def run_bounded_medicc2_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
    rss_limit_bytes: int,
    capture_limit_bytes: int,
) -> dict[str, Any]:
    """Run MEDICC2 with live timeout, stream-size, and process-tree RSS checks."""
    for field, value in (
        ("timeout_seconds", timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
    ):
        _validate_positive_integer(value, field)
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError("Refusing to overwrite MEDICC2 execution logs.")

    started_ns = time.perf_counter_ns()
    peak_rss_bytes = None
    sample_count = 0
    child_sampling_available = psutil is not None
    violation = None
    returncode = None
    process = None
    try:
        with stdout_path.open("xb") as stdout_handle, stderr_path.open(
            "xb"
        ) as stderr_handle:
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    env=dict(environment),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
            except OSError as error:
                violation = {
                    "code": "launch_failed",
                    "type": type(error).__name__,
                    "message": str(error)[:4096],
                }
            if process is not None:
                deadline = time.monotonic() + timeout_seconds
                while process.poll() is None:
                    rss, child_ok = _sample_process_tree_rss()
                    child_sampling_available = child_sampling_available and child_ok
                    if rss is not None:
                        sample_count += 1
                        peak_rss_bytes = rss if peak_rss_bytes is None else max(
                            peak_rss_bytes, rss
                        )
                    if rss is not None and rss > rss_limit_bytes:
                        violation = {
                            "code": "rss_limit_exceeded",
                            "message": (
                                f"Process-tree RSS {rss} exceeded {rss_limit_bytes}."
                            ),
                        }
                    elif (
                        stdout_path.stat().st_size > capture_limit_bytes
                        or stderr_path.stat().st_size > capture_limit_bytes
                    ):
                        violation = {
                            "code": "capture_limit_exceeded",
                            "message": "MEDICC2 stdout or stderr exceeded the byte limit.",
                        }
                    elif time.monotonic() >= deadline:
                        violation = {
                            "code": "timeout",
                            "message": f"MEDICC2 exceeded {timeout_seconds} seconds.",
                        }
                    if violation is not None:
                        _terminate_process_group(process)
                        break
                    time.sleep(MEMORY_SAMPLE_INTERVAL_SECONDS)
                returncode = process.poll()
                rss, child_ok = _sample_process_tree_rss()
                child_sampling_available = child_sampling_available and child_ok
                if rss is not None:
                    sample_count += 1
                    peak_rss_bytes = rss if peak_rss_bytes is None else max(
                        peak_rss_bytes, rss
                    )
    except BaseException:
        if process is not None:
            try:
                _terminate_process_group(process)
            except Exception:
                pass
        raise

    if psutil is None:
        memory_method = "unavailable_psutil_not_installed"
    elif child_sampling_available:
        memory_method = "psutil_process_tree_rss_10ms_live_limit"
    else:
        memory_method = "psutil_worker_rss_10ms_child_access_denied"
    stdout = _bounded_tail(stdout_path)
    stderr = _bounded_tail(stderr_path)
    if violation is None and (
        stdout["byte_count"] > capture_limit_bytes
        or stderr["byte_count"] > capture_limit_bytes
    ):
        violation = {
            "code": "capture_limit_exceeded",
            "message": "MEDICC2 stdout or stderr exceeded the byte limit.",
        }
    if violation is None and returncode != 0:
        violation = {
            "code": "external_process_failed",
            "message": f"MEDICC2 exited with status {returncode}.",
        }
    record = {
        "schema_version": MEDICC2_EXECUTION_SCHEMA_VERSION,
        "status": "success" if violation is None else "failure",
        "failure": violation,
        "command": list(command),
        "working_directory": str(cwd),
        "returncode": returncode,
        "wall_time_ns": time.perf_counter_ns() - started_ns,
        "timeout_seconds": timeout_seconds,
        "rss_limit_bytes": rss_limit_bytes,
        "capture_limit_bytes_per_stream": capture_limit_bytes,
        "memory": {
            "method": memory_method,
            "peak_rss_bytes": peak_rss_bytes,
            "sample_count": sample_count,
        },
        "stdout": stdout,
        "stderr": stderr,
    }
    return record


def expected_output_names(prefix: str) -> set[str]:
    return {f"{prefix}_{suffix}" for suffix in EXPECTED_OUTPUT_SUFFIXES}


def validate_output_inventory(output_root: Path, prefix: str) -> list[dict[str, Any]]:
    expected = expected_output_names(prefix)
    entries = list(output_root.iterdir())
    observed = {path.name for path in entries}
    if observed != expected or not all(path.is_file() for path in entries):
        raise ValueError(
            "MEDICC2 output inventory changed: "
            f"expected {sorted(expected)}, observed {sorted(observed)}."
        )
    return [
        {
            "name": name,
            "size_bytes": (output_root / name).stat().st_size,
            "sha256": file_sha256(output_root / name),
        }
        for name in sorted(expected)
    ]


def reconstruct_medicc2_case(
    reconstruction_input: Mapping[str, Any],
    case_root: Path | str,
    *,
    command_prefix: Sequence[Path | str],
    timeout_seconds: int,
    rss_limit_bytes: int,
    capture_limit_bytes: int,
    runtime_cache_root: Path | str | None = None,
) -> tuple[nx.DiGraph, Any, dict[str, Any], dict[str, Any]]:
    """Run one truth-free topology-only reconstruction in a new case directory."""
    validate_reconstruction_input(reconstruction_input)
    root = Path(case_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite MEDICC2 case root: {root}")
    root.mkdir(parents=True)
    output_root = root / "medicc2_output"
    output_root.mkdir()
    log_root = root / "execution"
    log_root.mkdir()
    cache_root = (
        root / "runtime_cache"
        if runtime_cache_root is None
        else Path(runtime_cache_root).expanduser().resolve()
    )
    prefix = str(reconstruction_input["case_id"])
    if not prefix or any(value in prefix for value in "/\\"):
        raise ValueError("Development case id is not a path-safe MEDICC2 prefix.")

    exported = export_observation_occurrences(
        reconstruction_input,
        root / "occurrence_input.tsv",
    )
    command, _interpreter = _normalized_command_prefix(command_prefix)
    full_command = topology_only_command(
        command,
        exported.path,
        output_root,
        prefix=prefix,
        n_cores=1,
        allow_wgd=False,
    )
    execution = run_bounded_medicc2_process(
        full_command,
        cwd=root,
        environment=_runtime_environment(cache_root),
        stdout_path=log_root / "stdout.log",
        stderr_path=log_root / "stderr.log",
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        capture_limit_bytes=capture_limit_bytes,
    )
    if execution["status"] != "success":
        raise Medicc2ExecutionError("MEDICC2 process failed.", execution)

    try:
        inventory = validate_output_inventory(output_root, prefix)
        distances = read_pairwise_distances(
            output_root / f"{prefix}_pairwise_distances.tsv"
        )
        expected_taxa = set(exported.taxon_to_state_label) | {
            ARTIFICIAL_NORMAL_TAXON
        }
        if set(distances) != expected_taxa:
            raise ValueError("MEDICC2 distance taxa differ from exported occurrences.")
        tree, returned_root, projection = project_phyloxml_to_partial_tree(
            output_root / f"{prefix}_final_tree.xml",
            exported.taxon_to_state_label,
        )
    except Exception as error:
        invalid = dict(execution)
        invalid["status"] = "failure"
        invalid["failure"] = {
            "code": "invalid_external_output",
            "type": type(error).__name__,
            "message": str(error)[:4096],
        }
        raise Medicc2ExecutionError("MEDICC2 output validation failed.", invalid) from error

    metadata = {
        "schema_version": MEDICC2_RECONSTRUCTION_SCHEMA_VERSION,
        "input_encoding": INPUT_ENCODING_VERSION,
        "projection": projection,
        "output_inventory": inventory,
        "external_process": execution,
        "wgd_policy": "disabled",
        "ancestor_policy": "topology_only_internal_nodes_unlabeled",
        "n_cores": 1,
        "truth_identity_used": False,
        "occurrence_input": {
            "occurrence_count": exported.occurrence_count,
            "non_diploid_occurrence_count": exported.non_diploid_occurrence_count,
            "unique_state_label_count": exported.unique_state_label_count,
            "cnp_length": exported.cnp_length,
            "chromosome_count": exported.chromosome_count,
            "taxon_to_observation": exported.taxon_to_observation,
        },
    }
    write_json(
        root / "reconstruction.json",
        {
            "schema_version": MEDICC2_RECONSTRUCTION_SCHEMA_VERSION,
            "case_id": reconstruction_input["case_id"],
            "arm_id": MEDICC2_DEVELOPMENT_ARM_ID,
            "tree": serialize_tree(tree),
            "metadata": metadata,
        },
    )
    runtime = {
        "wall_time_ns": execution["wall_time_ns"],
        "timeout_seconds": timeout_seconds,
        "rss_limit_bytes": rss_limit_bytes,
        "memory": execution["memory"],
        "external_execution_status": execution["status"],
    }
    return tree, returned_root, metadata, runtime


def _typed_error(error: BaseException, stage: str) -> dict[str, Any]:
    if isinstance(error, Medicc2ExecutionError):
        failure = error.record.get("failure") or {}
        return {
            "stage": stage,
            "type": "Medicc2ExecutionError",
            "code": failure.get("code", "external_failure"),
            "message": str(failure.get("message") or error)[:4096],
            "external_execution": error.record,
        }
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }


def _auditable_external_memory_error(runtime: Mapping[str, Any]) -> RuntimeError | None:
    memory = runtime.get("memory")
    if not isinstance(memory, Mapping) or memory.get("peak_rss_bytes") is None:
        return RuntimeError("MEDICC2 reconstruction has no peak-RSS measurement.")
    if memory.get("method") != "psutil_process_tree_rss_10ms_live_limit":
        return RuntimeError("MEDICC2 recursive-child RSS was not auditable.")
    return None


def _failure_record(
    case: Mapping[str, Any],
    error: BaseException,
    *,
    stage: str,
    reconstruction_runtime: Mapping[str, Any] | None,
    evaluation_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "block_index": int(case["block_index"]),
        "height": int(case["height"]),
        "arm_id": MEDICC2_DEVELOPMENT_ARM_ID,
        "family": CONTEXT_FAMILY,
        "problem": "partial",
        "primary_metric": "grf",
        "status": "failure",
        "failure": _typed_error(error, stage),
        "metrics": None,
        "observation_coverage": None,
        "tree_summary": None,
        "resources": {
            "reconstruction": reconstruction_runtime,
            "evaluation": evaluation_runtime,
        },
    }


def _read_observable_case_input(
    bank_root: Path, case: Mapping[str, Any]
) -> dict[str, Any]:
    payload = read_json(bank_root / str(case["input_path"]))
    validate_reconstruction_input(payload)
    if payload.get("case_id") != case.get("case_id"):
        raise ValueError("Stored MEDICC2 reconstruction input case id changed.")
    return payload


def _run_resource_gate_case(
    bank_root: str,
    case: Mapping[str, Any],
    case_root: str,
    command_prefix: Sequence[str],
    timeout_seconds: int,
    rss_limit_bytes: int,
    capture_limit_bytes: int,
    runtime_cache_root: str,
) -> dict[str, Any]:
    bank_path = Path(bank_root)
    input_payload = _read_observable_case_input(bank_path, case)
    runtime = None
    try:
        tree, _root, metadata, runtime = reconstruct_medicc2_case(
            input_payload,
            case_root,
            command_prefix=command_prefix,
            timeout_seconds=timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
            capture_limit_bytes=capture_limit_bytes,
            runtime_cache_root=runtime_cache_root,
        )
        memory_error = _auditable_external_memory_error(runtime)
        if memory_error is not None:
            raise memory_error
    except BaseException as error:
        external = error.record if isinstance(error, Medicc2ExecutionError) else None
        failure_runtime = runtime
        if failure_runtime is None and external is not None:
            failure_runtime = {
                "wall_time_ns": external["wall_time_ns"],
                "timeout_seconds": timeout_seconds,
                "rss_limit_bytes": rss_limit_bytes,
                "memory": external["memory"],
                "external_execution_status": external["status"],
            }
        return {
            "case_id": case["case_id"],
            "block_index": int(case["block_index"]),
            "height": int(case["height"]),
            "status": "failure",
            "failure": _typed_error(error, "reconstruction"),
            "resources": {"reconstruction": failure_runtime},
        }
    summary = tree_summary(tree)
    summary["inferred_copy_occurrence_count"] = 0
    return {
        "case_id": case["case_id"],
        "block_index": int(case["block_index"]),
        "height": int(case["height"]),
        "status": "success",
        "failure": None,
        "tree_summary": summary,
        "occurrence_input": metadata["occurrence_input"],
        "resources": {"reconstruction": runtime},
        "accuracy_evaluated": False,
    }


def _has_plus2_interval_event(event: Mapping[str, Any]) -> bool:
    return (
        event.get("event_class") == "interval_mode_cna"
        and event.get("direction") == "gain"
        and event.get("operator") == "additive"
        and int(event.get("magnitude", -1)) == 2
        and bool(event.get("multi_position"))
        and bool(event.get("effective"))
    )


def _observed_plus2_ancestry_count(
    input_payload: Mapping[str, Any],
    truth_tree: nx.DiGraph,
    height: int,
) -> int:
    prefix = truth_prefix(truth_tree, height)
    labels = set(observed_labels(input_payload))
    count = 0
    for _parent, child, attributes in prefix.edges(data=True):
        events = attributes.get("events", [])
        if not any(
            isinstance(event, Mapping) and _has_plus2_interval_event(event)
            for event in events
        ):
            continue
        descendants = nx.descendants(prefix, child) | {child}
        if any(prefix.nodes[node].get("cell_id") in labels for node in descendants):
            count += 1
    return count


def select_resource_gate_cases(
    bank_root: Path | str,
    *,
    expected_block_count: int = DEFAULT_BLOCK_COUNT,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Select min/max occurrence cases plus a distinct observed-path +2 case."""
    root, bank = load_bank_manifest(
        bank_root, expected_block_count=expected_block_count
    )
    rows = []
    truth_cache: dict[str, nx.DiGraph] = {}
    for case in sorted(
        bank["cases"], key=lambda value: (int(value["block_index"]), int(value["height"]))
    ):
        input_payload = _read_observable_case_input(root, case)
        metadata = read_json(root / str(case["metadata_path"]))
        if metadata.get("schema_version") != CASE_METADATA_SCHEMA_VERSION:
            raise ValueError("Unknown stored case-metadata schema.")
        if metadata.get("case_id") != case.get("case_id"):
            raise ValueError("Stored case metadata id changed.")
        if metadata.get("block_index") != int(case["block_index"]):
            raise ValueError("Stored case metadata block index changed.")
        if metadata.get("height") != int(case["height"]):
            raise ValueError("Stored case metadata height changed.")
        occurrence_count = sum(
            len(level["states"]) for level in input_payload["levels"]
        )
        if int(metadata.get("selected_occurrence_count", -1)) != occurrence_count:
            raise ValueError("Stored occurrence count differs from observable input.")
        truth_relative = str(case["truth_path"])
        if truth_relative not in truth_cache:
            truth_payload = read_json(root / truth_relative)
            if truth_payload.get("block_index") != int(case["block_index"]):
                raise ValueError("Stored truth block index changed.")
            truth_cache[truth_relative] = deserialize_truth(truth_payload)
        plus2_count = _observed_plus2_ancestry_count(
            input_payload,
            truth_cache[truth_relative],
            int(case["height"]),
        )
        rows.append(
            {
                "case": dict(case),
                "occurrence_count": occurrence_count,
                "observed_plus2_ancestry_count": plus2_count,
            }
        )
    return root, bank, _select_resource_gate_rows(rows)


def _select_resource_gate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Choose three distinct output-blind resource cases deterministically."""
    if len(rows) < 3:
        raise ValueError("The MEDICC2 resource gate requires at least three cases.")
    smallest = min(
        rows,
        key=lambda row: (
            row["occurrence_count"],
            int(row["case"]["block_index"]),
            int(row["case"]["height"]),
        ),
    )
    largest_candidates = [
        row
        for row in rows
        if row["case"]["case_id"] != smallest["case"]["case_id"]
    ]
    largest = min(
        largest_candidates,
        key=lambda row: (
            -row["occurrence_count"],
            int(row["case"]["block_index"]),
            int(row["case"]["height"]),
        ),
    )
    occupied = {smallest["case"]["case_id"], largest["case"]["case_id"]}
    plus2_candidates = [
        row
        for row in rows
        if row["observed_plus2_ancestry_count"] > 0
        and row["case"]["case_id"] not in occupied
    ]
    if not plus2_candidates:
        raise ValueError(
            "No distinct bank case has a realized +2 interval on observed ancestry."
        )
    plus2 = min(
        plus2_candidates,
        key=lambda row: (
            -row["occurrence_count"],
            int(row["case"]["block_index"]),
            int(row["case"]["height"]),
        ),
    )
    selected = []
    for role, row in (
        ("smallest_observed_occurrence_count", smallest),
        ("largest_observed_occurrence_count", largest),
        ("largest_distinct_realized_plus2_on_observed_ancestry", plus2),
    ):
        selected.append(
            {
                **row["case"],
                "selection_role": role,
                "selected_occurrence_count": row["occurrence_count"],
                "observed_plus2_ancestry_count": row[
                    "observed_plus2_ancestry_count"
                ],
            }
        )
    if len({row["case_id"] for row in selected}) != 3:
        raise AssertionError("MEDICC2 resource-gate case selection must be distinct.")
    return selected


def _resource_contract(
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
    capture_limit_bytes: int,
) -> dict[str, Any]:
    return {
        "reconstruction_timeout_seconds_per_arm": reconstruction_timeout_seconds,
        "evaluation_timeout_seconds_per_arm": evaluation_timeout_seconds,
        "rss_limit_bytes_per_stage": rss_limit_bytes,
        "stdout_stderr_limit_bytes_per_external_process": capture_limit_bytes,
        "memory_sampling_interval_ms": int(MEMORY_SAMPLE_INTERVAL_SECONDS * 1000),
        "record_worker_outer_timeout_seconds": (
            reconstruction_timeout_seconds
            + evaluation_timeout_seconds
            + OUTER_WORKER_GRACE_SECONDS
        ),
        "record_execution": fresh_process_contract(CASE_ARM_WORKER_UNIT),
        "external_execution": {
            "fresh_medicc2_subprocess_per_case": True,
            "n_cores": 1,
            "live_timeout_enforced": True,
            "live_process_tree_rss_limit_enforced": True,
            "live_capture_limit_enforced": True,
        },
    }


def run_resource_gate(
    *,
    bank_root: Path | str,
    output_root: Path | str,
    command_prefix: Sequence[Path | str],
    source_root: Path | str,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    expected_block_count: int = DEFAULT_BLOCK_COUNT,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    for field, value in (
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
        ("expected_block_count", expected_block_count),
    ):
        _validate_positive_integer(value, field)
    source = inspect_source_checkout(source_root)
    runtime = inspect_medicc2_runtime(command_prefix, source)
    root = ensure_new_output_root(output_root)
    bank_path, bank, selected = select_resource_gate_cases(
        bank_root, expected_block_count=expected_block_count
    )
    normalized_command, _interpreter = _normalized_command_prefix(command_prefix)
    preflight = run_fixture_preflight(
        command_prefix=normalized_command,
        source_root=source_root,
        output_root=root / "semantic_preflight",
        timeout_seconds=reconstruction_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        capture_limit_bytes=capture_limit_bytes,
        require_recursive_rss=True,
    )
    semantic_gate = {
        "status": "passed" if preflight["status"] == "success" else "failed",
        "fixture_results": [],
        "external_fixture_results": preflight["records"],
        "source_revision": source["revision"],
        "runtime_sha256": runtime["runtime_sha256"],
    }
    result = {
        "schema_version": MEDICC2_RESOURCE_GATE_SCHEMA_VERSION,
        "status": "in_progress",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "scientific_role": "output_blind_resource_gate_not_accuracy_evidence",
        "arm_spec": dict(MEDICC2_ARM_SPEC),
        "bank_id": bank["bank_id"],
        "bank_root": str(bank_path),
        "bank_manifest_sha256": file_sha256(bank_path / BANK_MANIFEST_NAME),
        "block_count": int(bank["block_count"]),
        "condition_count": int(bank["condition_count"]),
        "medicc2_source": source,
        "medicc2_runtime": runtime,
        "semantic_gate": semantic_gate,
        "selection": {
            "case_count": len(selected),
            "cases": selected,
            "accuracy_or_scores_examined": False,
            "truth_used_only_to_identify_realized_plus2_ancestry": True,
            "truth_passed_to_reconstruction": False,
        },
        "resources": _resource_contract(
            reconstruction_timeout_seconds,
            DEFAULT_EVALUATION_TIMEOUT_SECONDS,
            rss_limit_bytes,
            capture_limit_bytes,
        ),
        "records": [],
    }
    write_json(root / MEDICC2_RESOURCE_GATE_NAME, result)
    if semantic_gate["status"] == "passed":
        with FreshSpawnPerTaskExecutor() as executor:
            for selected_case in selected:
                case = {
                    key: value
                    for key, value in selected_case.items()
                    if key
                    not in {
                        "selection_role",
                        "selected_occurrence_count",
                        "observed_plus2_ancestry_count",
                    }
                }
                try:
                    record = executor.run(
                        _run_resource_gate_case,
                        str(bank_path),
                        case,
                        str(root / "bank_cases" / str(case["case_id"])),
                        normalized_command,
                        reconstruction_timeout_seconds,
                        rss_limit_bytes,
                        capture_limit_bytes,
                        str(root / "runtime_cache"),
                        timeout_seconds=(
                            reconstruction_timeout_seconds + OUTER_WORKER_GRACE_SECONDS
                        ),
                    )
                except FreshProcessTimeoutError as error:
                    record = {
                        "case_id": case["case_id"],
                        "block_index": int(case["block_index"]),
                        "height": int(case["height"]),
                        "status": "failure",
                        "failure": _typed_error(error, "outer_worker"),
                        "resources": {"reconstruction": None},
                    }
                record["selection_role"] = selected_case["selection_role"]
                result["records"].append(record)
                write_json(root / MEDICC2_RESOURCE_GATE_NAME, result)
    result["status"] = (
        "success"
        if semantic_gate["status"] == "passed"
        and len(result["records"]) == 3
        and all(record["status"] == "success" for record in result["records"])
        else "failure"
    )
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / MEDICC2_RESOURCE_GATE_NAME, result)
    return result


def _validate_resource_gate_for_run(
    gate_path: Path | str,
    *,
    bank_root: Path,
    bank: Mapping[str, Any],
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    reconstruction_timeout_seconds: int,
    rss_limit_bytes: int,
    capture_limit_bytes: int,
) -> tuple[Path, dict[str, Any]]:
    path = Path(gate_path).expanduser().resolve()
    if path.is_dir():
        path = path / MEDICC2_RESOURCE_GATE_NAME
    gate = read_json(path)
    if gate.get("schema_version") != MEDICC2_RESOURCE_GATE_SCHEMA_VERSION:
        raise ValueError("Unknown MEDICC2 resource-gate schema.")
    if gate.get("status") != "success":
        raise ValueError("MEDICC2 resource gate did not pass.")
    expected = {
        "bank_id": bank["bank_id"],
        "bank_root": str(bank_root),
        "bank_manifest_sha256": file_sha256(bank_root / BANK_MANIFEST_NAME),
        "block_count": int(bank["block_count"]),
        "condition_count": int(bank["condition_count"]),
    }
    for field, value in expected.items():
        if gate.get(field) != value:
            raise ValueError(f"MEDICC2 resource-gate {field} does not match this run.")
    if gate.get("medicc2_source") != source:
        raise ValueError("MEDICC2 source differs from the passing resource gate.")
    if gate.get("medicc2_runtime") != runtime:
        raise ValueError("MEDICC2 runtime differs from the passing resource gate.")
    resources = gate.get("resources", {})
    required_resources = {
        "reconstruction_timeout_seconds_per_arm": reconstruction_timeout_seconds,
        "rss_limit_bytes_per_stage": rss_limit_bytes,
        "stdout_stderr_limit_bytes_per_external_process": capture_limit_bytes,
    }
    for field, value in required_resources.items():
        if resources.get(field) != value:
            raise ValueError(f"MEDICC2 resource-gate {field} changed.")
    return path, gate


def _run_development_case(
    bank_root: str,
    case: Mapping[str, Any],
    case_root: str,
    command_prefix: Sequence[str],
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
    capture_limit_bytes: int,
    runtime_cache_root: str,
) -> dict[str, Any]:
    bank_path = Path(bank_root)
    input_payload = _read_observable_case_input(bank_path, case)
    reconstruction_runtime = None
    try:
        tree, _returned_root, metadata, reconstruction_runtime = (
            reconstruct_medicc2_case(
                input_payload,
                case_root,
                command_prefix=command_prefix,
                timeout_seconds=reconstruction_timeout_seconds,
                rss_limit_bytes=rss_limit_bytes,
                capture_limit_bytes=capture_limit_bytes,
                runtime_cache_root=runtime_cache_root,
            )
        )
        memory_error = _auditable_external_memory_error(reconstruction_runtime)
        if memory_error is not None:
            raise memory_error
    except BaseException as error:
        if isinstance(error, Medicc2ExecutionError):
            execution = error.record
            reconstruction_runtime = {
                "wall_time_ns": execution["wall_time_ns"],
                "timeout_seconds": reconstruction_timeout_seconds,
                "rss_limit_bytes": rss_limit_bytes,
                "memory": execution["memory"],
                "external_execution_status": execution["status"],
            }
        return _failure_record(
            case,
            error,
            stage="reconstruction",
            reconstruction_runtime=reconstruction_runtime,
        )

    # Truth is loaded only after the external reconstruction has terminated.
    truth_payload = read_json(bank_path / str(case["truth_path"]))
    if truth_payload.get("block_index") != int(case["block_index"]):
        return _failure_record(
            case,
            ValueError("Stored truth block index changed."),
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
        )
    truth_tree = truth_prefix(deserialize_truth(truth_payload), int(case["height"]))
    labels = observed_labels(input_payload)
    evaluation, evaluation_runtime, evaluation_error = measured_stage(
        lambda: evaluate_tree_pair_result(truth_tree, tree, labels),
        timeout_seconds=evaluation_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if evaluation_error is None:
        memory = evaluation_runtime.get("memory", {})
        if memory.get("peak_rss_bytes") is None:
            evaluation_error = RuntimeError("Evaluation has no peak-RSS measurement.")
    if evaluation_error is not None:
        return _failure_record(
            case,
            evaluation_error,
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=evaluation_runtime,
        )
    if evaluation is None:
        return _failure_record(
            case,
            RuntimeError("Evaluation returned no result."),
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=evaluation_runtime,
        )
    try:
        validate_evaluation_result(evaluation)
    except BaseException as error:
        return _failure_record(
            case,
            error,
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=evaluation_runtime,
        )
    if evaluation["status"] != "success":
        error = RuntimeError(str(evaluation.get("failure")))
        return _failure_record(
            case,
            error,
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=evaluation_runtime,
        )
    coverage = evaluation["inputs"]["observation_label_coverage"]
    if float(coverage["fraction"]) < 1.0:
        return _failure_record(
            case,
            RuntimeError("Reconstructed observation-label coverage is below one."),
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=evaluation_runtime,
        )
    projected_labels = [
        attributes.get("cell_id")
        for _node, attributes in tree.nodes(data=True)
        if attributes.get("cell_id") is not None
    ]
    input_occurrences = [
        state["state_label"]
        for level in input_payload["levels"]
        for state in level["states"]
    ]
    if Counter(projected_labels) != Counter(input_occurrences):
        return _failure_record(
            case,
            RuntimeError("MEDICC2 projection changed observation occurrence multiplicity."),
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=evaluation_runtime,
        )
    summary = tree_summary(tree)
    summary["inferred_copy_occurrence_count"] = 0
    summary["repeated_observed_occurrence_count"] = (
        len(projected_labels) - len(set(projected_labels))
    )
    return {
        "case_id": case["case_id"],
        "block_index": int(case["block_index"]),
        "height": int(case["height"]),
        "arm_id": MEDICC2_DEVELOPMENT_ARM_ID,
        "family": CONTEXT_FAMILY,
        "problem": "partial",
        "primary_metric": "grf",
        "status": "success",
        "failure": None,
        "metrics": {"grf": float(evaluation["metrics"]["grf"])},
        "observation_coverage": {
            "required_unique_label_count": coverage["required_unique_label_count"],
            "reconstructed_unique_label_count": coverage[
                "reconstructed_unique_label_count"
            ],
            "fraction": float(coverage["fraction"]),
        },
        "tree_summary": summary,
        "reconstruction_metadata": metadata,
        "resources": {
            "reconstruction": reconstruction_runtime,
            "evaluation": evaluation_runtime,
        },
    }


def _resume_run(
    *,
    output_root: Path | str,
    run_id: str,
    bank_root: Path,
    bank: Mapping[str, Any],
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    gate_path: Path,
    gate_sha256: str,
    resources: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = Path(output_root).expanduser().resolve()
    path = root / RESULT_NAME
    if not path.is_file():
        raise ValueError("A resumed MEDICC2 run requires an existing result.json.")
    result = read_json(path)
    exact = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "bank_id": bank["bank_id"],
        "bank_root": str(bank_root),
        "block_count": int(bank["block_count"]),
        "condition_count": int(bank["condition_count"]),
        "arm_specs": [dict(MEDICC2_ARM_SPEC)],
        "medicc2_source": source,
        "medicc2_runtime": runtime,
        "resource_gate_path": str(gate_path),
        "resource_gate_sha256": gate_sha256,
        "resources": dict(resources),
    }
    for field, value in exact.items():
        if result.get(field) != value:
            raise ValueError(f"Cannot resume because stored {field} changed.")
    if result.get("status") not in {"in_progress", "failure"}:
        raise ValueError("Only an interrupted or failed MEDICC2 run can resume.")
    records = result.get("records")
    if not isinstance(records, list):
        raise ValueError("Stored MEDICC2 record inventory is invalid.")
    result["status"] = "in_progress"
    result.pop("runner_failure", None)
    result.setdefault("resume_history", []).append(
        {
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "preserved_record_count": len(records),
        }
    )
    write_json(path, result)
    return root, result


def run_development_arm(
    *,
    bank_root: Path | str,
    output_root: Path | str,
    run_id: str,
    command_prefix: Sequence[Path | str],
    source_root: Path | str,
    resource_gate_path: Path | str,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    expected_block_count: int = DEFAULT_BLOCK_COUNT,
    created_at_utc: str | None = None,
    progress: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if not run_id or not run_id.strip():
        raise ValueError("run_id must be nonempty.")
    for field, value in (
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("evaluation_timeout_seconds", evaluation_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
        ("expected_block_count", expected_block_count),
    ):
        _validate_positive_integer(value, field)
    bank_path, bank = load_bank_manifest(
        bank_root, expected_block_count=expected_block_count
    )
    source = inspect_source_checkout(source_root)
    runtime = inspect_medicc2_runtime(command_prefix, source)
    normalized_command, _interpreter = _normalized_command_prefix(command_prefix)
    gate_file, gate = _validate_resource_gate_for_run(
        resource_gate_path,
        bank_root=bank_path,
        bank=bank,
        source=source,
        runtime=runtime,
        reconstruction_timeout_seconds=reconstruction_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        capture_limit_bytes=capture_limit_bytes,
    )
    resources = _resource_contract(
        reconstruction_timeout_seconds,
        evaluation_timeout_seconds,
        rss_limit_bytes,
        capture_limit_bytes,
    )
    cases = sorted(
        bank["cases"], key=lambda value: (int(value["block_index"]), int(value["height"]))
    )
    gate_sha256 = file_sha256(gate_file)
    if resume:
        root, result = _resume_run(
            output_root=output_root,
            run_id=run_id,
            bank_root=bank_path,
            bank=bank,
            source=source,
            runtime=runtime,
            gate_path=gate_file,
            gate_sha256=gate_sha256,
            resources=resources,
        )
    else:
        root = ensure_new_output_root(output_root)
        result = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "in_progress",
            "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
            "scientific_role": "method_development_only_not_paper_accuracy_evidence",
            "bank_id": bank["bank_id"],
            "bank_root": str(bank_path),
            "block_count": int(bank["block_count"]),
            "condition_count": int(bank["condition_count"]),
            "arm_specs": [dict(MEDICC2_ARM_SPEC)],
            "semantic_gate_by_arm": {
                MEDICC2_DEVELOPMENT_ARM_ID: gate["semantic_gate"]
            },
            "expected_record_count": len(cases),
            "resources": resources,
            "medicc2_source": source,
            "medicc2_runtime": runtime,
            "resource_gate_path": str(gate_file),
            "resource_gate_sha256": gate_sha256,
            "truth_identity_used_for_reconstruction": False,
            "records": [],
        }
        write_json(root / RESULT_NAME, result)

    completed = len(result["records"])
    expected_prefix = [str(case["case_id"]) for case in cases[:completed]]
    observed_prefix = [str(record.get("case_id")) for record in result["records"]]
    if observed_prefix != expected_prefix:
        raise ValueError("Stored MEDICC2 records are not the exact run-order prefix.")
    try:
        with FreshSpawnPerTaskExecutor() as executor:
            for index, case in enumerate(cases[completed:], start=completed):
                record = executor.run(
                    _run_development_case,
                    str(bank_path),
                    dict(case),
                    str(root / "cases" / str(case["case_id"])),
                    normalized_command,
                    reconstruction_timeout_seconds,
                    evaluation_timeout_seconds,
                    rss_limit_bytes,
                    capture_limit_bytes,
                    str(root / "runtime_cache"),
                    timeout_seconds=resources[
                        "record_worker_outer_timeout_seconds"
                    ],
                )
                result["records"].append(record)
                result["completed_record_count"] = len(result["records"])
                result["completed_condition_count"] = index + 1
                write_json(root / RESULT_NAME, result)
                if progress:
                    print(
                        f"{case['case_id']} {MEDICC2_DEVELOPMENT_ARM_ID}: "
                        f"{record['status']}",
                        file=sys.stderr,
                        flush=True,
                    )
    except BaseException as error:
        result["status"] = "failure"
        result["runner_failure"] = _typed_error(error, "runner")
        result["completed_record_count"] = len(result["records"])
        write_json(root / RESULT_NAME, result)
        raise
    result["status"] = "complete"
    result["completed_record_count"] = len(result["records"])
    result["completed_condition_count"] = len(cases)
    result["success_count"] = sum(
        record["status"] == "success" for record in result["records"]
    )
    result["failure_count"] = len(result["records"]) - result["success_count"]
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / RESULT_NAME, result)
    return result


def _command_prefix(arguments: argparse.Namespace) -> list[str]:
    values = [arguments.medicc2_executable]
    if arguments.medicc2_python:
        values.insert(0, arguments.medicc2_python)
    return values


def _add_common_external_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--medicc2-executable", required=True)
    parser.add_argument("--medicc2-python")
    parser.add_argument(
        "--source-root", default=str(Path("~/Work/Medicc").expanduser())
    )
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reconstruction-timeout-seconds",
        type=int,
        default=DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES
    )
    parser.add_argument(
        "--capture-limit-bytes", type=int, default=DEFAULT_CAPTURE_LIMIT_BYTES
    )
    parser.add_argument(
        "--expected-block-count", type=int, default=DEFAULT_BLOCK_COUNT
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    gate = subparsers.add_parser(
        "resource-gate", description="Run D0 plus exactly three bank resource cases."
    )
    _add_common_external_arguments(gate)
    run = subparsers.add_parser(
        "run", description="Run the admitted MEDICC2 arm over the complete bank."
    )
    _add_common_external_arguments(run)
    run.add_argument("--resource-gate-result", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument(
        "--evaluation-timeout-seconds",
        type=int,
        default=DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    )
    run.add_argument("--progress", action="store_true")
    run.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "resource-gate":
        result = run_resource_gate(
            bank_root=arguments.bank_root,
            output_root=arguments.output_root,
            command_prefix=_command_prefix(arguments),
            source_root=arguments.source_root,
            reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
            rss_limit_bytes=arguments.rss_limit_bytes,
            capture_limit_bytes=arguments.capture_limit_bytes,
            expected_block_count=arguments.expected_block_count,
        )
        print(
            f"MEDICC2 resource gate {result['status']}: "
            f"{sum(row['status'] == 'success' for row in result['records'])}/3 "
            "bank cases succeeded."
        )
        return 0 if result["status"] == "success" else 1
    result = run_development_arm(
        bank_root=arguments.bank_root,
        output_root=arguments.output_root,
        run_id=arguments.run_id,
        command_prefix=_command_prefix(arguments),
        source_root=arguments.source_root,
        resource_gate_path=arguments.resource_gate_result,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
        expected_block_count=arguments.expected_block_count,
        progress=arguments.progress,
        resume=arguments.resume,
    )
    print(
        f"complete: {result['completed_record_count']} MEDICC2 records; "
        f"{result['failure_count']} failures"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CAPTURE_LIMIT_BYTES",
    "DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS",
    "DEFAULT_RSS_LIMIT_BYTES",
    "EXPECTED_OUTPUT_SUFFIXES",
    "MEDICC2_ARM_SPEC",
    "MEDICC2_DEVELOPMENT_ARM_ID",
    "MEDICC2_RESOURCE_GATE_SCHEMA_VERSION",
    "Medicc2ExecutionError",
    "expected_output_names",
    "inspect_medicc2_runtime",
    "reconstruct_medicc2_case",
    "run_bounded_medicc2_process",
    "run_development_arm",
    "run_resource_gate",
    "select_resource_gate_cases",
    "validate_output_inventory",
]
