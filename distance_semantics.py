"""Shared distance semantics and provenance for CTBF reconstruction inputs."""

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import numpy as np


CNP2CNP_DISTANCE = "any"
CNP2CNP_SYMMETRIZATION = "minimum_bidirectional"
CNP2CNP_SEMANTICS_VERSION = "ctbf-cnp2cnp-any-min-bidirectional-v1"
DISTANCE_PROVENANCE_SCHEMA_VERSION = "ctbf-distance-provenance-v1"


def _validated_ids(ids):
    if ids is None:
        raise ValueError("Distance matrix ids are required.")

    values = list(ids)
    seen = set()
    for value in values:
        try:
            if value in seen:
                raise ValueError(f"Duplicate distance-matrix id {value!r}.")
            seen.add(value)
        except TypeError as exc:
            raise ValueError("Distance matrix ids must be hashable.") from exc
    return values


def validate_distance_matrix(ids, matrix):
    """Validate the exact symmetric dissimilarity contract used by CTBF."""
    values = _validated_ids(ids)
    try:
        array = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Distance matrix must contain numeric values.") from exc

    if array.ndim != 2:
        raise ValueError("Distance matrix must be two-dimensional.")
    rows, columns = array.shape
    if rows != columns:
        raise ValueError(f"Distance matrix must be square, got shape {array.shape}.")
    if len(values) != rows:
        raise ValueError(f"Distance matrix has {rows} rows but {len(values)} ids.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Distance matrix must contain only finite values.")
    if np.any(array < 0):
        raise ValueError("Distance matrix must contain nonnegative dissimilarities.")
    if not np.array_equal(array, array.T):
        raise ValueError("Distance matrix must be exactly symmetric.")
    if not np.all(np.diag(array) == 0):
        raise ValueError("Distance matrix diagonal must be exactly zero.")
    return values, np.array(array, copy=True)


def parse_cnp2cnp_directional_distance(output):
    """Parse one cnp2cnp ``dist`` result without accepting diagnostic text."""
    tokens = str(output).strip().split()
    if len(tokens) != 1:
        raise ValueError(
            "cnp2cnp directional output must contain exactly one numeric value."
        )
    try:
        value = float(tokens[0])
    except ValueError as exc:
        raise ValueError("cnp2cnp directional output is not numeric.") from exc
    if not np.isfinite(value):
        raise ValueError("cnp2cnp directional distance must be finite.")
    if value < 0:
        raise ValueError("cnp2cnp directional distance must be nonnegative.")
    return value


def minimum_bidirectional_distance(forward, reverse):
    """Return min(d(u,v), d(v,u)) after validating both directions."""
    forward_value = parse_cnp2cnp_directional_distance(forward)
    reverse_value = parse_cnp2cnp_directional_distance(reverse)
    return min(forward_value, reverse_value)


def combine_ordered_cnp2cnp_matrices(
    forward_ids,
    forward_matrix,
    reverse_ids,
    reverse_matrix,
):
    """Combine two opposite FASTA orders into the declared symmetric matrix.

    cnp2cnp matrix mode computes only the direction induced by row order and
    mirrors it. Reversing the complete row order evaluates the opposite
    direction for every unordered pair. This function realigns the second
    matrix and takes the elementwise minimum.
    """
    forward_ids, forward_matrix = validate_distance_matrix(
        forward_ids,
        forward_matrix,
    )
    reverse_ids, reverse_matrix = validate_distance_matrix(
        reverse_ids,
        reverse_matrix,
    )
    if set(forward_ids) != set(reverse_ids):
        raise ValueError("Opposite-order cnp2cnp matrices contain different ids.")

    reverse_index = {value: index for index, value in enumerate(reverse_ids)}
    order = [reverse_index[value] for value in forward_ids]
    reverse_aligned = reverse_matrix[np.ix_(order, order)]
    symmetric = np.minimum(forward_matrix, reverse_aligned)
    return validate_distance_matrix(forward_ids, symmetric)


def _file_sha256(path):
    digest = sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(source_dir):
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        description = subprocess.run(
            ["git", "-C", str(source_dir), "describe", "--always", "--dirty", "--tags"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return revision or None, description or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def cnp2cnp_provenance(runfile=None, *, construction, python_executable=None):
    """Build JSON-safe provenance for a newly constructed cnp2cnp matrix."""
    python_executable = str(python_executable or sys.executable)
    provenance = {
        "schema_version": DISTANCE_PROVENANCE_SCHEMA_VERSION,
        "metric": "cnp2cnp",
        "distance_mode": CNP2CNP_DISTANCE,
        "semantics_version": CNP2CNP_SEMANTICS_VERSION,
        "symmetrization": CNP2CNP_SYMMETRIZATION,
        "formula": "min(d_any(u,v),d_any(v,u))",
        "construction": construction,
        "directional_calls_per_unordered_pair": (
            0 if construction == "trivial_singleton" else 2
        ),
        "python_executable": python_executable,
        "cnp2cnp_executable": None,
        "cnp2cnp_source_revision": None,
        "cnp2cnp_source_description": None,
        "source_sha256": {},
        "command_template": None,
    }
    if runfile is None:
        return provenance

    executable = Path(runfile).expanduser().resolve()
    provenance["cnp2cnp_executable"] = str(executable)
    if construction == "opposite_order_matrix_mode":
        provenance["command_template"] = [
            python_executable,
            str(executable),
            "-m",
            "matrix",
            "-d",
            CNP2CNP_DISTANCE,
            "-i",
            "<ordered-or-reversed-input.fa>",
            "-o",
            "<temporary-output.phy>",
        ]
    elif construction == "bidirectional_pair_mode":
        provenance["command_template"] = [
            python_executable,
            str(executable),
            "-m",
            "dist",
            "-d",
            CNP2CNP_DISTANCE,
            "-i",
            "<forward-or-reverse-pair.fa>",
        ]
    elif construction == "direct_bidirectional_api":
        provenance["command_template"] = [
            "python-api",
            "CNPSolver.get_comparable_cnps",
            "CNPSolver.get_approximate_events",
            "<u-to-v-and-v-to-u>",
        ]

    if executable.is_file():
        source_files = [executable]
        solver = executable.with_name("cnpsolver.py")
        if solver.is_file():
            source_files.append(solver)
        provenance["source_sha256"] = {
            path.name: _file_sha256(path)
            for path in source_files
        }
        revision, description = _source_revision(executable.parent)
        provenance["cnp2cnp_source_revision"] = revision
        provenance["cnp2cnp_source_description"] = description
    return provenance


__all__ = [
    "CNP2CNP_DISTANCE",
    "CNP2CNP_SEMANTICS_VERSION",
    "CNP2CNP_SYMMETRIZATION",
    "DISTANCE_PROVENANCE_SCHEMA_VERSION",
    "cnp2cnp_provenance",
    "combine_ordered_cnp2cnp_matrices",
    "minimum_bidirectional_distance",
    "parse_cnp2cnp_directional_distance",
    "validate_distance_matrix",
]
