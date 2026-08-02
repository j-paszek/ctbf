"""Shared distance semantics and provenance for CTBF reconstruction inputs."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


CNP2CNP_DISTANCE = "any"
CNP2CNP_SYMMETRIZATION = "minimum_bidirectional"
CNP2CNP_SEMANTICS_VERSION = "ctbf-cnp2cnp-any-min-bidirectional-v1"
CNP2CNP_ORDERED_TRIANGLE_FAST = "ordered_triangle_fast"
CNP2CNP_ORDERED_TRIANGLE_FAST_SEMANTICS_VERSION = (
    "ctbf-cnp2cnp-any-ordered-triangle-fast-v1"
)
DIRECTED_DISTANCE_BUNDLE_SCHEMA_VERSION = "ctbf-directed-distance-bundle-v1"
DISTANCE_PROVENANCE_SCHEMA_VERSION = "ctbf-distance-provenance-v1"
DISTANCE_INPUT_CACHE_KEY_SCHEMA_VERSION = "ctbf-distance-input-cache-key-v1"


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


def parse_distance_label(value):
    """Parse the integer labels emitted by cnp2cnp, preserving other text."""
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def parse_labeled_distance_matrix(path):
    """Parse and validate the PHYLIP-like matrix format accepted by CTBF."""
    with open(path) as source:
        size_line = source.readline().strip()
        try:
            size = int(size_line)
        except ValueError as exc:
            raise ValueError("Distance matrix is missing a valid size line.") from exc
        if size < 0:
            raise ValueError("Distance matrix size must be nonnegative.")

        ids = []
        rows = []
        for row_index in range(size):
            parts = source.readline().strip().split()
            if len(parts) != size + 1:
                raise ValueError(
                    f"Distance matrix row {row_index} has "
                    f"{max(len(parts) - 1, 0)} values; expected {size}."
                )
            ids.append(parse_distance_label(parts[0]))
            rows.append(parts[1:])
        if any(line.strip() for line in source):
            raise ValueError("Distance matrix contains unexpected extra rows.")
    if size == 0:
        rows = np.zeros((0, 0), dtype=float)
    return validate_distance_matrix(ids, rows)


def validate_distance_label_coverage(ids, required_ids, *, allow_extra=True):
    """Require every observed CNP label to have a distance-matrix row."""
    matrix_ids = _validated_ids(ids)
    observed_ids = _validated_ids(required_ids)
    matrix_set = set(matrix_ids)
    observed_set = set(observed_ids)
    missing = sorted(observed_set - matrix_set, key=stable_distance_label_key)
    extra = sorted(matrix_set - observed_set, key=stable_distance_label_key)
    if missing or (extra and not allow_extra):
        qualifier = "cover" if allow_extra else "match"
        raise ValueError(
            f"Distance-matrix ids must {qualifier} the observed CNP labels "
            f"(missing={missing!r}, extra={extra!r})."
        )
    return matrix_ids


def validate_directed_distance_matrix(ids, matrix):
    """Validate an ordered dissimilarity matrix without requiring symmetry."""
    values = _validated_ids(ids)
    try:
        array = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Directed distance matrix must contain numeric values.") from exc

    if array.ndim != 2:
        raise ValueError("Directed distance matrix must be two-dimensional.")
    rows, columns = array.shape
    if rows != columns:
        raise ValueError(
            f"Directed distance matrix must be square, got shape {array.shape}."
        )
    if len(values) != rows:
        raise ValueError(
            f"Directed distance matrix has {rows} rows but {len(values)} ids."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Directed distance matrix must contain only finite values.")
    if np.any(array < 0):
        raise ValueError(
            "Directed distance matrix must contain nonnegative dissimilarities."
        )
    if not np.all(np.diag(array) == 0):
        raise ValueError("Directed distance matrix diagonal must be exactly zero.")
    return values, np.array(array, copy=True)


@dataclass(frozen=True, init=False, eq=False)
class DirectedDistanceBundle:
    """Immutable ordered counts plus their validated symmetric minimum.

    The asymmetric matrix is deliberately not a ``DistanceMatrix``. Consumers
    that do not explicitly declare directed-distance support receive only
    ``minimum_matrix``.
    """

    ids: tuple
    directed_matrix: np.ndarray
    minimum_matrix: np.ndarray
    _provenance_json: str | None

    def __init__(self, ids, directed_matrix, *, minimum_matrix=None, provenance=None):
        ids, directed = validate_directed_distance_matrix(ids, directed_matrix)
        derived_minimum = np.minimum(directed, directed.T)
        _, derived_minimum = validate_distance_matrix(ids, derived_minimum)

        if minimum_matrix is not None:
            _, supplied_minimum = validate_distance_matrix(ids, minimum_matrix)
            if not np.array_equal(supplied_minimum, derived_minimum):
                raise ValueError(
                    "Directed-distance minimum does not equal min(C, C.T)."
                )

        directed.setflags(write=False)
        derived_minimum.setflags(write=False)
        object.__setattr__(self, "ids", tuple(ids))
        object.__setattr__(self, "directed_matrix", directed)
        object.__setattr__(self, "minimum_matrix", derived_minimum)
        try:
            provenance_json = (
                None
                if provenance is None
                else json.dumps(provenance, sort_keys=True, separators=(",", ":"))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Directed-distance provenance must be JSON-safe.") from exc
        object.__setattr__(self, "_provenance_json", provenance_json)

    @property
    def provenance(self):
        """Return a copy so callers cannot mutate bundle provenance in place."""
        return (
            None
            if self._provenance_json is None
            else json.loads(self._provenance_json)
        )

    def build_tree_kwargs(self):
        """Expose the symmetric view plus this explicit directed side input."""
        return {
            "dist_matrix_path": None,
            "inids": list(self.ids),
            "indm": np.array(self.minimum_matrix, copy=True),
            "directed_distance_bundle": self,
        }


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


def directed_bundle_from_ordered_cnp2cnp_matrices(
    forward_ids,
    forward_matrix,
    reverse_ids,
    reverse_matrix,
    *,
    provenance=None,
):
    """Recover ``C[u,v]`` from two opposite cnp2cnp matrix invocations."""
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

    size = len(forward_ids)
    directed = np.zeros((size, size), dtype=float)
    upper_i, upper_j = np.triu_indices(size, k=1)
    directed[upper_i, upper_j] = forward_matrix[upper_i, upper_j]
    directed[upper_j, upper_i] = reverse_aligned[upper_i, upper_j]
    return DirectedDistanceBundle(
        forward_ids,
        directed,
        provenance=provenance,
    )


def _stable_label_record(value):
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        payload = repr(value)
    return {"type": type_name, "payload": payload}


def stable_distance_label_key(value):
    """Return a deterministic, biopsy-order-independent label sort key."""
    record = _stable_label_record(value)
    return record["type"], record["payload"]


def stable_row_order_digest(row_order):
    """Hash an ordered, type-preserving representation of matrix labels."""
    records = [_stable_label_record(value) for value in row_order]
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def distance_input_cache_key(records, provenance):
    """Return a deterministic key for profiles plus distance semantics/tool identity."""
    normalized_records = []
    record_ids = []
    for record in records:
        try:
            cell_id, cnp = record
        except (TypeError, ValueError) as exc:
            raise ValueError("Distance cache records must be (id, CNP) pairs.") from exc
        record_ids.append(cell_id)
        normalized_records.append(
            {
                "id": _stable_label_record(cell_id),
                "cnp": str(cnp),
            }
        )
    _validated_ids(record_ids)

    identity_fields = (
        "metric",
        "distance_mode",
        "semantics_version",
        "symmetrization",
        "formula",
        "construction",
        "python_executable",
        "cnp2cnp_executable",
        "cnp2cnp_source_revision",
        "source_sha256",
        "row_order_sha256",
        "retains_directed_counts",
    )
    identity = {
        field: provenance.get(field)
        for field in identity_fields
        if field in provenance
    }
    payload = {
        "schema_version": DISTANCE_INPUT_CACHE_KEY_SCHEMA_VERSION,
        "tool_and_semantics": identity,
        "ordered_profiles": normalized_records,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(serialized).hexdigest()


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


def cnp2cnp_provenance(
    runfile=None,
    *,
    construction,
    python_executable=None,
    semantic_mode=CNP2CNP_SYMMETRIZATION,
    row_order=None,
    profile_count=None,
    retains_directed=False,
):
    """Build JSON-safe provenance for a newly constructed cnp2cnp matrix."""
    python_executable = str(python_executable or sys.executable)
    if semantic_mode == CNP2CNP_SYMMETRIZATION:
        semantics_version = CNP2CNP_SEMANTICS_VERSION
        symmetrization = CNP2CNP_SYMMETRIZATION
        formula = "min(d_any(u,v),d_any(v,u))"
        calls_per_pair = 2
    elif semantic_mode == CNP2CNP_ORDERED_TRIANGLE_FAST:
        semantics_version = CNP2CNP_ORDERED_TRIANGLE_FAST_SEMANTICS_VERSION
        symmetrization = "ordered_triangle_mirrored"
        formula = "d_any(x_i,x_j) mirrored for recorded row order i<j"
        calls_per_pair = 1
        if row_order is None:
            raise ValueError("Ordered-triangle provenance requires row_order.")
    else:
        raise ValueError(f"Unknown cnp2cnp semantic mode {semantic_mode!r}.")

    if construction == "trivial_singleton":
        calls_per_pair = 0

    provenance = {
        "schema_version": DISTANCE_PROVENANCE_SCHEMA_VERSION,
        "metric": "cnp2cnp",
        "distance_mode": CNP2CNP_DISTANCE,
        "semantics_version": semantics_version,
        "symmetrization": symmetrization,
        "formula": formula,
        "construction": construction,
        "directional_calls_per_unordered_pair": calls_per_pair,
        "python_executable": python_executable,
        "cnp2cnp_executable": None,
        "cnp2cnp_source_revision": None,
        "cnp2cnp_source_description": None,
        "source_sha256": {},
        "command_template": None,
        "tool_identity_policy": "source_sha256_plus_git_revision",
    }
    if profile_count is not None:
        profile_count = int(profile_count)
        if profile_count < 0:
            raise ValueError("profile_count must be nonnegative.")
        unordered_pairs = profile_count * (profile_count - 1) // 2
        provenance["profile_count"] = profile_count
        provenance["directional_transformation_count"] = (
            calls_per_pair * unordered_pairs
        )
        if construction in {
            "opposite_order_matrix_mode",
            "opposite_order_matrix_mode_directed_bundle",
        }:
            process_count = 0 if profile_count <= 1 else 2
        elif construction == "ordered_triangle_matrix_mode":
            process_count = 0 if profile_count <= 1 else 1
        elif construction == "bidirectional_pair_mode":
            process_count = calls_per_pair * unordered_pairs
        else:
            process_count = 0
        provenance["external_process_count"] = process_count

    if row_order is not None:
        row_order = list(row_order)
        provenance["row_order"] = [
            _stable_label_record(value)
            for value in row_order
        ]
        provenance["row_order_sha256"] = stable_row_order_digest(row_order)

    if retains_directed:
        provenance["directed_bundle_schema_version"] = (
            DIRECTED_DISTANCE_BUNDLE_SCHEMA_VERSION
        )
        provenance["retains_directed_counts"] = True
        provenance["directed_formula"] = "C[u,v] = d_any(u,v)"

    if runfile is None:
        return provenance

    executable = Path(runfile).expanduser().resolve()
    provenance["cnp2cnp_executable"] = str(executable)
    if construction in {
        "opposite_order_matrix_mode",
        "opposite_order_matrix_mode_directed_bundle",
    }:
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
    elif construction == "ordered_triangle_matrix_mode":
        provenance["command_template"] = [
            python_executable,
            str(executable),
            "-m",
            "matrix",
            "-d",
            CNP2CNP_DISTANCE,
            "-i",
            "<recorded-ordered-input.fa>",
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
    "CNP2CNP_ORDERED_TRIANGLE_FAST",
    "CNP2CNP_ORDERED_TRIANGLE_FAST_SEMANTICS_VERSION",
    "CNP2CNP_SEMANTICS_VERSION",
    "CNP2CNP_SYMMETRIZATION",
    "DIRECTED_DISTANCE_BUNDLE_SCHEMA_VERSION",
    "DISTANCE_INPUT_CACHE_KEY_SCHEMA_VERSION",
    "DISTANCE_PROVENANCE_SCHEMA_VERSION",
    "DirectedDistanceBundle",
    "cnp2cnp_provenance",
    "combine_ordered_cnp2cnp_matrices",
    "directed_bundle_from_ordered_cnp2cnp_matrices",
    "distance_input_cache_key",
    "minimum_bidirectional_distance",
    "parse_cnp2cnp_directional_distance",
    "parse_distance_label",
    "parse_labeled_distance_matrix",
    "stable_distance_label_key",
    "stable_row_order_digest",
    "validate_directed_distance_matrix",
    "validate_distance_label_coverage",
    "validate_distance_matrix",
]
