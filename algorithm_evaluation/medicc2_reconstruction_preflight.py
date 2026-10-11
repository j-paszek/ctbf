"""Bounded MEDICC2 topology-only preflight and CTBF projection helpers.

The module keeps MEDICC2 behind a subprocess boundary.  Its input contains
only observable state labels, CNPs, and a declared chromosome layout; truth
node identities are neither accepted nor consulted.  MEDICC2 taxa represent
observation occurrences, so equal profiles and recurrent CTBF state labels are
preserved as distinct leaves.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import networkx as nx

from algorithm_evaluation.paper_pipeline_contract import (
    file_sha256,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import (
    PeakProcessTreeRss,
    validate_reconstruction_input,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    canonical_topology_digest,
)
from evaluation_contract import validate_rooted_labeled_tree


PREFLIGHT_SCHEMA_VERSION = "ctbf-v5-medicc2-reconstruction-preflight-v4"
PROJECTION_VERSION = "ctbf-medicc2-diploid-outgroup-removal-v1"
INPUT_ENCODING_VERSION = "ctbf-medicc2-total-cn-occurrence-tsv-v3"

PINNED_MEDICC2_REVISION = "3f1b8a69c1ad9ea1b4ec024faf69f5f3adcecaea"
PINNED_FSTLIB_REVISION = "deb4015f52d1f0b01b1e3ce58bce19973f418cc0"
PINNED_MEDICC2_VERSION = "1.1.2"
ARTIFICIAL_NORMAL_TAXON = "ctbf_artificial_diploid_normal"
MEDICC2_MAX_COPY_NUMBER = 8
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576

EXPECTED_TOPOLOGY_ONLY_OUTPUT_SUFFIXES = (
    "branch_lengths.tsv",
    "final_tree.new",
    "final_tree.xml",
    "pairwise_distances.tsv",
    "summary.tsv",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "test" / "fixtures" / "medicc2_total_cn"


@dataclass(frozen=True)
class ExportedOccurrenceInput:
    path: Path
    taxon_to_state_label: dict[str, Any]
    taxon_to_observation: dict[str, dict[str, int]]
    occurrence_count: int
    non_diploid_occurrence_count: int
    unique_state_label_count: int
    cnp_length: int
    chromosome_count: int


@dataclass(frozen=True)
class FixtureCase:
    case_id: str
    input_path: Path
    taxon_to_state_label: Mapping[str, Any]
    expected_distances: tuple[tuple[str, str, float], ...]
    segment_count: int


FIXTURE_CASES = (
    FixtureCase(
        case_id="simple",
        input_path=FIXTURE_ROOT / "simple.tsv",
        taxon_to_state_label={"A": "A", "B": "B", "C": "C", "D": "D"},
        expected_distances=(("A", "B", 1.0), ("C", "D", 1.0), ("A", "C", 2.0)),
        segment_count=6,
    ),
    FixtureCase(
        case_id="repeated",
        input_path=FIXTURE_ROOT / "repeated.tsv",
        taxon_to_state_label={
            "A1": "A",
            "A2": "A",
            "B": "B",
            "C": "C",
            "D": "D",
        },
        expected_distances=(("A1", "A2", 0.0), ("A1", "B", 1.0)),
        segment_count=6,
    ),
    FixtureCase(
        case_id="all_diploid",
        input_path=FIXTURE_ROOT / "all_diploid.tsv",
        taxon_to_state_label={"N1": "N", "N2": "N", "N3": "N"},
        expected_distances=(
            (ARTIFICIAL_NORMAL_TAXON, "N1", 0.0),
            ("N1", "N2", 0.0),
            ("N2", "N3", 0.0),
        ),
        segment_count=6,
    ),
    FixtureCase(
        case_id="plus2",
        input_path=FIXTURE_ROOT / "plus2.tsv",
        taxon_to_state_label={
            "U1": "U1",
            "U2": "U2",
            "V1": "V1",
            "V2": "V2",
        },
        expected_distances=(
            (ARTIFICIAL_NORMAL_TAXON, "U1", 1.0),
            (ARTIFICIAL_NORMAL_TAXON, "U2", 2.0),
            ("U1", "U2", 1.0),
            (ARTIFICIAL_NORMAL_TAXON, "V2", 2.0),
        ),
        segment_count=6,
    ),
)


def _write_bounded_scale_fixture(destination: Path) -> FixtureCase:
    """Create a deterministic 16-taxon, 100-segment runtime smoke case."""
    profiles: dict[str, tuple[int, ...]] = {}
    for sample_index in range(16):
        profile = [2] * 100
        for event_index in range(1 + sample_index % 4):
            start = (13 * sample_index + 29 * event_index) % 85
            width = 6 + (sample_index + 3 * event_index) % 12
            delta = (
                2
                if sample_index in {7, 15} and event_index == 0
                else (-1 if (sample_index + event_index) % 3 == 0 else 1)
            )
            for position in range(start, min(start + width, len(profile))):
                profile[position] += delta
        profiles[f"S{sample_index:02d}"] = tuple(profile)
    if len(set(profiles.values())) != len(profiles):
        raise AssertionError("The frozen MEDICC2 scale profiles must remain distinct.")

    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("sample_id", "chrom", "start", "end", "cn"))
        for taxon, profile in profiles.items():
            for position, copy_number in enumerate(profile):
                writer.writerow((taxon, "1", position, position + 1, copy_number))
    return FixtureCase(
        case_id="scale100_n16",
        input_path=destination,
        taxon_to_state_label={taxon: taxon for taxon in profiles},
        expected_distances=(),
        segment_count=100,
    )


def _write_fixture_variant(
    source: FixtureCase,
    destination: Path,
    *,
    case_id: str,
    reverse_taxon_order: bool = False,
    taxon_prefix: str | None = None,
) -> FixtureCase:
    with source.input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    if fieldnames != ("sample_id", "chrom", "start", "end", "cn"):
        raise ValueError("Frozen MEDICC2 fixture columns changed.")
    taxon_order = list(dict.fromkeys(row["sample_id"] for row in rows))
    if reverse_taxon_order:
        taxon_order.reverse()
    rows_by_taxon = {
        taxon: [row for row in rows if row["sample_id"] == taxon]
        for taxon in taxon_order
    }
    relabel = {
        taxon: (taxon if taxon_prefix is None else f"{taxon_prefix}{index:02d}")
        for index, taxon in enumerate(taxon_order)
    }
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for taxon in taxon_order:
            for row in rows_by_taxon[taxon]:
                writer.writerow({**row, "sample_id": relabel[taxon]})
    return FixtureCase(
        case_id=case_id,
        input_path=destination,
        taxon_to_state_label={
            relabel[taxon]: source.taxon_to_state_label[taxon]
            for taxon in taxon_order
        },
        expected_distances=(
            source.expected_distances
            if taxon_prefix is None
            else tuple(
                (
                    relabel.get(left, left),
                    relabel.get(right, right),
                    expected,
                )
                for left, right, expected in source.expected_distances
            )
        ),
        segment_count=source.segment_count,
    )


def _canonical_label(value: Any) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError("Observable state labels must canonicalize to non-empty strings.")
    return label


def _validate_chromosome_layout(
    chromosome_by_bin: Sequence[str] | None,
    cnp_length: int,
) -> tuple[str, ...]:
    if chromosome_by_bin is None:
        return ("1",) * cnp_length
    chromosomes = tuple(str(value).strip() for value in chromosome_by_bin)
    if len(chromosomes) != cnp_length or any(not value for value in chromosomes):
        raise ValueError("The chromosome layout must name every CNP bin exactly once.")
    closed = set()
    previous = None
    for chromosome in chromosomes:
        if chromosome != previous:
            if chromosome in closed:
                raise ValueError("Each chromosome must occupy one contiguous bin block.")
            if previous is not None:
                closed.add(previous)
            previous = chromosome
    return chromosomes


def export_observation_occurrences(
    reconstruction_input: Mapping[str, Any],
    destination: Path | str,
    *,
    chromosome_by_bin: Sequence[str] | None = None,
) -> ExportedOccurrenceInput:
    """Write MEDICC2 total-CN TSV without deduplicating occurrences."""
    validate_reconstruction_input(reconstruction_input)
    path = Path(destination).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite MEDICC2 input: {path}")
    if not path.parent.is_dir():
        raise ValueError(f"MEDICC2 input parent does not exist: {path.parent}")

    flattened = []
    cnp_length = None
    for level in reconstruction_input["levels"]:
        biopsy_level = int(level["biopsy_level"])
        for within_level_index, state in enumerate(level["states"]):
            cnp = tuple(state["cnp"])
            if cnp_length is None:
                cnp_length = len(cnp)
            if not cnp or len(cnp) != cnp_length:
                raise ValueError(
                    "Every MEDICC2 occurrence must have one common non-empty CNP length."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > MEDICC2_MAX_COPY_NUMBER
                for value in cnp
            ):
                raise ValueError(
                    "MEDICC2 total-CN input requires integer states in [0, 8]; "
                    "values above 8 must be rejected rather than capped."
                )
            flattened.append((biopsy_level, within_level_index, state, cnp))

    if cnp_length is None or len(flattened) < 2:
        raise ValueError("MEDICC2 tree inference requires at least two observed taxa.")
    label_by_cnp: dict[tuple[int, ...], str] = {}
    for _level, _index, state, cnp in flattened:
        label = _canonical_label(state["state_label"])
        if cnp in label_by_cnp and label_by_cnp[cnp] != label:
            raise ValueError(
                "Equal observable CNPs must share one canonical state label."
            )
        label_by_cnp[cnp] = label
    flattened.sort(key=lambda row: (row[3], row[0], row[1]))
    non_diploid_count = sum(
        any(copy_number != 2 for copy_number in cnp)
        for _level, _index, _state, cnp in flattened
    )
    chromosomes = _validate_chromosome_layout(chromosome_by_bin, cnp_length)
    local_positions = []
    previous = None
    local_index = 0
    for chromosome in chromosomes:
        if chromosome != previous:
            local_index = 0
            previous = chromosome
        local_positions.append(local_index)
        local_index += 1

    taxon_to_state_label: dict[str, Any] = {}
    taxon_to_observation: dict[str, dict[str, int]] = {}
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("sample_id", "chrom", "start", "end", "cn"))
        for occurrence_index, (
            biopsy_level,
            within_level_index,
            state,
            cnp,
        ) in enumerate(flattened):
            taxon = f"ctbf_o{occurrence_index:06d}"
            if taxon in taxon_to_state_label:
                raise AssertionError("Generated duplicate MEDICC2 occurrence taxon.")
            taxon_to_state_label[taxon] = state["state_label"]
            taxon_to_observation[taxon] = {
                "biopsy_level": biopsy_level,
                "within_level_index": within_level_index,
                "pooled_occurrence_index": occurrence_index,
            }
            for chromosome, start, copy_number in zip(chromosomes, local_positions, cnp):
                writer.writerow((taxon, chromosome, start, start + 1, copy_number))

    canonical_labels = [
        _canonical_label(value) for value in taxon_to_state_label.values()
    ]
    return ExportedOccurrenceInput(
        path=path,
        taxon_to_state_label=taxon_to_state_label,
        taxon_to_observation=taxon_to_observation,
        occurrence_count=len(flattened),
        non_diploid_occurrence_count=non_diploid_count,
        unique_state_label_count=len(set(canonical_labels)),
        cnp_length=cnp_length,
        chromosome_count=len(set(chromosomes)),
    )


def topology_only_command(
    command_prefix: Sequence[Path | str],
    input_path: Path | str,
    output_directory: Path | str,
    *,
    prefix: str,
    normal_taxon: str = ARTIFICIAL_NORMAL_TAXON,
    n_cores: int = 1,
    allow_wgd: bool = False,
) -> list[str]:
    """Return the exact headless MEDICC2 partial-tree command."""
    if not command_prefix:
        raise ValueError("A MEDICC2 executable command is required.")
    if n_cores < 1:
        raise ValueError("MEDICC2 core count must be positive.")
    if not prefix or any(character in prefix for character in "/\\"):
        raise ValueError("MEDICC2 output prefix must be a non-empty path-safe name.")
    command = [
        *(str(value) for value in command_prefix),
        str(Path(input_path).expanduser().resolve()),
        str(Path(output_directory).expanduser().resolve()),
        "--input-type",
        "tsv",
        "--total-copy-numbers",
        "--input-allele-columns",
        "cn",
        "--topology-only",
        "--normal-name",
        normal_taxon,
        "--maxcn",
        str(MEDICC2_MAX_COPY_NUMBER),
        "--plot",
        "none",
        "--no-plot-tree",
        "--n-cores",
        str(n_cores),
        "--prefix",
        prefix,
    ]
    if not allow_wgd:
        command.append("--no-wgd")
    return command


def read_pairwise_distances(path: Path | str) -> dict[str, dict[str, float]]:
    """Read and strictly validate MEDICC2's symmetric distance table."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("MEDICC2 distance table is empty.")
    labels = rows[0][1:]
    if len(labels) != len(set(labels)):
        raise ValueError("MEDICC2 distance columns are not unique.")
    matrix: dict[str, dict[str, float]] = {}
    for row in rows[1:]:
        if len(row) != len(labels) + 1:
            raise ValueError("MEDICC2 distance row has the wrong width.")
        label = row[0]
        if label in matrix:
            raise ValueError("MEDICC2 distance rows are not unique.")
        values = [float(value) for value in row[1:]]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("MEDICC2 distances must be finite and nonnegative.")
        matrix[label] = dict(zip(labels, values))
    if set(matrix) != set(labels):
        raise ValueError("MEDICC2 distance row and column labels differ.")
    for left in labels:
        if not math.isclose(matrix[left][left], 0.0, abs_tol=1e-12):
            raise ValueError("MEDICC2 distance diagonal must be zero.")
        for right in labels:
            if not math.isclose(
                matrix[left][right], matrix[right][left], rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("MEDICC2 distance table is not symmetric.")
    return matrix


def _clade_children(clade: ET.Element) -> list[ET.Element]:
    return list(clade.findall("{*}clade"))


def _clade_name(clade: ET.Element) -> str | None:
    value = clade.findtext("{*}name")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _branch_length(clade: ET.Element) -> float | None:
    value = clade.findtext("{*}branch_length")
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("MEDICC2 emitted a non-finite branch length.")
    return parsed


def project_phyloxml_to_partial_tree(
    path: Path | str,
    taxon_to_state_label: Mapping[str, Any],
    *,
    normal_taxon: str = ARTIFICIAL_NORMAL_TAXON,
) -> tuple[nx.DiGraph, Any, dict[str, Any]]:
    """Remove MEDICC2's normal outgroup and label only observed leaves."""
    if not taxon_to_state_label:
        raise ValueError("At least one observed MEDICC2 taxon is required.")
    if normal_taxon in taxon_to_state_label:
        raise ValueError("The artificial normal taxon is reserved.")

    document = ET.parse(Path(path))
    phylogenies = list(document.getroot().iterfind(".//{*}phylogeny"))
    if len(phylogenies) != 1 or phylogenies[0].get("rooted") != "true":
        raise ValueError("MEDICC2 PhyloXML must contain one explicitly rooted phylogeny.")
    top_clades = _clade_children(phylogenies[0])
    if len(top_clades) != 1:
        raise ValueError("MEDICC2 PhyloXML must contain one outer root clade.")
    outer_root = top_clades[0]
    outer_children = _clade_children(outer_root)
    normal_children = [
        child
        for child in outer_children
        if _clade_name(child) == normal_taxon and not _clade_children(child)
    ]
    tumor_children = [child for child in outer_children if child not in normal_children]
    if len(normal_children) != 1 or len(tumor_children) != 1:
        raise ValueError(
            "The rooted MEDICC2 tree must place one artificial normal leaf beside "
            "one tumor subtree."
        )
    tumor_root = tumor_children[0]

    terminal_names = []
    for clade in tumor_root.iter():
        if clade.tag.endswith("clade") and not _clade_children(clade):
            terminal_names.append(_clade_name(clade))
    if None in terminal_names or set(terminal_names) != set(taxon_to_state_label):
        raise ValueError("MEDICC2 terminal taxa do not exactly match input occurrences.")
    if len(terminal_names) != len(set(terminal_names)):
        raise ValueError("MEDICC2 emitted a duplicate terminal taxon identifier.")

    tree = nx.DiGraph()
    node_counter = 0

    def visit(clade: ET.Element) -> str:
        nonlocal node_counter
        node_id = f"medicc2_node_{node_counter:06d}"
        node_counter += 1
        children = _clade_children(clade)
        name = _clade_name(clade)
        if children:
            if name in taxon_to_state_label:
                raise ValueError("MEDICC2 placed an observed occurrence at an internal node.")
            tree.add_node(node_id, cell_id=None, medicc2_internal_name=name)
        else:
            tree.add_node(
                node_id,
                cell_id=taxon_to_state_label[name],
                medicc2_taxon=name,
            )
        for child in children:
            child_id = visit(child)
            attributes = {}
            length = _branch_length(child)
            if length is not None:
                attributes["weight"] = length
            tree.add_edge(node_id, child_id, **attributes)
        return node_id

    root = visit(tumor_root)
    if not nx.is_arborescence(tree) or tree.in_degree(root) != 0:
        raise ValueError("Projected MEDICC2 output is not one rooted arborescence.")
    validated = validate_rooted_labeled_tree(tree, "reconstructed_tree")
    labels = [_canonical_label(value) for value in taxon_to_state_label.values()]
    metadata = {
        "projection_version": PROJECTION_VERSION,
        "normal_taxon_removed": normal_taxon,
        "outer_root_removed": True,
        "observed_occurrence_count": len(taxon_to_state_label),
        "unique_observed_label_count": len(set(labels)),
        "repeated_observed_label_occurrence_count": len(labels) - len(set(labels)),
        "projected_tree": validated.metadata,
    }
    return tree, root, metadata


def inspect_source_checkout(source_root: Path | str) -> dict[str, Any]:
    """Return exact, read-only provenance for the external MEDICC2 checkout."""
    root = Path(source_root).expanduser().resolve()
    required = (
        root / "medicc2",
        root / "medicc" / "core.py",
        root / "medicc" / "io.py",
        root / "LICENSE",
    )
    if not all(path.is_file() for path in required):
        raise ValueError(f"Not a complete MEDICC2 source checkout: {root}")

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    submodule = git("submodule", "status", "submodule-fstlib")
    submodule_parts = submodule.lstrip("-+ ").split()
    submodule_revision = submodule_parts[0] if submodule_parts else None
    version_text = (root / "medicc" / "_version.py").read_text(encoding="utf-8")
    version = version_text.split("=", 1)[1].strip().strip("'\"")
    return {
        "source_root": str(root),
        "revision": revision,
        "working_tree_clean": not status,
        "submodule_revision": submodule_revision,
        "version": version,
        "license": "GPL-3.0",
        "sha256": {
            path.relative_to(root).as_posix(): file_sha256(path) for path in required
        },
    }


def _bounded_tail(value: str, maximum_characters: int = 4000) -> str:
    return value[-maximum_characters:]


def _run_fixture_case(
    case: FixtureCase,
    *,
    command_prefix: Sequence[Path | str],
    output_root: Path,
    timeout_seconds: int,
    rss_limit_bytes: int,
    capture_limit_bytes: int,
    require_recursive_rss: bool,
) -> dict[str, Any]:
    case_root = output_root / case.case_id
    case_root.mkdir()
    command = topology_only_command(
        command_prefix,
        case.input_path,
        case_root,
        prefix=case.case_id,
    )
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(output_root / "runtime-cache" / "matplotlib"))
    environment.setdefault("XDG_CACHE_HOME", str(output_root / "runtime-cache" / "xdg"))
    started = time.perf_counter_ns()
    with PeakProcessTreeRss() as memory:
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            completed = None
            timeout_error = error
        else:
            timeout_error = None
    wall_time_ns = time.perf_counter_ns() - started
    memory_report = memory.report()
    if timeout_error is not None:
        return {
            "case_id": case.case_id,
            "status": "failure",
            "failure": {"code": "timeout", "message": str(timeout_error)},
            "wall_time_ns": wall_time_ns,
            "memory": memory_report,
            "command": command,
        }
    assert completed is not None
    stdout_bytes = len(completed.stdout.encode("utf-8"))
    stderr_bytes = len(completed.stderr.encode("utf-8"))
    if stdout_bytes > capture_limit_bytes or stderr_bytes > capture_limit_bytes:
        return {
            "case_id": case.case_id,
            "status": "failure",
            "failure": {
                "code": "capture_limit_exceeded",
                "stdout_bytes": stdout_bytes,
                "stderr_bytes": stderr_bytes,
            },
            "wall_time_ns": wall_time_ns,
            "memory": memory_report,
            "command": command,
        }
    if (
        memory_report["peak_rss_bytes"] is not None
        and memory_report["peak_rss_bytes"] > rss_limit_bytes
    ):
        return {
            "case_id": case.case_id,
            "status": "failure",
            "failure": {
                "code": "rss_limit_exceeded",
                "peak_rss_bytes": memory_report["peak_rss_bytes"],
                "rss_limit_bytes": rss_limit_bytes,
            },
            "wall_time_ns": wall_time_ns,
            "memory": memory_report,
            "command": command,
        }
    if (
        require_recursive_rss
        and memory_report["method"] != "psutil_process_tree_rss_10ms"
    ):
        return {
            "case_id": case.case_id,
            "status": "failure",
            "failure": {
                "code": "recursive_rss_unavailable",
                "memory_method": memory_report["method"],
            },
            "wall_time_ns": wall_time_ns,
            "memory": memory_report,
            "command": command,
        }
    if completed.returncode != 0:
        return {
            "case_id": case.case_id,
            "status": "failure",
            "failure": {
                "code": "external_process_failed",
                "returncode": completed.returncode,
                "stdout_tail": _bounded_tail(completed.stdout),
                "stderr_tail": _bounded_tail(completed.stderr),
            },
            "wall_time_ns": wall_time_ns,
            "memory": memory_report,
            "command": command,
        }

    tree_path = case_root / f"{case.case_id}_final_tree.xml"
    distance_path = case_root / f"{case.case_id}_pairwise_distances.tsv"
    try:
        expected_names = {
            f"{case.case_id}_{suffix}"
            for suffix in EXPECTED_TOPOLOGY_ONLY_OUTPUT_SUFFIXES
        }
        entries = list(case_root.iterdir())
        observed_names = {path.name for path in entries}
        if observed_names != expected_names or not all(
            path.is_file() for path in entries
        ):
            raise ValueError(
                "MEDICC2 topology-only output inventory changed: "
                f"expected {sorted(expected_names)}, observed {sorted(observed_names)}."
            )
        tree, root, projection = project_phyloxml_to_partial_tree(
            tree_path,
            case.taxon_to_state_label,
        )
        distances = read_pairwise_distances(distance_path)
        expected_taxa = set(case.taxon_to_state_label) | {
            ARTIFICIAL_NORMAL_TAXON
        }
        if set(distances) != expected_taxa:
            raise ValueError("MEDICC2 preflight distance taxa changed.")
        for left, right, expected in case.expected_distances:
            observed = distances[left][right]
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"MEDICC2 preflight distance {left}/{right} was {observed}, "
                    f"expected {expected}."
                )
    except Exception as error:
        return {
            "case_id": case.case_id,
            "status": "failure",
            "failure": {
                "code": "invalid_external_output",
                "type": type(error).__name__,
                "message": str(error),
            },
            "wall_time_ns": wall_time_ns,
            "memory": memory_report,
            "command": command,
        }
    return {
        "case_id": case.case_id,
        "status": "success",
        "wall_time_ns": wall_time_ns,
        "memory": memory_report,
        "command": command,
        "input_sha256": file_sha256(case.input_path),
        "input_taxon_count": len(case.taxon_to_state_label),
        "input_segment_count": case.segment_count,
        "projection": projection,
        "projected_root": root,
        "projected_node_count": tree.number_of_nodes(),
        "projected_edge_count": tree.number_of_edges(),
        "canonical_topology_digest": canonical_topology_digest(tree),
        "checked_distances": [
            {"left": left, "right": right, "value": distances[left][right]}
            for left, right, _expected in case.expected_distances
        ],
        "stdout_tail": _bounded_tail(completed.stdout),
        "stderr_tail": _bounded_tail(completed.stderr),
    }


def run_fixture_preflight(
    *,
    command_prefix: Sequence[Path | str],
    source_root: Path | str,
    output_root: Path | str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    require_recursive_rss: bool = False,
) -> dict[str, Any]:
    """Execute semantic, stability, and bounded 100-segment scale cases."""
    if timeout_seconds < 1:
        raise ValueError("MEDICC2 timeout must be positive.")
    if rss_limit_bytes < 1 or capture_limit_bytes < 1:
        raise ValueError("MEDICC2 RSS and capture limits must be positive.")
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite MEDICC2 preflight root: {root}")
    source = inspect_source_checkout(source_root)
    if source["revision"] != PINNED_MEDICC2_REVISION:
        raise ValueError("MEDICC2 source revision differs from the preflight pin.")
    if source["submodule_revision"] != PINNED_FSTLIB_REVISION:
        raise ValueError("MEDICC2 fstlib revision differs from the preflight pin.")
    if source["version"] != PINNED_MEDICC2_VERSION or not source["working_tree_clean"]:
        raise ValueError("MEDICC2 source must have the pinned version and a clean tree.")

    root.mkdir(parents=True)
    scale_case = _write_bounded_scale_fixture(root / "scale100_n16_input.tsv")
    simple_repeat = FixtureCase(
        case_id="simple_repeat",
        input_path=FIXTURE_CASES[0].input_path,
        taxon_to_state_label=FIXTURE_CASES[0].taxon_to_state_label,
        expected_distances=FIXTURE_CASES[0].expected_distances,
        segment_count=FIXTURE_CASES[0].segment_count,
    )
    simple_permuted = _write_fixture_variant(
        FIXTURE_CASES[0],
        root / "simple_permuted_input.tsv",
        case_id="simple_permuted",
        reverse_taxon_order=True,
    )
    simple_relabelled = _write_fixture_variant(
        FIXTURE_CASES[0],
        root / "simple_relabelled_input.tsv",
        case_id="simple_relabelled",
        reverse_taxon_order=True,
        taxon_prefix="renamed_",
    )
    records = [
        _run_fixture_case(
            case,
            command_prefix=command_prefix,
            output_root=root,
            timeout_seconds=timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
            capture_limit_bytes=capture_limit_bytes,
            require_recursive_rss=require_recursive_rss,
        )
        for case in (
            *FIXTURE_CASES,
            scale_case,
            simple_repeat,
            simple_permuted,
            simple_relabelled,
        )
    ]
    by_id = {record["case_id"]: record for record in records}
    comparison_ids = (
        "simple",
        "simple_repeat",
        "simple_permuted",
        "simple_relabelled",
    )
    stability_available = all(
        by_id[case_id]["status"] == "success" for case_id in comparison_ids
    )
    stability = {
        "deterministic_rerun_same_topology": (
            stability_available
            and by_id["simple"]["canonical_topology_digest"]
            == by_id["simple_repeat"]["canonical_topology_digest"]
        ),
        "raw_taxon_row_permutation_same_topology": (
            stability_available
            and by_id["simple"]["canonical_topology_digest"]
            == by_id["simple_permuted"]["canonical_topology_digest"]
        ),
        "taxon_relabelling_same_topology_after_inverse_map": (
            stability_available
            and by_id["simple"]["canonical_topology_digest"]
            == by_id["simple_relabelled"]["canonical_topology_digest"]
        ),
        "permutation_and_relabelling_are_diagnostics_not_admission_filters": True,
    }
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": (
            "success"
            if all(row["status"] == "success" for row in records)
            and stability["deterministic_rerun_same_topology"]
            else "failure"
        ),
        "scientific_role": "bounded_technical_preflight_not_accuracy_evidence",
        "input_encoding": INPUT_ENCODING_VERSION,
        "projection": PROJECTION_VERSION,
        "medicc2_source": source,
        "timeout_seconds_per_case": timeout_seconds,
        "rss_limit_bytes_per_case": rss_limit_bytes,
        "capture_limit_bytes_per_stream": capture_limit_bytes,
        "recursive_rss_required": require_recursive_rss,
        "wgd_policy": "disabled_for_non_wgd_ctbf_toys",
        "ancestor_policy": "topology_only_internal_nodes_unlabeled",
        "truth_identity_used": False,
        "stability_diagnostics": stability,
        "records": records,
    }
    write_json_atomic(root / "preflight_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medicc2-executable", required=True)
    parser.add_argument(
        "--medicc2-python",
        help="Optional Python interpreter used to launch the MEDICC2 script.",
    )
    parser.add_argument(
        "--source-root",
        default=str(Path("~/Work/Medicc").expanduser()),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument(
        "--capture-limit-bytes", type=int, default=DEFAULT_CAPTURE_LIMIT_BYTES
    )
    parser.add_argument("--require-recursive-rss", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command_prefix = [arguments.medicc2_executable]
    if arguments.medicc2_python:
        command_prefix.insert(0, arguments.medicc2_python)
    report = run_fixture_preflight(
        command_prefix=command_prefix,
        source_root=arguments.source_root,
        output_root=arguments.output_root,
        timeout_seconds=arguments.timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
        require_recursive_rss=arguments.require_recursive_rss,
    )
    print(
        f"MEDICC2 preflight {report['status']}: "
        f"{sum(row['status'] == 'success' for row in report['records'])}/"
        f"{len(report['records'])} cases succeeded."
    )
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARTIFICIAL_NORMAL_TAXON",
    "DEFAULT_CAPTURE_LIMIT_BYTES",
    "DEFAULT_RSS_LIMIT_BYTES",
    "ExportedOccurrenceInput",
    "FIXTURE_CASES",
    "INPUT_ENCODING_VERSION",
    "MEDICC2_MAX_COPY_NUMBER",
    "PINNED_FSTLIB_REVISION",
    "PINNED_MEDICC2_REVISION",
    "PREFLIGHT_SCHEMA_VERSION",
    "PROJECTION_VERSION",
    "export_observation_occurrences",
    "inspect_source_checkout",
    "project_phyloxml_to_partial_tree",
    "read_pairwise_distances",
    "run_fixture_preflight",
    "topology_only_command",
]
