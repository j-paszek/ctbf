#!/usr/bin/env python3
"""Build the KTN152 empirical-feasibility figure as a self-contained SVG.

The profile heat map is rasterized inside the SVG because it contains more
than 900,000 values. All annotations, the biopsy timeline, and the inferred
tree remain vector elements.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import struct
import zlib
from collections import defaultdict, deque
from pathlib import Path


WIDTH = 1900
HEIGHT = 610

BIOPSY_COLORS = ("#377eb8", "#e69f00", "#cc6677")
COPY_COLOR = "#169c8d"
UNLABELED_COLOR = "#b9bec5"
INK = "#17202a"
MUTED = "#5f6b76"

COPY_NUMBER_COLORS = (
    "#252b3b",  # 0
    "#4f86c6",  # 1
    "#f4f1e8",  # 2
    "#f4c95d",  # 3
    "#ed8b45",  # 4
    "#d95f59",  # 5
    "#bd4e7a",  # 6
    "#8c4a8f",  # 7
    "#5e3c99",  # >=8
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--copykit-root",
        type=Path,
        required=True,
        help="Directory containing profiles_collapsed.tsv and bins.tsv.",
    )
    parser.add_argument(
        "--reconstruction",
        type=Path,
        required=True,
        help="Frozen M2 empirical reconstruction JSON.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_profiles(path: Path) -> tuple[list[dict[str, str]], list[str], list[list[int]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        bin_ids = [name for name in reader.fieldnames if name.startswith("bin_")]
        rows = list(reader)
    values = [[int(row[bin_id]) for bin_id in bin_ids] for row in rows]
    return rows, bin_ids, values


def _read_bins(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _png_data_url(rgb_rows: list[list[tuple[int, int, int]]]) -> str:
    height = len(rgb_rows)
    width = len(rgb_rows[0])
    raw = bytearray()
    for row in rgb_rows:
        if len(row) != width:
            raise ValueError("Inconsistent raster width")
        raw.append(0)
        for red, green, blue in row:
            raw.extend((red, green, blue))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 24,
    weight: str = "normal",
    anchor: str = "start",
    fill: str = INK,
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial,Helvetica,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{html.escape(value)}</text>'
    )


def _panel_label(x: float, letter: str, title: str) -> list[str]:
    return [
        _text(x, 42, letter, size=34, weight="bold"),
        _text(x + 38, 42, title, size=29, weight="bold"),
    ]


def _tree_layout(tree: dict, observed_count: int) -> tuple[dict[int, float], dict[int, int], dict[int, list[int]]]:
    nodes = {int(node["id"]): node for node in tree["nodes"]}
    children: dict[int, list[int]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in nodes}
    for edge in tree["links"]:
        parent = int(edge["source"])
        child = int(edge["target"])
        children[parent].append(child)
        indegree[child] += 1
    roots = [node_id for node_id, degree in indegree.items() if degree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one root, found {roots}")
    root = roots[0]

    minimum_observed: dict[int, int] = {}

    def key(node_id: int) -> int:
        if node_id in minimum_observed:
            return minimum_observed[node_id]
        descendants = children.get(node_id, [])
        if not descendants:
            result = node_id if node_id < observed_count else observed_count + node_id
        else:
            result = min(key(child) for child in descendants)
        minimum_observed[node_id] = result
        return result

    for node_id in children:
        children[node_id].sort(key=lambda child: (key(child), child))

    depth = {root: 0}
    queue = deque([root])
    while queue:
        parent = queue.popleft()
        for child in children.get(parent, []):
            depth[child] = depth[parent] + 1
            queue.append(child)

    leaves: list[int] = []

    def visit(node_id: int) -> None:
        if not children.get(node_id):
            leaves.append(node_id)
            return
        for child in children[node_id]:
            visit(child)

    visit(root)
    leaf_x = {node_id: float(index) for index, node_id in enumerate(leaves)}
    x_position: dict[int, float] = {}

    def assign_x(node_id: int) -> float:
        if node_id in leaf_x:
            x_position[node_id] = leaf_x[node_id]
        else:
            locations = [assign_x(child) for child in children[node_id]]
            x_position[node_id] = sum(locations) / len(locations)
        return x_position[node_id]

    assign_x(root)
    return x_position, depth, children


def build_figure(copykit_root: Path, reconstruction_path: Path, output: Path) -> None:
    profiles, bin_ids, matrix = _read_profiles(copykit_root / "profiles_collapsed.tsv")
    bins = _read_bins(copykit_root / "bins.tsv")
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    tree = reconstruction["tree"]
    summary = reconstruction["tree_summary"]
    audit = reconstruction["reconstruction_metadata"]["biopsy_layer_decision_audit"]

    if len(profiles) != 77 or len(bin_ids) != 12167:
        raise ValueError("The frozen KTN152 profile dimensions changed")
    if [row["bin_id"] for row in bins] != bin_ids:
        raise ValueError("Profile columns and genomic bins are not aligned")
    if summary["node_count"] != 148 or audit["copy_up_count"] != 31:
        raise ValueError("The frozen KTN152 reconstruction changed")

    order = sorted(
        range(len(profiles)),
        key=lambda index: (
            int(profiles[index]["biopsy_index"]),
            sum(matrix[index]) / len(matrix[index]),
            profiles[index]["state_label"],
        ),
    )
    palette = [_hex_rgb(color) for color in COPY_NUMBER_COLORS]
    raster = [
        [palette[min(value, 8)] for value in matrix[index]]
        for index in order
    ]
    heatmap_url = _png_data_url(raster)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<desc>KTN152 empirical feasibility: biopsy inventory, CopyKit-derived integer profiles, and CTBF M2 partial tree.</desc>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    # Panel A: longitudinal sampling inventory.
    parts.extend(_panel_label(20, "A", "Longitudinal input"))
    timeline_y = 226
    timeline_x = (68, 235, 410)
    parts.append(
        f'<line x1="{timeline_x[0]}" y1="{timeline_y}" x2="{timeline_x[-1]}" y2="{timeline_y}" '
        f'stroke="{MUTED}" stroke-width="4"/>'
    )
    biopsy_lines = (
        ("Pretreatment", "cycle 0", "28 cells", "28 occurrences"),
        ("On treatment", "after cycle 2", "16 cells", "16 occurrences"),
        ("Surgery", "after cycle 6", "35 cells", "33 occurrences"),
    )
    for index, (x, lines) in enumerate(zip(timeline_x, biopsy_lines)):
        parts.append(
            f'<circle cx="{x}" cy="{timeline_y}" r="11" fill="{BIOPSY_COLORS[index]}" '
            'stroke="white" stroke-width="3"/>'
        )
        parts.append(_text(x, 100, lines[0], size=22, weight="bold", anchor="middle"))
        parts.append(_text(x, 131, lines[1], size=21, anchor="middle", fill=MUTED))
        parts.append(_text(x, 284, lines[2], size=23, weight="bold", anchor="middle"))
        parts.append(_text(x, 314, lines[3], size=20, anchor="middle", fill=MUTED))
    parts.append(_text(230, 393, "79 retained cells", size=25, weight="bold", anchor="middle"))
    parts.append(_text(230, 427, "77 biopsy-indexed occurrences", size=23, anchor="middle"))
    parts.append(_text(230, 461, "76 distinct CNP states", size=23, anchor="middle"))
    parts.append(_text(230, 520, "Counts are not clone prevalences", size=20, anchor="middle", fill=MUTED))

    # Panel B: the full frozen CNP input, ordered without either reconstruction.
    panel_b_x = 475
    parts.extend(_panel_label(panel_b_x, "B", "CopyKit-derived integer CNPs"))
    heat_x, heat_y, heat_width, heat_height = 550, 92, 715, 352
    parts.append(
        f'<image x="{heat_x}" y="{heat_y}" width="{heat_width}" height="{heat_height}" '
        f'preserveAspectRatio="none" image-rendering="pixelated" xlink:href="{heatmap_url}"/>'
    )
    group_counts = (28, 16, 33)
    cumulative = 0
    for index, count in enumerate(group_counts):
        start = cumulative
        cumulative += count
        middle = heat_y + heat_height * (start + count / 2) / len(profiles)
        parts.append(
            f'<rect x="{heat_x - 12}" y="{heat_y + heat_height * start / len(profiles):.2f}" '
            f'width="7" height="{heat_height * count / len(profiles):.2f}" fill="{BIOPSY_COLORS[index]}"/>'
        )
        parts.append(_text(535, middle + 7, f"B{index + 1}", size=21, weight="bold", anchor="end"))
        if cumulative < len(profiles):
            separator_y = heat_y + heat_height * cumulative / len(profiles)
            parts.append(
                f'<line x1="{heat_x}" y1="{separator_y:.2f}" x2="{heat_x + heat_width}" '
                'y2="{:.2f}" stroke="white" stroke-width="3"/>'.format(separator_y)
            )

    chromosome_runs: list[tuple[str, int, int]] = []
    run_start = 0
    for index in range(1, len(bins) + 1):
        if index == len(bins) or bins[index]["chromosome"] != bins[run_start]["chromosome"]:
            chromosome_runs.append((bins[run_start]["chromosome"].removeprefix("chr"), run_start, index))
            run_start = index
    for chromosome, start, end in chromosome_runs:
        x_mid = heat_x + heat_width * (start + end) / (2 * len(bins))
        parts.append(_text(x_mid, 474, chromosome, size=15, anchor="middle", fill=MUTED))
        if end < len(bins):
            x_end = heat_x + heat_width * end / len(bins)
            parts.append(
                f'<line x1="{x_end:.2f}" y1="{heat_y}" x2="{x_end:.2f}" y2="{heat_y + heat_height}" '
                'stroke="#d2d5d9" stroke-width="0.8"/>'
            )
    parts.append(_text(907, 511, "Chromosome", size=20, weight="bold", anchor="middle"))
    parts.append(_text(907, 546, "Absolute copy-number scale unresolved", size=20, anchor="middle", fill=MUTED))

    legend_x = 620
    for index, (label, color) in enumerate(zip(("0", "1", "2", "3", "4", "5", "6", "7", "≥8"), COPY_NUMBER_COLORS)):
        x = legend_x + index * 63
        parts.append(f'<rect x="{x}" y="565" width="25" height="25" fill="{color}" stroke="#777" stroke-width="0.5"/>')
        parts.append(_text(x + 32, 585, label, size=18))
    parts.append(_text(550, 585, "CN", size=19, weight="bold", anchor="end"))

    # Panel C: inferred M2 partial tree.
    panel_c_x = 1305
    parts.extend(_panel_label(panel_c_x, "C", "Biopsy-guided partial tree"))
    observed_count = len(profiles)
    x_position, depth, children = _tree_layout(tree, observed_count)
    nodes = {int(node["id"]): node for node in tree["nodes"]}
    leaf_count = sum(not children.get(node_id) for node_id in nodes)
    if leaf_count < 2:
        raise ValueError("Tree layout requires at least two leaves")
    x_min, x_max = min(x_position.values()), max(x_position.values())
    plot_left, plot_right = 1325, 1875
    plot_top, plot_bottom = 82, 446
    max_depth = max(depth.values())

    def sx(value: float) -> float:
        return plot_left + (value - x_min) * (plot_right - plot_left) / (x_max - x_min)

    def sy(value: int) -> float:
        return plot_top + value * (plot_bottom - plot_top) / max_depth

    for parent, descendants in children.items():
        for child in descendants:
            parts.append(
                f'<line x1="{sx(x_position[parent]):.2f}" y1="{sy(depth[parent]):.2f}" '
                f'x2="{sx(x_position[child]):.2f}" y2="{sy(depth[child]):.2f}" '
                'stroke="#9da4ab" stroke-width="1.4" stroke-linecap="round"/>'
            )

    biopsy_queues: dict[str, deque[int]] = defaultdict(deque)
    for row in profiles:
        biopsy_queues[row["state_label"]].append(int(row["biopsy_index"]))
    observed_biopsy: dict[int, int] = {}
    for node_id in range(observed_count):
        state = nodes[node_id]["cell_id"]
        if not biopsy_queues[state]:
            raise ValueError("Observed node could not be assigned to a biopsy")
        observed_biopsy[node_id] = biopsy_queues[state].popleft()

    for node_id in sorted(nodes, key=lambda item: depth[item], reverse=True):
        if node_id < observed_count:
            fill = BIOPSY_COLORS[observed_biopsy[node_id]]
            radius = 4.8
            stroke = "white"
            stroke_width = 1.1
        elif node_id <= 107:
            fill = COPY_COLOR
            radius = 4.4
            stroke = "white"
            stroke_width = 1.0
        else:
            fill = UNLABELED_COLOR
            radius = 3.9
            stroke = "white"
            stroke_width = 0.8
        parts.append(
            f'<circle cx="{sx(x_position[node_id]):.2f}" cy="{sy(depth[node_id]):.2f}" '
            f'r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    parts.append(_text(1600, 482, "148 vertices · 40 unlabeled · 31 copy-up", size=20, anchor="middle"))
    parts.append(_text(1600, 512, "adaptive radii: 21, 11", size=21, weight="bold", anchor="middle"))
    legend_items = (
        (BIOPSY_COLORS[0], "B1"),
        (BIOPSY_COLORS[1], "B2"),
        (BIOPSY_COLORS[2], "B3"),
        (COPY_COLOR, "copy-up"),
        (UNLABELED_COLOR, "unlabeled"),
    )
    start_x = 1350
    for index, (color, label) in enumerate(legend_items):
        x = start_x + index * 112
        parts.append(f'<circle cx="{x}" cy="555" r="7" fill="{color}"/>')
        parts.append(_text(x + 13, 562, label, size=18))
    parts.append(_text(1600, 595, "Topology has no empirical ground truth", size=19, anchor="middle", fill=MUTED))

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = _arguments()
    build_figure(args.copykit_root, args.reconstruction, args.output)
