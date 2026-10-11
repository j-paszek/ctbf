#!/usr/bin/env python3
"""Read-only figure derivative with occurrence-provenance biopsy colours.

The frozen reconstruction and v1 figure are never modified. Only initialization
is replayed to recover observed graph IDs. No distances or trees are inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper.figures.build_ktn152_empirical_figure import (
    BIOPSY_COLORS, COPY_COLOR, COPY_NUMBER_COLORS, INK, MUTED,
    UNLABELED_COLOR, _hex_rgb, _png_data_url, _read_bins, _read_profiles,
    _text, _tree_layout,
)
from reconstructor_biopsy_blocks import (
    assign_compatible_node_ids, extend_biopsy_levels,
    initialize_biopsy_tree, make_unique_node_counter,
)
from simulator import Genotype


def observed_node_biopsies(payload: dict, state_map: dict, tree: dict) -> dict[int, int]:
    """Recover each observed occurrence from the exact initializer, not its state alone."""
    internal = {entry['external_state_label']: int(entry['internal_state_label'])
                for entry in state_map['entries']}
    observed = {}
    levels = []
    for index, level in enumerate(payload['levels']):
        cells = []
        for state in level['states']:
            label = state['state_label']
            cell = Genotype(list(state['cnp']), internal[label],
                            generation=level['generation'], cell_id=internal[label])
            cells.append(cell)
            observed[(index, label)] = cell
        levels.append(cells)
    extend_biopsy_levels(levels)
    assign_compatible_node_ids(levels)
    initialize_biopsy_tree(levels, make_unique_node_counter(list(internal.values())))
    nodes = {int(node['id']): node for node in tree['nodes']}
    mapping = {}
    for (index, label), cell in observed.items():
        node = nodes.get(cell.node_id)
        if node is None or node['cell_id'] != label or node['genome'] != list(cell.genome):
            raise ValueError('Observed occurrence does not match its frozen graph node')
        if cell.node_id in mapping:
            raise ValueError('Two observations were mapped to one graph occurrence')
        mapping[cell.node_id] = index
    # Check every ancestral pair of observations, including paths through copies.
    parents = {int(edge['target']): int(edge['source']) for edge in tree['links']}
    for node, index in mapping.items():
        ancestor = parents.get(node)
        seen = {node}
        while ancestor is not None:
            if ancestor in seen:
                raise ValueError('Cycle in frozen tree')
            seen.add(ancestor)
            if ancestor in mapping and mapping[ancestor] > index:
                raise ValueError('Observed ancestry violates biopsy order')
            ancestor = parents.get(ancestor)
    return mapping


def chromosome_runs(bins: list[dict]) -> list[tuple[str, int, int]]:
    runs = []
    start = 0
    for index in range(1, len(bins) + 1):
        if index == len(bins) or bins[index]['chromosome'] != bins[start]['chromosome']:
            runs.append((bins[start]['chromosome'].removeprefix('chr'), start, index))
            start = index
    return runs


def spaced_positions(centres: list[float], left: float, right: float,
                     minimum_gap: float = 27) -> list[float]:
    """Keep genomic centres separate from collision-free label positions."""
    if (len(centres) - 1) * minimum_gap > right - left:
        raise ValueError('Not enough space for chromosome labels')
    result = []
    for centre in centres:
        result.append(max(centre, left if not result else result[-1] + minimum_gap))
    result[-1] = min(result[-1], right)
    for index in range(len(result) - 2, -1, -1):
        result[index] = min(result[index], result[index + 1] - minimum_gap)
    return result


def high_copy_counts(matrix: list[list[int]]) -> dict:
    counts = {'profile_count': len(matrix), 'bin_count': len(matrix[0]),
              'entry_count': sum(len(row) for row in matrix),
              'maximum_copy_number': max(map(max, matrix))}
    for threshold in (8, 9):
        counts[f'cn_ge_{threshold}'] = {
            'entries': sum(value >= threshold for row in matrix for value in row),
            'profiles': sum(any(value >= threshold for value in row) for row in matrix),
            'bins': sum(any(row[index] >= threshold for row in matrix)
                        for index in range(len(matrix[0]))),
        }
    return counts


def build_figure(copykit_root: Path, reconstruction_path: Path,
                 state_map_path: Path, output: Path) -> None:
    audit_path = output.with_suffix('.audit.json')
    if output.exists() or audit_path.exists():
        raise FileExistsError('Use a new output path. Existing figure variants are immutable.')
    profiles, bin_ids, matrix = _read_profiles(copykit_root / 'profiles_collapsed.tsv')
    bins = _read_bins(copykit_root / 'bins.tsv')
    payload = json.loads((copykit_root / 'ctbf_input.json').read_text())
    state_map = json.loads(state_map_path.read_text())
    reconstruction = json.loads(reconstruction_path.read_text())
    tree = reconstruction['tree']
    decision = reconstruction['reconstruction_metadata']['biopsy_layer_decision_audit']
    if len(profiles) != 77 or len(bin_ids) != 12167:
        raise ValueError('Unexpected frozen KTN152 profile dimensions')
    if [row['bin_id'] for row in bins] != bin_ids:
        raise ValueError('Profile columns and genomic bins are misaligned')
    if {(int(row['biopsy_index']), row['state_label']) for row in profiles} != {
        (index, state['state_label']) for index, level in enumerate(payload['levels'])
        for state in level['states']
    }:
        raise ValueError('Input and heat-map occurrence identities disagree')
    observed = observed_node_biopsies(payload, state_map, tree)
    nodes = {int(node['id']): node for node in tree['nodes']}
    copies = {node for node, data in nodes.items()
              if data['cell_id'] is not None and node not in observed}
    latent = set(nodes) - set(observed) - copies
    if len(nodes) != 148 or len(copies) != 31 or len(latent) != 40:
        raise ValueError('Unexpected frozen tree occurrence counts')
    if decision['copy_up_count'] != len(copies):
        raise ValueError('Copy-up count and graph occurrence provenance disagree')
    order = sorted(range(len(profiles)), key=lambda index: (
        int(profiles[index]['biopsy_index']), sum(matrix[index]) / len(matrix[index]),
        profiles[index]['state_label']))
    palette = [_hex_rgb(color) for color in COPY_NUMBER_COLORS]
    raster = [[palette[min(value, 8)] for value in matrix[index]] for index in order]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'width="1900" height="850" viewBox="0 0 1900 850">',
        '<desc>KTN152 biopsy inventory above the full CNP heat map, with an enlarged '
        'partial tree. Biopsy colours follow observed occurrence provenance.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    def title(x, y, letter, value):
        parts.extend((_text(x, y, letter, size=34, weight='bold'),
                      _text(x + 38, y, value, size=29, weight='bold')))

    # Compact longitudinal inventory aligned with the heat map below.
    title(20, 38, 'A', 'Longitudinal input')
    parts.append(f'<line x1="100" y1="122" x2="485" y2="122" stroke="{MUTED}" stroke-width="3"/>')
    descriptions = (
        ('Pretreatment', 'cycle 0', '28 cells', '28 occurrences'),
        ('On treatment', 'after cycle 2', '16 cells', '16 occurrences'),
        ('Surgery', 'after cycle 6', '35 cells', '33 occurrences'),
    )
    for index, (x, lines) in enumerate(zip((100, 294, 485), descriptions)):
        parts.append(_text(x, 79, lines[0], size=23, weight='bold', anchor='middle'))
        parts.append(_text(x, 106, lines[1], size=22, anchor='middle', fill=MUTED))
        parts.append(f'<circle cx="{x}" cy="122" r="8" fill="{BIOPSY_COLORS[index]}"/>')
        parts.append(_text(x, 159, lines[2], size=24, weight='bold', anchor='middle'))
        parts.append(_text(x, 185, lines[3], size=21, anchor='middle', fill=MUTED))
    for y, line in zip((81, 112, 143, 183), (
        '79 retained cells', '77 biopsy-indexed occurrences',
        '76 distinct CNP states', 'Counts are not clone prevalences')):
        parts.append(_text(598, y, line, size=21 if y != 81 else 24,
                           weight='bold' if y == 81 else 'normal',
                           fill=MUTED if y == 183 else INK))

    title(20, 248, 'B', 'CopyKit-derived integer CNPs')
    hx, hy, hw, hh = 83, 280, 790, 409
    parts.append(f'<image x="{hx}" y="{hy}" width="{hw}" height="{hh}" '
                 f'preserveAspectRatio="none" image-rendering="pixelated" '
                 f'xlink:href="{_png_data_url(raster)}"/>')
    offset = 0
    for index, count in enumerate((28, 16, 33)):
        y = hy + hh * offset / len(profiles)
        height = hh * count / len(profiles)
        parts.append(f'<rect x="{hx - 13}" y="{y}" width="7" height="{height}" fill="{BIOPSY_COLORS[index]}"/>')
        parts.append(_text(hx - 21, y + height / 2 + 8, f'B{index + 1}',
                           size=23, anchor='end', weight='bold'))
        offset += count
        if offset < len(profiles):
            yend = hy + hh * offset / len(profiles)
            parts.append(f'<line x1="{hx}" x2="{hx + hw}" y1="{yend}" y2="{yend}" stroke="white" stroke-width="3"/>')
    runs = chromosome_runs(bins)
    centres = [hx + hw * (start + end) / (2 * len(bins)) for _, start, end in runs]
    label_x = spaced_positions(centres, hx, hx + hw)
    for (name, start, end), centre, label in zip(runs, centres, label_x):
        parts.append(f'<path d="M {centre:.2f} {hy + hh + 3} L {label:.2f} {hy + hh + 18}" '
                     'fill="none" stroke="#949ba3" stroke-width="0.8"/>')
        parts.append(_text(label, hy + hh + 39, name, size=19, anchor='middle', fill=MUTED))
        if end < len(bins):
            xe = hx + hw * end / len(bins)
            parts.append(f'<line x1="{xe}" x2="{xe}" y1="{hy}" y2="{hy + hh}" '
                         'stroke="#d2d5d9" stroke-width="0.8"/>')
    parts.append(_text(hx + hw / 2, 764, 'Chromosome', size=23, weight='bold', anchor='middle'))
    parts.append(_text(hx + hw / 2, 795, 'Absolute copy-number scale unresolved', size=23, anchor='middle', fill=MUTED))
    parts.append(_text(121, 832, 'CN', size=22, weight='bold', anchor='end'))
    for index, (label, color) in enumerate(zip(('0','1','2','3','4','5','6','7','≥8'), COPY_NUMBER_COLORS)):
        x = 140 + index * 78
        parts.append(f'<rect x="{x}" y="812" width="25" height="25" fill="{color}" stroke="#777" stroke-width="0.5"/>')
        parts.append(_text(x + 32, 832, label, size=21))

    title(940, 38, 'C', 'Biopsy-guided partial tree')
    xs, depth, children = _tree_layout(tree, len(profiles))
    low, high = min(xs.values()), max(xs.values())
    sx = lambda v: 955 + (xs[v] - low) * 920 / (high - low)
    sy = lambda v: 75 + depth[v] * 614 / max(depth.values())
    for parent, descendants in children.items():
        for child in descendants:
            parts.append(f'<line x1="{sx(parent):.2f}" y1="{sy(parent):.2f}" '
                         f'x2="{sx(child):.2f}" y2="{sy(child):.2f}" '
                         'stroke="#9da4ab" stroke-width="1.6"/>')
    for node in sorted(nodes, key=depth.get, reverse=True):
        color = BIOPSY_COLORS[observed[node]] if node in observed else (
            COPY_COLOR if node in copies else UNLABELED_COLOR)
        radius = 5.8 if node in observed else 5.0
        parts.append(f'<circle data-node-id="{node}" cx="{sx(node):.2f}" cy="{sy(node):.2f}" '
                     f'r="{radius}" fill="{color}" stroke="white" stroke-width="1.1"/>')
    centre = 1415
    parts.append(_text(centre, 731, '148 vertices · 40 unlabeled · 31 copy-up', size=24, anchor='middle'))
    parts.append(_text(centre, 764, 'Adaptive radii: 21, 11', size=24, weight='bold', anchor='middle'))
    for x, color, label in zip((1115, 1220, 1325, 1430, 1610),
                               (*BIOPSY_COLORS, COPY_COLOR, UNLABELED_COLOR),
                               ('B1', 'B2', 'B3', 'copy-up', 'unlabeled')):
        parts.append(f'<circle cx="{x}" cy="792" r="7" fill="{color}"/>')
        parts.append(_text(x + 15, 800, label, size=23))
    parts.append(_text(centre, 834, 'Topology has no empirical ground truth', size=23, anchor='middle', fill=MUTED))
    parts.append('</svg>')

    paths = [copykit_root / 'profiles_collapsed.tsv', copykit_root / 'bins.tsv',
             copykit_root / 'ctbf_input.json', reconstruction_path, state_map_path]
    audit = {
        'schema': 'ktn152-figure-v2-audit',
        'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'observed_node_biopsy_index': observed,
        'observed_occurrences': len(observed), 'copy_up_occurrences': len(copies),
        'unlabeled_occurrences': len(latent),
        'ancestral_observation_order_check': 'passed',
        'chromosome_runs': runs, 'chromosome_label_positions': label_x,
        'copy_number_counts': high_copy_counts(matrix),
        'count_unit': 'within-biopsy-collapsed occurrence by genomic bin',
        'medicc2_pinned_input_cap': {'maximum_accepted': 8, 'capped': '>8, not >=8'},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(parts) + '\n')
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    print(f'KTN152 figure v2: {len(observed)} provenance-matched observed occurrences; temporal check passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copykit-root', type=Path, required=True)
    parser.add_argument('--reconstruction', type=Path, required=True)
    parser.add_argument('--state-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_figure(args.copykit_root, args.reconstruction, args.state_map, args.output)
