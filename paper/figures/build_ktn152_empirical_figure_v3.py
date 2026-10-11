#!/usr/bin/env python3
"""Annotate an existing inferred cross-biopsy path without altering the tree.

All data panels, layout, observed-occurrence colours and tree edges come from
the delivered v2 SVG. The added layer identifies an illustrative hypothesis,
not a validated cell lineage or a reconstruction accuracy result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


SVG = 'http://www.w3.org/2000/svg'
INK = '#17202a'
ILLUSTRATIVE_PATH = (16, 32, 3)
LABELS = ('v₁', 'v₂', 'v₃')
OFFSETS = ((18, -14, 'start'), (-18, -18, 'end'), (15, 40, 'start'))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_ancestry(tree: dict, observed: dict[int, int], path: tuple[int, ...]) -> dict:
    """Count graph relations between observed occurrences, not clonal abundances."""
    nodes = {int(row['id']): row for row in tree['nodes']}
    parents, weights = {}, {}
    for edge in tree['links']:
        u, v = int(edge['source']), int(edge['target'])
        if u not in nodes or v not in nodes or v in parents or u == v:
            raise ValueError('Frozen graph is not a rooted tree')
        parents[v] = u
        weights[u, v] = edge.get('weight')
    if len(nodes.keys() - parents.keys()) != 1 or not set(observed) <= nodes.keys():
        raise ValueError('Expected one root and observed nodes in the tree')
    if (len(path) != 3 or any(v not in observed for v in path)
            or tuple(observed[v] for v in path) != (0, 1, 2)):
        raise ValueError('Illustration must use three observed biopsy occurrences')
    if any(parents.get(v) != u for u, v in zip(path, path[1:])):
        raise ValueError('Illustrative path edges must already exist in the tree')
    ancestry, direct = Counter(), Counter()
    descendants = {v: Counter() for v in path}
    for target, biopsy in observed.items():
        ancestor = parents.get(target)
        seen = {target}
        if ancestor in observed:
            direct[observed[ancestor]+1, biopsy+1] += 1
        while ancestor is not None:
            if ancestor in seen:
                raise ValueError('Cycle in frozen reconstruction')
            seen.add(ancestor)
            if ancestor in observed:
                if observed[ancestor] > biopsy:
                    raise ValueError('Frozen observation ancestry violates biopsy order')
                ancestry[observed[ancestor]+1, biopsy+1] += 1
            if ancestor in descendants:
                descendants[ancestor][biopsy+1] += 1
            ancestor = parents.get(ancestor)
    return {
        'count_unit': 'ordered pairs of observed graph occurrences',
        'cross_biopsy_ancestry_pairs': {f'B{i}_to_B{j}': n for (i,j),n in sorted(ancestry.items()) if i < j},
        'direct_observed_edges': {f'B{i}_to_B{j}': n for (i,j),n in sorted(direct.items()) if i < j},
        'illustrative_path': [
            {'figure_label': label, 'node_id': v, 'biopsy_number': observed[v]+1,
             'state_label': nodes[v]['cell_id'],
             'observed_descendants_by_biopsy': dict(sorted(descendants[v].items()))}
            for label,v in zip(LABELS,path)
        ],
        'illustrative_edges': [{'source': u, 'target': v, 'weight': weights[u,v]}
                               for u,v in zip(path,path[1:])],
        'interpretation': 'model_inferred_ancestry_not_validated_biological_lineage',
    }


def annotate_svg(svg: str, path: tuple[int, ...]) -> str:
    ET.register_namespace('', SVG)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    root = ET.fromstring(svg)
    if root.find(f".//{{{SVG}}}g[@id='illustrative-cross-biopsy-path']") is not None:
        raise ValueError('SVG already contains an annotation layer')
    positions = {}
    for v in path:
        found = root.findall(f".//{{{SVG}}}circle[@data-node-id='{v}']")
        if len(found) != 1:
            raise ValueError('Annotated node must have exactly one existing SVG circle')
        positions[v] = (float(found[0].attrib['cx']), float(found[0].attrib['cy']))
    group = ET.SubElement(root, f'{{{SVG}}}g', {'id': 'illustrative-cross-biopsy-path'})
    ET.SubElement(group, f'{{{SVG}}}desc').text = (
        'Illustrative inferred path through three observed biopsy occurrences. '
        'This annotation is not biological validation or a clone-prevalence estimate.')
    for u,v in zip(path,path[1:]):
        x1,y1 = positions[u]
        x2,y2 = positions[v]
        # Keep the observed-node colours visible inside their highlight rings.
        length = math.hypot(x2-x1, y2-y1)
        if length <= 19:
            raise ValueError('Illustrative nodes are too close for highlight rings')
        dx,dy = 9.5*(x2-x1)/length, 9.5*(y2-y1)/length
        ET.SubElement(group, f'{{{SVG}}}line', {
            'data-illustrative-edge': f'{u}-{v}',
            'x1': str(x1+dx), 'y1': str(y1+dy), 'x2': str(x2-dx), 'y2': str(y2-dy),
            'stroke': INK, 'stroke-width': '2.6', 'stroke-opacity': '.8',
        })
    for v,label,(dx,dy,anchor) in zip(path,LABELS,OFFSETS):
        x,y = positions[v]
        ET.SubElement(group, f'{{{SVG}}}circle', {
            'data-highlight-node': str(v), 'cx': str(x), 'cy': str(y),
            'r': '9', 'fill': 'none', 'stroke': INK, 'stroke-width': '1.8',
        })
        ET.SubElement(group, f'{{{SVG}}}text', {
            'x': str(x+dx), 'y': str(y+dy), 'text-anchor': anchor,
            'font-family': 'Arial, Helvetica, sans-serif', 'font-size': '26',
            'font-weight': 'bold', 'fill': INK, 'stroke': 'white',
            'stroke-width': '4', 'stroke-linejoin': 'round', 'paint-order': 'stroke',
        }).text = label
    return ET.tostring(root, encoding='unicode') + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('paper/figures/ktn152-empirical-v2.svg'))
    parser.add_argument('--previous-audit', type=Path, default=Path('paper/figures/ktn152-empirical-v2.audit.json'))
    parser.add_argument('--reconstruction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.output.with_suffix('.audit.json')
    if args.output.exists() or audit_path.exists():
        raise FileExistsError('Use a new output name. Delivered figures are immutable.')
    prior = json.loads(args.previous_audit.read_text())
    expected = [digest for name,digest in prior['sources'].items() if '/reconstructions/' in name]
    if len(expected) != 1 or sha256(args.reconstruction) != expected[0]:
        raise ValueError('Reconstruction differs from the frozen empirical figure')
    source_paths = (args.previous, args.previous_audit, args.reconstruction)
    hashes = {str(p): sha256(p) for p in source_paths}
    frozen = json.loads(args.reconstruction.read_text())
    observed = {int(k): v for k,v in prior['observed_node_biopsy_index'].items()}
    if len(observed) != 77 or len(frozen['tree']['nodes']) != 148:
        raise ValueError('Unexpected frozen KTN152 dimensions')
    summary = summarize_ancestry(frozen['tree'], observed, ILLUSTRATIVE_PATH)
    image = annotate_svg(args.previous.read_text(), ILLUSTRATIVE_PATH)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(image)
    audit = {'schema': 'ktn152-figure-v3-illustrative-path', 'source_sha256': hashes,
             'selection_rule': 'fixed illustrative three-biopsy path chosen for legibility, not confidence',
             'preservation': 'original heatmap, node circles, colours, coordinates and edges retained',
             **summary}
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    if any(sha256(Path(p)) != digest for p,digest in hashes.items()):
        raise RuntimeError('Frozen source changed during annotation')
    print('complete: frozen tree preserved; three observed nodes annotated; '
          'empirical relations remain hypotheses')


if __name__ == '__main__':
    main()
