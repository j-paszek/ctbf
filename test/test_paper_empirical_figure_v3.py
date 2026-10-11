"""Read-only ancestry/annotation checks without patient inputs."""
import copy
import xml.etree.ElementTree as ET

import pytest

from paper.figures.build_ktn152_empirical_figure_v3 import (
    SVG, annotate_svg, summarize_ancestry,
)


def fixture_tree():
    return {'nodes': [{'id': i, 'cell_id': str(i)} for i in range(4)],
            'links': [{'source': 0, 'target': 1, 'weight': 2},
                      {'source': 1, 'target': 2, 'weight': 3},
                      {'source': 1, 'target': 3, 'weight': 4}]}


def test_observed_ancestry_counts_are_occurrence_pairs_not_edges():
    tree = fixture_tree()
    before = copy.deepcopy(tree)
    report = summarize_ancestry(tree, {0:0, 1:1, 2:2, 3:2}, (0,1,2))
    assert report['cross_biopsy_ancestry_pairs'] == {
        'B1_to_B2': 1, 'B1_to_B3': 2, 'B2_to_B3': 2}
    assert report['direct_observed_edges'] == {'B1_to_B2': 1, 'B2_to_B3': 2}
    assert report['illustrative_path'][1]['observed_descendants_by_biopsy'] == {3:2}
    assert tree == before


def test_annotation_cannot_claim_an_unobserved_or_nonexistent_path():
    with pytest.raises(ValueError, match='observed biopsy occurrences'):
        summarize_ancestry(fixture_tree(), {0:0, 2:2, 3:2}, (0,1,2))
    with pytest.raises(ValueError, match='already exist'):
        summarize_ancestry(fixture_tree(), {0:0, 2:1, 3:2}, (0,2,3))


def test_svg_keeps_original_elements_and_adds_only_annotation_layer():
    svg = (f'<svg xmlns="{SVG}"><path id="tree-edge" d="M 0 0 L 1 1"/>'
           '<circle data-node-id="0" cx="10" cy="20" r="5" fill="blue"/>'
           '<circle data-node-id="1" cx="30" cy="40" r="5" fill="orange"/>'
           '<circle data-node-id="2" cx="50" cy="60" r="5" fill="pink"/></svg>')
    old, new = ET.fromstring(svg), ET.fromstring(annotate_svg(svg, (0,1,2)))
    assert len(new) == len(old)+1
    assert all(ET.tostring(a) == ET.tostring(b) for a,b in zip(old,new))
    group = new[-1]
    assert group.attrib['id'] == 'illustrative-cross-biopsy-path'
    assert len(group.findall(f'{{{SVG}}}text')) == 3
    assert len(group.findall(f'{{{SVG}}}line')) == 2
