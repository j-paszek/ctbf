#!/usr/bin/env python3
"""Add closed-state GRF to the frozen confirmation figure without rerunning trees.

Existing curves/intervals are copied from delivered v1/v3 summaries. Only
stored standard-condition closed-state GRF receives new descriptive summaries.
No evaluation, reconstruction, hypothesis test or method selection is run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithm_evaluation.v5_confirmation_common import PAPER_ID_BY_ARM
from paper.figures.build_confirmation_figure_v3 import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, COUNTS, FULL, PARTIAL, STYLES,
    audit_identity, bootstrap_means, checked_json, effect_summary, sha256,
    write_csv,
)

HEIGHTS = (14, 24, 34)
PLOTTED_CLOSED = ('Pooled-Q', 'M5', 'M6', 'M7')
COLUMNS = tuple((name, height) for name in FULL for height in HEIGHTS)


def method_label(name: str) -> str:
    if name == 'M0':
        return r'$M_0$ (cnp2cnp–NJ)'
    return f'$M_{name[1:]}$' if name.startswith('M') and name[1:].isdigit() else name


def summarize_closed_grf(matrix: np.ndarray, draws: int = BOOTSTRAP_DRAWS) -> tuple:
    """All columns share history resamples; pooled-height effects retain n blocks."""
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[1] != len(COLUMNS)
            or matrix.shape[0] == 0 or not np.isfinite(matrix).all()
            or np.any((matrix < 0) | (matrix > 1))):
        raise ValueError('Finite history-by-method/height GRF scores in [0,1] required')
    boot = bootstrap_means(matrix, draws=draws)
    points = []
    for j, (name, height) in enumerate(COLUMNS):
        summary = effect_summary(matrix[:, j], boot[:, j])
        points.append({'paper_id': name, 'condition_id': f'H{height}_rule_y',
                       'height': height, 'metric': 'grf', 'count': len(matrix),
                       **{key: summary[key] for key in ('mean', 'median', 'lower', 'upper')}})
    contrasts = []
    for left, right in (('M6', 'M5'), ('M7', 'Pooled-Q')):
        indices_left = [COLUMNS.index((left, h)) for h in HEIGHTS]
        indices_right = [COLUMNS.index((right, h)) for h in HEIGHTS]
        values = matrix[:, indices_left] - matrix[:, indices_right]
        resampled = boot[:, indices_left] - boot[:, indices_right]
        for k, height in enumerate((*HEIGHTS, 'block_mean')):
            delta = values[:, k] if k < 3 else values.mean(axis=1)
            sampled = resampled[:, k] if k < 3 else resampled.mean(axis=1)
            contrasts.append({'contrast': f'{left}_minus_{right}', 'height': height,
                              'metric': 'grf', **effect_summary(delta, sampled)})
    return points, contrasts


def load_closed_grf(run_root: Path, source: Path, previous: Path) -> tuple:
    """Read only 5 closed-state methods x 3 standard heights x 100 histories."""
    prior = json.loads((previous / 'audit.json').read_text())
    closure = run_root / 'run_checksums.sha256'
    evidence_path = source / 'evidence-summary.json'
    evidence = json.loads(evidence_path.read_text())
    expected_sources = prior['source_sha256']
    by_name = {Path(path).name: digest for path, digest in expected_sources.items()}
    for path in (closure, run_root / 'run_manifest.json', evidence_path,
                 source / 'main-figure-data.csv', source / 'primary-block-effects.csv'):
        if sha256(path) != by_name.get(path.name):
            raise ValueError(f'Previously frozen figure source changed: {path.name}')
    if sha256(closure) != evidence['input_evidence']['internal_run_checksum_sha256']:
        raise ValueError('Frozen run checksum differs from publication evidence')
    run = json.loads((run_root / 'run_manifest.json').read_text())
    if run['status'] != 'complete' or run['completed_record_count'] != 5600:
        raise ValueError('Complete frozen confirmation run required')
    specs = {row['arm_id']: row for row in run['arm_specs']}
    data, records = {}, []
    suffixes = tuple(f'-H{h}_rule_y' for h in HEIGHTS)
    for entry in run['record_inventory']:
        name = PAPER_ID_BY_ARM.get(entry['arm_id'])
        if name not in FULL or not entry['case_id'].endswith(suffixes):
            continue
        record = checked_json(run_root, entry)
        if record['status'] != 'success':
            raise ValueError('Do not silently omit failed frozen records')
        audit_identity(record, specs[entry['arm_id']])
        if record['case_id'] != entry['case_id']:
            raise ValueError('Record identity differs from inventory')
        condition = record['condition_id']
        height = next((h for h in HEIGHTS if condition == f'H{h}_rule_y'), None)
        block = record['block_index']
        key = (block, name, height)
        if height is None or not 0 <= block < 100 or key in data:
            raise ValueError('Unexpected or duplicate history/method/condition')
        value = float(record['metrics']['grf'])
        data[key] = value
        records.append({'block_index': block, 'paper_id': name, 'height': height,
                        'case_id': record['case_id'], 'metric': 'grf', 'value': value,
                        'record_path': entry['path'], 'record_sha256': entry['sha256']})
    expected = {(b, name, h) for b in range(100) for name, h in COLUMNS}
    if set(data) != expected:
        raise ValueError('Expected all 1500 closed-state standard-condition records')
    matrix = np.array([[data[b, name, h] for name, h in COLUMNS] for b in range(100)])
    for j, (name, height) in enumerate(COLUMNS):
        old = evidence['prespecified_descriptive_results']['absolute_standard_scores'][name]
        old = old[f'H{height}_rule_y']['grf']
        if old['count'] != 100 or not np.isclose(matrix[:, j].mean(), old['mean'], rtol=0, atol=1e-14):
            raise ValueError('GRF does not reproduce frozen absolute-score summary')
    return matrix, records


def render(source: Path, previous: Path, output: Path, grf_points: list[dict]) -> None:
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ctbf-paper-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with (source / 'main-figure-data.csv').open() as handle:
        heights = list(csv.DictReader(handle))
    with (previous / 'biopsy-count-curves.csv').open() as handle:
        counts = list(csv.DictReader(handle))

    def curve(axis, x, rows, name, frozen=False):
        y = np.array([float(r['mean']) for r in rows])
        lo = np.array([float(r['bootstrap_lower' if frozen else 'lower']) for r in rows])
        hi = np.array([float(r['bootstrap_upper' if frozen else 'upper']) for r in rows])
        color, marker, style = STYLES[name]
        axis.errorbar(x, y, yerr=[y-lo, hi-y], color=color, marker=marker,
                      linestyle=style, linewidth=1.7, markersize=4.5, capsize=2.5,
                      label=method_label(name))

    with plt.rc_context({'font.size': 12, 'axes.titlesize': 13,
                         'axes.labelsize': 12, 'xtick.labelsize': 11,
                         'ytick.labelsize': 11, 'legend.fontsize': 10.5,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(13.2, 6.0), layout='constrained')
        grid = fig.add_gridspec(2, 3, width_ratios=(1.12, 1, 1.06))
        count_partial = fig.add_subplot(grid[0, 0])
        count_full = fig.add_subplot(grid[1, 0])
        partial = fig.add_subplot(grid[:, 1])
        full_f1 = fig.add_subplot(grid[0, 2])
        full_grf = fig.add_subplot(grid[1, 2])
        for axis in (count_partial, count_full, partial, full_f1, full_grf):
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(axis='y', color='#D8DEE5', linewidth=.65)
        for axis, names in ((count_partial, PARTIAL), (count_full, FULL)):
            for name in names:
                rows = sorted((r for r in counts if r['paper_id'] == name),
                              key=lambda r: int(r['biopsies']))
                curve(axis, COUNTS, rows, name)
            axis.set_xticks(COUNTS)
            axis.legend(frameon=False, ncol=1 if names == PARTIAL else 3,
                        fontsize=10, loc='upper left' if names == PARTIAL else 'lower right')
        count_partial.set_title('A  More biopsies (H = 24)', loc='left', fontweight='bold')
        count_partial.set_ylabel('Rooted GRF')
        count_partial.set_ylim(.075, .25)
        count_full.set_xlabel('Number of biopsies')
        count_full.set_ylabel('Fixed-label AD-F1')
        count_full.set_ylim(.26, .9)
        for axis, names in ((partial, ('M0', 'MEDICC2', 'M2')), (full_f1, PLOTTED_CLOSED)):
            for name in names:
                rows = [next(r for r in heights if r['paper_id'] == name
                             and int(r['height']) == h) for h in HEIGHTS]
                curve(axis, HEIGHTS, rows, name, frozen=True)
            axis.set_xticks(HEIGHTS)
            axis.legend(frameon=False, ncol=1 if axis is partial else 2, fontsize=10)
        partial.set_box_aspect(1)
        partial.set_title('B  Partial-tree recovery', loc='left', fontweight='bold')
        partial.set(xlabel='True-tree height', ylabel='Rooted GRF')
        full_f1.set_title('C  Closed-state recovery', loc='left', fontweight='bold')
        full_f1.set_ylabel('AD-F1')
        full_f1.set_ylim(.43, 1.0)
        for name in PLOTTED_CLOSED:
            rows = [next(r for r in grf_points if r['paper_id'] == name
                         and r['height'] == h) for h in HEIGHTS]
            curve(full_grf, HEIGHTS, rows, name)
        full_grf.set_xticks(HEIGHTS)
        full_grf.set(xlabel='True-tree height', ylabel='Rooted GRF')
        full_grf.set_ylim(.025, .32)
        for suffix in ('pdf', 'svg', 'png'):
            opts = {'metadata': {'CreationDate': None, 'ModDate': None}} if suffix == 'pdf' else {}
            fig.savefig(output / f'main-figure.{suffix}', dpi=220, facecolor='white', **opts)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('paper/figures/v5-confirmation-v1'))
    parser.add_argument('--previous', type=Path, default=Path('paper/figures/v5-confirmation-v3'))
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError('Use a new output directory. Delivered figures are immutable.')
    matrix, records = load_closed_grf(args.run_root, args.source, args.previous)
    points, contrasts = summarize_closed_grf(matrix)
    inputs = [args.run_root / 'run_checksums.sha256', args.run_root / 'run_manifest.json',
              args.source / 'evidence-summary.json', args.source / 'main-figure-data.csv',
              args.previous / 'audit.json', args.previous / 'biopsy-count-curves.csv']
    source_hashes = {str(p): sha256(p) for p in inputs}
    args.output_root.mkdir(parents=True)
    write_csv(args.output_root / 'closed-state-grf-blocks.csv', records)
    write_csv(args.output_root / 'closed-state-grf-curves.csv', points)
    write_csv(args.output_root / 'closed-state-grf-contrasts.csv', contrasts)
    audit = {'schema': 'ctbf-confirmation-figure-v4', 'source_sha256': source_hashes,
             'blocks': 100, 'closed_state_standard_records': len(records),
             'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED,
             'existing_curves_and_intervals': 'copied_unchanged_from_v1_and_v3',
             'new_summary_role': 'descriptive_closed_state_complementary_grf_no_new_test',
             'grf_semantics': 'complete_tree_representations_not_fixed_label',
             'plotted_closed_methods': PLOTTED_CLOSED, 'paired_contrasts': contrasts}
    (args.output_root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    render(args.source, args.previous, args.output_root, points)
    if any(sha256(Path(p)) != digest for p, digest in source_hashes.items()):
        raise RuntimeError('A frozen source changed during plotting')
    print('complete: 1500 stored closed-state records; 15 GRF score cells; '
          '8 descriptive paired contrasts; existing curves preserved')


if __name__ == '__main__':
    main()
