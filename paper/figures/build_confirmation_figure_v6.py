#!/usr/bin/env python3
"""Read-only figure refinement: native AD-F1, fixed-label control, trimmed GRF.

Reaggregate sealed scores only. Never reconstruct, evaluate trees, retry a run,
or change frozen method IDs. M3 is a publication alias for Pooled-Q.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper.figures import build_confirmation_figure_v5 as previous
from paper.figures.build_confirmation_figure_v3 import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, COUNTS, FULL,
    bootstrap_means, effect_summary, write_csv,
)
from paper.figures.build_confirmation_figure_v4 import HEIGHTS, PLOTTED_CLOSED

PARTIAL = ('M0', 'M2', 'M4')
COUNT_CONDITIONS = ('H24_L2', 'H24_rule_y', 'H24_L4', 'H24_L5')


def display_id(source_id: str) -> str:
    return 'M3' if source_id == 'Pooled-Q' else source_id


def method_label(source_id: str) -> str:
    name = display_id(source_id)
    return f'$M_{{{name[1:]}}}$' if name.startswith('M') and name[1:].isdigit() else name


def summarize(rows: list[dict], methods: tuple, conditions: tuple,
              metric: str, draws: int = BOOTSTRAP_DRAWS) -> tuple:
    """Shared history indices preserve pairing across methods and conditions."""
    selected = [r for r in rows if r['method'] in methods and r['condition_id'] in conditions]
    index = {(int(r['block_index']), r['method'], r['condition_id']): r for r in selected}
    if len(index) != len(selected):
        raise ValueError('Duplicate method/condition/history score')
    blocks = sorted({b for b, _, _ in index})
    columns = [(m, c) for m in methods for c in conditions]
    if not blocks or any(index.get((b, m, c), {}).get('status') != 'success'
                         for b in blocks for m, c in columns):
        raise ValueError('Curves require complete paired histories')
    values = np.array([[float(index[b, m, c][metric]) for m, c in columns] for b in blocks])
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Scores must be finite and in [0,1]')
    boot = bootstrap_means(values, draws=draws)
    points, endpoints = [], []
    for j, (m, c) in enumerate(columns):
        point = effect_summary(values[:, j], boot[:, j])
        points.append({'source_id': m, 'paper_id': display_id(m), 'condition_id': c,
                       'metric': metric, **{k: point[k] for k in ('count', 'mean', 'lower', 'upper')}})
    for i, m in enumerate(methods):
        first, last = i * len(conditions), (i + 1) * len(conditions) - 1
        endpoints.append({'source_id': m, 'paper_id': display_id(m), 'metric': metric,
                          'contrast': f'{conditions[-1]}_minus_{conditions[0]}',
                          **effect_summary(values[:, last] - values[:, first],
                                           boot[:, last] - boot[:, first])})
    return points, endpoints


def render(output: Path, *, native, partial_trim, closed_trim,
           fixed, partial_full, matched, height_ad, height_grf) -> None:
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ctbf-paper-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    styles = dict(previous.COLORS)
    color, marker, _ = styles['MEDICC2']
    styles['MEDICC2'] = (color, marker, ':')

    def curve(axis, x, rows, name, frozen=False):
        y = np.array([float(r['mean']) for r in rows])
        lo = np.array([float(r['bootstrap_lower' if frozen else 'lower']) for r in rows])
        hi = np.array([float(r['bootstrap_upper' if frozen else 'upper']) for r in rows])
        color, marker, style = styles[name]
        axis.errorbar(x, y, yerr=[y-lo, hi-y], color=color, marker=marker,
                      linestyle=style, linewidth=1.6, markersize=4.5, capsize=2,
                      label=method_label(name))

    def decorate(axes):
        for axis in axes:
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(axis='y', color='#D8DEE5', linewidth=.65)

    def by_count(axis, points, names, new=False):
        for name in names:
            key = 'source_id' if new else 'paper_id'
            entries = [r for r in points if r[key] == name]
            if new:
                entries.sort(key=lambda r: COUNT_CONDITIONS.index(r['condition_id']))
            else:
                entries.sort(key=lambda r: int(r['biopsies']))
            curve(axis, COUNTS, entries, name)
        axis.set_xticks(COUNTS)

    def save(fig, stem):
        for suffix in ('pdf', 'svg', 'png'):
            opts = {'metadata': {'CreationDate': None, 'ModDate': None}} if suffix == 'pdf' else {}
            fig.savefig(output / f'{stem}.{suffix}', dpi=220, facecolor='white', **opts)
        plt.close(fig)

    with plt.rc_context({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'legend.fontsize': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(13.8, 7.1), layout='constrained')
        grid = fig.add_gridspec(2, 3, width_ratios=(1.12, 1.15, 1.06))
        cp, cf = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
        pf, pt = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
        ff, fg = fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2])
        decorate((cp, cf, pf, pt, ff, fg))
        by_count(cp, partial_full, PARTIAL)
        by_count(cf, native, FULL, new=True)
        cp.legend(frameon=False, fontsize=9, loc='upper left')
        cf.legend(frameon=False, ncol=3, fontsize=9, loc='lower right')
        cp.set_title('A  More biopsies (H = 24)', loc='left', fontweight='bold')
        cp.set(ylabel='Full-truth GRF', ylim=(.075, .25))
        cf.set(xlabel='Number of biopsies', ylabel='AD-F1 (all observed labels)', ylim=(.26, .9))
        for axis, metric in ((pf, 'grf_full'), (pt, 'grf_trim')):
            for j, name in enumerate(previous.METHODS):
                rows = sorted((r for r in matched if r['paper_id'] == name and r['metric'] == metric),
                              key=lambda r: int(r['height']))
                curve(axis, np.array([int(r['height']) for r in rows])+(j-2)*.24, rows, name)
            axis.set(xticks=HEIGHTS, xlim=(12.7, 35.2), ylim=(.065, .34))
            axis.text(.03, .03, 'Matched histories: 100 / 98 / 100',
                      transform=axis.transAxes, fontsize=9, color='#4B5563')
        pf.set_title('B  Partial-tree recovery', loc='left', fontweight='bold')
        pf.set_ylabel('Full-truth GRF')
        pt.set(xlabel='True-tree height', ylabel='Trimmed-truth GRF')
        pf.legend(frameon=False, fontsize=9, loc='upper right')
        for name in PLOTTED_CLOSED:
            for axis, points, frozen in ((ff, height_ad, True), (fg, height_grf, False)):
                rows = [next(r for r in points if r['paper_id'] == name and int(r['height']) == h)
                        for h in HEIGHTS]
                curve(axis, HEIGHTS, rows, name, frozen=frozen)
        ff.set_title('C  Closed-state recovery', loc='left', fontweight='bold')
        ff.set(ylabel='AD-F1', ylim=(.43, 1.0))
        fg.set(xlabel='True-tree height', ylabel='Full-truth GRF', ylim=(.025, .32))
        for axis in (ff, fg):
            axis.set_xticks(HEIGHTS)
        ff.legend(frameon=False, ncol=2, fontsize=10)
        save(fig, 'main-figure')

        fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8), layout='constrained')
        decorate(axes)
        a, b, c = axes
        by_count(a, fixed, FULL)
        by_count(b, partial_trim, PARTIAL, new=True)
        for name in PLOTTED_CLOSED:
            rows = [next(r for r in closed_trim if r['source_id'] == name
                         and r['condition_id'] == f'H{h}_rule_y') for h in HEIGHTS]
            curve(c, HEIGHTS, rows, name)
        a.set_title('A  Fixed-label ancestry control', loc='left', fontweight='bold')
        b.set_title('B  Partial trees: trimmed truth', loc='left', fontweight='bold')
        c.set_title('C  Closed-state: trimmed truth', loc='left', fontweight='bold')
        a.set(xlabel='Number of biopsies (H = 24)', ylabel='AD-F1 (two-biopsy labels)', ylim=(.26, .9))
        b.set(xlabel='Number of biopsies (H = 24)', ylabel='Trimmed-truth GRF')
        c.set(xlabel='True-tree height', ylabel='Trimmed-truth GRF', xticks=HEIGHTS)
        a.legend(frameon=False, ncol=3, fontsize=9, loc='lower right')
        b.legend(frameon=False, fontsize=9)
        c.legend(frameon=False, ncol=2, fontsize=10)
        save(fig, 'supplementary-controls')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension-root', type=Path, required=True)
    parser.add_argument('--figures-root', type=Path, default=Path('paper/figures'))
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError('Delivered figure derivatives are immutable; use a new output directory.')
    root = args.figures_root
    source, old, closed, recent = [root / f'v5-confirmation-v{v}' for v in (1, 3, 4, 5)]
    extension, _, _, _ = previous.load_campaign(args.extension_root)
    for path, expected in previous.read_json(recent / 'audit.json')['source_sha256'].items():
        previous.check_digest(Path(path), expected)
    contract = previous.read_json(args.extension_root / 'analysis_contract.json')
    score_path = Path(contract['trimmed_root']) / 'scores.csv'
    previous.check_digest(score_path, contract['protected_sha256'][str(score_path)])
    scores = previous.numeric_records(previous.read_csv(score_path))
    scores = [r for r in scores if r['method'] in ('M0', 'M2', *FULL)]
    scores += [r for r in extension if r['method'] == 'M4']
    if len(scores) != 6400 or any(r['status'] != 'success' for r in scores):
        raise ValueError('Expected all 6,400 successful internal scores')
    native, native_gains = summarize(scores, FULL, COUNT_CONDITIONS, 'ad_f1')
    partial_trim, trim_gains = summarize(scores, PARTIAL, COUNT_CONDITIONS, 'grf_trim')
    closed_trim, _ = summarize(scores, FULL, tuple(f'H{h}_rule_y' for h in HEIGHTS), 'grf_trim')
    if any(r['count'] != 100 for r in native + partial_trim + closed_trim):
        raise ValueError('Internal curves must use all 100 histories')
    fixed = previous.read_csv(old / 'biopsy-count-curves.csv')
    partial_full = fixed + previous.read_csv(recent / 'm4-biopsy-count-curves.csv')
    matched = previous.read_csv(recent / 'partial-matched-curves.csv')
    height_ad = previous.read_csv(source / 'main-figure-data.csv')
    height_grf = previous.read_csv(closed / 'closed-state-grf-curves.csv')
    for r in native:
        if r['condition_id'] == 'H24_L2':
            control = next(x for x in fixed if x['paper_id'] == r['source_id'] and x['biopsies'] == '2')
            if not np.isclose(r['mean'], float(control['mean']), rtol=0, atol=1e-14):
                raise ValueError('Native and fixed-U2 AD-F1 must agree with two biopsies')
    inputs = [score_path, source / 'main-figure-data.csv', old / 'biopsy-count-curves.csv',
              recent / 'm4-biopsy-count-curves.csv', recent / 'partial-matched-curves.csv',
              recent / 'audit.json', closed / 'closed-state-grf-curves.csv']
    audit = {'schema': 'ctbf-confirmation-figure-v6',
             'source_sha256': {str(p): previous.digest(p) for p in inputs},
             'source_aliases': {'Pooled-Q': 'M3'}, 'internal_records': len(scores),
             'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED,
             'bootstrap_unit': 'whole_independent_history', 'interval': 'pointwise_95_percentile',
             'main_A_closed_endpoint': 'native_ad_f1_all_observed_labels_per_condition',
             'supplementary_fixed_U2_curves': 'unchanged_from_v3',
             'main_B_and_C_scores_and_intervals': 'unchanged_from_v5',
             'native_ad_f1_five_minus_two': native_gains, 'trimmed_grf_five_minus_two': trim_gains,
             'reconstruction_or_tree_evaluation_performed': False,
             'original_primary_inference_changed': False}
    args.output_root.mkdir(parents=True)
    write_csv(args.output_root / 'native-biopsy-count-curves.csv', native)
    write_csv(args.output_root / 'partial-trimmed-biopsy-count-curves.csv', partial_trim)
    write_csv(args.output_root / 'closed-trimmed-height-curves.csv', closed_trim)
    (args.output_root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    render(args.output_root, native=native, partial_trim=partial_trim, closed_trim=closed_trim,
           fixed=fixed, partial_full=partial_full, matched=matched, height_ad=height_ad, height_grf=height_grf)
    for path, expected in audit['source_sha256'].items():
        previous.check_digest(Path(path), expected)
    print('complete: 6400 stored internal scores; native AD-F1 main panel; fixed-U2 and trimmed controls; '
          'B/C values preserved; no reconstruction or tree evaluation')


if __name__ == '__main__':
    main()
