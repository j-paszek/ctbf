#!/usr/bin/env python3
"""Redraw frozen confirmation summaries in three equally sized square panels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ctbf-paper-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(source: Path, output: Path) -> None:
    targets = [output.with_suffix(suffix) for suffix in ('.pdf', '.svg', '.png', '.audit.json')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Use a new output name. Existing figure variants are immutable.')
    with (source / 'main-figure-data.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    with (source / 'primary-block-effects.csv').open() as handle:
        effects = sorted(float(row['block_effect']) for row in csv.DictReader(handle))
    evidence = json.loads((source / 'evidence-summary.json').read_text())
    primary = evidence['registered_primary']
    cells = {(row['paper_id'], int(row['height'])): row for row in rows}
    styles = {'M0': ('#6B7280', 'o'), 'MEDICC2': ('#D97706', 's'),
              'M2': ('#087F8C', 'D'), 'Pooled-Q': ('#6B7280', 'o'),
              'M5': ('#D97706', 's'), 'M6': ('#087F8C', 'D'), 'M7': ('#263B70', '^')}
    with plt.rc_context({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'legend.fontsize': 10, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42}):
        figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.6), layout='constrained')
        for axis in axes:
            axis.set_box_aspect(1)
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(axis='y', color='#D8DEE5', linewidth=.65)
        for axis, names in zip((axes[0], axes[2]),
                               (('M0','MEDICC2','M2'), ('Pooled-Q','M5','M6','M7'))):
            for name in names:
                selected = [cells[(name, h)] for h in (14,24,34)]
                means = [float(row['mean']) for row in selected]
                lower = [v - float(row['bootstrap_lower']) for row, v in zip(selected, means)]
                upper = [float(row['bootstrap_upper']) - v for row, v in zip(selected, means)]
                color, marker = styles[name]
                label = f'$M_{name[1:]}$' if name.startswith('M') and name[1:].isdigit() else name
                axis.errorbar((14,24,34), means, yerr=[lower, upper], color=color,
                              marker=marker, linewidth=1.8, markersize=5, capsize=3, label=label)
            axis.set_xticks((14,24,34))
            axis.set_xlabel('True-tree height')
            axis.legend(frameon=False, ncol=1 if axis is axes[0] else 2)
        axes[0].set_title('A  Partial-tree recovery', loc='left', fontweight='bold')
        axes[0].set_ylabel('Rooted GRF similarity')
        axes[2].set_title('C  Closed-state recovery', loc='left', fontweight='bold')
        axes[2].set_ylabel('AD-F1')
        axis = axes[1]
        mean = primary['block_effect_summary']['mean']
        lower, upper = (primary['bootstrap_interval'][key] for key in ('lower','upper'))
        axis.scatter(range(1,len(effects)+1), effects, s=15, color='#263B70', linewidths=0)
        axis.axhline(0, color='#7A1F1F', linewidth=1, linestyle='--')
        axis.axhspan(lower, upper, color='#B8DED9', alpha=.8)
        axis.axhline(mean, color='#087F8C', linewidth=1.8)
        axis.set_xlim(0,101)
        axis.set_ylim(-.005,.139)
        axis.set_title('B  Primary paired effect', loc='left', fontweight='bold')
        axis.set_xlabel('History blocks, ordered by effect')
        axis.set_ylabel(r'Mean $M_2-M_0$ rooted-GRF effect')
        wtl = primary['block_wins_ties_losses']
        axis.text(.04,.97,f'Mean {mean:.3f} (95% CI {lower:.3f}–{upper:.3f})\n'
                  f"Block W/T/L {wtl['wins']}/{wtl['ties']}/{wtl['losses']}",
                  transform=axis.transAxes, va='top', fontsize=9.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        for path in targets[:3]:
            options = {'metadata': {'CreationDate': None, 'ModDate': None}} if path.suffix == '.pdf' else {}
            figure.savefig(path, dpi=220, facecolor='white', bbox_inches='tight', **options)
        plt.close(figure)
    paths = [source / name for name in ('main-figure-data.csv','primary-block-effects.csv','evidence-summary.json')]
    targets[3].write_text(json.dumps({
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'change': 'layout and typography only, all values and intervals copied unchanged',
        'panels': 'one row, three square axes',
    }, indent=2) + '\n')
    print('Confirmation figure v2: unchanged frozen summaries; three square panels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
