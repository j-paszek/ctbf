#!/usr/bin/env python3
"""Audit the completed 2M comparator extension and extend the publication figure.

Read-only with respect to all experiment and earlier figure directories.
No inference, evaluation, primary-test recalculation or seed selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithm_evaluation.v5_trimmed_truth_report import case_scores
from paper.figures.build_confirmation_figure_v3 import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, COUNTS, FULL, STYLES,
    bootstrap_means, effect_summary, write_csv,
)
from paper.figures.build_confirmation_figure_v4 import HEIGHTS, PLOTTED_CLOSED, method_label

METHODS = ('M0', 'M2', 'M4', 'MEDICC2', 'NestedBD-Long')
METRICS = ('grf_full', 'grf_trim')
COLORS = {**STYLES, 'M4': ('#263B70', '^', '-'),
          'NestedBD-Long': ('#875692', 'P', ':')}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def check_digest(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f'Sealed source changed: {path}')


def child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Source reference escapes its declared root')
    return path


def protected_hashes(root: Path, contract: dict, marker: dict) -> dict:
    """Apply only the completion-bound, owner-approved operational repair."""
    expected = dict(contract['protected_sha256'])
    repair_path = root / 'resume_repair_v1.json'
    if repair_path.exists():
        check_digest(repair_path, marker['artifacts']['resume_repair_v1.json'])
        repair = read_json(repair_path)
        if (repair['schema_version'] != 'ctbf-v5-partial-comparator-resume-repair-v1'
                or repair['original_contract_sha256'] != marker['contract_sha256']
                or repair['scope'] != 'terminal_outcome_reuse_and_selected_budget_reporting'
                or repair['scientific_rules_and_results_unchanged'] is not True):
            raise ValueError('Unrecognized operational repair record')
        allowed = {'v5_partial_comparator_extension.py', 'v5_partial_comparator_extension_2m_nested.py'}
        for path, replacement in repair['source_replacements'].items():
            if Path(path).name not in allowed or expected.get(path) != replacement['before']:
                raise ValueError('Repair does not match the original source declaration')
            expected[path] = replacement['after']
    return expected


def numeric_records(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        out = dict(row)
        for key in ('block_index', 'height', 'replicate', 'seed', 'inference_budget_iterations'):
            out[key] = int(out[key]) if out.get(key) not in ('', None) else None
        for key in (*METRICS, 'ad_f1'):
            out[key] = float(out[key]) if out.get(key) not in ('', None) else None
        if out['status'] == 'success':
            if any(out[k] is None or not np.isfinite(out[k]) or not 0 <= out[k] <= 1 for k in METRICS):
                raise ValueError('Successful records require finite GRF scores in [0,1]')
        elif any(out[k] is not None for k in METRICS):
            raise ValueError('Failed records must not supply imputed GRF scores')
        converted.append(out)
    return converted


def aggregate_diagnostics(diagnostic: dict) -> dict:
    """Summarize every successful seed, including a seed with a failed mate."""
    if diagnostic['iterations_per_seed'] != 2_000_000:
        raise ValueError('This publication derivative requires the uniform 2M series')
    runs = diagnostic['seed_records']
    if len({(r['case_id'], r['seed']) for r in runs}) != len(runs):
        raise ValueError('Duplicate diagnostic seed')
    result = {'heights': {}, 'failed_seeds': []}
    for row in runs:
        if row['inference_budget_iterations'] != 2_000_000:
            raise ValueError('Mixed inference budgets')
        if row['status'] != 'success':
            result['failed_seeds'].append({k: row[k] for k in
                ('case_id', 'height', 'seed', 'replicate', 'failure')})
            continue
        sample = row['sample_audit']
        if (sample['samples'], sample['discarded'], sample['retained']) != (2001, 200, 1801):
            raise ValueError('Unexpected retained-tree summary sample counts')
    for height in (14, 24):
        successful = [r for r in runs if r['height'] == height and r['status'] == 'success']
        paired = [v for k, v in diagnostic['clade_agreement'].items() if f'-H{height}_' in k]
        windows = {}
        for burnin in ('10', '25', '50'):
            scalar = {}
            for name in ('posterior', 'likelihood', 'TreeHeight'):
                values = [r['diagnostics']['scalar_windows'][burnin]
                          ['official_beast_statistics'][name]['ESS'] for r in successful]
                scalar[name] = {'n': len(values), 'minimum': min(values),
                    'median': statistics.median(values), 'maximum': max(values),
                    'at_least_200': sum(x >= 200 for x in values)}
            differences = [r[burnin]['maximum_clade_frequency_difference'] for r in paired]
            windows[burnin] = {'scalar_ess': scalar, 'paired_inputs': len(paired),
                'clade_max_difference_minimum': min(differences),
                'clade_max_difference_median': statistics.median(differences),
                'clade_max_difference_maximum': max(differences),
                'pairs_with_max_difference_above_025': sum(x > .25 for x in differences),
                'pairs_with_opposed_095_005_clades': sum(
                    any(r[burnin]['clades_ge95_in_one_le05_in_other']) for r in paired),
                'identical_majority_sets': sum(
                    r[burnin]['majority_clades_per_seed'] == [r[burnin]['shared_majority_clades']] * 2
                    for r in paired)}
        result['heights'][str(height)] = {'successful_seed_runs': len(successful), 'windows': windows}
    return result


def load_campaign(root: Path) -> tuple:
    """Check declared artifact closure, retained failures and exact reused scores."""
    marker = read_json(root / '_COMPLETE.json')
    contract = read_json(root / 'analysis_contract.json')
    if marker['schema_version'] != 'ctbf-v5-partial-comparator-extension-2m-v1':
        raise ValueError('Expected the completed uniform 2M extension')
    check_digest(root / 'analysis_contract.json', marker['contract_sha256'])
    report_dir = child(root, marker['report_directory'])
    for relative, expected in read_json(report_dir / '_SEALED.json').items():
        check_digest(child(report_dir, relative), expected)
    check_digest(report_dir / 'report.json', marker['report_sha256'])
    # This reads only explicitly declared files, never searches raw trace corpora.
    for relative, expected in marker['artifacts'].items():
        check_digest(child(root, relative), expected)
    print(f'audit: {len(marker["artifacts"])} completion-bound artifacts intact', flush=True)
    for path, expected in protected_hashes(root, contract, marker).items():
        check_digest(Path(path), expected)
    report = read_json(report_dir / 'report.json')
    diagnostic = read_json(report_dir / 'diagnostics.json')
    raw = read_csv(report_dir / 'scores.csv')
    records = numeric_records(raw)
    expected = {(r['case_id'], m, None) for r in contract['m4_cases'] for m in ('M0', 'M2', 'M4')}
    expected |= {(r['case_id'], 'MEDICC2', None) for r in contract['m4_cases']
                 if r['condition_id'] in tuple(f'H{h}_rule_y' for h in HEIGHTS)}
    expected |= {(r['case_id'], 'NestedBD-Long', s['replicate'])
                 for r in contract['nested_cases'] for s in r['seeds']}
    actual = {(r['case_id'], r['method'], r['replicate']) for r in records}
    if len(records) != 3100 or len(actual) != len(records) or actual != expected:
        raise ValueError('Declared 3,100-record roster is incomplete or duplicated')
    aggregate = case_scores(records)
    if aggregate != report['case_scores']:
        raise ValueError('Stored case scores differ from both-seed aggregation')
    old = read_csv(Path(contract['trimmed_root']) / 'scores.csv')
    old_index = {(r['case_id'], r['method']): r for r in old if r['method'] in ('M0', 'M2', 'MEDICC2')}
    reused = [r for r in raw if r['method'] in ('M0', 'M2', 'MEDICC2')]
    for row in reused:
        previous = old_index[row['case_id'], row['method']]
        if any(row[k] != previous[k] for k in previous):
            raise ValueError('Reused comparator fields changed')
    if len(reused) != 1900:
        raise ValueError('Expected 1,900 exact reused comparator rows')
    declared_seeds = {(c['case_id'], s['seed']) for c in contract['nested_cases'] for s in c['seeds']}
    for name in ('seed_records', 'inference_outcomes'):
        rows = diagnostic[name]
        if len(rows) != 400 or {(r['case_id'], r['seed']) for r in rows} != declared_seeds:
            raise ValueError('Diagnostic seed coverage differs from the declaration')
    summary = aggregate_diagnostics(diagnostic)
    summary.update(record_counts=report['record_counts_by_method'],
        completion_bound_artifacts=len(marker['artifacts']),
        protected_files_verified=len(contract['protected_sha256']),
        reused_comparator_rows_exact=len(reused), incomplete_record_count=0,
        input_ranges={str(h): {k: [min(c['input_summary'][k] for c in contract['nested_cases'] if c['height'] == h),
                                  max(c['input_summary'][k] for c in contract['nested_cases'] if c['height'] == h)]
                      for k in ('bins', 'occurrences', 'distinct_states', 'minimum_cn', 'maximum_cn', 'entries_above_9')}
                      for h in (14, 24)})
    return aggregate, report, summary, report_dir


def partial_points(rows: list[dict], draws: int = BOOTSTRAP_DRAWS) -> list[dict]:
    """One common successful-history mask per height for all available methods.

    Both truth targets and all methods share each height's bootstrap indices.
    Two seed scores are already averaged within an input, never resampled as
    independent observations. H34 has four methods and no NestedBD placeholder.
    """
    index = {(r['block_index'], r['condition_id'], r['method']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate case/method row')
    blocks = sorted({r['block_index'] for r in rows})
    points = []
    for height in HEIGHTS:
        methods = METHODS if height != 34 else METHODS[:-1]
        condition = f'H{height}_rule_y'
        complete = [b for b in blocks if all(index.get((b, condition, m), {}).get('status') == 'success'
                                            for m in methods)]
        if not complete:
            raise ValueError('No matched histories at a declared height')
        columns = [(m, k) for m in methods for k in METRICS]
        values = np.array([[index[b, condition, m][k] for m, k in columns] for b in complete])
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Invalid matched GRF scores')
        boot = bootstrap_means(values, draws=draws)
        for j, (method, metric) in enumerate(columns):
            point = effect_summary(values[:, j], boot[:, j])
            points.append({'paper_id': method, 'height': height, 'metric': metric,
                'planned': len(blocks), 'block_indices': complete,
                **{k: point[k] for k in ('count', 'mean', 'lower', 'upper')}})
    return points


def m4_count_points(rows: list[dict], draws: int = BOOTSTRAP_DRAWS) -> list[dict]:
    selected = {(r['block_index'], r['condition_id']): r for r in rows if r['method'] == 'M4'}
    blocks = sorted({b for b, _ in selected})
    conditions = ('H24_L2', 'H24_rule_y', 'H24_L4', 'H24_L5')
    if any(selected.get((b, c), {}).get('status') != 'success' for b in blocks for c in conditions):
        raise ValueError('M4 biopsy curves require complete paired histories')
    values = np.array([[selected[b, c]['grf_full'] for c in conditions] for b in blocks])
    boot = bootstrap_means(values, draws=draws)
    result = []
    for j, count in enumerate(COUNTS):
        summary = effect_summary(values[:, j], boot[:, j])
        result.append({'paper_id': 'M4', 'condition_id': conditions[j], 'biopsies': count,
            'metric': 'native_grf', **{k: summary[k] for k in ('count', 'mean', 'lower', 'upper')}})
    return result


def render(source: Path, previous: Path, closed: Path, output: Path,
           points: list[dict], m4_counts: list[dict]) -> None:
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ctbf-paper-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    heights = read_csv(source / 'main-figure-data.csv')
    counts = read_csv(previous / 'biopsy-count-curves.csv') + m4_counts
    grf = read_csv(closed / 'closed-state-grf-curves.csv')

    def curve(axis, x, rows, name, frozen=False):
        y = np.array([float(r['mean']) for r in rows])
        lo = np.array([float(r['bootstrap_lower' if frozen else 'lower']) for r in rows])
        hi = np.array([float(r['bootstrap_upper' if frozen else 'upper']) for r in rows])
        color, marker, style = COLORS[name]
        axis.errorbar(x, y, yerr=[y-lo, hi-y], color=color, marker=marker,
                      linestyle=style, linewidth=1.6, markersize=4.5, capsize=2,
                      label=method_label(name))

    with plt.rc_context({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'legend.fontsize': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(13.8, 7.1), layout='constrained')
        grid = fig.add_gridspec(2, 3, width_ratios=(1.12, 1.15, 1.06))
        cp, cf = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
        pf, pt = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
        ff, fg = fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2])
        for axis in (cp, cf, pf, pt, ff, fg):
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(axis='y', color='#D8DEE5', linewidth=.65)
        for axis, names in ((cp, ('M0', 'M2', 'M4')), (cf, FULL)):
            for name in names:
                rows = sorted((r for r in counts if r['paper_id'] == name), key=lambda r: int(r['biopsies']))
                curve(axis, COUNTS, rows, name)
            axis.set_xticks(COUNTS)
            axis.legend(frameon=False, ncol=1 if axis is cp else 3,
                        fontsize=9, loc='upper left' if axis is cp else 'lower right')
        cp.set_title('A  More biopsies (H = 24)', loc='left', fontweight='bold')
        cp.set(ylabel='Full-truth GRF', ylim=(.075, .25))
        cf.set(xlabel='Number of biopsies', ylabel='Fixed-label AD-F1', ylim=(.26, .9))
        for axis, metric in ((pf, 'grf_full'), (pt, 'grf_trim')):
            for j, name in enumerate(METHODS):
                rows = sorted((r for r in points if r['paper_id'] == name and r['metric'] == metric),
                              key=lambda r: r['height'])
                # Small fixed horizontal offsets separate M2/M4 and overlapping intervals.
                x = np.array([r['height'] for r in rows], dtype=float) + (j-2)*.24
                curve(axis, x, rows, name)
            axis.set_xticks(HEIGHTS)
            axis.set_xlim(12.7, 35.2)
            axis.set_ylim(.065, .34)
        pf.set_title('B  Partial-tree recovery', loc='left', fontweight='bold')
        pf.set_ylabel('Full-truth GRF')
        pt.set(xlabel='True-tree height', ylabel='Trimmed-truth GRF')
        pf.legend(frameon=False, fontsize=9, loc='upper right', ncol=1)
        for axis in (pf, pt):
            axis.text(.03, .03, 'Matched histories: 100 / 98 / 100',
                      transform=axis.transAxes, fontsize=9, color='#4B5563')
        for name in PLOTTED_CLOSED:
            rows = [next(r for r in heights if r['paper_id'] == name and int(r['height']) == h) for h in HEIGHTS]
            curve(ff, HEIGHTS, rows, name, frozen=True)
            rows = [next(r for r in grf if r['paper_id'] == name and int(r['height']) == h) for h in HEIGHTS]
            curve(fg, HEIGHTS, rows, name)
        ff.set_title('C  Closed-state recovery', loc='left', fontweight='bold')
        ff.set(ylabel='AD-F1', ylim=(.43, 1.0))
        fg.set(xlabel='True-tree height', ylabel='Full-truth GRF', ylim=(.025, .32))
        for axis in (ff, fg):
            axis.set_xticks(HEIGHTS)
        ff.legend(frameon=False, ncol=2, fontsize=10)
        for suffix in ('pdf', 'svg', 'png'):
            opts = {'metadata': {'CreationDate': None, 'ModDate': None}} if suffix == 'pdf' else {}
            fig.savefig(output / f'main-figure.{suffix}', dpi=220, facecolor='white', **opts)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('paper/figures/v5-confirmation-v1'))
    parser.add_argument('--previous', type=Path, default=Path('paper/figures/v5-confirmation-v3'))
    parser.add_argument('--closed', type=Path, default=Path('paper/figures/v5-confirmation-v4'))
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError('Use a new output directory. Delivered derivatives are immutable.')
    rows, report, diagnostic, report_dir = load_campaign(args.extension_root)
    old_audit = read_json(args.closed / 'audit.json')
    for path, expected in old_audit['source_sha256'].items():
        check_digest(Path(path), expected)
    inputs = [args.extension_root / '_COMPLETE.json', args.extension_root / 'analysis_contract.json',
              report_dir / '_SEALED.json', report_dir / 'report.json', report_dir / 'scores.csv',
              report_dir / 'diagnostics.json', args.source / 'main-figure-data.csv',
              args.previous / 'biopsy-count-curves.csv', args.closed / 'closed-state-grf-curves.csv',
              args.closed / 'audit.json']
    source_hashes = {str(p): digest(p) for p in inputs}
    points, counts = partial_points(rows), m4_count_points(rows)
    for entry in report['five_method_matched_scores']:
        for method in METHODS:
            for metric in METRICS:
                point = next(r for r in points if r['paper_id'] == method and r['metric'] == metric
                             and f'H{r["height"]}_rule_y' == entry['condition'])
                if point['block_indices'] != entry['block_indices'] or not np.isclose(
                        point['mean'], entry['scores'][method][metric], atol=1e-14, rtol=0):
                    raise ValueError('Figure differs from the sealed matched panel')
    args.output_root.mkdir(parents=True)
    write_csv(args.output_root / 'partial-matched-curves.csv', points)
    write_csv(args.output_root / 'm4-biopsy-count-curves.csv', counts)
    audit = {'schema': 'ctbf-confirmation-figure-v5', 'source_sha256': source_hashes,
        'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_unit': 'whole_simulated_history', 'matched_masks_per_height': True,
        'paired_seeds_averaged_before_bootstrap': True,
        'existing_closed_state_curves_and_intervals': 'copied_unchanged',
        'existing_M0_M2_biopsy_curves_and_intervals': 'copied_unchanged',
        'registered_primary_test_recomputed': False, 'new_inference_or_evaluation': False,
        'convergence_established': False, 'campaign_audit': diagnostic,
        'paired_contrasts': report['paired_contrasts'], 'sampling_contrasts': report['sampling_contrasts']}
    (args.output_root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    render(args.source, args.previous, args.closed, args.output_root, points, counts)
    for path, expected in source_hashes.items():
        check_digest(Path(path), expected)
    print('complete: 3100 records audited; 1900 reused rows exact; '
          'five-method full/trimmed panel; 100/98/100 matched histories; no inference')


if __name__ == '__main__':
    main()
