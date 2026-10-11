#!/usr/bin/env python3
"""Add the M4 placement summary from sealed scores, without rerunning trees.

The fourteen existing contrasts must reproduce exactly before new output is
written. Output must be a new directory. All experiment inputs are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))
from algorithm_evaluation.v5_confirmation_common import (  # noqa: E402
    PAPER_ARM_IDS, PAPER_ID_BY_ARM, derive_analysis_seed,
)
from algorithm_evaluation.v5_confirmation_report import _effect_summary  # noqa: E402

SOURCES = {
    'paper/figures/v5-confirmation-v1/evidence-summary.json':
        'eee6307bee86af260d7a7d20139719680f94323695a9b3af896462c81724728b',
    'algorithm_evaluation/results/v5_confirmation/trimmed_truth_v1/scores.csv':
        'afcd7215b23e30d6bceb85efd10dd564ab970c6ff0c38d4b290c360727396d1b',
    'algorithm_evaluation/results/v5_nestedbd_long/partial_comparators_100_2m_v1/'
    'reports/report_001/scores.csv':
        '962cd9cc9b6424078af4b8e9c7fdaea9c1e0f647e52b78d924899c0d42f1afc0',
}
CONDITIONS = ('H24_rule_y', 'H24_late', 'H24_random')
OLD_METHODS = ('M0', 'M2', 'M1', 'Pooled-Q', 'M5', 'M6', 'M7')
DRAW_COUNT = 100_000
TOLERANCE = 1e-12


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources() -> None:
    for relative, expected in SOURCES.items():
        if digest(PROJECT / relative) != expected:
            raise ValueError(f'Frozen score source changed: {relative}')


def load_selected(path: Path, methods: tuple[str, ...]) -> dict:
    selected = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['method'] not in methods or row['condition_id'] not in CONDITIONS:
                continue
            block = int(row['block_index'])
            key = (row['method'], block, row['condition_id'])
            if key in selected or row['status'] != 'success' or row['source_status'] != 'success':
                raise ValueError('Duplicate or unsuccessful selected score')
            if (int(row['height']) != 24 or row['replicate'] or row['seed']
                    or row['case_id'] != f'conf-b{block+1:03d}-{row["condition_id"]}'):
                raise ValueError('Unexpected history/condition identity')
            expected_problem = 'partial' if row['method'] in ('M0', 'M2', 'M4') else 'closed_state'
            if row['problem'] != expected_problem:
                raise ValueError('Unexpected reconstruction problem')
            for metric in ('grf_full', 'ad_f1'):
                row[metric] = float(row[metric])
                if not np.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
                    raise ValueError('Nonfinite or out-of-range metric')
            selected[key] = row
    expected = {(m, b, c) for m in methods for b in range(100) for c in CONDITIONS}
    if set(selected) != expected:
        raise ValueError('Expected all 100 histories at each of three placements')
    return selected


def summarize(index: dict, method: str, offset: int) -> tuple[dict, list[dict]]:
    metric = 'grf_full' if method in ('M0', 'M2', 'M4') else 'ad_f1'
    result, paired = {}, []
    base_seed = derive_analysis_seed('secondary_bootstrap') + 40
    for j, comparator in enumerate(CONDITIONS[1:]):
        contrast = f'H24_rule_y_minus_{comparator}'
        values = []
        for block in range(100):
            standard = index[method, block, CONDITIONS[0]][metric]
            other = index[method, block, comparator][metric]
            values.append(standard - other)
            paired.append({'block_index': block, 'method': method, 'metric': metric,
                           'contrast': contrast, 'standard_score': standard,
                           'comparator_score': other, 'paired_difference': standard-other})
        seed = base_seed + offset * 2 + j
        result[contrast] = {
            'bootstrap_seed': seed,
            **_effect_summary(values, bootstrap_seed=seed,
                              bootstrap_repetitions=DRAW_COUNT, tolerance=TOLERANCE),
        }
    return result, paired


def latex_cell(summary: dict) -> str:
    mean = summary['effect_summary']['mean']
    interval = summary['bootstrap_interval']
    wtl = summary['wins_ties_losses']
    return (f'${mean:.5f}$ [${interval["lower"]:.5f},{interval["upper"]:.5f}$], '
            f'{wtl["wins"]}/{wtl["ties"]}/{wtl["losses"]}')


def derive() -> tuple[dict, list[dict], str]:
    verify_sources()
    if tuple(PAPER_ID_BY_ARM[a] for a in PAPER_ARM_IDS) != OLD_METHODS:
        raise ValueError('Original method/seed-offset order changed')
    evidence_name, original_name, extension_name = SOURCES
    old = json.loads((PROJECT / evidence_name).read_text())['prespecified_descriptive_results']['placement']
    original = load_selected(PROJECT / original_name, OLD_METHODS)
    extended = load_selected(PROJECT / extension_name, ('M0', 'M2', 'M4'))
    for key, row in extended.items():
        if key[0] in ('M0', 'M2'):
            for field in ('case_id', 'problem', 'status', 'grf_full', 'ad_f1'):
                if row[field] != original[key][field]:
                    raise ValueError('Reused M0/M2 score differs from original campaign')
    verified = []
    for offset, method in enumerate(OLD_METHODS):
        computed, _ = summarize(original, method, offset)
        for contrast, result in computed.items():
            comparable = {k: v for k, v in result.items() if k != 'bootstrap_seed'}
            if comparable != old[method]['comparisons'][contrast]:
                raise ValueError(f'Original contrast no longer exact: {method}/{contrast}')
            verified.append(f'{method}/{contrast}')
    # Append after the seven historical seed offsets. Never reorder or reseed
    # an existing contrast; M4's display row can still follow M2 in the table.
    m4, paired = summarize(extended, 'M4', len(OLD_METHODS))
    row = '$M_4$ & GRF & ' + ' & '.join(latex_cell(v) for v in m4.values()) + r' \\' + '\n'
    helper_files = ('algorithm_evaluation/v5_confirmation_common.py',
                    'algorithm_evaluation/v5_confirmation_report.py')
    summary = {
        'schema': 'ctbf-v5-m4-placement-table-derivative-v1',
        'metric': 'full_true_tree_normalized_rooted_grf_similarity',
        'effect_direction': 'standard_minus_comparator',
        'history_count': 100, 'conditions': CONDITIONS,
        'bootstrap_unit': 'paired_whole_history', 'bootstrap_repetitions': DRAW_COUNT,
        'bootstrap_interval': 'two_sided_95_percentile_linear',
        'bootstrap_seed_rule': 'original secondary_bootstrap + 40 + 2*7 + comparator_offset',
        'tie_tolerance': TOLERANCE, 'old_contrasts_reproduced_exactly': verified,
        'shared_M0_M2_records_verified': 600,
        'new_reconstruction_or_evaluation': False,
        'source_sha256': SOURCES,
        'helper_sha256': {p: digest(PROJECT / p) for p in helper_files},
        'derivation_script_sha256': digest(Path(__file__)),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__},
        'M4': m4,
    }
    verify_sources()
    return summary, paired, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError('Use a new output directory; delivered evidence is immutable')
    summary, paired, row = derive()
    args.output_root.mkdir(parents=True)
    (args.output_root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.output_root / 'paired-differences.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    (args.output_root / 'm4-row.tex').write_text(row)
    print(row, end='')
    print('complete: M4 placement on 100 histories; 14/14 existing contrasts exact; '
          '600/600 reused records exact; no inference or evaluator replay')


if __name__ == '__main__':
    main()
