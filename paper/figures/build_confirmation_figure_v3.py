#!/usr/bin/env python3
"""Read-only method-identity and biopsy-number derivative of frozen confirmation.

No reconstruction, evaluation, or hypothesis test is run. Stored fixed-label
AD-F1 is used only for closed-state outputs. Partial curves use stored native
GRF and are explicitly not a fixed-target control.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithm_evaluation.v5_algorithm_development_common import ARM_SPEC_BY_ID
from algorithm_evaluation.v5_confirmation_common import (
    BIOPSY_COUNT_CONDITION_IDS, PAPER_ID_BY_ARM,
)

COUNTS = (2, 3, 4, 5)
PARTIAL = ('M0', 'M2')
FULL = ('M1', 'Pooled-Q', 'M5', 'M6', 'M7')
METHODS = PARTIAL + FULL
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 20260916
STYLES = {
    'M0': ('#6B7280', 'o', '--'), 'M2': ('#087F8C', 'D', '-'),
    'M1': ('#9A6DAD', 'v', '--'), 'Pooled-Q': ('#6B7280', 'o', '--'),
    'M5': ('#D97706', 's', '--'), 'M6': ('#087F8C', 'D', '-'),
    'M7': ('#263B70', '^', '-'), 'MEDICC2': ('#D97706', 's', '-'),
}
IDENTITY_FIELDS = {
    'algorithm': 'algorithm_name', 'input_mode': 'input_mode',
    'only_nj': 'only_nj', 'biopsy_preset': 'biopsy_preset',
    'radius_policy': 'radius_policy', 'top_output_projection': 'top_output_projection',
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_json(root: Path, entry: dict) -> dict:
    path = (root / entry['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Record path escapes its source directory')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError(f'Frozen record changed: {entry["path"]}')
    return json.loads(raw)


def audit_identity(record: dict, spec: dict) -> None:
    if record['arm_id'] != spec['arm_id'] or record['problem'] != spec['problem']:
        raise ValueError('Record method identity differs from frozen roster')
    metadata = record['reconstruction_metadata']
    for field, spec_field in IDENTITY_FIELDS.items():
        if metadata.get(field) != spec.get(spec_field):
            raise ValueError(f'Method identity differs for {field}: {record["case_id"]}')


def biopsy_value(record: dict, paper_id: str) -> tuple[float, tuple | None]:
    if paper_id in PARTIAL:
        if record.get('fixed_label_evaluation') is not None:
            raise ValueError('Unexpected fixed-label endpoint for a partial method')
        return float(record['metrics']['grf']), None
    fixed = record['fixed_label_evaluation']
    if not fixed or fixed['status'] != 'success':
        raise ValueError('Missing closed-state fixed-label AD-F1')
    if fixed['semantics']['endpoint_filter'] != 'bilateral_both_endpoints_in_fixed_labels':
        raise ValueError('Fixed-label filtering semantics changed')
    target = (tuple(fixed['fixed_labels']), fixed['metrics']['counts']['true_unique_pair_count'])
    return float(fixed['metrics']['ad_f1']), target


def bootstrap_means(values: np.ndarray, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    """Shared truth-block resamples for every column, with bounded workspace."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Finite block-by-endpoint matrix required')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty((draws, values.shape[1]))
    for start in range(0, draws, 512):
        stop = min(start + 512, draws)
        indices = rng.integers(0, len(values), size=(stop-start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return means


def effect_summary(values: np.ndarray, bootstrap: np.ndarray) -> dict:
    return {
        'count': len(values), 'mean': float(values.mean()),
        'median': float(np.median(values)),
        'lower': float(np.quantile(bootstrap, .025)),
        'upper': float(np.quantile(bootstrap, .975)),
        'wins': int(np.count_nonzero(values > 1e-12)),
        'ties': int(np.count_nonzero(np.abs(values) <= 1e-12)),
        'losses': int(np.count_nonzero(values < -1e-12)),
    }


def load_evidence(run_root: Path, source: Path) -> tuple:
    figure_inventory = {}
    for line in (source / 'checksums.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        figure_inventory[name.lstrip('*')] = digest
    for name in ('evidence-summary.json', 'main-figure-data.csv', 'primary-block-effects.csv'):
        if sha256(source / name) != figure_inventory.get(name):
            raise ValueError(f'Frozen publication input changed: {name}')
    evidence = json.loads((source / 'evidence-summary.json').read_text())
    closure = run_root / 'run_checksums.sha256'
    if sha256(closure) != evidence['input_evidence']['internal_run_checksum_sha256']:
        raise ValueError('Frozen internal-run checksum inventory changed')
    expected_manifest = None
    for line in closure.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name.lstrip('*') == 'run_manifest.json':
            expected_manifest = digest
    if sha256(run_root / 'run_manifest.json') != expected_manifest:
        raise ValueError('Frozen run manifest changed')
    run = json.loads((run_root / 'run_manifest.json').read_text())
    if run['status'] != 'complete' or run['completed_record_count'] != 5600:
        raise ValueError('All 5600 frozen records are required')
    specs = {row['arm_id']: row for row in run['arm_specs']}
    for arm, spec in specs.items():
        # The frozen serializer omits optional None fields and uses JSON lists.
        current = json.loads(json.dumps(asdict(ARM_SPEC_BY_ID[arm])))
        if any(current.get(key) != spec.get(key) for key in current.keys() | spec.keys()):
            raise ValueError(f'Current method specification differs from frozen {arm}')
    if set(specs) != set(PAPER_ID_BY_ARM):
        raise ValueError('Frozen and paper method rosters differ')
    data, targets, records = {}, {}, []
    seeds = {}
    for entry in run['record_inventory']:
        record = checked_json(run_root, entry)
        if record['status'] != 'success':
            raise ValueError('Do not silently omit a failed record')
        audit_identity(record, specs[record['arm_id']])
        case = record['case_id']
        seed = record['reconstruction_metadata']['reconstruction_seed']
        if seeds.setdefault(case, seed) != seed:
            raise ValueError('Methods no longer share the case reconstruction seed')
        condition = record['condition_id']
        if condition not in BIOPSY_COUNT_CONDITION_IDS:
            continue
        name = PAPER_ID_BY_ARM[record['arm_id']]
        block = record['block_index']
        value, target = biopsy_value(record, name)
        if target is not None and targets.setdefault(block, target) != target:
            raise ValueError('Fixed scoring target changed between methods or biopsy counts')
        key = (block, name, condition)
        if key in data:
            raise ValueError('Duplicate block-method-condition')
        data[key] = value
        records.append({'block_index': block, 'paper_id': name, 'condition_id': condition,
                        'biopsies': COUNTS[BIOPSY_COUNT_CONDITION_IDS.index(condition)],
                        'metric': 'native_grf' if name in PARTIAL else 'fixed_label_ad_f1',
                        'value': value, 'record_path': entry['path'], 'record_sha256': entry['sha256']})
    columns = [(name, condition) for name in METHODS for condition in BIOPSY_COUNT_CONDITION_IDS]
    matrix = np.array([[data[b, n, c] for n, c in columns] for b in range(100)])
    draws = bootstrap_means(matrix)
    points, endpoints = [], {}
    for j, (name, condition) in enumerate(columns):
        summary = effect_summary(matrix[:,j], draws[:,j])
        points.append({'paper_id': name, 'condition_id': condition,
                       'biopsies': COUNTS[BIOPSY_COUNT_CONDITION_IDS.index(condition)],
                       'metric': 'native_grf' if name in PARTIAL else 'fixed_label_ad_f1',
                       'count': 100, 'mean': summary['mean'],
                       'lower': summary['lower'], 'upper': summary['upper']})
    for j, name in enumerate(METHODS):
        first, last = j*4, j*4+3
        endpoints[name] = effect_summary(matrix[:,last]-matrix[:,first], draws[:,last]-draws[:,first])
    historical = evidence['prespecified_descriptive_results']['biopsy_count']
    for condition in BIOPSY_COUNT_CONDITION_IDS:
        j = columns.index(('M6',condition))
        if not np.isclose(matrix[:,j].mean(), historical['curves'][condition]['mean'], rtol=0, atol=1e-14):
            raise ValueError('M6 fixed-label means do not reproduce the existing report')
    old_endpoint = historical['endpoint_H24_L5_minus_H24_L2']
    endpoints['M6']['lower'] = old_endpoint['bootstrap_interval']['lower']
    endpoints['M6']['upper'] = old_endpoint['bootstrap_interval']['upper']
    interactions = {}
    for left, right in (('M6','M5'), ('M7','Pooled-Q')):
        i, j = METHODS.index(left)*4, METHODS.index(right)*4
        values = (matrix[:,i+3]-matrix[:,i])-(matrix[:,j+3]-matrix[:,j])
        boot = (draws[:,i+3]-draws[:,i])-(draws[:,j+3]-draws[:,j])
        interactions[f'{left}_minus_{right}_five_minus_two'] = effect_summary(values, boot)
    summary = {
        'schema': 'ctbf-confirmation-figure-v3', 'audited_method_records': 5600,
        'biopsy_curve_records': len(records), 'blocks': 100,
        'fixed_targets_identical_across_methods_and_counts': len(targets) == 100,
        'case_seeds_shared_across_methods': len(seeds),
        'prespecified_biopsy_count_method': 'M6',
        'other_curves_and_interactions': 'post_hoc_descriptive_extension_no_hypothesis_tests',
        'partial_endpoint': 'native_grf_not_fixed_label',
        'closed_state_endpoint': 'bilateral_fixed_label_ad_f1_on_H24_L2_labels',
        'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED,
        'M6_endpoint_interval': 'retained_from_frozen_report',
        'endpoints_five_minus_two': endpoints, 'gain_contrasts': interactions,
        'source_sha256': {str(p):sha256(p) for p in [closure, run_root/'run_manifest.json',
            source/'evidence-summary.json', source/'main-figure-data.csv', source/'primary-block-effects.csv']},
    }
    identities = [{'paper_id':PAPER_ID_BY_ARM[arm], **spec} for arm,spec in specs.items()]
    return points, records, identities, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(source: Path, output: Path, points: list[dict]) -> None:
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'ctbf-paper-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (source/'main-figure-data.csv').open() as handle:
        height_rows = list(csv.DictReader(handle))
    evidence = json.loads((source/'evidence-summary.json').read_text())
    label = lambda name: f'$M_{name[1:]}$' if name.startswith('M') and name[1:].isdigit() else name
    def decorate(axis):
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y', color='#D8DEE5', linewidth=.65)
    def curve(axis, x, rows, name, frozen=False):
        y=np.array([float(row['mean']) for row in rows])
        lo=np.array([float(row['bootstrap_lower' if frozen else 'lower']) for row in rows])
        hi=np.array([float(row['bootstrap_upper' if frozen else 'upper']) for row in rows])
        color, marker, style=STYLES[name]
        axis.errorbar(x,y,yerr=[y-lo,hi-y],color=color,marker=marker,linestyle=style,
                      linewidth=1.7,markersize=4.5,capsize=2.5,label=label(name))
    with plt.rc_context({'font.size':12, 'axes.titlesize':13, 'axes.labelsize':12,
                         'xtick.labelsize':11, 'ytick.labelsize':11, 'legend.fontsize':10.5,
                         'pdf.fonttype':42, 'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(13.2,5.7),layout='constrained')
        grid=fig.add_gridspec(2,3,width_ratios=(1.12,1,1))
        count_partial=fig.add_subplot(grid[0,0]); count_full=fig.add_subplot(grid[1,0])
        partial=fig.add_subplot(grid[:,1]); full=fig.add_subplot(grid[:,2])
        for axis in (count_partial,count_full,partial,full): decorate(axis)
        for axis,names in ((count_partial,PARTIAL),(count_full,FULL)):
            for name in names:
                curve(axis,COUNTS,[r for r in points if r['paper_id']==name],name)
            axis.set_xticks(COUNTS)
            axis.legend(frameon=False,ncol=2 if names==PARTIAL else 3,fontsize=10,
                        loc='upper left' if names==PARTIAL else 'lower right')
        count_partial.set_title('A  More biopsies (H = 24)',loc='left',fontweight='bold')
        count_partial.set_ylabel('Rooted GRF')
        count_partial.set_ylim(.075,.25)
        count_full.set_xlabel('Number of biopsies')
        count_full.set_ylabel('Fixed-label AD-F1')
        count_full.set_ylim(.26,.9)
        for axis,names in ((partial,('M0','MEDICC2','M2')),(full,('Pooled-Q','M5','M6','M7'))):
            for name in names:
                rows=[next(r for r in height_rows if r['paper_id']==name and int(r['height'])==h) for h in (14,24,34)]
                curve(axis,(14,24,34),rows,name,frozen=True)
            axis.set_xticks((14,24,34)); axis.set_xlabel('True-tree height')
            axis.set_box_aspect(1)
            axis.legend(frameon=False,ncol=1 if axis is partial else 2)
        partial.set_title('B  Partial-tree recovery',loc='left',fontweight='bold')
        partial.set_ylabel('Rooted GRF similarity')
        full.set_title('C  Closed-state recovery',loc='left',fontweight='bold')
        full.set_ylabel('AD-F1')
        for suffix in ('pdf','svg','png'):
            kwargs={'metadata':{'CreationDate':None,'ModDate':None}} if suffix=='pdf' else {}
            fig.savefig(output/f'main-figure.{suffix}',dpi=220,facecolor='white',**kwargs)
        plt.close(fig)
        fig,axis=plt.subplots(figsize=(6,4),layout='constrained'); decorate(axis)
        with (source/'primary-block-effects.csv').open() as handle:
            values=sorted(float(row['block_effect']) for row in csv.DictReader(handle))
        primary=evidence['registered_primary']; ci=primary['bootstrap_interval']
        axis.scatter(range(1,101),values,s=18,color='#263B70',zorder=3)
        axis.axhline(0,color='#7A1F1F',linestyle='--',linewidth=1)
        axis.axhspan(ci['lower'],ci['upper'],color='#B8DED9',alpha=.8)
        axis.axhline(primary['block_effect_summary']['mean'],color='#087F8C',linewidth=1.6)
        axis.set(xlabel='History blocks, ordered by effect',ylabel=r'Mean $M_2-M_0$ rooted-GRF effect',xlim=(0,101))
        axis.set_title('Prespecified paired effect: 100/100 positive blocks',loc='left')
        for suffix in ('pdf','svg','png'):
            kwargs={'metadata':{'CreationDate':None,'ModDate':None}} if suffix=='pdf' else {}
            fig.savefig(output/f'primary-block-effects.{suffix}',dpi=220,facecolor='white',**kwargs)
        plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('paper/figures/v5-confirmation-v1'))
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError('Use a new output directory. Delivered figures are immutable.')
    points,records,identities,summary=load_evidence(args.run_root,args.source)
    args.output_root.mkdir(parents=True)
    write_csv(args.output_root/'biopsy-count-curves.csv',points)
    write_csv(args.output_root/'biopsy-count-blocks.csv',records)
    (args.output_root/'method-identities.json').write_text(json.dumps(identities,indent=2)+'\n')
    (args.output_root/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    render(args.source,args.output_root,points)
    print('complete: 5600 method identities verified; 2800 biopsy-count records; fixed targets unchanged')


if __name__ == '__main__':
    main()
