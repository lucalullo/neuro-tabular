"""Present existing authenticated paired intervals. No inference or resampling."""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

NAMES = {
    'neurotabular200': 'NeuroTabular 2.0.0', 'neurotabular210': 'NeuroTabular 2.1.0',
    'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost',
    'realmlp': 'RealMLP', 'tabm': 'TabM', 'ft_transformer': 'FT-Transformer',
}
FAMILIES = {
    'bank_marketing': 'Bank Marketing', 'climate_crashes': 'Climate simulation crashes',
    'anneal': 'Anneal', 'glass': 'Glass', 'gas_nox': 'Gas NOX', 'fiat500': 'Fiat 500',
}
COLORS = {
    'neurotabular200': '#666666', 'xgboost': '#0072B2', 'lightgbm': '#009E73',
    'catboost': '#D55E00', 'ft_transformer': '#CC79A7',
    'realmlp': '#56B4E9', 'tabm': '#E69F00',
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def coordinates(record, baseline, metric):
    interval = record.get('paired_quality_uncertainty', {})
    available = interval.get('status') == 'MEASURED'
    if metric == 'balanced_accuracy':
        point = 100 * (baseline['metrics'][metric] - record['metrics'][metric])
        limits = interval.get('balanced_accuracy_difference_ci95') if available else None
        bounds = [-100 * limits[1], -100 * limits[0]] if limits is not None else None
    elif record['task'] == 'regression':
        point = record['metrics']['rmse'] / baseline['metrics']['rmse']
        bounds = interval.get('ci95') if available else None
    else:
        point = record['metrics']['nll'] - baseline['metrics']['nll']
        bounds = interval.get('ci95') if available else None
    return point, bounds


def rows_for(records, task, track, metric):
    references = {(r['family'], r['seed'], r['track']): r for r in records
                  if r['method'] == 'neurotabular210' and r['status'] == 'MEASURED'}
    rows = []
    for record in records:
        if record['task'] != task or record['track'] != track or record['method'] == 'neurotabular210':
            continue
        baseline = references.get((record['family'], record['seed'], track))
        if record['status'] != 'MEASURED' or baseline is None:
            continue
        if track == 'hpo4' and not all(r.get('hpo_complete_search') is True for r in (record, baseline)):
            continue
        point, bounds = coordinates(record, baseline, metric)
        rows.append(dict(family=record['family'], method=record['method'], seed=record['seed'],
                         point=point, ci95=bounds))
    return sorted(rows, key=lambda r: (r['family'], r['method'], r['seed']))


def selftest():
    native = dict(metrics=dict(balanced_accuracy=.8, nll=.4, rmse=2))
    other = dict(task='binary', metrics=dict(balanced_accuracy=.7, nll=.5),
                 paired_quality_uncertainty=dict(status='MEASURED', ci95=[.02, .15],
                     balanced_accuracy_difference_ci95=[-.2, -.05]))
    point, bounds = coordinates(other, native, 'balanced_accuracy')
    assert abs(point - 10) < 1e-12 and bounds == [5, 20]
    point, bounds = coordinates(other, native, 'primary')
    assert abs(point - .1) < 1e-12 and bounds == [.02, .15]
    other['paired_quality_uncertainty'] = dict(status='UNAVAILABLE')
    assert coordinates(other, native, 'primary')[1] is None
    other = dict(task='regression', metrics=dict(rmse=3),
                 paired_quality_uncertainty=dict(status='MEASURED', ci95=[1.2, 1.7]))
    assert coordinates(other, native, 'primary') == (1.5, [1.2, 1.7])
    return dict(PASS=True, pure_controls=4, model_operations=0, statistical_tests_added=0)


def render(result, output):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    artifacts = []
    for track in ('default', 'hpo4'):
        for task in ('binary', 'multiclass', 'regression'):
            for metric in (('primary',) if task == 'regression' else ('primary', 'balanced_accuracy')):
                rows = rows_for(result['records'], task, track, metric)
                if not rows:
                    continue
                families = sorted({r['family'] for r in rows})
                count = max(sum(r['family'] == f for r in rows) for f in families)
                fig, axes = plt.subplots(1, len(families), figsize=(14, max(5.8, .34 * count + 2)),
                                         squeeze=False, layout='constrained')
                for axis, family in zip(axes[0], families):
                    selected = [r for r in rows if r['family'] == family]
                    reference = 1 if task == 'regression' else 0
                    if task == 'regression':
                        axis.axvspan(.99, 1.01, color='#EEEEEE', zorder=0)
                    elif metric == 'balanced_accuracy':
                        axis.axvspan(-.5, .5, color='#EEEEEE', zorder=0)
                    axis.axvline(reference, color='#333333', linewidth=.8)
                    for index, row in enumerate(selected):
                        color = COLORS.get(row['method'], '#333333')
                        bounds = row['ci95']
                        if bounds is not None:
                            axis.hlines(index, bounds[0], bounds[1], colors=color, linewidth=1.7)
                            axis.plot(bounds, [index, index], '|', color=color, markersize=6)
                        axis.scatter([row['point']], [index], s=32, edgecolors=color,
                                     facecolors=color if bounds is not None else 'none', zorder=3)
                    axis.set_yticks(range(len(selected)),
                        [NAMES.get(r['method'], r['method']) + f" · seed {r['seed']}" for r in selected],
                        fontsize=8)
                    axis.invert_yaxis()
                    axis.set_title(FAMILIES.get(family, family), fontsize=12, loc='left')
                    axis.grid(axis='x', alpha=.2)
                    axis.spines[['top', 'right', 'left']].set_visible(False)
                    if task == 'regression':
                        axis.set_xlabel('Competitor / NeuroTabular 2.1.0 RMSE\nHigher favors NeuroTabular')
                    elif metric == 'balanced_accuracy':
                        axis.set_xlabel('NeuroTabular minus competitor balanced accuracy (pp)\nHigher favors NeuroTabular')
                    else:
                        axis.set_xlabel('Competitor minus NeuroTabular 2.1.0 NLL\nHigher favors NeuroTabular')
                fig.suptitle(f'Paired TEST evidence · {task} · {track}', fontsize=15)
                fig.supxlabel('Existing 95% intervals, conditional on frozen models and declared TEST units; no multiplicity correction.\n'
                    'Hollow marker: interval unavailable. Seeds are separate model repetitions. Complete failure roster is in PER_SEED.csv.',
                    fontsize=8)
                name = f'{track}-{task}-{metric}-paired-intervals.png'
                fig.savefig(output / name, dpi=170)
                plt.close(fig)
                artifacts.append(dict(name=name, plotted_cells=len(rows),
                    intervals_available=sum(r['ci95'] is not None for r in rows)))
    return artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--export-verification', type=Path)
    parser.add_argument('--expected-results-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps(selftest()))
        return
    source = args.report / 'results.json'
    if args.expected_results_sha256:
        verification = dict(PASS=digest(source) == args.expected_results_sha256,
            scope='PUBLISHED_BENCHMARK_RESULT_MANIFEST',
            source_results_sha256=args.expected_results_sha256)
    elif args.export_verification:
        verification = read(args.export_verification)
    else:
        parser.error('Provide --expected-results-sha256 or a local --export-verification manifest')
    assert verification['PASS'] is True and verification['scope'] in (
        'TECHNICAL_INTERNAL_REPORT_EXPORT_GATE_ONLY', 'LOCAL_REPRODUCTION_BOUND_REPORT',
        'PUBLISHED_BENCHMARK_RESULT_MANIFEST')
    assert digest(source) == verification['source_results_sha256']
    result = read(source)
    assert result['synthetic'] is False and len(result['records']) == 288
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    artifacts = render(result, args.output)
    for row in artifacts:
        path = args.output / row['name']
        row.update(bytes=path.stat().st_size, sha256=digest(path))
    receipt = dict(source_results_sha256=digest(source), source_code_sha256=digest(Path(__file__)),
        source_verification_scope=verification['scope'],
        files=artifacts, model_operations=0, scoring_operations=0, statistical_tests_added=0,
        scope='Presentation only, using already computed paired conditional intervals')
    with (args.output / 'FIGURE_PROVENANCE.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
