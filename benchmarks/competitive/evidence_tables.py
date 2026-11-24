"""Readable tables from the authenticated frozen report; no models or rescoring.

These exports add no statistical tests. Family medians are descriptive, and
existing intervals retain their original conditional uncertainty scope.
"""
import argparse
import csv
import importlib.util
import json
import statistics
from hashlib import sha256
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    'qualified_claim_interpretation', Path(__file__).with_name('claim_interpretation.py'))
_CLAIMS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLAIMS)
NLL_difference_verdict = _CLAIMS.NLL_difference_verdict
balanced_accuracy_verdict = _CLAIMS.balanced_accuracy_verdict
eligibility = _CLAIMS.eligibility
ratio_verdict = _CLAIMS.ratio_verdict

KEYS = ('family', 'task', 'seed', 'track', 'method')
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_recall', 'auc',
           'nll', 'average_precision', 'rmse', 'mae', 'r2')
COSTS = ('fit_cpu_s', 'fit_wall_s', 'fit_compute_cpu_s',
         'tuning_cpu_s', 'tuning_wall_s', 'training_total_cpu_s',
         'training_total_wall_s', 'fit_peak_rss_bytes', 'inference_peak_rss_bytes',
         'model_bytes', 'cold_predict_wall_ns', 'warm_full_predict_wall_ns',
         'warm_full_predict_cpu_ns', 'fit_optimizer_steps')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def csv_file(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ['status']
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def keyed(record):
    return {key: record[key] for key in KEYS}


def numerical(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flat_records(records, prepared=None):
    roles = {} if prepared is None else {
        (source['id'], int(seed)): split['roles']['test']
        for source in prepared['sources'] for seed, split in source['splits'].items()}
    rows = []
    for record in records:
        row = dict(keyed(record), status=record['status'],
                   reason=record.get('reason', record.get('failure_stage', '')),
                   inference_A_A_status=record.get('inference_A_A_status', ''),
                   hpo_complete_search=record.get('hpo_complete_search', ''),
                   hpo_selected_index=record.get('hpo_selected_index', ''))
        for name in METRICS:
            row[name] = record.get('metrics', {}).get(name)
        for name in COSTS:
            row[name] = record.get(name)
        if roles:
            role = roles[(record['family'], record['seed'])]
            if record['status'] == 'MEASURED':
                assert role['sha256'] == record['test_sha256'], 'Throughput requires the same frozen TEST workload'
            row['test_rows'] = role['rows']
            elapsed = record.get('warm_full_predict_wall_ns')
            row['warm_full_rows_per_s_point_only'] = (
                role['rows'] * 1e9 / elapsed if record['status'] == 'MEASURED'
                and record.get('inference_A_A_status') == 'VALID'
                and numerical(elapsed) and elapsed > 0 else None)
        rows.append(row)
    return rows


def paired_records(records):
    lookup = {(r['family'], r['seed'], r['track'], r['method']): r for r in records}
    rows = []
    for record in records:
        if record['method'] == 'neurotabular210':
            continue
        baseline = lookup.get((record['family'], record['seed'], record['track'],
                               'neurotabular210'), {})
        eligible = eligibility(record, baseline)
        row = dict(keyed(record), eligible=eligible, status=record['status'],
                   reference_status=baseline.get('status', 'MISSING'),
                   uncertainty_scope='Conditional on frozen models and declared TEST units; no multiplicity adjustment')
        interval = record.get('paired_quality_uncertainty', {})
        measured = eligible and interval.get('status') == 'MEASURED'
        bounds = interval.get('ci95')
        row['primary_metric'] = interval.get('primary', '')
        row['primary_ci95_low'], row['primary_ci95_high'] = bounds or (None, None)
        ba_bounds = interval.get('balanced_accuracy_difference_ci95')
        row['balanced_accuracy_difference_ci95_low'], row['balanced_accuracy_difference_ci95_high'] = ba_bounds or (None, None)
        row['balanced_accuracy_interpretation'] = balanced_accuracy_verdict(ba_bounds, measured)
        if record['task'] == 'regression':
            row['primary_interpretation'] = ratio_verdict(bounds, .01, measured)
        else:
            row['primary_interpretation'] = NLL_difference_verdict(bounds, measured)['primary']
        if eligible:
            for name in METRICS:
                a = record.get('metrics', {}).get(name)
                b = baseline.get('metrics', {}).get(name)
                if numerical(a) and numerical(b):
                    row[name + '_competitor_minus_nt210_point_only'] = a - b
            for name in COSTS:
                a, b = record.get(name), baseline.get(name)
                if numerical(a) and numerical(b) and b > 0:
                    # Warm ratios without valid A/A are deliberately absent.
                    if name.startswith('warm_') and not eligibility(record, baseline, clock_metric=True):
                        continue
                    row[name + '_competitor_over_nt210_point_only'] = a / b
        rows.append(row)
    return rows


def family_records(records, expected_seeds):
    groups = {}
    for record in records:
        key = tuple(record[k] for k in ('family', 'task', 'track', 'method'))
        groups.setdefault(key, []).append(record)
    output = []
    for key, group in sorted(groups.items()):
        complete = {r['seed'] for r in group} == set(expected_seeds) and all(
            r['status'] == 'MEASURED' and
            (r['track'] != 'hpo4' or r.get('hpo_complete_search') is True)
            for r in group)
        row = dict(zip(('family', 'task', 'track', 'method'), key))
        row.update(complete_three_seed_panel=complete,
                   measured_seeds=sum(r['status'] == 'MEASURED' for r in group),
                   descriptive_scope='Median and range over model seeds, not a confidence interval or independent dataset replication')
        if complete:
            for name in METRICS + COSTS:
                if name.startswith('warm_') and not all(r.get('inference_A_A_status') == 'VALID' for r in group):
                    continue
                values = [r.get('metrics', {}).get(name) if name in METRICS else r.get(name) for r in group]
                if all(numerical(v) for v in values):
                    row[name + '_median'] = statistics.median(values)
                    row[name + '_seed_min'], row[name + '_seed_max'] = min(values), max(values)
        output.append(row)
    return output


def timing_records(metrology):
    rows = []
    for pair, record in sorted(metrology.items()):
        aa_id = pair.rsplit('-', 1)[0] + '-neurotabular210'
        aa_status = metrology.get(aa_id, {}).get('status', 'UNAVAILABLE')
        if not record.get('blocks'):
            rows.append(dict(pair=pair, execution_status=record['status'],
                dedicated_A_A_status=aa_status, practical_interpretation='INSUFFICIENT_EVIDENCE',
                uncertainty_scope='No completed timing blocks; the failure is retained'))
        for batch, clocks in record.get('blocks', {}).items():
            for clock, result in clocks.items():
                lo, hi = result.get('ci95', (None, None))
                rows.append(dict(pair=pair, workload=batch, clock=clock,
                    A_A=record['A_A'], execution_status=record['status'],
                    dedicated_A_A_status=aa_status,
                    interval_status=result['status'],
                    competitor_over_nt210_ratio=result.get('ratio_B_over_A'),
                    ci95_low=lo, ci95_high=hi,
                    practical_interpretation='CONTROL_ONLY' if record['A_A'] else
                        ratio_verdict(result.get('ci95'), .10,
                            measured=result['status'] == 'MEASURED', AA_valid=aa_status == 'VALID'),
                    uncertainty_scope=record['uncertainty_scope']))
    return rows


def selftest():
    def record(method, status='MEASURED', seed=42):
        return dict(family='pure_fixture', task='binary', seed=seed, track='default',
                    method=method, status=status, inference_A_A_status='INCONCLUSIVE',
                    metrics=dict(nll=.4, balanced_accuracy=.8),
                    warm_full_predict_wall_ns=10, training_total_cpu_s=2,
                    paired_quality_uncertainty=dict(status='MEASURED',
                        primary='nll_difference', ci95=[.01, .02],
                        balanced_accuracy_difference_ci95=[-.02, -.01]))
    base, other = record('neurotabular210'), record('competitor')
    pair = paired_records([base, other])[0]
    assert pair['primary_interpretation'] == 'NT210_LOWER_NLL_CONDITIONAL_INTERVAL'
    assert pair['balanced_accuracy_interpretation'] == 'NT210_USEFUL_ADVANTAGE'
    assert not any(k.startswith('warm_') for k in pair)
    assert pair['training_total_cpu_s_competitor_over_nt210_point_only'] == 1
    assert not family_records([other], (42, 2026, 1947))[0]['complete_three_seed_panel']
    complete = family_records([record('competitor', seed=s) for s in (42, 2026, 1947)], (42, 2026, 1947))[0]
    assert complete['nll_median'] == .4 and complete['measured_seeds'] == 3
    assert not any(k.startswith('warm_') for k in complete)
    failed = paired_records([base, record('competitor', 'FAILED')])[0]
    assert not failed['eligible'] and failed['primary_interpretation'] == 'INSUFFICIENT_EVIDENCE'
    timings = timing_records({
        'default-fixture-42-neurotabular210': dict(A_A=True, status='INCONCLUSIVE'),
        'default-fixture-42-competitor': dict(A_A=False, status='VALID', uncertainty_scope='pure fixture',
            blocks={'full': {'wall_ns': dict(status='MEASURED', ratio_B_over_A=2, ci95=[1.8, 2.2])}})})
    measured = next(row for row in timings if row.get('clock'))
    assert measured['dedicated_A_A_status'] == 'INCONCLUSIVE'
    assert measured['practical_interpretation'] == 'INSUFFICIENT_EVIDENCE'
    assert timing_records({'default-fixture-42-failed': dict(status='FAILED')})[0]['execution_status'] == 'FAILED'
    base.update(test_sha256='pure_fixture_hash', inference_A_A_status='VALID', warm_full_predict_wall_ns=2e9)
    prepared = {'sources': [{'id': 'pure_fixture', 'splits': {
        '42': {'roles': {'test': {'rows': 40, 'sha256': 'pure_fixture_hash'}}}}}]}
    assert flat_records([base], prepared)[0]['warm_full_rows_per_s_point_only'] == 20
    base['inference_A_A_status'] = 'INCONCLUSIVE'
    assert flat_records([base], prepared)[0]['warm_full_rows_per_s_point_only'] is None
    base['test_sha256'] = 'different_workload'
    try:
        flat_records([base], prepared)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched workload admitted for throughput')
    return dict(PASS=True, checks=14, pure_reporting_only=True,
                new_model_operations=0, benchmark_scores_created=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--export-verification', type=Path)
    parser.add_argument('--expected-results-sha256')
    parser.add_argument('--prepared-sources', type=Path)
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
    assert len({tuple(r[k] for k in ('family', 'seed', 'track', 'method')) for r in result['records']}) == 288
    assert not args.output.exists(), 'Do not overwrite previously prepared evidence'
    args.output.mkdir(parents=True)
    records = result['records']
    prepared = read(args.prepared_sources) if args.prepared_sources else None
    csv_file(args.output / 'PER_SEED.csv', flat_records(records, prepared))
    csv_file(args.output / 'PAIRED_COMPARISONS.csv', paired_records(records))
    csv_file(args.output / 'FAMILY_SUMMARY.csv', family_records(records, (42, 2026, 1947)))
    csv_file(args.output / 'TIMING_CONTROLS.csv', timing_records(result['metrology']))
    inventory = [dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(args.output.iterdir())]
    receipt = dict(source_results_sha256=digest(source), source_gate_sha256=digest(args.export_verification) if args.export_verification else None,
                   source_verification_scope=verification['scope'],
                   prepared_sources_sha256=digest(args.prepared_sources) if args.prepared_sources else None,
                   files=inventory, statistical_tests_added=0, scoring_operations=0,
                   model_operations=0, descriptive_seed_ranges_are_not_confidence_intervals=True)
    with (args.output / 'TABLE_PROVENANCE.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
