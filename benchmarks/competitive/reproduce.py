"""Public reproduction CLI for the unchanged recorded benchmark implementation.

This portable CLI is separate from the private, content-bound admission runner.
It never imports or uses private grants. Each new experiment needs its own
operator authorization and resources; the published evidence is not extended.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'prepare', 'run', 'report'))
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--environment', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.execute and args.action == 'prepare' and args.data is None:
        parser.error('prepare --execute requires --data outside the repository')
    if args.execute and args.action == 'run' and any(
        value is None for value in (args.data, args.work, args.environment)
    ):
        parser.error('run --execute requires --data, --work and --environment')
    if args.execute and args.action == 'report' and args.work is None:
        parser.error('report --execute requires --work')
    if args.execute and args.action == 'run' and os.name != 'nt':
        parser.error('This recorded process-custody benchmark requires Windows; no operation started')
    return args


def outside(path, repository):
    resolved = path.resolve()
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError('Keep raw data, models and work receipts outside the repository')
    return resolved


def main(argv=None):
    args = parse(argv)
    from . import run

    protocol = args.protocol or run.HERE / 'protocol.json'
    config = run.load(protocol)
    if args.action == 'plan' or not args.execute:
        print(json.dumps(run.plan(config), indent=2))
        return
    repository = run.HERE.parents[1].resolve()
    if args.action == 'prepare':
        data = outside(args.data, repository)
        for spec in config['datasets']:
            run.prepare_family(spec, data / spec['id'], config['seeds'])
        return
    work = outside(args.work, repository)
    binding = work / 'RUN_BINDING.json'
    if binding.exists() and 'grant_sha256' in run.load(binding):
        raise ValueError('A content-bound private run must use its original admitted entrypoint')
    if args.action == 'run':
        environment = run.load(args.environment)
        environment['_path'] = str(args.environment.resolve())
        # Do not pre-populate work: execute verifies the empty-root condition
        # before registering a new run, and verifies every completed receipt.
        run.execute(work, outside(args.data, repository), config, environment, protocol)
        return
    from . import report
    from .control import Budget, atomic_json

    if not (work / 'PREDICTIONS_SEALED.json').is_file():
        raise ValueError('Complete and seal all model and replay work before TEST scoring')
    usage = run.load(work / 'USAGE.json')  # An established, budgeted work root is required.
    if usage.get('retries') != 0:
        raise ValueError('This protocol has no retry allocation')
    budget = Budget(work, config['budget'])
    try:
        completed = report.compile_results(work, config, budget, protocol)
        report.write_report(completed, work / 'report', budget=budget)
        atomic_json(work / 'LOCAL_REPORT_MANIFEST.json', dict(
            PASS=True, scope='LOCAL_REPRODUCTION_BOUND_REPORT',
            source_results_sha256=run.digest(work / 'report/results.json'),
            source_binding_sha256=run.digest(work / 'RUN_BINDING.json'),
            private_published_result_authentication=False,
            note='This local reproduction does not replace the published report provenance'))
    finally:
        budget.checkpoint()


if __name__ == '__main__':
    main()
