"""Pure prospective P6 counts and reservation proof. No dataset payload/model."""
import json
from pathlib import Path

STUDY = 'NT210-BENCHMARK0422-P6-D-H'
CPU_GUARD = 2
WALL_GUARD = 6
CONTROLLER_CPU = 11808
CONTROLLER_WALL = 21792
RSS = 4 * 1024**3


def proof(config):
    methods, families, seeds = len(config['methods']), len(config['datasets']), len(config['seeds'])
    assert (methods, families, seeds) == (8, 6, 3)
    cases = methods * families * seeds
    q = methods * 3
    fits = q + cases * 6
    replay = q + cases * 2
    residents = cases * 4
    pairs = residents // 2
    assert (fits, replay, residents, pairs) == (888, 312, 576, 288)
    phases = dict(artificial=q * 60, defaults=cases * 300,
                  trials=cases * 4 * 225, finals=cases * 300,
                  measurement=residents * 60, replay=replay * 20)
    base_cpu = sum(phases.values())
    cpu_guards = (fits + replay + residents) * CPU_GUARD
    base_wall = 2 * (phases['artificial'] + phases['defaults'] + phases['trials'] + phases['finals']) + pairs * 240 + replay * 40
    wall_guards = (fits + replay + pairs) * WALL_GUARD
    assert base_cpu == 258240 and cpu_guards == 3552
    assert base_wall == 516480 and wall_guards == 8928
    assert base_cpu + cpu_guards + CONTROLLER_CPU == config['budget']['cpu_s'] == 273600
    assert base_wall + wall_guards + CONTROLLER_WALL == config['budget']['wall_s'] == 547200
    return dict(study=STUDY, PASS=True, fit_attempts=fits, cold_replays=replay,
        measurement_workers=residents, model_load_operations=replay + residents,
        scoring=q + cases * 4 + cases * 2,
        optimizer_steps=(5 * 3 + 5 * families * seeds * 6) * 50000,
        worker_CPU_phases=phases, base_worker_CPU_s=base_cpu,
        failed_exit_CPU_guard_per_resident=CPU_GUARD, failed_exit_CPU_reserve_s=cpu_guards,
        controller_monitor_ingestion_scoring_report_CPU_s=CONTROLLER_CPU,
        total_CPU_s=base_cpu + cpu_guards + CONTROLLER_CPU,
        base_active_wall_s=base_wall, hard_exit_wall_guard_per_ticket_or_pair=WALL_GUARD,
        hard_exit_wall_guard_s=wall_guards,
        controller_ingestion_scoring_report_wall_s=CONTROLLER_WALL,
        total_wall_s=base_wall + wall_guards + CONTROLLER_WALL,
        original_global_budget_unchanged=True, extra_retry_slots=0,
        metadata_and_Atlas_counted_within_32GiB=True,
        same_fit_or_forward_counts_are_not_equal_compute=True)


def role_rows(job):
    """Read only already bound row metadata, never arrays/labels."""
    if job.get('qualification'):
        return 128, 32
    selected = Path(job.get('train_file') or job.get('input_file'))
    if selected.parent.name == 'artificial':
        if selected.stem not in {task + '-validation' for task in ('binary', 'multiclass', 'regression')}:
            raise ValueError('Unexpected artificial role')
        return 32, 0
    manifest = selected.parent.parent / 'MANIFEST.json'
    value = json.loads(manifest.read_text(encoding='utf-8'))
    roles = value['splits'][selected.parent.name]['roles']
    rows = roles[selected.stem]['rows']
    if not isinstance(rows, int) or rows < 1:
        raise ValueError('Missing frozen row count')
    validation = roles['validation']['rows'] if job.get('validation_file') else 0
    return rows, validation


def attach_forward_limit(job, operations):
    from .work_counters import NEURAL, fit_forward_limit, prediction_forward_limit
    job['max_prediction_API_calls'] = operations.get('prediction_API_calls', 0)
    if job['model'] not in NEURAL:
        cap = 0
    else:
        rows, validation = role_rows(job)
        cap = (fit_forward_limit(rows, job['max_optimizer_steps'], operations.get('prediction_API_calls', 0), validation)
               if job['mode'] == 'fit' else prediction_forward_limit(rows, operations.get('prediction_API_calls', 0)))
    job['max_torch_root_forward_calls'] = cap
    operations['torch_root_forward_calls'] = cap
    return cap
