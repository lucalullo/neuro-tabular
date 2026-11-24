"""Prospective learned-root forward safety guard, not FLOPs or equal compute.

No ML import at module load. Only the method's learned forward entry is wrapped;
nested calls through the same entry are counted once. Trees report NA, not an
invented neural forward count. Partial counters never replace reservations.
"""
from contextlib import contextmanager
from functools import wraps
from importlib import import_module
import threading

NEURAL = {'neurotabular200', 'neurotabular210', 'realmlp', 'tabm', 'ft_transformer'}
ROOTS = {
    'neurotabular200': ('neurotabular.network', 'TabularNetwork'),
    'neurotabular210': ('neurotabular.network', 'TabularNetwork'),
    'realmlp': ('pytabkit.models.nn_models.base', 'SequentialLayer'),
    'tabm': ('pytabkit.models.nn_models.tabm', 'Model'),
    'ft_transformer': ('pytabkit.models.nn_models.rtdl_resnet', 'FT_Transformer'),
}


class ForwardCounter:
    def __init__(self, method, limit):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError('Finite integral forward limit required')
        self.method, self.limit = method, limit
        self.calls = 0
        self.by_class = {}
        self.local = threading.local()
        self.fit_started = self.fit_completed = 0
        self.load_started = self.load_completed = 0
        self.internal_refits = 0
        self.blocked_refit_requests = 0
        self.optimizer_steps = 0
        self._restore = []

    def install_class(self, cls):
        owned = 'forward' in cls.__dict__
        original = cls.forward
        counter = self

        @wraps(original)
        def wrapped(model, *args, **kwargs):
            depth = getattr(counter.local, 'depth', 0)
            if depth == 0:
                if counter.calls >= counter.limit:
                    raise RuntimeError('LEARNED_ROOT_FORWARD_LIMIT')
                counter.calls += 1
                key = type(model).__module__ + '.' + type(model).__qualname__
                counter.by_class[key] = counter.by_class.get(key, 0) + 1
            counter.local.depth = depth + 1
            try:
                return original(model, *args, **kwargs)
            finally:
                counter.local.depth = depth

        cls.forward = wrapped
        self._restore.append((cls, original, owned))

    def install(self):
        if self.method not in NEURAL:
            if self.limit != 0:
                raise ValueError('Trees must have a zero neural reservation')
            return
        module, name = ROOTS[self.method]
        self.install_class(getattr(import_module(module), name))
        if self.method.startswith('neurotabular'):
            base = import_module('neurotabular._base')
            # A forbidden hidden full-data refit is rejected before any update.
            if hasattr(base, 'train_fixed_epochs'):
                original = base.train_fixed_epochs
                def forbidden(*args, **kwargs):
                    self.blocked_refit_requests += 1
                    raise RuntimeError('UNADMITTED_INTERNAL_REFIT')
                base.train_fixed_epochs = forbidden
                self._restore.append((base, original, 'train_fixed_epochs'))

    def close(self):
        for cls, original, owned in reversed(self._restore):
            if owned == 'train_fixed_epochs':
                cls.train_fixed_epochs = original
            elif owned:
                cls.forward = original
            else:
                del cls.forward
        self._restore.clear()

    def snapshot(self):
        return dict(torch_root_forward_calls=self.calls,
            forward_scope='learned forward entry per method; nested selected entries counted once; not FLOPs',
            forward_applicability='MEASURED' if self.method in NEURAL else 'NOT_APPLICABLE_TREE',
            forward_root_classes=dict(self.by_class), fit_calls_started=self.fit_started,
            fit_calls_completed=self.fit_completed, internal_refits=self.internal_refits,
            blocked_refit_requests=self.blocked_refit_requests,
            model_load_operations=self.load_completed,
            model_load_calls_started=self.load_started,
            optimizer_steps=self.optimizer_steps)


def fit_forward_limit(rows, max_steps=50000, external_API=0, validation_rows=0):
    """New finite safety envelope, deliberately conservative, not a free epoch cap.

    Four phase allowances (training, internal evaluation, preparation and final
    evaluation), max_steps+one rejected-step attempt, and 32-row units matching
    the already frozen middle inference workload. No training batch changes.
    This is a NEW safety hard stop, not a theoretical backend bound; unusual
    backend work may truncate before CPU and must be reported, never extended.
    """
    import math
    if min(rows, validation_rows, external_API, max_steps) < 0:
        raise ValueError('Negative work basis')
    return 4 * (max_steps + 1) * (1 + math.ceil(rows / 32)) + 4 * external_API * max(1, math.ceil(validation_rows / 32))


def prediction_forward_limit(rows, API_calls):
    import math
    # A deliberately conservative four-entry allowance per 32-row query block.
    # The guard is model-operation counting, not a claim of equal network size.
    return 4 * API_calls * max(1, math.ceil(rows / 32))
