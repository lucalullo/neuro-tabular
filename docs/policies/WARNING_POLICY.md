# Warning and error policy

Adopted for the 1.0.0 stable API scope. Existing runtime behavior is retained.
No current API is deprecated and no new runtime emitter is added.

| Category | Use |
| --- | --- |
| RuntimeWarning | A runtime failure with safe recovery: an automatic CUDA probe failure warns, records a reason and uses CPU |
| UserWarning | Actionable nonfatal user condition, only when the user should act; no current new emitter |
| FutureWarning | Public deprecation/default/behavior transitions, with replacement and schedule |
| DeprecationWarning | Developer-facing notices, not the main notice for public estimator users |

Ordinary reordered columns, missing features' **values**, unseen feature levels
and supported nullable dtypes are normal and quiet. Missing/extra/duplicate
column **names**, invalid targets/weights/device strings or unsafe values are
fatal existing exception paths. New eval_set target classes are errors. Explicit
CUDA requests raise on unavailability/incompatibility instead of falling back;
auto on a system reporting no CUDA chooses CPU quietly. Mocks do not prove
physical CUDA usability.

Preserve TypeError/ValueError/RuntimeError/NotFittedError distinctions. Exact
prose is actionable diagnostics rather than a general parsing API. Do not change
warning categories or convert fatal errors into permissive warnings silently.

Future warnings use the tested internal format:

```text
NeuroTabular: {feature} is deprecated since {since}; removal no earlier than {remove_in}. Use {replacement} instead.
```

Choose stacklevel to identify the user's call. Keep messages stable enough for
standard same-call-site filtering; no custom global warning registry, import-
time/per-row notices or repeated warnings merely because an option is labeled
experimental. Users retain normal filter control. No automatic version/load
warning is justified from missing old-model metadata.

The helper's numeric notice floor supplements, rather than replaces, actual
release-history review and [deprecation rules](DEPRECATION_POLICY.md). Tests may
exercise fictitious deprecations; doing so does not deprecate a real parameter.
