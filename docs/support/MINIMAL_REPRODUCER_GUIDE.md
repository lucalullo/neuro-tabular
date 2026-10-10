# Minimal reproducer and safe support report

P41 local development, public release 0.3.1. No private dataset or saved model
upload is required. Start with invented values and the smallest failing call.

## Runnable starting point

```python
import numpy as np
import pandas as pd
from neurotabular import NeuroTabularClassifier

X = pd.DataFrame(
    {
        "amount": np.linspace(-2, 2, 40),
        "city": pd.Categorical(["Rome", "Milan", None, "Rome"] * 10),
        "active": pd.Series([True, False, None, True] * 10, dtype="boolean"),
    }
)
X.loc[0, "amount"] = np.nan
y = np.arange(len(X)) % 2
model = NeuroTabularClassifier(
    hidden_dim=8,
    n_blocks=1,
    max_epochs=1,
    patience=1,
    device="cpu",
    random_state=7,
)
model.fit(X, y)
print(model.predict(X.iloc[:3]).shape)
print(model.predict_proba(X.iloc[:3]).shape)
print(model.classes_, model.n_features_in_, model.device_)
```

Expected shapes: `(3,)`, `(3, 2)`; classes `[0 1]`, three features, CPU.
Change only what is needed to reproduce the issue. Include the exact estimator
configuration, target construction, random_state and failing call. For regression
use NeuroTabularRegressor with a finite real one-dimensional target and omit
predict_proba; for multiclass supply at least three sufficiently represented
discrete labels. These short runs demonstrate software behavior, not quality.

## Environment without private information

From a development checkout run `python examples/support_environment.py`.
Alternatively import `environment_report` from `examples.support_environment`
when the checkout is importable. The optional helper is an example, not an
installed NeuroTabular API or CLI. It returns only NeuroTabular/Python/Torch/
NumPy/pandas/sklearn versions, OS family, requested/resolved device labels,
CUDA availability and Torch CUDA version. To include a fitted device explicitly:

```python
from examples.support_environment import environment_report

print(environment_report(requested_device="cpu", resolved_device=model.device_))
```

The helper does not load a model, resolve a device, launch a CUDA probe, read
environment variables or collect host/user identifiers. CUDA availability is
not proof of usable hardware. Its requested device defaults to auto; resolved
device is unknown until supplied. Review output before sharing. It requires
working imports; if import fails, report Python/package versions obtainable
locally and the reviewed traceback, without reinstalling or changing security
policy solely to generate a report.

## Error and recovery table

| Symptom | Recovery |
| --- | --- |
| NotFittedError | Fit the same estimator first; clones start unfitted |
| Missing/extra/duplicate columns | Supply the original unique feature names; fix upstream selection or rename duplicates before fitting |
| Columns reordered | Accepted and aligned by name; no need to refit |
| Numeric column contains text | Clean that column; if it is categorical, declare categorical_features and refit |
| New feature category | Accepted as unknown, without learning a new vocabulary; it differs from a new eval_set target class, which is rejected |
| Invalid y shape | Pass a one-dimensional single-output target aligned by row position; X index labels do not align y automatically |
| Non-finite regression target | Correct NaN/infinite values before fit; do not suppress the error |
| Invalid sample_weight | Supply finite nonnegative one-dimensional aligned weights with positive total |
| Invalid device | Use cpu, auto, cuda or cuda:N; explicit unavailable CUDA raises, auto with no CUDA uses CPU |
| Reported CUDA fails its probe | Auto warns and recovers to CPU; inspect device_info_; explicit CUDA requires a usable compatible runtime/hardware or choose CPU |
| load/predict fails after upgrade | Keep the original trusted artifact and pinned environment; test the documented compatibility scope before migration |

After a failed refit, the previous fitted model remains usable. set_params
changes pending next-fit settings; it does not retrain or move a saved model.
Do not edit fitted attributes to reconfigure it. Prediction preserves row order;
bitwise comparisons require the same ordered inputs/runtime/thread policy.
Different row/batch layouts can differ at floating-point precision.

## Issue contents

Supply expected and actual behavior, the synthetic script, version report and
complete relevant traceback (or "No exception"). Review traceback filenames and
messages for usernames, paths, secrets and private values; replace those parts
with placeholders while preserving error classes and useful stack context.
Do not attach private frames, model pickles, credentials, full pip configuration,
environment-variable dumps or machine identifiers. Report security issues via
SECURITY.md rather than a public bug report.

Only load trusted pickle/joblib artifacts. Same-version/build pinned CPU
round-trips are the tested scope; patch upgrades are best effort, and minor,
major, dependency, device and OS migration are not guaranteed. Keep a trusted
probe frame and expected outputs privately alongside the original environment.
