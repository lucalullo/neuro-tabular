# NeuroTabular 0.3.1

NeuroTabular 0.3.1 is a performance and product-hardening patch release. It
preserves the 0.3.0 algorithm, public API, defaults, and predictive behavior
while reducing avoidable CPU preprocessing overhead in prediction paths.

## Highlights

- Faster CPU `predict` and `predict_proba` paths for already-valid numeric and
  already-ordered DataFrame inputs.
- Bit-for-bit behavior preservation on frozen release validation, including the
  30-case binary equivalence panel and representative multiclass/regression
  checks.
- Additional product-path regression tests covering conversion, schema order,
  memory ownership, nullable numeric inputs, and mixed storage.
- Distribution-hygiene checks for wheel/sdist contents and accidental local or
  private artifact leakage.
- Release-branch CI coverage in addition to the existing Linux, Windows,
  minimum-PyTorch, Ruff, build, and installed-artifact checks.

On the internal P35 CPU/WSL validation panel, median latency improved by about
31% for `predict` and 36% for `predict_proba` versus the public 0.3.0 path.
Actual improvements depend on hardware, dependency versions, dataset shape, and
workload. See [docs/PERFORMANCE_0_3_1.md](docs/PERFORMANCE_0_3_1.md).

NeuroTabular remains experimental and pre-1.0. This patch makes no claim of
state-of-the-art performance, universal superiority over tree-based models, or
improved predictive accuracy. Physical CUDA hardware validation remains pending.

See the [usage guide](docs/USAGE.md), [API reference](docs/API.md), and
[changelog](CHANGELOG.md) for details.
