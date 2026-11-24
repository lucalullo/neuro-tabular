# Public comparison for NeuroTabular 2.1.0

**Status: measured CPU pilot; 181 valid final endpoints and all failures retained.**

See [When to choose NeuroTabular](WHEN_TO_CHOOSE.md), the [scorecard](SCORECARD.md),
[measured results](results/REPORT.md) and [measurement limits](MEASUREMENT_SCOPE.md).

This bounded six-family CPU pilot compares NeuroTabular 2.0.0 and 2.1.0,
RealMLP, XGBoost, LightGBM, CatBoost, TabM and FT-Transformer. It is a comparison
of frozen releases, not a search for changes to NeuroTabular. There are two
families per task and three seeds: 42, 2026 and 1947. This scope cannot support
a claim of universal superiority.

- [Protocol and decision rules](PROTOCOL.md)
- [Sources, roles and limitations](DATASETS.md)
- [Methods, versions and search settings](METHODS.md)
- [Executable configuration](protocol.json)
- [Software and wheel hashes](software.json)
- [Prepared source/split manifest](prepared_sources.json)
- [Result status](results/status.json) and [result schema](results/schema.json)

## Reproduce

Run commands from the repository root in a dedicated **Windows x86-64,
CPython 3.12.10** environment. The optional benchmark dependencies do not change
NeuroTabular's installation requirements. Other operating systems or versions
constitute a separately labeled environment, not this exact CPU comparison.

```powershell
py -3.12 -m venv ..\nt-benchmark-env
..\nt-benchmark-env\Scripts\python -m pip install --require-hashes --only-binary=:all: -r benchmarks/competitive/requirements-windows-py312.lock
```

Obtain the original 2.0.0 wheel and the frozen 2.1.0 wheel from their release
artifacts. The second version is not declared published by this preparation.
The following command checks both versions and **every runtime module hash**,
then stages separate import roots. It never fits a model. Documentation-only
wheel rebuilds are accepted only when the pinned runtime payload is identical;
the full wheel hashes are recorded in the environment file.

```powershell
..\nt-benchmark-env\Scripts\python -m benchmarks.competitive.environment --nt200 ..\wheels\neurotabular-2.0.0-py3-none-any.whl --nt210 ..\wheels\neurotabular-2.1.0-py3-none-any.whl --stage ..\nt-imports --python ..\nt-benchmark-env\Scripts\python.exe --output ..\benchmark-environment.json --execute
..\nt-benchmark-env\Scripts\python -m unittest discover -s tests -p test_public_benchmark.py
..\nt-benchmark-env\Scripts\python -m benchmarks.competitive.reproduce plan
```

The plan is inert. Downloading the six named public sources requires the
explicit prepare command; it verifies pinned source and official-split hashes.
Keep datasets, fitted models, environments and logs outside the repository.

```powershell
..\nt-benchmark-env\Scripts\python -m benchmarks.competitive.reproduce prepare --data ..\benchmark-data --execute
```

The maintainer must approve the frozen scientific admission before the project
agent runs the next command. It starts real fits, including the synthetic-data
backend qualification. The four-candidate search, final fit and all failures
are counted. No terminal ticket is automatically retried.

```powershell
..\nt-benchmark-env\Scripts\python -m benchmarks.competitive.reproduce run --data ..\benchmark-data --work ..\benchmark-work --environment ..\benchmark-environment.json --execute
..\nt-benchmark-env\Scripts\python -m benchmarks.competitive.reproduce report --work ..\benchmark-work --execute
```

Review the sanitized report before copying verified outputs into `results/`.
Do not copy raw data, weights, environments or work receipts. Publication and
Git operations remain the maintainer's responsibility.

## What the outputs mean

Each method retains task metrics, its loaded version/configuration, complete
fit cost, tuning cost, first and warm prediction time, isolated peak RSS,
model size, replay identity and failures. Process-import and persistence costs
are included in complete fit time; API-only fit time is also recorded.

Dedicated equally loaded A/A workers qualify the timing control. An unstable
A/A makes corresponding speed ratios inconclusive; it does not invalidate
independently verified predictions. The report never reruns a noisy timing
sample to obtain a favorable result.

Plots separate tasks, tracks and dataset panels. Family summaries use equal
family weights and median seed results, not an invented common accuracy scale
for regression and classification. Failures and incomplete searches remain
visible. Use the [limitations](DATASETS.md) when interpreting any future score.

## Coverage and source fidelity

Five methods complete all six families, three seeds and both regimes: NeuroTabular 2.0.0/2.1.0, XGBoost, LightGBM and CatBoost. RealMLP/TabM were excluded by CPU-limited qualification replays. FT-Transformer has one valid natural result; five Anneal measurement pairs and six Glass cold replays stop at their limits. No timeout counts as a predictive win.

The recorded model, adapter, data, scoring and process-custody implementation is retained verbatim. The portable `reproduce` CLI is separate from the original private admission wrapper; it never reuses that grant. Narrow style/unused-name lint exemptions preserve recorded instrumentation bytes; runtime-library rules remain unchanged. Windows process-custody tests are explicitly Windows-only.

The frozen JSON describes the executable protocol; its static PREPARED_NOT_ADMITTED field does not override the measured result status. A new run requires its own operator authorization and resources. The local report command writes LOCAL_REPORT_MANIFEST.json; it does not authenticate a new run as the original published evidence.

## Regenerate the readable tables and interval plots

For the verified delivered results, check the result digest recorded in the source manifest:

```powershell
..\nt-benchmark-env\Scripts\python -m benchmarks.competitive.evidence_tables --report benchmarks/competitive/results --expected-results-sha256 559bdd24fa632a80f9e9af9f04489fe634393cc74c58c814a88e3fde871deff7 --prepared-sources benchmarks/competitive/prepared_sources.json --output ..\derived-tables
..\nt-benchmark-env\Scripts\python -m benchmarks.competitive.paired_figures --report benchmarks/competitive/results --expected-results-sha256 559bdd24fa632a80f9e9af9f04489fe634393cc74c58c814a88e3fde871deff7 --output ..\derived-intervals
```

These commands read existing numerical evidence, without models or new scoring.
For a new local reproduction, use its `report` directory and the
`LOCAL_REPORT_MANIFEST.json` written by `reproduce report`. Keep the new outputs
separate from the delivered evidence. A local manifest does not authenticate a
new run as the original result. Optional plotting/benchmark dependencies stay
in the separate lock file; none becomes a mandatory library dependency.

### Archive integrity note

The [custody record](POSTPROCESSOR_CUSTODY.json) retains the historical hashes
and records a pre-existing source-hash discrepancy for two presentation-only
postprocessors. All four delivered CSV tables were regenerated byte for byte
from the frozen result JSON during repository review. The delivered paired
figure self-test passed, and the recorded PNG hashes match the supplied files;
exact graphical regeneration in a different environment is not claimed. No
model was retrained or rescored. The runtime modules and recorded benchmark
implementation remain byte-identical. Formatting exemptions and line-ending
attributes for the pinned benchmark sources preserve their reproducibility;
library lint, format and CI test requirements remain in force.
