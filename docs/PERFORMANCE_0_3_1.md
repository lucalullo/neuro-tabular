# NeuroTabular 0.3.1 performance validation

NeuroTabular 0.3.1 is a performance and product-hardening patch. It does not
change the model architecture, training objective, optimizer, scheduler,
default training policy, prediction formulas, or public estimator API.

The accepted optimization removes avoidable pandas work on two already-valid
prediction paths:

- real numeric columns no longer require a redundant per-column `to_numeric`
  reconstruction before the required owned float64 array is produced;
- a DataFrame whose columns are already in the learned schema order no longer
  requires a redundant column-selection reconstruction.

All fallback conversion, schema-reordering, missing-value, categorical, and
caller-ownership behavior remains in place.

## Validation result

On the frozen six-fixture P35 CPU/WSL software panel, the final two-change path
measured the following median changes versus the public 0.3.0 implementation:

| Measurement | Median change |
| --- | ---: |
| `predict` latency | -30.7% |
| `predict_proba` latency | -35.8% |
| fit wall time | -2.8% |
| fit CPU time | -1.2% |
| peak fit RSS | approximately unchanged |

Negative latency/time values mean faster execution. These are equal-weight
software-fixture aggregates, not a production traffic model and not a claim that
every call, dataset, operating system, or machine is faster.

A release-candidate spot check preserved the optimization; its medians differed
from the P35 confirmation by roughly -0.3% for `predict` and -1.6% for
`predict_proba`, within the documented timing variability.

## Behavior preservation

The release validation found bit-for-bit behavior preservation on the frozen
checks, including 30 binary equivalence cases and representative multiclass and
regression cases. Product-path tests cover native and nullable numeric storage,
reordered schemas, categorical/missing inputs, and caller ownership.

The patch does not claim improved predictive accuracy. The frozen 0.3 benchmark
remains the relevant quality evidence; see [BENCHMARK_0_3.md](BENCHMARK_0_3.md).

## Scope and limits

The P35 measurements were collected on CPU under WSL2/Ubuntu. Performance is
hardware-, dependency-, dataset-, and workload-specific. Physical CUDA hardware
validation remains pending. Linux/Windows compatibility is enforced through the
configured CI matrix before release publication.
