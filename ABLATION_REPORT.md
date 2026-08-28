# NeuroTabular 0.2.0 ablation report

## Protocol

Architecture screening used six deterministic dataset families and seeds 19
and 31, a stratified 75/25 split, at most 20 epochs, and the same optimizer,
validation, and model budget. The families cover numeric, missing, categorical,
high-cardinality, imbalanced, and medium synthetic data. Every transform was
fitted only on the training partition.

Focused follow-ups reused the same seeds and workloads where the component was
relevant. Results are means across runs unless labelled median. Small deltas are
treated conservatively because twelve screening outcomes are not a substitute
for a large benchmark suite.

## Numerical representations

| Variant | Mean ROC-AUC | Mean log loss | Median fit s | Median predict s |
|---|---:|---:|---:|---:|
| scalar | 0.904508 | 0.342421 | 0.7470 | 0.0176 |
| affine | 0.904000 | 0.347604 | 0.9697 | 0.0205 |
| periodic | 0.792801 | 0.496602 | 0.7317 | 0.0175 |
| piecewise | 0.882937 | 0.398508 | 1.0519 | 0.0193 |
| piecewise + frequency | 0.883459 | not selected | not selected | not selected |
| piecewise + frequency + gate | 0.886431 | not selected | not selected | not selected |

Decision: retain all four representations as tested opt-in modes, but keep
`scalar` as the release default. Affine added cost without a mean gain;
piecewise and periodic were materially worse on this protocol. Quantile knots
remain useful infrastructure for explicit dataset-level experimentation.

## Category frequency and input gating

The focused comparison found:

| Variant | Mean ROC-AUC | Mean log loss | Median fit s | Median predict s |
|---|---:|---:|---:|---:|
| scalar | 0.904508 | 0.342421 | 0.7470 | 0.0176 |
| scalar + frequency | 0.906508 | 0.339034 | 0.6535 | 0.0164 |
| scalar + gate | 0.902472 | not selected | not selected | not selected |
| scalar + frequency + gate | 0.905628 | not selected | not selected | not selected |
| affine + frequency | 0.908474 | higher cost | higher cost | higher cost |

An early frequency implementation assigned individual rare values their own
frequency. The final release instead aggregates counts at the actual rare ID
and uses a NumPy lookup by encoded ID. The definitive paired 0.1.1/0.2.0 release
matrix measured +0.003849 mean ROC-AUC, with improvements concentrated in all
four categorical workloads and exact parity on numeric workloads.

Decision: enable scalar + aggregate category frequency by default. Keep the
feature gate off. Affine + frequency was not selected because its small
screening mean advantage was not broad, it regressed one pure-numeric case by
0.0024, and it increased parameters and prediction work.

## Categorical regularization

Categorical-heavy and moderately high-cardinality datasets were evaluated with
two seeds:

| Variant | Mean ROC-AUC | Mean log loss | Median fit s |
|---|---:|---:|---:|
| scalar + frequency | 0.860722 | 0.469896 | 0.6033 |
| 5% categorical ID dropout | 0.860600 | 0.469496 | 0.6333 |
| 5% embedding dropout | 0.860533 | 0.469691 | 0.6070 |

Decision: do not enable either regularizer. Log loss moved slightly in the
desired direction, but mean ROC-AUC declined and there was no robust efficiency
benefit. The implementation remains internal research support, not public API.

## Full-data refit

| Variant | Mean ROC-AUC | Median fit s |
|---|---:|---:|
| scalar + frequency | 0.906508 | 0.5942 |
| scalar + frequency + refit | 0.901626 | 0.9989 |

Decision: retain `full_data_refit` as an explicit public option, default false.
It is useful when a user has independent validation evidence for the protocol,
but it added roughly 68% median fit time and reduced mean AUC here.

## High-cardinality overflow

A frequency-ranked vocabulary cap and optional stable hash buckets were tested
on moderate and extreme high-cardinality synthetic tables, two seeds each:

| Variant | Mean ROC-AUC | Mean log loss | Median fit s | Median predict s |
|---|---:|---:|---:|---:|
| uncapped scalar + frequency | 0.867874 | 0.458545 | 0.7314 | 0.0152 |
| top 64 + one rare bucket | 0.868496 | 0.457902 | 0.4650 | 0.0136 |
| top 64 + 16 hash buckets | 0.869386 | 0.458899 | 0.4691 | 0.0165 |

The first uncapped fit paid a large cold-start cost, so its median fit advantage
must not be interpreted as a reliable 36% speedup. Capping reduced parameters
by roughly 3-9% on these schemas. AUC changes were +0.0006/+0.0015 overall but
changed sign across seeds; hashing slowed median prediction.

Decision: do not expose or enable the controls in 0.2.0. They remain isolated
research controls for future large-dataset validation and have no effect on
normal estimator construction.

## Execution-engine experiments

### `torch.compile`

| Mode | Cold workload result |
|---|---|
| eager | 8.23 s, completed |
| `reduce-overhead` | 17.14 s before `InvalidCxxCompiler`; `cl` unavailable |

Decision: no compile default and no advertised compile speedup. The experiment
is retained only in the benchmark script so a properly provisioned host can
repeat it.

### AdamW strategy

Independent cold runs measured optimizer-step time of about 0.064 s for
`foreach` and 0.026 s for `fused`, versus 0.096 s in an earlier automatic run.
The processes were contended, and fused CPU behavior is not a safe guarantee
across the declared PyTorch 2.0+ range.

Decision: use AdamW automatic selection. Explicit foreach/fused settings remain
training-engine research controls, not public classifier parameters.

## Final release configuration

```text
numerical_embedding = "scalar"
use_category_frequency = True
feature_gating = False
full_data_refit = False
categorical dropout = 0
embedding dropout = 0
category cap/hash = disabled
torch.compile = disabled
AdamW strategy = automatic
```

This configuration is a measured incremental change over 0.1.1. It avoids
stacking components whose individual evidence was weak or negative.
