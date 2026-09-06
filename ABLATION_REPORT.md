# NeuroTabular 0.2.0 ablation report

## Protocol

Internal architecture screening used six deterministic dataset families and
seeds 19 and 31, a stratified 75/25 split, at most 20 epochs, and the same
optimizer/validation/model budget. The families cover numeric, missing,
categorical, high-cardinality, imbalanced, and medium synthetic data. Every
transform was fitted only on the training partition.

These are development ablations, not a substitute for a broad real-world
benchmark suite.

## Numerical representations

| Variant | Mean ROC-AUC | Mean log loss | Median fit s | Median predict s |
|---|---:|---:|---:|---:|
| scalar | 0.904508 | 0.342421 | 0.7470 | 0.0176 |
| affine | 0.904000 | 0.347604 | 0.9697 | 0.0205 |
| periodic | 0.792801 | 0.496602 | 0.7317 | 0.0175 |
| piecewise | 0.882937 | 0.398508 | 1.0519 | 0.0193 |
| piecewise + frequency | 0.883459 | not selected | not selected | not selected |
| piecewise + frequency + gate | 0.886431 | not selected | not selected | not selected |

Decision: keep `scalar` as the release default. Other modes remain explicit
experimental options.

## Category frequency and input gating

| Variant | Mean ROC-AUC | Mean log loss | Median fit s | Median predict s |
|---|---:|---:|---:|---:|
| scalar | 0.904508 | 0.342421 | 0.7470 | 0.0176 |
| scalar + frequency | 0.906508 | 0.339034 | 0.6535 | 0.0164 |
| scalar + gate | 0.902472 | not selected | not selected | not selected |
| scalar + frequency + gate | 0.905628 | not selected | not selected | not selected |
| affine + frequency | 0.908474 | higher cost | higher cost | higher cost |

Decision: enable scalar + aggregate category frequency by default and keep the
feature gate off. The frequency implementation aggregates counts at the actual
rare ID and uses encoded-ID lookup, preserving training-only semantics.

## Categorical regularization

| Variant | Mean ROC-AUC | Mean log loss | Median fit s |
|---|---:|---:|---:|
| scalar + frequency | 0.860722 | 0.469896 | 0.6033 |
| 5% categorical ID dropout | 0.860600 | 0.469496 | 0.6333 |
| 5% embedding dropout | 0.860533 | 0.469691 | 0.6070 |

Decision: do not enable either regularizer by default.

## Full-data refit

| Variant | Mean ROC-AUC | Median fit s |
|---|---:|---:|
| scalar + frequency | 0.906508 | 0.5942 |
| scalar + frequency + refit | 0.901626 | 0.9989 |

Decision: retain `full_data_refit` as an explicit option, default `False`.

## High-cardinality overflow controls

| Variant | Mean ROC-AUC | Mean log loss | Median fit s | Median predict s |
|---|---:|---:|---:|---:|
| uncapped scalar + frequency | 0.867874 | 0.458545 | 0.7314 | 0.0152 |
| top 64 + one rare bucket | 0.868496 | 0.457902 | 0.4650 | 0.0136 |
| top 64 + 16 hash buckets | 0.869386 | 0.458899 | 0.4691 | 0.0165 |

Changes were small and inconsistent across seeds; hashing also increased
prediction work in the recorded screen. Decision: do not expose these controls
as release defaults.

## Execution-engine experiments

`torch.compile` did not provide a portable result on the recorded Windows host
because the required compiler toolchain was unavailable. AdamW backend timing
was also host-dependent. Decision: no compile default and no forced optimizer
backend; use the supported PyTorch/AdamW automatic path.

## Release configuration

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

The release intentionally avoids stacking components whose internal evidence
was weak, inconsistent, specialized, or too costly.
