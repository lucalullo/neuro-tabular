# NeuroTabular 0.2.0 research and design notes

## Scope and originality

The 0.2.0 work reviewed primary papers and official implementation or framework
documentation to identify testable ideas. NeuroTabular's code was written for
this project; no third-party source code, trained weights, or dataset is copied
into the package. The cited systems are research context, not dependencies or
claims of equivalent implementation.

## Sources reviewed

### RealMLP

- Paper: [Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular
  Data](https://openreview.net/forum?id=fwajDrDy89)

Relevant idea: a carefully designed and pre-tuned MLP pipeline can be a strong
tabular baseline; preprocessing, initialization, regularization, and training
details matter at least as much as adding a large architecture.

NeuroTabular application: test inexpensive numerical representations,
frequency-aware categorical input, prior output bias, stable preprocessing, and
execution-path improvements while keeping a small residual MLP. NeuroTabular is
not a RealMLP reproduction and does not copy its training recipe.

Outcome: scalar numerical input remained stronger than the tested periodic and
piecewise expansions. Aggregate categorical frequency improved the release
matrix and became the only new default representation component.

### TabM

- Paper: [TabM: Advancing Tabular Deep Learning with Parameter-Efficient
  Ensembling](https://openreview.net/forum?id=Sd4wYYOhmY)
- Official repository: [yandex-research/tabm](https://github.com/yandex-research/tabm)

Relevant idea: parameter-efficient ensembling can improve tabular neural
networks without simply replicating complete models.

NeuroTabular application: the concept was evaluated at the design level, but a
multi-branch ensemble would expand training and inference cost, serialization,
and public hyperparameters beyond the evidence available for this release.

Outcome: not implemented in 0.2.0. The release instead uses one compact model
and adaptive embedding widths. A future ensemble would require an isolated
quality-per-parameter and latency ablation.

### Numerical feature embeddings

- Paper: [On Embeddings for Numerical Features in Tabular Deep
  Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)

Relevant idea: piecewise-linear and periodic representations can make numerical
features easier for tabular neural networks to use.

NeuroTabular application: implement compact affine, periodic, and
piecewise-linear modes behind one tested numerical embedding module. Piecewise
knots are quantiles fitted on training rows only; missingness remains explicit.
The dimensions and projections are project-specific compact choices rather than
a reproduction of the paper's models.

Outcome: all modes pass forward/backward tests, but scalar won the release
screening. Optional modes remain available for dataset-specific experiments and
are not advertised as universally better.

### PyTorch execution guidance

- Official documentation: [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- Official tutorial: [Performance Tuning
  Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- Official API: [`torch.optim.AdamW`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

Relevant ideas: reduce Python/framework overhead where workloads justify it,
choose transfer and precision features according to the actual device, and
measure optimizer backends rather than assuming equivalence.

NeuroTabular application: tables remain device-resident, batches use index
tensors, validation avoids unnecessary probability collection for loss-only
evaluation, device auto-selection performs a synchronized kernel probe, and
AMP is restricted to compatible CUDA hardware. Compile and optimizer modes are
benchmark-script experiments rather than defaults.

Outcome: compile was unavailable on the Windows host because no supported C++
compiler was installed and its attempted run was slower before failure.
Explicit optimizer modes produced suggestive but non-portable/noisy results.
Eager execution and automatic AdamW are therefore retained.

### Gradient-boosted tree context

- Official LightGBM documentation: [Features](https://lightgbm.readthedocs.io/en/latest/Features.html)
- CatBoost paper: [CatBoost: unbiased boosting with categorical
  features](https://arxiv.org/abs/1706.09516)
- Official CatBoost documentation: [Categorical features](https://catboost.ai/en/docs/features/categorical-features)

Relevant idea: modern boosted trees are strong tabular references, particularly
for categorical handling, speed, and small/medium structured datasets.

NeuroTabular application: the benchmark runner includes leakage-safe ordinal
encoding for the bundled sklearn histogram baseline and optional LightGBM and
CatBoost comparisons when installed. Neither library is imported by normal
NeuroTabular execution.

Outcome: the release makes no superiority claim. The project brief's external
Kaggle reference favored LightGBM in both AUC and runtime and could not be
reproduced locally because its data/environment were unavailable.

## Leakage rules applied

- validation and test rows never fit medians, centers, scales, quantiles,
  vocabularies, rare buckets, or category frequencies;
- category frequency is indexed by the encoded training-derived ID;
- optional full-data refit occurs only after epoch selection and builds fresh
  preprocessing from primary data, never from an external validation set;
- tree baselines use a scikit-learn pipeline so encoders are fitted only on the
  training partition;
- synthetic target construction precedes train/test splitting and does not use
  model predictions or test statistics.

## Components tested and rejected as defaults

- periodic and piecewise numerical embeddings: lower mean ROC-AUC;
- affine numerical embedding: no aggregate gain and higher cost;
- feature gate: no robust gain;
- categorical/embedding dropout: tiny mean ROC-AUC regressions;
- full-data refit: lower mean AUC and about 68% higher median fit time;
- category cap/hash: small inconsistent quality changes, insufficient evidence;
- `torch.compile`: compiler unavailable and failed after additional cold time;
- forced AdamW fused/foreach: insufficient portability evidence;
- parameter-efficient multi-branch ensembling: deferred before implementation
  because the release lacked evidence to justify its cost surface.

## Evidence limits

The screening suite is intentionally compact and uses only two seeds. Public
no-regression data is numerical and therefore does not independently validate
the categorical gain. The release host has no compatible CUDA runtime, so GPU,
AMP, pinned-memory throughput, and VRAM remain unmeasured. These limitations are
reported in the benchmark and performance documents rather than filled with
estimates.
