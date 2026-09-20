# Historical NeuroTabular 0.2.0 research and design notes

## Design goal

NeuroTabular aims to keep the ergonomics of a conventional tabular estimator
while using a compact neural model. The public release favors simple,
reproducible defaults over architectural novelty.

## Principles used for the release

- preprocessing statistics must be learned from training rows only;
- missing, unknown, and rare categories must have explicit semantics;
- small real and synthetic diagnostics are useful, but broad claims require
  broader real-world evidence;
- model size, fit cost, prediction cost, and failure modes matter alongside
  predictive quality;
- experimental components stay opt-in or out of the public API until evidence
  justifies them;
- results from one dataset or one seed are not treated as general conclusions.

## Numerical features

The default representation is a standardized scalar plus missingness signal.
Affine, periodic, and piecewise representations are implemented for controlled
experimentation. Internal ablations did not justify replacing scalar inputs as
the default.

## Categorical features

Each categorical column has explicit missing, unknown, and rare IDs plus a
learned embedding. Aggregate log-frequency can be appended as a numerical side
feature. Frequencies and vocabularies are learned only from training rows;
declared but unobserved pandas categorical levels are not treated as observed.

Embedding width adapts to sample count, cardinality, categorical feature count,
and a compact aggregate width budget. The width budget is not a total-memory
cap, so extreme cardinality still requires care.

## Training

The release uses AdamW with cosine scheduling, deterministic automatic batching,
stratified internal validation when no external set is supplied, and early
stopping. `min_delta` governs patience; strict best-checkpoint restoration is
kept separate so a smaller strict improvement is not discarded.

Re-fitting is transactional and caller RNG states are restored after fitting.
Rows with zero effective sample weight are removed before target discovery,
splitting, and preprocessing.

## Device behavior

Automatic CUDA use requires a successful synchronized kernel probe. If CUDA is
visible but unusable, `device="auto"` warns and falls back to CPU. Explicit CUDA
requests fail instead of silently changing device. AMP is only enabled after a
compatible CUDA probe.

## Current research boundary

Exploratory work has investigated activation changes, update granularity,
checkpoint/ranking criteria, calibration, robust preprocessing, categorical
regularization, and other mechanisms. None has broad enough evidence to replace
the public 0.2.0 defaults. NeuroTabular 0.3.0 keeps the same validated binary
default behavior while extending the shared core to multiclass classification
and regression. Future architecture changes should continue to require broader
validation before replacing the standard algorithm.

## Benchmark discipline

The bundled benchmark scripts fit preprocessing independently inside each
training split. Contextual tree baselines use scikit-learn pipelines so encoders
are learned only from training data. Optional LightGBM/CatBoost comparisons are
benchmark-only and do not become runtime dependencies.

No state-of-the-art claim is made.
