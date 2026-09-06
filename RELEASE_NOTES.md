# NeuroTabular 0.2.0 release notes

NeuroTabular 0.2.0 is the first public release in the current public history.
It provides a compact neural binary classifier for heterogeneous pandas
DataFrames with a scikit-learn-style API, automatic preprocessing, conservative
defaults, and release-focused correctness checks.

## Core capabilities

- `NeuroTabularClassifier` with `fit`, `predict`, and `predict_proba`;
- automatic numerical and categorical feature handling;
- missing, unknown, and rare-category semantics;
- training-only categorical log-frequency features;
- adaptive categorical embedding widths;
- compact residual MLP backbone;
- AdamW optimization, cosine scheduling, automatic batching, and early stopping;
- sample weights and balanced class weights;
- CPU/CUDA device selection and explicit diagnostics;
- scikit-learn clone, pipeline, and cross-validation compatibility.

## Reliability and data integrity

- Declared but unobserved pandas `Categorical` levels remain unknown rather than
  being learned as rare training categories.
- Complex-valued numerical inputs are rejected explicitly.
- Finite numerical values that would overflow float32 conversion are rejected.
- Zero-weight rows are ignored before target discovery, splitting, and
  preprocessing.
- Weight normalization validates dynamic range before tensor conversion.
- Re-fitting is transactional: a failed fit preserves a previously fitted model.
- Caller Python, NumPy, and PyTorch CPU RNG states are restored after fitting.
- `min_delta` controls patience while strict best-checkpoint restoration remains
  independent.
- Aggregate categorical embedding widths are bounded by the configured compact
  budget.

## Release engineering

- Python support is declared and tested for 3.10, 3.11, and 3.12.
- PyTorch 2.0 compatibility is checked in CI.
- Linux and Windows CI jobs run the test suite.
- Source and wheel distributions are built and metadata-checked.
- The built wheel is installed into a clean environment and smoke-tested outside
  the source tree.
- Manual publication requires a real Git tag matching the package version.

## Research status

The release deliberately does not enable experimental GELU defaults, changed
mini-batch policies, ranking-oriented checkpoint selection, automatic
calibration, or other exploratory mechanisms. Internal research produced useful
signals but not enough broad real-dataset evidence to justify changing the
public defaults.

## Compatibility

- Python: `>=3.10,<3.13`
- PyTorch: `>=2.0`
- NumPy: `>=1.24`
- pandas: `>=2.0`
- scikit-learn: `>=1.3`

NeuroTabular remains experimental, pre-1.0, binary-classification-only, and
pandas-DataFrame-only.
