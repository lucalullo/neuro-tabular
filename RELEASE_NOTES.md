# NeuroTabular 0.1.0 release notes

NeuroTabular 0.1.0 establishes a new experimental foundation for neural binary
classification on pandas DataFrames.

The release provides `NeuroTabularClassifier` with automatic leakage-safe
handling of numerical NaNs, categorical detection, embedding vocabularies,
rare and unseen categories, internal or explicit validation, early stopping,
sample/class weights, device selection, automatic batches, and scikit-learn
compatibility.

The standard network is a compact residual MLP with categorical embeddings,
LayerNorm, SiLU, AdamW, cosine learning-rate decay, and one binary logit. The
in-memory engine converts tensors once and uses direct indexing instead of a
DataLoader for small and medium materialized datasets.

Release validation includes 67 passing CPU tests and one conditional CUDA test
skip on the CPU-only benchmark host. Synthetic benchmarks, profiling, and
ablation results are published with the source. They support the engineering
defaults but do not establish state-of-the-art or universal superiority.

Known limitations include binary classification only, pandas DataFrame input,
simple rare-category handling, no public serialization, no advanced numerical
embeddings, and no measured CUDA benchmark for this release.

