# NeuroTabular 0.1.1 release notes

NeuroTabular 0.1.1 is a maintenance and robustness release for the 0.1.x line
and is the recommended 0.1.x release. It preserves the public constructor,
standard preprocessing, residual MLP architecture, loss, defaults, and binary
classification scope of 0.1.0.

## CUDA compatibility fix

PyTorch can report CUDA available even when its installed binaries cannot run a
kernel on the visible GPU. This was reproduced externally with a Tesla P100
(`sm_60`) and a PyTorch build compiled only for newer architectures. In 0.1.0,
`device="auto"` selected CUDA and training later failed with `no kernel image is
available for execution on the device`.

0.1.1 checks availability and device count, gathers GPU/build metadata, then
launches and synchronizes a minimal CUDA kernel before creating the network,
training tensors, GradScaler, or optimizer. Automatic selection warns and falls
back to CPU when the probe fails. Explicit `"cuda"` and `"cuda:N"` requests do
not fall back; they raise a focused diagnostic containing all available device,
compute-capability, PyTorch, compiled-architecture, and probe information.

CUDA random seeding now occurs only after a verified CUDA path is selected.
Verified devices with compute capability 7.0 or newer use float16 AMP; CPU and
older or metadata-unknown CUDA devices use FP32. NeuroTabular never changes or
installs PyTorch/CUDA on the user's behalf.

## Performance-engine maintenance

AdamW no longer forces the newer fused implementation; PyTorch chooses its
supported default implementation. Profiling now records target preparation,
scheduler time, best-state restoration, AMP/optimizer path, and device
diagnostics.

The new warm, interleaved same-process comparison ran the immutable 0.1.0 and
the 0.1.1 candidate on seven synthetic datasets, two seeds, and three timing
repetitions. All 14 paired dataset/seed results had identical ROC-AUC and epoch
counts; mean ROC-AUC was `0.874559` for both. The candidate won 9/14 fit pairs
(median paired ratio `0.9702`) and 7/14 prediction pairs (median paired ratio
`1.0106`). Large scheduling outliers made the arithmetic means unfavorable:
`1.6023 s` versus `1.7040 s` fit and `0.01882 s` versus `0.02102 s` prediction.
Therefore no speedup is claimed and the attempted hot-path changes were
removed. The final candidate retains the 0.1.0 hot path apart from low-overhead
profiling and device setup. Run `benchmarks/compare_0_1_x.py` on stable hardware
for release performance qualification.

The historical published 0.1.0 synthetic report remains `0.8735` mean ROC-AUC,
with its separate matched-legacy `+0.0064` ROC-AUC, `1.49x` fit, and `1.56x`
prediction results. Those are 0.1.0 results, not new 0.1.1 claims. The external
Kaggle S6E8 result (`0.940020` mean ROC-AUC and about `536.98 s` for six CPU
folds) was supplied by the maintainer and was not reproduced locally.

## Verification

- 77 tests passed and one real-CUDA integration test was skipped on the
  CPU-only host, compared with 67 passed and one skipped for the immutable
  0.1.0 baseline in the same environment.
- Device tests simulate unavailable CUDA, incompatible visible CUDA,
  compatible CUDA, invalid indices, automatic fallback, explicit failure, and
  CUDA seeding without requiring a GPU in CI.
- The candidate is checked with Ruff, warnings-as-errors pytest, build, Twine,
  and a clean-wheel import plus synthetic fit/predict smoke test.

## Limitations

The release remains binary-classification-only and pandas-DataFrame-only. It
does not add serialization, multiclass, regression, advanced numerical
embeddings, target encoding, attention, Transformers, neural ensembles, or
AutoML. No compatible GPU was available on the candidate host, so CUDA speed,
VRAM, AMP throughput, and fused-optimizer behavior were not benchmarked. The
included `benchmarks/kaggle_cuda_smoke.py` script is provided for later
measurement on a compatible GPU without distributing a dataset.

This file prepares release content only. It does not assert that tag `v0.1.1`,
a GitHub release, or a PyPI package already exists.
