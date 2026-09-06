# NeuroTabular 0.2.0 performance profile

## Objective

This profile records where end-to-end time is spent on one reproducible CPU
workload. It is diagnostic rather than a throughput guarantee.

## Recorded environment and workload

- Windows 11, Python 3.12.13;
- PyTorch 2.13.0+cpu;
- 2 physical / 4 logical CPU cores, 11.71 GiB RAM;
- 5,000 mixed rows, 15 columns;
- five epochs, hidden width 32, one residual block;
- ROC-AUC validation metric;
- CPU only, no AMP, no VRAM measurement.

## Stage timings

| Stage | Seconds |
|---|---:|
| End-to-end fit | 2.771824 |
| Preprocessing | 0.037767 |
| Tensor conversion | 0.000078 |
| Device transfer | 0.000011 |
| Batch construction | 0.000033 |
| Forward | 0.101932 |
| Backward | 0.070493 |
| Optimizer | 0.019705 |
| Validation inference | 0.018597 |
| Metric calculation | 0.029753 |
| Checkpoint copy | 0.003144 |
| Best-state restoration | 0.001453 |
| Engine setup/first-use | 2.456397 |
| Training engine total | 2.708069 |
| End-to-end prediction | 0.023689 |
| Prediction preprocessing | 0.013382 |
| Prediction network, derived | 0.010307 |

The dominant cost in this cold process was engine setup/first-use overhead.
Absolute timings should not be extrapolated to other machines or workloads.

## Code-path findings

### Preprocessing

Categorical frequency transform indexes a learned NumPy frequency table using
already encoded category IDs, avoiding a second pandas mapping pass. Numerical
quantile knots are fitted so all supported numerical representations can share
training-only preprocessing state; scalar mode consumes only scaled values and
missing indicators at runtime.

### Training and validation

Processed tables are converted once and kept on the selected device. Batches
use index tensors rather than repeated host-to-device table copies. When
`eval_metric="loss"`, validation avoids probability-array construction and
sigmoid work that the metric does not need. Best weights are copied only on a
strict improvement and restored once.

### CPU autocast and CUDA

When AMP is disabled, CPU execution uses `contextlib.nullcontext`; no disabled
CPU `torch.autocast` context is constructed. AMP is enabled only after a
successful CUDA probe and compatibility check.

### Adaptive embeddings

Categorical widths are bounded and adapt to cardinality, sample count, feature
count, and a compact aggregate width budget. On this recorded schema the model
used 9,063 trainable parameters. The budget controls embedding widths, not total
RAM/VRAM bytes.

## Memory

The recorded cold profile peak RSS delta was about 86.8 MiB. RSS includes the
Python process, native libraries, allocator behavior, and first-use effects.
VRAM was not measured because CUDA was unavailable.

## Practical guidance

- benchmark on the target hardware;
- avoid judging speed from one tiny dataset;
- use the built-in profiling attributes to separate preprocessing, training,
  validation, and prediction costs;
- treat high-cardinality categoricals separately because embedding table size
  depends on cardinality even when width is bounded.
