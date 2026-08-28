# NeuroTabular 0.2.0 performance profile

## Profile objective

The release profile identifies where end-to-end time and memory are spent; it
is not a microbenchmark contest. It compares the final 0.1.1 baseline with the
0.2.0 candidate on one reproducible CPU workload and supplements that cold
profile with the paired benchmark ratios in `BENCHMARK_REPORT.md`.

## Environment and workload

- Windows 11, Python 3.12.13;
- PyTorch 2.13.0+cpu;
- 2 physical / 4 logical CPU cores, 11.71 GiB RAM;
- 5,000 mixed rows, 15 columns;
- five epochs, hidden width 32, one residual block;
- ROC-AUC validation metric;
- CPU only, no AMP, no VRAM measurement.

Each detailed profile ran in a separate cold process. Cold-start and host-load
effects therefore make its absolute cross-version delta contextual. The paired,
warm, alternating benchmark is used for release speed claims.

## Stage timings

| Stage | 0.1.1 seconds | 0.2.0 seconds |
|---|---:|---:|
| End-to-end fit | 5.099872 | 2.771824 |
| Preprocessing | 0.044050 | 0.037767 |
| Tensor conversion | 0.000177 | 0.000078 |
| Device transfer | 0.000015 | 0.000011 |
| Batch construction | 0.000037 | 0.000033 |
| Forward | 0.155676 | 0.101932 |
| Backward | 0.113537 | 0.070493 |
| Optimizer | 0.036594 | 0.019705 |
| Validation inference | 0.029929 | 0.018597 |
| Metric calculation | 0.053041 | 0.029753 |
| Checkpoint copy | 0.003831 | 0.003144 |
| Best-state restoration | 0.002289 | 0.001453 |
| Engine setup/first-use | 4.623094 | 2.456397 |
| Training engine total | 5.026674 | 2.708069 |
| End-to-end prediction | 0.043944 | 0.023689 |
| Prediction preprocessing | 0.025768 | 0.013382 |
| Prediction network, derived | 0.018176 | 0.010307 |

The dominant component is engine setup and first-use overhead, not tensor
transfer or batch construction. The absolute reduction between two independent
cold processes must not be read as a guaranteed 46% production speedup. In the
paired release matrix, the robust aggregate was a 0.950 median fit-time ratio
and a 0.986 median prediction-time ratio.

## Code-path findings

### Preprocessing

The categorical frequency transform initially performed a second pandas map.
The release implementation instead indexes a learned NumPy frequency table with
already encoded category IDs. This preserves training-only semantics and avoids
duplicating the expensive mapping pass.

Numerical quantile knots are fitted for every model so all supported numerical
representations can share preprocessing state. Their cost is included in the
profile. Scalar mode remains the default and consumes only scaled values and
missing indicators at runtime.

### Training and validation

Processed tables are converted once and kept on the selected device. Training
batches are index tensors rather than repeated host-to-device copies.

When `eval_metric="loss"`, validation no longer accumulates probability and
target arrays or applies a sigmoid solely for a metric that does not require
them. ROC-AUC and accuracy retain the full metric path.

Best weights are copied only on a qualifying improvement and restored once.
The profile separates checkpoint and restoration time.

### CPU autocast and PyTorch 2.0

The final 0.1.1 maintenance behavior is preserved: `_autocast_context(False)`
returns `contextlib.nullcontext`. CPU execution with AMP disabled never calls
`torch.autocast`. This avoids unsupported arguments/overhead and is protected by
a regression test. AMP remains a CUDA-only optimization after a successful
device probe.

### Adaptive embeddings

0.2.0 computes bounded categorical widths using cardinality, sample count,
feature count, and a compact memory budget. On the profiled schema, trainable
parameters decreased from 9,401 to 9,063. This is not guaranteed for every
schema, but widths remain bounded between 2 and 16.

## Memory

| Measure | 0.1.1 | 0.2.0 |
|---|---:|---:|
| Trainable parameters | 9,401 | 9,063 |
| Cold profile peak RSS delta | 86.703 MiB | 86.816 MiB |
| Peak VRAM | N/A | N/A |

RSS is effectively unchanged. The sampler measures the whole process and is
sensitive to allocator high-water marks and first-use library initialization.
The host used a CPU-only PyTorch wheel, so CUDA transfer, pinned memory,
non-blocking copies, AMP throughput, and VRAM could not be measured physically.
Their device-selection and fallback semantics are covered by tests, not by a
fabricated hardware result.

## Performance experiments not selected

- `torch.compile(mode="reduce-overhead")` spent 17.14 s before failing because
  a supported C++ compiler (`cl`) was unavailable; eager mode completed the
  same cold workload in 8.23 s. Compile remains experimental and off.
- Explicit AdamW `foreach` and `fused` modes showed lower optimizer timings in
  noisy independent runs, but fused CPU support is not a safe PyTorch 2.0-wide
  default. Automatic optimizer selection remains the release path.
- Full-data refit increased median fit time by about 68% and reduced screening
  mean AUC; it remains opt-in.
- Category cap/hash reduced parameters on high-cardinality workloads, but the
  mean AUC gain was only 0.0006/0.0015 across four cases and changed sign across
  seeds; hashing also increased median prediction time. These controls remain
  research-only and do not affect defaults.

## Remaining bottlenecks

For small and medium CPU tables, PyTorch process and kernel initialization is a
large fraction of wall time. For repeated production fits, persistent worker
processes may matter more than another small model-level optimization. On large
categorical schemas, pandas conversion and categorical lookup remain important.

Future performance claims require:

- an idle, controlled multi-core CPU host;
- at least one physical CUDA GPU with recorded model and compute capability;
- repeated warm and cold measurements;
- representative large public datasets;
- throughput, latency distribution, RSS, and VRAM reported together;
- quality parity constraints for every optimization.
