# NeuroTabular 0.1.x performance profile

## Objective

The legacy prototype was slow on small datasets. This profile separates data
preparation, tensor creation, batch overhead, neural compute, validation,
metrics, checkpointing, prediction, and total fit time before increasing model
complexity.

All detailed timings below were collected on the CPU environment documented in
`BENCHMARK_REPORT.md`. CUDA transfer and VRAM could not be measured because the
host had a CPU-only PyTorch build.

## Legacy profile

A 5,000-row mixed DataFrame with three numerical features, two categorical
features, and numerical missing values was profiled with a 32-unit, one-layer
network and batch size 256.

| Phase | Seconds |
| --- | ---: |
| Preprocessor fit + transform | 0.08523 |
| Repeated transform | 0.03901 |
| NumPy-to-tensor conversion | 0.00094 |
| DataLoader construction | 0.00036 |
| DataLoader batch iteration only | 0.25107 |
| One training epoch total | 0.12717 |
| Forward within epoch | 0.03535 |
| Backward within epoch | 0.03553 |
| Optimizer steps within epoch | 0.04535 |
| Validation inference | 0.25171 |
| ROC-AUC calculation | 0.00776 |
| CPU checkpoint clone | 0.00077 |

Batch iteration alone took roughly twice the entire measured neural compute of
the epoch. Validation repeated the same high-overhead loader path. The training
loop also transferred and synchronized batch values repeatedly and validated
plus checkpointed every epoch with an effective improvement threshold near
machine precision.

The historical 600-row, 12-feature smoke benchmark with five epochs measured
4.2170 s fit, 0.0129 s prediction, and 0.7500 ROC-AUC.

## NeuroTabular detailed profile

The same 5,000-row mixed schema was profiled in a fresh process using a
32-unit, one-block residual model, five-epoch ceiling, patience 2, and automatic
batching. The internal split produced 4,000 training rows and a full training
batch of 4,000.

| Phase | Seconds |
| --- | ---: |
| Total fit | 3.32437 |
| Preprocessing total | 0.02447 |
| Schema and statistics fit | 0.00978 |
| Training transform | 0.00774 |
| Validation transform | 0.00688 |
| Tensor conversion | 0.00019 |
| Device transfer | 0.00002 |
| Batch creation, 5 epochs | 0.00004 |
| Forward, 5 epochs | 0.13522 |
| Backward, 5 epochs | 0.07171 |
| Optimizer steps, 5 epochs | 0.02583 |
| Training compute total | 0.23906 |
| Validation inference | 0.02442 |
| Metric calculation | 0.00001 |
| Checkpoint clones | 0.00372 |
| Validation wall time | 0.02613 |
| Prediction, 5,000 rows | 0.02908 |
| Trainable parameters | 4,776 |

The apparent gap between total fit and compute is a 3.00827 s one-time optimizer
and PyTorch runtime cold start in the fresh process. Later fits in the
multi-dataset benchmark commonly completed in 0.38–0.94 s for 600–1,200 rows.
The cold start is reported rather than hidden because it affects first-use
latency.

## Engine comparison

The release ablation independently measured tensor traversal on 5,000 rows:

```text
DataLoader:       0.18012 s
direct indexing:  0.00208 s
speedup:         86.49x
```

This is an iteration-only microbenchmark. End-to-end paired results across
seven datasets showed a 1.49× aggregate fit-time speedup and 1.56× aggregate
prediction speedup relative to the legacy prototype while improving mean
ROC-AUC by 0.0064.

## Current engine behavior

- DataFrame validation and preprocessing run once per fit/transform.
- Contiguous NumPy arrays become tensors once.
- Small datasets use a full batch; larger in-memory datasets use large shuffled
  index tensors.
- No DataLoader worker process is created for already-materialized arrays.
- Training loss aggregates on device and crosses to CPU once per epoch.
- Validation uses `torch.inference_mode()` and transfers only arrays required by
  sklearn metrics.
- `optimizer.zero_grad(set_to_none=True)` avoids redundant gradient writes.
- Checkpoints are cloned only when improvement exceeds `min_delta`.
- Prediction uses bounded tensor batches and `torch.inference_mode()`.
- No global PyTorch thread count, deterministic-mode, TF32, or user device
  setting is changed.

On CUDA, the engine attempts one-time data residency when combined tensors use
less than 20% of reported free memory. Otherwise it pins CPU tensors and uses
non-blocking batch transfers. CUDA enables autocast/gradient scaling and tries
fused AdamW with a safe fallback. Fine-grained CUDA phase timings are marked
approximate to avoid a synchronization after every operation.

## Remaining performance work

- Reduce or amortize first-fit optimizer/runtime cold start where PyTorch permits
  it without moving hidden work to import time.
- Profile real CUDA hardware, including host-to-device transfer, pinned staging,
  AMP, fused AdamW, VRAM, and auto-batch memory margins.
- Test larger-than-memory and truly large mini-batch workloads before adding a
  DataLoader path.
- Profile categorical encoding on million-row and extreme-cardinality frames.
- Add public-dataset continuous performance regression thresholds after stable
  CI hardware is available.

## 0.1.1 maintenance profile

Version 0.1.1 adds explicit `target_preparation_seconds`, `scheduler_seconds`,
`best_state_restoration_seconds`, AMP state, optimizer-path labeling, and device
diagnostics to the existing profile. CUDA fine-grained timings remain
asynchronous/approximate; the compatibility probe synchronizes exactly once
during device resolution, not inside the training hot path.

The maintenance audit evaluated and then discarded:

- preallocated prediction output, because the measured small-batch path was not
  faster than list collection plus one concatenation;
- Python-side guards around `.to(device)`, because the added dispatch did not
  improve the measured aggregate;
- validation slicing and loss-only probability suppression, because the noisy
  end-to-end matrix did not establish an overall gain;
- automatic CUDA OOM batch retries, because no compatible GPU was available to
  validate cleanup/retry semantics safely; and
- forced fused AdamW fallback after a failed real-model first step, because a
  partially executed optimizer step is not a safe recovery boundary.

The final candidate instead stops forcing `fused=True` and lets PyTorch select
its supported AdamW implementation. This follows the official AdamW guidance
and avoids treating arbitrary CUDA, numerical, or OOM failures as evidence that
only the fused implementation failed.

The final warm/interleaved comparison preserved exactly the same ROC-AUC and
epoch count for every one of 14 dataset/seed pairs. Median paired fit ratio was
0.9702 and median paired prediction ratio was 1.0106, while arithmetic means
were distorted by large scheduling outliers. See `BENCHMARK_REPORT.md` for all
rows. No 0.1.1 speedup is claimed.
