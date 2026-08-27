# NeuroTabular 0.1.0 ablation report

## Method

`benchmarks/run_ablations.py` evaluates two 1,200-row synthetic tasks (numeric
and mixed) with seeds 19 and 31. Every variant uses the same external test split,
internal validation split, hidden width 64, two blocks/layers, dropout 0.1,
12-epoch ceiling, patience 3, AdamW, and CPU. One factor changes at a time from
the residual reference configuration.

The first residual run includes a roughly three-second PyTorch optimizer cold
start. Median fit time is therefore more representative for architecture
comparisons than mean fit time.

## Results

| Variant | Mean ROC-AUC | Median fit s | Mean epochs |
| --- | ---: | ---: | ---: |
| Residual reference | 0.87219 | 0.4881 | 10.50 |
| Plain MLP | 0.86800 | 0.2271 | 12.00 |
| GELU instead of SiLU | 0.87042 | 0.3974 | 10.00 |
| No normalization | 0.87058 | 0.4876 | 12.00 |
| Standard numerical preprocessing | 0.87262 | 0.4258 | 10.50 |
| Rare handling disabled | 0.87219 | 0.4341 | 10.50 |
| Validation every epoch | 0.87228 | 0.3693 | 8.25 |
| Validation every 3 epochs | 0.87430 | 0.4859 | 12.00 |
| `min_delta=1e-12` | 0.87219 | 0.4347 | 10.50 |
| Constant learning rate | 0.87140 | 0.4964 | 10.50 |
| Short warmup + cosine | 0.87050 | 0.4744 | 10.50 |

## Decisions

- **Residual MLP retained.** It improved mean ROC-AUC by 0.00419 over the plain
  MLP. The plain model was about twice as fast and used far fewer parameters,
  but the release prioritizes the measured accuracy gain while remaining under
  roughly 37k parameters on benchmark schemas.
- **SiLU retained.** GELU was 0.00177 lower in mean ROC-AUC.
- **LayerNorm retained.** Removing normalization was 0.00161 lower on average,
  although individual seeds differed.
- **Standard preprocessing selected.** Median imputation, standard scaling, and
  missing indicators slightly exceeded robust scaling plus smooth clipping by
  0.00043 mean ROC-AUC and had lower median fit time. The robust path remains in
  the ablation harness, not the public classifier default.
- **Rare bucket retained.** This particular matrix contained too few genuinely
  rare values to separate the variants. The bucket has low complexity, avoids
  one-embedding-per-singleton growth, and is covered by focused tests. Its
  quality effect remains unproven.
- **Validation every epoch selected.** Direct-tensor validation was cheap and
  earlier stopping reduced mean epochs from 10.5 to 8.25 without a material
  ROC-AUC change. Validation every three epochs produced the highest mean score
  but always consumed the 12-epoch budget.
- **`min_delta=1e-4` retained.** This short matrix did not distinguish it from
  `1e-12`, but the behavioral test demonstrates that only significant changes
  reset patience. It prevents the known failure mode of chasing microscopic
  metric movement.
- **Cosine selected without warmup.** It exceeded constant by 0.00079 and
  warmup+cosine by 0.00169 mean ROC-AUC. Warmup added no measured benefit.

## Training-engine ablation

A 5,000-row, 20-numerical-plus-4-categorical tensor traversal compared a
standard DataLoader (`batch_size=256`, shuffled) with direct shuffled tensor
indexing:

| Engine | Traversal time |
| --- | ---: |
| DataLoader traversal | 0.18012 s |
| Direct tensor indexing | 0.00208 s |
| Speedup | 86.49× |

This microbenchmark isolates batch materialization and iteration, not forward
or backward compute. It directly supports the 0.1.0 in-memory engine choice.

## Deferred or rejected components

- Transformers and attention were not evaluated as a release default because
  they conflict with the small, fast first-release scope.
- Learned numerical embeddings were deferred: the standard representation won
  the scoped preprocessing ablation and the release should not become a
  numerical-embedding laboratory.
- Parameter-efficient internal ensembling was deferred so the release measures
  one neural network.
- `torch.compile` was not enabled because cold compilation is unlikely to repay
  its cost on the target small workloads without stronger evidence.
- DataLoader workers, pinned staging, and non-blocking transfers remain relevant
  for large CUDA datasets, but CPU arrays already materialized in memory did not
  justify worker overhead.
