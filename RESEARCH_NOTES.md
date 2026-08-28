# NeuroTabular 0.1.x research and design notes

## Scope and originality

These notes record concepts studied for independent implementation. No external
model source was copied, vendored, or imported as NeuroTabular's internal model.
LightGBM, CatBoost, RealMLP, and TabM are not runtime dependencies.

## Primary sources reviewed

- Ke et al., [LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html),
  NeurIPS 2017.
- Prokhorenkova et al., [CatBoost: Unbiased Boosting with Categorical
  Features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html),
  NeurIPS 2018.
- Holzmüller, Grinsztajn, and Steinwart, [Better by Default: Strong Pre-Tuned
  MLPs and Boosted Trees on Tabular Data](https://openreview.net/pdf?id=3BNPUDvqMt),
  NeurIPS 2024.
- Gorishniy, Kotelnikov, and Babenko, [TabM: Advancing Tabular Deep Learning
  with Parameter-Efficient Ensembling](https://openreview.net/pdf?id=Sd4wYYOhmY),
  ICLR 2025.
- Gorishniy, Rubachev, and Babenko, [On Embeddings for Numerical Features in
  Tabular Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html),
  NeurIPS 2022.
- PyTorch, [Performance Tuning
  Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
  and [Automatic Mixed Precision
  documentation](https://docs.pytorch.org/docs/stable/accelerator/amp.html).
- PyTorch CUDA API documentation for
  [`is_available`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available),
  [`get_arch_list`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html),
  [`get_device_capability`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_device_capability.html),
  and [`synchronize`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize).
- PyTorch,
  [`AdamW`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW)
  documentation.
- NVIDIA CUDA Programming Guide,
  [CUDA platform and binary/PTX compatibility](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html).

## Lessons applied

### Efficiency before architectural scale

LightGBM's implementation is tree-based and was not reproduced. Its engineering
lesson—compact representations, one-time preprocessing, minimal scans, and
dataset-size-aware work—motivated profiling the data path before enlarging the
network. The measured DataLoader overhead justified direct in-memory indexing.

### Leakage prevention

CatBoost's paper emphasizes leakage risks in categorical target statistics.
NeuroTabular 0.1.0 therefore uses no target encoding. Vocabularies and numerical
statistics are fitted on training rows only; validation and unseen categories
cannot alter them.

### Strong small MLPs and preprocessing

RealMLP demonstrates that preprocessing, defaults, schedules, and engineering
can matter as much as exotic backbones. NeuroTabular evaluated standard versus
robust/smooth-clipped numerical preprocessing, residual versus plain MLPs,
activation, normalization, and learning-rate schedules. Standard preprocessing,
residual blocks, SiLU, LayerNorm, and cosine decay were selected from the scoped
ablation rather than copied from an implementation.

### Parameter-efficient ensembling

TabM provides evidence that parameter-efficient neural ensembles can improve
tabular MLPs. NeuroTabular 0.1.0 deliberately exposes one network so its
strength, cost, and failure modes remain measurable. The module boundaries do
not prevent a future shared-backbone/adapters design, but no unused ensemble
abstraction is included.

### Numerical embeddings

The numerical-embedding paper shows that piecewise-linear and periodic
representations can improve multiple backbones. This is a credible future
direction. It was deferred because the release ablation slightly favored the
simple standard representation and the first release needs a small, auditable
baseline.

### PyTorch execution

The PyTorch guide supports inference without gradients, gradients set to `None`,
pinned CUDA staging, and workload-specific data-loading decisions. The release
uses `torch.inference_mode()`, `zero_grad(set_to_none=True)`, direct indexing for
materialized tensors, optional pinned staging, non-blocking CUDA transfers,
autocast, and a fused-AdamW attempt. It does not globally change thread counts,
TF32, or deterministic settings.

### CUDA compatibility in 0.1.1

`torch.cuda.is_available()` answers whether CUDA is available at runtime, but
the externally reproduced P100 failure demonstrates that this alone is not a
sufficient application-level guarantee that the installed PyTorch binary can
execute a kernel on the visible device. `get_arch_list()` reports architectures
compiled into the library and is valuable diagnostic evidence.

NeuroTabular does not treat exact `sm_XY` membership as the final compatibility
test. NVIDIA documents that cubins have same-major compatibility constraints
while PTX can be JIT-compiled for later compute capabilities. Newer CUDA targets
also include family- and architecture-specific variants. A string-membership
rule would therefore reject valid forward-compatible cases or oversimplify
future targets. Version 0.1.1 instead launches a one-element PyTorch CUDA kernel
and calls `torch.cuda.synchronize()` so asynchronous launch failures surface
before model setup. Architecture metadata remains in diagnostics.

AMP is restricted to a verified CUDA path with compute capability 7.0 or newer,
where Tensor Core acceleration makes float16 autocast an appropriate default.
CPU never enables AMP. The optimizer no longer forces `fused=True`: current
PyTorch documentation describes fused as newer and lets the default prefer a
supported CUDA foreach implementation. This also avoids a class of failures
where fused construction and first-step support differ across versions.

## Decisions not adopted

- No tree, boosting, target encoding, or external model inside the estimator.
- No Transformer or attention default in 0.1.0.
- No internal ensemble in 0.1.0.
- No advanced numerical embedding in 0.1.0.
- No automatic `torch.compile`; measured target jobs are too short to assume
  compile cost is recovered.
- No DataLoader workers for already-materialized small/medium tensors.
- No global PyTorch configuration mutation for application-wide performance.

## Evidence limits

The release benchmark is synthetic, CPU-only, and small. It is sufficient for
engineering choices and regression baselines, not for broad scientific claims.
Future architecture changes should use more seeds, public datasets, separate
meta-train/meta-test reasoning for defaults, and real CUDA profiling.
