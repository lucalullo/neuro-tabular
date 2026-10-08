# P36 — NeuroTabular 0.3.1 release-candidate validation summary

Verdict at local close: **RC_LOCAL_VALIDATED**.

## Identity

- Public baseline: NeuroTabular 0.3.0 (`2a218f2987fbb748f963fc2ab38a26c4a43c99ff`).
- P35 product-hardening source: `f9dfd79d`.
- Locally validated P36 RC reference: `a2559ff327f302893ff5383175eab29837d9c4a1`.
- Package version: 0.3.1.

## Local validation

- 178 tests passed; 3 physical-CUDA tests skipped.
- Ruff checks passed in the P36 validation environment.
- Frozen equivalence: 30/30 binary and 4/4 representative multiclass/regression
  cases bit-for-bit; persistence validation passed.
- Wheel and sdist builds passed, as did clean-install smoke tests for binary,
  multiclass, and regression tasks.
- The clean source manifest verified 67/67 entries.
- The P35 prediction optimization remained present in the clean integration;
  release-candidate spot measurements were consistent with the validated P35
  result within documented timing variability.

## Public-boundary decisions

Internal P35/P36 runtime instrumentation, local environments, caches, models,
raw measurement artifacts, and machine-specific receipts are not part of this
public tree. The public patch contains only the product optimization, regression
tests, release/packaging hygiene, documentation, and CI changes needed to
maintain the release.

## Remaining release gate

Remote CI, including the Windows Python 3.12 job, must pass on the exact Git
commit that will be tagged. Physical CUDA validation remains pending and no new
physical-CUDA claim is made by 0.3.1.

No tag or publication is created by this report.
