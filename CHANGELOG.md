# Changelog

All notable public changes to NeuroTabular are recorded here. Published tags and
release artifacts are immutable.

## [1.0.3] - 2026-10-18

- Finalize the maintenance release from the qualified RC2, preserving public
  API, defaults, dependency bounds and exact same-runtime numerical behavior.
- Retain measured categorical/schema preprocessing and CPU inference improvements,
  LF source identity, source-manifest verification and private-file exclusions.
- Record RC2 CI #17: all 17 jobs passed on the exact RC2 commit. The stable
  version requires its own exact-commit CI before tag and registry publication.
- Promote the original verified TestPyPI wheel and sdist to PyPI by immutable
  run/artifact identity and SHA256, without rebuilding them for promotion.
- Record the maintainer's release-specific waiver of the final professional
  independent-attestation gate. This is not independent certification.
- No general improvement in predictive accuracy, training time or process peak
  RAM is claimed. Historical prerelease entries retain their freeze-time status.

## [1.0.3rc2] - 2026-10-18 (local release candidate; unpublished)

- Normalize five Markdown documents to LF to match Git attributes and regenerate
  the source manifest from repository bytes. This repairs source identity across
  clean Git checkouts without changing the meaning of the normalized text.
- Assign a distinct RC2 version to metadata and release documentation. Preserve
  frozen RC1 artifacts, hashes and qualification evidence without alteration.
- No computational implementation, API, defaults, dependency bounds or measured
  performance claims change. Independent review and exact-commit CI remain open.

## [1.0.3rc1] - 2026-10-18 (local release candidate; unpublished)

- Preserve the stable 1.0 API/defaults while reducing categorical preprocessing,
  schema validation and CPU inference overhead. Numerical and persistence
  equivalence require qualification in the same runtime.
- Reject `.local` and `.codex-context` in distribution content checks and
  explicitly prune both from source distributions.
- Reduce redundant categorical Index construction, schema work, numerical
  matrix copies and CPU inference allocations without persistent caches.
- In the frozen Linux CPU comparison with original 1.0.2, one 1,200-row multiclass
  `predict_proba` workload improved by 14.14% (six paired comparisons, all faster).
  Temporary array allocation reductions are workload-specific; small
  inputs can allocate slightly more. No general peak-RAM saving is established.
- Preserve exact predictions, trained state and persistence in the qualified
  same-runtime comparisons. No accuracy or general full-training gain is claimed.
- Record native Windows 11 / CPython 3.12 x64 qualification of the unchanged
  computational implementation, in addition to the Linux compatibility matrix.
- Assign a distinct patch prerelease identity and refresh public documentation,
  source hashes and artifact metadata. Independent final review and publication
  remain separate maintainer gates. No research prototype is promoted.

## [1.0.2] - Compatibility patch (historical baseline)

- Remove the unnecessary Python <3.13 metadata cap; retain Requires-Python >=3.10.
- Validate conventional CPython 3.11–3.14 on Linux CPU, including exact 3.13.15,
  ordinary PyPI dependency installs, minimum/current stacks and warnings as errors.
- Retain Python 3.10 legacy compatibility; distinguish Python 3.15 code compatibility
  on the official Torch CPU channel from its ordinary PyPI distribution limitation.
- Expand the proposed Linux/Windows CI matrix and add Apple Silicon macOS coverage.
- Update installation/support guidance, version metadata and release notes.
- Preserve runtime implementation bytes except the version constant and frozen
  30/30 binary plus 4/4 representative predictions bit for bit.
- Retain the four existing direct dependency bounds. Require joblib>=1.5 on
  Python 3.12; >=1.4.2 excluding 1.5.0/1.5.1 on Python 3.13; >=1.5.2 on
  Python 3.14; and >=1.6 on Python>=3.15. These prevent removed-distutils,
  pickle-hashing and loky worker/resource-tracker failures. Add cached-Pipeline
  and parallel cross-validation regression tests.
- Keep public API, training, predictive defaults and persistence contracts unchanged.
  Remote CI and publication remain separate maintainer gates.

## [1.0.1] - 2026-10-11

- Make the package README suitable for PyPI with absolute documentation links
  and stable installation/status guidance.
- Align version/citation metadata, release notes and the version regression test.
- Retain the existing manual Trusted Publishing workflow and separate
  TestPyPI/PyPI environments.
- Preserve all runtime implementation bytes except the version constant;
  no algorithm, API, default, feature or dependency-bound change.

## [1.0.0] - 2026-10-11

- Stabilize the reviewed public estimator/API scope for the first stable release.
- Retain RC1 target hardening and earlier product/persistence/packaging work.
- Promote version, stable metadata, installation guidance and support policies;
  no algorithm, predictive default, dependency-bound or feature change.
- Complete fresh full validation, frozen equivalence, clean installs and an
  exact stable-commit seven-job remote CI result before publication.
- Disclose independent human pilots, macOS and physical CUDA evidence limits.

## [1.0.0rc1] - Published release candidate

### Target contract hardening

- Classifier fit, eval_set and score share single-output binary/multiclass
  target validation. Fractional, non-finite and unsupported multioutput targets
  raise ValueError before training, encoding or metrics.
- Object-backed scalar labels receive value-preserving pandas inference before
  sklearn target typing. Numeric strings stay strings. Zero-weight training rows
  remain ignored before target discovery, including their invalid labels.
- Regression score now shares the existing finite numeric-target contract of
  fit and eval_set. Valid regression values and R2 semantics are unchanged.
- Constructor defaults, prediction math, training, preprocessing, checkpoint
  selection and valid-model persistence are unchanged by this repair.
- Add focused and randomized target tests, transactional refit checks and an
  explicit target contract. This intentionally removes erroneous acceptance of
  unsupported targets before 1.0; it is not an artifact migration mechanism.

### Candidate scope

- Include the previously unreleased product diagnostics, support guidance and
  maintenance policies listed below. Physical CUDA remains optional and unvalidated.
- RC1 was subsequently published with exact-commit seven-job CI PASS.
  This historical result does not replace stable-commit remote CI.
- Independent user pilots remain outstanding evidence and a disclosed soft risk.
  Stable 1.0 approval and publication are separate maintainer decisions.

### Previously unreleased product hardening

- Reconcile the public-0.3.1-based local stabilization candidate and canonical
  policy/support scope; classify the intended 1.0 boundary without enacting a freeze.
- Preserve Python 3.10–3.12 intent, scope CUDA to optional best effort and prepare
  independent acceptance evidence. These earlier changes did not alter algorithm or dependency bounds.

- Adopt local-development API maintenance, deprecation, warning, persistence and
  versioning policies without declaring a 1.0 freeze or changing the runtime.
- Add support reporting/reproducer guidance and exact privacy-output checks;
  clarify activation, standalone sklearn examples and optional diagnostics.
- Keep historical compatibility models/receipts in the repository only; exclude
  them and internal research memories from public distributions. Enforce strict
  artifact hygiene in CI and manual-build workflows before any future upload.
- Add a compact, training-only `input_schema_` fitted diagnostic without new
  constructor options or changes to predictions and training semantics.
- Identify invalid numeric and duplicate columns and clarify schema/DataFrame
  and CUDA recovery messages, preserving exception types and strictness.
- Extend mixed-data persistence, refit/RNG and sklearn regression coverage;
  document fitted inspection and a DataFrame-preserving pipeline example.

## [0.3.1] - 2026-10-08

### Changed

- Reduce CPU prediction-path overhead for already numeric columns and DataFrames
  whose columns already match the learned schema order.
- Preserve the original conversion, schema-reordering, missing-value,
  categorical, and caller-ownership paths when the fast paths do not apply.
- Preserve prediction/probability behavior bit for bit on the frozen release
  validation, including the 30-case binary equivalence panel.

### Validation and packaging

- Add product-path regression tests for numeric conversion, memory ownership,
  schema ordering, and mixed/nullable numeric inputs.
- Add distribution-hygiene checks for wheel/sdist contents and local/private
  artifact leakage.
- Run CI on release branches in addition to `main`, while retaining Python
  3.10-3.12, Windows, minimum-PyTorch, Ruff, build, and installed-artifact jobs.
- Document the scoped P35 CPU/WSL performance measurements in
  `docs/PERFORMANCE_0_3_1.md`.

NeuroTabular 0.3.1 is a compatible performance/product-hardening patch. It does
not intentionally change the algorithm, defaults, model capacity, or predictive
quality. Physical CUDA validation remains pending.

## [0.3.0] - 2026-09-20

### Added

- Automatic multiclass classification with ordered labels, cross entropy,
  normalized softmax probabilities, weights and loss/accuracy stopping.
- `NeuroTabularRegressor` with train-only weighted target standardization,
  inverse transformation and loss/RMSE/MAE/R² evaluation.
- Shared preprocessing, neural backbone and multi-task training core.
- Pickle/joblib validation in fresh processes for all three tasks.
- Expanded schema, weighting, target, device, sklearn and packaging validation,
  with documentation and frozen real-data benchmark evidence.

### Changed

- Refactor internal estimator/training code while preserving exact v0.2.0 binary
  behavior on the frozen 30-case equivalence panel and retaining public defaults.

## [0.2.0] - 2026-09-06

### First public release

- Add `NeuroTabularClassifier`, a scikit-learn-compatible neural binary
  classifier for heterogeneous pandas DataFrames.
- Add automatic numerical/categorical detection, missing-value handling,
  training-only preprocessing, categorical embeddings, rare/unknown IDs, and
  categorical log-frequency side features.
- Add scalar numerical inputs by default plus explicit affine, periodic, and
  piecewise experimental representations.
- Add adaptive categorical embedding widths, residual MLP backbone, optional
  feature gating, and optional full-data refit.
- Add AdamW, cosine scheduling, automatic batching, internal/external validation,
  early stopping, strict best-checkpoint restoration, sample weights, and
  `class_weight="balanced"`.
- Add transactional re-fitting and restoration of caller Python/NumPy/PyTorch
  CPU RNG states.
- Treat declared-but-unobserved pandas categorical levels as unknown.
- Reject complex values and unsafe float32 conversions explicitly.
- Ignore zero-weight rows before target discovery and preprocessing.
- Add CPU/CUDA diagnostics, synchronized CUDA probing, safe automatic fallback,
  and AMP only on compatible CUDA devices.
- Add Python 3.10-3.12, PyTorch 2.0+, Linux/Windows CI, wheel/sdist verification,
  and clean-wheel smoke tests.
- Add benchmark, performance, ablation, research, usage, API, security,
  contribution, publishing, and release documentation.

### Status

NeuroTabular 0.2.0 is experimental and pre-1.0. No state-of-the-art or universal
performance claim is made.
