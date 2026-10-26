# NeuroTabular 2.0.0

## Scope

This is the maintainer-designated major maturity/consolidation release. The
number does not imply a new architecture, state-of-the-art accuracy, or a general
speed/RAM improvement. Binary classification, multiclass classification and
single-output regression retain the public 1.x estimator API and defaults.

The numerical-input validation defect involving complex columns beside nullable
pandas columns is corrected before a real conversion can discard information.
The correction reuses the already established real-dtype predicate and inspects
converted column dtypes where needed. Valid imputation, scaling, categories,
tensor preparation, losses, threshold, scheduler and network calculations stay
unchanged. No persistent cache or dependency on an experimental fast path.

Versioned private-folder exclusions, manifest/Git newline reproducibility and
the public packaging safeguards are retained. Prior 1.0.3 local inference and
temporary-array benefits are inherited, not counted as new 2.0 gains.

Extreme finite float64 inputs whose training-statistic calculation overflows
are rejected with a controlled ValueError. The numerical error-policy context
is restored on both success and failure. Complex sample weights are rejected
before any conversion can discard imaginary values. These are input-safety
corrections; they do not change the predictive architecture or establish a
generalization or speed benefit.

Non-representable numeric conversions, including oversized Python integers
and wider floating-point values outside float64 range, are rejected explicitly
for targets, weights and numerical features. Overflow during conversion does
not leak a RuntimeWarning or an uncaught OverflowError. Ordinary values retain
the same conversions; no new precision or performance guarantee is implied.

## Compatibility and security boundaries

Numerical/predictive equivalence, dtype/error/refit/RNG contracts and trusted
pickle/joblib boundaries require the exact-release validation receipts. Loading
untrusted pickle/joblib is unsafe. A successful tested 1.0.3 artifact replay does
not establish general major-version, dependency, device or OS portability.

The old pinned Windows qualification stack is historical and is not reused as a
security certificate. The new native qualification resolves and hash-locks
patched dependencies and audits their actual versions, including transitive
filelock and build tooling. Package audits are not exhaustive OS/security proof.
See the support policy for tested platforms and explicit unverified hardware.

## Performance evidence and release decision

Dedicated native Linux/Windows A/A, A/B and process-RSS qualification is not
complete. Inconclusive metrology does not establish either a slowdown or a
speedup. The maintainer explicitly accepts this residual performance risk for
2.0.0 publication while retaining the functional guard. Frozen protocols,
thresholds and historical outcomes are unchanged; native qualification remains
future work. This is a release-policy decision, not a performance PASS or an
independent certification. No general new training, inference, accuracy or RAM
benefit is claimed for 2.0.0.

The local Windows Smart App Control denial concerns an exact unsigned pandas
extension in one host configuration. It is recorded separately from library
correctness; no protection was disabled. Windows CI on the exact release commit
remains mandatory and must not be replaced by WSL performance evidence.

## Publication and future releases

These release sources and local artifacts do not authorize publication. Exact
commit CI, required release review/decision and the maintainer's separate
publication actions remain mandatory. No preliminary public releases are planned.
The published 1.0.3 artifacts and all earlier history remain immutable.

After publication, 2.0 is the recommended current line. Ordinary later releases
must bring concrete validated benefits; critical fixes are not delayed merely
to reduce release frequency. Historical releases are preserved by default;
deletion/yanking requires a separate justified maintainer decision.
