# NeuroTabular 2.0.0 local qualification checklist

- Freeze source, version, public allowlist and SHA256 manifest.
- Verify unchanged API/defaults/valid-input calculations against immutable S103;
  isolate the numerical-validation correction and version metadata.
- Pass full Linux and native Windows software suites, targeted privacy/metadata
  checks and explicit skip review; retain errors and technical interruptions.
- Verify same-runtime exact predictions, preprocessing, tensors, RNG, failed
  refits and trusted pickle/joblib behavior on all three tasks.
- Verify clean-checkout Git/LF identity, wheel/sdist/RECORD, strict Twine,
  private-material exclusion, dependency checks and clean installations.
- Run executable examples and document compatibility/security evidence limits.
- Verify the minimal complex+nullable correction. Record the explicit 2.0.0
  maintainer decision accepting incomplete native cost qualification as a
  nonblocking residual risk. Preserve old inconclusive/unexecuted measurements
  and frozen gates; do not claim performance PASS or integrate the experimental
  nullable fast path. Native cost qualification remains future work.
- Obtain exact-candidate CI on an authorized future release/2.0.0 commit;
  old release CI is historical. Disclose untested macOS/physical CUDA coverage.
- Complete the release-specific final review and maintainer decision. The old
  1.0.3 waiver is not automatic approval or third-party certification here.
- Only after separately authorized actions: immutable tag, TestPyPI publication,
  real download/hash/clean-install smoke, then same-artifact PyPI promotion.

No local qualification step authorizes commit, push, merge, tag or publication.
