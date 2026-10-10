# NeuroTabular 1.0.0rc1 local candidate checklist

This checklist authorizes no remote writes or publication. Public 0.3.1 identity
remains immutable. Stable 1.0 acceptance is a separate maintainer decision.

- Confirm the intended baseline, final local commit and clean working tree.
- Align RC version metadata, citation, changelog and release notes.
- Pass full tests, Ruff check and Ruff format --check on the exact RC source.
- Close the continuous/non-finite target blockers across fit/eval_set/score.
- Preserve valid binary/multiclass/regression, transactional refit and caller RNG.
- Require 30/30 frozen binary and 4/4 representative exact equivalence.
- Verify weights, schema, clone/Pipeline/CV and trusted persistence.
- Load current and scoped public-0.3.1 pickle/joblib models in fresh processes.
- Build wheel/sdist from a clean source archive; run Twine and distribution hygiene.
- Install wheel, sdist, source and editable source in separate clean environments.
- Execute documentation, privacy/path audits and performance safety comparisons.
- Freeze source metadata, regenerate the source manifest, verify complete N/N,
  and record source/artifact SHA256 hashes.
- Preserve only public material in the GitHub-ready repository and source ZIP.
- Mark exact-corrected-commit remote CI PENDING until the maintainer supplies it.
- Disclose native Windows host-policy blocking separately from remote Windows CI.
- Disclose independent-user pilots as outstanding soft risk and physical CUDA
  as optional PENDING; do not count local tests as independent-user evidence.
- Stop after the local commit and evidence. No push, tag, release or upload.
