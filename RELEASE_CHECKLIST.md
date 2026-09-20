# NeuroTabular 0.3.0 release checklist

- Confirm the intended release commit and a clean working tree.
- Keep package metadata, package version, citation, docs, tests, and tag aligned.
- Pass Ruff check, Ruff format, and the complete test suite.
- Preserve binary compatibility evidence and validate multiclass/regression behavior.
- Verify weights, feature schema, sklearn clone/pipeline/CV, and persistence.
- Load pickle/joblib models in new processes outside the checkout.
- Build wheel and sdist from clean source and inspect their contents.
- Install both artifacts in clean environments and run three-task smoke tests.
- Verify source/artifact SHA256 manifests and security/path audits.
- Run configured Linux/Windows CI on the exact release commit.
- Disclose that physical CUDA hardware validation is not part of 0.3.0 evidence.
- Create immutable tag `v0.3.0` only after the exact commit is accepted.
- Publish release artifacts manually; never move or overwrite an existing tag.
