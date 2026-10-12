# NeuroTabular 1.0.1

NeuroTabular 1.0.1 is a packaging and documentation patch for the stable 1.0
public API. It provides PyPI-ready absolute README links, stable installation
guidance and consistent version/citation metadata. The existing manual OIDC
publishing workflow and separate TestPyPI/PyPI environments are retained.

Binary classification, multiclass classification, single-output regression,
pandas mixed-data preprocessing, sklearn workflows and trusted persistence
retain the behavior of 1.0.0. All runtime implementation files are identical to
1.0.0 except the version constant. No algorithm, API, predictive default,
feature or dependency-bound change is introduced.

The validation contract is the full regression suite, 30 frozen binary cases
and four representative cases with exact bitwise comparisons, isolated wheel
and sdist installs, fresh-process pickle/joblib checks, a cleanroom rebuild,
strict Twine rendering, distribution hygiene and the source manifest.
Exact-commit remote CI and package-index publication are separate maintainer
gates; earlier 1.0.0 CI is not evidence for a new 1.0.1 commit.

Python 3.10–3.12 and PyTorch 2.0+ remain the declared compatibility scope.
Linux/Windows CPU CI is required before publication. Physical NVIDIA CUDA is
best effort with physical validation pending; macOS and independent human
pilots remain outside the validated evidence. Same-runtime persistence does
not promise cross-version, dependency, OS or device portability.

This patch makes no SOTA or universal/tree-model superiority claim. Published
1.0.0 and earlier tags and release assets remain immutable.

NeuroTabular 1.0.0 was published after its exact stable commit passed the
seven-job remote CI matrix. Its immutable release identity and historical RC
evidence remain distinct from this 1.0.1 handoff.
