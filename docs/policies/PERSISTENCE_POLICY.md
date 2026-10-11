# Persistence and upgrade policy

Adopted for 1.0.0. This deliberately promises less than a portable model format.
Load only trusted pickle/joblib artifacts: unpickling can execute code. No new
serialization API or loader is introduced.

| Boundary | Policy | Required evidence / recovery |
| --- | --- | --- |
| Same version **and source build**, pinned Python/dependencies, CPU | SUPPORTED in tested same-runtime binary/multiclass/regression scope | Fresh-process round-trip with retained probe outputs/class order/schema |
| Same version string, different unpublished source build | Only explicitly tested build pairs | Record source hashes; version equality is insufficient |
| Patch version | BEST EFFORT, not universal backward compatibility | Test original trusted artifact and predictions in the target runtime before adoption |
| Minor version | NOT GUARANTEED unless an explicit migration/load path is validated | Keep original environment/artifact; validated migration or retrain with original training data |
| Major version | NOT GUARANTEED | Release notes must describe supported migration or retraining; API stability alone is insufficient |
| Different Python/Torch/NumPy/pandas/sklearn/joblib, OS or device | NOT GUARANTEED merely by install support | Validate the exact boundary separately; set_params(device='cpu') is not an artifact remapper |

The CPU scope is grounded in fresh-process same-runtime tests and specific
trusted public-0.3.1 -> this candidate fixture comparisons. Those immutable
fixtures stay in the local repository and are excluded from public packages.
Historical-reference tests explicitly require their recorded runtime. This
policy does not infer all patch upgrades, future runtimes or physical CUDA.

Store original environment/source identity, artifact hash, small trusted probe
frame, expected outputs/class order and schema privately alongside important
models. Internal `_fitted_neurotabular_version_` records fit-time version when
present; example JSON records export-time version. Neither is a security
signature or compatibility validator. Old public 0.3.1 models lack the newer
schema/version fields; absence means UNKNOWN and loading does not invent them.

On upgrade failure, preserve the original files/environment; do not edit pickles,
drop state or suppress incompatibility silently. Compare load **and prediction**,
then adopt a specifically validated migration or refit intentionally. For a
public issue, use a new synthetic reproducer, not a private model upload.

Any future supported migration needs source/runtime-bound tests and documented
failure/recovery semantics before its guarantee is expanded. No cross-version
artifact guarantee is a prerequisite for 1.0 if its limits are explicit.
