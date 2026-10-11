# API maintenance policy

Adopted for the 1.0.0 stable public scope. RC1's authorized target hardening
is retained without constructor-default or valid-input numerical changes.
The exact stable/experimental/internal boundary is [API_BOUNDARY](../API_BOUNDARY.md).

The public surface is NeuroTabularClassifier, NeuroTabularRegressor and package
__version__; documented constructor parameters, methods, outputs, error families
and fitted observations have the scopes in API.md. Constructor storage/clone,
DataFrame schema alignment, train-only preprocessing, positional target/weight
alignment, class order, single-output shapes, transactional refit and caller
CPU RNG restoration must remain tested. set_params changes pending next-fit
settings and retains the current trained predictor; clone is unfitted.

All fitted state is read-only by convention. Live class/device/target state is
not a reconfiguration interface. The detached optional input_schema_ snapshot
does not control prediction. Timing/profile structures and heuristics are
diagnostics, not numerical or service guarantees. Internal fit-version metadata
does not control loading. Private implementation modules are not stable APIs.

Before changing a documented public item, classify compatibility, update tests
and docs, follow [deprecation](DEPRECATION_POLICY.md) and
[versioning](VERSIONING_POLICY.md), and validate affected persistence/device/
sklearn behavior. Experimental public parameters still need explicit notice.
Do not claim full sklearn check_estimator or automatic eval_set transformation
from the tested subset of workflows.

The reviewed stable contract does not require every input type, calibration,
mandatory CUDA, a new backbone, SOTA or universal tree superiority. Experimental
features remain available with explicit limits and notice discipline. Technical
validation and public release decisions remain separate.

No default/architecture/preprocessing/optimization change is authorized by this
policy. Release/tag/publication decisions remain separate, and research remains
frozen under D37/R38.
