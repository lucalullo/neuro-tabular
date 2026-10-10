# API maintenance policy

P42, 2026-10-09. **ADOPTED_FOR_LOCAL_DEVELOPMENT** as a compatibility/review
process. This does **not** enact the proposed 1.0 API freeze. Public release is
pre-1.0 0.3.1, with no signature/default/runtime changes in this phase.

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

P43 defines the intended stable/experimental boundary in [API_BOUNDARY](../API_BOUNDARY.md)
and [support scope](SUPPORT_SCOPE.md); both remain prospective, not an actual freeze.
Before an actual 1.0 freeze, review that boundary against external findings,
publish the supported platform/dependency boundaries and policies, obtain exact-
source compatibility evidence and address concrete usability failures. A version
number, new backbone, SOTA claim or universal tree superiority is insufficient.
Features left experimental can stay available with explicit limits; 1.0 need
not add every possible input type, calibration or CUDA support.

No default/architecture/preprocessing/optimization change is authorized by this
policy. Release/tag/publication decisions remain separate, and research remains
frozen under D37/R38.
