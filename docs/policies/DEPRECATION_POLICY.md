# Deprecation and behavior-change policy

Adopted for the 1.0.0 stable scope. No current feature is deprecated and no
removal or schedule is announced. Release/publication remains a separate action.

Documented public classes, constructor parameters, methods, output/schema/weight
contracts and documented fitted observations are covered. Experimental public
options also require notice; that label does not permit silent removal. Private
underscore modules/state, incidental timing values and unspecified diagnostic
subkeys are not stable interfaces.

1. Announce a removal, rename, changed default or incompatible accepted-input/
   output behavior in CHANGELOG, API/USAGE and release notes. State the reason,
   old/new behavior, replacement, migration example, first warning version and
   earliest removal version.
2. Emit **FutureWarning** from the relevant public operation with an actionable
   replacement/schedule and stacklevel pointing to the caller. Use the tested
   internal canonical helper where appropriate. No import-time/per-row spam;
   preserve normal warning-filter behavior. Do not silently alter a fitted model.
3. Keep the old path supported through **at least one full intervening minor
   release series after first shipped notice**. First warning in 0.3.x preserves
   0.4.x; earliest announced removal is 0.5.0. Patches cannot shorten notice.
   Apply the same full-series floor to 1.x; after 1.0 a breaking removal also
   needs a major version. A numeric version check cannot prove actual shipped
   notice, especially across a major transition: verify release history.
4. Pre-1.0 incompatible evolution can use an announced minor only after notice
   and explicit compatibility review. The first 1.0 release is not permission
   to waive notice. After 1.0, incompatible defaults/shapes/dtypes/class order/
   schema/preprocessing/weights or public semantics need a major version and
   migration guidance. Compatible additions belong in a minor.
5. During notice, test old and replacement paths, warnings and affected
   serialization. Retain migration notes and actionable errors after removal.
   Dependency support changes need notice/matrix updates too; an API notice
   window never implies cross-version pickle compatibility.

Urgent correctness/security exceptions require an explicit maintainer decision,
documented scope, disclosure and recovery instructions. Do not hide a breaking
change in a performance patch. Unchanged documented behavior remains the default
for compatible fixes. See [versioning](VERSIONING_POLICY.md) and
[warning categories](WARNING_POLICY.md).
