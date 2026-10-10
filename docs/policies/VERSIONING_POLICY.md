# Versioning and stabilization policy

P42, 2026-10-09. **ADOPTED_FOR_LOCAL_DEVELOPMENT**. Public release remains
0.3.1; no future version/date is assigned here. Semver is a practical change-
classification rule, not evidence that an artifact or platform has been tested.

| Change class | Rule |
| --- | --- |
| PATCH | Compatible bugfix, usability, documentation, packaging or performance work; preserve documented defaults/contracts and scientific behavior. Do not disguise an algorithm or breaking change as optimization |
| MINOR | Backward-compatible features/optional capabilities under the existing defaults/contracts. Pre-1.0 intentional incompatible evolution needs explicit review, notice and migration; experimental is not a silent-removal exemption |
| MAJOR | First explicit public contract stabilization at 1.0, or later incompatible public contract changes. 1.0 may stabilize the current backbone; it does not imply SOTA, universal tree superiority, every feature or mandatory CUDA |

After 1.0, changed public shapes/dtypes/class order/schema/weight semantics,
incompatible defaults or removals require a major version plus the deprecation
notice floor. Before 1.0, a breaking minor can occur only after an announced
full intervening minor series of notice and explicit migration review. Patches
never shorten notice. Verify actual shipped history across major transitions;
version arithmetic alone cannot prove notice. Correctness/security exceptions
need explicit documented maintainer decisions and recovery advice.

Reaching 1.0 means a reviewed stable scope, reliable packaging and trustworthy
behavior, clear platform/dependency/persistence boundaries, compatibility and
deprecation discipline, documentation and evidence for the claims made. It is
not blanket backward pickle portability or validation of untested platforms.
API and serialized-artifact compatibility are separate policies.

Each eventual release has matching package metadata/version module/CITATION,
changelog/release notes/docs, artifacts and immutable tag/commit. Official tags
and published files must not move or be overwritten. Unpublished builds sharing
a version require source/artifact hashes; do not distribute this rehearsal as a
replacement for the existing public 0.3.1 files.

Review README, API, USAGE, changelog and release notes for every release. Obtain
exact integrated-source CI/platform evidence, inspect distributions, run Twine
and fresh installs before a separately authorized publication. No P-phase
completion automatically creates a version or releases software.
