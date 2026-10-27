# Versioning and stabilization policy

Adopted for 1.0.0. Semver is a practical change-classification rule, not proof
that a platform/artifact has been tested. Published older identities remain immutable.

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
a version require source/artifact hashes; do not replace any existing published version with a different build.

Review README, API, USAGE, changelog and release notes for every release. Obtain
exact integrated-source CI/platform evidence, inspect distributions, run Twine
and fresh installs before a separately authorized publication. No P-phase
completion automatically creates a version or releases software.


## Maintainer decision for the 2.0 release line

On 24 October 2026 the maintainer explicitly selected `2.0.0` / `v2.0.0` as a
major maturity/consolidation milestone. This choice is not evidence of an API
break, a new predictive architecture or superior accuracy. Preserve the tested
1.x contracts unless a deliberate reviewed migration is separately documented.
Private validation builds may be hash-identified; no public rc/alpha/beta/dev
release is planned for this consolidation. Publish only after mandatory gates
and explicit maintainer authorization, with one exact final artifact identity.

After 2.0 publication, ordinary releases require concrete validated predictive,
performance, memory, stable-feature, compatibility or usability value. Do not
release for every private experiment. Security, correctness, compatibility and
critical regression fixes must not be delayed for release-frequency targets.
Every release still needs tests, documentation and comparison with its baseline.

Recommend the current stable line; preserve older working versions, tags and
assets for reproducibility. Never delete or yank history for cosmetic reasons.
Any justified destructive cleanup or security yank needs its own explicit
maintainer decision, affected-item inventory and non-destructive alternative.
