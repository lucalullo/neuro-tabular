# NeuroTabular performance profile

The current release profile is
[PERFORMANCE_PROFILE_0_2.md](PERFORMANCE_PROFILE_0_2.md).

Version 0.2.0 preserves the final 0.1.1 CPU/CUDA maintenance behavior and adds
stage-level preprocessing, training, prediction, parameter, RSS, compile, and
optimizer evidence. The canonical report also explains why cold-process
timings are not used as the main release speed claim and records that GPU/VRAM
measurements were unavailable on the release host.

Historical 0.1.0 and 0.1.1 changes remain described in `CHANGELOG.md` and their
immutable release artifacts. This file intentionally follows the current
release rather than duplicating versioned measurements.
