# Security policy

NeuroTabular is experimental and pre-1.0. Security fixes are provided for the
latest released line when practical; older pre-1.0 releases may not receive
fixes.

## Reporting a vulnerability

After a canonical public repository exists, use GitHub Private Vulnerability
Reporting from its Security tab. Do not publish confidential data, credentials,
exploit details, or an undisclosed vulnerability in a public issue.

If private reporting is unavailable, open a public issue requesting a private
contact channel without including vulnerability details. No email address is
listed until the project has a real monitored security inbox.

Privately include the affected version and environment, a minimal reproduction
without private data, expected and observed impact, and possible mitigation.

NeuroTabular 0.1.0 does not deserialize model files, access accounts, download
models or datasets, make fit-time network requests, or collect telemetry.
