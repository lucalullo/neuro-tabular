# Release publication on main

Two manually dispatched workflows publish verified distributions directly using
PyPI Trusted Publishing. Neither runs on push or requires a personal API token.
The workflow dispatch is the maintainer's explicit publication decision for the
selected index. Both preserve the exact-commit 17-job CI gate.

## One-time setup

Register a GitHub Trusted Publisher for `neurotabular` on each index (use the
project's Publishing settings, or a pending publisher if the project does not
exist there yet). Use these exact values:

| Field | TestPyPI | PyPI |
| --- | --- | --- |
| Owner | `lucalullo` | `lucalullo` |
| Repository | `neuro-tabular` | `neuro-tabular` |
| Workflow filename | `publish-testpypi.yml` | `publish-pypi.yml` |
| Environment | `testpypi` | `pypi` |

See [Trusted Publishing setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
Existing environment protection rules remain in effect. Keep the production
approval rules required by the repository's security policy. OIDC permissions
are granted only to the separate publication jobs; build and test jobs have
read-only permissions.

## Publication sequence

1. Push the final repository to `main` and wait for all 17 CI jobs to succeed on
   the exact commit.
2. In Actions, select **Publish to TestPyPI**, leave the branch set to `main`,
   and click **Run workflow**. The job checks CI, source hashes, lint, tests,
   metadata, archive contents and installation. A separate job automatically
   uploads the verified wheel and sdist to TestPyPI using Trusted Publishing.
3. Review the successful run and the TestPyPI project page. Do not change `main`
   or rebuild the distributions before production promotion.
4. In Actions, select **Publish to PyPI** on the same `main` commit and click
   **Run workflow**. It automatically finds the successful TestPyPI publication
   on that exact commit, downloads its original locked bundle, and verifies
   the hashes and actual downloaded bytes from TestPyPI. A missing, ambiguous,
   expired or mismatched source causes a failure. No replacement build is made.
   The separate production job publishes the identical files to PyPI.
5. Create the GitHub version tag and release on the verified commit when ready.

No additional branch, tag, manual upload or user-entered run ID is required for
these two workflows. The separate **Manual trusted publication** workflow retains
its original tag, release-branch CI and immutable-promotion guards.

Published versions cannot be overwritten with changed files. Do not retry an
upload blindly after a partial publication. Investigate the registry and the
workflow receipts first. Do not commit downloaded distributions or credentials.
