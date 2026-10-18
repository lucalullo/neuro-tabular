# NeuroTabular 1.0.3 publishing guide

The release identity is 1.0.3, with immutable tag v1.0.3. Published versions and
the frozen RC1/RC2 archives must not be overwritten, moved or reused for changed
source. The maintainer approved a release-specific waiver of final professional
independent attestation. That decision is not independent certification and does
not waive technical checks, staged registry verification or separate approvals.

RC2 commit d969836a63b0bacacc19ed8f1f04ee75581144ff passed all 17 jobs of
[CI #17](https://github.com/lucalullo/neuro-tabular/actions/runs/39141613971).
The stable commit has different metadata and publication safeguards; it must
pass its own exact-commit CI. No publication is performed by local preparation.

## Local and exact-commit checks

Run Ruff check/format, the tests with warnings as errors, the complete source
manifest, package builds, member hygiene, strict Twine and fresh wheel/sdist
installation checks. Preserve equivalence and same/fresh-process persistence
receipts in the qualified runtime. Do not infer cross-runtime portability.

~~~bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
python tools/release_artifacts.py source
python -m build
python tools/check_distribution.py dist/*.whl dist/*.tar.gz
python -m twine check --strict dist/*
~~~

Freeze final metadata/source bytes before generating SOURCE_MANIFEST.sha256.
Preserve its sorted path list and two-space format; exclude the manifest itself,
Git/workflow-only files and the existing public research/fixture exclusions.
Every source and sdist entry must verify. Renew affected validation after edits.

The maintainer commits and pushes release/1.0.3 manually only after reviewing the
local inventory. Record the full commit SHA as S and compare its tree with the
local qualification receipt. Confirm all 17 required jobs of the push CI on S:
Ruff; Linux Python 3.10-3.14; Windows 3.11-3.14; macOS/Python 3.14;
PyTorch 2.0-family; useful binary minima; build/hygiene/Twine/install-smokes.
Skipped jobs are not successful jobs; documented optional test skips remain
distinct. Physical CUDA and broader platform/dependency pairings are not implied.

## Tag: a separately authorized operation

After the stable CI passes and the maintainer authorizes the tag, create it
on exactly S. Do not silently merge, amend or rebase between CI and tagging.
A changed commit needs new qualification.

~~~bash
git tag -a v1.0.3 S -m "NeuroTabular 1.0.3"
git rev-parse "v1.0.3^{commit}"
git push origin refs/tags/v1.0.3
~~~

S is a placeholder for the reviewed full SHA, not a literal Git ref.
No existing published tag may be replaced or force-pushed.

## Trusted Publishing configuration

Configure separate publishers in the TestPyPI and PyPI project accounts:

| Setting | TestPyPI | PyPI |
| --- | --- | --- |
| Owner | lucalullo | lucalullo |
| Repository | neuro-tabular | neuro-tabular |
| Workflow filename | publish.yml | publish.yml |
| GitHub environment | testpypi | pypi |

Check the exact case and account/project identity. The workflow uses
id-token: write only in publishing jobs and the official PyPA action; it does
not require stored API-token secrets. GitHub environment existence alone does
not prove a registry publisher is configured or that OIDC authentication works.
Review environment approval/tag restrictions before dispatch. No setting is
created or changed by local qualification.

See [publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [publisher usage](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
Automatic cryptographic publish attestations, when produced by PyPA, bind a
workflow identity to files. They are distinct from professional quality review.

## First dispatch: TestPyPI only

After separate authorization, run publish.yml on v1.0.3 with target=testpypi
and confirmation=v1.0.3. The existing workflow must be present on the default
branch; selecting the immutable tag chooses its version of the workflow.
The CLI makes the ref and all inputs explicit if the web UI shows older inputs:

~~~bash
gh workflow run publish.yml --repo lucalullo/neuro-tabular --ref v1.0.3 --field target=testpypi --field confirmation=v1.0.3
~~~

The tag/version/confirmation and exact stable-branch CI must match.
Only this target builds distributions. It validates and locks them in an
immutable artifact named release-distributions-RUN-ATTEMPT. Record RUN,
ATTEMPT, artifact ID, manifest.json, wheel and sdist SHA256, logs and exit status.
The TestPyPI job publishes that locked bundle. Do not rebuild it for promotion.

If a version/file already exists or an upload is partial, stop and reconcile
its actual hashes. There is no automatic skip-existing, replacement or rebuild
fallback. Never upload the private maintainer/research review packet.

## Verify the actual TestPyPI installation

Wait for the successful TestPyPI run; the PyPI job must be skipped.
Download its exact immutable artifact into a new private directory, for example:

~~~bash
gh run download RUN --repo lucalullo/neuro-tabular --name release-distributions-RUN-ATTEMPT --dir .local/testpypi-bundle
~~~

Verify that its source run belongs to publish.yml on the exact v1.0.3 commit.
Record both artifact hashes as WHEEL_HASH and SDIST_HASH. The local verifier uses
the tag's source, not an edited checkout. Set these environment variables:
GITHUB_REPOSITORY=lucalullo/neuro-tabular, GITHUB_SHA=S, SOURCE_RUN_ID=RUN,
EXPECTED_SOURCE_ATTEMPT=ATTEMPT, RELEASE_TARGET=pypi,
WHEEL_SHA256=WHEEL_HASH, SDIST_SHA256=SDIST_HASH.

~~~bash
python tools/release_artifacts.py testpypi --bundle .local/testpypi-bundle --output .local/testpypi-download
python -m venv .local/testpypi-install
~~~

The helper checks TestPyPI JSON and downloads the two actual published files.
Both downloaded bytes must match the original immutable bundle hashes and
source identity. Keep the JSON and downloaded files.

Install the downloaded wheel into that new environment, resolving dependencies
only from production PyPI. Use its interpreter for every following command:

~~~bash
python -m pip install --index-url https://pypi.org/simple --report resolver.json .local/testpypi-download/neurotabular-1.0.3-py3-none-any.whl
python -m pip check
python -m pip freeze --all
~~~

Replace python with .local/testpypi-install/bin/python on POSIX or its
Scripts/python.exe on Windows. Disable inherited pip extra-index/config settings
for this check; never mix TestPyPI with PyPI using --extra-index-url.
The wheel is the actual verified TestPyPI download; dependencies come from PyPI.

Confirm installed version/origin, then run the maintainer handoff's isolated
three-task smoke with binary/multiclass probabilities, regression outputs,
column reordering and pickle/joblib persistence. Keep the commands, resolver,
freeze, outputs, hashes and exit status. Scores from a smoke are not accuracy
or generalization evidence. This verification requires its own approval before
the final PyPI dispatch; do not infer completion from a successful upload.

## Promotion: same files, no rebuild

Only after TestPyPI verification and a separate maintainer approval:

~~~bash
gh workflow run publish.yml --repo lucalullo/neuro-tabular --ref v1.0.3 --field target=pypi --field confirmation=v1.0.3 --field source_run_id=RUN --field wheel_sha256=WHEEL_HASH --field sdist_sha256=SDIST_HASH
~~~

This path retrieves the immutable artifact from the successful TestPyPI run,
checks its workflow/ref/commit/attempt, rejects expired or ambiguous artifacts,
and verifies the approved hashes and actual TestPyPI downloads before publishing.
It never rebuilds wheel/sdist or falls back to another run/artifact. Both
registry file digests must remain identical. Retain the bundle before its
14-day GitHub retention expires; expiration is a stop condition, not permission
to rebuild or replace an existing release.

Finally verify PyPI JSON and actual file hashes, then a clean installation.
Review existing support/security/legal/naming notes for the final maintainer
go/no-go. No local check creates professional legal or third-party clearance.
