# Release preparation on main

The two manual preparation workflows use the same release sequence as the other
repositories. Both preserve the repository and require only read permissions; neither
uploads a distribution to TestPyPI or PyPI.

1. Push the final repository to `main` and wait for all 17 CI jobs to succeed on
   the exact commit.
2. In Actions, select **TestPyPI preparation - no upload**, choose `main`, and
   click **Run workflow**. This checks CI, source hashes, lint, tests, metadata,
   archive contents, and installation. Download the successful run's
   `testpypi-distributions-*` artifact and extract it.
3. Manually upload only its `dist/` wheel and sdist to TestPyPI, using your
   usual authorized publication method. Review the rendered project page and
   test the installed package. Do not change `main` or rebuild the artifacts.
4. In Actions, select **PyPI preparation - no upload**, choose the same `main`
   commit, and click **Run workflow**. If multiple successful TestPyPI
   preparation runs exist for that commit, supply the corresponding run ID.
   The job retrieves the original locked bundle and compares the hashes and
   actual downloaded bytes with the files on TestPyPI. A missing or different
   TestPyPI release causes a failure; no replacement build is performed.
5. Download the successful `pypi-distributions-*` artifact and, after approval,
   manually upload only its two `dist/` files to PyPI. Keep the manifest for your
   records. The published bytes must be identical to the TestPyPI files.
6. Create the GitHub version tag and release on the verified commit when ready.

No additional branch or tag is needed for these two preparation workflows. They
are distinct from the existing **Manual trusted publication** workflow, whose
original tag, release-branch CI, and immutable-promotion guards remain intact.
The latter does not accept preparation-only runs as publication receipts.

Preparation is not publication. If any check fails, resolve the issue before
proceeding. Do not commit the downloaded distribution files or credentials.
