"""The publication path retains exact-commit and artifact guards."""

import pytest

from tools import prepare_release as prep
from tools import release_artifacts as release

SHA = "a" * 40


def test_main_guard_requires_exact_ci():
    run = dict(
        id=100,
        head_sha=SHA,
        head_branch="main",
        event="push",
        path=".github/workflows/tests.yml",
        status="completed",
        conclusion="success",
        run_attempt=1,
    )
    jobs = [
        dict(name=n, status="completed", conclusion="success")
        for n in release.expected_jobs()
    ]

    def api(path):
        if "/jobs?" in path:
            return {"total_count": len(jobs), "jobs": jobs}
        assert "branch=main" in path
        return {"workflow_runs": [run]}

    assert prep.guard("owner/repo", SHA, "branch", "main", SHA, api) == 100
    jobs[0]["conclusion"] = "failure"
    with pytest.raises(ValueError):
        prep.guard("owner/repo", SHA, "branch", "main", SHA, api)


@pytest.mark.parametrize(
    "ref_type,ref_name,checkout",
    [("tag", "v2.1.0", SHA), ("branch", "dev", SHA), ("branch", "main", "b" * 40)],
)
def test_guard_rejects_wrong_checkout(ref_type, ref_name, checkout):
    with pytest.raises(ValueError, match="exact main commit"):
        prep.guard("owner/repo", SHA, ref_type, ref_name, checkout)


def preparation_fixture():
    run = dict(
        id=123,
        head_sha=SHA,
        head_branch="main",
        event="workflow_dispatch",
        path=prep.WORKFLOW,
        status="completed",
        conclusion="success",
        run_attempt=1,
    )
    jobs = [
        dict(name=name, status="completed", conclusion="success")
        for name in (prep.JOB, prep.PUBLISH_JOB)
    ]
    artifacts = [
        dict(
            id=99,
            name="testpypi-distributions-123-1",
            expired=False,
            workflow_run={"head_sha": SHA},
        )
    ]
    return run, jobs, artifacts


def preparation_api(run, jobs, artifacts, runs=None):
    def api(path):
        if "/jobs?" in path:
            return {"total_count": len(jobs), "jobs": jobs}
        if "/artifacts?" in path:
            return {"total_count": len(artifacts), "artifacts": artifacts}
        if "/workflows/" in path:
            choices = [run] if runs is None else runs
            return {"total_count": len(choices), "workflow_runs": choices}
        return run

    return api


@pytest.mark.parametrize("run_id", ["", "123"])
def test_resolve_preparation_artifact(run_id):
    api = preparation_api(*preparation_fixture())
    assert prep.preparation_source("owner/repo", SHA, run_id, api) == ("123", 1, 99)


@pytest.mark.parametrize(
    "change", ["commit", "branch", "workflow", "failure", "job", "expired", "artifact"]
)
def test_reject_unverified_preparation(change):
    run, jobs, artifacts = preparation_fixture()
    if change == "commit":
        run["head_sha"] = "b" * 40
    elif change == "branch":
        run["head_branch"] = "dev"
    elif change == "workflow":
        run["path"] = ".github/workflows/publish.yml"
    elif change == "failure":
        run["conclusion"] = "failure"
    elif change == "job":
        jobs[0]["conclusion"] = "failure"
    elif change == "expired":
        artifacts[0]["expired"] = True
    else:
        artifacts[0]["workflow_run"]["head_sha"] = "b" * 40
    with pytest.raises(ValueError):
        prep.preparation_source(
            "owner/repo", SHA, "123", preparation_api(run, jobs, artifacts)
        )


def test_multiple_runs_require_explicit_choice():
    run, jobs, artifacts = preparation_fixture()
    another = dict(run, id=124)
    api = preparation_api(run, jobs, artifacts, [run, another])
    with pytest.raises(ValueError, match="source_run_id"):
        prep.preparation_source("owner/repo", SHA, api=api)


@pytest.mark.parametrize("change", ["missing", "failed", "skipped", "duplicate"])
def test_reject_source_without_successful_testpypi_publication(change):
    run, jobs, artifacts = preparation_fixture()
    if change == "missing":
        jobs.pop()
    elif change == "duplicate":
        jobs[1]["name"] = prep.JOB
    else:
        jobs[1]["conclusion"] = "failure" if change == "failed" else "skipped"
    with pytest.raises(ValueError, match="publication must both succeed"):
        prep.preparation_source(
            "owner/repo", SHA, api=preparation_api(run, jobs, artifacts)
        )
