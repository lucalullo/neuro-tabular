"""Release guards fail closed before any registry publication."""

import copy
import hashlib
import io
import json

import pytest

from tools import release_artifacts as release

SHA = "a" * 40


def test_exact_tag_and_commit():
    release.check_tag("1.0.3", "tag", "v1.0.3", "v1.0.3", SHA, SHA)


@pytest.mark.parametrize(
    "values",
    [
        ("branch", "v1.0.3", "v1.0.3", SHA, SHA),
        ("tag", "v1.0.2", "v1.0.3", SHA, SHA),
        ("tag", "v1.0.3", "v1.0.2", SHA, SHA),
        ("tag", "v1.0.3", "v1.0.3", SHA, "b" * 40),
        ("tag", "v1.0.3", "v1.0.3", "bad", "bad"),
    ],
)
def test_invalid_tag_identity(values):
    with pytest.raises(ValueError):
        release.check_tag("1.0.3", *values)


def test_source_manifest(tmp_path):
    data = b"public source\n"
    (tmp_path / "a.py").write_bytes(data)
    (tmp_path / "SOURCE_MANIFEST.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "  a.py\n"
    )
    assert release.manifest(tmp_path) == {"a.py": release.digest(data)}
    (tmp_path / "a.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="Source hash"):
        release.manifest(tmp_path)


@pytest.mark.parametrize("name", ["../outside", "/absolute", ".local/x", "a\\b", "C:x"])
def test_source_manifest_rejects_unsafe_path(tmp_path, name):
    (tmp_path / "SOURCE_MANIFEST.sha256").write_text("0" * 64 + "  " + name + "\n")
    with pytest.raises(ValueError):
        release.manifest(tmp_path)


def ci_fixture():
    run = dict(
        head_sha=SHA,
        head_branch="release/1.0.3",
        event="push",
        path=".github/workflows/tests.yml",
        status="completed",
        conclusion="success",
        id=100,
        run_attempt=1,
    )
    jobs = [
        dict(name=n, status="completed", conclusion="success")
        for n in release.expected_jobs()
    ]
    return run, jobs


def test_exact_commit_ci():
    run, jobs = ci_fixture()
    api = lambda path: (  # noqa: E731
        {"total_count": len(jobs), "jobs": jobs}
        if "/jobs?" in path
        else {"workflow_runs": [run]}
    )
    assert release.successful_ci("owner/repo", SHA, "1.0.3", api) == 100


@pytest.mark.parametrize("change", ["sha", "branch", "missing", "failed", "wrong_name"])
def test_ci_must_cover_exact_commit_and_all_jobs(change):
    run, jobs = ci_fixture()
    if change == "sha":
        run["head_sha"] = "b" * 40
    elif change == "branch":
        run["head_branch"] = "main"
    elif change == "missing":
        jobs.pop()
    elif change == "failed":
        jobs[0]["conclusion"] = "failure"
    else:
        jobs[0]["name"] = "unrelated"

    def api(path):
        if "/jobs?" in path:
            return {"total_count": len(jobs), "jobs": jobs}
        return {"workflow_runs": [run]}

    with pytest.raises(ValueError):
        release.successful_ci("owner/repo", SHA, "1.0.3", api)


def publication_fixture():
    run = dict(
        head_sha=SHA,
        head_branch="v1.0.3",
        event="workflow_dispatch",
        path=".github/workflows/publish.yml",
        status="completed",
        conclusion="success",
        run_attempt=1,
    )
    jobs = [
        dict(name="Publish to TestPyPI", conclusion="success"),
        dict(name="Publish to PyPI", conclusion="skipped"),
    ]
    artifact = dict(
        id=99,
        name="release-distributions-123-1",
        expired=False,
        workflow_run={"head_sha": SHA},
    )
    return run, jobs, artifact


def source_api(run, jobs, artifacts):
    def api(path):
        if "/jobs?" in path:
            return {"total_count": len(jobs), "jobs": jobs}
        if "/artifacts?" in path:
            return {"total_count": len(artifacts), "artifacts": artifacts}
        return run

    return api


def test_resolve_original_testpypi_artifact():
    run, jobs, artifact = publication_fixture()
    assert release.source_run(
        "owner/repo", SHA, "1.0.3", "123", source_api(run, jobs, [artifact])
    ) == (artifact, 1)


@pytest.mark.parametrize(
    "change", ["sha", "branch", "workflow", "failed", "wrong_index", "expired", "extra"]
)
def test_source_artifact_provenance_rejected(change):
    run, jobs, artifact = publication_fixture()
    artifacts = [artifact]
    if change in {"sha", "branch", "workflow", "failed"}:
        field, value = {
            "sha": ("head_sha", "b" * 40),
            "branch": ("head_branch", "release/1.0.3"),
            "workflow": ("path", ".github/workflows/other.yml"),
            "failed": ("conclusion", "failure"),
        }[change]
        run[field] = value
    elif change == "wrong_index":
        jobs[0]["conclusion"] = "skipped"
        jobs[1]["conclusion"] = "success"
    elif change == "expired":
        artifact["expired"] = True
    else:
        artifacts.append(copy.deepcopy(artifact))
    with pytest.raises(ValueError):
        release.source_run(
            "owner/repo", SHA, "1.0.3", "123", source_api(run, jobs, artifacts)
        )


def test_only_two_distribution_files(tmp_path):
    for n in release.names("1.0.3"):
        (tmp_path / n).write_bytes(b"distribution")
    assert len(release.hashes(tmp_path, "1.0.3")) == 2
    (tmp_path / "extra.txt").write_bytes(b"extra")
    with pytest.raises(ValueError):
        release.hashes(tmp_path, "1.0.3")


@pytest.mark.parametrize("change", ["digest", "bytes", "host", "yanked", "extra"])
def test_actual_testpypi_download_checks(monkeypatch, change):
    payload = b"verified bytes"
    file_hash = release.digest(payload)
    names = release.names("1.0.3")
    locked = {"files": {n: file_hash for n in names}}
    items = [
        dict(
            filename=n,
            digests={"sha256": file_hash},
            url="https://test-files.pythonhosted.org/" + n,
            yanked=False,
        )
        for n in names
    ]
    if change == "digest":
        items[0]["digests"]["sha256"] = "0" * 64
    elif change == "bytes":
        payload = b"tampered bytes"
    elif change == "host":
        items[0]["url"] = "https://untrusted.example/file"
    elif change == "yanked":
        items[0]["yanked"] = True
    else:
        items.append(copy.deepcopy(items[0]))
    monkeypatch.setattr(release, "verify", lambda *a: locked)
    monkeypatch.setattr(release, "version", lambda: "1.0.3")

    def opener(url, **kwargs):
        if url.endswith("/json"):
            return io.BytesIO(
                json.dumps({"info": {"version": "1.0.3"}, "urls": items}).encode()
            )
        return io.BytesIO(payload)

    with pytest.raises(ValueError):
        release.testpypi(None, opener=opener)


@pytest.mark.parametrize(
    "first,second",
    [
        (
            "joblib>=1.4.2,!=1.5.0,!=1.5.1; python_version == '3.13'",
            'joblib!=1.5.1,>=1.4.2,!=1.5.0; python_version=="3.13"',
        ),
        ("Scikit_Learn>=1.3", "scikit-learn>=1.3"),
        ("build>=1.2; extra == 'dev'", 'build>=1.2; extra=="dev"'),
    ],
)
def test_equivalent_dependency_metadata(first, second):
    assert release.normalized_requirement(first) == release.normalized_requirement(
        second
    )


def test_changed_dependency_bounds_differ():
    assert release.normalized_requirement(
        "torch>=2.0"
    ) != release.normalized_requirement("torch>=3.0")
