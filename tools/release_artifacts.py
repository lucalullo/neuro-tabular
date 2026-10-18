"""Verify release identity and promote immutable distributions; never publish."""

from __future__ import annotations

import argparse
import base64
import csv
import email.parser
import hashlib
import io
import json
import os
import re
import runpy
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def version():
    import tomllib

    return tomllib.loads((ROOT / "pyproject.toml").read_text("utf8"))["project"][
        "version"
    ]


def manifest(root=ROOT):
    entries = {}
    for line in (root / "SOURCE_MANIFEST.sha256").read_text("utf8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, "Invalid source manifest line")
        expected, name = fields
        path = PurePosixPath(name)
        require(
            re.fullmatch("[0-9a-f]{64}", expected)
            and not path.is_absolute()
            and ".." not in path.parts
            and "\\" not in name
            and ":" not in name
            and name not in entries
            and not any(p in {".git", ".local", ".codex-context"} for p in path.parts)
            and (root / name).resolve().is_relative_to(root.resolve()),
            "Unsafe or duplicate source manifest path",
        )
        require(digest((root / name).read_bytes()) == expected, f"Source hash: {name}")
        entries[name] = expected
    require(entries and list(entries) == sorted(entries), "Manifest must be sorted")
    return entries


def check_tag(v, ref_type, ref_name, confirmation, sha, checkout):
    require(
        ref_type == "tag"
        and ref_name == confirmation == "v" + v
        and re.fullmatch("[0-9a-f]{40}", sha or "")
        and sha == checkout,
        "Exact version tag, confirmation and commit required",
    )


def github(path):
    token = os.environ.get("GH_TOKEN")
    require(bool(token), "Read-only GitHub token missing")
    request = urllib.request.Request(
        "https://api.github.com/" + path,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def expected_jobs():
    return {
        "Ruff",
        "Build distributions",
        "PyTorch 2.0 compatibility",
        "macOS Python 3.14 (new platform evidence)",
        *("Python " + p for p in ("3.10", "3.11", "3.12", "3.13", "3.14")),
        *("Windows Python " + p for p in ("3.11", "3.12", "3.13", "3.14")),
        *(
            "Lowest binary dependencies Python " + p
            for p in ("3.11", "3.12", "3.13", "3.14")
        ),
    }


def successful_ci(repo, sha, v, api=github):
    query = urllib.parse.urlencode(
        {"head_sha": sha, "event": "push", "branch": "release/" + v, "per_page": 100}
    )
    runs = api(f"repos/{repo}/actions/workflows/tests.yml/runs?{query}")
    for run in runs["workflow_runs"]:
        if (
            run["head_sha"] == sha
            and run["head_branch"] == "release/" + v
            and run["event"] == "push"
            and run["path"] == ".github/workflows/tests.yml"
            and run["status"] == "completed"
            and run["conclusion"] == "success"
        ):
            data = api(
                f"repos/{repo}/actions/runs/{run['id']}"
                f"/attempts/{run['run_attempt']}/jobs?per_page=100"
            )
            jobs = data["jobs"]
            if (
                data["total_count"] == len(jobs) == 17
                and {j["name"] for j in jobs} == expected_jobs()
                and all(
                    j["status"] == "completed" and j["conclusion"] == "success"
                    for j in jobs
                )
            ):
                return run["id"]
    raise ValueError("No successful exact-commit 17-job stable branch CI")


def source_run(repo, sha, v, run_id, api=github):
    require(re.fullmatch("[1-9][0-9]*", str(run_id)), "Invalid source run ID")
    base = f"repos/{repo}/actions/runs/{run_id}"
    run = api(base)
    require(
        run["head_sha"] == sha
        and run["head_branch"] == "v" + v
        and run["event"] == "workflow_dispatch"
        and run["path"] == ".github/workflows/publish.yml"
        and run["status"] == "completed"
        and run["conclusion"] == "success",
        "Source must be a successful publication run on this exact version tag",
    )
    jobs = api(base + f"/attempts/{run['run_attempt']}/jobs?per_page=100")
    require(jobs["total_count"] == len(jobs["jobs"]), "Incomplete source job list")
    by_name = {j["name"]: j for j in jobs["jobs"]}
    require(
        by_name.get("Publish to TestPyPI", {}).get("conclusion") == "success"
        and by_name.get("Publish to PyPI", {}).get("conclusion") == "skipped",
        "Source run did not publish only to TestPyPI",
    )
    artifacts = api(base + "/artifacts?per_page=100")
    require(
        artifacts["total_count"] == len(artifacts["artifacts"]),
        "Incomplete artifact list",
    )
    name = f"release-distributions-{run_id}-{run['run_attempt']}"
    matches = [a for a in artifacts["artifacts"] if a["name"] == name]
    require(
        len(matches) == 1
        and not matches[0]["expired"]
        and matches[0]["workflow_run"]["head_sha"] == sha,
        "Missing, ambiguous, expired or unrelated immutable release artifact",
    )
    return matches[0], run["run_attempt"]


def names(v):
    return (f"neurotabular-{v}-py3-none-any.whl", f"neurotabular-{v}.tar.gz")


def hashes(directory, v):
    expected = set(names(v))
    require(
        {p.name for p in directory.iterdir()} == expected,
        "Distribution directory must contain exactly wheel and sdist",
    )
    require(
        all(
            (directory / n).is_file() and not (directory / n).is_symlink()
            for n in expected
        ),
        "Not regular files",
    )
    return {n: digest((directory / n).read_bytes()) for n in sorted(expected)}


def normalized_requirement(value):
    compact = re.sub(r"\s+", "", value).replace("'", '"')
    requirement, separator, marker = compact.partition(";")
    name = re.match(r"[A-Za-z0-9_.-]+", requirement)
    require(name is not None, "Invalid dependency declaration")
    package = re.sub(r"[-_.]+", "-", name.group()).lower()
    specifiers = tuple(sorted(requirement[name.end() :].split(",")))
    return package, specifiers, marker if separator else ""


def validate_metadata(data, v, root):
    import tomllib

    project = tomllib.loads((root / "pyproject.toml").read_text("utf8"))["project"]
    meta = email.parser.BytesParser().parsebytes(data)
    require(
        meta["Name"] == project["name"] == "neurotabular"
        and meta["Version"] == v
        and meta["Requires-Python"] == project["requires-python"],
        "Package identity or Python bounds",
    )
    declared = list(project["dependencies"])
    for group, dependencies in project.get("optional-dependencies", {}).items():
        declared.extend(d + "; extra == '" + group + "'" for d in dependencies)
    require(
        sorted(map(normalized_requirement, meta.get_all("Requires-Dist", [])))
        == sorted(map(normalized_requirement, declared)),
        "Package dependency bounds",
    )


def validate_archives(directory, v, root=ROOT):
    entries = manifest(root)
    hygiene = runpy.run_path(str(root / "tools/check_distribution.py"))["check_archive"]
    for name in names(v):
        hygiene(directory / name)
    with zipfile.ZipFile(directory / names(v)[0]) as z:
        require(z.testzip() is None, "Wheel CRC")
        regular = [i for i in z.infolist() if not i.is_dir()]
        require(
            len({i.filename for i in regular}) == len(regular),
            "Duplicate wheel members",
        )
        files = {i.filename: z.read(i) for i in regular}
        prefix = f"neurotabular-{v}.dist-info/"
        validate_metadata(files[prefix + "METADATA"], v, root)
        runtime = {
            n[4:]: h for n, h in entries.items() if n.startswith("src/neurotabular/")
        }
        require(
            {n for n in files if n.startswith("neurotabular/")} == set(runtime),
            "Unexpected runtime members",
        )
        require(
            set(files)
            == set(runtime)
            | {
                prefix + n
                for n in (
                    "METADATA",
                    "WHEEL",
                    "RECORD",
                    "top_level.txt",
                    "licenses/LICENSE",
                )
            },
            "Unexpected wheel file list",
        )
        for name, expected in runtime.items():
            require(digest(files[name]) == expected, f"Wheel source hash: {name}")
        for name, expected, size in csv.reader(
            io.StringIO(files[prefix + "RECORD"].decode())
        ):
            if expected:
                actual = (
                    base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest())
                    .rstrip(b"=")
                    .decode()
                )
                require(
                    expected == "sha256=" + actual and int(size) == len(files[name]),
                    "Wheel RECORD",
                )
    with tarfile.open(directory / names(v)[1]) as t:
        members = t.getmembers()
        require(not any(m.issym() or m.islnk() for m in members), "Sdist link")
        regular = [m for m in members if m.isfile()]
        require(
            len({m.name for m in regular}) == len(regular), "Duplicate sdist members"
        )
        files = {
            m.name.split("/", 1)[1]: t.extractfile(m).read()
            for m in t.getmembers()
            if m.isfile()
        }
        metadata = {
            "PKG-INFO",
            "setup.cfg",
            *(
                "src/neurotabular.egg-info/" + n
                for n in (
                    "PKG-INFO",
                    "SOURCES.txt",
                    "dependency_links.txt",
                    "requires.txt",
                    "top_level.txt",
                )
            ),
        }
        require(
            set(files) == set(entries) | {"SOURCE_MANIFEST.sha256"} | metadata,
            "Unexpected sdist member list",
        )
        require(
            files["SOURCE_MANIFEST.sha256"]
            == (root / "SOURCE_MANIFEST.sha256").read_bytes(),
            "Sdist source manifest identity",
        )
        for name, expected in entries.items():
            require(digest(files[name]) == expected, f"Sdist source hash: {name}")
        validate_metadata(files["PKG-INFO"], v, root)


def identity():
    return {
        "project": "neurotabular",
        "version": version(),
        "commit": os.environ["GITHUB_SHA"],
        "repository": os.environ["GITHUB_REPOSITORY"],
        "source_manifest_sha256": digest(
            (ROOT / "SOURCE_MANIFEST.sha256").read_bytes()
        ),
    }


def lock(directory, bundle):
    v = version()
    validate_archives(directory, v)
    values = hashes(directory, v)
    bundle.mkdir()
    (bundle / "dist").mkdir()
    for name in values:
        shutil.copyfile(directory / name, bundle / "dist" / name)
    data = {
        **identity(),
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "files": values,
    }
    (bundle / "manifest.json").write_text(json.dumps(data, indent=2) + "\n", "utf8")
    return data


def verify(bundle, expected_run=None):
    require(
        {p.name for p in bundle.iterdir()} == {"dist", "manifest.json"},
        "Unexpected release bundle members",
    )
    data = json.loads((bundle / "manifest.json").read_bytes())
    require(
        set(data) == set(identity()) | {"run_id", "run_attempt", "files"},
        "Unexpected release manifest fields",
    )
    require(all(data[k] == v for k, v in identity().items()), "Bundle identity")
    if expected_run is not None:
        require(data["run_id"] == int(expected_run), "Bundle source run")
    expected_attempt = os.environ.get("EXPECTED_SOURCE_ATTEMPT")
    if expected_attempt:
        require(data["run_attempt"] == int(expected_attempt), "Bundle source attempt")
    require(data["files"] == hashes(bundle / "dist", version()), "Bundle file hashes")
    for name, key in zip(names(version()), ("WHEEL_SHA256", "SDIST_SHA256")):
        expected = os.environ.get(key)
        if os.environ.get("RELEASE_TARGET") == "pypi":
            require(
                bool(re.fullmatch("[0-9a-f]{64}", expected or "")),
                "Expected maintainer-approved wheel/sdist SHA256 required",
            )
        if expected:
            require(data["files"][name] == expected, "Approved artifact hash mismatch")
    validate_archives(bundle / "dist", version())
    return data


def testpypi(bundle, output=None, opener=urllib.request.urlopen):
    data = verify(bundle, os.environ.get("SOURCE_RUN_ID"))
    url = f"https://test.pypi.org/pypi/neurotabular/{version()}/json"
    with opener(url, timeout=60) as response:
        registry = json.load(response)
    require(registry["info"]["version"] == version(), "TestPyPI version")
    require(
        len(registry["urls"]) == 2
        and {x["filename"] for x in registry["urls"]} == set(data["files"]),
        "TestPyPI file list",
    )
    if output:
        output.mkdir()
        (output / "registry.json").write_text(json.dumps(registry, indent=2), "utf8")
    for item in registry["urls"]:
        name = item["filename"]
        require(
            not item.get("yanked") and item["digests"]["sha256"] == data["files"][name],
            "TestPyPI recorded hash or yank status",
        )
        parsed = urllib.parse.urlsplit(item["url"])
        require(
            parsed.scheme == "https"
            and parsed.hostname
            in {"test-files.pythonhosted.org", "files.pythonhosted.org"}
            and parsed.port in {None, 443},
            "Unexpected TestPyPI file host",
        )
        with opener(item["url"], timeout=60) as response:
            payload = response.read(128 * 1024 * 1024 + 1)
        require(
            len(payload) <= 128 * 1024 * 1024
            and digest(payload) == data["files"][name],
            "Actual TestPyPI download hash",
        )
        if output:
            with (output / name).open("xb") as f:
                f.write(payload)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["source", "guard", "source-run", "lock", "verify", "testpypi"],
    )
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    parser.add_argument("--bundle", type=Path, default=Path("release-bundle"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "source":
        print(json.dumps({"source_manifest": "PASS", "entries": len(manifest())}))
        return
    if args.command == "guard":
        checkout = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        check_tag(
            version(),
            os.environ["GITHUB_REF_TYPE"],
            os.environ["GITHUB_REF_NAME"],
            os.environ["CONFIRMATION"],
            os.environ["GITHUB_SHA"],
            checkout,
        )
        target = os.environ["RELEASE_TARGET"]
        require(target in {"testpypi", "pypi"}, "Unknown publication target")
        if target == "testpypi":
            require(
                not os.environ.get("SOURCE_RUN_ID"), "No source run for first build"
            )
        result = successful_ci(os.environ["GITHUB_REPOSITORY"], checkout, version())
        print(json.dumps({"exact_commit_CI_run": result, "tag": "v" + version()}))
    elif args.command == "source-run":
        for key in ("WHEEL_SHA256", "SDIST_SHA256"):
            require(
                bool(re.fullmatch("[0-9a-f]{64}", os.environ.get(key, ""))),
                "Expected maintainer-approved artifact hashes required",
            )
        artifact, attempt = source_run(
            os.environ["GITHUB_REPOSITORY"],
            os.environ["GITHUB_SHA"],
            version(),
            os.environ["SOURCE_RUN_ID"],
        )
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf8") as f:
            f.write(f"artifact_id={artifact['id']}\nsource_attempt={attempt}\n")
        print(json.dumps({"source_artifact_id": artifact["id"], "attempt": attempt}))
    elif args.command == "lock":
        print(json.dumps(lock(args.directory, args.bundle), indent=2))
    elif args.command == "verify":
        print(
            json.dumps(verify(args.bundle, os.environ.get("SOURCE_RUN_ID")), indent=2)
        )
    elif args.command == "testpypi":
        print(json.dumps(testpypi(args.bundle, args.output), indent=2))


if __name__ == "__main__":
    main()
