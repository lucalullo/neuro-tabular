"""Prepare distributions on main; never upload to a package index."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.parse
from pathlib import Path

from tools import release_artifacts as release

WORKFLOW = ".github/workflows/publish-testpypi.yml"
JOB = "Prepare TestPyPI distributions"


def guard(repo, sha, ref_type, ref_name, checkout, api=release.github):
    release.require(
        ref_type == "branch"
        and ref_name == "main"
        and bool(re.fullmatch("[0-9a-f]{40}", sha))
        and checkout == sha,
        "Preparation requires the exact main commit",
    )
    return release.successful_ci(repo, sha, release.version(), api, branch="main")


def matching_run(run, sha):
    return (
        run["head_sha"] == sha
        and run["head_branch"] == "main"
        and run["event"] == "workflow_dispatch"
        and run["path"] == WORKFLOW
        and run["status"] == "completed"
        and run["conclusion"] == "success"
    )


def preparation_source(repo, sha, run_id="", api=release.github):
    if run_id:
        release.require(
            bool(re.fullmatch("[1-9][0-9]*", str(run_id))), "Invalid source run ID"
        )
    else:
        query = urllib.parse.urlencode(
            {
                "head_sha": sha,
                "event": "workflow_dispatch",
                "branch": "main",
                "per_page": 100,
            }
        )
        data = api(f"repos/{repo}/actions/workflows/publish-testpypi.yml/runs?{query}")
        release.require(
            data["total_count"] == len(data["workflow_runs"]),
            "Incomplete preparation run list",
        )
        runs = [r for r in data["workflow_runs"] if matching_run(r, sha)]
        release.require(
            len(runs) == 1,
            "Need one successful TestPyPI preparation on this SHA; "
            "if multiple exist, supply its source_run_id",
        )
        run_id = str(runs[0]["id"])
    base = f"repos/{repo}/actions/runs/{run_id}"
    run = api(base)
    release.require(
        str(run["id"]) == str(run_id) and matching_run(run, sha),
        "Source preparation is not successful on this SHA",
    )
    attempt = run["run_attempt"]
    data = api(base + f"/attempts/{attempt}/jobs?per_page=100")
    jobs = data["jobs"]
    release.require(
        data["total_count"] == len(jobs) == 1
        and jobs[0]["name"] == JOB
        and jobs[0]["status"] == "completed"
        and jobs[0]["conclusion"] == "success",
        "Source preparation job did not succeed",
    )
    data = api(base + "/artifacts?per_page=100")
    release.require(
        data["total_count"] == len(data["artifacts"]), "Incomplete artifact list"
    )
    expected = f"testpypi-distributions-{run_id}-{attempt}"
    matches = [a for a in data["artifacts"] if a["name"] == expected]
    release.require(
        len(matches) == 1
        and not matches[0]["expired"]
        and matches[0]["workflow_run"]["head_sha"] == sha,
        "Missing, expired or unrelated TestPyPI preparation artifact",
    )
    return str(run_id), attempt, matches[0]["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["guard", "source"])
    args = parser.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    sha = os.environ["GITHUB_SHA"]
    if args.command == "guard":
        checkout = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        run_id = guard(
            repo,
            sha,
            os.environ["GITHUB_REF_TYPE"],
            os.environ["GITHUB_REF_NAME"],
            checkout,
        )
        print(json.dumps({"exact_main_CI_run": run_id, "source_commit": sha}))
    else:
        run_id, attempt, artifact_id = preparation_source(
            repo, sha, os.environ.get("PREPARATION_SOURCE_RUN_ID", "")
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf8") as f:
            f.write(
                f"run_id={run_id}\nsource_attempt={attempt}\n"
                f"artifact_id={artifact_id}\n"
            )
        print(json.dumps({"source_run_id": run_id, "artifact_id": artifact_id}))


if __name__ == "__main__":
    main()
