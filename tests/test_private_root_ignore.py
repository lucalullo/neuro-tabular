"""Private roots must be ignored without a maintainer-local Git exclude file."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name,ignored",
    [
        (".local/research/synthetic.txt", True),
        (".codex-context/checkpoint.json", True),
        ("src/neurotabular/classifier.py", False),
        ("docs/NEUROTABULAR_VISION.md", False),
    ],
)
def test_versioned_private_root_rules(tmp_path, name, ignored):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git-only source hygiene check: Git executable unavailable")
    root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    work.mkdir()
    (work / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    for key in ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"]:
        env.pop(key, None)
    subprocess.run([git, "init", str(work)], env=env, check=True, capture_output=True)
    probe = subprocess.run(
        [git, "check-ignore", "--no-index", name],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
    )
    assert probe.returncode in {0, 1}, probe.stderr
    assert (probe.returncode == 0) is ignored
