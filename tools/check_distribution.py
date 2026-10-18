"""Check wheel/sdist contents without importing NeuroTabular or dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN = {
    ".local",
    ".codex-context",
    ".p35-local",
    ".p36-local",
    ".p35-product",
    ".p36-product",
    ".p37-local",
    ".p37-product",
    "product_validation",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
    "build",
    "dist",
    ".cache",
}
SDIST_METADATA = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "requires.txt",
    "top_level.txt",
}
PERSONAL_PATH = re.compile(
    rb"(?i)(?:[a-z]:[/\\]+Users[/\\]+|/(?:home|Users)/[a-z0-9_.-]+/|/mnt/[a-z]/Users/)"
)
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".cff",
    ".sha256",
}


def check_member(name, data, *, sdist=False):
    path = PurePosixPath(name.replace("\\", "/"))
    parts = path.parts
    if (
        path.is_absolute()
        or ".." in parts
        or any(
            p in FORBIDDEN or re.fullmatch(r"\.p[0-9]+-(?:local|product)", p)
            for p in parts
        )
    ):
        raise ValueError(f"Forbidden distribution member: {name}")
    relative = parts[1:] if sdist else parts
    if "research" in relative and (relative[0] == "research" or "raw" in relative):
        raise ValueError(f"Raw research member: {name}")
    if relative[:2] == ("docs", "research") or relative[:2] == ("tests", "fixtures"):
        raise ValueError(f"Internal archive/fixture member: {name}")
    if path.suffix == ".json" and re.search(
        r"(?i)provenance|receipt|validation", path.name
    ):
        raise ValueError(f"Internal receipt member: {name}")
    if path.suffix in {
        ".pyc",
        ".pyo",
        ".pkl",
        ".pickle",
        ".joblib",
        ".npz",
        ".pt",
        ".pth",
    }:
        raise ValueError(f"Local model/runtime artifact: {name}")
    if any(p.endswith(".egg-info") for p in parts):
        allowed = (
            sdist
            and len(parts) == 4
            and parts[1:3] == ("src", "neurotabular.egg-info")
            and parts[-1] in SDIST_METADATA
        )
        if not allowed:
            raise ValueError(f"Unexpected egg-info member: {name}")
    if path.suffix in TEXT_SUFFIXES or path.name in {
        "PKG-INFO",
        "METADATA",
        "SOURCES.txt",
    }:
        if PERSONAL_PATH.search(data):
            raise ValueError(f"Personal filesystem path in distribution member: {name}")


def check_archive(artifact):
    artifact = Path(artifact)
    members = []
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    check_member(info.filename, archive.read(info))
                    members.append(info.filename)
    elif artifact.name.endswith(".tar.gz"):
        with tarfile.open(artifact, "r:gz") as archive:
            for info in archive.getmembers():
                if info.issym() or info.islnk():
                    raise ValueError(f"Unexpected distribution link: {info.name}")
                if info.isfile():
                    extracted = archive.extractfile(info)
                    assert extracted is not None
                    check_member(info.name, extracted.read(), sdist=True)
                    members.append(info.name)
    else:
        raise ValueError(f"Unsupported distribution type: {artifact.name}")
    assert any(name.endswith("neurotabular/__init__.py") for name in members)
    return {
        "artifact": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "bytes": artifact.stat().st_size,
        "members": len(members),
        "status": "PASS",
        "metadata_exception": (
            "Only intentional standard setuptools source egg-info; "
            "wheel dist-info is required metadata"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = [check_archive(path) for path in args.artifacts]
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
