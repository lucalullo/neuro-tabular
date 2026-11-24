"""Verify pure NeuroTabular wheels and stage two isolated import roots, no fits."""

from __future__ import annotations

import argparse
import json
import stat
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from .control import atomic_json
from .data import digest

HERE = Path(__file__).resolve().parent


def verified_wheel(path, expected_version, pins):
    import hashlib

    with ZipFile(path) as archive:
        if (
            len(archive.infolist()) > 100
            or sum(item.file_size for item in archive.infolist()) > 64 * 1024**2
        ):
            raise ValueError("Wheel total size/member cap")
        contents = {}
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.filename
                or ":" in member.filename
            ):
                raise ValueError("Unsafe wheel path")
            if (
                stat.S_ISLNK(member.external_attr >> 16)
                or member.file_size > 64 * 1024**2
            ):
                raise ValueError("Linked or oversized wheel member")
            if not member.is_dir():
                if member.filename.casefold() in {name.casefold() for name in contents}:
                    raise ValueError("Duplicate wheel member")
                contents[member.filename] = archive.read(member)
    metadata = [
        raw for name, raw in contents.items() if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata) != 1:
        raise ValueError("Unexpected wheel metadata")
    meta = BytesParser().parsebytes(metadata[0])
    if meta["Name"] != "neurotabular" or meta["Version"] != expected_version:
        raise ValueError("Wrong wheel identity/version")
    runtime = {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in contents.items()
        if name.startswith("neurotabular/")
    }
    if runtime != pins:
        raise ValueError("Wheel runtime differs from the frozen benchmark")
    if any(
        not (
            name.startswith("neurotabular/")
            or name.startswith("neurotabular-" + expected_version + ".dist-info/")
        )
        for name in contents
    ):
        raise ValueError("Unexpected wheel installation path")
    return contents


def stage_wheel(path, destination, version, pins):
    contents = verified_wheel(path, version, pins)
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Do not overwrite an existing import root")
    destination.mkdir(parents=True)
    for name, raw in contents.items():
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Wheel extraction escaped its root")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nt200", type=Path, required=True)
    parser.add_argument("--nt210", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    pins = json.loads((HERE / "runtimes.json").read_text(encoding="utf-8"))
    for version, path in (("2.0.0", args.nt200), ("2.1.0", args.nt210)):
        verified_wheel(path, version, pins[version])
    if not args.execute:
        print(
            "Wheel runtimes verified. Add --execute to stage them outside the repository."
        )
        return
    if args.stage.resolve().is_relative_to(HERE.parents[1].resolve()):
        raise ValueError("Keep environment files outside the public repository")
    environment = {
        "python": str(args.python.resolve()),
        "nt_import_roots": {},
        "wheel_sha256": {},
    }
    for version, path in (("2.0.0", args.nt200), ("2.1.0", args.nt210)):
        key = "neurotabular" + version.replace(".", "")
        root = args.stage / key
        stage_wheel(path, root, version, pins[version])
        environment["nt_import_roots"][key] = str(root.resolve())
        environment["wheel_sha256"][key] = digest(path)
    atomic_json(args.output, environment)


if __name__ == "__main__":
    main()
