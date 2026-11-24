"""Source-bound ingestion and grouped roles; no model training or selection."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path
from zipfile import ZipFile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


ALLOWED_HOSTS = {
    "archive.ics.uci.edu",
    "openml.org",
    "www.openml.org",
    "data.openml.org",
    "api.openml.org",
}


def check_source(raw, spec):
    expected = spec.get("md5")
    if expected and hashlib.md5(raw, usedforsecurity=False).hexdigest() != expected:
        raise ValueError("Source bytes differ from pinned OpenML metadata")
    if spec.get("sha256") and hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("Source bytes differ from frozen SHA256")


def fetch_source(spec, destination):
    """Explicit download, called only by the opt-in prepare command."""
    url = urllib.parse.urlparse(spec["url"])
    if url.scheme != "https" or url.hostname not in ALLOWED_HOSTS:
        raise ValueError("Unexpected source URL")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Preserve earlier source files; do not silently redownload")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        spec["url"], headers={"User-Agent": "NeuroTabular-public-benchmark"}
    )
    with (
        urllib.request.urlopen(request, timeout=30) as response,
        temporary.open("xb") as stream,
    ):
        resolved = urllib.parse.urlparse(response.geturl())
        if resolved.scheme != "https" or resolved.hostname not in ALLOWED_HOSTS:
            raise ValueError("Unexpected source redirect")
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > spec.get("max_source_bytes", 128 * 1024**2):
                raise ValueError("Source size cap")
            stream.write(chunk)
    raw = temporary.read_bytes()
    check_source(raw, spec)
    temporary.replace(destination)
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source": spec["source_identity"],
    }


def parse_source(path, spec):
    import numpy as np
    import pandas as pd

    if spec["format"] == "arff":
        from scipy.io import arff

        values, metadata = arff.loadarff(path)
        frame = pd.DataFrame(values)
        categorical = [
            name for name in metadata.names() if metadata[name][0] == "nominal"
        ]
        for name in categorical:
            frame[name] = frame[name].map(
                lambda value: (
                    value.decode("utf-8") if isinstance(value, bytes) else value
                )
            )
            frame[name] = frame[name].replace("?", np.nan)
    elif spec["format"] == "zip_csv":
        raw = Path(path).read_bytes()
        for member in spec["members"]:
            with ZipFile(io.BytesIO(raw)) as archive:
                if archive.getinfo(member).file_size > spec.get(
                    "max_uncompressed_bytes", 32 * 1024**2
                ):
                    raise ValueError("Archive member size cap")
                raw = archive.read(member)
        frame = pd.read_csv(
            io.BytesIO(raw),
            sep=spec.get("separator", ","),
            float_precision="round_trip",
            na_values=["?"],
        )
        categorical = list(spec.get("categorical", []))
    elif spec["format"] == "csv":
        frame = pd.read_csv(
            path,
            sep=spec.get("separator", ","),
            float_precision="round_trip",
            na_values=["?"],
        )
        categorical = list(spec.get("categorical", []))
    else:
        raise ValueError("Unsupported frozen source format")
    if len(frame) != spec["expected_rows"]:
        raise ValueError("Unexpected source row count")
    if spec.get("positional"):
        if frame.shape[1] != 21:
            raise ValueError("Climate source must have 21 columns")
        frame.columns = [
            "study_id",
            "run_id",
            *[f"parameter_{j:02d}" for j in range(18)],
            "outcome",
        ]
    target_name = spec["target"]
    if target_name not in frame:
        raise ValueError("Missing pinned target column")
    y = frame[target_name].copy()
    if y.isna().any():
        raise ValueError("Missing target: no outcome-dependent row removal allowed")
    excluded = [target_name, *spec.get("exclude", [])]
    if not set(excluded).issubset(frame.columns):
        raise ValueError("Missing pinned ID/group/co-target column")
    X = frame.drop(columns=excluded)
    categorical = [name for name in categorical if name in X]
    for name in X:
        if name in categorical:
            X[name] = X[name].astype(object)
        else:
            X[name] = pd.to_numeric(X[name], errors="raise").astype("float64")
            if np.isinf(X[name]).any():
                raise ValueError("Infinite source input")
    if X.shape[1] != spec["expected_features"]:
        raise ValueError("Unexpected feature schema")
    if spec["task"] != "regression":
        y = y.astype(str)
        if y.nunique() != spec["expected_classes"]:
            raise ValueError("Unexpected class support")
    else:
        y = pd.to_numeric(y, errors="raise").astype("float64")
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite target")
    return X, y, frame, categorical


def linked_groups(X, declared=None):
    """Union declared groups and exact input equality; no hash-only equivalence."""
    import numpy as np
    import pandas as pd

    parent = list(range(len(X)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        a, b = root(first), root(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    signatures = {}
    for row, values in enumerate(X.itertuples(index=False, name=None)):
        key = tuple(
            ("missing",) if pd.isna(value) else (type(value).__name__, value)
            for value in values
        )
        if key in signatures:
            union(row, signatures[key])
        else:
            signatures[key] = row
    if declared is not None:
        seen = {}
        for row, value in enumerate(declared):
            if pd.isna(value):
                raise ValueError("Missing group identifier")
            if value in seen:
                union(row, seen[value])
            else:
                seen[value] = row
    return np.asarray([root(row) for row in range(len(X))], dtype="int64")


def make_roles(X, y, groups, frame, spec, seed, official=None):
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    policy = spec["split_policy"]
    if official is not None:
        pool, test = official
        if not (set(groups[pool]) & set(groups[test])):
            if spec["task"] != "regression":
                local = StratifiedGroupKFold(
                    n_splits=5, shuffle=True, random_state=seed
                )
                fit, validation = next(
                    local.split(X.iloc[pool], y.iloc[pool], groups[pool])
                )
            else:
                local = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                fit, validation = next(local.split(X.iloc[pool], groups=groups[pool]))
            roles = {"fit": pool[fit], "validation": pool[validation], "test": test}
        else:
            # An unusable official split triggers the predeclared group fallback,
            # never a search among alternative official folds using test scores.
            return make_roles(X, y, groups, frame, spec, seed)
    elif policy == "ordered_contacts":
        first, second = int(0.6 * len(X)), int(0.8 * len(X))
        roles = {
            "fit": np.arange(first),
            "validation": np.arange(first, second),
            "test": np.arange(second, len(X)),
        }
    elif policy == "climate_studies":
        study = np.asarray(frame["study_id"])
        roles = {
            "fit": np.flatnonzero(study == 1),
            "validation": np.flatnonzero(study == 2),
            "test": np.flatnonzero(study == 3),
        }
    elif policy == "gas_years":
        years = np.asarray(frame["year"])
        roles = {
            "fit": np.flatnonzero(np.isin(years, [2011, 2012])),
            "validation": np.flatnonzero(years == 2013),
            "test": np.flatnonzero(np.isin(years, [2014, 2015])),
        }
    elif spec["task"] != "regression":
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = [held for _, held in splitter.split(X, y, groups)]
        roles = {
            "fit": np.concatenate(folds[2:]),
            "validation": folds[1],
            "test": folds[0],
        }
    else:
        first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        pool, test = next(first.split(X, groups=groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        fit, validation = next(second.split(X.iloc[pool], groups=groups[pool]))
        roles = {"fit": pool[fit], "validation": pool[validation], "test": test}
    # Temporal/source-defined roles remain fixed. Purge earlier duplicate/group
    # exposure; never move future records into training or inspect their labels.
    purged = []
    if policy in {"climate_studies", "gas_years", "ordered_contacts"}:
        later = set()
        for name in ("test", "validation", "fit"):
            indices = roles[name]
            keep = np.asarray([groups[row] not in later for row in indices])
            purged.extend(indices[~keep].tolist())
            roles[name] = indices[keep]
            later.update(groups[roles[name]].tolist())
    for name, indices in roles.items():
        if not len(indices):
            raise ValueError("Empty statistical role")
        if spec["task"] != "regression" and set(y.iloc[indices]) != set(y):
            raise ValueError("A frozen split lacks class support; do not redraw it")
        for other, opposite in roles.items():
            if name != other and set(groups[indices]) & set(groups[opposite]):
                raise ValueError("Statistical role leakage")
    roles["pool"] = np.concatenate([roles["fit"], roles["validation"]])
    return roles, purged


def write_bundle(path, X, y, groups, indices, categorical, units=None):
    import numpy as np
    import pandas as pd

    arrays = {
        "columns": np.asarray(X.columns, dtype="str"),
        "categorical": np.asarray(categorical, dtype="str"),
        "y": np.asarray(y.iloc[indices]),
        "row_ids": np.asarray(indices),
        "groups": np.asarray(groups[indices]),
        "units": np.asarray(groups[indices] if units is None else units[indices]),
    }
    if arrays["y"].dtype.kind == "O":
        arrays["y"] = arrays["y"].astype("str")
    for index, name in enumerate(X):
        series = X[name].iloc[indices]
        if name in categorical:
            mask = pd.isna(series).to_numpy()
            if any(not isinstance(value, str) for value in series[~mask]):
                raise ValueError(
                    "Bundle schema requires explicit string categories; no silent type collapse"
                )
            arrays[f"x{index}"] = np.asarray(
                ["" if missing else str(value) for value, missing in zip(series, mask)],
                dtype="str",
            )
            arrays[f"missing{index}"] = mask
        else:
            arrays[f"x{index}"] = series.to_numpy(dtype="float64")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def read_bundle(path, include_target=True):
    import numpy as np
    import pandas as pd

    with np.load(path, allow_pickle=False) as arrays:
        categorical = arrays["categorical"].tolist()
        columns = arrays["columns"].tolist()
        values = {}
        for index, name in enumerate(columns):
            column = arrays[f"x{index}"].copy()
            if name in categorical:
                column = column.astype(object)
                column[arrays[f"missing{index}"]] = np.nan
            values[name] = column
        target = arrays["y"].copy() if include_target else None
        return (
            pd.DataFrame(values, columns=columns),
            target,
            arrays["groups"].copy(),
            categorical,
        )


def prepare_family(spec, output, seeds, source_file=None, official_split_file=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Do not overwrite prepared source/roles")
    output.mkdir(parents=True)
    source = output / ("source." + spec["format"])
    if source_file is None:
        receipt = fetch_source(spec, source)
    else:
        raw = Path(source_file).read_bytes()
        check_source(raw, spec)
        with source.open("xb") as stream:
            stream.write(raw)
        receipt = {
            "bytes": len(raw),
            "sha256": digest(source),
            "source": spec["source_identity"],
        }
    X, y, original, categorical = parse_source(source, spec)
    declared = (
        original[spec["group_column"]].to_numpy() if spec.get("group_column") else None
    )
    groups = linked_groups(X, declared)
    import numpy as np

    units = groups.copy()
    if spec["split_policy"] == "ordered_contacts":
        units = np.arange(len(X)) // 100
    elif spec["split_policy"] == "gas_years":
        for year in sorted(original["year"].unique()):
            indices = np.flatnonzero(np.asarray(original["year"]) == year)
            units[indices] = int(year) * 100000 + np.arange(len(indices)) // 168
    elif spec["split_policy"] == "climate_studies":
        units = original["study_id"].to_numpy(dtype="int64")
    official_splits = None
    if spec.get("official_splits_url"):
        import numpy as np
        import pandas as pd
        from scipy.io import arff

        split_file = output / "official_splits.arff"
        split_spec = {
            "url": spec["official_splits_url"],
            "source_identity": spec["source_identity"] + ":splits",
            "sha256": spec.get("official_splits_sha256"),
        }
        if official_split_file is None:
            fetch_source(split_spec, split_file)
        else:
            raw = Path(official_split_file).read_bytes()
            check_source(raw, split_spec)
            split_file.write_bytes(raw)
        values, _ = arff.loadarff(split_file)
        official_splits = pd.DataFrame(values)
        if "sample" not in official_splits:
            official_splits["sample"] = 0
        choices = sorted(
            set(
                zip(
                    official_splits["repeat"],
                    official_splits["fold"],
                    official_splits["sample"],
                )
            )
        )
        if len(choices) < len(seeds):
            raise ValueError("Insufficient frozen official splits")
    splits = {}
    for split_index, seed in enumerate(seeds):
        official = None
        official_info = None
        if official_splits is not None:
            repeat, fold, sample = choices[split_index]
            chosen = official_splits[
                (official_splits["repeat"] == repeat)
                & (official_splits["fold"] == fold)
                & (official_splits["sample"] == sample)
            ]
            types = (
                chosen["type"]
                .map(
                    lambda value: value.decode() if isinstance(value, bytes) else value
                )
                .str.upper()
            )
            train_rows = chosen.loc[types == "TRAIN", "rowid"].to_numpy(dtype=int)
            test_rows = chosen.loc[types == "TEST", "rowid"].to_numpy(dtype=int)
            if set(train_rows) & set(test_rows) or set(train_rows) | set(
                test_rows
            ) != set(range(len(X))):
                raise ValueError("Malformed official row roles")
            compatible = not bool(set(groups[train_rows]) & set(groups[test_rows]))
            official_info = {
                "repeat": int(repeat),
                "fold": int(fold),
                "sample": int(sample),
                "group_compatible": compatible,
                "decision": "used"
                if compatible
                else "predeclared_group_fallback_due_to_overlap",
            }
            official = train_rows, test_rows
        roles, purged = make_roles(X, y, groups, original, spec, seed, official)
        folder = output / str(seed)
        folder.mkdir()
        hashes = {}
        for name, indices in roles.items():
            path = folder / (name + ".npz")
            write_bundle(path, X, y, groups, indices, categorical, units)
            hashes[name] = {"sha256": digest(path), "rows": len(indices)}
        splits[str(seed)] = {
            "roles": hashes,
            "purged_earlier_duplicates": purged,
            "official": official_info,
        }
    record = {
        "schema": "NT_PUBLIC_SOURCE/1",
        "family_id": spec["id"],
        "source": receipt,
        "feature_columns": list(X),
        "categorical": categorical,
        "splits": splits,
        "statistical_unit": spec["statistical_unit"],
        "predictive_models_used": 0,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record
