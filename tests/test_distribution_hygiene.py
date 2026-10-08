"""Product distribution boundaries, exercised without installing packages."""

import pytest

from tools.check_distribution import check_member


@pytest.mark.parametrize(
    "name",
    [
        "neurotabular/__pycache__/a.pyc",
        "root/research/raw/data.json",
        "root/.p35-local/results/x.json",
        "root/.p36-local/results/x.json",
        "root/.p36-product/src/x.py",
        "root/build/lib/x.py",
        "root/../escape.py",
        "root/models/frozen.npz",
        "root/src/other.egg-info/PKG-INFO",
    ],
)
def test_rejects_accidental_distribution_artifacts(name):
    with pytest.raises(ValueError):
        check_member(name, b"", sdist=True)


@pytest.mark.parametrize(
    "content",
    [
        b"C:" + b"\\Users\\someone\\project",
        b"/" + b"home" + b"/someone/project",
        b"/mnt/c/" + b"Users/someone/project",
    ],
)
def test_rejects_personal_paths_in_packaged_text(content):
    with pytest.raises(ValueError, match="Personal filesystem path"):
        check_member("root/docs/report.md", content, sdist=True)


def test_allows_only_intentional_source_metadata_and_compact_docs():
    check_member(
        "root/src/neurotabular.egg-info/PKG-INFO", b"Name: neurotabular", sdist=True
    )
    check_member(
        "root/docs/research/RESEARCH_MASTER.md", b"Compact frozen reference", sdist=True
    )
    with pytest.raises(ValueError, match="egg-info"):
        check_member("root/src/neurotabular.egg-info/local.txt", b"", sdist=True)
    with pytest.raises(ValueError, match="egg-info"):
        check_member("root/src/neurotabular.egg-info/PKG-INFO", b"", sdist=False)
