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
        "root/.p37-local/results/x.json",
        "root/.p37-product/src/x.py",
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
    check_member("root/docs/API.md", b"Public API reference", sdist=True)
    with pytest.raises(ValueError, match="egg-info"):
        check_member("root/src/neurotabular.egg-info/local.txt", b"", sdist=True)
    with pytest.raises(ValueError, match="egg-info"):
        check_member("root/src/neurotabular.egg-info/PKG-INFO", b"", sdist=False)


def test_public_distributions_reject_models_receipts_and_local_archives():
    for name in (
        "tests/fixtures/public_0_3_1/binary.joblib",
        "tests/fixtures/public_0_3_1/PROVENANCE.json",
        "models/probe.pickle",
        "models/outputs.npz",
        "DELIVERY_RECEIPT.json",
        "docs/research/RESEARCH_STATE.json",
        ".p42-product/README.md",
        ".p99-local/result.txt",
    ):
        for prefix, sdist in (("root/", True), ("", False)):
            with pytest.raises(ValueError):
                check_member(prefix + name, b"synthetic payload", sdist=sdist)


@pytest.mark.parametrize("private_root", [".local", ".codex-context"])
@pytest.mark.parametrize("sdist", [True, False])
def test_rejects_private_workspace_text(private_root, sdist):
    prefix = "root/" if sdist else ""
    with pytest.raises(ValueError, match="Forbidden distribution member"):
        check_member(prefix + private_root + "/notes.md", b"private notes", sdist=sdist)
