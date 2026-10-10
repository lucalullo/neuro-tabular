"""Dormant future-deprecation infrastructure, not current API deprecations."""

import inspect
import warnings

import pytest

from neurotabular._warnings import _warn_deprecated


def test_canonical_future_warning_and_caller_location():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        _warn_deprecated(
            "example.old", since="0.3.1", remove_in="0.5.0", replacement="example.new"
        )
    assert len(caught) == 1
    assert caught[0].category is FutureWarning
    assert (
        str(caught[0].message)
        == "NeuroTabular: example.old is deprecated since 0.3.1; removal no earlier than 0.5.0. Use example.new instead."
    )
    assert caught[0].filename == __file__ and caught[0].lineno == line


@pytest.mark.parametrize("remove_in", ["0.3.2", "0.4.0", "0.2.0"])
def test_notice_cannot_be_a_patch_or_immediately_next_minor(remove_in):
    with pytest.raises(ValueError, match="complete minor"):
        _warn_deprecated("old", since="0.3.1", remove_in=remove_in, replacement="new")


@pytest.mark.parametrize(
    "field,value",
    [("since", "next"), ("remove_in", "0.5"), ("replacement", ""), ("feature", "")],
)
def test_malformed_deprecation_metadata(field, value):
    args = dict(feature="old", since="0.3.1", remove_in="0.5.0", replacement="new")
    args[field] = value
    with pytest.raises(ValueError):
        _warn_deprecated(**args)


def test_default_warning_filter_deduplicates_a_call_site():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(3):
            _warn_deprecated("old", since="1.2.0", remove_in="1.4.0", replacement="new")
    assert len(caught) == 1
