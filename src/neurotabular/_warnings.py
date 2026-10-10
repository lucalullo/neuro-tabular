"""Internal, dormant infrastructure for future announced API deprecations."""

import re
import warnings


def _warn_deprecated(feature, *, since, remove_in, replacement, stacklevel=2):
    """Emit the canonical FutureWarning; no current feature calls this helper.

    The numeric floor does not replace an actual full minor release of notice,
    migration documentation or a maintainer-approved removal announcement.
    """
    if not all(
        isinstance(value, str) and value.strip() for value in (feature, replacement)
    ):
        raise ValueError("feature and replacement must be nonempty strings.")
    versions = []
    for value in (since, remove_in):
        if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+", value):
            raise ValueError("Deprecation versions must use MAJOR.MINOR.PATCH.")
        versions.append(tuple(map(int, value.split("."))))
    first, last = versions
    if last <= first or (last[0] == first[0] and last[1] < first[1] + 2):
        raise ValueError(
            "Removal requires at least one complete minor series of notice."
        )
    warnings.warn(
        f"NeuroTabular: {feature} is deprecated since {since}; "
        f"removal no earlier than {remove_in}. Use {replacement} instead.",
        FutureWarning,
        stacklevel=stacklevel,
    )
