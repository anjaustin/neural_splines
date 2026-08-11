"""Placeholder geometric validation helper.

Not implemented. This previously had a body of ``pass``, so it returned ``None``
silently -- indistinguishable from a validation that passed. It now raises.
"""

from typing import Any, NoReturn


def geometric_validation(*args: Any, **kwargs: Any) -> NoReturn:
    """Not implemented. Raises rather than silently reporting success."""
    raise NotImplementedError("geometric_validation() is not implemented.")
