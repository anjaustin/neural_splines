"""Placeholder spline interpolation helper.

Not implemented. This previously had a body of ``pass``, so it returned ``None``
silently -- a caller could not tell it had done nothing. It now raises.

The interpolation that actually runs lives in
:meth:`neural_splines.SplineLinear._interpolate_weights`.
"""

from typing import Any, NoReturn


def spline_interpolation(*args: Any, **kwargs: Any) -> NoReturn:
    """Not implemented; see :meth:`neural_splines.SplineLinear._interpolate_weights`."""
    raise NotImplementedError(
        "spline_interpolation() is not implemented. The interpolation used by "
        "the package is SplineLinear._interpolate_weights()."
    )
