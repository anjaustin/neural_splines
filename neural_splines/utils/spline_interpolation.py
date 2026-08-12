# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: spline_interpolation.py
#
# This file is part of the Neural Splines project, licensed under the GNU
# Affero General Public License, version 3 or later. See the LICENSE file at
# the repository root for the full text.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
# -----------------------------------------------------------------------------

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
