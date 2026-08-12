# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: geometric_validation.py
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

"""Placeholder geometric validation helper.

Not implemented. This previously had a body of ``pass``, so it returned ``None``
silently -- indistinguishable from a validation that passed. It now raises.
"""

from typing import Any, NoReturn


def geometric_validation(*args: Any, **kwargs: Any) -> NoReturn:
    """Not implemented. Raises rather than silently reporting success."""
    raise NotImplementedError("geometric_validation() is not implemented.")
