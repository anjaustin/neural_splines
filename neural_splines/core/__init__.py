# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: __init__.py
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

"""Core module for Neural Splines."""

# Explicit re-exports rather than `import *`: the star import pulled this
# module's third-party and typing imports (torch, nn, F, np, ThreadPoolExecutor,
# as_completed, Dict, List, Tuple, Optional, Union) into `neural_splines.core`,
# where they looked like part of the package's own API.
from .neural_spline import (
    DenseMLP,
    aspect_grid,
    HarmonicCollapseConverter,
    SplineLinear,
    SplineMLP,
)
from .spline_conv import SplineConv2d

__all__ = [
    "DenseMLP",
    "aspect_grid",
    "HarmonicCollapseConverter",
    "SplineLinear",
    "SplineMLP",
    "SplineConv2d",
]
