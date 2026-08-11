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
