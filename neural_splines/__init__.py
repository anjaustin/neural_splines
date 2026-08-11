"""Simplified working layout for Neural Splines."""

__version__ = "1.0.0"

# torch is a hard dependency, so this import is deliberately unguarded: a
# failure here should surface the real ImportError rather than be masked by
# binding the public names to None (which only defers the error to a
# confusing "NoneType is not callable" at the call site).
from .core.neural_spline import (
    SplineLinear,
    aspect_grid,
    SplineMLP,
    DenseMLP,
    HarmonicCollapseConverter,
    HarmonicCollapseConverter as SplineConverter,
)
from .core.spline_conv import SplineConv2d
from .models.base_neural import BaseNeuralModel
from .compression.adaptive import DeepSeekSplineAdapter
from .compression.optimizer import CompressionOptimizer
from .utils.geometric_validation import geometric_validation
from .utils.spline_interpolation import spline_interpolation

__all__ = [
    "__version__",
    "SplineLinear",
    "SplineConv2d",
    "aspect_grid",
    "SplineMLP",
    "DenseMLP",
    "HarmonicCollapseConverter",
    "SplineConverter",
    "BaseNeuralModel",
    "DeepSeekSplineAdapter",
    "CompressionOptimizer",
    "geometric_validation",
    "spline_interpolation",
]
