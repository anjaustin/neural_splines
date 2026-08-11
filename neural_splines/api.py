"""Planned high-level API for Neural Splines.

None of these are implemented. They are kept as the intended surface so the
shape of the eventual API is visible, but every one raises rather than
returning a plausible-looking result.

For functionality that does exist today, use:

* :class:`neural_splines.SplineLinear` / :class:`neural_splines.SplineMLP`
* ``python -m neural_splines.core.train``
* ``python -m neural_splines.core.inference``
* ``python -m neural_splines.core.dense_to_neural_spline``
"""

from typing import Any, NoReturn

_NOT_IMPLEMENTED = (
    "{name}() is not implemented. See the neural_splines.api module docstring "
    "for the parts of the package that are usable today."
)


def convert_model_to_splines(*args: Any, **kwargs: Any) -> NoReturn:
    """Not implemented. See :mod:`neural_splines.api`."""
    raise NotImplementedError(_NOT_IMPLEMENTED.format(name="convert_model_to_splines"))


def load_neural_splines_model(*args: Any, **kwargs: Any) -> NoReturn:
    """Not implemented. See :mod:`neural_splines.api`."""
    raise NotImplementedError(_NOT_IMPLEMENTED.format(name="load_neural_splines_model"))


def visualize_spline_structure(*args: Any, **kwargs: Any) -> NoReturn:
    """Not implemented. See :mod:`neural_splines.api`."""
    raise NotImplementedError(_NOT_IMPLEMENTED.format(name="visualize_spline_structure"))
