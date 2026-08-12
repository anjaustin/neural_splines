# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: base_neural.py
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

"""Placeholder base model.

Not implemented. ``__init__`` previously accepted any arguments and did
nothing, producing an object that looked constructed but held no state.
"""

from typing import Any, NoReturn


class BaseNeuralModel:
    """Not implemented. Kept as the intended base class for future work."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "BaseNeuralModel is not implemented. Use neural_splines.SplineMLP "
            "for a working model."
        )

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> NoReturn:
        """Not implemented."""
        raise NotImplementedError("BaseNeuralModel.from_pretrained is not implemented.")
