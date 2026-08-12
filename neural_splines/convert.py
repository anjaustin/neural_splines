# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: convert.py
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

"""Placeholder conversion command-line entry point.

Not implemented. This previously printed "Conversion CLI placeholder" and
exited successfully while registered as the ``spline-convert`` console script.

Working dense-to-spline conversion lives in
:mod:`neural_splines.core.dense_to_neural_spline`; note the limitation recorded
against item C0 in REMEDIATION.md before relying on it.
"""

from typing import NoReturn


def cli_main() -> NoReturn:
    """Not implemented; points at the module that does the work."""
    raise SystemExit(
        "spline-convert is not implemented. Use:\n"
        "  python -m neural_splines.core.dense_to_neural_spline --help"
    )
