# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: cli.py
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

"""Placeholder command-line entry point.

Not implemented. This previously printed "Neural Splines CLI placeholder" and
exited successfully, while being registered as the ``neural-splines`` console
script -- so installing the package put a command on the PATH that appeared to
work and did nothing. The console script has been removed until this does
something real.
"""

from typing import NoReturn

_USAGE = """\
neural-splines has no CLI yet. The working entry points are:

  python -m neural_splines.core.train                 # train a spline MLP on MNIST
  python -m neural_splines.core.inference             # evaluate a densified model
  python -m neural_splines.core.dense_to_neural_spline  # dense -> spline conversion
"""


def main() -> NoReturn:
    """Not implemented; prints the working entry points and exits non-zero."""
    raise SystemExit(_USAGE)
