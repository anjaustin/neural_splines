# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: test_spline_conv.py
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

"""Tests for SplineConv2d."""

import pytest
import torch
import torch.nn as nn

import neural_splines


def test_shapes_and_forward():
    conv = neural_splines.SplineConv2d(3, 16, 9, cp_h=4)
    assert conv.control_points.shape == (16, 3, 4, 4)
    assert conv._interpolate_kernel().shape == (16, 3, 9, 9)
    assert conv(torch.randn(2, 3, 28, 28)).shape == (2, 16, 20, 20)


def test_padding_and_stride_match_conv2d_semantics():
    conv = neural_splines.SplineConv2d(1, 8, 9, cp_h=4, padding=4, stride=2)
    out = conv(torch.randn(2, 1, 28, 28))
    ref = nn.Conv2d(1, 8, 9, padding=4, stride=2)(torch.randn(2, 1, 28, 28))
    assert out.shape == ref.shape


def test_to_dense_conv_reproduces_outputs():
    torch.manual_seed(0)
    conv = neural_splines.SplineConv2d(2, 6, 7, cp_h=4, padding=3).eval()
    dense = conv.to_dense_conv().eval()
    x = torch.randn(3, 2, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(conv(x), dense(x))


def test_channels_are_not_mixed():
    """The spline must span only the spatial axes.

    Channel ordering is arbitrary, so interpolating across it is the mistake
    that costs ~10 accuracy points in the dense case. Perturbing one channel
    pair's control points must leave every other pair's kernel untouched.
    """
    torch.manual_seed(0)
    conv = neural_splines.SplineConv2d(3, 4, 7, cp_h=4)
    before = conv._interpolate_kernel().detach().clone()
    with torch.no_grad():
        conv.control_points[2, 1] += 5.0
    after = conv._interpolate_kernel().detach()

    changed = (after - before).abs().amax(dim=(2, 3)) > 1e-6
    assert changed[2, 1], "the perturbed channel pair should have changed"
    changed[2, 1] = False
    assert not changed.any(), "no other channel pair may be affected"


def test_rejects_too_few_control_points():
    with pytest.raises(ValueError):
        neural_splines.SplineConv2d(1, 4, 9, cp_h=3)


def test_rejects_a_grid_that_would_not_compress():
    """A grid at least as large as the kernel stores more than nn.Conv2d."""
    with pytest.raises(ValueError):
        neural_splines.SplineConv2d(1, 4, 5, cp_h=5)
    with pytest.raises(ValueError):
        neural_splines.SplineConv2d(1, 4, 4, cp_h=4)


def test_parameter_saving_is_reported_honestly():
    conv = neural_splines.SplineConv2d(16, 32, 9, cp_h=4)
    dense = nn.Conv2d(16, 32, 9)
    assert conv.control_points.numel() < dense.weight.numel()
    expected = dense.weight.numel() / conv.control_points.numel()
    assert conv.parameter_saving() == pytest.approx(expected)


def test_groups_are_supported():
    conv = neural_splines.SplineConv2d(8, 8, 7, cp_h=4, groups=8, padding=3)
    assert conv.control_points.shape == (8, 1, 4, 4)
    assert conv(torch.randn(2, 8, 16, 16)).shape == (2, 8, 16, 16)
    with pytest.raises(ValueError):
        neural_splines.SplineConv2d(3, 8, 7, cp_h=4, groups=2)


def test_gradients_reach_the_control_points():
    torch.manual_seed(0)
    conv = neural_splines.SplineConv2d(2, 4, 7, cp_h=4, padding=3)
    conv(torch.randn(2, 2, 12, 12)).sum().backward()
    assert conv.control_points.grad is not None
    assert torch.isfinite(conv.control_points.grad).all()
    assert conv.control_points.grad.abs().sum() > 0


def test_bias_can_be_disabled():
    conv = neural_splines.SplineConv2d(1, 4, 7, cp_h=4, bias=False)
    assert conv.bias is None
    assert conv(torch.randn(1, 1, 12, 12)).shape == (1, 4, 6, 6)
