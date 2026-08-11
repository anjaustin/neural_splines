"""Smoke tests guarding the import path and the spline/dense round-trip.

These are deliberately minimal. Their main job is to fail loudly if the
package ever again becomes unimportable (as it was when
``neural_splines/__init__.py`` referenced a module that did not exist),
or if densification stops reproducing the spline model's outputs.
"""

import importlib
import importlib.util

import pytest
import torch

import neural_splines

# train.py and inference.py import torchvision (an `examples` extra, not a
# core dependency), so tests touching them must skip when it is absent.
_HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None
_needs_torchvision = pytest.mark.skipif(
    not _HAS_TORCHVISION, reason="requires torchvision (pip install -e '.[examples]')"
)


def test_package_imports_and_exports():
    """The top-level package must import and expose its public names."""
    assert neural_splines.__version__
    for name in neural_splines.__all__:
        assert hasattr(neural_splines, name), f"missing export: {name}"
    # Regression: these must be real classes, not None placeholders.
    assert neural_splines.SplineLinear is not None
    assert neural_splines.SplineMLP is not None


@pytest.mark.parametrize(
    "module",
    [
        "neural_splines.core.neural_spline",
        "neural_splines.core.dense_to_neural_spline",
    ],
)
def test_submodules_import(module):
    """Regression: these used flat ``from neural_spline import ...`` imports."""
    assert importlib.import_module(module) is not None


@_needs_torchvision
@pytest.mark.parametrize(
    "module", ["neural_splines.core.train", "neural_splines.core.inference"]
)
def test_example_scripts_import(module):
    """Same regression, for the two modules that additionally need torchvision."""
    assert importlib.import_module(module) is not None


def test_spline_linear_shapes():
    layer = neural_splines.SplineLinear(64, 32, cp_h=4, cp_w=4)
    assert layer._interpolate_weights().shape == (32, 64)
    assert layer._interpolate_bias().shape == (32,)
    assert layer(torch.randn(8, 64)).shape == (8, 32)


def test_spline_linear_rejects_too_few_control_points():
    with pytest.raises(ValueError):
        neural_splines.SplineLinear(64, 32, cp_h=3, cp_w=4)


def test_interpolated_weight_is_low_rank():
    """A cp_h x cp_w grid can only ever produce a rank <= min(cp_h, cp_w) matrix.

    This is the capacity ceiling documented in the README's Results section;
    pin it so the trade-off is not silently misunderstood.
    """
    layer = neural_splines.SplineLinear(784, 256, cp_h=4, cp_w=4)
    W = layer._interpolate_weights().detach()
    assert torch.linalg.matrix_rank(W).item() <= 4


def test_densification_preserves_outputs():
    """to_dense_mlp() must reproduce the spline model's outputs bit-for-bit."""
    torch.manual_seed(0)
    spline = neural_splines.SplineMLP(64, 32, 10, 4, 4).eval()
    dense = spline.to_dense_mlp().eval()
    x = torch.randn(16, 64)
    with torch.no_grad():
        torch.testing.assert_close(spline(x), dense(x), rtol=0, atol=0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "HarmonicCollapseConverter is broken and has never run successfully: "
        "torch.exp() is called on a Python float in "
        "subspace_iteration_resonance(). Behind that, the same routine assigns "
        "the QR factor of a non-square product back to V, so it only ever "
        "works on square weight matrices. See the audit notes."
    ),
)
def test_converter_is_known_broken():
    """Documents the converter's state so a green suite does not imply it works."""
    converter = neural_splines.HarmonicCollapseConverter()
    result = converter.convert_layer(torch.nn.Linear(784, 256), 0.05)
    assert result["control_points"].numel() > 0
