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


def test_bias_is_linearly_interpolated_between_control_points():
    """Pin the bias interpolation mode.

    Found by mutation testing: switching `_interpolate_bias` from "linear" to
    "nearest" changed the model's semantics while the whole suite stayed green.
    With control points [0, 1] spread over 5 outputs, linear interpolation must
    give an even ramp; "nearest" would produce plateaus instead.
    """
    layer = neural_splines.SplineLinear(8, 5, cp_h=4, cp_w=4, cp_bias=2)
    with torch.no_grad():
        layer.bias_control_points.copy_(torch.tensor([0.0, 1.0]))
        bias = layer._interpolate_bias()
    torch.testing.assert_close(bias, torch.linspace(0.0, 1.0, 5))


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


def test_separable_forward_matches_the_materializing_path():
    """The memory-frugal path must compute the same function.

    Bicubic interpolation is separable, so W == A @ control_points @ B.T and
    the dense matrix never has to be built. Checked in float64, where the
    two orderings agree to machine precision.
    """
    torch.manual_seed(0)
    layer = neural_splines.SplineLinear(128, 64, 6, 6).double()
    x = torch.randn(8, 128, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), layer.forward_separable(x))


def test_separable_forward_matches_gradients():
    """Gradients must match too, not just outputs."""
    torch.manual_seed(0)
    layer = neural_splines.SplineLinear(128, 64, 6, 6).double()
    x = torch.randn(8, 128, dtype=torch.float64)

    grads = {}
    for separable in (False, True):
        layer.zero_grad()
        layer.separable_forward = separable
        layer(x).sum().backward()
        grads[separable] = layer.weight_control_points.grad.clone()
    torch.testing.assert_close(grads[False], grads[True])


def test_separable_buffers_stay_out_of_state_dict():
    """The cached interpolation operators must not become checkpoint content."""
    layer = neural_splines.SplineLinear(64, 32, 4, 4)
    layer.separable_forward = True
    layer(torch.randn(2, 64))  # populate the cache
    assert set(layer.state_dict()) == {"weight_control_points", "bias_control_points"}


def test_densification_preserves_outputs():
    """to_dense_mlp() must reproduce the spline model's outputs bit-for-bit."""
    torch.manual_seed(0)
    spline = neural_splines.SplineMLP(64, 32, 10, 4, 4).eval()
    dense = spline.to_dense_mlp().eval()
    x = torch.randn(16, 64)
    with torch.no_grad():
        torch.testing.assert_close(spline(x), dense(x), rtol=0, atol=0)


def test_converter_runs_on_non_square_layers():
    """Regression: the converter used to crash on every input.

    ``torch.exp()`` was called on a Python float, and behind that the QR factor
    of a non-square product was assigned back to ``V``. Every ``nn.Linear`` in a
    real network is non-square, so nothing ever converted.
    """
    converter = neural_splines.HarmonicCollapseConverter()
    result = converter.convert_layer(torch.nn.Linear(128, 64), control_ratio=0.05)
    assert result["control_points"].numel() > 0
    assert tuple(result["control_grid"]) == tuple(result["control_points"].shape)


def test_converter_honours_an_explicit_control_grid():
    """Regression: a 4x4 request against a 256x784 weight silently returned 4x7."""
    converter = neural_splines.HarmonicCollapseConverter()
    result = converter.convert_layer(
        torch.nn.Linear(784, 256), control_grid=(4, 4)
    )
    assert tuple(result["control_points"].shape) == (4, 4)

    # A grid the weight matrix cannot support must be refused, not truncated.
    with pytest.raises(ValueError):
        converter.convert_layer(torch.nn.Linear(3, 3), control_grid=(8, 8))


def test_spline_cannot_represent_a_trained_weight_matrix():
    """Characterization test for the converter's central limitation.

    Interpolation imposes smoothness on the weight matrix, but a trained
    weight matrix is not smooth in its index coordinates. Fitting control
    points to one reconstructs it no better than predicting zeros. This is
    what makes the compression premise fail, independently of the bugs above.

    The test is deliberately loose: it asserts only that the error is large.
    If a future change makes splines genuinely fit trained weights, this test
    fails and should be revisited rather than relaxed.
    """
    torch.manual_seed(0)
    linear = torch.nn.Linear(128, 64)
    # Give it a weight matrix with realistic (non-smooth) structure.
    torch.nn.init.kaiming_normal_(linear.weight)
    W = linear.weight.data

    converter = neural_splines.HarmonicCollapseConverter()
    result = converter.convert_layer(linear, control_grid=(8, 8))
    reconstructed = torch.nn.functional.interpolate(
        result["control_points"][None, None].float(),
        size=tuple(W.shape),
        mode="bicubic",
        align_corners=True,
    )[0, 0]

    rel_error = (torch.norm(reconstructed - W) / torch.norm(W)).item()
    assert rel_error > 0.5, (
        f"relative reconstruction error {rel_error:.4f} is far better than the "
        "~1.0 previously measured; the compression premise may warrant re-evaluation"
    )
