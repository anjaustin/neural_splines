# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: test_core.py
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

"""Behavioural tests for the core spline layers and the train/inference contract.

These complement ``test_smoke.py``, which covers imports, shapes, the ValueError
guard, the rank ceiling and the in-memory densification round-trip. Here we
cover the parts that are actually load-bearing at runtime but were untested:
single-layer densification, the *on-disk* spline -> dense -> inference
hand-off, input flattening, gradient flow into the control points, the
``cp_bias`` default, and the data-loader helpers (without touching the network).

Nothing here downloads MNIST. The loader tests substitute a tiny in-memory
dataset for ``torchvision.datasets.MNIST``.
"""

import importlib
import importlib.util

import pytest
import torch
import torch.nn as nn

import neural_splines
from neural_splines import DenseMLP, SplineLinear, SplineMLP

# train.py and inference.py import torchvision (an `examples` extra, not a
# core dependency), so tests touching them must skip when it is absent.
_HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None
_needs_torchvision = pytest.mark.skipif(
    not _HAS_TORCHVISION, reason="requires torchvision (pip install -e '.[examples]')"
)


# ---------------------------------------------------------------------------
# SplineLinear.to_dense_linear
# ---------------------------------------------------------------------------


def test_to_dense_linear_shapes_and_type():
    """The densified layer must be a real nn.Linear with the declared shapes."""
    torch.manual_seed(0)
    layer = SplineLinear(64, 32, cp_h=5, cp_w=6)
    dense = layer.to_dense_linear()
    assert isinstance(dense, nn.Linear)
    assert dense.in_features == 64
    assert dense.out_features == 32
    assert dense.weight.shape == (32, 64)
    assert dense.bias is not None
    assert dense.bias.shape == (32,)


def test_to_dense_linear_matches_spline_output():
    """A densified layer must reproduce the spline layer's output exactly."""
    torch.manual_seed(0)
    layer = SplineLinear(64, 32, cp_h=5, cp_w=6).eval()
    dense = layer.to_dense_linear().eval()
    x = torch.randn(8, 64)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), dense(x), rtol=0, atol=0)


def test_to_dense_linear_is_detached_copy():
    """The dense layer must not alias or back-propagate into the control points."""
    torch.manual_seed(0)
    layer = SplineLinear(16, 8, cp_h=4, cp_w=4)
    dense = layer.to_dense_linear()
    assert dense.weight.grad_fn is None
    assert dense.bias.grad_fn is None

    # Mutating the control points afterwards must not change the dense layer.
    before = dense.weight.detach().clone()
    with torch.no_grad():
        layer.weight_control_points.add_(1.0)
    torch.testing.assert_close(dense.weight.detach(), before, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# The real train -> disk -> inference contract
# ---------------------------------------------------------------------------


def test_dense_mlp_state_dict_round_trip_through_disk(tmp_path):
    """train.py saves ``to_dense_mlp().state_dict()``; inference.py rebuilds a
    ``DenseMLP(nn.Linear(...), nn.Linear(...))`` and loads it. That hand-off is
    the actual deliverable of this project, so exercise it end to end.
    """
    torch.manual_seed(0)
    input_size, hidden_size, output_size = 784, 32, 10

    spline = SplineMLP(input_size, hidden_size, output_size, 4, 4).eval()
    dense = spline.to_dense_mlp().eval()

    path = tmp_path / "dense_model.pth"
    torch.save(dense.state_dict(), path)

    # Rebuilt exactly the way neural_splines/core/inference.py main() does it.
    layer1 = nn.Linear(input_size, hidden_size)
    layer2 = nn.Linear(hidden_size, output_size)
    reloaded = DenseMLP(layer1, layer2)
    # ``weights_only=True`` mirrors inference.py and also asserts the saved file
    # is a plain tensor state dict, safe to load without executing pickled code.
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    # A strict load also pins the state_dict key names ("layer1.weight", ...),
    # which are part of the on-disk format inference.py depends on.
    reloaded.load_state_dict(state_dict, strict=True)
    reloaded.eval()

    x = torch.randn(4, input_size)
    with torch.no_grad():
        expected = spline(x)
        torch.testing.assert_close(dense(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(reloaded(x), expected, rtol=0, atol=0)


def test_dense_mlp_state_dict_keys():
    """Pin the on-disk key names that inference.py's load_state_dict relies on."""
    torch.manual_seed(0)
    dense = SplineMLP(64, 32, 10, 4, 4).to_dense_mlp()
    assert set(dense.state_dict().keys()) == {
        "layer1.weight",
        "layer1.bias",
        "layer2.weight",
        "layer2.bias",
    }


# ---------------------------------------------------------------------------
# Input flattening
# ---------------------------------------------------------------------------


def test_spline_mlp_accepts_flat_and_image_shaped_input():
    """(N, 784) and (N, 1, 28, 28) must give identical results.

    train.py feeds the model raw MNIST batches of shape (N, 1, 28, 28) while
    inference.py flattens before calling the model; both paths must agree.
    """
    torch.manual_seed(0)
    model = SplineMLP(28 * 28, 32, 10, 4, 4).eval()
    images = torch.randn(6, 1, 28, 28)
    flat = images.view(6, -1)
    with torch.no_grad():
        torch.testing.assert_close(model(images), model(flat), rtol=0, atol=0)
    assert model(flat).shape == (6, 10)


def test_dense_mlp_accepts_flat_and_image_shaped_input():
    """The densified model must flatten identically, or inference silently drifts."""
    torch.manual_seed(0)
    dense = SplineMLP(28 * 28, 32, 10, 4, 4).to_dense_mlp().eval()
    images = torch.randn(6, 1, 28, 28)
    with torch.no_grad():
        torch.testing.assert_close(
            dense(images), dense(images.view(6, -1)), rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_control_points_receive_gradients():
    """Both control point tensors must actually train."""
    torch.manual_seed(0)
    layer = SplineLinear(64, 32, cp_h=4, cp_w=4)
    out = layer(torch.randn(8, 64))
    out.pow(2).sum().backward()

    assert layer.weight_control_points.grad is not None
    assert layer.bias_control_points.grad is not None
    assert torch.isfinite(layer.weight_control_points.grad).all()
    assert torch.isfinite(layer.bias_control_points.grad).all()
    assert layer.weight_control_points.grad.abs().sum() > 0
    assert layer.bias_control_points.grad.abs().sum() > 0


def test_spline_mlp_control_points_receive_gradients():
    """Gradients must reach the control points of *every* layer of the MLP."""
    torch.manual_seed(0)
    model = SplineMLP(64, 32, 10, 4, 4)
    logits = model(torch.randn(8, 64))
    target = torch.arange(8) % 10
    nn.CrossEntropyLoss()(logits, target).backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert param.grad.abs().sum() > 0, f"zero grad for {name}"

    # And the whole model is only these four small tensors.
    assert {name for name, _ in model.named_parameters()} == {
        "spline1.weight_control_points",
        "spline1.bias_control_points",
        "spline2.weight_control_points",
        "spline2.bias_control_points",
    }


def test_control_point_step_changes_the_output():
    """A gradient step on the control points must move the layer's output."""
    torch.manual_seed(0)
    layer = SplineLinear(32, 16, cp_h=4, cp_w=4)
    x = torch.randn(4, 32)
    before = layer(x).detach().clone()

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    optimizer.zero_grad()
    layer(x).pow(2).sum().backward()
    optimizer.step()

    after = layer(x).detach()
    assert not torch.allclose(before, after)


# ---------------------------------------------------------------------------
# cp_bias defaulting
# ---------------------------------------------------------------------------


def test_cp_bias_defaults_to_cp_h():
    layer = SplineLinear(64, 32, cp_h=7, cp_w=5)
    assert layer.cp_bias == 7
    assert layer.bias_control_points.shape == (7,)


def test_cp_bias_explicit_value_is_honoured():
    layer = SplineLinear(64, 32, cp_h=7, cp_w=5, cp_bias=4)
    assert layer.cp_bias == 4
    assert layer.bias_control_points.shape == (4,)
    assert layer._interpolate_bias().shape == (32,)


def test_cp_bias_is_not_validated_against_the_cubic_minimum():
    """Documents that only ``cp_h``/``cp_w`` are range-checked.

    ``cp_bias`` feeds *linear* interpolation, so fewer than four points is
    legitimate. Even the degenerate ``cp_bias=1`` is well defined: it broadcasts
    to a constant bias vector rather than raising.
    """
    layer = SplineLinear(64, 32, cp_h=4, cp_w=4, cp_bias=2)
    assert layer.bias_control_points.shape == (2,)
    assert layer._interpolate_bias().shape == (32,)

    degenerate = SplineLinear(64, 32, cp_h=4, cp_w=4, cp_bias=1)
    bias = degenerate._interpolate_bias()
    assert bias.shape == (32,)
    torch.testing.assert_close(
        bias, degenerate.bias_control_points.detach().expand(32), rtol=0, atol=0
    )


def test_spline_mlp_uses_cp_for_both_axes():
    """SplineMLP forwards a single ``cp_hidden``/``cp_output`` to both axes."""
    model = SplineMLP(64, 32, 10, cp_hidden=5, cp_output=6)
    assert (model.spline1.cp_h, model.spline1.cp_w) == (5, 5)
    assert (model.spline2.cp_h, model.spline2.cp_w) == (6, 6)
    # ...and therefore to the bias, via the cp_bias default.
    assert model.spline1.cp_bias == 5
    assert model.spline2.cp_bias == 6


# ---------------------------------------------------------------------------
# Data loader helpers -- no network, no MNIST download
# ---------------------------------------------------------------------------


class _FakeMNIST(torch.utils.data.Dataset):
    """Stand-in for ``torchvision.datasets.MNIST`` that touches no disk or network."""

    calls: list = []

    def __init__(self, root, train=True, download=False, transform=None):
        type(self).calls.append(
            {"root": root, "train": train, "download": download, "transform": transform}
        )
        self.n = 8 if train else 4
        self.data = torch.randn(self.n, 1, 28, 28)
        self.targets = torch.arange(self.n) % 10

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def fake_mnist(monkeypatch):
    """Patch ``datasets.MNIST`` out from under train.py / inference.py."""
    from torchvision import datasets

    _FakeMNIST.calls = []
    monkeypatch.setattr(datasets, "MNIST", _FakeMNIST)
    yield _FakeMNIST


@_needs_torchvision
def test_get_data_loaders_wiring(fake_mnist):
    """Check batch size, shuffling and the train/test split without downloading."""
    train_mod = importlib.import_module("neural_splines.core.train")
    train_loader, test_loader = train_mod.get_data_loaders(batch_size=2)

    assert train_loader.batch_size == 2
    assert test_loader.batch_size == 2
    assert len(train_loader.dataset) == 8
    assert len(test_loader.dataset) == 4
    # The training loader shuffles; the test loader must not.
    assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)
    assert isinstance(test_loader.sampler, torch.utils.data.SequentialSampler)

    # One dataset built for train, one for test.
    assert [c["train"] for c in fake_mnist.calls] == [True, False]


@_needs_torchvision
def test_get_data_loaders_applies_normalising_transform(fake_mnist):
    """The MNIST mean/std normalisation is part of the trained model's contract."""
    from torchvision import transforms

    train_mod = importlib.import_module("neural_splines.core.train")
    train_mod.get_data_loaders(batch_size=2)

    transform = fake_mnist.calls[0]["transform"]
    assert isinstance(transform, transforms.Compose)
    kinds = [type(t) for t in transform.transforms]
    assert transforms.ToTensor in kinds
    assert transforms.Normalize in kinds
    normalize = next(t for t in transform.transforms if isinstance(t, transforms.Normalize))
    assert tuple(normalize.mean) == (0.1307,)
    assert tuple(normalize.std) == (0.3081,)


@_needs_torchvision
def test_get_data_loaders_wraps_dataset_failures(monkeypatch):
    """A missing/undownloadable dataset must surface the documented RuntimeError."""
    from torchvision import datasets

    def boom(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(datasets, "MNIST", boom)
    train_mod = importlib.import_module("neural_splines.core.train")
    with pytest.raises(RuntimeError, match="MNIST"):
        train_mod.get_data_loaders(batch_size=2)


@_needs_torchvision
def test_get_test_loader_wiring(fake_mnist):
    """inference.get_test_loader must request the *test* split, unshuffled."""
    inference_mod = importlib.import_module("neural_splines.core.inference")
    loader = inference_mod.get_test_loader(batch_size=2)

    assert loader.batch_size == 2
    assert len(loader.dataset) == 4
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    assert [c["train"] for c in fake_mnist.calls] == [False]


@_needs_torchvision
def test_get_test_loader_does_not_swallow_dataset_failures(fake_mnist, monkeypatch):
    """A dataset that cannot be loaded must raise, never yield an empty loader.

    Note the asymmetry with ``train.get_data_loaders``, which wraps the cause in
    a friendly RuntimeError; ``get_test_loader`` currently lets it through raw.
    Either is acceptable here -- silently returning nothing is not.
    """
    from torchvision import datasets

    def boom(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(datasets, "MNIST", boom)
    inference_mod = importlib.import_module("neural_splines.core.inference")
    with pytest.raises((OSError, RuntimeError)):
        inference_mod.get_test_loader(batch_size=2)


@_needs_torchvision
def test_batches_from_the_loader_flow_through_the_model(fake_mnist):
    """End-to-end shape check: a loader batch must be consumable by SplineMLP."""
    torch.manual_seed(0)
    train_mod = importlib.import_module("neural_splines.core.train")
    train_loader, _ = train_mod.get_data_loaders(batch_size=4)
    model = SplineMLP(28 * 28, 32, 10, 4, 4).eval()

    data, target = next(iter(train_loader))
    assert data.shape == (4, 1, 28, 28)
    with torch.no_grad():
        output = model(data)
    assert output.shape == (4, 10)
    assert target.shape == (4,)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_construction_is_seed_deterministic():
    """Same seed, same control points -- the premise of every test above."""
    torch.manual_seed(1234)
    a = SplineLinear(32, 16, cp_h=4, cp_w=4)
    torch.manual_seed(1234)
    b = SplineLinear(32, 16, cp_h=4, cp_w=4)
    torch.testing.assert_close(
        a.weight_control_points.detach(), b.weight_control_points.detach(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        a.bias_control_points.detach(), b.bias_control_points.detach(), rtol=0, atol=0
    )


def test_spline_forward_is_deterministic():
    """Repeated forward passes must be bit-identical (no stochastic interpolation)."""
    torch.manual_seed(0)
    layer = SplineLinear(32, 16, cp_h=4, cp_w=4).eval()
    x = torch.randn(4, 32)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), layer(x), rtol=0, atol=0)


def test_public_alias_matches_underlying_class():
    """``neural_splines.SplineConverter`` is an alias, not a second class."""
    assert neural_splines.SplineConverter is neural_splines.HarmonicCollapseConverter
