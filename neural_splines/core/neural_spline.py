# -----------------------------------------------------------------------------
# Project Name: Neural Splines
# File: neural_spline.py
# Author: Robert Sitton
# Contact: rsitton@quholo.com
#
# This file is part of a project licensed under the GNU Affero General Public License.
#
# GNU AFFERO GENERAL PUBLIC LICENSE
# Version 3, 19 November 2007
#
# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

"""
neural_spline.py
================

This module provides PyTorch components that implement weight
compression using two‑dimensional spline interpolation.  Traditional
dense layers store one parameter per connection, which quickly
becomes prohibitive on resource‑constrained hardware.  Neural
splines address this by representing the full weight matrix with a
much smaller grid of **control points**.  During the forward
pass the grid is up‑sampled using bicubic interpolation to
construct the dense weight matrix on the fly.  A separate vector of
control points defines the bias.

The implementation below is intentionally concise yet educational.
It depends only on the core PyTorch library and does not require
any external runtime.  After training you can convert a spline
layer into an equivalent dense layer so that inference can proceed
without on‑the‑fly interpolation.  Readers interested in the
mathematics of splines will find an overview in the accompanying
paper and LaTeX document.

This version includes the HarmonicCollapseConverter for converting
existing dense neural networks into spline representations through
harmonic decomposition and subspace resonance.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

class SplineLinear(nn.Module):
    """Linear layer with weights defined by a bicubic spline.

    Instead of storing a full ``out_features × in_features`` weight
    matrix, this layer maintains a much smaller set of control
    points.  During the forward pass the control point grid is
    interpolated to the required shape using bicubic interpolation
    provided by :func:`torch.nn.functional.interpolate`.

    A separate vector of control points is used to generate the
    bias.  This approach dramatically reduces the number of trainable
    parameters while still allowing the network to approximate the
    expressivity of a dense layer.

    This implementation follows the theoretical discussion in
    Section~2 (Spline--Based Compression) and Section~3 (Reference
    Implementation) of the accompanying LaTeX document.  For a detailed
    derivation of the interpolation formula see equation~(1) in
    Section~2.

    Parameters
    ----------
    in_features : int
        The dimensionality of the input.
    out_features : int
        The dimensionality of the output.
    cp_h : int, optional
        Number of control points along the output (height) axis.  A
        larger value allows the spline to model finer variation along
        the output dimension.  Defaults to 4.
    cp_w : int, optional
        Number of control points along the input (width) axis.
        Defaults to 4.
    cp_bias : int, optional
        Number of control points used to generate the bias.  If
        omitted, ``cp_bias`` defaults to the same value as
        ``cp_h``.
    """

    # Declared so that mypy knows the type of the lazily populated buffers
    # registered in __init__; register_buffer(..., None) alone leaves them
    # untyped.
    _interp_a: Optional[torch.Tensor]
    _interp_b: Optional[torch.Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cp_h: int = 4,
        cp_w: int = 4,
        cp_bias: int | None = None,
    ) -> None:
        """
        Initialize a spline‐linear layer.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        cp_h : int, default=4
            Number of control points along the output dimension (height).
        cp_w : int, default=4
            Number of control points along the input dimension (width).
        cp_bias : int, optional
            Number of control points for the bias spline.  If omitted,
            defaults to ``cp_h``.

        Notes
        -----
        Cubic B‑spline interpolation requires at least four control points in
        each dimension.  If ``cp_h`` or ``cp_w`` are less than 4 a
        ``ValueError`` is raised.  See Section \ref{sec:spline-compression}
        of the accompanying LaTeX document for the mathematical
        formulation of the interpolation.
        """
        super().__init__()
        if cp_h < 4 or cp_w < 4:
            raise ValueError(
                "cp_h and cp_w must be at least 4 for cubic spline interpolation"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.cp_h = cp_h
        self.cp_w = cp_w
        self.cp_bias = cp_bias if cp_bias is not None else cp_h

        # Weight control points.  Initialized with a small random
        # distribution to avoid large initial outputs.  Shape: (Hcp, Wcp)
        self.weight_control_points = nn.Parameter(
            torch.randn(self.cp_h, self.cp_w) * 0.02
        )

        # Bias control points.  Shape: (Bcp,)
        self.bias_control_points = nn.Parameter(
            torch.randn(self.cp_bias) * 0.02
        )

        # Opt-in flag for the memory-frugal forward path; see
        # :meth:`forward_separable`. Off by default so numerics are unchanged.
        self.separable_forward = False

        # Cached 1-D interpolation operators, built on first use. Registered as
        # non-persistent buffers up front so they follow .to(device) and stay
        # out of state_dict; register_buffer would reject the name later if it
        # already existed as a plain attribute.
        self.register_buffer("_interp_a", None, persistent=False)
        self.register_buffer("_interp_b", None, persistent=False)

    @classmethod
    def from_dense(cls, linear: nn.Linear, cp_h: int = 4, cp_w: int = 4, 
                   cp_bias: int | None = None) -> 'SplineLinear':
        """Create a SplineLinear layer from a dense Linear layer.
        
        Parameters
        ----------
        linear : nn.Linear
            The dense linear layer to convert.
        cp_h : int, default=4
            Number of control points along the output dimension.
        cp_w : int, default=4
            Number of control points along the input dimension.
        cp_bias : int, optional
            Number of control points for the bias spline.
            
        Returns
        -------
        SplineLinear
            A spline layer initialized to approximate the dense layer.
        """
        out_features, in_features = linear.weight.shape
        spline_layer = cls(in_features, out_features, cp_h, cp_w, cp_bias)

        # Use HarmonicCollapseConverter to find optimal control points. The grid
        # is requested explicitly: deriving it from a ratio does not round-trip
        # to the (cp_h, cp_w) this layer was just constructed with.
        converter = HarmonicCollapseConverter()
        layer_data = converter.convert_layer(
            linear, control_grid=(spline_layer.cp_h, spline_layer.cp_w)
        )

        # Set the control points
        control_points = layer_data['control_points']
        if tuple(control_points.shape) != tuple(spline_layer.weight_control_points.shape):
            raise ValueError(
                f"converter returned control points of shape "
                f"{tuple(control_points.shape)}, expected "
                f"{tuple(spline_layer.weight_control_points.shape)}"
            )
        spline_layer.weight_control_points.data = control_points.to(
            spline_layer.weight_control_points.device
        )

        # Handle bias - interpolate down to cp_bias points
        if linear.bias is not None:
            bias_cp_count = spline_layer.cp_bias
            bias_indices = torch.linspace(0, out_features-1, bias_cp_count).long()
            spline_layer.bias_control_points.data = linear.bias.data[bias_indices]
        
        return spline_layer

    def _interpolate_weights(self) -> torch.Tensor:
        """Generate the dense weight matrix by bicubic interpolation.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(out_features, in_features)``.
        """
        # Reshape control point grid to match the expected input of
        # torch.nn.functional.interpolate.  The function expects
        # input with shape (N, C, H, W).  We use N=C=1 because we
        # interpolate a single 2‑D surface.
        cp = self.weight_control_points.unsqueeze(0).unsqueeze(0)
        # Perform bicubic interpolation to the desired spatial size.
        # ``align_corners=True`` ensures that the extreme control
        # points influence the extreme weights exactly.
        dense = F.interpolate(
            cp,
            size=(self.out_features, self.in_features),
            mode="bicubic",
            align_corners=True,
        )
        # Remove batch and channel dimensions.
        return dense.squeeze(0).squeeze(0)

    def _interpolate_bias(self) -> torch.Tensor:
        """Generate the dense bias vector by linear interpolation.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(out_features,)``.
        """
        # For a 1‑D bias we treat the control points as a length
        # ``cp_bias`` sequence.  By adding dummy dimensions we can
        # reuse :func:`torch.nn.functional.interpolate` for linear
        # interpolation.  The input shape becomes (N, C, L), and the
        # output will be (N, C, L_out).  We again set N=C=1.
        cp = self.bias_control_points.unsqueeze(0).unsqueeze(0)
        dense = F.interpolate(
            cp,
            size=(self.out_features),
            mode="linear",
            align_corners=True,
        )
        return dense.squeeze(0).squeeze(0)

    def _interpolation_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the two 1-D interpolation operators ``(A, B)``.

        Bicubic interpolation is a tensor product of two 1-D cubic
        interpolations, so the dense weight factors exactly as
        ``W = A @ control_points @ B.T`` with ``A`` of shape
        ``(out_features, cp_h)`` and ``B`` of shape ``(in_features, cp_w)``.
        Both are constants determined by the layer's shape, so they are built
        once and cached as non-persistent buffers (they follow ``.to(device)``
        but stay out of ``state_dict``).
        """
        if self._interp_a is None or self._interp_b is None:
            def axis_operator(n_cp: int, n_out: int) -> torch.Tensor:
                identity = torch.eye(n_cp, device=self.weight_control_points.device,
                                     dtype=self.weight_control_points.dtype)
                return F.interpolate(
                    identity.unsqueeze(0).unsqueeze(0),
                    size=(n_out, n_cp),
                    mode="bicubic",
                    align_corners=True,
                ).squeeze(0).squeeze(0)

            self._interp_a = axis_operator(self.cp_h, self.out_features)
            self._interp_b = axis_operator(self.cp_w, self.in_features)
        assert self._interp_a is not None and self._interp_b is not None
        return self._interp_a, self._interp_b

    def forward_separable(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass that never materializes the dense weight matrix.

        Mathematically identical to :meth:`forward` up to floating point
        accumulation order, but it contracts the input against the two 1-D
        interpolation operators instead of building the full
        ``out_features x in_features`` matrix:

            x @ W.T  ==  ((x @ B) @ control_points.T) @ A.T

        The largest intermediate is ``(batch, cp_w)`` rather than
        ``(out_features, in_features)``. For ``SplineMLP(784, 256, 10)`` at
        ``cp = 6`` this cuts the peak intermediate from 200,704 elements to
        4,704. This is what makes the layer's storage saving translate into a
        runtime memory saving; :meth:`forward` alone does not.
        """
        a, b = self._interpolation_matrices()
        hidden = input @ b                       # (batch, cp_w)
        hidden = hidden @ self.weight_control_points.t()   # (batch, cp_h)
        return hidden @ a.t() + self._interpolate_bias()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the linear transformation with spline‑generated weights.

        Set ``layer.separable_forward = True`` to route through
        :meth:`forward_separable`, which avoids materializing the dense weight
        matrix. It is off by default so that existing numerics are unchanged.

        Parameters
        ----------
        input : torch.Tensor
            A tensor of shape ``(batch_size, in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, out_features)``.
        """
        if self.separable_forward:
            return self.forward_separable(input)
        weight = self._interpolate_weights()
        bias = self._interpolate_bias()
        return F.linear(input, weight, bias)

    def to_dense_linear(self) -> nn.Linear:
        """Return an equivalent dense :class:`~torch.nn.Linear` layer.

        Converting a spline layer to a dense layer fixes the
        interpolated weights and biases at their current values.
        This is useful for inference when you no longer need to
        update the control points and want to avoid the overhead of
        interpolation on every forward pass.

        Returns
        -------
        nn.Linear
            A dense linear layer with identical output to this
            spline layer at the current state of the control points.
        """
        dense_weight = self._interpolate_weights()
        dense_bias = self._interpolate_bias()
        linear = nn.Linear(self.in_features, self.out_features)
        # Copy the tensors into the new layer without tracking
        # gradients.  The ``detach()`` call prevents PyTorch from
        # storing a backward reference to the original control
        # points.  ``clone()`` ensures the data is copied.
        linear.weight.data = dense_weight.detach().clone()
        linear.bias.data = dense_bias.detach().clone()
        return linear


def aspect_grid(out_features: int, in_features: int, budget: int) -> Tuple[int, int]:
    """Split ``budget`` control points across a ``(out_features, in_features)``
    weight matrix, allocating more of them to the input axis.

    A square grid is a poor fit for a weight matrix that is not square. What
    limits accuracy in practice is not the rank bound
    ``rank <= min(cp_h, cp_w)`` but how finely the grid can vary across the
    *input* dimension. Measured on MNIST at a fixed 2,954 parameters, changing
    only the aspect ratio of the 256x784 first layer:

        8x256 -> 92.03%    32x64 -> 95.76%    64x32 -> 85.25%
       16x128 -> 95.37%    45x45 -> 93.23%   256x8  -> 78.21%

    ``32x64`` and ``64x32`` have the same rank and the same parameter count and
    differ by 10.5 points, so the asymmetry is real and worth spending the
    budget on.

    This allocates ``cp_h / cp_w = sqrt(out_features / in_features)``, then
    clamps each axis to ``[4, dimension]``. Clamping ``cp_h`` to
    ``out_features`` matters for classifier heads: with 10 output classes it
    yields one control point per class, so the layer never interpolates across
    an arbitrary class ordering.

    Parameters
    ----------
    out_features, in_features : int
        Shape of the weight matrix the grid will be expanded to.
    budget : int
        Approximate number of control points to spend. The product of the
        returned grid is close to, but not exactly, this value.

    Returns
    -------
    (int, int)
        ``(cp_h, cp_w)`` suitable for :class:`SplineLinear`.
    """
    if budget < 16:
        raise ValueError("budget must be at least 16 (a 4x4 grid)")
    ratio = math.sqrt(out_features / in_features)
    cp_h = int(round(math.sqrt(budget * ratio)))
    cp_h = max(4, min(cp_h, out_features))
    cp_w = max(4, min(budget // cp_h, in_features))
    return cp_h, cp_w


def _as_grid(cp: int | Tuple[int, int]) -> Tuple[int, int]:
    """Accept either a single int (square grid) or an explicit ``(cp_h, cp_w)``."""
    if isinstance(cp, int):
        return cp, cp
    cp_h, cp_w = cp
    return int(cp_h), int(cp_w)


class SplineMLP(nn.Module):
    """A simple multilayer perceptron using spline linear layers.

    The network contains two spline linear layers separated by a
    rectified linear unit.  It is designed for small image
    classification tasks such as MNIST, where the input is flattened
    into a one‑dimensional vector.  By adjusting the number of
    control points you can trade expressivity for parameter count.

    This architecture corresponds to the reference implementation
    described in Section~3 of the LaTeX document.  See Section~4
    (Training on MNIST) for usage and empirical results.

    Parameters
    ----------
    input_size : int
        Number of input features.  For 28×28 MNIST images this is
        784.
    hidden_size : int
        Number of hidden units in the intermediate representation.
    output_size : int
        Number of output units (e.g. 10 for MNIST digit classes).
    cp_hidden : int or (int, int), optional
        Control points for the hidden layer. An ``int`` gives a square
        ``cp x cp`` grid; a tuple gives an explicit ``(cp_h, cp_w)``.
        Defaults to 4.
    cp_output : int or (int, int), optional
        Control points for the output layer, same convention.
        Defaults to 4.

    Notes
    -----
    A square grid is rarely the best use of a parameter budget -- see
    :func:`aspect_grid`, and :meth:`with_budget` which applies it. Passing an
    ``int`` here reproduces the original square-grid behaviour exactly.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        cp_hidden: int | Tuple[int, int] = 4,
        cp_output: int | Tuple[int, int] = 4,
    ) -> None:
        super().__init__()
        h_h, h_w = _as_grid(cp_hidden)
        o_h, o_w = _as_grid(cp_output)
        self.spline1 = SplineLinear(input_size, hidden_size, h_h, h_w)
        self.spline2 = SplineLinear(hidden_size, output_size, o_h, o_w)

    @classmethod
    def with_budget(
        cls,
        input_size: int,
        hidden_size: int,
        output_size: int,
        budget_hidden: int,
        budget_output: int | None = None,
    ) -> "SplineMLP":
        """Build an MLP by parameter budget, choosing each grid's aspect ratio.

        Prefer this over passing a single ``cp``: a square grid starves the
        input axis of a non-square weight matrix. On MNIST at roughly 2,950
        control points this reaches ~95.8% where the equivalent square-grid
        configuration reaches ~84%. See :func:`aspect_grid` for the
        measurements and the allocation rule.

        Parameters
        ----------
        budget_hidden : int
            Control points for the ``hidden_size x input_size`` layer.
        budget_output : int, optional
            Control points for the ``output_size x hidden_size`` layer.
            Defaults to a quarter of ``budget_hidden`` (minimum 16), since the
            output layer is usually much smaller.
        """
        if budget_output is None:
            budget_output = max(16, budget_hidden // 4)
        return cls(
            input_size,
            hidden_size,
            output_size,
            aspect_grid(hidden_size, input_size, budget_hidden),
            aspect_grid(output_size, hidden_size, budget_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input if it has more than two dimensions
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.spline1(x)
        x = F.relu(x)
        x = self.spline2(x)
        return x

    def to_dense_mlp(self) -> nn.Module:
        """Convert the spline MLP into an equivalent dense MLP.

        The returned network consists of ordinary
        :class:`~torch.nn.Linear` layers whose weights and biases are
        fixed to the current interpolated values of the spline
        network.  This conversion eliminates interpolation overhead
        during inference and decouples the dense model from the
        control points.

        Returns
        -------
        nn.Module
            A dense neural network with the same architecture and
            output as the original spline MLP.
        """
        dense1 = self.spline1.to_dense_linear()
        dense2 = self.spline2.to_dense_linear()
        return DenseMLP(dense1, dense2)


class DenseMLP(nn.Module):
    """A dense multilayer perceptron with fixed weights.

    This class mirrors the architecture of :class:`~SplineMLP` but
    uses standard dense layers.  It is used after training a
    :class:`~SplineMLP` to provide a version of the network that can
    be exported and executed on devices that do not support spline
    interpolation at runtime.

    The motivation for this class is discussed in Section~6
    (Densification and Inference) of the LaTeX document.  After
    densification the network no longer references control points or
    interpolation operations, simplifying inference.
    """

    def __init__(self, layer1: nn.Linear, layer2: nn.Linear) -> None:
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x


class HarmonicCollapseConverter:
    """
    Converts traditional weight matrices to Neural Splines through
    harmonic decomposition and subspace resonance.

    .. warning::

       This works mechanically but does not achieve useful compression of a
       fully connected layer, and the limitation is in the approach rather than
       the implementation. Fitting a spline to a trained weight matrix gives
       ~98% relative reconstruction error -- no better than predicting zeros --
       because interpolation assumes neighbouring rows and columns are related
       and a trained network's neuron ordering is arbitrary. The fit is at a
       provable ceiling: the objective is convex in the control points and
       LBFGS reaches its closed-form optimum.

       Retained as a reference implementation and a documented negative
       result. For a case where the premise does hold, see
       :class:`neural_splines.SplineConv2d`, which interpolates only the
       spatial axes of a convolution kernel. Measurements for both are in
       ``experiments/``.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        self.device = device
        self.resonance_cache: Dict[str, torch.Tensor] = {}
        
    def forward_difference_cascade(self, W: torch.Tensor, order: int = 3) -> List[torch.Tensor]:
        """
        Compute cascading forward differences to reveal the underlying
        smoothness structure. Higher order differences expose control point locations.
        """
        # Handle edge case: matrices too small for differences
        if W.shape[0] < 2 or W.shape[1] < 2:
            return [W]
            
        differences = [W]
        current = W.clone()
        
        for k in range(order):
            # Check if we can still compute differences
            if current.shape[0] < 2 or current.shape[1] < 2:
                break
                
            # Horizontal differences
            h_diff = torch.zeros_like(current)
            if current.shape[1] > 1:
                h_diff[:, :-1] = current[:, 1:] - current[:, :-1]
            
            # Vertical differences  
            v_diff = torch.zeros_like(current)
            if current.shape[0] > 1:
                v_diff[:-1, :] = current[1:, :] - current[:-1, :]
            
            # Diagonal difference (the hidden dimension)
            d_diff = torch.zeros_like(current)
            if current.shape[0] > 1 and current.shape[1] > 1:
                d_diff[:-1, :-1] = current[1:, 1:] - current[:-1, :-1]
            
            # Harmonic mean of directional differences
            with torch.no_grad():
                # Avoid division by zero
                epsilon = 1e-8
                h_safe = torch.where(torch.abs(h_diff) > epsilon, h_diff, torch.ones_like(h_diff) * epsilon)
                v_safe = torch.where(torch.abs(v_diff) > epsilon, v_diff, torch.ones_like(v_diff) * epsilon)
                d_safe = torch.where(torch.abs(d_diff) > epsilon, d_diff, torch.ones_like(d_diff) * epsilon)
                
                current = 3.0 / (1.0/h_safe + 1.0/v_safe + 1.0/d_safe)
            
            differences.append(current)
            
        return differences
    
    def subspace_iteration_resonance(self, W: torch.Tensor, 
                                    num_modes: int = 16,
                                    iterations: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract dominant subspace modes through power iteration with
        harmonic dampening. Returns modes and their resonance strengths.
        """
        m, n = W.shape
        
        # Handle edge cases
        if m == 0 or n == 0:
            return torch.zeros((n, 0), device=self.device), torch.zeros(0, device=self.device)
            
        # Adjust num_modes for small matrices
        num_modes = min(num_modes, min(m, n))
        if num_modes == 0:
            return torch.zeros((n, 0), device=self.device), torch.zeros(0, device=self.device)
        
        # Initialize with quantum-inspired random phases
        V = torch.randn(n, num_modes, device=self.device)
        V = V / (torch.norm(V, dim=0, keepdim=True) + 1e-8)
        
        # Add harmonic perturbation
        harmonics = torch.stack([
            torch.sin(torch.linspace(0, k*np.pi, n, device=self.device))
            for k in range(1, num_modes+1)
        ]).T
        V = 0.7 * V + 0.3 * harmonics
        V = V / (torch.norm(V, dim=0, keepdim=True) + 1e-8)
        
        # Subspace iteration with resonance tracking
        resonances = []
        
        for iteration in range(iterations):
            # One step of subspace iteration on W^T W. Both factors are applied
            # so that V keeps its (n, num_modes) shape; taking the QR factor of
            # W @ V alone would silently reshape V to (m, num_modes) and break
            # the next iteration for every non-square W -- that is, for every
            # weight matrix in a real network.
            U = W @ V
            B = W.T @ U

            # Harmonic dampening based on iteration number. math.exp, not
            # torch.exp: the argument is a Python float, and torch.exp rejects
            # it with a TypeError.
            damping = math.exp(-iteration / 20.0)
            B = B + damping * torch.randn_like(B) * 0.01

            # QR decomposition for orthogonalization
            try:
                Q, _ = torch.linalg.qr(B)
                V = Q
            except (torch.linalg.LinAlgError, RuntimeError):
                # Fallback for numerical instability
                V = B / (torch.norm(B, dim=0, keepdim=True) + 1e-8)

            # Track resonance (how well modes capture the space)
            reconstruction = (V @ V.T @ W.T).T
            resonance = torch.norm(reconstruction - W) / (torch.norm(W) + 1e-8)
            resonances.append(resonance.item())

            # An "adaptive convergence with golden ratio" step used to sit here,
            # replacing 38% of V with fresh noise whenever the resonance ticked
            # up. Near convergence the resonance rises from floating-point noise
            # alone, so it fired repeatedly and destroyed the very subspace it
            # was meant to refine. Measured against the true right-singular
            # subspace of a random 256x784 matrix over 200 iterations
            # (principal angles, degrees, lower is better):
            #
            #   clean iteration          [0.0, 0.0, 0.0, 4.2]    0 restarts
            #   with this step           [2.4, 5.6, 13.9, 59.8]  7 restarts
            #   this step, without noise [11.1, 25.4, 47.6, 75.1] 11 restarts
            #
            # It made the result worse in every configuration tested, so it is
            # removed rather than tuned. The residual ~4 degrees on the trailing
            # mode is inherent: for a random matrix the singular values are
            # nearly equal (s5/s4 = 0.994 here), so the last mode of the block
            # converges very slowly. Oversampling the block would fix that.

        # Extract eigenvalues through Rayleigh quotient
        eigenvalues = torch.diag(V.T @ W.T @ W @ V)

        return V, eigenvalues
    
    def parallel_spline_synthesis(self, W: torch.Tensor, 
                                 control_points: Tuple[int, int] = (6, 6)) -> torch.Tensor:
        """
        Synthesize spline control points through parallel harmonic collapse.
        This is where the magic happens - finding the minimal representation.
        """
        m, n = W.shape
        cp_m, cp_n = control_points
        
        # Handle edge cases: ensure control points don't exceed matrix dimensions
        cp_m = min(cp_m, m)
        cp_n = min(cp_n, n)
        
        # Ensure minimum control points for cubic interpolation
        if cp_m < 4 or cp_n < 4:
            cp_m = max(4, cp_m)
            cp_n = max(4, cp_n)
            # If original matrix is too small, just return it padded
            if m < cp_m or n < cp_n:
                padded = torch.zeros(cp_m, cp_n, device=self.device)
                padded[:m, :n] = W
                return padded
        
        # Forward difference cascade
        diff_cascade = self.forward_difference_cascade(W, order=3)
        
        # Subspace modes
        modes, eigenvalues = self.subspace_iteration_resonance(W, num_modes=min(16, min(m, n)//2))
        
        # Initialize control points using multiple strategies in parallel
        strategies = []
        
        # Strategy 1: Uniform sampling with jitter
        uniform_cp = torch.zeros(cp_m, cp_n, device=self.device)
        for i in range(cp_m):
            for j in range(cp_n):
                # Sample with golden ratio offsets
                row_idx = int((i + 0.618) * m / cp_m) % m
                col_idx = int((j + 0.618) * n / cp_n) % n
                uniform_cp[i, j] = W[row_idx, col_idx]
        strategies.append(uniform_cp)
        
        # Strategy 2: Importance sampling based on gradient magnitude
        if len(diff_cascade) > 1:
            grad_magnitude = torch.abs(diff_cascade[1])
        else:
            grad_magnitude = torch.abs(W)
            
        importance_cp = torch.zeros(cp_m, cp_n, device=self.device)
        
        # Flatten and sort by importance
        flat_grad = grad_magnitude.flatten()
        k = min(cp_m * cp_n, flat_grad.shape[0])
        if k > 0:
            values, indices = torch.topk(flat_grad, k=k)
            
            for idx in range(k):
                i, j = idx // cp_n, idx % cp_n
                if i < cp_m and j < cp_n:
                    pos = int(indices[idx].item())
                    row, col = pos // n, pos % n
                    if row < m and col < n:
                        importance_cp[i, j] = W[row, col]
        strategies.append(importance_cp)
        
        # Strategy 3: Subspace projection
        subspace_cp = torch.zeros(cp_m, cp_n, device=self.device)
        if modes.shape[1] > 0:  # Check if we have modes
            mode_projection = W @ modes
            
            for i in range(cp_m):
                for j in range(cp_n):
                    mode_i = min(i * modes.shape[1] // cp_m, modes.shape[1]-1)
                    row_idx = min(i * m // cp_m, m-1)
                    col_idx = min(j * n // cp_n, n-1)
                    if mode_i < mode_projection.shape[1]:
                        subspace_cp[i, j] = mode_projection[row_idx, mode_i]
        strategies.append(subspace_cp)
        
        # Parallel optimization of each strategy
        def optimize_strategy(cp_init: torch.Tensor) -> torch.Tensor:
            cp = cp_init.clone().requires_grad_(True)
            optimizer = torch.optim.LBFGS([cp], lr=0.1, max_iter=20, line_search_fn='strong_wolfe')
            
            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                # Bicubic interpolation to reconstruct
                cp_4d = cp.unsqueeze(0).unsqueeze(0)
                
                # Handle edge case where target size is smaller than control points
                target_size = (m, n)
                if m < 4 or n < 4:
                    # Use bilinear for very small targets
                    mode = 'bilinear'
                else:
                    mode = 'bicubic'
                    
                reconstructed = F.interpolate(
                    cp_4d,
                    size=target_size,
                    mode=mode,
                    align_corners=True
                ).squeeze()
                
                # Multi-scale loss
                loss = torch.norm(reconstructed - W)
                
                # Add smoothness regularization
                if cp.shape[0] > 1 and cp.shape[1] > 1:
                    smooth_loss = torch.norm(cp[1:, :] - cp[:-1, :]) + \
                                 torch.norm(cp[:, 1:] - cp[:, :-1])
                    loss = loss + 0.01 * smooth_loss
                
                loss.backward()
                return cast(torch.Tensor, loss)
            
            try:
                optimizer.step(closure)
            except (RuntimeError, ValueError) as exc:
                # Fall back to the unoptimized initialization, but say so: a
                # bare `except: pass` here made a failed LBFGS run
                # indistinguishable from a successful one.
                print(f"  LBFGS refinement failed ({type(exc).__name__}: {exc}); "
                      f"using unoptimized control points for this strategy")

            return cp.detach()
        
        # Execute optimization
        optimized = []
        for strategy in strategies:
            try:
                opt_cp = optimize_strategy(strategy)
                optimized.append(opt_cp)
            except Exception as e:
                # If optimization fails, use the initial strategy
                optimized.append(strategy)
        
        # Select best based on reconstruction error
        best_cp = None
        best_error = float('inf')
        
        for cp in optimized:
            cp_4d = cp.unsqueeze(0).unsqueeze(0)
            mode = 'bilinear' if m < 4 or n < 4 else 'bicubic'
            
            reconstructed = F.interpolate(
                cp_4d,
                size=(m, n),
                mode=mode,
                align_corners=True
            ).squeeze()
            
            error = torch.norm(reconstructed - W).item()
            
            if error < best_error:
                best_error = error
                best_cp = cp
        
        return best_cp if best_cp is not None else strategies[0]
    
    def convert_layer(self, layer: nn.Module,
                     control_ratio: float = 0.1,
                     control_grid: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Convert a single layer to Neural Spline representation.
        Returns control points and metadata.

        Parameters
        ----------
        control_ratio : float
            Target fraction of the original parameter count, used to derive a
            grid size when ``control_grid`` is not given.
        control_grid : (int, int), optional
            Exact control point grid to produce. Callers that must receive a
            specific shape back -- notably :meth:`SplineLinear.from_dense`,
            whose layer is already constructed around ``(cp_h, cp_w)`` -- should
            pass it here. Deriving the grid from ``control_ratio`` alone does
            not round-trip: a ratio computed from a 4x4 request against a
            256x784 weight yields a 4x7 grid, and assigning that to the layer's
            ``.data`` silently leaves ``cp_h``/``cp_w`` describing a shape the
            parameter no longer has.
        """
        if not hasattr(layer, 'weight'):
            raise ValueError("Layer must have weight attribute")

        # nn.Module.__getattr__ is typed as returning Tensor | Module | Size, so
        # the attribute has to be narrowed before it can be used as a tensor.
        weight = cast(torch.Tensor, layer.weight)
        W = weight.data.to(self.device)

        # Handle different layer types
        original_shape = W.shape
        if len(W.shape) == 4:  # Conv2d
            # Flatten to 2D for processing
            out_c, in_c, k_h, k_w = W.shape
            W = W.reshape(out_c, in_c * k_h * k_w)
        elif len(W.shape) == 1:  # Bias or BatchNorm
            # Treat as column vector
            W = W.unsqueeze(1)
        elif len(W.shape) != 2:
            # Flatten any other shape to 2D
            W = W.reshape(W.shape[0], -1)
        
        # Handle empty tensors
        if W.numel() == 0:
            return {
                'control_points': torch.zeros(4, 4, device=self.device),
                'original_shape': original_shape,
                'compression_ratio': 1.0,
                'control_grid': (4, 4)
            }
        
        # Determine control point grid size
        m, n = W.shape
        if control_grid is not None:
            cp_m, cp_n = control_grid
        else:
            total_params = m * n
            target_params = max(16, int(total_params * control_ratio))  # Minimum 16 control points

            # Calculate grid dimensions
            cp_m = max(4, int(np.sqrt(target_params * m / n)))
            cp_n = max(4, int(np.sqrt(target_params * n / m)))

            # Ensure we don't exceed original dimensions
            cp_m = min(cp_m, m)
            cp_n = min(cp_n, n)

        # The harmonic collapse
        control_points = self.parallel_spline_synthesis(W, (cp_m, cp_n))

        # parallel_spline_synthesis clamps and pads, so the grid it returns is
        # not always the grid it was asked for. Report what actually came back
        # rather than what was requested, and hold it to an explicit request.
        actual_grid = tuple(control_points.shape)
        if control_grid is not None and actual_grid != tuple(control_grid):
            raise ValueError(
                f"requested control grid {tuple(control_grid)} but synthesis "
                f"produced {actual_grid}; the weight matrix {tuple(W.shape)} "
                f"cannot support that grid"
            )
        
        # Compute compression ratio
        original_params = W.numel()
        compressed_params = control_points.numel()
        compression_ratio = original_params / compressed_params if compressed_params > 0 else 1.0
        
        result = {
            'control_points': control_points.cpu(),  # Move back to CPU
            'original_shape': original_shape,
            'compression_ratio': compression_ratio,
            'control_grid': actual_grid,
            'weight_shape': W.shape  # Store the 2D shape used for processing
        }
        
        # Add bias if present
        if hasattr(layer, 'bias') and layer.bias is not None:
            result['bias'] = layer.bias.data.cpu()
        
        return result
    
    def convert_network(self, model: nn.Module, 
                       control_ratio: float = 0.1,
                       parallel: bool = True) -> Dict[str, Dict]:
        """
        Convert entire network to Neural Splines representation.
        The full harmonic collapse of artificial neurons to mathematical curves.
        """
        model = model.to(self.device)
        spline_model: Dict[str, Dict] = {}
        
        layers_to_convert = [
            (name, module) for name, module in model.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))
        ]
        
        if not layers_to_convert:
            print("No convertible layers found in the model.")
            return spline_model
        
        if parallel and len(layers_to_convert) > 1:
            # Parallel conversion of all layers
            with ThreadPoolExecutor(max_workers=min(4, len(layers_to_convert))) as executor:
                futures = {
                    executor.submit(self.convert_layer, layer, control_ratio): name
                    for name, layer in layers_to_convert
                }
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        spline_model[name] = future.result()
                        print(f"Converted {name}: compression {spline_model[name]['compression_ratio']:.1f}x")
                    except Exception as e:
                        print(f"Failed to convert {name}: {e}")
                        import traceback
                        traceback.print_exc()
        else:
            # Sequential conversion
            for name, layer in layers_to_convert:
                try:
                    spline_model[name] = self.convert_layer(layer, control_ratio)
                    print(f"Converted {name}: compression {spline_model[name]['compression_ratio']:.1f}x")
                except Exception as e:
                    print(f"Failed to convert {name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        return spline_model
