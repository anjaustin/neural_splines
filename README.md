# Neural Splines for Resource-Constrained Deep Learning

This repository presents a self-contained example of training and
deploying a compressed neural network using spline interpolation.
Neural networks typically store one parameter per connection, which
quickly becomes burdensome on devices with limited memory.  A **neural
spline** represents an entire weight matrix with a much smaller grid
of control points and reconstructs the dense weights by bicubic
interpolation on the fly.  Fewer stored parameters mean much smaller
checkpoints, at a measurable cost in accuracy and, at inference time,
in latency and working memory (see [Results](#results) and
[Caveats](#caveats)).

The implementation is deliberately simple and educational.  All code
uses vanilla PyTorch and runs on CPU or GPU without any special
runtime.  The PDFs in `docs/` offer a concise introduction to the
theory of spline-based compression and explain how the provided code
works.  This project is intended as a study of an extreme point on the
compression/accuracy curve for researchers in resource-constrained
environments seeking to understand and extend neural splines.

## Overview

The project comprises two stages:

1. **Training a compressed model** on the MNIST digit dataset using
   spline layers (see `neural_splines/core/train.py`).  During training
   the network learns a small set of control points and compares its
   performance against a densified version of itself.
2. **Densifying and running inference** with standard PyTorch.  After
   training the spline network is converted to an equivalent dense
   model and saved.  The module `neural_splines/core/inference.py`
   loads this dense model and evaluates it on the MNIST test set.

## Getting Started

### Prerequisites

* A Python 3.8+ interpreter.
* Internet access to download PyTorch, torchvision and the MNIST dataset.

### Setup

On any platform, create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[examples]'
```

The library itself needs only `torch` and `numpy`.  The `examples`
extra adds `torchvision`, which is required by the MNIST training and
inference scripts but not by the spline layers.  Use `pip install -e .`
if you only need the library.

Alternatively, **on Ubuntu only**, `setup.sh` installs system packages
via `sudo apt-get`, creates `.venv`, and pins torch 2.3.1 /
torchvision 0.18.1:

```bash
bash setup.sh
```

`setup.sh` invokes `sudo` and is not portable to macOS or Windows;
prefer the manual steps above unless you are on Ubuntu.

### Training

To train the spline network on MNIST run:

```bash
python3 -m neural_splines.core.train --epochs 5 --batch-size 128 --cp 6 --hidden-size 256
```

The script downloads the MNIST dataset, constructs a network with
spline layers, trains it and writes two checkpoint files in
`./checkpoints`:

* `spline_model.pth` - the raw spline model containing control
  points.
* `dense_model.pth` - a densified version of the model where the
  interpolation has been baked into fixed weights.

After training the script prints the accuracy of both the spline and
dense models.  These are expected to be identical: densification
reproduces the spline model's outputs exactly, it does not recover
accuracy.

### Inference

After training you can evaluate the densified model using the provided
inference script.  The script reconstructs the dense architecture, loads
the saved weights and reports the test accuracy:

```bash
python3 -m neural_splines.core.inference \
    --model-path checkpoints/dense_model.pth \
    --input-size 784 --hidden-size 256 --output-size 10
```

The `input-size`, `hidden-size` and `output-size` arguments should
match the architecture used during training.  In the MNIST example
the input is 28x28 pixels (784 features) and the output has 10
classes.

## Results

Measured on the MNIST test set: `SplineMLP(784, 256, 10)`, 5 epochs,
Adam at lr 1e-3, batch size 128, seed 0, CPU.  `--cp` sets the number
of control points along **each** axis of both layers.

These numbers describe `SplineMLP` as shipped. They are *not* the limit
of the spline representation — see
[the square-grid section](#that-plateau-is-an-artifact-of-the-square-grid-not-a-real-limit)
below, where the same parameter budget reaches 95.76%.

| `--cp` | trainable params | test accuracy |
| -----: | ---------------: | ------------: |
|      4 |               40 |        18.25% |
|      6 |               84 |        54.50% |
|     12 |              312 |        80.18% |
|     16 |              544 |        80.57% |
|     24 |            1,200 |        83.02% |
|     32 |            2,112 |        83.38% |
| *dense baseline* |    203,530 |    **97.75%** |

Single run per row. Run-to-run spread across seeds is roughly 0.7pp at
`--cp 12` and 1.3pp at `--cp 32`, so the `--cp 12` and `--cp 16` rows
are not meaningfully different.  Training 8x longer helps only
marginally and does not close the gap:

| epochs | `--cp 12` | `--cp 32` |
| -----: | --------: | --------: |
|      5 |    80.18% |    83.38% |
|     10 |    80.92% |    85.43% |
|     20 |    81.10% |    84.94% |
|     40 |    80.82% |    85.66% |

Accuracy rises steeply out of the degenerate regime and then flattens
in the low-to-mid 80s, and training 8x longer does not close the gap.

### That plateau is an artifact of the square grid, not a real limit

**The tables above are not the ceiling of this representation.** `--cp N`
gives every layer an `N x N` control grid, and a square grid is a poor
fit for a weight matrix that is not square. Layer 1 here is 256x784, so
a 32x32 grid spreads only 32 control points across 784 input pixels.

Holding the parameter count fixed at 2,954 and changing *only* the grid
shape (5 epochs, otherwise identical):

| layer-1 grid | rank <= | control points across the 784 inputs | accuracy |
| --- | ---: | ---: | ---: |
| 8x256 | 8 | 256 | 92.03% |
| 16x128 | 16 | 128 | 95.37% |
| **32x64** | **32** | **64** | **95.76%** |
| 45x45 | 45 | 45 | 93.23% |
| **64x32** | **32** | **32** | **85.25%** |
| 128x16 | 16 | 16 | 84.84% |
| 256x8 | 8 | 8 | 78.21% |

`32x64` and `64x32` have the **same rank and the same parameter count**
and differ by 10.5 points; rank-16 beats rank-45. So the binding
constraint is **resolution along the input axis**, not the rank bound.

It is true that bicubic upsampling of a `cp_h x cp_w` grid yields a
weight matrix of rank at most `min(cp_h, cp_w)` — a 4x4 grid expanded to
256x784 measures **rank 4** — and that the basis is fixed rather than
learned. But at useful budgets that bound is not what limits accuracy;
how finely the grid can vary across the input dimension is. This is
consistent with the permutation result in [Caveats](#caveats): both say
the layer's leverage comes from structure along the input axis.

With a rectangular grid and a per-class output layer, **2,954
parameters reach 95.76% against the dense baseline's 203,530 and
~97.8%** — a 69x reduction for about 2 points.

### Use `--budget`, not `--cp`

`SplineMLP.with_budget` spends a control-point budget on each layer with
the aspect ratio chosen by :func:`aspect_grid`, instead of forcing a
square grid. From the CLI:

```bash
python3 -m neural_splines.core.train --budget 2048 --hidden-size 256 --epochs 5
```

Measured over two seeds, 5 epochs, at matched parameter counts:

| configuration | params | accuracy |
| --- | ---: | ---: |
| `--cp 32` (square) | 2,112 | 82.88% |
| `--cp 36` (square, param-matched) | 2,664 | 85.08% |
| **`--budget 2048`** (34x60 and 10x51) | 2,594 | **93.88%** |

**+8.8 points for the same parameter count**, purely from how the budget
is divided. A hand-tuned grid does slightly better still (95.76% for
32x64 with uncompressed biases), so `aspect_grid` is a good default
rather than an optimum — pass an explicit `(cp_h, cp_w)` tuple to
`SplineMLP` or use `SplineLinear` directly if you want to tune it.

Measurements in [`experiments/`](experiments/).

## Caveats

Two properties of this approach are easy to misread, so they are stated
explicitly.

**The saving is in storage, not in runtime memory.** A `SplineLinear`
layer must materialize the full dense weight matrix on *every* forward
pass, so it does not reduce peak inference memory - it only moves those
bytes from persistent storage to a transient allocation, and adds the
interpolation cost on top. Measured for `SplineMLP(784, 256, 10)` at
`--cp 6`, batch size 1, on CPU:

| | spline | dense |
| --- | ---: | ---: |
| checkpoint on disk | 2,781 B | 816,349 B |
| allocated per forward | 876,704 B | 4,176 B |
| latency | 443 us | 8.9 us |

So the default forward pass is ~293x smaller on disk but ~50x slower,
with no peak-memory benefit.

**This is fixable, and is fixed.** Bicubic interpolation is separable, so
the dense weight factors exactly as `W = A @ control_points @ B.T`, and
the output can be computed as `((x @ B) @ control_points.T) @ A.T`
without ever building `W`. Set `layer.separable_forward = True` to use
it:

| `SplineMLP(784, 256, 10)`, `--cp 6`, batch 1 | default | `separable_forward` |
| --- | ---: | ---: |
| allocated per forward | 868,400 B | 12,880 B |
| latency | 324 us | 24.5 us |

Same function to floating-point precision -- outputs and gradients agree
to ~1e-14 in float64, and two epochs of training give 78.77% vs 78.76%
while running about 40% faster. It is opt-in rather than the default
only so that existing numerics are unchanged.

**The layer assumes neighbouring rows and columns are related.**
Interpolation imposes smoothness along both weight-matrix axes, which
implicitly assumes that adjacent input features - and adjacent output
units - should have similar weights. For raster-ordered image pixels
that assumption partly holds; in general it does not. Shuffling the
784 input pixels with a fixed permutation costs the spline model 19
points of accuracy (80.18% -> 61.27% at `--cp 12`) while leaving an
equivalent dense MLP unaffected (97.87% -> 98.01%). Much of what the
spline layer buys on MNIST therefore comes from the spatial structure
of the input, and should not be expected to transfer to arbitrary
feature orderings or to layers whose unit ordering is arbitrary.

## Project Structure

```
neural_splines/
├── README.md                        # This document
├── setup.sh                         # Environment setup script (Ubuntu only)
├── CMakeLists.txt                   # ExecuTorch C++ runner (see note below)
├── pyproject.toml
├── docs/
│   ├── neural_spline_paper.pdf      # Theory and exposition
│   └── neural spline draft.pdf
├── neural_splines/
│   ├── core/
│   │   ├── neural_spline.py         # Spline layer and MLP implementation
│   │   ├── train.py                 # Training script for MNIST
│   │   ├── inference.py             # Dense model inference script
│   │   └── dense_to_neural_spline.py# Dense -> spline conversion (experimental)
│   ├── models/, compression/, utils/# Placeholder modules, not yet implemented
│   ├── api.py, cli.py               # Placeholder entry points
│   └── visualization.py, convert.py
├── tests/
│   └── test_smoke.py
└── checkpoints/ (created at runtime)
    ├── spline_model.pth
    └── dense_model.pth
```

Note that `CMakeLists.txt` builds `src/run_inference.cpp`, which is not
present in this repository; the C++ runner is not currently buildable.
The `api`, `cli`, `convert`, `visualization`, `models`, `compression`
and `utils` modules are placeholders that do not yet implement
functionality.

## Philosophy and Future Work

While this repository focuses on practical code, the underlying
concept arises from a much broader epistemological perspective.  The
documents in `docs/` distil the ideas presented in the original paper
into a concise narrative that bridges abstract notions of continuous
parameter spaces with implementable algorithms.  Readers are
encouraged to explore the theory to appreciate how neural splines can
serve as a bridge between discrete computation and continuous
knowledge representations.

Future improvements might include convolutional spline layers,
integration with quantization techniques, learned (rather than fixed)
interpolation bases to lift the rank ceiling described above, and
support for more complex datasets.  Contributions are welcome.

## Acknowledgements

This project draws inspiration from a wide range of literature on
model compression and functional approximation.  It was prepared
with care to empower researchers worldwide to experiment with
compressed neural models and to encourage further exploration into
transcendent parameter structures.
