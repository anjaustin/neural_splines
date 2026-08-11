# What actually gets a spline MLP to 98%

Scripts: `atomics.py` (isolation), `attribute.py` (multi-resolution control),
`fair_baseline.py` (dense references), `push98.py` (final configurations),
`seedcheck.py` (seed stability). Logs beside each.

All runs use AdamW + OneCycle, batch 128, seed 0 unless stated, MNIST test set.

## Answer

**98.56% at 23,130 parameters — 8.8x fewer than the dense baseline**, using a
single `108x188` control grid for layer 1, a dense 10-unit head, 40 epochs and
light affine augmentation. Mean over 3 seeds (98.51 / 98.51 / 98.67), spread
0.16pp; the tables below are seed 0 unless stated.

Two levers account for essentially all of it, and neither is exotic:

| lever | movement | cost |
| --- | --- | --- |
| **control-point budget** (at the right aspect ratio) | 95.95% -> 98.21% | 2,040 -> 20,304 control points |
| **training recipe** (AdamW + OneCycle, 15 epochs) | 93.88% -> 95.95% | free — no architecture change |
| augmentation, at a *long* schedule | 98.22% -> 98.56% | free |

## Correcting the dense baseline

The `~97.8%` figure quoted elsewhere in this repository came from plain Adam,
5 epochs, no augmentation. That understates the dense architecture. Under the
same recipe the spline models get:

| dense reference | params | acc |
| --- | ---: | ---: |
| 784-256-10, 15ep, no aug | 203,530 | 98.33% |
| 784-256-10 + BN, 40ep + aug | 204,042 | **99.09%** |
| 784-1024-10 + BN, 40ep + aug | 816,138 | 99.26% |

So **98% does not beat dense.** The honest claim is 8.8x fewer parameters for
about 0.5 points (98.56% against 99.09% at a matched recipe). That is a good
compression result, and it is a different claim from "matching dense".

## Isolation: one factor at a time

Reference: single 1D grid `34x60`, dense head, 15 epochs, 4,866 params, 95.95%.

| atomic | acc | delta | params |
| --- | ---: | ---: | ---: |
| 2D grid over (out, 28, 28), `34x8x8` | 96.72% | **+0.77** | +136 |
| 2D grid, `24x9x9` | 96.48% | +0.53 | -96 |
| multi-resolution (coarse + fine) | 98.10% | +2.15 | **+18,360** |
| BatchNorm | 95.89% | -0.06 | +512 |
| hidden 1024 | 96.09% | +0.14 | +8,448 |
| spline head instead of dense `fc2` | 95.72% | -0.23 | -1,920 |
| compressed bias (`cp_bias = cp_h`) | 95.95% | **0.00** | **-222** |
| augmentation (at 15 epochs) | 94.67% | -1.28 | 0 |

`compressed bias` is free: it costs nothing in accuracy and saves 222
parameters, so `cp_bias = cp_h` is the right default.

## Two findings that did not survive their controls

**Multi-resolution is a misattribution.** It looked like +2.15, but it also
spent 10x the control points. Pinned to ~20,400 control points:

| configuration | cp params | total | acc |
| --- | ---: | ---: | ---: |
| **single 1D `108x188`** | 20,304 | 23,130 | **98.21%** |
| multires 1D `34x60` + `102x180` | 20,400 | 23,226 | 98.10% |
| single 2D `60x18x18` | 19,440 | 22,266 | 97.82% |
| multires 2D `24x8x8` + `72x16x16` | 19,968 | 22,794 | 98.11% |

A single grid of the same size is *slightly better* than the multi-resolution
pair. The gain was the parameter count, not the structure.

**The 2D grid reverses sign with budget.** The input axis is raster order, so
pixels one row apart sit 28 indices apart and a 1D grid cannot express 2D
locality — which predicts 2D should win. It does at a small budget (+0.77 at
~5,000 params) and *loses* at a large one (97.82% against 98.21% at ~23,000).
At 20k parameters a `108x188` 1D grid resolves ~4 pixels per control point
directly, and spending the budget on output-axis resolution (108 rows against
60) beats spending it on input locality. So 2D is worth it only when the grid
is starved.

## Final configurations

| configuration | params | vs dense | acc |
| --- | ---: | ---: | ---: |
| single 1D `108x188`, 40ep, no aug | 23,130 | 8.8x | 98.22% |
| **single 1D `108x188`, 40ep + aug** | **23,130** | **8.8x** | **98.67%** (98.56% over 3 seeds) |
| + BatchNorm | 23,642 | 8.6x | 98.60% |
| `150x260` + BN, 40ep + aug | 42,338 | 4.8x | 98.67% |
| `108x188` + BN, hidden 512, 40ep + aug | 26,970 | 7.5x | 98.55% |

Note the saturation: **doubling the grid to 42,338 parameters buys nothing**
(98.67% either way), and neither BatchNorm nor a wider hidden layer helps. The
representation tops out near 98.7% in this architecture. Closing the last ~0.4
points to dense, or reaching 99%+, needs a different architecture rather than a
bigger grid.

Augmentation flips sign with schedule length: -1.28 at 15 epochs, +0.34 at 40
(98.22% -> 98.56% on seed-averaged numbers).

## Practical recommendation

```python
from neural_splines import SplineLinear

layer = SplineLinear(784, 256, cp_h=108, cp_w=188)   # 8.8x smaller, ~98.6%
```

or, letting the aspect ratio be chosen for you:

```python
from neural_splines import SplineMLP
model = SplineMLP.with_budget(784, 256, 10, budget_hidden=20304)
```

Train with AdamW + OneCycle for 40 epochs with light affine augmentation.
Keep `cp_bias` at its default, skip BatchNorm, and do not bother with
multi-resolution grids.
