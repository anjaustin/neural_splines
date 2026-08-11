# SplineConv2d: where the spline premise finally holds

Scripts: `conv_premise.py` (can a spline represent a *trained* filter?),
`conv_compare.py` (from-scratch accuracy at matched budgets). Logs beside each.

## Summary

The spline premise, which fails outright for fully connected weights, **holds
for convolution**. The layer works and compresses conv weights ~3.2x for about
0.1 accuracy points.

It nevertheless **does not beat a plain 3x3 convolution**, which is what anyone
would actually reach for on this task. Both claims are below.

## The premise holds — and the control says why

Fitting control points to a *trained* weight, in closed form (the interpolation
is linear in the control points, so this is the provable optimum, not a search):

| target | cp | stored/dense | rel. error |
| --- | ---: | ---: | ---: |
| fully connected weight matrix | — | 0.20x | **0.99** |
| conv1 (1->16, 9x9) | 4 | 0.20x | 0.7126 |
| conv2 (16->32, 9x9) | 4 | 0.20x | 0.6622 |
| conv2 (16->32, 9x9) | 5 | 0.31x | 0.5438 |
| conv2 (16->32, 9x9) | 7 | 0.60x | **0.3369** |
| **conv2 with the 81 taps shuffled** | 4 | 0.20x | **0.8690** |

Relative error ~1.0 means no better than predicting zeros, which is where the
fully connected case sits at every budget. Conv filters are different: at 5x
compression a spline captures roughly a third of the filter's energy.

The last row is the control. Permuting the 81 spatial taps keeps the value
distribution exactly and destroys only the spatial arrangement — and
reconstruction degrades from 0.6622 to 0.8690. So the gain comes from genuine
**spatial structure**, not from anything about the numbers themselves. That is
the same test that condemned the fully connected case, run the other way round.

## Why the axes matter

A conv weight is `(out_channels, in_channels, kH, kW)`. `SplineConv2d`
interpolates **only the spatial axes** and leaves the channel axes alone,
because channel ordering is arbitrary — a network is invariant to permuting
channels, so channel *c* and *c+1* have no reason to resemble each other.
Interpolating across an arbitrarily-ordered axis is precisely the mistake that
costs ~10 accuracy points in the dense case (`ATOMICS.md`).

The saving is `(kH*kW) / (cp_h*cp_w)`. Cubic interpolation needs 4 control
points per axis, so there is nothing to gain below a 5x5 kernel; the layer
raises `ValueError` if the grid would not actually compress.

## From scratch: the honest comparison

Same architecture (conv -> ReLU -> pool -> conv -> ReLU -> pool -> linear),
AdamW + OneCycle, 8 epochs, seed 0.

| configuration | total params | conv weights | acc | train |
| --- | ---: | ---: | ---: | ---: |
| **dense 3x3** (the real baseline) | 20,490 | 4,752 | 99.14% | 56s |
| dense 9x9 | 58,506 | 42,768 | **99.38%** | 163s |
| spline 9x9 cp=4 | 24,186 | 8,448 | 99.17% | 162s |
| **spline 9x9 cp=5** | 28,938 | 13,200 | 99.27% | 163s |
| spline 13x13 cp=4 | 24,186 | 8,448 | 99.03% | 316s |
| dense 13x13 | 104,970 | 89,232 | 99.28% | 319s |

**Against the same kernel size, the layer does its job.** `spline 9x9 cp=5`
reaches 99.27% against dense 9x9's 99.38% — 3.2x fewer conv weights (13,200
against 42,768) for 0.11 points.

**Against a 3x3 convolution, it does not.** Dense 3x3 reaches 99.14% with fewer
total parameters (20,490 against 28,938) and a third of the training time.
`spline 9x9 cp=4` is +0.03 points over it for 18% more parameters and 2.9x the
training time — a wash on accuracy and a loss everywhere else.

Larger kernels do not rescue it: 13x13 is *worse* than 9x9 at the same control
budget (99.03% against 99.17%) and twice as slow.

## Inference cost, and how to remove it

Batch 1, CPU, a single 16->32 9x9 layer:

| | latency | checkpoint |
| --- | ---: | ---: |
| `SplineConv2d` 9x9 cp=5 | 197.7 us | **53,157 B** |
| `.to_dense_conv()` of it | **68.1 us** | 167,845 B |
| `nn.Conv2d` 9x9 | 63.2 us | 167,845 B |
| `nn.Conv2d` 3x3 | 22.9 us | 20,389 B |

Interpolating every forward costs about 3x. `to_dense_conv()` removes it
entirely — 68.1 us against native dense's 63.2 us, with bit-identical outputs —
so the deployment pattern is: ship the small checkpoint, densify on load. The
storage saving survives; the latency cost does not have to.

Note that dense 3x3 is still smaller *and* faster than the densified 9x9, which
is the same conclusion as above from a different direction.

## Verdict

`SplineConv2d` is a legitimate compression for **large-kernel** convolutions,
and it is the one place in this repository where the neural-spline idea is
supported by evidence rather than contradicted by it. Its value depends
entirely on whether large kernels are warranted for the task. On MNIST they are
not — 3x3 is sufficient and cheaper on every axis.

Where it would plausibly pay: architectures that genuinely want large receptive
fields per layer (7x7 and up, as in ConvNeXt-style or RepLKNet-style designs),
or settings where checkpoint size is the binding constraint and inference can
densify at load. Neither is demonstrated here, and neither should be claimed
until it is measured.
