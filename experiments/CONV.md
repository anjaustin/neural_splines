# SplineConv2d

Scripts: `conv_premise.py`, `conv_compare.py`, and the red-team follow-ups
`rt_conv_fair.py` (matched-compression comparison, controls, init scale),
`rt_conv_seeds.py` and `rt_conv_final.py` (seed replication, matched budgets).
Logs beside each.

> **This document has been substantially corrected.** The first version made
> two claims that adversarial review overturned — one too generous, one too
> harsh. Both corrections are recorded below rather than edited away.

## Summary

`SplineConv2d` reaches **99.34%** at 28,938 parameters, against **99.13%** for a
parameter-matched dense 3x3 and **99.37%** for a dense 9x9 using twice the
parameters. It gets large-kernel accuracy at small-kernel parameter cost.

It does **not** save computation. A 9x9 convolution performs ~9x the multiply-
accumulates of a 3x3 one no matter how its kernel is stored, and this layer
changes only the storage.

## Correction 1: the premise contrast was a compression-ratio artifact

The first version reported conv at **0.66** relative reconstruction error against
a fully connected weight matrix's **0.99**, and concluded the spline premise
"holds for convolution but fails for dense layers".

Those two numbers were taken at different compression ratios. The conv figure
came from a 4x4 grid on a 9x9 filter — **5x** compression. The dense figure came
from a 4,096-point grid on a 200,704-entry matrix — **49x**. Swept across the
same ratios, in closed form:

| compression | dense 256x784 | conv 9x9 stack |
| ---: | ---: | ---: |
| 5.06x | 0.7443 | 0.6708 |
| 3.24x | 0.6320 | 0.5568 |
| 2.25x | 0.5136 | 0.4492 |
| 1.65x | 0.3489 | 0.3437 |

At matched compression the two are close, and converge almost exactly at 1.65x.
Conv filters are modestly more spline-compressible than dense weights — not
categorically different. The dramatic contrast was the ratio, not the layer.

Note also that a 9x9 filter holds only 81 values and cubic interpolation needs a
4x4 grid, so **5x is the most compression a spline can extract from a 9x9
kernel at all**. The high-compression regime where dense weights fail outright
is simply not reachable here, so the two cases cannot be compared there.

## What survives: the spatial structure is real

The mechanism claim holds, and now has a proper null:

| conv 9x9 filters, cp=4 (5.06x) | rel. error |
| --- | ---: |
| trained filters | **0.6708** |
| same filters, 81 spatial taps shuffled | 0.8677 |
| random filters, matched std | **0.8944** |

Shuffling the taps preserves the value distribution exactly and destroys only
the spatial arrangement — and that alone moves reconstruction most of the way
to the random floor. So what the spline exploits in a conv kernel is genuine
spatial structure, which is the thing that does not exist along the arbitrarily
ordered axes of a fully connected layer.

## Correction 2: it does beat a 3x3, which the first version denied

The first version concluded "against a plain 3x3 convolution it does not [work]".
That rested on a single seed, and compared the wrong configuration (`cp=4`, a
smaller model). With three seeds and a parameter-matched baseline:

| configuration | params | s0 | s1 | s2 | mean | spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense 3x3 (c2=32) | 20,490 | 99.14 | 99.04 | 99.05 | 99.08 | 0.10 |
| dense 3x3 (c2=48, param-matched) | 30,650 | 99.16 | 99.12 | 99.12 | 99.13 | 0.04 |
| **spline 9x9 cp=5** | **28,938** | 99.40 | 99.30 | 99.31 | **99.34** | 0.10 |
| dense 9x9 | 58,506 | 99.38 | 99.43 | 99.30 | 99.37 | 0.13 |

* Against the **parameter-matched** 3x3 the spline wins by **0.21 points with
  non-overlapping seed ranges**, using 6% fewer parameters.
* Against **dense 9x9** the gap is **0.03** — comfortably inside noise — at 2.0x
  fewer total parameters and 3.2x fewer convolution weights.

So the layer delivers 9x9-kernel accuracy at roughly 3x3-kernel parameter cost.

## The cost it does not remove

Parameters are not FLOPs. A 9x9 convolution computes nine times the multiply-
accumulates of a 3x3 one; storing its kernel compactly does not change that.

| | latency (batch 1) | checkpoint |
| --- | ---: | ---: |
| `SplineConv2d` 9x9 cp=5 | 197.7 us | **53,157 B** |
| `.to_dense_conv()` of it | 68.1 us | 167,845 B |
| `nn.Conv2d` 9x9 | 63.2 us | 167,845 B |
| `nn.Conv2d` 3x3 | **22.9 us** | 20,389 B |

Two things to read from this. First, `to_dense_conv()` removes the
interpolation overhead entirely (68.1us against native 63.2us, bit-identical
outputs), so the deployment pattern is to ship the 3.2x smaller checkpoint and
densify on load. Second, even densified it remains ~3x slower than a 3x3,
because that is what a 9x9 convolution costs.

A caveat on an earlier claim: the training times in `conv_compare.py` showed
spline 9x9 at 162s and dense 9x9 at 163s — **identical**. The interpolation is
performed once per forward call and amortised across the batch, so at training
batch sizes it is free. The 3x figure above is a batch-1 inference artifact and
should not be read as a general overhead.

## Init scale

Interpolation is linear, so it rescales the control points' standard deviation
by a fixed factor that depends on the grid and kernel sizes. Initialising the
grid directly at the Kaiming scale produced kernels **1.48x** larger than
`nn.Conv2d`'s default. The layer now computes that gain and divides it out;
measured ratio is 1.00 across every kernel and grid size tested. The effect on
results was small — 99.32% before the fix against 99.34% after — but the layer
is now a genuine drop-in for `nn.Conv2d`.

## Verdict

Worth using when large kernels are wanted and parameters are the binding
constraint. Not worth using to save computation, which it does not do, and not
obviously worth it on a task where a 3x3 already suffices — though on this task
it did, in fact, win at matched parameters.
