# Experiments

Evidence behind item C0 in [`../REMEDIATION.md`](../REMEDIATION.md): whether
`HarmonicCollapseConverter`'s approach can work, and what the spline
representation is actually capable of.

Each script writes the `.log` beside it. All run on CPU. They download MNIST to
a hard-coded scratch path — change `D` at the top before running.

> **Read this first.** The first three scripts produced two conclusions that
> the later four *overturned*. Both the wrong and the corrected results are
> kept here deliberately, because the corrections are the useful part.

| script | question | verdict |
| --- | --- | --- |
| `coordnet.py` | Would a richer weight generator beat the fixed spline basis? | superseded by `rt_siren.py` |
| `objective.py` | Does the fitting objective matter more than the representation? | superseded by `rt_objective.py` |
| `objective_control.py` | Is the gain the spline, or just retraining the head? | stands |
| `rt_siren.py` | Was the SIREN result an artifact of training budget? | **yes — it was** |
| `rt_objective.py` | Is the Frobenius fit optimal? Does the teacher matter? | **teacher does not matter** |
| `rt_param.py` | Is the accuracy plateau an artifact of the parameterisation? | **yes — it is** |
| `rt_aspect.py` | Is the binding constraint rank, or input-axis resolution? | **resolution** |

---

## 1. The Frobenius objective genuinely cannot work

`W_hat = A C B^T` is *linear* in `C`, so `||A C B^T - W||_F` is a convex
quadratic with a unique global minimum and closed form `C* = A⁺ W (B⁺)^T`.
LBFGS matches that closed form to five decimals at every grid tested:

| grid | LBFGS | closed form | gap |
| --- | ---: | ---: | ---: |
| 4x4 | 0.9978 | 0.9978 | +0.00000 |
| 16x16 | 0.9946 | 0.9946 | -0.00000 |
| 32x64 | 0.9805 | 0.9805 | +0.00000 |

So the converter's ~98% reconstruction error is a *provable ceiling* for its
objective, not an optimiser failure. This is the one original conclusion that
survived red-teaming intact.

## 2. Distillation does not rescue it either — nothing is being compressed

`objective.py` appeared to show that swapping the loss for distillation took the
same spline grid from 23.89% to 92.37%. The control in `rt_objective.py` shows
the teacher contributes **nothing**:

| grid | head | init | objective | accuracy |
| --- | --- | --- | --- | ---: |
| 32x64 | dense | teacher | distill | 94.45% |
| 32x64 | dense | teacher | labels | 94.55% |
| 32x64 | dense | **random** | **labels** | **94.34%** |
| 32x64 | spline | random | labels | 94.76% |

A randomly initialised student trained on labels alone, never seeing the
teacher, matches distillation. There is no compression happening — this is
simply training a small model. **Any claim that the converter can be fixed by
changing its objective is wrong**, and an earlier revision of
`../REMEDIATION.md` said exactly that. It was incorrect.

`objective_control.py` separately shows that retraining the head on top of the
Frobenius-fitted grid already recovers 23.89% -> 87.38%, so most apparent
"compression success" is downstream adaptation.

## 3. A coordinate network CAN beat the spline — the first result was undertrained

`coordnet.py` gave every method 3,000 steps and concluded SIREN was worse than
the spline. That was a training artifact. The same SIREN, same budget, run
longer:

| w0 | 1k | 3k | 6k | 10k | 20k |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 0.9606 | 0.9313 | 0.8944 | 0.8606 | **0.8171** |
| 100 | 0.9477 | 0.9802 | 0.9999 | 1.0001 | 0.8632 |
| 300 | 1.0025 | 1.0029 | 1.0021 | 1.0012 | 0.9130 |

At 3k it reproduces the original 0.963; by 20k it reaches 0.8171, **better than
the spline's 0.8909**, and is still falling. The `smooth` positive control in
`coordnet.py` did not catch this, because smooth targets are easy for SIREN —
it validated the harness only in the easy regime.

Two hypotheses were tested for the original failure and only one held. The
sweep above also refutes the second: `w0` sets SIREN's frequency prior, so a
low `w0` was suspected of imposing the same smoothness bias the spline suffers
from. Raising it makes things *worse* (0.8632 and 0.9130 against 0.8171), so
the standard `w0 = 30` was the right setting and **undertraining was the sole
cause**.

Still true from the original run: SVD remains best at 0.6924, and the CfC-style
arm is poor. That arm is weak evidence about CfCs regardless — a real CfC is a
continuous-time *sequence* model and a static weight matrix has no time axis;
this transplants only the gated closed-form interpolation, and it underperforms
even on the smooth control.

## 4. The real finding: the parameterisation, not the representation

`SplineMLP` uses a square `cp x cp` grid for every layer. That is a poor default
for a non-square weight matrix, and it — not any fundamental limit — produces
the plateau in the low 80s that the README used to report as structural.

| config | params | acc (5 epochs) |
| --- | ---: | ---: |
| repo default (32x32 + 32x32, spline biases) | 2,112 | 84.05% |
| + full biases | 2,314 | 84.11% |
| + wide L1 grid (16x128) | 3,338 | **95.32%** |
| + per-class head (10x64) | 2,954 | 95.37% |
| 32x64 L1 + per-class head | 2,954 | **95.76%** |

**It is input-axis resolution that binds, not rank.** Holding the parameter
count fixed at 2,954 and varying only the aspect ratio:

| L1 grid | rank ≤ | cp across the 784 inputs | acc |
| --- | ---: | ---: | ---: |
| 8x256 | 8 | 256 | 92.03% |
| 16x128 | 16 | 128 | 95.37% |
| **32x64** | **32** | **64** | **95.76%** |
| 45x45 | 45 | 45 | 93.23% |
| **64x32** | **32** | **32** | **85.25%** |
| 128x16 | 16 | 16 | 84.84% |
| 256x8 | 8 | 8 | 78.21% |

`32x64` and `64x32` have **identical rank and identical parameter count** and
differ by 10.5 points. Rank-16 (`16x128`) beats rank-45 (`45x45`). The rank
bound `rank(W) <= min(cp_h, cp_w)` is real but is not what limits accuracy at
useful budgets; what limits it is how finely the grid can vary across the input
dimension. That is consistent with the permutation result — shuffling input
pixels costs the model 18.9 points — since both say the layer's leverage comes
from structure along the input axis.

The best configuration measured, **95.76% at 2,954 parameters against a dense
203,530-parameter baseline at ~97.8%**, is a 69x reduction for ~2 points. That
is a genuinely useful result, and it is reachable only by *not* using
`SplineMLP`'s square-grid default.

## What this implies

- The converter's objective is unfixable, and distillation does not rescue it,
  because there is nothing there to compress in the first place.
- The spline *layer* is worth keeping. Its weakness was never the fixed basis;
  it was a default grid shape that starves the input axis.
- `SplineMLP`'s constructor cannot express a non-square grid at all (it passes
  `cp_hidden` to both axes). Use `SplineLinear` directly until that is fixed —
  tracked as item C6.
