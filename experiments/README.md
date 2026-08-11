# Experiments

Evidence behind item C0 in [`../REMEDIATION.md`](../REMEDIATION.md): whether
`HarmonicCollapseConverter`'s approach can work, and if not, what does.

Each script writes the `.log` beside it. All run on CPU; `coordnet.py` takes
about 5 minutes, the other two a couple of minutes each. They download MNIST
to a hard-coded scratch path — change `D` at the top before running.

| script | question |
| --- | --- |
| `coordnet.py` | Would a richer weight generator (SIREN, CfC-style) beat the fixed spline basis? |
| `objective.py` | Does the fitting *objective* matter more than the representation? |
| `objective_control.py` | Control for `objective.py`: is the gain the spline, or just retraining the head? |

## 1. A richer basis does not help (`coordnet.py`)

Relative reconstruction error against a trained `Linear(784, 256)` weight, at
matched parameter budgets:

| budget | spline | svd | siren | cfc-style |
| -----: | -----: | --: | ----: | --------: |
| 256 | 0.9956 | 0.9679 * | 0.9942 | 1.0157 |
| 1024 | 0.9885 | 0.9679 * | 0.9895 | 1.0076 |
| 4096 | 0.9634 | 0.9114 | 0.9809 | 0.9973 |
| 16384 | 0.8909 | **0.6924** | 0.9630 | 0.9907 |

\* over budget: rank-1 costs `m + n = 1040` parameters regardless. At 4096 and
16384 the SVD arm is *under* budget (3,120 and 15,600), so the last row is fair.

SIREN is worse than the spline; the CfC-style variant is worse than emitting
zeros at three of four budgets. Replacing the fixed basis with a learned
nonlinear one buys nothing.

**Controls make that interpretable.** At budget 16384:

| target | spline | svd | siren | cfc |
| --- | -----: | --: | ----: | --: |
| smooth surface (positive control) | 0.0022 | 0.0000 | 0.0023 | 0.0424 |
| **trained weight** | 0.8909 | 0.6924 | 0.9630 | 0.9907 |
| i.i.d. Gaussian (entropy ceiling) | 0.9587 | 0.9326 | 0.9938 | 0.9987 |

Both the spline and SIREN reach ~0.2% error on a genuinely smooth target, so
the harness works and nothing is undertrained. A trained weight matrix sits far
closer to the noise control than to the smooth one.

Caveat: the CfC arm is weak evidence about CfCs. A real CfC is a
continuous-time *sequence* model with recurrent state, and a static weight
matrix has no time axis; this transplants only the gated closed-form
interpolation. It underperforms even on the smooth control, which points at the
adaptation being poor rather than the idea being refuted.

## 2. The objective is what is broken (`objective.py`, `objective_control.py`)

Same spline grid, same budget, same target network — only the fitting
objective changes. `frob_only` is what the converter does today; `distill_cp`
trains **the same tensor** against the teacher's logits instead of its weights:

| grid | cp params | arm | trainable | accuracy |
| ---: | --------: | --- | --- | -------: |
| 4x4 | 16 | frob_only | none (cp fitted to W) | 10.96% |
| 4x4 | 16 | frob_then_head | head only, cp frozen | 53.42% |
| 4x4 | 16 | distill_cp | cp only, head frozen | 24.22% |
| 4x4 | 16 | distill_both | cp + head | 57.47% |
| 32x64 | 2048 | frob_only | none (cp fitted to W) | 23.89% |
| 32x64 | 2048 | frob_then_head | head only, cp frozen | 87.38% |
| 32x64 | 2048 | **distill_cp** | **cp only, head frozen** | **92.37%** |
| 32x64 | 2048 | distill_both | cp + head | 94.45% |

Teacher: 96.87%. Random chance: ~9.8%.

`frob_only` vs `distill_cp` is the clean comparison — identical representation,
identical trainable tensor, identical budget, only the loss differs:
**23.89% against 92.37%.** That is 98x compression of layer 1 (200,704 -> 2,048)
for 4.5 points of accuracy.

The control also shows something the naive comparison hid: retraining the head
on top of the *Frobenius* grid already recovers 23.89% -> 87.38%. Much of what
looks like compression success in the literature is downstream adaptation, which
is why real methods always fine-tune. The spline still contributes on top of
that (92.37% with the head frozen).

Note the reconstruction error of the good solutions is **greater than 1.0** —
they are further from the teacher's weights than zero is, while computing
nearly the same function. Reconstruction error does not merely mispredict
functional quality here, it inverts it.

## What this implies for the converter

The neural-spline representation is adequate. `parallel_spline_synthesis`
minimizing `‖W_hat - W‖` is not. Replacing that objective with distillation
against the source model's activations or logits turns the converter from
something that cannot work into something competitive.

One fairness note before treating this as "splines beat low-rank": the SVD arm
was fitted to reconstruct `W`, not distilled. A distilled low-rank
factorization would very likely improve in the same way, and has not been
measured here.
