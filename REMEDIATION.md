# Remediation Tracker

Findings from the repository audit and adversarial review. Licensing is
tracked separately and deliberately excluded here.

**Status key:** `open` · `done` · `needs decision` · `won't fix`

Each item marked `done` has been fixed, independently red-teamed, and
verified by running the code — not by inspection alone.

**Summary: 25 done · 1 needs a decision · 1 won't fix.**
No open work items. C0 is decided — the converter is kept, with its limitation
documented at the point of use. The remaining decision is B3 (asset size).

Licensing was tracked separately and is now resolved: the project is aligned on
AGPL-3.0-or-later, matching the `LICENSE` file and the source headers. The
packaging metadata had declared MIT, which advertised rights the project does
not grant.

> Several conclusions in this document were **overturned by red-teaming their
> own evidence**, and the corrections are recorded rather than quietly edited
> away. In particular: "coordinate networks do not help" was an undertraining
> artifact, "replace the converter's objective with distillation" was refuted by
> a control, and "the accuracy gap to dense is structural" was an artifact of a
> square-grid default. See `experiments/README.md`.

---

## 1. The converter

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| C0 | Decide: cut the converter, or keep it | — | High | `done` | **Decided: keep it.** Kept as a reference implementation and a documented negative result, with the limitation stated where a user will meet it — the class docstring, the module docstring, and here — rather than only in `experiments/`. The evidence is unchanged and is not softened by the decision: (1) the Frobenius objective sits at a *provable* ceiling, since `W_hat = A C B^T` is linear in `C` and LBFGS matches the closed-form optimum `A⁺W(B⁺)ᵀ` to five decimals, so ~98% reconstruction error cannot be optimised away; (2) changing the objective does not rescue it either — a randomly initialized student trained on labels alone, never seeing the teacher, matches distillation (94.34% vs 94.45%), so nothing is being compressed. An earlier revision of this row proposed "replace the objective with distillation" — **that was wrong**, and `experiments/rt_objective.py` is the control that refuted it. The bugs C1-C5 are fixed regardless, so what is kept is correct code with an honest description. |
| C6 | `SplineMLP` cannot express a non-square control grid | `neural_spline.py` (`SplineMLP.__init__`) | **High** | `done` | **Found while red-teaming C0, and it was the most valuable finding here.** `SplineMLP` passed `cp_hidden` to *both* axes, forcing a square grid on non-square weight matrices. `32x64` and `64x32` have identical rank and identical parameter count yet differ by 10.5 points, so the constraint is input-axis resolution, not the rank bound. Fixed: `cp_hidden`/`cp_output` now accept `(cp_h, cp_w)` (an `int` still means square, so nothing breaks), plus `aspect_grid()`, `SplineMLP.with_budget()` and a `--budget` flag on the training CLI. Verified on MNIST over two seeds at matched parameter counts: **93.88% at 2,594 params against 85.08% for the square `--cp 36` at 2,664** — +8.8 points for the same budget. |
| C1 | `torch.exp()` called on a Python float | `core/neural_spline.py` | High | `done` | Now `math.exp`. Verified: converter runs on every shape tested. |
| C2 | QR factor of non-square product assigned back to `V` | `core/neural_spline.py` | High | `done` | Both factors now applied so `V` keeps shape `(n, k)`. Verified against SVD: principal angles ~0°, Rayleigh quotients match true squared singular values. |
| C3 | `from_dense` grid contract mismatch | `neural_spline.py` | High | `done` | `convert_layer` takes an explicit `control_grid`, reports the grid actually produced, and refuses impossible requests. `state_dict` round-trip verified. |
| C4 | `nn.ModuleDict` rejects dotted names | `dense_to_neural_spline.py` | Med | `done` | Names mapped through `_module_key()`. Verified on a nested model. |
| C5 | Golden-ratio restart destroyed convergence | `core/neural_spline.py` | High | `done` | **Found by red-teaming the C1/C2 fix** — previously unreachable because the function always crashed. It replaced 38% of `V` with noise whenever resonance ticked up, which near convergence happens from float noise alone. Measured worse in every configuration; removed. |

## 2. Correctness and safety

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| S1 | `torch.load` without explicit `weights_only` | 3 sites | High | `done` | **Verified exploitable**: a proof-of-concept payload executed under the old call and does not under the new one. State-dict reads use `weights_only=True`; the full-module path refuses unless `--trust-checkpoint` is passed (exit code 1). |
| S2 | Two bare `except:` | `neural_spline.py` | Med | `done` | Typed exceptions; LBFGS failures are now reported instead of silently returning unoptimized points. |
| S3 | `load_spline_model` rebinds Parameter attribute | `dense_to_neural_spline.py` | Low | `done` | Shape-checked copy; mismatches raise at load. |
| S4 | Return annotation lies | `dense_to_neural_spline.py` | Low | `done` | Now `Tuple[nn.ModuleDict, Dict[str, Dict]]`. |
| S5 | `import *` leaks 11 names | `core/__init__.py` | Low | `done` | Explicit re-exports; verified no leakage. |

## 3. Packaging and metadata

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| P1 | `setup.py` contradicts `pyproject.toml` | `setup.py` | High | `done` | Deleted; `pyproject.toml` is the single source. Install verified from outside the repo. |
| P2 | Placeholder URLs and emails ship | `pyproject.toml` | Med | `done` | Real repository URL; invented readthedocs URL dropped. Verified none remain. |
| P3 | Author attribution inconsistent | `pyproject.toml` | Low | `done` | Attributed to Robert Sitton, matching every source header and every commit. |

## 4. Dead surface

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| D1 | Three console scripts that only print | `pyproject.toml` | Med | `done` | `[project.scripts]` removed; verified absent after reinstall. |
| D2 | `api.py` is three `NotImplementedError`s | `api.py` | Med | `done` | Documented, annotated, points at what works. |
| D3 | `models/`, `compression/`, `utils/` are stubs | those packages | Low | `done` | All now raise instead of returning `None`. `geometric_validation()` returning `None` was indistinguishable from a validation that passed. |
| D4 | `deepseek` extra with no consumer | `pyproject.toml` | Low | `done` | Retained but documented as having no consumer. |

## 5. Build, docs, assets

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| B1 | `CMakeLists.txt` builds a missing file | `CMakeLists.txt` | Med | `done` | Fails configuration with an explanation. Verified by running `cmake`. |
| B2 | PDFs unreviewed | `docs/*.pdf` | Med | `done` | Both read in full; findings below. |
| B3 | 5.1MB of assets | `assets/` | Low | `needs decision` | `profile.png` 3.2MB, `icon.png` 1.9MB. Already permanent in git history, so removing them from the working tree does not reclaim clone size — that needs a history rewrite, which is your call. |
| B4 | Commit `644c6b1` message | git history | Low | `won't fix` | Immutable without a history rewrite. |

## 6. Quality tooling

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| Q1 | mypy config strict, code untyped | `pyproject.toml` | Med | `done` | First run reported 30 errors; now clean across 18 files, and enforced by CI. Also corrected `requires-python` from `>=3.8` (never installable with the pinned torch floor) to `>=3.10`. |
| Q2 | No CI | `.github/workflows/ci.yml` | Med | `done` | pytest on Python 3.10-3.13 with CPU-only torch wheels, plus a mypy job. |
| Q3 | Coverage is smoke-only | `tests/` | Low | `done` | 39 tests. Mutation testing found the suite could not detect the bias interpolation mode changing from linear to nearest; that gap is closed. |
| Q4 | No streaming interpolation | `neural_spline.py` | Low | `done` | Better than streaming: bicubic interpolation is *separable*, so `W = A @ C @ B.T` and the dense matrix need never exist. `separable_forward` cuts allocation 868,400 B → 12,880 B and latency 324 us → 24.5 us, with outputs and gradients matching to ~1e-14 in float64 (gradcheck passes). |

---

## B2: what the papers actually claim

Both PDFs were read in full and the key claims verified directly against
the extracted text.

- **`neural_spline_paper.pdf`** (5pp) makes exactly one accuracy claim:
  *"achieves around 96 accuracy"* — literally, with **no percent sign** —
  for the recipe `--epochs 5 --batch-size 128 --cp 6 --hidden-size 256`.
  Measured for that exact recipe: **54.5%**. It gives no baseline, no
  seeds, no variance, no results table. Its one compression claim,
  *"replace tens of thousands of parameters with fewer than 100 control
  values"*, is prefaced *"in our experiments"* with no experiment described.
- **`neural spline draft.pdf`** (7pp) contains **no experiments and no
  numbers at all** — no dataset, no MNIST, no measurement.
- The README's former "95-97%" claim appears in **neither** document. It
  originated in the README.
- The draft's **Theorem 3** is the claim the measurements bear on: for any
  `W` *"with bounded second derivatives"* there exists a control-point
  configuration within a stated error bound. The document never argues
  that trained weight matrices satisfy that hypothesis, never bounds the
  quantity for any real network, and gives no proof. Of its 8 numbered
  results, only Theorem 1 carries a proof.
- Neither document claims a speedup or a peak-memory reduction, so the
  storage-only finding does not contradict them. The paper in fact states
  the dense weights are *"synthesize[d] on demand"* each forward pass.
- Neither document mentions input ordering or permutation.

---

## Evidence appendix

Measurements behind the items above, all reproducible on CPU.

**Accuracy (MNIST, `SplineMLP(784, 256, 10)`, Adam lr 1e-3, batch 128, seed 0):**

| `--cp` | params | 5 epochs | 40 epochs |
| -----: | -----: | -------: | --------: |
| 4 | 40 | 18.25% | - |
| 6 | 84 | 54.50% | - |
| 12 | 312 | 80.18% | 80.82% |
| 32 | 2,112 | 83.38% | 85.66% |
| *dense* | 203,530 | 97.75% | - |

Seed spread: 0.7pp at `--cp 12`, 1.3pp at `--cp 32`.

**Spline reconstruction of a trained 256x784 weight matrix**, control
points fitted optimally by LBFGS (far better than the repo's heuristic):

| budget | spline rel. error | optimal SVD rel. error |
| -----: | ----------------: | ---------------------: |
| 16 | 0.9982 | 0.9679 |
| 256 | 0.9946 | 0.9679 |
| 4096 | 0.9596 | 0.9114 |

Relative error ~1.0 means no better than predicting all zeros. The
repaired converter's own demo independently reports 0.978 on a 3-layer MLP.

**Runtime cost** (`--cp 6`, batch 1, CPU): checkpoint 2,781 B vs 816,349 B
dense (293x smaller). The default forward allocates 868,400 B and takes
324 us; `separable_forward` allocates 12,880 B and takes 24.5 us (Q4).

**Input-ordering dependence:** shuffling the 784 input pixels costs the
spline model 18.9 points (80.18% -> 61.27%) while leaving a dense MLP
unaffected (97.87% -> 98.01%).
