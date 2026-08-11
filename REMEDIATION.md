# Remediation Tracker

Findings from the repository audit and adversarial review. Licensing is
tracked separately and deliberately excluded here.

**Status key:** `open` · `in progress` · `done` · `needs decision` · `won't fix`

Each item marked `done` has been fixed, independently red-teamed, and
verified by running the code — not by inspection alone.

---

## 1. The converter

`HarmonicCollapseConverter` is the repository's advertised feature.
**C0 governs whether C1-C4 are worth fixing at all.**

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| C0 | Decide: repair, mark experimental, or cut | — | High | `needs decision` | Measured ~99% reconstruction error against a trained weight matrix *with optimally fitted control points*. Fixing C1-C4 yields a working program that still cannot do the advertised thing. |
| C1 | `torch.exp()` called on a Python float | `core/neural_spline.py:489` | High | `open` | `TypeError` on every call, every input. |
| C2 | QR factor of non-square product assigned back to `V` | `core/neural_spline.py:495` | High | `open` | Only works on square weight matrices; every real `nn.Linear` is non-square. |
| C3 | `from_dense` grid contract mismatch | `neural_spline.py:186-190` / `:711-712` | High | `open` | Requests 4x4, receives 4x7, assigned via `.data` so it is silent. Breaks `state_dict` round-trips later. |
| C4 | `nn.ModuleDict` rejects dotted names | `dense_to_neural_spline.py:111` | Med | `open` | `KeyError` on any nested model. |

## 2. Correctness and safety

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| S1 | `torch.load` without explicit `weights_only` | `dense_to_neural_spline.py:149,308`; `inference.py:105` | High | `open` | `:308` loads a full pickled object from a user-supplied path: arbitrary code execution on an untrusted checkpoint. |
| S2 | Two bare `except:` | `neural_spline.py:496,633` | Med | `open` | Swallow `KeyboardInterrupt`; `:633` hides LBFGS failure. |
| S3 | `load_spline_model` rebinds Parameter attribute | `dense_to_neural_spline.py:158-159` | Low | `open` | Should be `.data.copy_()`; bypasses shape validation. |
| S4 | Return annotation lies | `dense_to_neural_spline.py:64-65` | Low | `open` | Annotated `-> nn.Module`, returns a 2-tuple. |
| S5 | `from .neural_spline import *` leaks 11 names | `core/__init__.py:2` | Low | `open` | `torch`, `nn`, `F`, `np`, `ThreadPoolExecutor`, `as_completed`, and five `typing` names. |

## 3. Packaging and metadata

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| P1 | `setup.py` contradicts `pyproject.toml` | `setup.py:34-44` | High | `open` | Still lists all 7 unused deps. Delete or sync. |
| P2 | Placeholder URLs and emails ship | `pyproject.toml:79-83`, `setup.py:135-141` | Med | `open` | `your-username` x4, `example.com` x2, a likely-nonexistent readthedocs URL. |
| P3 | Author attribution inconsistent | headers / `pyproject.toml:11-16` / git config | Low | `open` | Three different answers. |

## 4. Dead surface shipped as real API

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| D1 | Three console scripts that only print | `cli.py`, `convert.py`, `visualization.py` | Med | `open` | Wired live in `pyproject.toml`; `pip install` yields three no-op commands. |
| D2 | `api.py` is three `NotImplementedError`s | `api.py` | Med | `open` | Named as the public API. |
| D3 | `models/`, `compression/`, `utils/` are stubs | those packages | Low | `open` | Exported from `__init__.py`, so they look real. |
| D4 | `deepseek` extra with no consumer | `pyproject.toml` | Low | `open` | The module it existed for never existed. |

## 5. Build, docs, assets

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| B1 | `CMakeLists.txt` builds a missing file | `CMakeLists.txt:64` | Med | `open` | `src/run_inference.cpp` absent; file is unbuildable. |
| B2 | PDFs unreviewed | `docs/*.pdf` | Med | `open` | May assert the same claims the measurements refute. |
| B3 | 5.1MB of assets | `assets/` | Low | `open` | `profile.png` 3.2MB, `icon.png` 1.9MB, permanent in history. |
| B4 | Commit `644c6b1` message | git history | Low | `won't fix` | "128x compression achieved... Same accuracy." on a commit containing no implementation. Immutable without a history rewrite. |

## 6. Quality tooling

| # | Item | Location | Sev | Status | Notes |
|---|---|---|---|---|---|
| Q1 | mypy config is strict, code is not typed | `pyproject.toml:164-177` | Med | `open` | 20 functions across 11 files would fail `disallow_untyped_defs`. |
| Q2 | No CI | — | Med | `open` | Nothing runs the tests. The original blockers reached `main` because nothing did. |
| Q3 | Coverage is smoke-only | `tests/` | Low | `open` | No coverage of the training loop, save/load round-trip, or `inference.py`. |
| Q4 | No streaming interpolation | `neural_spline.py:200-223` | Low | `open` | Would deliver the memory saving the README's premise implies. |

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

Relative error ~1.0 means no better than predicting all zeros.

**Runtime cost** (`--cp 6`, batch 1, CPU): checkpoint 2,781 B vs 816,349 B
dense (293x smaller), but 876,704 B allocated per forward vs 4,176 B
(no peak-memory benefit) and 443 us vs 8.9 us (50x slower).

**Input-ordering dependence:** shuffling the 784 input pixels costs the
spline model 18.9 points (80.18% -> 61.27%) while leaving a dense MLP
unaffected (97.87% -> 98.01%).
