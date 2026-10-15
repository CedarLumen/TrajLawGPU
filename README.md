# TrajLawGPU

Recovering the closed-form interaction law between particles from position trajectories alone, with the
whole pipeline running as tensor operations on the GPU (PyTorch, CUDA Graph replay).

Repository: <https://github.com/CedarLumen/TrajLawGPU>

This is the reference implementation behind the accompanying manuscript, together with every measured output
it uses. The reports are written by the scripts themselves, and the accuracy claims are checked by the
self-test suite rather than asserted.

## Layout

```
tl/        core library: force laws, simulator, design matrices, sparse fitting, estimators, thermodynamics
exp/       E0-E7 experiments (thin drivers; all logic lives in tl/)
tests/     equivalence with the reference implementation, sentinels, defect evidence, PBC checks, full rerun
bench/     cost accounting per integration step and CUDA Graph comparison
reports/   measured outputs, written by the scripts as UTF-8
```

## Requirements

- Python 3.11 and PyTorch with CUDA (developed on an RTX 5060 Laptop, 8 GB, sm_120)
- numpy

## Running

```text
python tests/test_parity.py         # item-by-item equivalence with the reference implementation, incl. the three CUDA Graph paths
python tests/test_sentinels.py      # quantities that must be exactly zero or reach machine precision
python tests/test_bug_evidence.py   # defect evidence, incl. measured memory
python tests/test_lj_pbc.py         # independent recomputation of three-dimensional PBC with Lennard-Jones
python tests/report_aux.py          # coverage by observation regime and per-truth convergence order
python tests/verify_all.py --run    # rerun every experiment and compare the numbers against the reports

python exp/e0.py                    # simulator self-check         ~33 s
python exp/e1.py                    # identifiability              ~18 s
python exp/e2.py                    # estimator gain               ~5 s
python exp/e3.py                    # representation comparison    ~2 min
python exp/e4.py                    # known-term ablation          ~52 s
python exp/e5.py                    # migration                    ~9 s
python exp/e6.py                    # many-body test               ~12 s
python exp/e7_argon.py              # liquid-argon inversion       ~4 min

python bench/step.py                # cost accounting for one integration step
python bench/graph.py               # CUDA Graph replay against eager execution
```

Switching the backend: `from tl import rt; rt.use("cpu")`.

## Notes

- The experiment scripts write their own reports; do not redirect their output with `>` in PowerShell 5.1,
  which produces UTF-16LE.
- `tests/verify_all.py` compares numeric tokens against the reports as sets, so report wording can change
  freely while the numbers stay pinned.
- `tests/test_parity.py`, and the reference comparison inside `tests/verify_all.py`, compare against the
  original numpy implementation. That implementation is not part of this repository; the archived comparison
  outputs are in `reports/`.
