# PE-MFPBO

Physics-embedded multi-fidelity Bayesian optimisation of polymer formulation and processing.

This repository contains the implementation accompanying the manuscript *Embedding Cure and
Crystallization Kinetics in Multifidelity Bayesian Optimization of Polymer Formulation and
Processing*. Formulation and thermal-history variables are searched as one design space. Classical
polymer relations (Fox, Halpin–Tsai, Williams–Landel–Ferry, autocatalytic cure, Avrami) supply a
learned warping of the input space, a physical mean function and thermodynamic sign constraints
shared across fidelity levels. A cost-aware Pareto front entropy criterion with a prior-disagreement
bonus selects batches through a determinantal point process, and risk-adjusted decoding returns sets
of admissible recipes.

## Contents

| Component | Manuscript | Module |
|---|---|---|
| Mixed design space, nested initial design, budgeted loop with the nesting rule | §3.1, Eqs. (1)–(7), (37) | `pemfpbo/utils/space.py`, `pemfpbo/runner.py` |
| Constitutive prior library | §3.2, Eqs. (8)–(15) | `pemfpbo/physics/library.py` |
| Physics warping map and mean bases, non-physical control map | Eqs. (16)–(17) | `pemfpbo/models/features.py` |
| Multi-output level GP: ARD × categorical embedding kernel, nonlinear autoregressive coupling, coregionalisation, derivative covariances | Eqs. (18)–(22), (24) | `pemfpbo/models/levelgp.py` |
| Virtual derivative sites with probit likelihood and expectation propagation | Eqs. (23), (25) | `pemfpbo/models/ep.py` |
| Multi-fidelity surrogate: MAP fit with prior anchors, trust score and annealing, augmented predictive moments, posterior sample paths | Eqs. (26)–(30) | `pemfpbo/models/mfgp.py` |
| Pareto front entropy search, cost-aware criterion with disagreement bonus, feasibility weighting, DPP batches | Eqs. (31)–(36) | `pemfpbo/acquisition/core.py`, `pemfpbo/methods/pemfpbo.py` |
| CVaR under input perturbation, attainment probability, robust inverse set | Eqs. (38)–(39) | `pemfpbo/robust/cvar.py` |
| Hyperparameter-frozen variant used in Appendix A | Appendix A | option `freeze_after` |
| Case Study I, copper clad laminate (Table 2) | §4.3 | `pemfpbo/problems/laminate.py` |
| Case Study II, talc-filled polypropylene (Table 5) | §4.4 | `pemfpbo/problems/polypropylene.py` |
| MF-ZDT1, MF-ZDT3, MF-DTLZ2 and CureWarp-4 | §4.2 | `pemfpbo/problems/synthetic.py` |
| Residual-network emulator of the level-three solver | §4.1 | `pemfpbo/problems/emulator.py` |
| Ten baselines | §4.1 | `pemfpbo/methods/baselines.py` |
| Ablation variants A1–A10 | §4.6 | `pemfpbo/methods/pemfpbo.py` |
| Metrics (normalised HV, IGD+, cost to target), Wilcoxon + Holm, A12 | §4.1 | `scripts/summarize.py` |

## Installation

Python 3.9 or later.

```bash
pip install -r requirements.txt
pip install -e .
```

The implementation uses NumPy and SciPy only; no GPU is required.

## Quick start

```bash
python -m unittest discover -s tests
python scripts/reference_front.py --problem mf_zdt1 --n 4000
python scripts/run_experiment.py --problem mf_zdt1 --methods PE-MFPBO,MF-OSEMO --seeds 0-2 \
    --profile smoke --budget 8
python scripts/summarize.py --problem mf_zdt1
```

## Reproducing the numerical study

`scripts/run_all.sh` runs every step of Section 4 in order and can be resumed (existing runs are
skipped):

1. `train_emulator.py`: emulator of the level-three solver (512 Latin hypercube training analyses,
   128 held out; five-fold cross-validated R² and held-out NRMSE are written to
   `results/<problem>/emulator_report.json`).
2. `reference_front.py`: reference front from a 20 000-point Sobol sweep of the highest fidelity.
3. `run_experiment.py`: methods × seeds; `--methods all | baselines | ablations | <list>`.
4. `run_sensitivity.py`: the disagreement-weight × cost-exponent grid of `configs/sensitivity.yaml`.
5. `summarize.py`: per-run and aggregated metrics (`runs.csv`, `summary.csv`, `summary.md`).
6. `robust_decode.py`: robust inverse set, representative designs and verification with the
   noise-free level-three solver (`robust_designs.csv`).
7. `plot_figures.py`: Figures 2–5.

Budgets and nested initial designs are in `configs/problems.yaml`
(Case Study I: 120 HFE with 40/12/4 initial points; Case Study II: 90 HFE with 32/10/3).

### Profiles and run time

`configs/profiles.yaml` defines computational profiles; `--set key=value` overrides any option.

| Profile | Sample fronts | Candidate pool | Active points | Virtual sites |
|---|---|---|---|---|
| `paper` (Section 4.1) | 64 | 4096 | 256 | 96 |
| `fast` | 16 | 1024 | 96 | 48 |
| `smoke` | 8 | 256 | 48 | 16 |

With the `paper` profile on a single CPU core, a PE-MFPBO iteration takes a few seconds and a
hyperparameter refit about ten seconds at the start of a run; both grow with the active-set size.
Because cheap levels dominate the allocation, a full 120 HFE run involves on the order of a thousand
iterations. Seeds and methods are independent processes and can be distributed across nodes.

### Option reference

Model options (`ModelConfig`): `warp` (`physics`, `identity`, `random`), `learn_warp`, `mean`
(`physics`, `constant`, `linear`), `monotone`, `coupling` (`nonlinear`, `linear`), `icm_rank`,
`single_fidelity`, `n_active`, `n_fit`, `n_virtual`, `probit_scale`, `anneal`, `gamma_beta`,
`gamma_theta`, `fit_maxiter`, `n_rff`.

Acquisition options (`AcqConfig`): `q`, `n_samples`, `n_candidates`, `kappa`, `eta`, `refit_every`,
`batch` (`dpp`, `random`, `topq`), `ea_generations`, `ea_offspring`, `level_corr`, `freeze_after`.

## Implementation notes

* **Fidelity levels.** Level one evaluates the constitutive library with handbook parameters. The
  level-two and level-three solvers in this repository are reduced-order Python solvers (lumped and
  one-dimensional implicit thermo-chemical solutions with diffusion-controlled kinetics,
  DiBenedetto glass transition and Prony-series stress for the laminate; lumped and one-dimensional
  cooling with latent heat and a skin–core description for polypropylene). They keep the structure
  of the hierarchy (systematic level-one bias, model-form and discretisation differences, the cost
  model of Table 3) but they are not the RVE homogenisation and board-level finite element models
  described in the manuscript. The finite element solver and its datasets are not part of this
  repository, so the numbers produced here are not expected to match the tables of the manuscript.
* **Synthetic benchmarks.** MF-ZDT1, MF-ZDT3 and MF-DTLZ2 use the standard test functions at level
  three with smooth, objective-specific bias added at lower levels (`pemfpbo/problems/synthetic.py`).
* **Surrogate.** Rows of every level GP are (design, objective) pairs, so coregionalisation,
  autoregressive coupling and derivative observations share one covariance. Training inputs of
  level `l` take the observed level `l-1` outputs (available through the nesting rule); prediction
  inputs take the posterior mean with first-order variance propagation. Hyperparameters, mean
  coefficients and warping parameters are fitted jointly by L-BFGS-B with analytic gradients
  (sensitivities of the warped coordinates to the warping parameters by finite differences) on a
  nested space-filling subset of `n_fit` points per level; the posterior uses an active set of
  `n_active` points per level, which plays the role of the inducing set.
* **Mean coefficients** start from the library calibration and are refined by ridge regression on
  level-one data, then on the data of each level (Section 3.2). The two prior weights of Eq. (27)
  are selected by held-out predictive likelihood on the level-one initial design.
* **Sign constraints.** Each constraint is a (objective, warped coordinate, sign, design variable)
  tuple. The contribution of the physical mean to the derivative is obtained by a finite
  difference along the named design variable. With the identity map (A1) the constraint applies to
  the raw variable; with the non-physical map (A9) the coordinates carry no physical meaning and the
  constraints are not applied.
* **Sample paths** (Eq. 30) are random Fourier feature prior draws corrected by a pathwise update
  through the exact augmented posterior, so that virtual derivative sites enter on the same footing
  as data.
* **Trust score.** The reference of Eq. (28) is the level-one library prediction; its prior
  variance is the top-level prior variance.
* **Baselines** reproduce the defining surrogate and acquisition principle of each cited method on
  the shared infrastructure (same initial design, candidate pool, inner evolutionary solver, GP
  code and cost accounting). They are re-implementations, not the authors' original code.
* **Metrics.** Hypervolume is computed from feasible level-three observations only, in a space
  normalised by the ideal and nadir of the reference front, with the reference point set to 1.1
  times the worst normalised value over all runs of all methods and then frozen.

## Repository layout

```
pemfpbo/
  physics/library.py        constitutive relations, Eqs. (8)-(15)
  problems/                 case studies, synthetic benchmarks, emulator
  models/                   level GP, feature maps, EP, multi-fidelity surrogate
  acquisition/core.py       PFES, CoPF-ES, DPP, EI, MC-EHVI, HVKG, inner solver
  methods/                  PE-MFPBO, ablations, baselines
  robust/cvar.py            CVaR decoding
  runner.py                 archive, nested design, budgeted loop
scripts/                    command-line entry points
configs/                    budgets, profiles, sensitivity grid
tests/                      unit tests
```

## License

MIT, see `LICENSE`.
