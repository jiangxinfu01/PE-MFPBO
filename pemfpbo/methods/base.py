"""Method interface and shared model-based machinery."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..acquisition.core import feasibility_probability
from ..models.mfgp import ModelConfig, MultiFidelityGP
from ..problems.base import Problem
from ..runner import Archive, candidate_pool


@dataclass
class AcqConfig:
    q: int = 4
    n_samples: int = 64
    n_candidates: int = 4096
    kappa: float = 1.0
    eta: float = 0.10
    refit_every: int = 10
    batch: str = "dpp"            # dpp | random | topq
    ea_generations: int = 2
    ea_offspring: int = 64
    tol: float = 1e-10
    level_corr: float = 0.5
    freeze_after: float = 0.0     # >0: no hyperparameter refits after this cost (Appendix A variant)


class Method:
    name = "method"
    multi_fidelity = False

    def __init__(self, problem: Problem, seed: int = 0):
        self.problem = problem
        self.seed = seed
        self.rng = np.random.default_rng(10_000 + seed)

    def initialize(self, archive: Archive) -> None:
        pass

    def propose(self, archive: Archive, it: int) -> List[Tuple[np.ndarray, int]]:
        raise NotImplementedError

    def diagnostics(self, archive: Archive) -> Dict:
        return {}


def split_config(opts: Dict) -> Tuple[ModelConfig, AcqConfig]:
    mk = {f.name for f in fields(ModelConfig)}
    ak = {f.name for f in fields(AcqConfig)}
    unknown = set(opts) - mk - ak
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    return (ModelConfig(**{k: v for k, v in opts.items() if k in mk}),
            AcqConfig(**{k: v for k, v in opts.items() if k in ak}))


class ModelBased(Method):
    """Owns the objective surrogate, the constraint surrogate and the pool."""

    defaults: Dict = {}

    def __init__(self, problem: Problem, seed: int = 0, **opts):
        super().__init__(problem, seed)
        merged = dict(self.defaults)
        merged.update(opts)
        self.mcfg, self.acfg = split_config(merged)
        self.multi_fidelity = not self.mcfg.single_fidelity
        self.to_min = problem.senses.copy()
        self.to_max = -problem.senses
        self.model: Optional[MultiFidelityGP] = None
        self.cmodel: Optional[MultiFidelityGP] = None

    @property
    def levels(self) -> List[int]:
        return self.model.levels

    def _levels_used(self) -> List[int]:
        L = self.problem.n_levels
        return [L] if self.mcfg.single_fidelity else list(range(1, L + 1))

    def initialize(self, archive: Archive) -> None:
        data = archive.data(self._levels_used())
        Y = np.vstack([v[1] for v in data.values()])
        y_mu, y_sd = Y.mean(axis=0), np.maximum(Y.std(axis=0), 1e-9)
        self.model = MultiFidelityGP(self.problem, self.mcfg, y_mu, y_sd, seed=self.seed)
        if self.problem.J > 0:
            gdata = archive.data(self._levels_used(), constraints=True)
            Gs = np.vstack([v[1] for v in gdata.values()])
            g_mu, g_sd = Gs.mean(axis=0), np.maximum(Gs.std(axis=0), 1e-9)
            ccfg = ModelConfig(warp=self.mcfg.warp, learn_warp=False, mean=self.mcfg.mean, monotone=False,
                               coupling=self.mcfg.coupling, icm_rank=0,
                               single_fidelity=self.mcfg.single_fidelity, n_active=self.mcfg.n_active,
                               n_fit=self.mcfg.n_fit, fit_maxiter=min(self.mcfg.fit_maxiter, 50),
                               gamma_beta=1.0, gamma_theta=1.0, anneal=False)
            self.cmodel = MultiFidelityGP(self.problem, ccfg, g_mu, g_sd, seed=self.seed + 1,
                                          n_outputs=self.problem.J, fmap=self.model.fmap,
                                          fixed_p=self.model.p_warp)
            self.g_threshold = -g_mu / g_sd

    def update_models(self, archive: Archive, it: int) -> bool:
        levels = self._levels_used()
        self.model.set_data(archive.data(levels))
        refit = (it % self.acfg.refit_every == 0) or self.model.params[self.model.top] is None
        if self.acfg.freeze_after > 0 and archive.total_cost > self.acfg.freeze_after \
                and self.model.params[self.model.top] is not None:
            refit = False
        if refit:
            self.model.fit()
            obs = np.vstack([v[0] for v in archive.data(levels).values()])
            self.model.update_annealing(obs)
        else:
            self.model.condition()
        if self.cmodel is not None:
            if refit:
                self.cmodel.p_warp = self.model.p_warp.copy()
                self.cmodel.set_data(archive.data(levels, constraints=True))
                self.cmodel.fit()
            else:
                self.cmodel.set_data(archive.data(levels, constraints=True))
                self.cmodel.condition()
        return refit

    def feasibility(self, X: np.ndarray) -> np.ndarray:
        if self.cmodel is None:
            return np.ones(np.atleast_2d(X).shape[0])
        mu, var = self.cmodel.predict(X)
        return feasibility_probability(mu, np.sqrt(var), self.g_threshold)

    def pool(self, it: int) -> np.ndarray:
        return candidate_pool(self.problem, self.acfg.n_candidates, self.seed, it)

    def charged_costs(self) -> np.ndarray:
        c = self.problem.costs[np.array(self.model.levels) - 1]
        return np.cumsum(self.problem.costs)[np.array(self.model.levels) - 1] if self.multi_fidelity else c

    def top_observations(self, archive: Archive) -> Tuple[np.ndarray, np.ndarray]:
        L = self.problem.n_levels
        d = archive.data([L])
        return d[L] if L in d else (np.zeros((0, self.problem.space.d)), np.zeros((0, self.problem.M)))

    def lengthscale_features(self, X: np.ndarray) -> np.ndarray:
        m = self.model
        gp, p = m.gps[m.top], m.params[m.top]
        name = "log_ls" if "log_ls" in gp.layout else "log_ls_d"
        ls = np.exp(gp.layout.get(p, name))
        z, _ = m.fmap(X, m.p_warp)
        return z / ls

    def diagnostics(self, archive: Archive) -> Dict:
        m = self.model
        if m is None or m.params.get(m.top) is None:
            return {}
        out = {"theta": m.p_warp, "theta_names": m.fmap.problem.theta_names if m.fmap.warp == "physics" else [],
               "wbar_history": m.wbar_history, "gamma0": [m.gamma_beta0, m.gamma_theta0]}
        if m.M > 1:
            out["coregionalization"] = m.coregionalization(normalized=True)
        out["monotonicity_violation"] = monotonicity_violation(self, archive)
        return out


def monotonicity_violation(method: ModelBased, archive: Archive, n: int = 1024) -> Dict:
    """Share of candidates whose posterior-mean response has the inadmissible sign
    along each physically constrained input, inside and outside the sampled range."""
    prob, m = method.problem, method.model
    if not prob.monotone:
        return {}
    X = prob.space.sobol(n, seed=4242)
    Xtop, _ = method.top_observations(archive)
    mu0, _ = m.predict(X)
    out = {}
    for (obj, _, sign, raw) in prob.monotone:
        v = prob.space.variables[raw]
        step = 0.02 * (v.high - v.low)
        X1 = X.copy()
        X1[:, raw] = np.minimum(X[:, raw] + step, v.high)
        mu1, _ = m.predict(X1)
        wrong = sign * (mu1[:, obj] - mu0[:, obj]) < 0
        if Xtop.shape[0]:
            lo, hi = Xtop[:, raw].min(), Xtop[:, raw].max()
            outside = (X[:, raw] < lo) | (X[:, raw] > hi)
        else:
            outside = np.ones(n, dtype=bool)
        key = f"{prob.objective_names[obj]}|{v.name}"
        out[key] = {"all": float(wrong.mean()),
                    "outside_sampled_range": float(wrong[outside].mean()) if outside.any() else None}
    return out
