"""Robust decoding under input perturbation (Section 3.5, Eqs. 38-39)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from ..problems.base import Perturbation, Problem


def skewnorm_from_moments(scale: float, skew: float):
    """Skew-normal with zero mean, standard deviation ``scale`` and sample skewness ``skew``."""
    g = np.clip(abs(skew), 0.0, 0.99)
    if g == 0.0:
        return stats.norm(loc=0.0, scale=scale)
    r = (2.0 * g / (4.0 - np.pi)) ** (1.0 / 3.0)
    delta = np.sign(skew) * np.sqrt(np.pi / 2.0 * r ** 2 / (1.0 + r ** 2))
    alpha = delta / np.sqrt(1.0 - delta ** 2)
    omega = scale / np.sqrt(1.0 - 2.0 * delta ** 2 / np.pi)
    xi = -omega * delta * np.sqrt(2.0 / np.pi)
    return stats.skewnorm(alpha, loc=xi, scale=omega)


def sample_perturbations(problem: Problem, n: int, rng: np.random.Generator,
                         perturbations: Optional[List[Perturbation]] = None) -> np.ndarray:
    perturbations = problem.perturbations if perturbations is None else perturbations
    D = np.zeros((n, problem.space.d))
    for p in perturbations:
        if p.kind == "normal":
            D[:, p.variable] = p.scale * rng.standard_normal(n)
        elif p.kind == "skewnormal":
            D[:, p.variable] = skewnorm_from_moments(p.scale, p.skew).rvs(size=n, random_state=rng)
        else:
            raise ValueError(p.kind)
    return D


def fit_perturbation(deviations: np.ndarray, variable: int) -> Perturbation:
    """Fit a perturbation model to logged deviations (setpoint minus actual)."""
    d = np.asarray(deviations, dtype=float)
    d = d[np.isfinite(d)]
    skew = float(stats.skew(d))
    scale = float(np.std(d, ddof=1))
    if abs(skew) < 0.1:
        return Perturbation(variable, "normal", scale)
    return Perturbation(variable, "skewnormal", scale, skew)


def cvar_unfavourable(values: np.ndarray, senses: np.ndarray, a: float = 0.05, axis: int = -2) -> np.ndarray:
    """CVaR of the unfavourable tail, Eq. (38), in natural units.

    For minimised objectives the mean of the largest ``a`` fraction; for
    maximised objectives the mean of the smallest ``a`` fraction.
    """
    v = np.moveaxis(values, axis, -2) * senses
    n = v.shape[-2]
    k = max(1, int(np.ceil(a * n)))
    tail = np.sort(v, axis=-2)[..., n - k:, :]
    return tail.mean(axis=-2) * senses


def risk_adjusted_moments(paths, model, problem: Problem, X: np.ndarray, n_pert: int = 512, a: float = 0.05,
                          rng: Optional[np.random.Generator] = None,
                          perturbations: Optional[List[Perturbation]] = None) -> Dict[str, np.ndarray]:
    """Mean and covariance across posterior sample paths of the CVaR objectives."""
    rng = np.random.default_rng(0) if rng is None else rng
    X = np.atleast_2d(X)
    n, M = X.shape[0], problem.M
    mu = np.zeros((n, M))
    cov = np.zeros((n, M, M))
    D = sample_perturbations(problem, n_pert, rng, perturbations)
    for i, x in enumerate(X):
        Xp = problem.space.clip(x[None, :] + D)
        vals = model.unstandardize_mean(paths(Xp))          # (S, n_pert, M)
        c = cvar_unfavourable(vals, problem.senses, a, axis=1)  # (S, M)
        mu[i] = c.mean(axis=0)
        cov[i] = np.atleast_2d(np.cov(c, rowvar=False)) + 1e-12 * np.eye(M)
    return {"mu": mu, "cov": cov}


def attainment_probability(problem: Problem, mu: np.ndarray, cov: np.ndarray, n_mc: int = 4096,
                           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Probability, Eq. (39), that the risk-adjusted objectives lie in the target region."""
    rng = np.random.default_rng(1) if rng is None else rng
    out = np.zeros(mu.shape[0])
    E = rng.standard_normal((n_mc, mu.shape[1]))
    for i in range(mu.shape[0]):
        L = np.linalg.cholesky(cov[i] + 1e-10 * np.eye(mu.shape[1]))
        Ys = mu[i] + E @ L.T
        out[i] = problem.in_target(Ys).mean()
    return out


def representative(problem: Problem, Y: np.ndarray, k: int = 5) -> np.ndarray:
    """``k`` spread members: the point nearest the ideal, then max-min in normalised objectives."""
    Ym = Y * problem.senses
    Z = (Ym - Ym.min(0)) / np.maximum(Ym.max(0) - Ym.min(0), 1e-12)
    chosen = [int(np.argmin(np.linalg.norm(Z, axis=1)))]
    d = np.linalg.norm(Z - Z[chosen[0]], axis=1)
    while len(chosen) < min(k, len(Y)):
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(Z - Z[j], axis=1))
    return np.array(chosen)


def decode(method, problem: Problem, archive_X: np.ndarray, n_pool: int = 2048, rho: float = 0.10,
           a: float = 0.05, n_pert: int = 512, n_paths: int = 64, max_candidates: int = 200,
           seed: int = 0) -> Dict[str, np.ndarray]:
    """Robust inverse solution set of Eq. (39) from a fitted method surrogate."""
    rng = np.random.default_rng(seed)
    model = method.model
    pool = np.vstack([problem.space.sobol(n_pool, seed=777 + seed), archive_X])
    mu, var = model.predict_raw(pool)
    pf = method.feasibility(pool)
    nominal_ok = problem.in_target(mu) & (pf >= 0.5)
    cand = np.where(nominal_ok)[0]
    if cand.size == 0:
        cand = np.argsort(-pf)[:max_candidates]
    if cand.size > max_candidates:
        Ymin = mu[cand] * problem.senses
        from ..utils.pareto import nondominated_mask
        nd = cand[nondominated_mask(Ymin)]
        rest = np.setdiff1d(cand, nd)
        extra = rng.choice(rest, size=max(0, max_candidates - nd.size), replace=False) if rest.size else []
        cand = np.concatenate([nd, extra]).astype(int)[:max_candidates]
    paths = model.sample_paths(n_paths, rng)
    rm = risk_adjusted_moments(paths, model, problem, pool[cand], n_pert, a, rng)
    pi = attainment_probability(problem, rm["mu"], rm["cov"], rng=rng)
    keep = pi >= 1.0 - rho
    return {"X": pool[cand], "nominal": mu[cand], "risk_mean": rm["mu"], "risk_cov": rm["cov"],
            "attainment": pi, "admissible": keep, "p_feasible": pf[cand]}
