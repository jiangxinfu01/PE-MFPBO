"""Acquisition functions.

Orientation conventions: ``to_max`` / ``to_min`` convert standardised model
outputs to maximisation / minimisation form using the problem senses.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import log_ndtr, ndtr

from ..utils.pareto import MCHypervolume, nondominated_mask


# ----------------------------------------------------------------------
# Pareto front entropy search, Eqs. (31)-(33)
# ----------------------------------------------------------------------
def pfes(mu: np.ndarray, sd: np.ndarray, fstar: np.ndarray) -> np.ndarray:
    """Truncated-Gaussian approximation of Eq. (33).

    ``mu``, ``sd``: (n, M) predictive moments in maximisation orientation;
    ``fstar``: (S, M) per-objective maxima of the sampled fronts.
    """
    sd = np.maximum(sd, 1e-9)
    gamma = (fstar[:, None, :] - mu[None, :, :]) / sd[None, :, :]
    logcdf = log_ndtr(gamma)
    ratio = np.exp(-0.5 * gamma ** 2 - 0.5 * np.log(2.0 * np.pi) - logcdf)
    val = gamma * ratio / 2.0 - logcdf
    return np.maximum(val.sum(axis=2).mean(axis=0), 0.0)


def inner_evolution(path_fn: Callable[[np.ndarray], np.ndarray], space, X0: np.ndarray, values0: np.ndarray,
                    rng: np.random.Generator, generations: int = 2, offspring: int = 64,
                    sigma: float = 0.08, weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
                    ) -> np.ndarray:
    """Evolutionary refinement of the per-path, per-objective maxima (Eq. 31).

    ``values0`` (S, n, M) are path values (maximisation) at ``X0``. Returns the
    refined maxima (S, M). All methods that use sampled fronts share this solver.
    """
    S, n, M = values0.shape
    w0 = np.ones(n) if weight_fn is None else weight_fn(X0)
    masked = np.where(w0[None, :, None] > 0.5, values0, -np.inf)
    if not np.isfinite(masked).any():
        masked = values0
    fstar = masked.max(axis=1)
    if generations <= 0:
        return fstar
    parents_idx = np.unique(masked.argmax(axis=1).reshape(-1))
    parents = X0[parents_idx]
    U_par = space.to_unit(parents)
    cat = space.cat_idx
    for _ in range(generations):
        pick = rng.integers(len(U_par), size=offspring)
        U = U_par[pick] + sigma * rng.standard_normal((offspring, space.d))
        if len(cat):
            flip = rng.random((offspring, len(cat))) < 0.2
            U[:, cat] = np.where(flip, rng.random((offspring, len(cat))), U[:, cat])
        U = np.clip(U, 0.0, 1.0 - 1e-9)
        Xo = space.from_unit(U)
        vals = path_fn(Xo)
        w = np.ones(offspring) if weight_fn is None else weight_fn(Xo)
        vals = np.where(w[None, :, None] > 0.5, vals, -np.inf)
        fstar = np.maximum(fstar, vals.max(axis=1))
        best = np.unique(vals.argmax(axis=1).reshape(-1))
        U_par = np.vstack([U_par, U[best]])[-64:]
    return fstar


# ----------------------------------------------------------------------
# Cost-aware criterion with disagreement bonus, Eqs. (34)-(35)
# ----------------------------------------------------------------------
def copf(alpha_pfes: np.ndarray, costs: Sequence[float], kappa: float, eta: float, tau: np.ndarray,
         sigbar: np.ndarray, p_feas: Optional[np.ndarray] = None) -> np.ndarray:
    """``alpha_pfes`` (n, L) -> CoPF-ES scores (n, L)."""
    lam = np.asarray(costs, dtype=float)[None, :] ** kappa
    a = alpha_pfes / lam + eta * (tau * sigbar)[:, None]
    if p_feas is not None:
        a = a * p_feas[:, None]
    return a


def feasibility_probability(mu: np.ndarray, sd: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Product over constraints of Pr[g <= threshold] (standardised units)."""
    if mu.shape[1] == 0:
        return np.ones(mu.shape[0])
    return np.prod(ndtr((threshold[None, :] - mu) / np.maximum(sd, 1e-9)), axis=1)


# ----------------------------------------------------------------------
# Determinantal point process batch selection, Eq. (36)
# ----------------------------------------------------------------------
def dpp_greedy(scores: np.ndarray, feats: np.ndarray, levels: np.ndarray, q: int, ids: np.ndarray,
               level_corr: float = 0.5, top: int = 256) -> List[int]:
    """Greedy maximisation of ``log det(diag(a) K diag(a))`` over candidate pairs.

    ``feats`` are lengthscale-scaled warped coordinates, ``levels`` the fidelity
    of each pair and ``ids`` the design index (no design is selected twice).
    """
    order = np.argsort(-scores)[:top]
    a = np.maximum(scores[order], 1e-12)
    a = a / a.max()
    F = feats[order]
    lv = levels[order].astype(float)
    sq = (F * F).sum(1)
    K = np.exp(-0.5 * np.maximum(sq[:, None] + sq[None, :] - 2.0 * F @ F.T, 0.0))
    K *= level_corr ** np.abs(lv[:, None] - lv[None, :])
    Lm = a[:, None] * K * a[None, :] + 1e-10 * np.eye(len(order))
    n = len(order)
    cis = np.zeros((q, n))
    di2 = np.diag(Lm).copy()
    chosen: List[int] = []
    used_ids = set()
    for k in range(q):
        cand = np.array([i for i in range(n) if i not in chosen and ids[order[i]] not in used_ids])
        if cand.size == 0:
            break
        j = int(cand[np.argmax(np.log(np.maximum(di2[cand], 1e-300)))])
        chosen.append(j)
        used_ids.add(ids[order[j]])
        if k == q - 1:
            break
        e = (Lm[j] - cis[:k, j] @ cis[:k]) / np.sqrt(max(di2[j], 1e-300))
        cis[k] = e
        di2 = di2 - e ** 2
    return [int(order[j]) for j in chosen]


def top_q_distinct(scores: np.ndarray, ids: np.ndarray, U: np.ndarray, q: int, min_dist: float = 0.05
                   ) -> List[int]:
    """Top-``q`` selection with distinct designs and a minimum unit-cube distance."""
    out: List[int] = []
    for i in np.argsort(-scores):
        if len(out) == q:
            break
        if any(ids[i] == ids[j] for j in out):
            continue
        if any(np.linalg.norm(U[ids[i]] - U[ids[j]]) < min_dist for j in out):
            continue
        out.append(int(i))
    return out


# ----------------------------------------------------------------------
# Scalar expected improvement (ParEGO)
# ----------------------------------------------------------------------
def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    """EI for minimisation."""
    sd = np.maximum(sd, 1e-12)
    z = (best - mu) / sd
    return sd * (z * ndtr(z) + np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi))


# ----------------------------------------------------------------------
# Monte Carlo noisy expected hypervolume improvement (qNEHVI-type)
# ----------------------------------------------------------------------
def qnehvi_greedy(samples_obs: np.ndarray, samples_cand: np.ndarray, ref: np.ndarray, q: int,
                  rng: np.random.Generator, n_mc: int = 1024) -> List[int]:
    """Sequential-greedy batch maximising the MC expected joint HV improvement.

    ``samples_obs`` (S, n_obs, M) and ``samples_cand`` (S, n_c, M) are joint
    posterior samples in minimisation orientation; the baseline front of every
    sample is taken from the sampled values at observed designs (noisy baseline).
    """
    S = samples_cand.shape[0]
    allv = np.concatenate([samples_obs, samples_cand], axis=1)
    lo = allv.reshape(-1, allv.shape[2]).min(axis=0)
    hv = MCHypervolume(lo, ref, n_mc, rng)
    doms = [hv.dominated(samples_obs[s]) if samples_obs.shape[1] else np.zeros(n_mc, bool) for s in range(S)]
    chosen: List[int] = []
    for _ in range(q):
        gain = np.zeros(samples_cand.shape[1])
        for s in range(S):
            gain += hv.improvement(doms[s], samples_cand[s])
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        if not np.isfinite(gain[j]):
            break
        chosen.append(j)
        for s in range(S):
            doms[s] |= np.all(samples_cand[s, j][None, :] <= hv.U, axis=1)
    return chosen


# ----------------------------------------------------------------------
# Hypervolume knowledge gradient (one-step, cost-aware)
# ----------------------------------------------------------------------
def hvkg_scores(mu_D: np.ndarray, cand_mu: np.ndarray, cand_cov: np.ndarray, cross: np.ndarray,
                noise: np.ndarray, to_min: np.ndarray, ref: np.ndarray, rng: np.random.Generator,
                n_fantasy: int = 8, n_mc: int = 1024) -> np.ndarray:
    """Expected increase of the hypervolume of the posterior-mean front over a
    discrete set ``D`` after one (vector) observation at each candidate.

    ``mu_D`` (|D|, M) top-level means at D; ``cand_mu`` (n, M) and ``cand_cov``
    (n, M, M) moments of the candidate observation; ``cross`` (n, |D|, M, M)
    covariance between top-level values at D and the candidate observation;
    ``noise`` (n, M) observation noise. All standardised; ``to_min`` holds the
    signs mapping to minimisation.
    """
    n, M = cand_mu.shape
    Ymin = mu_D * to_min
    lo = Ymin.min(axis=0) - 0.5 * np.abs(Ymin.std(axis=0)) - 0.5
    hv = MCHypervolume(lo, ref, n_mc, rng)
    base = hv.volume(Ymin[nondominated_mask(Ymin)])
    xi = rng.standard_normal((n_fantasy, M))
    out = np.zeros(n)
    for i in range(n):
        Sx = cand_cov[i] + np.diag(noise[i])
        try:
            Lx = np.linalg.cholesky(Sx + 1e-9 * np.eye(M))
        except np.linalg.LinAlgError:
            continue
        gain_mat = np.einsum("dmk,kj->dmj", cross[i], np.linalg.inv(Sx))
        dy = xi @ Lx.T
        vals = []
        for f in range(n_fantasy):
            new = mu_D + np.einsum("dmj,j->dm", gain_mat, dy[f])
            Ym = new * to_min
            vals.append(hv.volume(Ym[nondominated_mask(Ym)]))
        out[i] = np.mean(vals) - base
    return np.maximum(out, 0.0)
