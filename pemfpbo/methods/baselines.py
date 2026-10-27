"""Baseline methods (Section 4.1).

Each baseline reproduces the defining surrogate/acquisition principle of the
cited method on the shared infrastructure (same initial design, candidate
pool, inner evolutionary solver, GP implementation and cost accounting), so
that differences reflect the surrogate and the acquisition rather than
implementation detail. They are re-implementations, not the original code.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..acquisition.core import (expected_improvement, hvkg_scores, inner_evolution, pfes,
                                qnehvi_greedy, top_q_distinct)
from ..models.mfgp import ModelConfig, MultiFidelityGP
from ..runner import Archive
from ..utils.pareto import MCHypervolume, chebyshev, das_dennis, nondominated_mask, random_simplex
from .base import Method, ModelBased

COMMON = dict(n_active=256, n_fit=128, q=4, n_samples=64, n_candidates=4096, refit_every=10)
PLAIN_SF = dict(COMMON, warp="identity", mean="constant", monotone=False, icm_rank=0, single_fidelity=True)
PLAIN_MF = dict(COMMON, warp="identity", mean="constant", monotone=False, icm_rank=0, single_fidelity=False,
                coupling="linear")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _ref_point(Ymin: np.ndarray, margin: float = 0.1) -> np.ndarray:
    lo, hi = Ymin.min(axis=0), Ymin.max(axis=0)
    return hi + margin * np.maximum(hi - lo, 1e-6)


def _greedy_hvi(Ycand: np.ndarray, Ybase: np.ndarray, ref: np.ndarray, q: int, rng: np.random.Generator,
                groups: Optional[np.ndarray] = None, n_mc: int = 2048) -> List[int]:
    allv = np.vstack([Ycand, Ybase]) if Ybase.size else Ycand
    hv = MCHypervolume(allv.min(axis=0), ref, n_mc, rng)
    dom = hv.dominated(Ybase) if Ybase.size else np.zeros(n_mc, dtype=bool)
    chosen: List[int] = []
    used = set()
    for _ in range(q):
        gain = hv.improvement(dom, Ycand)
        gain[chosen] = -np.inf
        if groups is not None:
            gain[np.isin(groups, list(used))] = -np.inf
        if not np.isfinite(gain).any():
            break
        j = int(np.argmax(gain))
        if gain[j] <= 0:
            score = -np.abs(Ycand - Ycand.mean(0)).sum(1)
            score[chosen] = -np.inf
            j = int(np.argmax(score))
        chosen.append(j)
        if groups is not None:
            used.add(groups[j])
        dom |= np.all(Ycand[j][None, :] <= hv.U, axis=1)
    return chosen


def _kmeans(U: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25) -> np.ndarray:
    k = min(k, len(U))
    C = U[rng.choice(len(U), k, replace=False)]
    lab = np.zeros(len(U), dtype=int)
    for _ in range(iters):
        lab = np.argmin(((U[:, None, :] - C[None, :, :]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if np.any(lab == j):
                C[j] = U[lab == j].mean(axis=0)
    return lab


class _ObsMixin:
    def observed_min_std(self, archive: Archive) -> np.ndarray:
        X, Y = self.top_observations(archive)
        return self.model.standardize(Y) * self.to_min


# ----------------------------------------------------------------------
# ParEGO [18]
# ----------------------------------------------------------------------
class ParEGO(ModelBased, _ObsMixin):
    name = "ParEGO"
    defaults = PLAIN_SF

    def propose(self, archive: Archive, it: int):
        self.update_models(archive, it)
        X, Y = self.top_observations(archive)
        Ym = Y * self.to_min
        Yn = (Ym - Ym.min(0)) / np.maximum(Ym.max(0) - Ym.min(0), 1e-9)
        pool = self.pool(it)
        pf = self.feasibility(pool)
        L = self.problem.n_levels
        chosen: List[int] = []
        for k in range(self.acfg.q):
            w = random_simplex(1, self.problem.M, self.rng)[0]
            s = chebyshev(Yn, w)
            cfg = ModelConfig(warp="identity", mean="constant", monotone=False, icm_rank=0, single_fidelity=True,
                              fit_maxiter=30, gamma_beta=1.0, gamma_theta=1.0, anneal=False)
            gp = MultiFidelityGP(self.problem, cfg, np.array([s.mean()]), np.array([max(s.std(), 1e-9)]),
                                 seed=self.seed + it, n_outputs=1)
            gp.set_data({L: (X, s[:, None])})
            gp.fit()
            mu, var = gp.predict(pool)
            ei = expected_improvement(mu[:, 0], np.sqrt(var[:, 0]), gp.standardize(s[:, None]).min()) * pf
            ei[chosen] = -np.inf
            chosen.append(int(np.argmax(ei)))
        return [(pool[i], L) for i in chosen]


# ----------------------------------------------------------------------
# NSGA-III [19]
# ----------------------------------------------------------------------
class NSGA3(Method):
    name = "NSGA-III"
    multi_fidelity = False

    def __init__(self, problem, seed: int = 0, q: int = 4, pop_size: Optional[int] = None, **_):
        super().__init__(problem, seed)
        M = problem.M
        p = {2: 11, 3: 4, 4: 3}.get(M, 2)
        self.refs = das_dennis(p, M)
        self.pop_size = pop_size or max(len(self.refs), 2 * q)
        self.q = q
        self.queue: List[np.ndarray] = []
        self.to_min = problem.senses

    def _fitness(self, Y: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        viol = np.maximum(G, 0.0).sum(axis=1) if G.size else np.zeros(len(Y))
        return Y * self.to_min, viol

    def _select(self, F: np.ndarray, viol: np.ndarray, N: int) -> np.ndarray:
        idx = np.arange(len(F))
        feas = viol <= 0
        if feas.sum() < N:
            order = np.concatenate([idx[feas], idx[~feas][np.argsort(viol[~feas])]])
            return order[:N]
        idx = idx[feas]
        F = F[feas]
        fronts, remaining = [], np.arange(len(F))
        while remaining.size:
            nd = nondominated_mask(F[remaining])
            fronts.append(remaining[nd])
            remaining = remaining[~nd]
        sel: List[int] = []
        for fr in fronts:
            if len(sel) + len(fr) <= N:
                sel.extend(fr.tolist())
                continue
            St = np.array(sel + fr.tolist())
            ideal = F[St].min(0)
            nadir = np.maximum(F[St].max(0), ideal + 1e-9)
            Fn = (F[St] - ideal) / (nadir - ideal)
            W = self.refs / np.maximum(np.linalg.norm(self.refs, axis=1, keepdims=True), 1e-12)
            proj = Fn @ W.T
            dist = np.sqrt(np.maximum((Fn ** 2).sum(1)[:, None] - proj ** 2, 0.0))
            assoc, d = dist.argmin(1), dist.min(1)
            niche = np.bincount(assoc[: len(sel)], minlength=len(W))
            last = list(range(len(sel), len(St)))
            while len(sel) < N and last:
                j = int(np.argmin(np.where(np.isin(np.arange(len(W)), assoc[last]), niche, np.inf)))
                members = [i for i in last if assoc[i] == j]
                pick = min(members, key=lambda i: d[i]) if niche[j] == 0 else members[self.rng.integers(len(members))]
                sel.append(int(St[pick]))
                last.remove(pick)
                niche[j] += 1
            break
        return idx[np.array(sel, dtype=int)]

    def _offspring(self, parents: np.ndarray, n: int) -> np.ndarray:
        sp = self.problem.space
        U = sp.to_unit(parents)
        out = []
        eta_c, eta_m = 15.0, 20.0
        while len(out) < n:
            a, b = U[self.rng.integers(len(U), size=2)]
            u = self.rng.random(len(a))
            beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
            c = 0.5 * ((1 + beta) * a + (1 - beta) * b)
            r = self.rng.random(len(c))
            delta = np.where(r < 0.5, (2 * r) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - r)) ** (1 / (eta_m + 1)))
            mut = self.rng.random(len(c)) < 1.0 / len(c)
            c = np.clip(c + mut * delta, 0.0, 1.0 - 1e-9)
            out.append(c)
        return sp.from_unit(np.array(out))

    def propose(self, archive: Archive, it: int):
        L = self.problem.n_levels
        if not self.queue:
            a = archive.arrays()
            sel = a["level"] == L
            F, viol = self._fitness(a["Y"][sel], a["G"][sel])
            parents = a["X"][sel][self._select(F, viol, min(self.pop_size, int(sel.sum())))]
            self.queue = list(self._offspring(parents, self.pop_size))
        batch = [self.queue.pop(0) for _ in range(min(self.q, len(self.queue)))]
        return [(x, L) for x in batch]


# ----------------------------------------------------------------------
# MESMO [20] and MF-OSEMO [9]: output-space entropy search
# ----------------------------------------------------------------------
class MESMO(ModelBased):
    name = "MESMO"
    defaults = PLAIN_SF
    cost_aware = False

    def propose(self, archive: Archive, it: int):
        a = self.acfg
        self.update_models(archive, it)
        m = self.model
        pool = self.pool(it)
        paths = m.sample_paths(a.n_samples, self.rng)
        samples, preds = paths.evaluate_with_predictions(pool)
        fstar = inner_evolution(lambda X: paths(X) * self.to_max, self.problem.space, pool,
                                samples * self.to_max, self.rng, a.ea_generations, a.ea_offspring)
        alpha = np.column_stack([pfes(preds[l][0] * self.to_max, np.sqrt(preds[l][1]), fstar) for l in m.levels])
        if self.cost_aware:
            alpha = alpha / self.charged_costs()[None, :] ** a.kappa
        alpha = alpha * self.feasibility(pool)[:, None]
        flat = alpha.reshape(-1)
        ids = np.repeat(np.arange(len(pool)), len(m.levels))
        lv = np.tile(np.array(m.levels), len(pool))
        chosen = top_q_distinct(flat, ids, self.problem.space.to_unit(pool), a.q)
        return [(pool[ids[j]], int(lv[j])) for j in chosen]


class MFOSEMO(MESMO):
    name = "MF-OSEMO"
    defaults = PLAIN_MF
    cost_aware = True


# ----------------------------------------------------------------------
# DGEMO [21]
# ----------------------------------------------------------------------
class DGEMO(ModelBased, _ObsMixin):
    name = "DGEMO"
    defaults = PLAIN_SF

    def propose(self, archive: Archive, it: int):
        self.update_models(archive, it)
        pool = self.pool(it)
        mu, _ = self.model.predict(pool)
        pf = self.feasibility(pool)
        Ym = mu * self.to_min
        cand = np.where(pf >= 0.5)[0] if np.any(pf >= 0.5) else np.arange(len(pool))
        nd = cand[nondominated_mask(Ym[cand])]
        if len(nd) < self.acfg.q:
            nd = cand[np.argsort(Ym[cand].sum(1))[: max(self.acfg.q, 16)]]
        if len(nd) > 256:
            nd = self.rng.choice(nd, 256, replace=False)
        groups = _kmeans(self.problem.space.to_unit(pool[nd]), 2 * self.acfg.q, self.rng)
        base = self.observed_min_std(archive)
        ref = _ref_point(np.vstack([base, Ym[nd]]))
        chosen = _greedy_hvi(Ym[nd], base, ref, self.acfg.q, self.rng, groups)
        L = self.problem.n_levels
        return [(pool[nd[j]], L) for j in chosen]


# ----------------------------------------------------------------------
# qNEHVI [8] and PIBO-MO [14]
# ----------------------------------------------------------------------
class QNEHVI(ModelBased, _ObsMixin):
    name = "qNEHVI"
    defaults = PLAIN_SF

    def propose(self, archive: Archive, it: int):
        self.update_models(archive, it)
        m = self.model
        pool = self.pool(it)
        mu, var = m.predict(pool)
        pf = self.feasibility(pool)
        base = self.observed_min_std(archive)
        ref = _ref_point(base)
        opt = mu * self.to_min - 2.0 * np.sqrt(var)
        hv = MCHypervolume(np.vstack([opt, base]).min(0), ref, 1024, self.rng)
        gain = hv.improvement(hv.dominated(base), opt) * pf
        top = np.argsort(-gain)[:128]
        rand = self.rng.choice(len(pool), 64, replace=False)
        cand = np.unique(np.concatenate([top, rand]))
        if np.any(pf[cand] >= 0.5):
            cand = cand[pf[cand] >= 0.5]
        Xobs, _ = self.top_observations(archive)
        paths = m.sample_paths(min(self.acfg.n_samples, 32), self.rng)
        vals = paths(np.vstack([Xobs, pool[cand]])) * self.to_min
        chosen = qnehvi_greedy(vals[:, : len(Xobs)], vals[:, len(Xobs):], ref, self.acfg.q, self.rng)
        L = self.problem.n_levels
        return [(pool[cand[j]], L) for j in chosen]


class PIBOMO(QNEHVI):
    """Physics-informed multi-output GP (physics mean, partial sign information)
    with an expected-hypervolume acquisition, single fidelity."""
    name = "PIBO-MO"
    defaults = dict(COMMON, warp="identity", mean="physics", monotone=True, icm_rank=2, single_fidelity=True)


# ----------------------------------------------------------------------
# PSL-MOBO [22]
# ----------------------------------------------------------------------
class PSLMOBO(ModelBased, _ObsMixin):
    name = "PSL-MOBO"
    defaults = PLAIN_SF

    def propose(self, archive: Archive, it: int):
        self.update_models(archive, it)
        m, sp, M = self.model, self.problem.space, self.problem.M
        X, Y = self.top_observations(archive)
        base = self.observed_min_std(archive)
        ideal, nadir = base.min(0), base.max(0)
        cont, cats = sp.cont_idx, sp.cat_idx
        cat_val = None
        if len(cats):
            nd = nondominated_mask(base)
            vals, counts = np.unique(X[nd][:, cats], axis=0, return_counts=True)
            cat_val = vals[np.argmax(counts)]
        dc = len(cont)

        def decode(params, lam):
            A = params[: dc * M].reshape(dc, M)
            b = params[dc * M:]
            u = 1.0 / (1.0 + np.exp(-(lam @ A.T + b)))
            U = np.full((len(lam), sp.d), 0.5)
            U[:, cont] = np.clip(u, 0.0, 1.0 - 1e-9)
            Xd = sp.from_unit(U)
            if cat_val is not None:
                Xd[:, cats] = cat_val
            return Xd

        lam_fit = random_simplex(32, M, self.rng)

        def loss(params):
            Xd = decode(params, lam_fit)
            mu, var = m.predict(Xd)
            lcb = mu * self.to_min - np.sqrt(var)
            Yn = (lcb - ideal) / np.maximum(nadir - ideal, 1e-9)
            return float(np.mean(chebyshev(Yn, lam_fit)))

        mean = np.zeros(dc * M + dc)
        sigma, best, best_val = 0.5, mean.copy(), loss(mean)
        for _ in range(20):
            pop = mean + sigma * self.rng.standard_normal((10, mean.size))
            vals = np.array([loss(p) for p in pop])
            elite = pop[np.argsort(vals)[:3]]
            mean = elite.mean(0)
            if vals.min() < best_val:
                best_val, best = vals.min(), pop[np.argmin(vals)]
            sigma *= 0.9
        Xc = decode(best, random_simplex(256, M, self.rng))
        mu, _ = m.predict(Xc)
        Ym = mu * self.to_min
        ref = _ref_point(np.vstack([base, Ym]))
        chosen = _greedy_hvi(Ym, base, ref, self.acfg.q, self.rng)
        L = self.problem.n_levels
        return [(Xc[j], L) for j in chosen]


# ----------------------------------------------------------------------
# HVKG-MF [12] and RESCUE [13]: cost-aware hypervolume knowledge gradient
# ----------------------------------------------------------------------
class HVKGMF(ModelBased, _ObsMixin):
    name = "HVKG-MF"
    defaults = PLAIN_MF
    n_discrete = 32
    n_cand = 24

    def propose(self, archive: Archive, it: int):
        self.update_models(archive, it)
        m, M = self.model, self.problem.M
        pool = self.pool(it)
        mu_top, var_top = m.predict(pool)
        Ym = mu_top * self.to_min
        nd = np.where(nondominated_mask(Ym))[0]
        if len(nd) > self.n_discrete:
            nd = self.rng.choice(nd, self.n_discrete, replace=False)
        Xobs, _ = self.top_observations(archive)
        base = self.observed_min_std(archive)
        keep = np.where(nondominated_mask(base))[0][:16]
        D = np.vstack([pool[nd], Xobs[keep]])
        sd_sum = np.sqrt(var_top).sum(1)
        idx = np.unique(np.concatenate([np.argsort(-sd_sum)[: self.n_cand // 2],
                                        self.rng.choice(len(pool), self.n_cand // 2, replace=False)]))
        Xc = pool[idx]
        rD = m._predict_levels(D)
        mu_D = rD[m.top][0]
        ref = _ref_point(np.vstack([mu_D * self.to_min, base]))
        rC = m._predict_levels(Xc)
        costs = self.charged_costs()
        nc, nD = len(Xc), len(D)
        scores = np.zeros((nc, len(m.levels)))
        for k, l in enumerate(m.levels):
            C = m.posterior_cov(l, D, Xc).reshape(nD, M, nc, M).transpose(2, 0, 1, 3)
            J = np.ones((nD, M))
            for j in m.levels:
                if j > l:
                    J = J * rD[j][3]
            C = C * J[None, :, :, None]
            Cc = m.posterior_cov(l, Xc, Xc).reshape(nc, M, nc, M)
            blocks = np.stack([Cc[i, :, i, :] for i in range(nc)])
            noise = np.tile(m.gps[l].noise(m.params[l]), (nc, 1))
            scores[:, k] = hvkg_scores(mu_D, rC[l][0], blocks, C, noise, self.to_min, ref, self.rng) / costs[k]
        scores = scores * self.feasibility(Xc)[:, None]
        flat = scores.reshape(-1)
        ids = np.repeat(np.arange(nc), len(m.levels))
        lv = np.tile(np.array(m.levels), nc)
        chosen = top_q_distinct(flat, ids, self.problem.space.to_unit(Xc), self.acfg.q)
        return [(Xc[ids[j]], int(lv[j])) for j in chosen]


class RESCUE(HVKGMF):
    """Multi-fidelity surrogate with a statistical (data-calibrated linear)
    prior mean and coregionalised outputs, cost-aware HVKG acquisition."""
    name = "RESCUE"
    defaults = dict(COMMON, warp="identity", mean="linear", monotone=False, icm_rank=2,
                    single_fidelity=False, coupling="linear")
