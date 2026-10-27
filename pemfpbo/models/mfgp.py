"""Physics-warped multi-fidelity Gaussian process (Section 3.3).

Each level is a :class:`LevelGP` over warped coordinates. Levels are linked
through the nonlinear autoregressive composition of Eq. (20): the lower-level
output enters the kernel of the next level (Eq. 21), using observed values at
training inputs (guaranteed by the nesting rule of Eq. 37) and the posterior
mean at prediction inputs, with first-order propagation of its variance.

Sign constraints are imposed through virtual derivative sites (Eqs. 23-25)
at every level, with sites placed on a space-filling set in warped
coordinates and refreshed at every hyperparameter refit. Hyperparameters,
mean coefficients and warping parameters are estimated jointly by maximum a
posteriori (Eq. 27); the prior strengths are annealed by the trust weight of
Eqs. (28)-(29).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

from ..problems.base import Problem
from .ep import ep_probit
from .features import FeatureMap
from .levelgp import EMB_DIM, LevelGP, Rows, expand_rows


@dataclass
class ModelConfig:
    warp: str = "physics"          # physics | identity | random
    learn_warp: bool = True
    mean: str = "physics"          # physics | constant | linear
    monotone: bool = True
    coupling: str = "nonlinear"    # nonlinear | linear
    icm_rank: int = 2              # 0 gives independent outputs
    single_fidelity: bool = False
    n_active: int = 256
    n_fit: int = 128
    n_virtual: int = 96
    probit_scale: float = 0.1
    anneal: bool = True
    gamma_beta: Optional[float] = None
    gamma_theta: Optional[float] = None
    gamma_grid: Sequence[float] = (0.1, 1.0, 10.0)
    fit_maxiter: int = 100
    n_rff: int = 512
    jitter: float = 1e-8


def _keys(X: np.ndarray) -> List[bytes]:
    return [np.round(np.asarray(x, dtype=float), 9).tobytes() for x in np.atleast_2d(X)]


def _maximin(Z: np.ndarray, k: int, start: Sequence[int]) -> np.ndarray:
    n = Z.shape[0]
    if k >= n:
        return np.arange(n)
    chosen = list(dict.fromkeys(int(i) for i in start))[:k]
    if not chosen:
        chosen = [0]
    dmin = np.min(((Z[:, None, :] - Z[None, chosen, :]) ** 2).sum(-1), axis=1)
    while len(chosen) < k:
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, ((Z - Z[j]) ** 2).sum(1))
    return np.array(chosen, dtype=int)


class MultiFidelityGP:
    def __init__(self, problem: Problem, cfg: ModelConfig, y_mu: np.ndarray, y_sd: np.ndarray,
                 seed: int = 0, n_outputs: Optional[int] = None, fmap: Optional[FeatureMap] = None,
                 fixed_p: Optional[np.ndarray] = None, use_monotone: Optional[bool] = None):
        self.problem = problem
        self.cfg = cfg
        self.M = problem.M if n_outputs is None else int(n_outputs)
        L = problem.n_levels
        self.levels = [L] if cfg.single_fidelity else list(range(1, L + 1))
        self.top = self.levels[-1]
        self.fmap = fmap if fmap is not None else FeatureMap(problem, cfg.warp, cfg.mean, cfg.learn_warp, seed)
        self.p_warp = self.fmap.p0.copy() if fixed_p is None else np.asarray(fixed_p, dtype=float).copy()
        self.learn_warp = self.fmap.learn and fixed_p is None
        self.y_mu = np.asarray(y_mu, dtype=float)
        self.y_sd = np.asarray(y_sd, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        cats = problem.space.cat_sizes
        rank = cfg.icm_rank if self.M > 1 else 0
        self.gps: Dict[int, LevelGP] = {}
        for i, l in enumerate(self.levels):
            floor = 1e-6 if (l == 1 and not cfg.single_fidelity) else 1e-5
            self.gps[l] = LevelGP(self.M, self.fmap.dz, cats, self.fmap.K, first=(i == 0),
                                  coupling=cfg.coupling, icm_rank=rank, noise_floor=floor)
        self.params: Dict[int, Optional[np.ndarray]] = {l: None for l in self.levels}
        self.X: Dict[int, np.ndarray] = {}
        self.Y: Dict[int, np.ndarray] = {}
        self.F: Dict[int, Optional[np.ndarray]] = {}
        self.Zc: Dict[int, np.ndarray] = {}
        self.Hc: Dict[int, np.ndarray] = {}
        self.active: Dict[int, np.ndarray] = {}
        self.post: Dict[int, dict] = {}
        self.ep: Dict[int, Optional[tuple]] = {}
        mono = cfg.monotone if use_monotone is None else use_monotone
        self.constraints = self._monotone_spec() if (mono and self.M == problem.M) else []
        self.sites: Optional[dict] = None
        self.gamma_beta0 = cfg.gamma_beta
        self.gamma_theta0 = cfg.gamma_theta
        self.gamma_beta = 1.0 if cfg.gamma_beta is None else cfg.gamma_beta
        self.gamma_theta = 1.0 if cfg.gamma_theta is None else cfg.gamma_theta
        self.wbar = 1.0
        self.wbar_history: List[float] = []
        self.coreg_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # data handling
    # ------------------------------------------------------------------
    def _monotone_spec(self) -> List[Tuple[int, int, int, int]]:
        p = self.problem
        if not p.monotone:
            return []
        if self.fmap.warp == "physics":
            return list(p.monotone)
        if self.fmap.warp == "identity":
            pos = {int(r): i for i, r in enumerate(p.space.cont_idx)}
            return [(o, pos[raw], s, raw) for (o, _, s, raw) in p.monotone if raw in pos]
        return []

    def standardize(self, Y: np.ndarray) -> np.ndarray:
        return (np.asarray(Y) - self.y_mu) / self.y_sd

    def unstandardize_mean(self, mu: np.ndarray) -> np.ndarray:
        return mu * self.y_sd + self.y_mu

    def set_data(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> None:
        for l in self.levels:
            if l not in data:
                continue
            X, Y = data[l]
            X = np.atleast_2d(np.asarray(X, dtype=float))
            n_old = 0 if l not in self.X else self.X[l].shape[0]
            self.X[l] = X
            self.Y[l] = self.standardize(Y)
            if l in self.Zc and n_old <= X.shape[0] and n_old > 0:
                if X.shape[0] > n_old:
                    z, h = self.fmap(X[n_old:], self.p_warp)
                    self.Zc[l] = np.vstack([self.Zc[l], z])
                    self.Hc[l] = np.vstack([self.Hc[l], h])
                    if l in self.active:
                        self.active[l] = np.concatenate([self.active[l], np.arange(n_old, X.shape[0])])
            else:
                self.Zc[l], self.Hc[l] = self.fmap(X, self.p_warp)
        for i, l in enumerate(self.levels):
            if i == 0 or l not in self.X:
                self.F[l] = None
                continue
            lo = self.levels[i - 1]
            lookup = {k: j for j, k in enumerate(_keys(self.X[lo]))}
            F = np.full((self.X[l].shape[0], self.M), np.nan)
            for j, k in enumerate(_keys(self.X[l])):
                if k in lookup:
                    F[j] = self.Y[lo][lookup[k]]
            self.F[l] = F

    def _refresh_cache(self) -> None:
        for l in self.levels:
            if l in self.X:
                self.Zc[l], self.Hc[l] = self.fmap(self.X[l], self.p_warp)

    def _cats(self, X: np.ndarray) -> np.ndarray:
        return self.problem.space.cats(X)

    def _subsets(self, n_max: int) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        upper_keys: Optional[set] = None
        for l in reversed(self.levels):
            if l not in self.X:
                continue
            n = self.X[l].shape[0]
            keys = _keys(self.X[l])
            prio = [] if upper_keys is None else [j for j, k in enumerate(keys) if k in upper_keys]
            if n <= n_max:
                idx = np.arange(n)
            else:
                prio = prio[: n_max // 2]
                recent = list(range(max(0, n - 8), n))
                idx = _maximin(self.Zc[l], n_max, prio + recent)
            out[l] = np.sort(idx)
            upper_keys = set(keys[j] for j in out[l])
        return out

    def _filled_F(self, l: int, idx: np.ndarray) -> Optional[np.ndarray]:
        F = self.F.get(l)
        if F is None:
            return None
        F = F[idx].copy()
        miss = np.isnan(F).any(axis=1)
        if miss.any():
            i = self.levels.index(l)
            if self.levels[i - 1] in self.post:
                mu = self._predict_levels(self.X[l][idx][miss], upto=self.levels[i - 1])[self.levels[i - 1]][0]
                F[miss] = mu
            else:
                F[miss] = 0.0
        return F

    # ------------------------------------------------------------------
    # MAP fitting (Eq. 27)
    # ------------------------------------------------------------------
    def _init_params(self, l: int, idx: np.ndarray) -> np.ndarray:
        """Mean coefficients start from the library calibration, refined by ridge
        regression on level-one data and then on the data of level ``l``."""
        b_lit = self.fmap.literature_beta(self.M, self.y_mu, self.y_sd) if self.M == self.problem.M \
            else np.zeros((self.M, self.fmap.K))

        def ridge(H, Y, prior, lam):
            A = H.T @ H + lam * np.eye(H.shape[1])
            return np.linalg.solve(A, H.T @ Y + lam * prior.T).T

        lb = self.levels[0]
        b_ref = ridge(self.Hc[lb], self.Y[lb], b_lit, 1.0 if self.cfg.single_fidelity else 1e-2)
        beta0 = b_ref if l == lb else ridge(self.Hc[l][idx], self.Y[l][idx], b_ref, 1.0)
        noise0 = np.full(self.M, 1e-5 if l == 1 else 1e-3)
        return self.gps[l].init_vector(self.rng, beta0, noise0)

    def _fit(self, levels: List[int], subsets: Dict[int, np.ndarray], params: Dict[int, np.ndarray],
             p_warp: np.ndarray, gamma_beta: float, gamma_theta: float, maxiter: int, learn: bool):
        Xs = np.vstack([self.X[l][subsets[l]] for l in levels])
        keys = _keys(Xs)
        uniq: Dict[bytes, int] = {}
        for k in keys:
            uniq.setdefault(k, len(uniq))
        pos = np.array([uniq[k] for k in keys])
        first_idx = np.zeros(len(uniq), dtype=int)
        for i, k in enumerate(keys):
            first_idx[uniq[k]] = i
        Xu = Xs[first_idx]
        Cu = self._cats(Xu)
        offs, start = {}, 0
        for l in levels:
            n = len(subsets[l])
            offs[l] = pos[start:start + n]
            start += n
        Ys = {l: self.Y[l][subsets[l]].reshape(-1) for l in levels}
        Fs = {l: self._filled_F(l, subsets[l]) for l in levels}
        for l in levels:
            self.gps[l].gamma_beta = gamma_beta
        sizes = [params[l].size for l in levels]
        nt = p_warp.size if learn else 0
        # warping parameters are optimised in units of their prior scale
        p_ref = self.fmap.p0
        p_scl = np.where(self.fmap.pscale > 0, self.fmap.pscale, 1.0)
        x0 = np.concatenate([params[l] for l in levels] + ([(p_warp - p_ref) / p_scl] if learn else []))
        bounds = sum((self.gps[l].bounds() for l in levels), [])
        if learn:
            bounds += [((lo - r) / s_, (hi - r) / s_) for (lo, hi), r, s_ in zip(self.fmap.pbounds, p_ref, p_scl)]
        fixed = None if learn else self.fmap(Xu, p_warp)

        def objective(vec):
            pw = p_ref + vec[len(vec) - nt:] * p_scl if learn else p_warp
            if learn:
                z, h, Jz, Jh = self.fmap.jacobian(Xu, pw)
            else:
                z, h = fixed
                Jz = Jh = None
            total, grads, gt = 0.0, [], np.zeros(nt)
            off = 0
            for l, sz in zip(levels, sizes):
                p = vec[off:off + sz]
                off += sz
                ii = offs[l]
                rows = expand_rows(z[ii], Cu[ii], self.M, Fs[l])
                Hr = np.repeat(h[ii], self.M, axis=0)
                Jzr = np.repeat(Jz[ii], self.M, axis=0) if learn else None
                Jhr = np.repeat(Jh[ii], self.M, axis=0) if learn else None
                v, g, gth = self.gps[l].mll(p, rows, Ys[l], Hr, Jzr, Jhr, jitter=self.cfg.jitter)
                lp, glp = self.gps[l].log_prior(p)
                total += v + lp
                grads.append(g + glp)
                if learn and gth is not None:
                    gt += gth
            fval, fgrad = -total, -np.concatenate(grads)
            if learn:
                pen, gpen = self.fmap.penalty(pw, gamma_theta)
                fval += pen
                fgrad = np.concatenate([fgrad, (-gt + gpen) * p_scl])
            if not np.isfinite(fval):
                return 1e12, np.zeros_like(vec)
            return fval, fgrad

        try:
            res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": int(maxiter)})
            x = res.x if np.isfinite(res.fun) else x0
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            x = x0
        out, off = {}, 0
        for l, sz in zip(levels, sizes):
            out[l] = x[off:off + sz].copy()
            off += sz
        pw = p_ref + x[len(x) - nt:] * p_scl if learn else p_warp.copy()
        return out, pw

    def _select_prior_weights(self) -> None:
        """Initial penalty weights by held-out predictive likelihood on level one."""
        needs = (self.fmap.mean == "physics") or self.learn_warp
        l1 = self.levels[0]
        if self.gamma_beta0 is not None and self.gamma_theta0 is not None:
            return
        if not needs or l1 not in self.X or self.X[l1].shape[0] < 12:
            self.gamma_beta0 = 1.0 if self.gamma_beta0 is None else self.gamma_beta0
            self.gamma_theta0 = 1.0 if self.gamma_theta0 is None else self.gamma_theta0
            self.gamma_beta, self.gamma_theta = self.gamma_beta0, self.gamma_theta0
            return
        n = self.X[l1].shape[0]
        perm = np.random.default_rng(self.seed + 7).permutation(n)
        n_te = max(4, n // 4)
        te, tr = perm[:n_te], perm[n_te:]
        best, best_score = (1.0, 1.0), -np.inf
        grid = list(self.cfg.gamma_grid)
        gb_grid = grid if self.gamma_beta0 is None else [self.gamma_beta0]
        gt_grid = grid if (self.gamma_theta0 is None and self.learn_warp) else [self.gamma_theta0 or 1.0]
        for gb in gb_grid:
            for gt in gt_grid:
                p0 = {l1: self._init_params(l1, tr)}
                prm, pw = self._fit([l1], {l1: tr}, p0, self.fmap.p0.copy(), gb, gt, 30, self.learn_warp)
                score = self._heldout_loglik(l1, prm[l1], pw, tr, te)
                if score > best_score:
                    best, best_score = (gb, gt), score
        self.gamma_beta0, self.gamma_theta0 = best
        self.gamma_beta, self.gamma_theta = best

    def _heldout_loglik(self, l: int, p: np.ndarray, pw: np.ndarray, tr: np.ndarray, te: np.ndarray) -> float:
        gp = self.gps[l]
        X = self.X[l]
        z, h = self.fmap(X, pw)
        C = self._cats(X)
        rt = expand_rows(z[tr], C[tr], self.M)
        rs = expand_rows(z[te], C[te], self.M)
        beta = gp.beta(p)
        mt = (np.repeat(h[tr], self.M, 0) * beta[rt.O]).sum(1)
        ms = (np.repeat(h[te], self.M, 0) * beta[rs.O]).sum(1)
        K = gp.k_vv(p, rt, rt) + np.diag(gp.noise(p)[rt.O] + 1e-8)
        try:
            cf = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf
        Ks = gp.k_vv(p, rs, rt)
        mu = ms + Ks @ cho_solve(cf, self.Y[l][tr].reshape(-1) - mt)
        v = solve_triangular(cf[0], Ks.T, lower=True)
        var = np.maximum(gp.k_diag(p, rs) - (v * v).sum(0) + gp.noise(p)[rs.O], 1e-10)
        y = self.Y[l][te].reshape(-1)
        return float(np.sum(-0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var))

    def fit(self, maxiter: Optional[int] = None) -> None:
        """Re-estimate hyperparameters, refresh active sets, sites and posterior."""
        maxiter = self.cfg.fit_maxiter if maxiter is None else maxiter
        if self.gamma_beta0 is None or self.gamma_theta0 is None:
            self._select_prior_weights()
        self._refresh_cache()
        subsets = self._subsets(self.cfg.n_fit)
        levels = [l for l in self.levels if l in subsets]
        for l in levels:
            if self.params[l] is None:
                self.params[l] = self._init_params(l, subsets[l])
        new, pw = self._fit(levels, subsets, {l: self.params[l] for l in levels}, self.p_warp,
                            self.gamma_beta, self.gamma_theta, maxiter, self.learn_warp)
        for l in levels:
            self.params[l] = new[l]
        self.p_warp = pw
        self._refresh_cache()
        self.active = self._subsets(self.cfg.n_active)
        if self.constraints:
            self._place_sites()
        self.condition(refresh_ep=True)
        if self.M > 1:
            self.coreg_history.append(self.coregionalization(normalized=True))

    # ------------------------------------------------------------------
    # virtual derivative sites
    # ------------------------------------------------------------------
    def _place_sites(self) -> None:
        pool = self.problem.space.sobol(1024, seed=int(self.rng.integers(1 << 30)))
        z, _ = self.fmap(pool, self.p_warp)
        idx = _maximin(z, self.cfg.n_virtual, [int(self.rng.integers(len(pool)))])
        nc = len(self.constraints)
        spec = np.array([self.constraints[i % nc] for i in range(len(idx))], dtype=int)
        self.sites = {"X": pool[idx], "O": spec[:, 0], "J": spec[:, 1], "s": spec[:, 2].astype(float),
                      "raw": spec[:, 3]}
        self.ep = {}

    def _site_rows(self, l: int, F: Optional[np.ndarray]) -> Rows:
        s = self.sites
        z, _ = self.fmap(s["X"], self.p_warp)
        Fs = None if F is None else F[np.arange(len(s["O"])), s["O"]]
        return Rows(z, self._cats(s["X"]), s["O"], Fs, s["J"])

    def _mean_derivative(self, l: int, p: np.ndarray) -> np.ndarray:
        s = self.sites
        sp = self.problem.space
        X0 = s["X"]
        X1 = X0.copy()
        for i, raw in enumerate(s["raw"]):
            v = sp.variables[raw]
            step = 0.02 * (v.high - v.low)
            X1[i, raw] = X0[i, raw] + step if X0[i, raw] + step <= v.high else X0[i, raw] - step
        z0, h0 = self.fmap(X0, self.p_warp)
        z1, h1 = self.fmap(X1, self.p_warp)
        beta = self.gps[l].beta(p)
        m0 = (h0 * beta[s["O"]]).sum(1)
        m1 = (h1 * beta[s["O"]]).sum(1)
        ii = np.arange(len(s["O"]))
        dz = z1[ii, s["J"]] - z0[ii, s["J"]]
        return np.where(np.abs(dz) > 1e-6, (m1 - m0) / np.where(np.abs(dz) > 1e-6, dz, 1.0), 0.0)

    # ------------------------------------------------------------------
    # posterior
    # ------------------------------------------------------------------
    def condition(self, refresh_ep: bool = False) -> None:
        jit = self.cfg.jitter
        for i, l in enumerate(self.levels):
            if l not in self.X or self.params[l] is None:
                continue
            gp, p = self.gps[l], self.params[l]
            idx = self.active.get(l, np.arange(self.X[l].shape[0]))
            Z, H, C = self.Zc[l][idx], self.Hc[l][idx], self._cats(self.X[l][idx])
            Fv = self._filled_F(l, idx)
            rows_v = expand_rows(Z, C, self.M, Fv)
            beta = gp.beta(p)
            y = self.Y[l][idx].reshape(-1)
            m = (np.repeat(H, self.M, 0) * beta[rows_v.O]).sum(1)
            noise = gp.noise(p)
            Kff = gp.k_vv(p, rows_v, rows_v)
            Kff[np.diag_indices_from(Kff)] += noise[rows_v.O] + jit
            rows_d = None
            if self.sites is not None and self.constraints:
                Fsite = None
                if i > 0:
                    Fsite = self._predict_levels(self.sites["X"], upto=self.levels[i - 1])[self.levels[i - 1]][0]
                rows_d = self._site_rows(l, Fsite)
                if refresh_ep or self.ep.get(l) is None:
                    dm = self._mean_derivative(l, p)
                    Kfd = gp.k_vd(p, rows_v, rows_d)
                    Kdd = gp.k_dd(p, rows_d, rows_d)
                    try:
                        cf = cho_factor(Kff, lower=True)
                        mu0 = dm + Kfd.T @ cho_solve(cf, y - m)
                        S0 = Kdd - Kfd.T @ cho_solve(cf, Kfd)
                        tau, mt = ep_probit(mu0, S0, self.sites["s"], nu=self.cfg.probit_scale)
                        self.ep[l] = (tau, mt, dm)
                    except np.linalg.LinAlgError:
                        self.ep[l] = None
            if rows_d is not None and self.ep.get(l) is not None:
                tau, mt, dm = self.ep[l]
                Kfd = gp.k_vd(p, rows_v, rows_d)
                Kdd = gp.k_dd(p, rows_d, rows_d) + np.diag(1.0 / tau + jit)
                A = np.block([[Kff, Kfd], [Kfd.T, Kdd]])
                r = np.concatenate([y - m, mt - dm])
            else:
                rows_d = None
                A, r = Kff, y - m
            for extra in (0.0, 1e-6, 1e-4, 1e-2):
                try:
                    Lc = np.linalg.cholesky(A + extra * np.eye(A.shape[0]))
                    break
                except np.linalg.LinAlgError:
                    continue
            a = cho_solve((Lc, True), r)
            self.post[l] = {"p": p, "rows_v": rows_v, "rows_d": rows_d, "L": Lc, "a": a, "r": r,
                            "noise_v": noise[rows_v.O],
                            "noise_d": None if rows_d is None else 1.0 / self.ep[l][0]}

    def _level_cross(self, l: int, rows: Rows) -> np.ndarray:
        post, gp = self.post[l], self.gps[l]
        K = gp.k_vv(post["p"], rows, post["rows_v"])
        if post["rows_d"] is not None:
            K = np.hstack([K, gp.k_vd(post["p"], rows, post["rows_d"])])
        return K

    def _level_dcross(self, l: int, rows: Rows) -> np.ndarray:
        post, gp = self.post[l], self.gps[l]
        K = gp.dk_vv_dF(post["p"], rows, post["rows_v"])
        if post["rows_d"] is not None:
            K = np.hstack([K, gp.dk_vd_dF(post["p"], rows, post["rows_d"])])
        return K

    def _predict_levels(self, X: np.ndarray, upto: Optional[int] = None, with_cross: bool = False,
                        chunk: int = 512) -> Dict[int, tuple]:
        """Predictive mean and variance (standardised) at every level up to ``upto``.

        Returns ``{level: (mu (n, M), var (n, M), var_latent (n, M), dmu_dF (n, M))}`` and, if
        ``with_cross``, the top-level cross covariance and rows under key ``"cross"``.
        """
        X = np.atleast_2d(X)
        upto = self.top if upto is None else upto
        lv = [l for l in self.levels if l <= upto and l in self.post]
        n = X.shape[0]
        out = {l: [np.zeros((n, self.M)) for _ in range(4)] for l in lv}
        crosses = []
        for s in range(0, n, chunk):
            Xc = X[s:s + chunk]
            nc = Xc.shape[0]
            z, h = self.fmap(Xc, self.p_warp)
            C = self._cats(Xc)
            Fp, vp = None, None
            for l in lv:
                post, gp = self.post[l], self.gps[l]
                p = post["p"]
                rows = expand_rows(z, C, self.M, Fp)
                m = (np.repeat(h, self.M, 0) * gp.beta(p)[rows.O]).sum(1)
                Ks = self._level_cross(l, rows)
                mu = m + Ks @ post["a"]
                v = solve_triangular(post["L"], Ks.T, lower=True, check_finite=False)
                var = np.maximum(gp.k_diag(p, rows) - (v * v).sum(0), 1e-12)
                var_lat = var.copy()
                dmu = np.zeros_like(mu)
                if Fp is not None:
                    dmu = self._level_dcross(l, rows) @ post["a"]
                    var = var + dmu ** 2 * vp.reshape(-1)
                res = out[l]
                res[0][s:s + nc] = mu.reshape(nc, self.M)
                res[1][s:s + nc] = var.reshape(nc, self.M)
                res[2][s:s + nc] = var_lat.reshape(nc, self.M)
                res[3][s:s + nc] = dmu.reshape(nc, self.M)
                if with_cross and l == lv[-1]:
                    crosses.append((rows, Ks, m))
                Fp, vp = mu.reshape(nc, self.M), var.reshape(nc, self.M)
        result: Dict = {l: tuple(out[l]) for l in lv}
        if with_cross:
            result["cross"] = crosses
        return result

    def predict(self, X: np.ndarray, level: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        level = self.top if level is None else level
        r = self._predict_levels(X, upto=level)
        return r[level][0], r[level][1]

    def predict_all(self, X: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        r = self._predict_levels(X)
        return {l: (r[l][0], r[l][1]) for l in self.levels if l in r}

    def predict_raw(self, X: np.ndarray, level: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        mu, var = self.predict(X, level)
        return self.unstandardize_mean(mu), var * self.y_sd ** 2

    def prior_reference(self, X: np.ndarray) -> np.ndarray:
        """Analytical prior of Eq. (28): the constitutive library evaluated with
        handbook parameters (the level-one expression), standardised."""
        X = np.atleast_2d(X)
        if self.M == self.problem.M:
            return self.standardize(self.problem.analytic_prior(X))
        return np.zeros((X.shape[0], self.M))

    def prior_var_top(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        l = self.top
        gp, p = self.gps[l], self.params[l]
        z, _ = self.fmap(X, self.p_warp)
        F = None
        if len(self.levels) > 1:
            F = self._predict_levels(X, upto=self.levels[-2])[self.levels[-2]][0]
        rows = expand_rows(z, self._cats(X), self.M, F)
        return gp.k_diag(p, rows).reshape(X.shape[0], self.M)

    def trust(self, X: np.ndarray, mu: Optional[np.ndarray] = None, var: Optional[np.ndarray] = None
              ) -> np.ndarray:
        """Scalar trust score tau(x) of Eq. (28)."""
        if mu is None or var is None:
            mu, var = self.predict(X)
        ref = self.prior_reference(X)
        pv = self.prior_var_top(X)
        tau_m = np.abs(mu - ref) / np.sqrt(var + pv)
        return np.sqrt(np.mean(tau_m ** 2, axis=1))

    def update_annealing(self, X_obs: np.ndarray, max_points: int = 512) -> float:
        """Average trust weight over observed designs and anneal the anchors (Eq. 29)."""
        X_obs = np.atleast_2d(X_obs)
        if X_obs.shape[0] > max_points:
            X_obs = X_obs[self.rng.choice(X_obs.shape[0], max_points, replace=False)]
        tau = self.trust(X_obs)
        self.wbar = float(np.mean(np.exp(-0.5 * tau ** 2)))
        self.wbar_history.append(self.wbar)
        if self.cfg.anneal:
            self.gamma_beta = self.gamma_beta0 * self.wbar
            self.gamma_theta = self.gamma_theta0 * self.wbar
        return self.wbar

    def coregionalization(self, normalized: bool = True) -> np.ndarray:
        B = self.gps[self.top].B(self.params[self.top])
        if not normalized:
            return B
        d = np.sqrt(np.diag(B))
        return B / np.outer(d, d)

    def posterior_cov(self, l: int, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        """Posterior covariance at level ``l`` between value rows of ``Xa`` and ``Xb``
        (point-major, lower-level inputs at their posterior means)."""
        rows = []
        for X in (Xa, Xb):
            X = np.atleast_2d(X)
            z, _ = self.fmap(X, self.p_warp)
            F = None
            i = self.levels.index(l)
            if i > 0:
                F = self._predict_levels(X, upto=self.levels[i - 1])[self.levels[i - 1]][0]
            rows.append(expand_rows(z, self._cats(X), self.M, F))
        gp, post = self.gps[l], self.post[l]
        Ka = self._level_cross(l, rows[0])
        Kb = self._level_cross(l, rows[1])
        va = solve_triangular(post["L"], Ka.T, lower=True)
        vb = solve_triangular(post["L"], Kb.T, lower=True)
        return gp.k_vv(post["p"], rows[0], rows[1]) - va.T @ vb

    # ------------------------------------------------------------------
    # posterior sample paths (Eq. 30)
    # ------------------------------------------------------------------
    def sample_paths(self, S: int, rng: np.random.Generator, D: Optional[int] = None) -> "SamplePaths":
        return SamplePaths(self, S, rng, self.cfg.n_rff if D is None else D)


class SamplePaths:
    """Top-level posterior sample paths: a random-Fourier-feature prior draw
    corrected by the pathwise (Matheron) update through the exact posterior."""

    def __init__(self, model: MultiFidelityGP, S: int, rng: np.random.Generator, D: int):
        self.model = model
        self.S = S
        l = model.top
        self.l = l
        gp = model.gps[l]
        post = model.post[l]
        p = post["p"]
        self.gp, self.p = gp, p
        Lay = gp.layout
        nE = EMB_DIM * len(gp.cat_sizes)
        dz = gp.dz
        comps = []
        if gp.first:
            comps.append(dict(scale=1.0, ls=np.exp(Lay.get(p, "log_ls")), use_f=False, ffac=False, D=D))
        else:
            s_lin = float(np.exp(Lay.get(p, "log_slin"))[0])
            s_d = float(np.exp(Lay.get(p, "log_sd"))[0])
            ls_d = np.exp(Lay.get(p, "log_ls_d"))
            if gp.coupling == "nonlinear":
                ls = np.exp(Lay.get(p, "log_ls"))
                lf = float(np.exp(Lay.get(p, "log_lf"))[0])
                s_nl = float(np.exp(Lay.get(p, "log_snl"))[0])
                comps.append(dict(scale=s_nl, ls=ls, use_f=True, lf=lf, ffac=False, D=D * 3 // 8))
                comps.append(dict(scale=s_lin, ls=ls, use_f=False, ffac=True, D=D // 4))
            else:
                comps.append(dict(scale=s_lin, ls=None, use_f=False, ffac=True, D=D // 4 if nE else 1))
            comps.append(dict(scale=s_d, ls=ls_d, use_f=False, ffac=False, D=D * 3 // 8))
        for c in comps:
            d_in = nE + (dz if c["ls"] is not None else 0) + (1 if c["use_f"] else 0)
            c["d_in"] = d_in
            if d_in == 0:
                c["D"] = 1
            c["Om"] = rng.standard_normal((c["D"], d_in))
            c["b"] = rng.uniform(0.0, 2.0 * np.pi, c["D"])
        self.comps = comps
        self.nE = nE
        D_tot = sum(c["D"] for c in comps)
        M = model.M
        v = np.exp(Lay.get(p, "log_v"))
        wbar = np.sqrt(v)[:, None, None] * rng.standard_normal((M, D_tot, S))
        if gp.icm_rank > 0:
            W = Lay.get(p, "W")
            wg = rng.standard_normal((gp.icm_rank, D_tot, S))
            wbar = wbar + np.einsum("mr,rds->mds", W, wg)
        self.wbar = wbar
        rows_v, rows_d = post["rows_v"], post["rows_d"]
        prior_v = self._prior(rows_v, deriv=False)
        eps_v = rng.standard_normal(prior_v.shape) * np.sqrt(post["noise_v"])[:, None]
        resid = post["r"][: rows_v.n, None] - prior_v - eps_v
        if rows_d is not None:
            prior_d = self._prior(rows_d, deriv=True)
            eps_d = rng.standard_normal(prior_d.shape) * np.sqrt(post["noise_d"])[:, None]
            resid = np.vstack([resid, post["r"][rows_v.n:, None] - prior_d - eps_d])
        self.V = cho_solve((post["L"], True), resid)

    def _phi(self, rows: Rows, deriv: bool) -> np.ndarray:
        E = self.gp._emb(self.p, rows.C)
        blocks = []
        for c in self.comps:
            parts = [E]
            if c["ls"] is not None:
                parts.append(rows.Z / c["ls"])
            if c["use_f"]:
                parts.append((rows.F / c["lf"])[:, None])
            U = np.hstack(parts) if parts else np.zeros((rows.n, 0))
            amp = np.sqrt(c["scale"])
            if c["d_in"] == 0:
                phi = np.full((rows.n, 1), amp)
                if deriv:
                    phi = np.zeros((rows.n, 1))
            else:
                arg = U @ c["Om"].T + c["b"]
                k = amp * np.sqrt(2.0 / c["D"])
                if not deriv:
                    phi = k * np.cos(arg)
                elif c["ls"] is None:
                    phi = np.zeros_like(arg)
                else:
                    col = self.nE + rows.J
                    phi = -k * np.sin(arg) * c["Om"].T[col] / c["ls"][rows.J][:, None]
            if c["ffac"]:
                phi = phi * rows.F[:, None]
            blocks.append(phi)
        return np.hstack(blocks)

    def _prior(self, rows: Rows, deriv: bool) -> np.ndarray:
        Phi = self._phi(rows, deriv)
        out = np.empty((rows.n, self.S))
        for m in range(self.model.M):
            sel = rows.O == m
            if sel.any():
                out[sel] = Phi[sel] @ self.wbar[m]
        return out

    def __call__(self, X: np.ndarray, chunk: int = 512) -> np.ndarray:
        """Sample path values ``(S, n, M)`` in standardised units."""
        X = np.atleast_2d(X)
        model = self.model
        n, M = X.shape[0], model.M
        out = np.empty((self.S, n, M))
        for s in range(0, n, chunk):
            Xc = X[s:s + chunk]
            r = model._predict_levels(Xc, with_cross=True)
            rows, Ks, m = r["cross"][0]
            f = m[:, None] + self._prior(rows, deriv=False) + Ks @ self.V
            out[:, s:s + Xc.shape[0], :] = f.reshape(Xc.shape[0], M, self.S).transpose(2, 0, 1)
        return out

    def evaluate_with_predictions(self, X: np.ndarray, chunk: int = 512):
        """Sample paths and all-level predictions from one pass over ``X``."""
        X = np.atleast_2d(X)
        model = self.model
        n, M = X.shape[0], model.M
        samples = np.empty((self.S, n, M))
        preds = {l: [np.zeros((n, M)), np.zeros((n, M))] for l in model.levels}
        for s in range(0, n, chunk):
            Xc = X[s:s + chunk]
            nc = Xc.shape[0]
            r = model._predict_levels(Xc, with_cross=True)
            rows, Ks, m = r["cross"][0]
            f = m[:, None] + self._prior(rows, deriv=False) + Ks @ self.V
            samples[:, s:s + nc, :] = f.reshape(nc, M, self.S).transpose(2, 0, 1)
            for l in model.levels:
                if l in r:
                    preds[l][0][s:s + nc] = r[l][0]
                    preds[l][1][s:s + nc] = r[l][1]
        return samples, {l: tuple(v) for l, v in preds.items()}
