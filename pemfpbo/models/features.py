"""Warped coordinates and mean-function bases.

``FeatureMap`` returns, for a design matrix ``X`` and a learnable parameter
vector ``p``, standardised warped coordinates ``z`` (Eq. 16) and
standardised mean bases ``h`` (Eq. 17).

warp = "physics"   z = Psi_theta(x), p = theta (learned jointly, Section 3.2)
warp = "identity"  z = continuous coordinates scaled to the unit cube
warp = "random"    control map of variant A9: a fixed random projection to
                   d_z outputs followed by a learned element-wise transform
                   with exactly as many parameters as theta

mean = "physics"   bases of the constitutive library
mean = "constant"  a single constant basis
mean = "linear"    constant plus the continuous coordinates (statistical prior)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..problems.base import Problem


class FeatureMap:
    def __init__(self, problem: Problem, warp: str = "physics", mean: str = "physics",
                 learn: bool = True, seed: int = 0, n_ref: int = 512):
        self.problem = problem
        if not problem.has_library:
            warp = "identity" if warp in ("physics", "random") else warp
            mean = "constant" if mean == "physics" else mean
        self.warp, self.mean = warp, mean
        self.theta0 = problem.theta0.copy() if problem.has_library else np.zeros(0)
        d_cont = len(problem.space.cont_idx)
        if warp == "physics":
            self.n_params = len(self.theta0)
            self.p0 = self.theta0.copy()
            self.pscale = problem.theta_scale.copy()
            self.pbounds = problem.theta_bounds.copy()
            self.dz = problem.dz
        elif warp == "random":
            rng = np.random.default_rng(10_000 + seed)
            self.dz = problem.dz
            Q, _ = np.linalg.qr(rng.standard_normal((d_cont, max(d_cont, self.dz))))
            self.R = Q[:, : self.dz] if Q.shape[1] >= self.dz else rng.standard_normal((d_cont, self.dz))
            self.n_params = len(problem.theta0)
            k = min(self.n_params, self.dz)
            self.p0 = np.concatenate([np.ones(k), np.zeros(self.n_params - k)])
            self.pscale = np.ones(self.n_params)
            self.pbounds = np.tile([-5.0, 5.0], (self.n_params, 1))
        else:
            self.n_params = 0
            self.p0 = np.zeros(0)
            self.pscale = np.zeros(0)
            self.pbounds = np.zeros((0, 2))
            self.dz = d_cont
        self.learn = bool(learn and self.n_params > 0)
        Xref = problem.space.sobol(n_ref, seed=12345)
        z, h = self._raw(Xref, self.p0)
        self.z_mu, self.z_sd = z.mean(axis=0), np.maximum(z.std(axis=0), 1e-9)
        self.h_mu, self.h_sd = h.mean(axis=0), np.maximum(h.std(axis=0), 1e-9)
        const = np.all(np.abs(h - h[:1]) < 1e-12, axis=0)
        self.h_mu[const] = 0.0
        self.h_sd[const] = 1.0

    def literature_beta(self, M: int, y_mu: np.ndarray, y_sd: np.ndarray) -> np.ndarray:
        """Mean coefficients that reproduce the library prediction of each objective."""
        beta = np.zeros((M, self.K))
        ob = getattr(self.problem, "objective_basis", [])
        if self.mean != "physics" or len(ob) != M:
            return beta
        for m, k in enumerate(ob):
            beta[m, k] = self.h_sd[k] / y_sd[m]
            beta[m, 0] = (self.h_mu[k] - y_mu[m]) / y_sd[m]
        return beta

    @property
    def K(self) -> int:
        if self.mean == "physics":
            return self.problem.K
        if self.mean == "linear":
            return 1 + len(self.problem.space.cont_idx)
        return 1

    # ------------------------------------------------------------------
    def _raw(self, X: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)
        sp = self.problem.space
        theta = p if self.warp == "physics" else self.theta0
        need_lib = self.problem.has_library and (self.warp == "physics" or self.mean == "physics")
        if need_lib and self.warp != "physics":
            key = X.tobytes()
            if getattr(self, "_cache_key", None) != key:
                self._cache_key, self._cache_val = key, self.problem.library(X, theta)
            zl, hl = self._cache_val
        else:
            zl, hl = self.problem.library(X, theta) if need_lib else (None, None)
        if self.warp == "physics":
            z = zl
        elif self.warp == "random":
            z = self._random_map(X, p)
        else:
            z = sp.cont_unit(X)
        if self.mean == "physics":
            h = hl
        elif self.mean == "linear":
            h = np.column_stack([np.ones(X.shape[0]), sp.cont_unit(X)])
        else:
            h = np.ones((X.shape[0], 1))
        return z, h

    def _random_map(self, X: np.ndarray, p: np.ndarray) -> np.ndarray:
        u = 2.0 * self.problem.space.cont_unit(X) - 1.0
        proj = u @ self.R
        k = min(self.n_params, self.dz)
        gains = np.ones(self.dz)
        gains[:k] = p[:k]
        bias = np.zeros(self.dz)
        extra = p[self.dz:][: self.dz]
        bias[: len(extra)] = extra
        return np.tanh(gains * proj + bias)

    def __call__(self, X: np.ndarray, p: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        p = self.p0 if p is None else p
        z, h = self._raw(X, p)
        return (z - self.z_mu) / self.z_sd, (h - self.h_mu) / self.h_sd

    def jacobian(self, X: np.ndarray, p: np.ndarray, rel_step: float = 1e-4
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Values and forward-difference sensitivities with respect to ``p``."""
        X = np.atleast_2d(X)
        nt = self.n_params
        if self.warp == "physics":
            steps = np.array([rel_step * max(abs(p[k]), self.pscale[k]) for k in range(nt)])
            Th = np.repeat(p[None, :], nt + 1, axis=0)
            Th[1:] += np.diag(steps)
            n = X.shape[0]
            Xs = np.tile(X, (nt + 1, 1))
            Ts = np.repeat(Th, n, axis=0)
            zl, hl = self.problem.library(Xs, Ts)
            zl = ((zl - self.z_mu) / self.z_sd).reshape(nt + 1, n, -1)
            if self.mean == "physics":
                hl = ((hl - self.h_mu) / self.h_sd).reshape(nt + 1, n, -1)
            else:
                h0 = self(X, p)[1]
                hl = np.repeat(h0[None], nt + 1, axis=0)
            Jz = np.moveaxis((zl[1:] - zl[0][None]) / steps[:, None, None], 0, -1)
            Jh = np.moveaxis((hl[1:] - hl[0][None]) / steps[:, None, None], 0, -1)
            return zl[0], hl[0], Jz, Jh
        z, h = self(X, p)
        if self.warp == "random":
            Jz = np.zeros(z.shape + (nt,))
            for k in range(nt):
                pk = p.copy()
                pk[k] += rel_step
                Jz[:, :, k] = (self._random_map(X, pk) - self._random_map(X, p)) / self.z_sd / rel_step
            return z, h, Jz, np.zeros(h.shape + (nt,))
        Jz = np.zeros(z.shape + (nt,))
        Jh = np.zeros(h.shape + (nt,))
        for k in range(nt):
            step = rel_step * max(abs(p[k]), self.pscale[k] if self.pscale[k] > 0 else 1.0)
            pk = p.copy()
            pk[k] += step
            zk, hk = self(X, pk)
            Jz[:, :, k] = (zk - z) / step
            Jh[:, :, k] = (hk - h) / step
        return z, h, Jz, Jh

    def penalty(self, p: np.ndarray, gamma: float) -> Tuple[float, np.ndarray]:
        """Anchor of Eq. (27) on the warping parameters (physics map only)."""
        if not self.learn or self.warp != "physics":
            return 0.0, np.zeros_like(p)
        d = (p - self.p0) / self.pscale
        return 0.5 * gamma * float(d @ d), gamma * d / self.pscale
