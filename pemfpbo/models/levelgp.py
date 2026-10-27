"""One fidelity level of the physics-warped multi-fidelity GP.

The level models all objectives jointly through an intrinsic
coregionalisation structure (Eq. 22), ``B = W W^T + diag(v)``, with base kernel

* level one:            k(u, u') = k_cat(c, c') k_z(z, z')                    (Eq. 19)
* higher levels (Eq. 21):
      nonlinear coupling  k = k_cat [k_rho(z, z') (s_lin f f' + s_nl k_f(f, f')) + s_d k_d(z, z')]
      linear coupling     k = k_cat [s_lin f f' + s_d k_d(z, z')]

where ``z`` are warped coordinates, ``c`` categorical codes with learned
latent embeddings, and ``f`` the output of the next-lower level for the same
objective (Eq. 20). Rows are (point, objective) pairs; derivative rows hold
partial derivatives with respect to one warped coordinate (Eq. 24).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

EMB_DIM = 2


@dataclass
class Rows:
    Z: np.ndarray                     # (n, dz) warped coordinates
    C: np.ndarray                     # (n, nc) categorical codes
    O: np.ndarray                     # (n,) objective index
    F: Optional[np.ndarray] = None    # (n,) lower-level input
    J: Optional[np.ndarray] = None    # (n,) derivative coordinate for derivative rows

    @property
    def n(self) -> int:
        return self.Z.shape[0]


def expand_rows(Z: np.ndarray, C: np.ndarray, M: int, Fprev: Optional[np.ndarray] = None) -> Rows:
    """Point-major expansion of ``P`` points into ``P * M`` value rows."""
    P = Z.shape[0]
    Zr = np.repeat(Z, M, axis=0)
    Cr = np.repeat(C, M, axis=0)
    Or = np.tile(np.arange(M), P)
    Fr = None if Fprev is None else Fprev.reshape(-1)
    return Rows(Zr, Cr, Or, Fr)


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def _rbf(A: np.ndarray, B: np.ndarray, ls: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * _sqdist(A / ls, B / ls))


class ParamLayout:
    def __init__(self):
        self.slices: Dict[str, slice] = {}
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        self.size = 0

    def add(self, name: str, shape: Tuple[int, ...]):
        n = int(np.prod(shape)) if len(shape) else 1
        self.slices[name] = slice(self.size, self.size + n)
        self.shapes[name] = shape
        self.size += n

    def get(self, vec: np.ndarray, name: str) -> np.ndarray:
        return vec[self.slices[name]].reshape(self.shapes[name])

    def put(self, vec: np.ndarray, name: str, value) -> None:
        vec[self.slices[name]] = np.asarray(value, dtype=float).reshape(-1)

    def __contains__(self, name: str) -> bool:
        return name in self.slices


class LevelGP:
    def __init__(self, M: int, dz: int, cat_sizes: List[int], K: int, top: bool = False,
                 first: bool = True, coupling: str = "nonlinear", icm_rank: int = 2,
                 noise_floor: float = 1e-6):
        self.M, self.dz, self.cat_sizes, self.K = M, dz, list(cat_sizes), K
        self.first = first
        self.coupling = coupling
        self.icm_rank = icm_rank
        self.noise_floor = noise_floor
        L = ParamLayout()
        if first:
            L.add("log_ls", (dz,))
        else:
            if coupling == "nonlinear":
                L.add("log_ls", (dz,))
                L.add("log_lf", (1,))
                L.add("log_snl", (1,))
            L.add("log_slin", (1,))
            L.add("log_ls_d", (dz,))
            L.add("log_sd", (1,))
        for i, s in enumerate(self.cat_sizes):
            L.add(f"emb{i}", (s, EMB_DIM))
        if icm_rank > 0:
            L.add("W", (M, icm_rank))
        L.add("log_v", (M,))
        L.add("log_noise", (M,))
        L.add("beta", (M, K))
        self.layout = L
        self.beta0 = np.zeros((M, K))
        self.gamma_beta = 1.0

    # ------------------------------------------------------------------
    # parameter helpers
    # ------------------------------------------------------------------
    def init_vector(self, rng: np.random.Generator, beta0: np.ndarray, noise0: np.ndarray) -> np.ndarray:
        L = self.layout
        p = np.zeros(L.size)
        if "log_ls" in L:
            L.put(p, "log_ls", np.log(np.full(self.dz, 1.5)))
        if not self.first:
            if "log_lf" in L:
                L.put(p, "log_lf", np.log(1.0))
                L.put(p, "log_snl", np.log(0.1))
            L.put(p, "log_slin", np.log(1.0))
            L.put(p, "log_ls_d", np.log(np.full(self.dz, 1.5)))
            L.put(p, "log_sd", np.log(0.1))
        for i, s in enumerate(self.cat_sizes):
            L.put(p, f"emb{i}", 0.3 * rng.standard_normal((s, EMB_DIM)))
        if self.icm_rank > 0:
            L.put(p, "W", 0.3 * rng.standard_normal((self.M, self.icm_rank)))
        L.put(p, "log_v", np.log(np.full(self.M, 0.5)))
        L.put(p, "log_noise", np.log(np.maximum(noise0, self.noise_floor)))
        L.put(p, "beta", beta0)
        self.beta0 = beta0.copy()
        return p

    def bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        L = self.layout
        b: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * L.size
        def setb(name, lo, hi):
            s = L.slices[name]
            for i in range(s.start, s.stop):
                b[i] = (lo, hi)
        for name in ("log_ls", "log_ls_d"):
            if name in L:
                setb(name, np.log(0.05), np.log(50.0))
        if "log_lf" in L:
            setb("log_lf", np.log(0.05), np.log(50.0))
        for name in ("log_snl", "log_slin", "log_sd"):
            if name in L:
                setb(name, np.log(1e-6), np.log(1e2))
        setb("log_v", np.log(1e-6), np.log(1e2))
        setb("log_noise", np.log(self.noise_floor), np.log(1.0))
        if self.icm_rank > 0:
            setb("W", -10.0, 10.0)
        for i in range(len(self.cat_sizes)):
            setb(f"emb{i}", -5.0, 5.0)
        return b

    def B(self, p: np.ndarray) -> np.ndarray:
        v = np.exp(self.layout.get(p, "log_v"))
        B = np.diag(v)
        if self.icm_rank > 0:
            W = self.layout.get(p, "W")
            B = B + W @ W.T
        return B

    def noise(self, p: np.ndarray) -> np.ndarray:
        return np.exp(self.layout.get(p, "log_noise"))

    def beta(self, p: np.ndarray) -> np.ndarray:
        return self.layout.get(p, "beta")

    def _emb(self, p: np.ndarray, C: np.ndarray) -> np.ndarray:
        if not self.cat_sizes:
            return np.zeros((C.shape[0], 0))
        return np.concatenate([self.layout.get(p, f"emb{i}")[C[:, i]] for i in range(len(self.cat_sizes))],
                              axis=1)

    def kcat(self, p: np.ndarray, Ca: np.ndarray, Cb: np.ndarray) -> np.ndarray:
        if not self.cat_sizes:
            return np.ones((Ca.shape[0], Cb.shape[0]))
        Ea, Eb = self._emb(p, Ca), self._emb(p, Cb)
        return np.exp(-0.5 * _sqdist(Ea, Eb))

    # ------------------------------------------------------------------
    # kernel blocks
    # ------------------------------------------------------------------
    def _parts(self, p: np.ndarray, A: Rows, Bset: Rows) -> Dict[str, np.ndarray]:
        L = self.layout
        out = {"kc": self.kcat(p, A.C, Bset.C), "Bm": self.B(p)[A.O][:, Bset.O]}
        if self.first:
            ls = np.exp(L.get(p, "log_ls"))
            out["ls"] = ls
            out["R"] = _rbf(A.Z, Bset.Z, ls)
            return out
        ls_d = np.exp(L.get(p, "log_ls_d"))
        s_d = float(np.exp(L.get(p, "log_sd"))[0])
        s_lin = float(np.exp(L.get(p, "log_slin"))[0])
        out.update(ls_d=ls_d, s_d=s_d, s_lin=s_lin, Rd=_rbf(A.Z, Bset.Z, ls_d))
        FF = np.outer(A.F, Bset.F)
        out["FF"] = FF
        if self.coupling == "nonlinear":
            ls = np.exp(L.get(p, "log_ls"))
            lf = float(np.exp(L.get(p, "log_lf"))[0])
            s_nl = float(np.exp(L.get(p, "log_snl"))[0])
            Rf = np.exp(-0.5 * (A.F[:, None] - Bset.F[None, :]) ** 2 / lf ** 2)
            out.update(ls=ls, lf=lf, s_nl=s_nl, Rf=Rf, R=_rbf(A.Z, Bset.Z, ls))
            out["KF"] = s_lin * FF + s_nl * Rf
        else:
            out["KF"] = s_lin * FF
        return out

    def _G(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        if self.first:
            return parts["R"]
        if self.coupling == "nonlinear":
            return parts["R"] * parts["KF"] + parts["s_d"] * parts["Rd"]
        return parts["KF"] + parts["s_d"] * parts["Rd"]

    def k_vv(self, p: np.ndarray, A: Rows, Bset: Rows) -> np.ndarray:
        parts = self._parts(p, A, Bset)
        return parts["Bm"] * parts["kc"] * self._G(parts)

    def k_diag(self, p: np.ndarray, A: Rows) -> np.ndarray:
        Bd = np.diag(self.B(p))[A.O]
        if self.first:
            return Bd
        L = self.layout
        s_d = float(np.exp(L.get(p, "log_sd"))[0])
        s_lin = float(np.exp(L.get(p, "log_slin"))[0])
        if self.coupling == "nonlinear":
            s_nl = float(np.exp(L.get(p, "log_snl"))[0])
            return Bd * (s_lin * A.F ** 2 + s_nl + s_d)
        return Bd * (s_lin * A.F ** 2 + s_d)

    def _dR_dsecond(self, R: np.ndarray, A: Rows, D: Rows, ls: np.ndarray) -> np.ndarray:
        """d R(z_a, z_d) / d z_d[J_d]."""
        idx = np.arange(D.n)
        diff = A.Z[:, D.J] - D.Z[idx, D.J][None, :]
        return R * diff / (ls[D.J] ** 2)[None, :]

    def k_vd(self, p: np.ndarray, A: Rows, D: Rows) -> np.ndarray:
        """cov(f(A), d f(D) / d z_J) for value rows A and derivative rows D."""
        parts = self._parts(p, A, D)
        pref = parts["Bm"] * parts["kc"]
        if self.first:
            return pref * self._dR_dsecond(parts["R"], A, D, parts["ls"])
        out = parts["s_d"] * self._dR_dsecond(parts["Rd"], A, D, parts["ls_d"])
        if self.coupling == "nonlinear":
            out = out + parts["KF"] * self._dR_dsecond(parts["R"], A, D, parts["ls"])
        return pref * out

    def _d2R(self, R: np.ndarray, D1: Rows, D2: Rows, ls: np.ndarray) -> np.ndarray:
        i1 = np.arange(D1.n)
        i2 = np.arange(D2.n)
        a_i = D1.Z[i1, D1.J][:, None] - D2.Z[:, D1.J].T          # (n1, n2): z1_i - z2_i
        a_j = D1.Z[:, D2.J] - D2.Z[i2, D2.J][None, :]            # (n1, n2): z1_j - z2_j
        same = (D1.J[:, None] == D2.J[None, :]).astype(float)
        li2 = (ls[D1.J] ** 2)[:, None]
        lj2 = (ls[D2.J] ** 2)[None, :]
        return R * (same / li2 - a_i * a_j / (li2 * lj2))

    def k_dd(self, p: np.ndarray, D1: Rows, D2: Rows) -> np.ndarray:
        parts = self._parts(p, D1, D2)
        pref = parts["Bm"] * parts["kc"]
        if self.first:
            return pref * self._d2R(parts["R"], D1, D2, parts["ls"])
        out = parts["s_d"] * self._d2R(parts["Rd"], D1, D2, parts["ls_d"])
        if self.coupling == "nonlinear":
            out = out + parts["KF"] * self._d2R(parts["R"], D1, D2, parts["ls"])
        return pref * out

    # derivatives with respect to the lower-level input of the first argument
    def dk_vv_dF(self, p: np.ndarray, A: Rows, Bset: Rows) -> np.ndarray:
        parts = self._parts(p, A, Bset)
        dKF = parts["s_lin"] * np.broadcast_to(Bset.F[None, :], (A.n, Bset.n))
        if self.coupling == "nonlinear":
            dKF = dKF - parts["s_nl"] * parts["Rf"] * (A.F[:, None] - Bset.F[None, :]) / parts["lf"] ** 2
            return parts["Bm"] * parts["kc"] * parts["R"] * dKF
        return parts["Bm"] * parts["kc"] * dKF

    def dk_vd_dF(self, p: np.ndarray, A: Rows, D: Rows) -> np.ndarray:
        if self.coupling != "nonlinear":
            return np.zeros((A.n, D.n))
        parts = self._parts(p, A, D)
        dKF = parts["s_lin"] * np.broadcast_to(D.F[None, :], (A.n, D.n))
        dKF = dKF - parts["s_nl"] * parts["Rf"] * (A.F[:, None] - D.F[None, :]) / parts["lf"] ** 2
        return parts["Bm"] * parts["kc"] * dKF * self._dR_dsecond(parts["R"], A, D, parts["ls"])

    # ------------------------------------------------------------------
    # marginal likelihood with analytic gradients
    # ------------------------------------------------------------------
    def log_prior(self, p: np.ndarray) -> Tuple[float, np.ndarray]:
        L = self.layout
        g = np.zeros_like(p)
        val = 0.0
        def gauss(name, mu, sd):
            nonlocal val
            x = L.get(p, name).reshape(-1)
            val += -0.5 * np.sum((x - mu) ** 2) / sd ** 2
            g[L.slices[name]] += -(x - mu) / sd ** 2
        for name in ("log_ls", "log_ls_d"):
            if name in L:
                gauss(name, np.log(1.5), 1.0)
        if "log_lf" in L:
            gauss("log_lf", 0.0, 1.0)
            gauss("log_snl", np.log(0.1), 1.5)
        if "log_slin" in L:
            gauss("log_slin", 0.0, 1.0)
            gauss("log_sd", np.log(0.1), 1.5)
        for i in range(len(self.cat_sizes)):
            gauss(f"emb{i}", 0.0, 1.0)
        if self.icm_rank > 0:
            gauss("W", 0.0, 1.0)
        gauss("log_v", np.log(0.5), 1.5)
        gauss("log_noise", np.log(1e-3), 3.0)
        b = L.get(p, "beta").reshape(-1)
        b0 = self.beta0.reshape(-1)
        val += -0.5 * self.gamma_beta * np.sum((b - b0) ** 2)
        g[L.slices["beta"]] += -self.gamma_beta * (b - b0)
        return val, g

    def mll(self, p: np.ndarray, rows: Rows, y: np.ndarray, Hr: np.ndarray,
            Jz: Optional[np.ndarray] = None, Jh: Optional[np.ndarray] = None,
            jitter: float = 1e-8) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
        """Log marginal likelihood, its gradient w.r.t. ``p`` and w.r.t. theta.

        ``Hr`` holds the mean-function bases of every row, ``Jz`` (n, dz, nt)
        and ``Jh`` (n, K, nt) their sensitivities to the warping parameters.
        """
        L = self.layout
        N = rows.n
        parts = self._parts(p, rows, rows)
        kc, Bm = parts["kc"], parts["Bm"]
        G = self._G(parts)
        noise = self.noise(p)
        K = Bm * kc * G
        K[np.diag_indices(N)] += noise[rows.O] + jitter
        beta = self.beta(p)
        mvec = (Hr * beta[rows.O]).sum(axis=1)
        r = y - mvec
        try:
            cf = cho_factor(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -1e10, np.zeros_like(p), None
        alpha = cho_solve(cf, r, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
        val = -0.5 * r @ alpha - 0.5 * logdet - 0.5 * N * np.log(2.0 * np.pi)
        Kinv = cho_solve(cf, np.eye(N), check_finite=False)
        Wm = 0.5 * (np.outer(alpha, alpha) - Kinv)
        g = np.zeros_like(p)
        # noise
        diagW = np.diag(Wm)
        for m in range(self.M):
            sel = rows.O == m
            g[L.slices["log_noise"].start + m] = np.sum(diagW[sel]) * noise[m]
        # coregionalisation
        S = Wm * kc * G
        OH = np.eye(self.M)[rows.O]
        Sagg = OH.T @ S @ OH
        v = np.exp(L.get(p, "log_v"))
        g[L.slices["log_v"]] = np.diag(Sagg) * v
        if self.icm_rank > 0:
            W = L.get(p, "W")
            g[L.slices["W"]] = (2.0 * Sagg @ W).reshape(-1)
        # categorical embeddings
        if self.cat_sizes:
            Hc = Wm * Bm * G * kc
            s = Hc.sum(axis=1)
            for i, size in enumerate(self.cat_sizes):
                E = L.get(p, f"emb{i}")[rows.C[:, i]]
                OHc = np.eye(size)[rows.C[:, i]]
                gE = -2.0 * OHc.T @ (E * s[:, None] - Hc @ E)
                g[L.slices[f"emb{i}"]] = gE.reshape(-1)
        Wbk = Wm * Bm * kc
        gz = np.zeros((N, self.dz))
        Z = rows.Z

        def ls_grad(Hmat, ls):
            s_ = Hmat.sum(axis=1)
            HZ = Hmat @ Z
            gd = 2.0 * (Z ** 2 * s_[:, None]).sum(axis=0) - 2.0 * (Z * HZ).sum(axis=0)
            dz_ = -2.0 * (Z * s_[:, None] - HZ) / ls ** 2
            return gd / ls ** 2, dz_

        if self.first:
            Hm = Wbk * parts["R"]
            gl, dz_ = ls_grad(Hm, parts["ls"])
            g[L.slices["log_ls"]] = gl
            gz += dz_
        else:
            Hd = Wbk * parts["s_d"] * parts["Rd"]
            gl, dz_ = ls_grad(Hd, parts["ls_d"])
            g[L.slices["log_ls_d"]] = gl
            g[L.slices["log_sd"]] = Hd.sum()
            gz += dz_
            if self.coupling == "nonlinear":
                R = parts["R"]
                Hr_ = Wbk * R * parts["KF"]
                gl, dz_ = ls_grad(Hr_, parts["ls"])
                g[L.slices["log_ls"]] = gl
                gz += dz_
                g[L.slices["log_slin"]] = np.sum(Wbk * R * parts["s_lin"] * parts["FF"])
                Hnl = Wbk * R * parts["s_nl"] * parts["Rf"]
                g[L.slices["log_snl"]] = Hnl.sum()
                d2 = (rows.F[:, None] - rows.F[None, :]) ** 2
                g[L.slices["log_lf"]] = np.sum(Hnl * d2) / parts["lf"] ** 2
            else:
                g[L.slices["log_slin"]] = np.sum(Wbk * parts["s_lin"] * parts["FF"])
        # mean coefficients
        gb = np.zeros((self.M, self.K))
        for m in range(self.M):
            sel = rows.O == m
            gb[m] = alpha[sel] @ Hr[sel]
        g[L.slices["beta"]] = gb.reshape(-1)
        gtheta = None
        if Jz is not None:
            gtheta = np.einsum("nd,ndt->t", gz, Jz)
            if Jh is not None:
                gtheta = gtheta + np.einsum("n,nk,nkt->t", alpha, beta[rows.O], Jh)
        return float(val), g, gtheta
