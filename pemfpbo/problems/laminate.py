"""Case Study I: epoxy-based copper clad laminate (Table 2).

Fidelity hierarchy (Table 3):

level 1  analytical evaluation of Eqs. (8)-(14) at the platen temperature
         with handbook parameters and classical laminate relations;
level 2  lumped through-thickness thermal lag, diffusion-controlled cure,
         DiBenedetto glass transition and a single-mode viscoelastic solution;
level 3  implicit one-dimensional thermo-chemical solution through the press
         book, node-wise cure, glass transition and Prony-series stress, and a
         stress-gradient contribution to panel curvature.

The level-two and level-three solvers are reduced-order Python solvers. They
reproduce the structure of the fidelity hierarchy (systematic level-one bias,
discretisation error, distinct model physics) and are used here in place of
the RVE and board-level finite element analyses.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..physics import library as lib
from ..utils.space import Space, Variable
from .base import Perturbation, Problem, Target

BACKBONES = ["DGEBA", "DGEBF", "novolac", "biphenyl"]

# Handbook-level resin data indexed by backbone.
RESIN = {
    "tg_inf": np.array([168.0, 152.0, 205.0, 188.0]) + lib.T_ZERO_C,
    "tg0": np.array([-18.0, -25.0, 15.0, 5.0]) + lib.T_ZERO_C,
    "lam_db": np.array([0.42, 0.40, 0.48, 0.45]),
    "rate_mult": np.array([1.00, 1.15, 0.80, 0.90]),
    "E_glassy": np.array([3.0, 2.9, 3.4, 3.2]),          # GPa
    "cte": np.array([62.0, 64.0, 55.0, 50.0]),           # ppm/K, glassy
    "H_r": np.array([450.0, 470.0, 400.0, 420.0]),       # J/g
    "shrink": np.array([0.050, 0.052, 0.040, 0.042]),    # volumetric
}

T_AMB = 25.0 + lib.T_ZERO_C
T_REF_KIN = 150.0 + lib.T_ZERO_C
TG_RUBBER = -45.0 + lib.T_ZERO_C
RAMP2 = 2.5           # K/min, dwell to peak
COOL = 3.0            # K/min
BOOK_HALF = 12.0      # mm, half thickness of the press book
K_RESIN, K_SILICA = 0.20, 1.40   # W/m/K
RHO_C = 1.9e6         # J/m^3/K
E_SILICA = 70.0       # GPa
CTE_SILICA = 0.5      # ppm/K
CTE_REF = 16.0        # ppm/K, copper/glass constraint
V_GLASS, E_GLASS, K_WEAVE = 0.40, 72.0, 0.55
PANEL_GEOM = 180.0    # mm, curvature-to-warpage factor for a 250 mm coupon
H0 = 150.0            # W/m^2/K, platen contact conductance
TAU0 = 100.0          # min, handbook relaxation time
ALPHA_GEL = 0.58
ALPHA_MIN = 0.955     # minimum acceptable degree of cure
BOARD_FACTOR = 1.15   # board-level curvature amplification (levels 2 and 3)


def mobility_shift(T, tg, C1, C2):
    """log10 shift factor normalised to unity at ``Tg + 100 K`` (WLF branch)."""
    return lib.wlf_log_shift(T, tg, C1, C2) + C1 * 100.0 / (C2 + 100.0)


class CopperCladLaminate(Problem):
    name = "laminate"

    # learnable warping parameters theta (Eq. 16)
    THETA_NAMES = ["C1", "C2", "dE1", "dE2", "n_avrami", "lnK_avrami", "ln_h_scale", "ln_tau0"]
    THETA0 = np.array([17.4, 51.6, 0.0, 0.0, 2.0, -6.8, 0.0, 0.0])
    THETA_SCALE = np.array([3.0, 10.0, 5.0e3, 5.0e3, 0.5, 1.0, 0.5, 1.0])
    THETA_BOUNDS = np.array([[8.0, 30.0], [25.0, 90.0], [-1.5e4, 1.5e4], [-1.5e4, 1.5e4],
                             [1.0, 4.0], [-11.0, -3.0], [-1.5, 1.5], [-3.0, 3.0]])
    # parameters of the higher-fidelity solvers (not visible to the prior)
    TRUE = {"C1": 15.5, "C2": 46.0, "dE1": -2.5e3, "dE2": 2.0e3, "k1_mult": 1.08, "k2_mult": 0.93,
            "n_avrami": 2.4, "lnK_avrami": -7.3, "h_scale": 0.85, "tau_scale": 1.6}

    def __init__(self, n_seg_library: int = 20):
        super().__init__()
        self.space = Space([
            Variable("resin_backbone", "categorical", levels=BACKBONES, block="structure"),
            Variable("stoichiometric_ratio", "continuous", 0.80, 1.20, block="structure"),
            Variable("silica_volume_fraction", "continuous", 0.25, 0.50, block="structure"),
            Variable("filler_aspect_ratio", "continuous", 1.0, 25.0, block="structure"),
            Variable("silane_loading_wt", "continuous", 0.0, 2.0, unit="wt%", block="structure"),
            Variable("toughener_phr", "continuous", 0.0, 12.0, unit="phr", block="structure"),
            Variable("ramp_rate", "continuous", 0.5, 5.0, unit="K/min", block="process"),
            Variable("dwell_temperature", "continuous", 110.0, 150.0, unit="C", block="process"),
            Variable("dwell_time", "continuous", 20.0, 90.0, unit="min", block="process"),
            Variable("peak_temperature", "continuous", 160.0, 200.0, unit="C", block="process"),
            Variable("post_cure_time", "continuous", 30.0, 120.0, unit="min", block="process"),
        ])
        self.costs = np.array([0.004, 0.031, 1.0])
        self.objective_names = ["warpage_mm", "storage_modulus_GPa", "Tg_C", "cure_energy_kWh"]
        self.senses = np.array([1.0, -1.0, -1.0, 1.0])
        self.constraint_names = ["min_conversion"]
        self.targets = [Target(0, "<=", 0.30), Target(1, ">=", 22.0), Target(2, ">=", 160.0),
                        Target(3, "<=", 6.0)]
        typical = np.array([0.30, 24.0, 165.0, 5.0])
        self.noise_sd = np.outer([0.0, 0.003, 0.006], typical)
        self.perturbations = [Perturbation(7, "normal", 2.4), Perturbation(6, "skewnormal", 0.15, -0.83)]
        self.has_library = True
        self.theta_names = list(self.THETA_NAMES)
        self.theta0 = self.THETA0.copy()
        self.theta_scale = self.THETA_SCALE.copy()
        self.theta_bounds = self.THETA_BOUNDS.copy()
        self.z_names = ["log_reduced_time", "conversion", "transformed_fraction", "phi_eff",
                        "log_Deborah", "log_Fourier", "log_Biot", "reduced_temperature"]
        self.basis_names = ["const", "Tg_fox", "E_halpin_tsai", "log_reduced_time", "conversion",
                            "transformed_fraction", "warpage_eq14", "energy"]
        self.monotone = [(1, 3, +1, 2), (0, 3, -1, 2), (2, 1, +1, 10), (3, 5, +1, 8)]
        self.objective_basis = [6, 2, 1, 7]
        self.n_seg_library = n_seg_library

    # ------------------------------------------------------------------
    # Composition-derived quantities
    # ------------------------------------------------------------------
    @staticmethod
    def _composition(X: np.ndarray) -> Dict[str, np.ndarray]:
        b = X[:, 0].astype(int)
        r, phi, ar, sil, phr = X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
        w_t = phr / (100.0 + phr)
        dr = r - 1.0
        tg_inf = RESIN["tg_inf"][b] - np.where(dr < 0, 300.0, 200.0) * dr ** 2
        eta_s = 0.80 + 0.20 * (1.0 - np.exp(-sil / 0.5))
        k_c = lib.maxwell_conductivity(K_RESIN, K_SILICA, phi)
        kappa = k_c / RHO_C * 1e6 * 60.0  # mm^2/min
        return {
            "b": b, "r": r, "phi": phi, "ar": ar, "sil": sil, "w_t": w_t,
            "tg_inf": tg_inf, "tg0": RESIN["tg0"][b], "lam": RESIN["lam_db"][b],
            "rate": RESIN["rate_mult"][b] * np.sqrt(r), "E_g": RESIN["E_glassy"][b],
            "cte": RESIN["cte"][b], "H_r": RESIN["H_r"][b], "shrink": RESIN["shrink"][b],
            "eta_s": eta_s, "phi_eff": phi * eta_s * (1.0 + 0.25 * np.log10(ar)),
            "k_c": k_c, "kappa": kappa,
            "alpha_gel": np.clip(ALPHA_GEL / np.sqrt(np.minimum(r, 1.0)), 0.0, 0.95),
            "tg_pen": 3.0 * np.maximum(sil - 1.2, 0.0),
        }

    @staticmethod
    def _schedule(X: np.ndarray, n_seg: int):
        return lib.ramp_dwell_schedule(T_AMB, X[:, 6], X[:, 7] + lib.T_ZERO_C, X[:, 8],
                                       X[:, 9] + lib.T_ZERO_C, X[:, 10], RAMP2, COOL, n_seg)

    @staticmethod
    def _rates(c, T, dE1, dE2, k1m=1.0, k2m=1.0):
        inv = 1.0 / T - 1.0 / T_REF_KIN
        k1 = 0.010 * k1m * c["rate"] * np.exp(-(75.0e3 + dE1) / lib.R_GAS * inv)
        k2 = 0.060 * k2m * c["rate"] * np.exp(-(60.0e3 + dE2) / lib.R_GAS * inv)
        return k1, k2

    @staticmethod
    def _energy(t: np.ndarray, T: np.ndarray, overhead: float = 0.0, factor: float = 1.0) -> np.ndarray:
        dt = np.diff(t, axis=1)
        mean_excess = 0.5 * (T[:, 1:] + T[:, :-1]) - T_AMB
        loss = 95.0 * (mean_excess * dt).sum(axis=1) * 60.0        # J
        heat = 2.6e5 * np.maximum(np.diff(T, axis=1), 0.0).sum(axis=1)  # J
        return factor * (loss + heat) / 3.6e6 / 10.0 + overhead       # kWh per panel

    @staticmethod
    def _energy_correction(c) -> np.ndarray:
        """Conductive losses through the book and exotherm credit (levels 2, 3)."""
        return 0.9 * (c["k_c"] - 0.32) / 0.32 - 0.35 * c["H_r"] * (1.0 - c["phi"]) / 450.0

    @staticmethod
    def _modulus(c, alpha_f, w_d, phi_r, zeta_fac: float, zeta_pow: float, interphase: float = 0.0):
        E_m = c["E_g"] * (0.80 + 0.20 * alpha_f) * (1.0 + 0.22 * (1.0 - c["r"])) * (1.0 - 0.9 * w_d)
        E_m = E_m * (1.0 - phi_r) / (1.0 + 0.5 * phi_r)
        zeta = zeta_fac * c["ar"] ** zeta_pow
        E_c = lib.halpin_tsai(E_m, E_SILICA, c["phi"] * c["eta_s"], zeta) * (1.0 + interphase * c["sil"])
        E_lam = V_GLASS * E_GLASS * K_WEAVE + (1.0 - V_GLASS) * E_c
        return E_c, E_lam

    @staticmethod
    def _cte(c, w_d, phi_r, kerner: bool):
        a_m = c["cte"] * (1.0 + 0.5 * w_d + 0.8 * phi_r)
        rom = a_m * (1.0 - c["phi"]) + CTE_SILICA * c["phi"]
        if kerner:
            rom = rom - 0.25 * c["phi"] * (1.0 - c["phi"]) * (a_m - CTE_SILICA)
        return rom

    # ------------------------------------------------------------------
    # Level one / constitutive library
    # ------------------------------------------------------------------
    def _core(self, X: np.ndarray, theta: np.ndarray, n_seg: int) -> Dict[str, np.ndarray]:
        th = np.asarray(theta, dtype=float)
        C1, C2, dE1, dE2, nA, lnKA, lnh, lntau = th.T if th.ndim == 2 else th
        c = self._composition(X)
        t, T, _ = self._schedule(X, n_seg)
        n, N = t.shape
        alpha = np.zeros(n)
        xi_ps = np.zeros(n)
        xi_tot = np.zeros(n)
        sigma = np.zeros(n)
        E_c0, _ = self._modulus(c, np.ones(n), np.zeros(n), np.zeros(n), 2.0, 0.5)
        cte0 = self._cte(c, np.zeros(n), np.zeros(n), kerner=False)
        d_alpha_cte = cte0 - CTE_REF
        tau = TAU0 * np.exp(lntau)
        for k in range(N - 1):
            dt = t[:, k + 1] - t[:, k]
            Tm = 0.5 * (T[:, k] + T[:, k + 1])
            k1, k2 = self._rates(c, Tm, dE1, dE2)
            a_new = lib.kamal_step(alpha, k1, k2, 0.5, 1.6, dt)
            tg_a = lib.fox_tg_conversion(0.5 * (alpha + a_new), c["tg0"], c["tg_inf"])
            log_a_ps = mobility_shift(Tm, tg_a, C1, C2)
            pre_gel = alpha < c["alpha_gel"]
            xi_ps = xi_ps + np.where(pre_gel, lib.reduced_time_increment(dt, log_a_ps), 0.0)
            xi_tot = xi_tot + lib.reduced_time_increment(dt, log_a_ps)
            dxi = lib.reduced_time_increment(dt, lib.wlf_log_shift(Tm, tg_a, C1, C2))
            # Eq. (14): stress develops in the vitrified state
            glassy = 1.0 / (1.0 + np.exp((Tm - tg_a) / 4.0))
            E_t = (0.03 + 0.97 * glassy) * E_c0 * 1e3  # MPa
            d_eps = d_alpha_cte * 1e-6 * (T[:, k + 1] - T[:, k])
            d_eps = d_eps - c["shrink"] / 3.0 * (a_new - alpha) * (alpha > c["alpha_gel"]) * 0.15
            sigma = lib.stress_step(sigma, E_t, d_eps, dxi, tau)
            alpha = a_new
        X_ps = lib.avrami(xi_ps, np.exp(lnKA), nA)
        w_d = c["w_t"] * (1.0 - X_ps)
        phi_r = 1.05 * c["w_t"] * X_ps
        tg_net = lib.fox_tg_conversion(alpha, c["tg0"], c["tg_inf"])
        tg = lib.fox_tg([1.0 - w_d, w_d], [tg_net, np.full(n, TG_RUBBER)]) - c["tg_pen"]
        tg_inf_blend = lib.fox_tg([1.0 - c["w_t"], c["w_t"]], [c["tg_inf"], np.full(n, TG_RUBBER)])
        E_c, E_lam = self._modulus(c, alpha, w_d, phi_r, 2.0, 0.5)
        cte = self._cte(c, w_d, phi_r, kerner=False)
        warp = PANEL_GEOM * np.abs(sigma) * (cte - CTE_REF) / d_alpha_cte / (E_lam * 1e3)
        energy = self._energy(t, T)
        T_peak = X[:, 9] + lib.T_ZERO_C
        relax = TAU0 * np.exp(lntau) * np.power(10.0, lib.wlf_log_shift(T_peak, tg, C1, C2))
        h = H0 * np.exp(lnh)
        return {
            "alpha": alpha, "tg": tg - lib.T_ZERO_C, "E_lam": E_lam, "warp": warp, "energy": energy,
            "X_ps": X_ps, "xi": np.log10(xi_tot + 1e-3), "phi_eff": c["phi_eff"],
            "De": np.log10(np.clip(relax / X[:, 10], 1e-30, 1e30)),
            "Fo": np.log10(lib.fourier(c["kappa"], X[:, 8] + X[:, 10], BOOK_HALF)),
            "Bi": np.log10(lib.biot(h, BOOK_HALF * 1e-3, c["k_c"])),
            "Theta": lib.reduced_temperature(T_peak, tg_inf_blend),
        }

    def library(self, X: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)
        o = self._core(X, np.asarray(theta, dtype=float), self.n_seg_library)
        z = np.column_stack([o["xi"], o["alpha"], o["X_ps"], o["phi_eff"], o["De"], o["Fo"],
                             o["Bi"], o["Theta"]])
        h = np.column_stack([np.ones(X.shape[0]), o["tg"], o["E_lam"], o["xi"], o["alpha"],
                             o["X_ps"], o["warp"], o["energy"]])
        return z, h

    # ------------------------------------------------------------------
    # Level two
    # ------------------------------------------------------------------
    def _level2(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = self.TRUE
        c = self._composition(X)
        t, Tp, _ = self._schedule(X, 40)
        n, N = t.shape
        Bi = lib.biot(H0 * P["h_scale"], BOOK_HALF * 1e-3, c["k_c"])
        lam1 = Bi * (np.pi ** 2 / 4.0) / (Bi + np.pi ** 2 / 4.0)
        tau_th = BOOK_HALF ** 2 / (c["kappa"] * lam1)
        dT_ad = 0.35 * c["H_r"] / 1.2 * (1.0 - c["phi"])
        Tc = Tp[:, 0].copy()
        alpha = np.zeros(n)
        xi_ps = np.zeros(n)
        sigma = np.zeros(n)
        E_c0, _ = self._modulus(c, np.ones(n), np.zeros(n), np.zeros(n), 1.7, 0.5)
        cte0 = self._cte(c, np.zeros(n), np.zeros(n), kerner=True)
        d_alpha_cte = cte0 - CTE_REF
        tau = TAU0 * P["tau_scale"]
        T_hist = [Tc.copy()]
        for k in range(N - 1):
            dt = t[:, k + 1] - t[:, k]
            Tpm = 0.5 * (Tp[:, k] + Tp[:, k + 1])
            tg_a = lib.dibenedetto_tg(alpha, c["tg0"], c["tg_inf"], c["lam"])
            a_c = lib.dibenedetto_inverse(Tc + 12.0, c["tg0"], c["tg_inf"], c["lam"])
            f_d = 1.0 / (1.0 + np.exp(np.clip(30.0 * (alpha - a_c), -50, 50)))
            k1, k2 = self._rates(c, Tc, P["dE1"], P["dE2"], P["k1_mult"], P["k2_mult"])
            a_new = lib.kamal_step(alpha, k1, k2, 0.5, 1.6, dt, f_d)
            decay = np.exp(-dt / tau_th)
            Tc_new = Tpm + (Tc - Tpm) * decay + dT_ad * (a_new - alpha)
            Tm = 0.5 * (Tc + Tc_new)
            log_a_ps = mobility_shift(Tm, tg_a, P["C1"], P["C2"])
            xi_ps = xi_ps + np.where(alpha < c["alpha_gel"], lib.reduced_time_increment(dt, log_a_ps), 0.0)
            dxi = lib.reduced_time_increment(dt, lib.wlf_log_shift(Tm, tg_a, P["C1"], P["C2"]))
            glassy = 1.0 / (1.0 + np.exp((Tm - tg_a) / 4.0))
            E_t = (0.03 + 0.97 * glassy) * E_c0 * 1e3
            d_eps = d_alpha_cte * 1e-6 * (Tc_new - Tc)
            d_eps = d_eps - c["shrink"] / 3.0 * (a_new - alpha) * (alpha > c["alpha_gel"]) * 0.15
            sigma = lib.stress_step(sigma, E_t, d_eps, dxi, tau)
            alpha, Tc = a_new, Tc_new
            T_hist.append(Tc.copy())
        X_ps = lib.avrami(xi_ps, np.exp(P["lnK_avrami"]), P["n_avrami"])
        w_d = c["w_t"] * (1.0 - X_ps)
        phi_r = 1.05 * c["w_t"] * X_ps
        tg_net = lib.dibenedetto_tg(alpha, c["tg0"], c["tg_inf"], c["lam"])
        tg = lib.fox_tg([1.0 - w_d, w_d], [tg_net, np.full(n, TG_RUBBER)]) - c["tg_pen"]
        _, E_lam = self._modulus(c, alpha, w_d, phi_r, 1.7, 0.5)
        cte = self._cte(c, w_d, phi_r, kerner=True)
        warp = BOARD_FACTOR * PANEL_GEOM * np.abs(sigma) * (cte - CTE_REF) / d_alpha_cte / (E_lam * 1e3)
        energy = self._energy(t, Tp, overhead=0.12, factor=1.03) + self._energy_correction(c)
        Y = np.column_stack([warp, E_lam, tg - lib.T_ZERO_C, energy])
        G = (ALPHA_MIN - alpha)[:, None]
        return Y, G

    # ------------------------------------------------------------------
    # Level three
    # ------------------------------------------------------------------
    def _level3(self, X: np.ndarray, n_nodes: int = 9, n_seg: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        P = self.TRUE
        c = self._composition(X)
        t, Tp, _ = self._schedule(X, n_seg)
        n, N = t.shape
        J = n_nodes
        dx = BOOK_HALF / (J - 1)
        h_over_k = H0 * P["h_scale"] / c["k_c"] / 1000.0  # 1/mm
        dT_ad = 0.35 * c["H_r"] / 1.2 * (1.0 - c["phi"])
        T = np.repeat(Tp[:, :1], J, axis=1)
        alpha = np.zeros((n, J))
        xi_ps = np.zeros((n, J))
        weights = np.array([0.3, 0.4, 0.3])
        taus = TAU0 * P["tau_scale"] * np.array([0.1, 1.0, 10.0])
        sig = np.zeros((n, J, 3))
        E_c0, _ = self._modulus(c, np.ones(n), np.zeros(n), np.zeros(n), 1.6, 0.55, interphase=0.02)
        cte0 = self._cte(c, np.zeros(n), np.zeros(n), kerner=True)
        d_alpha_cte = (cte0 - CTE_REF)[:, None]
        cp, cg, cs = c["tg0"][:, None], c["tg_inf"][:, None], c["lam"][:, None]
        for k in range(N - 1):
            dt = t[:, k + 1] - t[:, k]
            a = c["kappa"] * dt / dx ** 2
            tg_a = lib.dibenedetto_tg(alpha, cp, cg, cs)
            a_c = lib.dibenedetto_inverse(T + 12.0, cp, cg, cs)
            f_d = 1.0 / (1.0 + np.exp(np.clip(30.0 * (alpha - a_c), -50, 50)))
            k1, k2 = self._rates({"rate": c["rate"][:, None]}, T, P["dE1"], P["dE2"], P["k1_mult"], P["k2_mult"])
            a_new = lib.kamal_step(alpha, k1, k2, 0.5, 1.6, dt[:, None], f_d)
            rhs = T + dT_ad[:, None] * (a_new - alpha)
            T_new = _implicit_conduction(T, rhs, a, h_over_k * dx, Tp[:, k + 1])
            Tm = 0.5 * (T + T_new)
            log_a_ps = mobility_shift(Tm, tg_a, P["C1"], P["C2"])
            xi_ps = xi_ps + np.where(alpha < c["alpha_gel"][:, None],
                                     lib.reduced_time_increment(dt[:, None], log_a_ps), 0.0)
            dxi = lib.reduced_time_increment(dt[:, None], lib.wlf_log_shift(Tm, tg_a, P["C1"], P["C2"]))
            glassy = 1.0 / (1.0 + np.exp((Tm - tg_a) / 4.0))
            E_t = (0.03 + 0.97 * glassy) * E_c0[:, None] * 1e3 * (0.5 + 0.5 * np.clip(alpha, 0, 1))
            d_eps = d_alpha_cte * 1e-6 * (T_new - T)
            d_eps = d_eps - c["shrink"][:, None] / 3.0 * (a_new - alpha) * (alpha > c["alpha_gel"][:, None]) * 0.15
            sig = lib.prony_stress_step(sig, E_t, weights, taus, d_eps, dxi)
            alpha, T = a_new, T_new
        X_ps = lib.avrami(xi_ps, np.exp(P["lnK_avrami"]), P["n_avrami"]).mean(axis=1)
        alpha_m = alpha.mean(axis=1)
        w_d = c["w_t"] * (1.0 - X_ps)
        phi_r = 1.05 * c["w_t"] * X_ps
        tg_net = lib.dibenedetto_tg(alpha, cp, cg, cs).mean(axis=1)
        tg = lib.fox_tg([1.0 - w_d, w_d], [tg_net, np.full(n, TG_RUBBER)]) - c["tg_pen"]
        _, E_lam = self._modulus(c, alpha_m, w_d, phi_r, 1.6, 0.55, interphase=0.02)
        cte = self._cte(c, w_d, phi_r, kerner=True)
        s_node = sig.sum(axis=2)
        s_mean = np.abs(s_node.mean(axis=1))
        s_grad = np.abs(s_node[:, -1] - s_node[:, 0])
        warp_lin = BOARD_FACTOR * PANEL_GEOM * (s_mean + 0.5 * s_grad) * (cte - CTE_REF) / d_alpha_cte[:, 0] / (E_lam * 1e3)
        warp = warp_lin / (1.0 + 0.4 * warp_lin ** 2)
        energy = self._energy(t, Tp, overhead=0.15, factor=1.04) + 1.05 * self._energy_correction(c)
        Y = np.column_stack([warp, E_lam, tg - lib.T_ZERO_C, energy])
        G = (ALPHA_MIN - alpha.min(axis=1))[:, None]
        return Y, G

    # ------------------------------------------------------------------
    def _evaluate(self, X: np.ndarray, level: int) -> Tuple[np.ndarray, np.ndarray]:
        if level == 1:
            o = self._core(X, self.theta0, 24)
            Y = np.column_stack([o["warp"], o["E_lam"], o["tg"], o["energy"]])
            G = (ALPHA_MIN - o["alpha"])[:, None]
            return Y, G
        if level == 2:
            return self._level2(X)
        if level == 3:
            out_Y, out_G = [], []
            for s in range(0, X.shape[0], 2048):
                Y, G = self._level3(X[s:s + 2048])
                out_Y.append(Y)
                out_G.append(G)
            return np.vstack(out_Y), np.vstack(out_G)
        raise ValueError(f"unknown level {level}")


def _implicit_conduction(T: np.ndarray, rhs: np.ndarray, a: np.ndarray, beta: np.ndarray,
                         T_platen: np.ndarray) -> np.ndarray:
    """Backward-Euler step of 1D conduction with a symmetry plane at node 0 and
    a convective boundary at the last node; Thomas algorithm, vectorised."""
    n, J = T.shape
    a = a[:, None]
    lower = -a * np.ones((n, J))
    upper = -a * np.ones((n, J))
    diag = (1.0 + 2.0 * a) * np.ones((n, J))
    upper[:, 0] = -2.0 * a[:, 0]
    lower[:, -1] = -2.0 * a[:, 0]
    diag[:, -1] = 1.0 + 2.0 * a[:, 0] + 2.0 * a[:, 0] * beta
    d = rhs.copy()
    d[:, -1] = d[:, -1] + 2.0 * a[:, 0] * beta * T_platen
    cp = np.zeros((n, J))
    dp = np.zeros((n, J))
    cp[:, 0] = upper[:, 0] / diag[:, 0]
    dp[:, 0] = d[:, 0] / diag[:, 0]
    for j in range(1, J):
        den = diag[:, j] - lower[:, j] * cp[:, j - 1]
        cp[:, j] = upper[:, j] / den if j < J - 1 else 0.0
        dp[:, j] = (d[:, j] - lower[:, j] * dp[:, j - 1]) / den
    out = np.empty((n, J))
    out[:, -1] = dp[:, -1]
    for j in range(J - 2, -1, -1):
        out[:, j] = dp[:, j] - cp[:, j] * out[:, j + 1]
    return out
