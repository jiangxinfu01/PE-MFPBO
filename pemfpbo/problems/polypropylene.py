"""Case Study II: talc-filled polypropylene housing (Table 5).

Crystallisation rather than cure governs the microstructure, so the library
uses Eq. (13) (Avrami statistics in reduced time, here in the Nakamura form
with a Ziabicki rate function) in place of Eq. (12).

level 1  one-term analytical cooling of the mid-plane, handbook parameters;
level 2  lumped cooling with latent heat release and cooling-rate dependent
         crystallinity;
level 3  implicit one-dimensional cooling across the wall with latent heat,
         pressure-shifted crystallisation and a skin-core description of
         stiffness and impact strength.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..physics import library as lib
from ..utils.space import Space, Variable
from .base import Perturbation, Problem, Target
from .laminate import _implicit_conduction

MODIFIERS = ["none", "EPR", "EOC", "SEBS"]
MOD_PHI = np.array([0.0, 0.11, 0.11, 0.11])
MOD_TOUGH = np.array([1.0, 2.6, 3.1, 2.3])
MOD_TG = np.array([-10.0, -55.0, -50.0, -60.0]) + lib.T_ZERO_C
TG_PP = -10.0 + lib.T_ZERO_C
HALF = 1.25            # mm, half wall thickness
K_PP, K_TALC = 0.20, 10.0
RHO_C = 1.8e6
RHO_PP, RHO_TALC = 0.905, 2.75
E_TALC = 120.0
T_MAX0 = 65.0 + lib.T_ZERO_C
D_ZIAB = 55.0
T_MELT_REF = 230.0 + lib.T_ZERO_C
FLOW_LENGTH = 140.0    # mm
H_CONTACT = 2500.0     # W/m^2/K
DH_LATENT = 25.0       # K of adiabatic rise for complete crystallisation


class TalcPolypropylene(Problem):
    name = "polypropylene"

    THETA_NAMES = ["C1", "C2", "n_avrami", "lnK_avrami", "dT_nucleation", "ln_h_scale"]
    THETA0 = np.array([4.4, 210.0, 2.5, -3.0, 12.0, 0.0])
    THETA_SCALE = np.array([1.0, 40.0, 0.4, 0.5, 4.0, 0.5])
    THETA_BOUNDS = np.array([[1.0, 10.0], [80.0, 400.0], [1.5, 4.0], [-6.0, -0.5], [0.0, 30.0], [-1.5, 1.5]])
    TRUE = {"C1": 5.0, "C2": 190.0, "n_avrami": 2.8, "lnK_avrami": -3.25, "dT_nucleation": 16.0,
            "h_scale": 0.80}

    def __init__(self):
        super().__init__()
        self.space = Space([
            Variable("talc_content_wt", "continuous", 5.0, 30.0, unit="wt%", block="structure"),
            Variable("talc_median_size", "continuous", 1.0, 12.0, unit="um", block="structure"),
            Variable("nucleating_agent_ppm", "continuous", 0.0, 2500.0, unit="ppm", block="structure"),
            Variable("impact_modifier", "categorical", levels=MODIFIERS, block="structure"),
            Variable("melt_temperature", "continuous", 200.0, 250.0, unit="C", block="process"),
            Variable("mold_temperature", "continuous", 20.0, 70.0, unit="C", block="process"),
            Variable("injection_speed", "continuous", 20.0, 120.0, unit="mm/s", block="process"),
            Variable("packing_pressure", "continuous", 20.0, 60.0, unit="MPa", block="process"),
            Variable("cooling_time", "continuous", 8.0, 40.0, unit="s", block="process"),
        ])
        self.costs = np.array([0.004, 0.031, 1.0])
        self.objective_names = ["flexural_modulus_GPa", "notched_izod_kJm2", "cycle_time_s"]
        self.senses = np.array([-1.0, -1.0, 1.0])
        self.constraint_names = ["solidification_at_ejection"]
        self.targets = [Target(0, ">=", 3.2), Target(1, ">=", 4.0), Target(2, "<=", 42.0)]
        typical = np.array([2.5, 4.0, 35.0])
        self.noise_sd = np.outer([0.0, 0.003, 0.006], typical)
        self.perturbations = [Perturbation(5, "normal", 1.5), Perturbation(4, "normal", 2.0)]
        self.has_library = True
        self.theta_names = list(self.THETA_NAMES)
        self.theta0 = self.THETA0.copy()
        self.theta_scale = self.THETA_SCALE.copy()
        self.theta_bounds = self.THETA_BOUNDS.copy()
        self.z_names = ["log_reduced_time", "crystallinity", "phi_eff", "log_Deborah", "log_Fourier",
                        "log_Biot", "reduced_temperature"]
        self.basis_names = ["const", "E_halpin_tsai", "izod_fox", "crystallinity", "log_reduced_time",
                            "cycle_time", "relative_crystallinity_eject"]
        self.monotone = [(0, 2, +1, 0), (1, 2, -1, 0), (2, 4, +1, 8)]
        self.objective_basis = [1, 2, 5]

    # ------------------------------------------------------------------
    @staticmethod
    def _composition(X: np.ndarray) -> Dict[str, np.ndarray]:
        talc, d50, ppm = X[:, 0], X[:, 1], X[:, 2]
        mod = X[:, 3].astype(int)
        phi_v = lib.mixture_density_fraction(talc / 100.0, RHO_TALC, RHO_PP)
        k = lib.maxwell_conductivity(K_PP, K_TALC, phi_v)
        f_nuc = 1.0 - np.exp(-ppm / 600.0)
        return {
            "talc": talc, "d50": d50, "f_nuc": f_nuc, "mod": mod, "phi_v": phi_v, "k": k,
            "kappa": k / RHO_C * 1e6,  # mm^2/s
            "ar": 3.0 + 1.6 * d50,
            "T_melt": X[:, 4] + lib.T_ZERO_C, "T_mold": X[:, 5] + lib.T_ZERO_C,
            "speed": X[:, 6], "P": X[:, 7], "t_cool": X[:, 8],
            "t_pack": 3.0 + 0.05 * (X[:, 7] - 20.0),
        }

    @staticmethod
    def _cycle(c) -> np.ndarray:
        return FLOW_LENGTH / c["speed"] + c["t_pack"] + c["t_cool"] + 6.5

    @staticmethod
    def _rate(c, T, lnK, dT_nuc, extra_shift=0.0):
        T_max = T_MAX0 + dT_nuc * c["f_nuc"] + 0.15 * c["talc"] + extra_shift
        K_max = np.exp(lnK) * (1.0 + 3.0 * c["f_nuc"] + 0.04 * c["talc"])
        return K_max * np.exp(-4.0 * np.log(2.0) * (T - T_max) ** 2 / D_ZIAB ** 2), T_max

    @staticmethod
    def _orientation(c, C1, C2):
        log_a = -C1 * (c["T_melt"] - T_MELT_REF) / (C2 + c["T_melt"] - T_MELT_REF)
        relax = 0.05 * np.power(10.0, log_a)
        shear = 4.0 * c["speed"] / (2.0 * HALF)
        De = relax * shear
        o = (0.45 + 0.25 * np.tanh((c["speed"] - 60.0) / 40.0)) * (0.6 + 0.4 * De / (1.0 + De))
        return o, De

    @staticmethod
    def _x_inf(c, cool_rate=None):
        x = 0.43 + 0.0018 * (c["T_mold"] - lib.T_ZERO_C - 20.0) + 0.035 * c["f_nuc"] + 0.001 * c["talc"]
        if cool_rate is not None:
            x = x - 0.012 * np.log10(np.maximum(cool_rate, 1e-3) / 10.0)
        return x

    @staticmethod
    def _modulus(c, Xc, o, zeta_fac=1.0):
        E_pp = 1.30 * (1.0 + 2.8 * (Xc - 0.45))
        phi_m = MOD_PHI[c["mod"]]
        E_m = E_pp * (1.0 - phi_m) / (1.0 + 0.5 * phi_m)
        E = lib.halpin_tsai(E_m, E_TALC, c["phi_v"], zeta_fac * c["ar"] * o)
        return E * (1.0 + 0.0015 * (c["P"] - 20.0))

    @staticmethod
    def _izod(c, Xc, T_melt_bonus=1.0):
        phi_m = MOD_PHI[c["mod"]]
        tg_blend = lib.fox_tg([1.0 - 0.15 * phi_m, 0.15 * phi_m], [np.full(phi_m.shape, TG_PP), MOD_TG[c["mod"]]])
        ductility = 1.0 + 0.8 * (TG_PP - tg_blend) / 10.0
        I = 2.4 * (1.0 - 1.6 * (Xc - 0.45)) * (1.0 + 0.15 * c["f_nuc"])
        I = I * MOD_TOUGH[c["mod"]] * ductility * T_melt_bonus
        I = I * (1.0 - 1.8 * c["phi_v"] * (c["d50"] / 6.0) ** 0.4)
        return I * (1.0 + 0.1 * np.tanh((c["speed"] - 60.0) / 40.0))

    # ------------------------------------------------------------------
    def _core(self, X: np.ndarray, theta: np.ndarray, n_steps: int = 40) -> Dict[str, np.ndarray]:
        th = np.asarray(theta, dtype=float)
        C1, C2, nA, lnK, dTn, lnh = th.T if th.ndim == 2 else th
        c = self._composition(X)
        n = X.shape[0]
        Bi = lib.biot(H_CONTACT * np.exp(lnh), HALF * 1e-3, c["k"])
        lam2 = Bi * (np.pi ** 2 / 4.0) / (Bi + np.pi ** 2 / 4.0)
        t_e = c["t_pack"] + c["t_cool"]
        xi = np.zeros(n)
        frac = np.linspace(0.0, 1.0, n_steps + 1)
        T_prev = c["T_melt"]
        for k in range(1, n_steps + 1):
            tk = frac[k] * t_e
            dt = (frac[k] - frac[k - 1]) * t_e
            Fo = c["kappa"] * tk / HALF ** 2
            Tk = c["T_mold"] + (c["T_melt"] - c["T_mold"]) * np.minimum(1.27 * np.exp(-lam2 * Fo), 1.0)
            K, T_max = self._rate(c, 0.5 * (Tk + T_prev), lnK, dTn)
            xi = xi + K * dt
            T_prev = Tk
        X_rel = 1.0 - np.exp(-np.power(xi, nA))
        Xc = self._x_inf(c) * (1.0 - 0.1 * (1.0 - X_rel))
        o, De = self._orientation(c, C1, C2)
        E = self._modulus(c, Xc, o)
        I = self._izod(c, Xc)
        cycle = self._cycle(c)
        phi_eff = c["phi_v"] * np.sqrt(c["ar"]) * o
        Fo_e = c["kappa"] * t_e / HALF ** 2
        _, T_max = self._rate(c, c["T_mold"], lnK, dTn)
        return {
            "E": E, "I": I, "cycle": cycle, "X_rel": X_rel, "Xc": Xc,
            "xi": np.log10(xi + 1e-6), "phi_eff": phi_eff, "De": np.log10(De),
            "Fo": np.log10(Fo_e), "Bi": np.log10(Bi), "Theta": (c["T_mold"] - T_max) / T_max,
        }

    def library(self, X: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)
        o = self._core(X, np.asarray(theta, dtype=float))
        z = np.column_stack([o["xi"], o["Xc"], o["phi_eff"], o["De"], o["Fo"], o["Bi"], o["Theta"]])
        h = np.column_stack([np.ones(X.shape[0]), o["E"], o["I"], o["Xc"], o["xi"], o["cycle"], o["X_rel"]])
        return z, h

    # ------------------------------------------------------------------
    def _level2(self, X: np.ndarray, n_steps: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        P = self.TRUE
        c = self._composition(X)
        n = X.shape[0]
        Bi = lib.biot(H_CONTACT * P["h_scale"], HALF * 1e-3, c["k"])
        lam2 = Bi * (np.pi ** 2 / 4.0) / (Bi + np.pi ** 2 / 4.0)
        tau = HALF ** 2 / (c["kappa"] * lam2)
        t_e = c["t_pack"] + c["t_cool"]
        dt = t_e / n_steps
        T = c["T_melt"].copy()
        xi = np.zeros(n)
        X_rel = np.zeros(n)
        cool_rate = np.zeros(n)
        for k in range(n_steps):
            K, T_max = self._rate(c, T, P["lnK_avrami"], P["dT_nucleation"])
            xi = xi + K * dt
            X_new = 1.0 - np.exp(-np.power(xi, P["n_avrami"]))
            T_new = c["T_mold"] + (T - c["T_mold"]) * np.exp(-dt / tau) + DH_LATENT * (X_new - X_rel)
            near = np.abs(T - T_max) < 15.0
            cool_rate = np.where(near & (cool_rate == 0.0), (T - T_new) / dt, cool_rate)
            T, X_rel = T_new, X_new
        cool_rate = np.where(cool_rate == 0.0, 10.0, cool_rate)
        Xc = self._x_inf(c, cool_rate) * (1.0 - 0.1 * (1.0 - X_rel))
        o, _ = self._orientation(c, P["C1"], P["C2"])
        E = self._modulus(c, Xc, o, zeta_fac=0.92)
        I = self._izod(c, Xc, T_melt_bonus=1.0 + 0.004 * (c["T_melt"] - T_MELT_REF))
        Y = np.column_stack([E, I, self._cycle(c) + 0.4])
        return Y, (0.95 - X_rel)[:, None]

    def _level3(self, X: np.ndarray, n_nodes: int = 9, n_steps: int = 160) -> Tuple[np.ndarray, np.ndarray]:
        P = self.TRUE
        c = self._composition(X)
        n, J = X.shape[0], n_nodes
        dx = HALF / (J - 1)
        beta = H_CONTACT * P["h_scale"] / c["k"] / 1000.0 * dx
        t_e = c["t_pack"] + c["t_cool"]
        dt = t_e / n_steps
        a = c["kappa"] * dt / dx ** 2
        T = np.repeat(c["T_melt"][:, None], J, axis=1)
        xi = np.zeros((n, J))
        X_rel = np.zeros((n, J))
        T_cryst = np.zeros((n, J))
        shift = 0.2 * c["P"][:, None]
        cc = {k: (v[:, None] if isinstance(v, np.ndarray) else v) for k, v in c.items()}
        for k in range(n_steps):
            K, _ = self._rate(cc, T, P["lnK_avrami"], P["dT_nucleation"], extra_shift=shift)
            xi = xi + K * dt[:, None]
            X_new = 1.0 - np.exp(-np.power(xi, P["n_avrami"]))
            half_now = (X_new >= 0.5) & (X_rel < 0.5)
            T_cryst = np.where(half_now, T, T_cryst)
            T = _implicit_conduction(T, T + DH_LATENT * (X_new - X_rel), a, beta, c["T_mold"])
            X_rel = X_new
        T_cryst = np.where(T_cryst == 0.0, c["T_mold"][:, None], T_cryst)
        x_inf = (0.43 + 0.0018 * (T_cryst - lib.T_ZERO_C - 20.0) * 0.8 + 0.035 * cc["f_nuc"]
                 + 0.001 * cc["talc"])
        Xc_nodes = x_inf * (1.0 - 0.1 * (1.0 - X_rel))
        o, _ = self._orientation(c, P["C1"], P["C2"])
        skin = np.linspace(0.0, 1.0, J) ** 2
        o_nodes = o[:, None] * (0.75 + 0.5 * skin)
        E_nodes = self._modulus(cc, Xc_nodes, o_nodes, zeta_fac=0.9)
        w = (np.linspace(0.0, 1.0, J) ** 2)[None, :] + 0.2
        E = (E_nodes * w).sum(axis=1) / w.sum(axis=1)
        Xc_core = Xc_nodes[:, : J // 2].mean(axis=1)
        I = self._izod(c, Xc_core, T_melt_bonus=1.0 + 0.004 * (c["T_melt"] - T_MELT_REF))
        Y = np.column_stack([E, I, self._cycle(c) + 0.4])
        return Y, (0.95 - X_rel[:, 0])[:, None]

    def _evaluate(self, X: np.ndarray, level: int) -> Tuple[np.ndarray, np.ndarray]:
        if level == 1:
            o = self._core(X, self.theta0)
            return np.column_stack([o["E"], o["I"], o["cycle"]]), (0.95 - o["X_rel"])[:, None]
        if level == 2:
            return self._level2(X)
        if level == 3:
            return self._level3(X)
        raise ValueError(f"unknown level {level}")
