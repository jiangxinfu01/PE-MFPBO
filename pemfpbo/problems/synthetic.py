"""Synthetic multi-fidelity benchmarks (Section 4.2).

MF-ZDT1, MF-ZDT3 and MF-DTLZ2 are three-level variants of the standard test
functions: level three is the original function and lower levels add smooth,
objective-specific bias of decreasing amplitude. No constitutive library
exists for them, so the framework runs with an identity warping map, a zero
mean and no sign constraints; the disagreement bonus is measured against
the level-one expression.

CureWarp-4 is an analytic four-objective problem built from Eqs. (10)-(14)
with an injected multiplicative model-error field of amplitude 0.15 and
correlation length 0.3 in the unit cube.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..utils.space import Space, Variable
from .base import Problem
from .laminate import CopperCladLaminate


def zdt1(U: np.ndarray) -> np.ndarray:
    f1 = U[:, 0]
    g = 1.0 + 9.0 * U[:, 1:].mean(axis=1)
    return np.column_stack([f1, g * (1.0 - np.sqrt(f1 / g))])


def zdt3(U: np.ndarray) -> np.ndarray:
    f1 = U[:, 0]
    g = 1.0 + 9.0 * U[:, 1:].mean(axis=1)
    r = f1 / g
    return np.column_stack([f1, g * (1.0 - np.sqrt(r) - r * np.sin(10.0 * np.pi * f1))])


def dtlz2(U: np.ndarray, M: int = 3) -> np.ndarray:
    g = ((U[:, M - 1:] - 0.5) ** 2).sum(axis=1)
    x = U[:, : M - 1] * np.pi / 2.0
    F = np.empty((U.shape[0], M))
    for m in range(M):
        f = 1.0 + g
        for j in range(M - 1 - m):
            f = f * np.cos(x[:, j])
        if m > 0:
            f = f * np.sin(x[:, M - 1 - m])
        F[:, m] = f
    return F


class _SyntheticMF(Problem):
    base_fn = None
    scales = None

    def __init__(self, d: int, M: int, name: str):
        super().__init__()
        self.name = name
        self.space = Space([Variable(f"x{i + 1}", "continuous", 0.0, 1.0) for i in range(d)])
        self.objective_names = [f"f{m + 1}" for m in range(M)]
        self.senses = np.ones(M)
        self.noise_sd = np.outer([0.0, 0.002, 0.004], self.scales)

    def _bias(self, U: np.ndarray) -> np.ndarray:
        M = len(self.objective_names)
        xbar = U.mean(axis=1)
        return np.column_stack([
            np.sin(2.0 * np.pi * (U[:, 0] + 0.3 * m)) * np.cos(np.pi * xbar) + 0.5 * (xbar - 0.5)
            for m in range(M)
        ]) * self.scales

    def _evaluate(self, X: np.ndarray, level: int) -> Tuple[np.ndarray, np.ndarray]:
        F = self.base_fn(X)
        gap = 3 - level
        if gap == 0:
            Y = F
        else:
            Y = (1.0 - 0.04 * gap) * F + 0.06 * gap * self._bias(X)
        return Y, np.zeros((X.shape[0], 0))


class MFZDT1(_SyntheticMF):
    scales = np.array([1.0, 3.0])

    def __init__(self, d: int = 6):
        self.base_fn = zdt1
        super().__init__(d, 2, "mf_zdt1")


class MFZDT3(_SyntheticMF):
    scales = np.array([1.0, 3.0])

    def __init__(self, d: int = 6):
        self.base_fn = zdt3
        super().__init__(d, 2, "mf_zdt3")


class MFDTLZ2(_SyntheticMF):
    scales = np.array([1.0, 1.0, 1.0])

    def __init__(self, d: int = 6):
        self.base_fn = lambda U: dtlz2(U, 3)
        super().__init__(d, 3, "mf_dtlz2")


class CureWarp4(Problem):
    """Analytic four-objective cure benchmark with injected model error."""

    name = "curewarp4"
    THETA_STAR = np.array([16.0, 47.0, -3.0e3, 2.5e3, 2.3, -7.1, -0.15, 0.4])

    def __init__(self, amplitude: float = 0.15, corr_length: float = 0.3, field_seed: int = 2024):
        super().__init__()
        self._lam = CopperCladLaminate()
        self.space = Space([
            Variable("ramp_rate", "continuous", 0.5, 5.0, block="process"),
            Variable("dwell_temperature", "continuous", 110.0, 150.0, block="process"),
            Variable("dwell_time", "continuous", 20.0, 90.0, block="process"),
            Variable("peak_temperature", "continuous", 160.0, 200.0, block="process"),
            Variable("post_cure_time", "continuous", 30.0, 120.0, block="process"),
            Variable("silica_volume_fraction", "continuous", 0.25, 0.50, block="structure"),
        ])
        lam = self._lam
        self.objective_names = list(lam.objective_names)
        self.senses = lam.senses.copy()
        self.noise_sd = lam.noise_sd.copy()
        self.has_library = True
        self.theta_names = list(lam.theta_names)
        self.theta0 = lam.theta0.copy()
        self.theta_scale = lam.theta_scale.copy()
        self.theta_bounds = lam.theta_bounds.copy()
        self.z_names = list(lam.z_names)
        self.basis_names = list(lam.basis_names)
        self.monotone = [(1, 3, +1, 5), (0, 3, -1, 5), (2, 1, +1, 4), (3, 5, +1, 2)]
        self.objective_basis = list(lam.objective_basis)
        rng = np.random.default_rng(field_seed)
        D = 200
        self._omega = rng.standard_normal((4, D, 6)) / corr_length
        self._phase = rng.uniform(0.0, 2.0 * np.pi, (4, D))
        self._amp = amplitude

    def lift(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        n = X.shape[0]
        full = np.empty((n, 11))
        full[:, 0] = 0.0
        full[:, 1] = 1.0
        full[:, 2] = X[:, 5]
        full[:, 3] = 5.0
        full[:, 4] = 1.0
        full[:, 5] = 6.0
        full[:, 6:11] = X[:, 0:5]
        return full

    def field(self, X: np.ndarray) -> np.ndarray:
        U = self.space.to_unit(X)
        D = self._phase.shape[1]
        G = np.stack([np.sqrt(2.0 / D) * np.cos(U @ self._omega[m].T + self._phase[m]).sum(axis=1)
                      for m in range(4)], axis=1)
        return G

    def _truth(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        o = self._lam._core(self.lift(X), theta, 24)
        return np.column_stack([o["warp"], o["E_lam"], o["tg"], o["energy"]])

    def _evaluate(self, X: np.ndarray, level: int) -> Tuple[np.ndarray, np.ndarray]:
        G0 = np.zeros((X.shape[0], 0))
        if level == 1:
            return self._truth(X, self.theta0), G0
        F = self._truth(X, self.THETA_STAR)
        amp = self._amp * (1.0 if level == 3 else 0.5)
        return F * (1.0 + amp * self.field(X)), G0

    def library(self, X: np.ndarray, theta: np.ndarray):
        return self._lam.library(self.lift(X), theta)
