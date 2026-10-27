"""Expectation propagation for virtual derivative observations.

Given a Gaussian prior ``N(mu0, S0)`` over derivative values at virtual
sites and probit likelihoods ``Phi(s_i u_i / nu)`` (Eq. 23), returns
Gaussian site parameters ``(tau_i, mu_i)`` so that the sites can be added to
the GP as pseudo-observations with precision ``tau_i`` (Eq. 25).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import log_ndtr


def _ratio(z: np.ndarray) -> np.ndarray:
    """N(z) / Phi(z), computed stably."""
    return np.exp(-0.5 * z ** 2 - 0.5 * np.log(2.0 * np.pi) - log_ndtr(z))


def ep_probit(mu0: np.ndarray, S0: np.ndarray, signs: np.ndarray, nu: float = 0.1,
              n_sweeps: int = 20, damping: float = 0.5, tol: float = 1e-5
              ) -> Tuple[np.ndarray, np.ndarray]:
    n = mu0.size
    tau = np.zeros(n)
    nut = np.zeros(n)
    S0 = 0.5 * (S0 + S0.T) + 1e-10 * np.eye(n)
    S0inv = np.linalg.inv(S0)
    b0 = S0inv @ mu0
    Sigma, mu = S0.copy(), mu0.copy()
    for _ in range(n_sweeps):
        d = np.maximum(np.diag(Sigma), 1e-12)
        tau_c = np.maximum(1.0 / d - tau, 1e-10)
        nu_c = mu / d - nut
        m_c = nu_c / tau_c
        v_c = 1.0 / tau_c
        den = np.sqrt(nu ** 2 + v_c)
        z = signs * m_c / den
        r = _ratio(z)
        m_hat = m_c + signs * v_c * r / den
        v_hat = np.maximum(v_c - v_c ** 2 * r * (z + r) / den ** 2, 1e-12)
        tau_new = np.maximum(1.0 / v_hat - tau_c, 1e-8)
        nut_new = m_hat / v_hat - nu_c
        tau_upd = (1.0 - damping) * tau + damping * tau_new
        nut_upd = (1.0 - damping) * nut + damping * nut_new
        change = np.max(np.abs(tau_upd - tau)) + np.max(np.abs(nut_upd - nut))
        tau, nut = tau_upd, nut_upd
        P = S0inv + np.diag(tau)
        Sigma = np.linalg.inv(P)
        mu = Sigma @ (b0 + nut)
        if change < tol:
            break
    return tau, nut / tau
