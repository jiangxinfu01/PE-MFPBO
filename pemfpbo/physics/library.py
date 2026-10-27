"""Classical polymer relations used as the constitutive prior library.

All functions are vectorised over designs. Time is in minutes, temperature
in kelvin, moduli in GPa, activation energies in J/mol.

    Eq. (8)   Fox rule for the glass transition of a blend
    Eq. (9)   Halpin-Tsai homogenisation
    Eq. (10)  Williams-Landel-Ferry shift factor
    Eq. (11)  reduced time along an arbitrary thermal path
    Eq. (12)  autocatalytic (Kamal) cure kinetics
    Eq. (13)  Avrami transformation statistics in reduced time
    Eq. (14)  residual stress as a convolution in reduced time
    Eq. (15)  Deborah, Fourier and Biot numbers and a reduced temperature
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

R_GAS = 8.314462618
T_ZERO_C = 273.15


# ----------------------------------------------------------------------
# Eq. (8): Fox rule
# ----------------------------------------------------------------------
def fox_tg(weights: Sequence[np.ndarray], tgs: Sequence[np.ndarray]) -> np.ndarray:
    """Glass transition of a blend, ``1/Tg = sum_i w_i / Tg_i`` (kelvin)."""
    inv = 0.0
    for w, tg in zip(weights, tgs):
        inv = inv + np.asarray(w) / np.asarray(tg)
    return 1.0 / inv


def fox_tg_conversion(alpha: np.ndarray, tg0: np.ndarray, tg_inf: np.ndarray) -> np.ndarray:
    """Fox rule applied to a partially cured network (unreacted/cured mixture)."""
    return fox_tg([1.0 - alpha, alpha], [tg0, tg_inf])


def dibenedetto_tg(alpha: np.ndarray, tg0: np.ndarray, tg_inf: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """DiBenedetto relation (used only inside the higher-fidelity solvers)."""
    return tg0 + (tg_inf - tg0) * lam * alpha / (1.0 - (1.0 - lam) * alpha)


def dibenedetto_inverse(tg: np.ndarray, tg0: np.ndarray, tg_inf: np.ndarray, lam: np.ndarray) -> np.ndarray:
    x = (tg - tg0) / (tg_inf - tg0)
    return x / (lam + x * (1.0 - lam))


# ----------------------------------------------------------------------
# Eq. (9): Halpin-Tsai
# ----------------------------------------------------------------------
def halpin_tsai(Em: np.ndarray, Ef: np.ndarray, phi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    ratio = Ef / Em
    eta = (ratio - 1.0) / (ratio + zeta)
    return Em * (1.0 + zeta * eta * phi) / (1.0 - eta * phi)


def maxwell_conductivity(km: float, kf: float, phi: np.ndarray) -> np.ndarray:
    beta = (kf - km) / (kf + 2.0 * km)
    return km * (1.0 + 2.0 * beta * phi) / (1.0 - beta * phi)


# ----------------------------------------------------------------------
# Eqs. (10)-(11): WLF shift and reduced time
# ----------------------------------------------------------------------
def wlf_log_shift(T: np.ndarray, Tref: np.ndarray, C1: float, C2: float,
                  cap: float = 100.0, Ea_glass: float = 2.5e5) -> np.ndarray:
    """log10 of the shift factor ``a_T``.

    The WLF form is used above ``Tref``; below it an Arrhenius continuation
    with the slope of the glassy state avoids the WLF singularity.
    """
    dT = np.minimum(T - Tref, cap)
    above = -C1 * dT / (C2 + np.maximum(dT, 0.0))
    below = Ea_glass / (np.log(10.0) * R_GAS) * (1.0 / np.maximum(T, 1.0) - 1.0 / Tref)
    return np.where(dT >= 0.0, above, below)


def reduced_time_increment(dt: np.ndarray, log_aT: np.ndarray) -> np.ndarray:
    return dt / np.power(10.0, log_aT)


def reduced_time(t: np.ndarray, log_aT: np.ndarray) -> np.ndarray:
    """Cumulative reduced time, Eq. (11), by the trapezoidal rule along axis 1."""
    inv = np.power(10.0, -log_aT)
    dt = np.diff(t, axis=1)
    inc = 0.5 * (inv[:, 1:] + inv[:, :-1]) * dt
    return np.concatenate([np.zeros((t.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)


# ----------------------------------------------------------------------
# Eq. (12): Kamal kinetics
# ----------------------------------------------------------------------
def arrhenius(A: np.ndarray, Ea: np.ndarray, T: np.ndarray) -> np.ndarray:
    return A * np.exp(-Ea / (R_GAS * T))


def kamal_rate(alpha: np.ndarray, k1: np.ndarray, k2: np.ndarray, m: float, n: float) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0)
    return (k1 + k2 * a ** m) * (1.0 - a) ** n


def kamal_step(alpha: np.ndarray, k1: np.ndarray, k2: np.ndarray, m: float, n: float,
               dt: np.ndarray, diffusion: np.ndarray | float = 1.0) -> np.ndarray:
    """Exponential-Euler step of Eq. (12) with a midpoint correction.

    The equation is written as ``d(1-a)/dt = -r(a) (1-a)`` with
    ``r = (k1 + k2 a^m)(1-a)^(n-1)`` frozen over the step, which is stable
    for any step length.
    """
    a = np.clip(alpha, 0.0, 0.99999)
    r0 = (k1 + k2 * a ** m) * (1.0 - a) ** (n - 1.0) * diffusion
    a_half = 1.0 - (1.0 - a) * np.exp(-0.5 * r0 * dt)
    r1 = (k1 + k2 * a_half ** m) * (1.0 - a_half) ** (n - 1.0) * diffusion
    return 1.0 - (1.0 - a) * np.exp(-r1 * dt)


# ----------------------------------------------------------------------
# Eq. (13): Avrami
# ----------------------------------------------------------------------
def avrami(xi: np.ndarray, K: np.ndarray, n: np.ndarray, X_inf: np.ndarray | float = 1.0) -> np.ndarray:
    return X_inf * (1.0 - np.exp(-K * np.power(np.maximum(xi, 0.0), n)))


# ----------------------------------------------------------------------
# Eq. (14): residual stress in reduced time
# ----------------------------------------------------------------------
def stress_step(sigma: np.ndarray, E: np.ndarray, d_eps: np.ndarray, d_xi: np.ndarray,
                tau: np.ndarray) -> np.ndarray:
    """One step of the Maxwell form of the convolution in Eq. (14).

    Exact for a strain rate that is constant in reduced time over the step.
    ``sigma`` and ``E`` share units (MPa if ``E`` is in MPa).
    """
    x = np.maximum(d_xi / tau, 1e-12)
    decay = np.exp(-x)
    gain = np.where(x > 1e-6, (1.0 - decay) / x, 1.0 - 0.5 * x)
    return sigma * decay + E * d_eps * gain


def prony_stress_step(sigmas: np.ndarray, E: np.ndarray, weights: np.ndarray, taus: np.ndarray,
                      d_eps: np.ndarray, d_xi: np.ndarray) -> np.ndarray:
    """Generalised Maxwell (Prony series) version of :func:`stress_step`.

    ``sigmas`` has a trailing axis over the Prony terms.
    """
    out = np.empty_like(sigmas)
    for j in range(sigmas.shape[-1]):
        out[..., j] = stress_step(sigmas[..., j], E * weights[j], d_eps, d_xi, taus[j])
    return out


# ----------------------------------------------------------------------
# Eq. (15): dimensionless groups
# ----------------------------------------------------------------------
def deborah(relax_time: np.ndarray, process_time: np.ndarray) -> np.ndarray:
    return relax_time / process_time


def fourier(diffusivity: np.ndarray, t: np.ndarray, L: float) -> np.ndarray:
    return diffusivity * t / L ** 2


def biot(h: np.ndarray, L: float, k: np.ndarray) -> np.ndarray:
    return h * L / k


def reduced_temperature(T: np.ndarray, Tg: np.ndarray) -> np.ndarray:
    return (T - Tg) / Tg


# ----------------------------------------------------------------------
# Thermal histories
# ----------------------------------------------------------------------
def ramp_dwell_schedule(T0: float, ramp1: np.ndarray, T_dwell: np.ndarray, t_dwell: np.ndarray,
                        T_peak: np.ndarray, t_post: np.ndarray, ramp2: float, cool: float,
                        n_seg: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-segment ramp and dwell cure cycle followed by controlled cooling.

    Returns time ``t`` (n, N), platen temperature ``T`` (n, N) and the
    segment index of every interval (N-1,).
    """
    n = np.asarray(ramp1).shape[0]
    durations = [
        (T_dwell - T0) / ramp1,
        t_dwell,
        np.maximum(T_peak - T_dwell, 0.0) / ramp2,
        t_post,
        (T_peak - T0) / cool,
    ]
    starts = [np.full(n, T0), T_dwell, T_dwell, T_peak, T_peak]
    ends = [T_dwell, T_dwell, T_peak, T_peak, np.full(n, T0)]
    frac = np.linspace(0.0, 1.0, n_seg + 1)[1:]
    t_cols, T_cols, seg = [np.zeros((n, 1))], [np.full((n, 1), T0)], []
    t_offset = np.zeros(n)
    for s, (dur, a, b) in enumerate(zip(durations, starts, ends)):
        t_cols.append(t_offset[:, None] + dur[:, None] * frac[None, :])
        T_cols.append(a[:, None] + (b - a)[:, None] * frac[None, :])
        seg.extend([s] * n_seg)
        t_offset = t_offset + dur
    return np.concatenate(t_cols, axis=1), np.concatenate(T_cols, axis=1), np.array(seg)


def mixture_density_fraction(w_filler: np.ndarray, rho_f: float, rho_m: float) -> np.ndarray:
    """Volume fraction of a filler from its weight fraction."""
    vf = w_filler / rho_f
    vm = (1.0 - w_filler) / rho_m
    return vf / (vf + vm)
