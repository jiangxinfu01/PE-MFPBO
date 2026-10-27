"""Pareto dominance, hypervolume and IGD+ (all for minimisation)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def nondominated_mask(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of ``Y`` (minimisation)."""
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort(Y.T[::-1])
    Ys = Y[order]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        y = Ys[i]
        rest = Ys[i + 1:]
        dominated = np.all(y <= rest, axis=1)  # strictly dominated or an exact duplicate
        keep[i + 1:] &= ~dominated
    mask = np.zeros(n, dtype=bool)
    mask[order[keep]] = True
    return mask


def pareto_front(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    if Y.shape[0] == 0:
        return Y
    return Y[nondominated_mask(Y)]


def _hv2d(P: np.ndarray, ref: np.ndarray) -> float:
    P = P[np.argsort(P[:, 0])]
    vol, y_prev = 0.0, ref[1]
    for x, y in P:
        if y < y_prev:
            vol += (ref[0] - x) * (y_prev - y)
            y_prev = y
    return vol


def _hv_rec(P: np.ndarray, ref: np.ndarray) -> float:
    d = P.shape[1]
    if P.shape[0] == 0:
        return 0.0
    if d == 1:
        return float(ref[0] - P[:, 0].min())
    if d == 2:
        return _hv2d(P, ref)
    P = P[np.argsort(P[:, -1])]
    vol = 0.0
    for i in range(P.shape[0]):
        lo = P[i, -1]
        hi = P[i + 1, -1] if i + 1 < P.shape[0] else ref[-1]
        if hi <= lo:
            continue
        sub = P[: i + 1, :-1]
        sub = sub[nondominated_mask(sub)]
        vol += _hv_rec(sub, ref[:-1]) * (hi - lo)
    return vol


def hypervolume(Y: np.ndarray, ref: np.ndarray, exact_max: int = 250, n_mc: int = 400_000,
                seed: int = 0) -> float:
    """Dominated hypervolume of ``Y`` with respect to ``ref`` (minimisation).

    Exact slicing is used for fronts up to ``exact_max`` points; larger fronts
    use a fixed-seed Monte Carlo estimate.
    """
    Y = np.asarray(Y, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if Y.size == 0:
        return 0.0
    Y = Y[np.all(Y < ref, axis=1)]
    if Y.shape[0] == 0:
        return 0.0
    Y = Y[nondominated_mask(Y)]
    if Y.shape[0] <= exact_max:
        return float(_hv_rec(Y, ref))
    return hypervolume_mc(Y, ref, n_mc=n_mc, seed=seed)


def hypervolume_mc(Y: np.ndarray, ref: np.ndarray, n_mc: int = 400_000, seed: int = 0,
                   chunk: int = 4096) -> float:
    Y = np.asarray(Y, dtype=float)
    lo = Y.min(axis=0)
    box = float(np.prod(ref - lo))
    rng = np.random.default_rng(seed)
    hit = 0
    done = 0
    while done < n_mc:
        m = min(chunk, n_mc - done)
        U = lo + rng.random((m, Y.shape[1])) * (ref - lo)
        dom = np.zeros(m, dtype=bool)
        for start in range(0, Y.shape[0], 512):
            Yb = Y[start:start + 512]
            dom |= np.any(np.all(Yb[None, :, :] <= U[:, None, :], axis=2), axis=1)
        hit += int(dom.sum())
        done += m
    return box * hit / n_mc


def igd_plus(Y: np.ndarray, reference_front: np.ndarray) -> float:
    """Modified inverted generational distance IGD+ (minimisation)."""
    Y = np.asarray(Y, dtype=float)
    R = np.asarray(reference_front, dtype=float)
    if Y.shape[0] == 0:
        return float("inf")
    out = np.empty(R.shape[0])
    for start in range(0, R.shape[0], 1024):
        Rb = R[start:start + 1024]
        diff = np.maximum(Y[None, :, :] - Rb[:, None, :], 0.0)
        out[start:start + 1024] = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
    return float(out.mean())


class MCHypervolume:
    """Monte Carlo hypervolume on a fixed set of uniform points.

    Used inside acquisition functions where many hypervolume improvements
    must be compared on a common sample (common random numbers).
    """

    def __init__(self, lo: np.ndarray, ref: np.ndarray, n: int, rng: np.random.Generator):
        self.lo = np.asarray(lo, dtype=float)
        self.ref = np.asarray(ref, dtype=float)
        self.U = self.lo + rng.random((n, self.lo.size)) * (self.ref - self.lo)
        self.box = float(np.prod(self.ref - self.lo))

    def dominated(self, P: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(P)
        dom = np.zeros(self.U.shape[0], dtype=bool)
        for start in range(0, P.shape[0], 256):
            Pb = P[start:start + 256]
            dom |= np.any(np.all(Pb[None, :, :] <= self.U[:, None, :], axis=2), axis=1)
        return dom

    def volume(self, P: np.ndarray) -> float:
        if P.size == 0:
            return 0.0
        return self.box * float(self.dominated(P).mean())

    def improvement(self, base_dom: np.ndarray, Yc: np.ndarray) -> np.ndarray:
        """Hypervolume improvement of each row of ``Yc`` over a dominated mask."""
        free = ~base_dom
        Uf = self.U[free]
        if Uf.shape[0] == 0:
            return np.zeros(Yc.shape[0])
        out = np.empty(Yc.shape[0])
        for start in range(0, Yc.shape[0], 128):
            Yb = Yc[start:start + 128]
            cnt = np.all(Yb[:, None, :] <= Uf[None, :, :], axis=2).sum(axis=1)
            out[start:start + 128] = cnt
        return self.box * out / self.U.shape[0]


def das_dennis(n_partitions: int, n_dim: int) -> np.ndarray:
    """Uniformly spaced reference directions on the simplex."""
    def rec(left, depth):
        if depth == n_dim - 1:
            return [[left]]
        out = []
        for i in range(left + 1):
            for tail in rec(left - i, depth + 1):
                out.append([i] + tail)
        return out
    W = np.array(rec(n_partitions, 0), dtype=float) / n_partitions
    return W


def normalize_objectives(Y: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    span = np.where(np.abs(nadir - ideal) > 1e-12, nadir - ideal, 1.0)
    return (Y - ideal) / span


def random_simplex(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    W = rng.exponential(size=(n, m))
    return W / W.sum(axis=1, keepdims=True)


def chebyshev(Y: np.ndarray, w: np.ndarray, rho: float = 0.05) -> np.ndarray:
    """Augmented Chebyshev scalarisation (minimisation, normalised objectives)."""
    wy = Y * w
    return wy.max(axis=1) + rho * wy.sum(axis=1)


def front_or_empty(Y: Optional[np.ndarray], m: int) -> np.ndarray:
    if Y is None or len(Y) == 0:
        return np.zeros((0, m))
    return pareto_front(Y)
