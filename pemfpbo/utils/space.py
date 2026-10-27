"""Mixed continuous/categorical design spaces.

A design matrix ``X`` has shape ``(n, d)`` and stores continuous variables in
natural units and categorical variables as integer codes (stored as floats).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.stats import qmc


@dataclass
class Variable:
    name: str
    kind: str  # "continuous" or "categorical"
    low: float = 0.0
    high: float = 1.0
    levels: Optional[Sequence[str]] = None
    unit: str = ""
    block: str = "structure"  # "structure" or "process"

    @property
    def n_levels(self) -> int:
        return 0 if self.levels is None else len(self.levels)


@dataclass
class Space:
    variables: List[Variable] = field(default_factory=list)

    @property
    def d(self) -> int:
        return len(self.variables)

    @property
    def cont_idx(self) -> np.ndarray:
        return np.array([i for i, v in enumerate(self.variables) if v.kind == "continuous"], dtype=int)

    @property
    def cat_idx(self) -> np.ndarray:
        return np.array([i for i, v in enumerate(self.variables) if v.kind == "categorical"], dtype=int)

    @property
    def cat_sizes(self) -> List[int]:
        return [self.variables[i].n_levels for i in self.cat_idx]

    @property
    def lows(self) -> np.ndarray:
        return np.array([self.variables[i].low for i in self.cont_idx], dtype=float)

    @property
    def highs(self) -> np.ndarray:
        return np.array([self.variables[i].high for i in self.cont_idx], dtype=float)

    def names(self) -> List[str]:
        return [v.name for v in self.variables]

    # ------------------------------------------------------------------
    # Conversions between the unit hypercube and natural units
    # ------------------------------------------------------------------
    def from_unit(self, U: np.ndarray) -> np.ndarray:
        """Map points of the unit hypercube ``[0, 1]^d`` to designs."""
        U = np.atleast_2d(np.asarray(U, dtype=float))
        X = np.empty_like(U)
        for i, v in enumerate(self.variables):
            if v.kind == "continuous":
                X[:, i] = v.low + U[:, i] * (v.high - v.low)
            else:
                X[:, i] = np.minimum(np.floor(U[:, i] * v.n_levels), v.n_levels - 1)
        return X

    def to_unit(self, X: np.ndarray) -> np.ndarray:
        """Map designs to the unit hypercube (categoricals to bin centres)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        U = np.empty_like(X)
        for i, v in enumerate(self.variables):
            if v.kind == "continuous":
                U[:, i] = (X[:, i] - v.low) / (v.high - v.low)
            else:
                U[:, i] = (X[:, i] + 0.5) / v.n_levels
        return U

    def cont_unit(self, X: np.ndarray) -> np.ndarray:
        """Continuous coordinates of ``X`` scaled to ``[0, 1]``."""
        X = np.atleast_2d(X)
        idx = self.cont_idx
        return (X[:, idx] - self.lows) / (self.highs - self.lows)

    def cats(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return X[:, self.cat_idx].astype(int)

    def clip(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float, copy=True)
        for i, v in enumerate(self.variables):
            if v.kind == "continuous":
                X[:, i] = np.clip(X[:, i], v.low, v.high)
            else:
                X[:, i] = np.clip(np.round(X[:, i]), 0, v.n_levels - 1)
        return X

    # ------------------------------------------------------------------
    # Space-filling designs
    # ------------------------------------------------------------------
    def sobol(self, n: int, seed: int) -> np.ndarray:
        sampler = qmc.Sobol(d=self.d, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n, 2))))
        U = sampler.random_base2(m)[:n]
        return self.from_unit(U)

    def lhs(self, n: int, seed: int) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.d, seed=seed)
        return self.from_unit(sampler.random(n))

    def uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.from_unit(rng.random((n, self.d)))


def maximin_subset(U: np.ndarray, k: int, rng: np.random.Generator, first: Optional[int] = None) -> np.ndarray:
    """Greedy max-min distance subset of the rows of ``U``; returns indices."""
    n = U.shape[0]
    if k >= n:
        return np.arange(n)
    start = int(rng.integers(n)) if first is None else int(first)
    chosen = [start]
    dmin = np.sum((U - U[start]) ** 2, axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, np.sum((U - U[j]) ** 2, axis=1))
    return np.array(chosen, dtype=int)
