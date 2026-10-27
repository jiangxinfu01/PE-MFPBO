"""Common interface of multi-fidelity, multi-objective design problems."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.space import Space


@dataclass
class Target:
    """One component of the engineering target region of Eq. (7)."""
    objective: int
    op: str  # "<=" or ">="
    value: float
    risk_adjusted: bool = True

    def satisfied(self, y: np.ndarray) -> np.ndarray:
        v = y[..., self.objective]
        return v <= self.value if self.op == "<=" else v >= self.value


@dataclass
class Perturbation:
    """Input perturbation model used for robust decoding, Eq. (38).

    ``kind`` is ``"normal"`` or ``"skewnormal"``; ``scale`` is in the natural
    units of the variable and ``skew`` is the sample skewness.
    """
    variable: int
    kind: str
    scale: float
    skew: float = 0.0


class Problem:
    """Base class. Sub-classes populate the attributes set in ``__init__``."""

    name: str = "problem"

    def __init__(self):
        self.space: Space = Space()
        self.n_levels: int = 3
        self.costs: np.ndarray = np.array([0.004, 0.031, 1.0])
        self.objective_names: List[str] = []
        self.senses: np.ndarray = np.zeros(0)  # +1 minimise, -1 maximise
        self.constraint_names: List[str] = []
        self.targets: List[Target] = []
        self.noise_sd: np.ndarray = np.zeros((3, 0))
        self.perturbations: List[Perturbation] = []
        # constitutive library
        self.has_library: bool = False
        self.theta_names: List[str] = []
        self.theta0: np.ndarray = np.zeros(0)
        self.theta_scale: np.ndarray = np.zeros(0)
        self.theta_bounds: np.ndarray = np.zeros((0, 2))
        self.z_names: List[str] = []
        self.basis_names: List[str] = []
        self.monotone: List[Tuple[int, int, int, int]] = []  # (objective, z index, sign, raw variable)
        self.objective_basis: List[int] = []  # library basis that predicts each objective

    # ------------------------------------------------------------------
    @property
    def M(self) -> int:
        return len(self.objective_names)

    @property
    def J(self) -> int:
        return len(self.constraint_names)

    @property
    def dz(self) -> int:
        return len(self.z_names)

    @property
    def K(self) -> int:
        return len(self.basis_names)

    def to_min(self, Y: np.ndarray) -> np.ndarray:
        """Objectives in minimisation form."""
        return np.asarray(Y) * self.senses

    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, level: int, rng: Optional[np.random.Generator] = None,
                 noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate designs at fidelity ``level`` (1-based).

        Returns objectives ``(n, M)`` and constraint values ``(n, J)`` with
        the convention ``g <= 0`` for feasibility.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y, G = self._evaluate(X, level)
        if noise and rng is not None:
            Y = Y + rng.standard_normal(Y.shape) * self.noise_sd[level - 1]
        return Y, G

    def _evaluate(self, X: np.ndarray, level: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def library(self, X: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Warped coordinates ``z`` (Eq. 16) and mean-function bases ``h`` (Eq. 17)."""
        raise NotImplementedError

    def analytic_prior(self, X: np.ndarray) -> np.ndarray:
        """Analytical (level-one) prediction used as the reference of Eq. (28)."""
        return self._evaluate(np.atleast_2d(X), 1)[0]

    def in_target(self, Y: np.ndarray) -> np.ndarray:
        ok = np.ones(Y.shape[:-1], dtype=bool)
        for t in self.targets:
            ok &= t.satisfied(Y)
        return ok

    def description(self) -> Dict:
        return {
            "name": self.name,
            "variables": [v.name for v in self.space.variables],
            "objectives": self.objective_names,
            "senses": self.senses.tolist(),
            "constraints": self.constraint_names,
            "costs": self.costs.tolist(),
        }
