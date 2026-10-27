"""Budgeted optimisation loop with the nested evaluation rule of Eq. (37)."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .problems.base import Problem
from .utils.space import maximin_subset


class Archive:
    """All evaluations of one run, in evaluation order."""

    def __init__(self, problem: Problem):
        self.problem = problem
        self.X: List[np.ndarray] = []
        self.level: List[int] = []
        self.Y: List[np.ndarray] = []
        self.G: List[np.ndarray] = []
        self.cost: List[float] = []
        self.cum_cost: List[float] = []
        self.iteration: List[int] = []
        self.meta: Dict = {}

    @property
    def total_cost(self) -> float:
        return self.cum_cost[-1] if self.cum_cost else 0.0

    def add(self, x: np.ndarray, level: int, y: np.ndarray, g: np.ndarray, cost: float, it: int) -> None:
        self.X.append(np.asarray(x, dtype=float))
        self.level.append(int(level))
        self.Y.append(np.asarray(y, dtype=float))
        self.G.append(np.asarray(g, dtype=float))
        self.cost.append(float(cost))
        self.cum_cost.append(self.total_cost + float(cost))
        self.iteration.append(int(it))

    def arrays(self) -> Dict[str, np.ndarray]:
        d = self.problem.space.d
        M, J = self.problem.M, self.problem.J
        n = len(self.X)
        return {
            "X": np.array(self.X, dtype=float).reshape(n, d), "level": np.array(self.level, dtype=int),
            "Y": np.array(self.Y, dtype=float).reshape(n, M), "G": np.array(self.G, dtype=float).reshape(n, J),
            "cost": np.array(self.cost), "cum_cost": np.array(self.cum_cost),
            "iteration": np.array(self.iteration, dtype=int),
        }

    def data(self, levels: Optional[Sequence[int]] = None, constraints: bool = False
             ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        a = self.arrays()
        levels = range(1, self.problem.n_levels + 1) if levels is None else levels
        out = {}
        for l in levels:
            sel = a["level"] == l
            if sel.any():
                out[l] = (a["X"][sel], a["G"][sel] if constraints else a["Y"][sel])
        return out

    def count(self, level: int) -> int:
        return int(np.sum(np.array(self.level) == level))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **self.arrays())
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump(self.meta, f, indent=2, default=_jsonable)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def load_archive(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    meta_path = os.path.splitext(path)[0] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            out["meta"] = json.load(f)
    return out


def nested_initial_design(problem: Problem, sizes: Sequence[int], seed: int) -> Dict[int, np.ndarray]:
    """Latin hypercube at level one with maximin nested subsets at higher levels."""
    rng = np.random.default_rng(seed)
    X = problem.space.lhs(int(sizes[0]), seed=seed)
    designs = {1: X}
    for l in range(2, len(sizes) + 1):
        U = problem.space.to_unit(designs[l - 1])
        idx = maximin_subset(U, int(sizes[l - 1]), rng)
        designs[l] = designs[l - 1][idx]
    return designs


def candidate_pool(problem: Problem, n: int, seed: int, it: int) -> np.ndarray:
    """Candidate pool shared by all methods for a given seed and iteration."""
    return problem.space.sobol(n, seed=1_000_003 * (seed + 1) + it)


def run(problem: Problem, method, budget: float, seed: int, init_sizes: Sequence[int],
        max_iter: int = 100_000, verbose: bool = False, log_every: int = 25) -> Archive:
    rng = np.random.default_rng(seed)
    arch = Archive(problem)
    designs = nested_initial_design(problem, init_sizes, seed)
    for l, X in designs.items():
        Y, G = problem.evaluate(X, l, rng)
        for x, y, g in zip(X, Y, G):
            arch.add(x, l, y, g, problem.costs[l - 1], -1)
    method.initialize(arch)
    t0 = time.time()
    it = 0
    while arch.total_cost < budget - 1e-9 and it < max_iter:
        proposals = method.propose(arch, it)
        if not proposals:
            break
        evaluated = 0
        for x, l in proposals:
            lv = list(range(1, l + 1)) if method.multi_fidelity else [l]
            c = float(sum(problem.costs[j - 1] for j in lv))
            if arch.total_cost + c > budget + 1e-9:
                continue
            for j in lv:
                Y, G = problem.evaluate(x[None, :], j, rng)
                arch.add(x, j, Y[0], G[0], problem.costs[j - 1], it)
            evaluated += 1
        if evaluated == 0:
            break
        if verbose and it % log_every == 0:
            print(f"[{method.name}] it={it} cost={arch.total_cost:.2f} "
                  f"L3={arch.count(problem.n_levels)} t={time.time() - t0:.0f}s", flush=True)
        it += 1
    arch.meta.update({"method": method.name, "seed": seed, "budget": budget, "iterations": it,
                      "wall_time_s": time.time() - t0, "init_sizes": list(init_sizes)})
    arch.meta.update(method.diagnostics(arch))
    return arch
