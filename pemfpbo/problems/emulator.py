"""Residual-network emulator of the level-three solver (Section 4.1).

Three hidden layers of 256 units with residual connections, trained by Adam
on Latin hypercube evaluations of the level-three solver. The emulator is a
generic function approximator fitted to solver output; it is not built from
the constitutive library.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .base import Problem


def encode(problem: Problem, X: np.ndarray) -> np.ndarray:
    sp = problem.space
    parts = [sp.cont_unit(X) * 2.0 - 1.0]
    for j, size in zip(sp.cat_idx, sp.cat_sizes):
        parts.append(np.eye(size)[X[:, j].astype(int)])
    return np.hstack(parts)


class ResidualMLP:
    def __init__(self, d_in: int, d_out: int, width: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        def init(a, b):
            return rng.standard_normal((a, b)) * np.sqrt(1.0 / a)
        self.p = {"W1": init(d_in, width), "b1": np.zeros(width),
                  "W2": init(width, width) * 0.5, "b2": np.zeros(width),
                  "W3": init(width, width) * 0.5, "b3": np.zeros(width),
                  "W4": init(width, d_out), "b4": np.zeros(d_out)}
        self.x_mu = self.y_mu = 0.0
        self.y_sd = 1.0

    def _forward(self, X):
        p = self.p
        h1 = np.tanh(X @ p["W1"] + p["b1"])
        u2 = np.tanh(h1 @ p["W2"] + p["b2"])
        h2 = h1 + u2
        u3 = np.tanh(h2 @ p["W3"] + p["b3"])
        h3 = h2 + u3
        return h3 @ p["W4"] + p["b4"], (X, h1, u2, h2, u3, h3)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 3000, lr: float = 2e-3, wd: float = 1e-5,
            batch: int = 128, seed: int = 0) -> "ResidualMLP":
        rng = np.random.default_rng(seed)
        self.y_mu, self.y_sd = Y.mean(0), Y.std(0) + 1e-12
        T = (Y - self.y_mu) / self.y_sd
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v) for k, v in self.p.items()}
        step = 0
        n = X.shape[0]
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                out, (x, h1, u2, h2, u3, h3) = self._forward(X[idx])
                dy = 2.0 * (out - T[idx]) / len(idx)
                p = self.p
                g = {"W4": h3.T @ dy, "b4": dy.sum(0)}
                dh3 = dy @ p["W4"].T
                dz3 = dh3 * (1 - u3 ** 2)
                g["W3"], g["b3"] = h2.T @ dz3, dz3.sum(0)
                dh2 = dh3 + dz3 @ p["W3"].T
                dz2 = dh2 * (1 - u2 ** 2)
                g["W2"], g["b2"] = h1.T @ dz2, dz2.sum(0)
                dh1 = dh2 + dz2 @ p["W2"].T
                dz1 = dh1 * (1 - h1 ** 2)
                g["W1"], g["b1"] = x.T @ dz1, dz1.sum(0)
                step += 1
                rate = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)) + 0.01)
                for k in p:
                    gk = g[k] + wd * p[k]
                    m[k] = 0.9 * m[k] + 0.1 * gk
                    v[k] = 0.999 * v[k] + 0.001 * gk ** 2
                    mh = m[k] / (1 - 0.9 ** step)
                    vh = v[k] / (1 - 0.999 ** step)
                    p[k] -= rate * mh / (np.sqrt(vh) + 1e-8)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[0] * self.y_sd + self.y_mu

    def save(self, path: str) -> None:
        np.savez_compressed(path, y_mu=self.y_mu, y_sd=self.y_sd, **self.p)

    @classmethod
    def load(cls, path: str) -> "ResidualMLP":
        z = np.load(path)
        obj = cls.__new__(cls)
        obj.p = {k: z[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")}
        obj.y_mu, obj.y_sd = z["y_mu"], z["y_sd"]
        return obj


def r2(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return 1.0 - ((y - yhat) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def nrmse(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return np.sqrt(((y - yhat) ** 2).mean(0)) / (y.max(0) - y.min(0))


def train_emulator(problem: Problem, n_train: int = 512, n_test: int = 128, seed: int = 0, epochs: int = 3000,
                   folds: int = 5) -> Tuple[ResidualMLP, Dict]:
    X = problem.space.lhs(n_train + n_test, seed=seed)
    Yo, G = problem.evaluate(X, problem.n_levels, noise=False)
    Y = np.hstack([Yo, G])
    E = encode(problem, X)
    Etr, Ytr, Ete, Yte = E[:n_train], Y[:n_train], E[n_train:], Y[n_train:]
    cv_pred = np.zeros_like(Ytr)
    fold = np.arange(n_train) % folds
    for k in range(folds):
        tr, va = fold != k, fold == k
        net = ResidualMLP(E.shape[1], Y.shape[1], seed=seed + k).fit(Etr[tr], Ytr[tr], epochs=epochs, seed=k)
        cv_pred[va] = net.predict(Etr[va])
    net = ResidualMLP(E.shape[1], Y.shape[1], seed=seed).fit(Etr, Ytr, epochs=epochs, seed=seed)
    report = {"cv_r2": r2(Ytr, cv_pred).tolist(), "heldout_nrmse": nrmse(Yte, net.predict(Ete)).tolist(),
              "outputs": problem.objective_names + problem.constraint_names, "n_train": n_train,
              "n_test": n_test}
    return net, report


class EmulatedProblem(Problem):
    """Wraps a problem so that the highest fidelity is served by an emulator."""

    def __init__(self, base: Problem, net: ResidualMLP):
        self.__dict__.update(base.__dict__)
        self.base, self.net = base, net
        self.name = base.name + "_emulated"

    def _evaluate(self, X: np.ndarray, level: int):
        if level < self.n_levels:
            return self.base._evaluate(X, level)
        out = self.net.predict(encode(self.base, X))
        return out[:, : self.base.M], out[:, self.base.M:]

    def library(self, X, theta):
        return self.base.library(X, theta)

    def analytic_prior(self, X):
        return self.base.analytic_prior(X)
