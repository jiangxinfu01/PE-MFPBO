"""Reference front from an exhaustive Sobol sweep of the highest fidelity."""
from __future__ import annotations

import argparse
import os

import numpy as np

import _common
from pemfpbo.utils.pareto import nondominated_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--emulator", default="")
    ap.add_argument("--results", default=os.path.join(_common.ROOT, "results"))
    args = ap.parse_args()
    problem = _common.build_problem(args.problem, args.emulator)
    X = problem.space.sobol(args.n, seed=31415)
    Ys, Gs = [], []
    for s in range(0, args.n, 2000):
        Y, G = problem.evaluate(X[s:s + 2000], problem.n_levels, noise=False)
        Ys.append(Y)
        Gs.append(G)
    Y, G = np.vstack(Ys), np.vstack(Gs)
    feas = np.all(G <= 0, axis=1) if G.size else np.ones(len(Y), dtype=bool)
    Ymin = problem.to_min(Y[feas])
    nd = nondominated_mask(Ymin)
    out = os.path.join(args.results, args.problem, "reference_front.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, X=X[feas][nd], Y=Y[feas][nd], front_min=Ymin[nd], n_sweep=args.n)
    print(f"{args.problem}: {feas.sum()} feasible of {args.n}, front size {nd.sum()} -> {out}")


if __name__ == "__main__":
    main()
