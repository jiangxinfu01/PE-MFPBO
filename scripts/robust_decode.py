"""Robust inverse solution set (Eq. 39) and independent verification (Table 7).

The surrogate of a finished PE-MFPBO run is re-fitted on its archive, the
risk-adjusted moments are estimated from posterior sample paths under the
input perturbation model, admissible designs are decoded and representative
members are verified with the noise-free level-three solver.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import _common
from pemfpbo.methods import make_method
from pemfpbo.problems.base import Perturbation
from pemfpbo.robust.cvar import decode, representative
from pemfpbo.runner import Archive, load_archive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="laminate")
    ap.add_argument("--run", required=True, help="archive .npz of a PE-MFPBO run")
    ap.add_argument("--profile", default="paper")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-pert", type=int, default=512)
    ap.add_argument("--risk", type=float, default=0.05)
    ap.add_argument("--rho", type=float, default=0.10)
    ap.add_argument("--perturbations", default="", help="JSON written by fit_perturbation.py")
    ap.add_argument("--emulator", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    problem = _common.build_problem(args.problem, args.emulator)
    truth = _common.build_problem(args.problem)
    if args.perturbations:
        names = problem.space.names()
        with open(args.perturbations) as f:
            problem.perturbations = [Perturbation(names.index(d["variable"]), d["kind"], d["scale"], d["skew"])
                                     for d in json.load(f)]
    a = load_archive(args.run)
    arch = Archive(problem)
    for x, l, y, g, c, it in zip(a["X"], a["level"], a["Y"], a["G"], a["cost"], a["iteration"]):
        arch.add(x, l, y, g, c, it)
    opts = dict(_common.load_yaml("profiles.yaml").get(args.profile) or {})
    seed = int(a["meta"]["seed"]) if "meta" in a else 0
    method = make_method("PE-MFPBO", problem, seed, **opts)
    method.initialize(arch)
    method.update_models(arch, 0)
    res = decode(method, problem, a["X"][a["level"] == problem.n_levels], rho=args.rho, a=args.risk,
                 n_pert=args.n_pert, seed=seed)
    adm = np.where(res["admissible"])[0]
    print(f"candidates {len(res['X'])}, admissible (attainment >= {1 - args.rho:.2f}): {len(adm)}")
    if adm.size == 0:
        return
    pick = adm[representative(problem, res["risk_mean"][adm], args.k)]
    Yv, Gv = truth.evaluate(res["X"][pick], truth.n_levels, noise=False)
    names = problem.space.names()
    rows = []
    for i, j in enumerate(pick):
        row = {"ID": f"D{i + 1}"}
        for v in problem.space.variables:
            col = names.index(v.name)
            row[v.name] = v.levels[int(res["X"][j, col])] if v.kind == "categorical" else res["X"][j, col]
        for m, name in enumerate(problem.objective_names):
            row[f"{name}_predicted"] = res["nominal"][j, m]
            row[f"{name}_cvar"] = res["risk_mean"][j, m]
            row[f"{name}_verified"] = Yv[i, m]
            row[f"{name}_error_pct"] = 100.0 * (Yv[i, m] - res["nominal"][j, m]) / Yv[i, m]
        row["attainment"] = res["attainment"][j]
        row["verified_in_target"] = bool(problem.in_target(Yv[i:i + 1])[0])
        row["verified_feasible"] = bool(np.all(Gv[i] <= 0)) if Gv.size else True
        rows.append(row)
    df = pd.DataFrame(rows)
    out = args.out or os.path.join(os.path.dirname(args.run), "robust_designs.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.precision", 4):
        print(df)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
