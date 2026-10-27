"""Fit the input-perturbation model of Eq. (38) to logged process deviations.

The CSV must contain one column per perturbed variable (named as in the
problem definition) holding logged deviations from the set point.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import _common
from pemfpbo.problems import get_problem
from pemfpbo.robust.cvar import fit_perturbation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="laminate")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="perturbations.json")
    args = ap.parse_args()
    problem = get_problem(args.problem)
    names = problem.space.names()
    df = pd.read_csv(args.csv)
    fitted = []
    for col in df.columns:
        if col not in names:
            continue
        p = fit_perturbation(df[col].to_numpy(dtype=float), names.index(col))
        fitted.append({"variable": col, "kind": p.kind, "scale": p.scale, "skew": p.skew})
        print(f"{col:24s} {p.kind:10s} scale {p.scale:.4g} skew {p.skew:+.3f}")
    with open(args.out, "w") as f:
        json.dump(fitted, f, indent=2)


if __name__ == "__main__":
    main()
