"""Train the residual-network emulator of the level-three solver."""
from __future__ import annotations

import argparse
import json
import os

import _common
from pemfpbo.problems import get_problem
from pemfpbo.problems.emulator import train_emulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="laminate")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default=os.path.join(_common.ROOT, "results"))
    args = ap.parse_args()
    problem = get_problem(args.problem)
    net, report = train_emulator(problem, args.n_train, args.n_test, args.seed, args.epochs)
    out_dir = os.path.join(args.results, args.problem)
    os.makedirs(out_dir, exist_ok=True)
    net.save(os.path.join(out_dir, "emulator.npz"))
    with open(os.path.join(out_dir, "emulator_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    for name, r2, e in zip(report["outputs"], report["cv_r2"], report["heldout_nrmse"]):
        print(f"{name:28s} 5-fold CV R2 {r2:.3f}   held-out NRMSE {100 * e:.1f}%")


if __name__ == "__main__":
    main()
