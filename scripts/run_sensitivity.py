"""Disagreement weight / cost exponent grid of Section 4.7."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import _common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="laminate")
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--profile", default="paper")
    ap.add_argument("--emulator", default="")
    ap.add_argument("--results", default=os.path.join(_common.ROOT, "results"))
    args = ap.parse_args()
    grid = _common.load_yaml("sensitivity.yaml")
    script = os.path.join(_common.ROOT, "scripts", "run_experiment.py")
    for eta in grid["eta"]:
        for kappa in grid["kappa"]:
            tag = f"sens_eta{eta:.2f}_kappa{kappa:.1f}"
            cmd = [sys.executable, script, "--problem", args.problem, "--methods", "PE-MFPBO", "--seeds",
                   args.seeds, "--profile", args.profile, "--tag", tag, "--results", args.results,
                   "--set", f"eta={eta}", f"kappa={kappa}"]
            if args.emulator:
                cmd += ["--emulator", args.emulator]
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
