"""Run one or more methods over several seeds on one problem.

Examples
--------
python scripts/run_experiment.py --problem laminate --methods all --seeds 0-19
python scripts/run_experiment.py --problem laminate --methods ablations --seeds 0-19
python scripts/run_experiment.py --problem curewarp4 --methods PE-MFPBO --set eta=0.2 --tag eta0.20_kappa1.0
"""
from __future__ import annotations

import argparse
import os
import time

import _common
from pemfpbo.methods import ABLATIONS, BASELINES, make_method
from pemfpbo.runner import run


def expand_methods(spec: str):
    if spec == "all":
        return ["PE-MFPBO"] + list(BASELINES)
    if spec == "baselines":
        return list(BASELINES)
    if spec == "ablations":
        return ["PE-MFPBO"] + [f"PE-MFPBO-{t}" for t in ABLATIONS]
    return [m.strip() for m in spec.split(",") if m.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--methods", default="PE-MFPBO")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--profile", default="paper")
    ap.add_argument("--set", nargs="*", default=[], help="option overrides key=value")
    ap.add_argument("--tag", default="", help="result sub-directory (defaults to the method name)")
    ap.add_argument("--emulator", default="", help="emulator .npz serving the highest fidelity")
    ap.add_argument("--results", default=os.path.join(_common.ROOT, "results"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = _common.load_yaml("problems.yaml")[args.problem]
    budget = args.budget if args.budget is not None else cfg["budget"]
    opts = dict(_common.load_yaml("profiles.yaml").get(args.profile) or {})
    opts.update(_common.parse_overrides(args.set))
    problem = _common.build_problem(args.problem, args.emulator)
    for name in expand_methods(args.methods):
        tag = args.tag or name
        out_dir = _common.run_dir(args.results, args.problem, tag)
        for seed in _common.parse_seeds(args.seeds):
            path = os.path.join(out_dir, f"seed_{seed:02d}.npz")
            if os.path.exists(path) and not args.overwrite:
                print(f"skip {path}")
                continue
            t0 = time.time()
            method = make_method(name, problem, seed, **opts)
            arch = run(problem, method, budget, seed, cfg["init"], verbose=args.verbose)
            arch.meta.update({"problem": args.problem, "profile": args.profile, "options": opts, "tag": tag})
            arch.save(path)
            print(f"{tag:22s} seed {seed:2d}  cost {arch.total_cost:7.2f}  "
                  f"L3 calls {arch.count(problem.n_levels):4d}  {time.time() - t0:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
