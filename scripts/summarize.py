"""Metrics and statistics of Section 4.1 for all runs of one problem.

Normalised hypervolume is computed from feasible level-three observations
only. Objectives are mapped to minimisation and scaled by the ideal and nadir
of the reference front; the reference point is 1.1 times the worst value of
the union of all observations of all methods and seeds, then frozen.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import _common
from pemfpbo.problems import get_problem
from pemfpbo.runner import load_archive
from pemfpbo.utils.pareto import hypervolume, igd_plus


def holm(p: List[float]) -> List[float]:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney effect size: probability that a draw of x exceeds one of y."""
    gt = (x[:, None] > y[None, :]).mean()
    eq = (x[:, None] == y[None, :]).mean()
    return float(gt + 0.5 * eq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--results", default=os.path.join(_common.ROOT, "results"))
    ap.add_argument("--reference-method", default="PE-MFPBO")
    ap.add_argument("--target", type=float, default=0.65, help="normalised HV for cost to target")
    ap.add_argument("--tags", default="", help="comma-separated result directories (default: all)")
    args = ap.parse_args()

    problem = get_problem(args.problem)
    base = os.path.join(args.results, args.problem)
    ref = np.load(os.path.join(base, "reference_front.npz"))
    front = ref["front_min"]
    ideal, nadir = front.min(0), front.max(0)
    scale = np.maximum(nadir - ideal, 1e-12)
    norm = lambda Y: (problem.to_min(Y) - ideal) / scale
    tags = [t for t in args.tags.split(",") if t] or sorted(
        d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    runs: Dict[str, List[dict]] = {}
    worst = np.full(problem.M, -np.inf)
    L = problem.n_levels
    for tag in tags:
        for path in sorted(glob.glob(os.path.join(base, tag, "seed_*.npz"))):
            a = load_archive(path)
            sel = a["level"] == L
            feas = np.all(a["G"][sel] <= 0, axis=1) if a["G"].size else np.ones(sel.sum(), bool)
            Yn = norm(a["Y"][sel])
            worst = np.maximum(worst, Yn.max(0))
            runs.setdefault(tag, []).append({"seed": int(a["meta"]["seed"]), "a": a, "Yn": Yn, "feas": feas,
                                             "cost": a["cum_cost"][sel]})
    ref_pt = np.maximum(1.1 * worst, worst + 0.1)
    hv_ref = hypervolume((front - ideal) / scale, ref_pt)
    rows, traj = [], {}
    for tag, rr in runs.items():
        for r in rr:
            Yn, feas, cost = r["Yn"], r["feas"], r["cost"]
            hv_t = np.array([hypervolume(Yn[: i + 1][feas[: i + 1]], ref_pt) / hv_ref for i in range(len(Yn))])
            hit = np.where(hv_t >= args.target)[0]
            a = r["a"]
            lv = a["level"]
            rows.append({
                "method": tag, "seed": r["seed"], "hypervolume": hv_t[-1] if len(hv_t) else 0.0,
                "igd_plus": igd_plus(Yn[feas], (front - ideal) / scale) if feas.any() else np.nan,
                "cost_to_target": float(cost[hit[0]]) if hit.size else np.nan,
                "hf_calls": int(np.sum(lv == L)), "total_cost": float(a["cum_cost"][-1]),
                **{f"calls_L{l}": int(np.sum(lv == l)) for l in range(1, L + 1)},
                **{f"cost_L{l}": float(a["cost"][lv == l].sum()) for l in range(1, L + 1)},
            })
            traj.setdefault(tag, []).append(np.column_stack([cost, hv_t]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base, "runs.csv"), index=False)
    agg = df.groupby("method").agg(
        hv_mean=("hypervolume", "mean"), hv_sd=("hypervolume", "std"), igd_mean=("igd_plus", "mean"),
        igd_sd=("igd_plus", "std"), ctt=("cost_to_target", lambda s: s.mean() if s.notna().all() else np.nan),
        hf_calls=("hf_calls", "mean"), n_seeds=("seed", "count"),
        **{f"calls_L{l}": (f"calls_L{l}", "mean") for l in range(1, L + 1)},
        **{f"cost_L{l}": (f"cost_L{l}", "mean") for l in range(1, L + 1)})
    refm = args.reference_method
    if refm in runs:
        x = df[df.method == refm].set_index("seed")["hypervolume"]
        others = [t for t in agg.index if t != refm]
        raw = {}
        for t in others:
            y = df[df.method == t].set_index("seed")["hypervolume"]
            common = x.index.intersection(y.index)
            if len(common) >= 2 and np.any(x[common].values != y[common].values):
                raw[t] = float(wilcoxon(x[common].values, y[common].values, alternative="two-sided").pvalue)
            else:
                raw[t] = np.nan
            agg.loc[t, "A12"] = a12(x[common].values, y[common].values) if len(common) else np.nan
        valid = [t for t in others if np.isfinite(raw[t])]
        adj = dict(zip(valid, holm([raw[t] for t in valid])))
        for t in others:
            agg.loc[t, "raw_p"] = raw[t]
            agg.loc[t, "holm_p"] = adj.get(t, np.nan)
    agg.to_csv(os.path.join(base, "summary.csv"))
    np.savez_compressed(os.path.join(base, "trajectories.npz"),
                        **{t: np.array(v, dtype=object) for t, v in traj.items()})
    with open(os.path.join(base, "summary.md"), "w") as f:
        f.write(f"| Method | Hypervolume | IGD+ | CTT (HFE) | HF calls | Raw p | Holm p | A12 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for t, r in agg.sort_values("hv_mean").iterrows():
            ctt = "n.a." if not np.isfinite(r["ctt"]) else f"{r['ctt']:.1f}"
            if t == refm:
                stat = "ref. | ref. | ref."
            else:
                stat = f"{r.get('raw_p', np.nan):.2g} | {r.get('holm_p', np.nan):.2g} | {r.get('A12', np.nan):.2f}"
            f.write(f"| {t} | {r['hv_mean']:.4f} ± {r['hv_sd']:.4f} | {r['igd_mean']:.4f} ± {r['igd_sd']:.4f} | "
                    f"{ctt} | {r['hf_calls']:.0f} | {stat} |\n")
    print(open(os.path.join(base, "summary.md")).read())
    print(f"reference point (normalised): {np.round(ref_pt, 4)}")


if __name__ == "__main__":
    main()
