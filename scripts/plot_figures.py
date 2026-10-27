"""Figures 2-5 from the outputs of summarize.py."""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common

BASELINE_ORDER = ["ParEGO", "NSGA-III", "MESMO", "DGEMO", "MF-OSEMO", "qNEHVI", "PSL-MOBO", "HVKG-MF",
                  "PIBO-MO", "RESCUE", "PE-MFPBO"]
ABL = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]


def _summary(results, problem):
    return pd.read_csv(os.path.join(results, problem, "summary.csv"), index_col=0)


def fig_benchmarks(results, problems, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, prob in zip(axes.ravel(), problems):
        s = _summary(results, prob)
        names = [m for m in BASELINE_ORDER if m in s.index]
        vals = s.loc[names, "hv_mean"].values
        colors = ["#b5413b" if n == "PE-MFPBO" else "#2f5d7c" for n in names]
        ax.barh(names, vals, color=colors)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:.4f}", va="center", fontsize=7)
        ax.set_title(prob, loc="right", style="italic")
        ax.invert_yaxis()
    for ax in axes[-1]:
        ax.set_xlabel("Normalized hypervolume")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def fig_allocation(results, problem, out):
    s = _summary(results, problem)
    names = [m for m in ["MF-OSEMO", "HVKG-MF", "RESCUE", "PE-MFPBO"] if m in s.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    cols = ["#1f3a5f", "#2e7d8c", "#e0a845"]
    for k, l in enumerate((1, 2, 3)):
        a1.barh(np.arange(len(names)) + 0.25 * (k - 1), s.loc[names, f"calls_L{l}"], height=0.25,
                color=cols[k], label=f"Level {l}")
    a1.set_xscale("log")
    a1.set_yticks(range(len(names)), names)
    a1.set_xlabel("Number of calls (log scale)")
    a1.legend(frameon=False)
    tot = sum(s.loc[names, f"cost_L{l}"] for l in (1, 2, 3))
    left = np.zeros(len(names))
    for k, l in enumerate((1, 2, 3)):
        share = 100 * s.loc[names, f"cost_L{l}"] / tot
        a2.barh(names, share, left=left, color=cols[k])
        left += share.values
    a2.set_xlabel("Share of the budget (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def fig_ablation(results, problem, out):
    s = _summary(results, problem)
    names = [f"PE-MFPBO-{a}" for a in ABL if f"PE-MFPBO-{a}" in s.index] + ["PE-MFPBO"]
    full = s.loc["PE-MFPBO", "hv_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    y = np.arange(len(names))
    axes[0].errorbar(s.loc[names, "hv_mean"], y, xerr=s.loc[names, "hv_sd"], fmt="o")
    axes[0].axvline(full, ls="--", color="r")
    for yi, n in enumerate(names[:-1]):
        axes[0].text(full, yi, f" {100 * (s.loc[n, 'hv_mean'] - full) / full:+.2f}%", fontsize=7, va="center")
    axes[1].errorbar(s.loc[names, "igd_mean"], y, xerr=s.loc[names, "igd_sd"], fmt="s")
    axes[2].barh(y, s.loc[names, "hf_calls"])
    axes[0].set_yticks(y, [n.replace("PE-MFPBO-", "") for n in names[:-1]] + ["Full"])
    for ax, lab in zip(axes, ["Hypervolume", "IGD+", "Level 3 calls"]):
        ax.set_xlabel(lab)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def fig_sensitivity(results, problem, out):
    grid = _common.load_yaml("sensitivity.yaml")
    base = os.path.join(results, problem)
    runs = pd.read_csv(os.path.join(base, "runs.csv"))
    H = np.full((len(grid["eta"]), len(grid["kappa"])), np.nan)
    for i, e in enumerate(grid["eta"]):
        for j, k in enumerate(grid["kappa"]):
            sel = runs[runs.method == f"sens_eta{e:.2f}_kappa{k:.1f}"]
            if len(sel):
                H[i, j] = sel.hypervolume.mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for j, k in enumerate(grid["kappa"]):
        a1.plot(range(len(grid["eta"])), H[:, j], marker="o", label=f"kappa={k}")
    a1.set_xticks(range(len(grid["eta"])), [f"{e:.2f}" for e in grid["eta"]])
    a1.set_xlabel("Disagreement weight eta")
    for i, e in enumerate(grid["eta"]):
        a2.plot(grid["kappa"], H[i], marker="o", label=f"eta={e}")
    a2.set_xlabel("Cost exponent kappa")
    for ax in (a1, a2):
        ax.set_ylabel("Hypervolume")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(_common.ROOT, "results"))
    ap.add_argument("--figures", default=os.path.join(_common.ROOT, "figures"))
    ap.add_argument("--which", default="all")
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    jobs = {
        "benchmarks": lambda: fig_benchmarks(args.results, ["mf_zdt1", "mf_zdt3", "mf_dtlz2", "curewarp4"],
                                             os.path.join(args.figures, "fig2_benchmarks.png")),
        "allocation": lambda: fig_allocation(args.results, "laminate", os.path.join(args.figures, "fig3_allocation.png")),
        "ablation": lambda: fig_ablation(args.results, "laminate", os.path.join(args.figures, "fig4_ablation.png")),
        "sensitivity": lambda: fig_sensitivity(args.results, "laminate", os.path.join(args.figures, "fig5_sensitivity.png")),
    }
    for name, job in jobs.items():
        if args.which in ("all", name):
            try:
                job()
                print(f"figure {name} written")
            except FileNotFoundError as e:
                print(f"figure {name} skipped: {e}")


if __name__ == "__main__":
    main()
