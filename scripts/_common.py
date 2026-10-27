"""Shared helpers for the command-line scripts."""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_yaml(name: str) -> Dict:
    with open(os.path.join(ROOT, "configs", name)) as f:
        return yaml.safe_load(f)


def parse_seeds(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_overrides(items: List[str]) -> Dict:
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def build_problem(name: str, emulator: str = ""):
    from pemfpbo.problems import get_problem
    problem = get_problem(name)
    if emulator:
        from pemfpbo.problems.emulator import EmulatedProblem, ResidualMLP
        problem = EmulatedProblem(problem, ResidualMLP.load(emulator))
    return problem


def run_dir(results: str, problem: str, tag: str) -> str:
    return os.path.join(results, problem, tag)
