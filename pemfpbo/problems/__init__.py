"""Problem registry."""
from __future__ import annotations

from .base import Perturbation, Problem, Target
from .laminate import CopperCladLaminate
from .polypropylene import TalcPolypropylene
from .synthetic import MFDTLZ2, MFZDT1, MFZDT3, CureWarp4

PROBLEMS = {
    "laminate": CopperCladLaminate,
    "polypropylene": TalcPolypropylene,
    "mf_zdt1": MFZDT1,
    "mf_zdt3": MFZDT3,
    "mf_dtlz2": MFDTLZ2,
    "curewarp4": CureWarp4,
}


def get_problem(name: str, **kwargs) -> Problem:
    if name not in PROBLEMS:
        raise KeyError(f"unknown problem {name}; available: {sorted(PROBLEMS)}")
    return PROBLEMS[name](**kwargs)
