"""Method registry."""
from __future__ import annotations

from typing import Dict

from .baselines import DGEMO, HVKGMF, MESMO, MFOSEMO, NSGA3, PIBOMO, PSLMOBO, QNEHVI, RESCUE, ParEGO
from .pemfpbo import ABLATIONS, PAPER_DEFAULTS, PEMFPBO, make_variant

BASELINES = {
    "ParEGO": ParEGO, "NSGA-III": NSGA3, "MESMO": MESMO, "DGEMO": DGEMO, "MF-OSEMO": MFOSEMO,
    "qNEHVI": QNEHVI, "PSL-MOBO": PSLMOBO, "HVKG-MF": HVKGMF, "PIBO-MO": PIBOMO, "RESCUE": RESCUE,
}
METHODS: Dict[str, type] = {"PE-MFPBO": PEMFPBO, **BASELINES}
METHODS.update({f"PE-MFPBO-{tag}": make_variant(tag) for tag in ABLATIONS})


def make_method(name: str, problem, seed: int, **opts):
    if name not in METHODS:
        raise KeyError(f"unknown method {name}; available: {sorted(METHODS)}")
    cls = METHODS[name]
    if cls is NSGA3:
        return cls(problem, seed, q=opts.get("q", 4))
    return cls(problem, seed, **opts)
