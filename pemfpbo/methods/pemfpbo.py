"""PE-MFPBO: physics-warped multi-fidelity GP with cost-aware Pareto front
entropy search, disagreement bonus and determinantal batch selection."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..acquisition.core import copf, dpp_greedy, inner_evolution, pfes, top_q_distinct
from ..runner import Archive
from .base import ModelBased

PAPER_DEFAULTS = dict(
    warp="physics", learn_warp=True, mean="physics", monotone=True, coupling="nonlinear", icm_rank=2,
    single_fidelity=False, n_active=256, n_fit=128, n_virtual=96, q=4, n_samples=64, n_candidates=4096,
    kappa=1.0, eta=0.10, refit_every=10, batch="dpp",
)

# Ablation variants of Section 4.6 (Figure 4)
ABLATIONS: Dict[str, Dict] = {
    "A1": dict(warp="identity"),           # no physics warping
    "A2": dict(mean="constant"),           # no physics mean
    "A3": dict(monotone=False),            # no monotonicity constraints
    "A4": dict(eta=0.0),                   # no disagreement bonus
    "A5": dict(single_fidelity=True),      # level three only
    "A6": dict(coupling="linear"),         # linear autoregressive coupling
    "A7": dict(batch="random"),            # random batch from the acquisition
    "A8": dict(icm_rank=0),                # independent GPs
    "A9": dict(warp="random"),             # non-physical map, same output dimension and parameter count
    "A10": dict(learn_warp=False),         # warping parameters fixed at handbook values
}


class PEMFPBO(ModelBased):
    name = "PE-MFPBO"
    defaults = PAPER_DEFAULTS

    def propose(self, archive: Archive, it: int) -> List[Tuple[np.ndarray, int]]:
        a = self.acfg
        self.update_models(archive, it)
        m = self.model
        pool = self.pool(it)
        paths = m.sample_paths(a.n_samples, self.rng)
        samples, preds = paths.evaluate_with_predictions(pool)
        fstar = inner_evolution(lambda X: paths(X) * self.to_max, self.problem.space, pool,
                                samples * self.to_max, self.rng, a.ea_generations, a.ea_offspring,
                                weight_fn=self.feasibility if self.cmodel is not None else None)
        levels = m.levels
        alpha = np.column_stack([pfes(preds[l][0] * self.to_max, np.sqrt(preds[l][1]), fstar) for l in levels])
        mu_top, var_top = preds[m.top]
        tau = m.trust(pool, mu_top, var_top) if a.eta > 0 else np.zeros(len(pool))
        sigbar = np.sqrt(var_top).mean(axis=1)
        pf = self.feasibility(pool)
        scores = copf(alpha, self.charged_costs(), a.kappa, a.eta, tau, sigbar, pf)
        if scores.max() < a.tol:
            return []
        flat = scores.reshape(-1)
        ids = np.repeat(np.arange(len(pool)), len(levels))
        lv = np.tile(np.array(levels), len(pool))
        if a.batch == "dpp":
            feats = np.repeat(self.lengthscale_features(pool), len(levels), axis=0)
            chosen = dpp_greedy(flat, feats, lv, a.q, ids, level_corr=a.level_corr)
        elif a.batch == "random":
            top = np.argsort(-flat)[:256]
            w = np.maximum(flat[top], 1e-12)
            chosen, seen = [], set()
            for j in self.rng.choice(top, size=len(top), replace=False, p=w / w.sum()):
                if ids[j] not in seen:
                    chosen.append(int(j))
                    seen.add(ids[j])
                if len(chosen) == a.q:
                    break
        else:
            chosen = top_q_distinct(flat, ids, self.problem.space.to_unit(pool), a.q)
        return [(pool[ids[j]], int(lv[j])) for j in chosen]


def make_variant(tag: str):
    """PE-MFPBO with the overrides of an ablation ``tag`` (e.g. ``"A1"``)."""
    overrides = ABLATIONS[tag]

    class Variant(PEMFPBO):
        name = f"PE-MFPBO-{tag}"
        defaults = {**PAPER_DEFAULTS, **overrides}

    Variant.__name__ = f"PEMFPBO_{tag}"
    return Variant
