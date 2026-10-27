import unittest

import numpy as np

from pemfpbo.methods import make_method
from pemfpbo.problems import get_problem
from pemfpbo.runner import run

SMOKE = dict(n_samples=4, n_candidates=128, n_active=32, n_fit=24, n_virtual=8, fit_maxiter=8, refit_every=4,
             ea_offspring=8)


class TestRunner(unittest.TestCase):
    def test_budget_and_nesting(self):
        p = get_problem("mf_zdt1")
        m = make_method("PE-MFPBO", p, 0, **SMOKE)
        arch = run(p, m, budget=4.0, seed=0, init_sizes=(10, 4, 2), max_iter=8)
        a = arch.arrays()
        self.assertLessEqual(arch.total_cost, 4.0 + 1e-9)
        keys1 = {x.tobytes() for x in a["X"][a["level"] == 1]}
        for x in a["X"][a["level"] == 3]:
            self.assertIn(x.tobytes(), keys1)

    def test_single_fidelity_baseline(self):
        p = get_problem("mf_zdt1")
        m = make_method("NSGA-III", p, 0)
        arch = run(p, m, budget=6.0, seed=0, init_sizes=(10, 4, 2), max_iter=3)
        a = arch.arrays()
        self.assertTrue(np.all(a["level"][a["iteration"] >= 0] == 3))


if __name__ == "__main__":
    unittest.main()
