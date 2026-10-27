import unittest

import numpy as np

from pemfpbo.problems import PROBLEMS, get_problem


class TestProblems(unittest.TestCase):
    def test_shapes_and_costs(self):
        for name in PROBLEMS:
            p = get_problem(name)
            X = p.space.sobol(16, seed=0)
            for level in (1, 2, 3):
                Y, G = p.evaluate(X, level, noise=False)
                self.assertEqual(Y.shape, (16, p.M), name)
                self.assertEqual(G.shape, (16, p.J), name)
                self.assertTrue(np.all(np.isfinite(Y)), name)
            self.assertTrue(np.all(np.diff(p.costs) > 0))

    def test_library(self):
        for name in ("laminate", "polypropylene", "curewarp4"):
            p = get_problem(name)
            X = p.space.sobol(8, seed=1)
            z, h = p.library(X, p.theta0)
            self.assertEqual(z.shape, (8, p.dz))
            self.assertEqual(h.shape, (8, p.K))
            T = np.repeat(p.theta0[None, :], 8, axis=0)
            z2, h2 = p.library(X, T)
            np.testing.assert_allclose(z, z2)

    def test_level_hierarchy_is_informative(self):
        p = get_problem("laminate")
        X = p.space.sobol(128, seed=2)
        Y1, _ = p.evaluate(X, 1, noise=False)
        Y3, _ = p.evaluate(X, 3, noise=False)
        corr = [np.corrcoef(Y1[:, m], Y3[:, m])[0, 1] for m in range(p.M)]
        self.assertTrue(min(corr) > 0.8)


if __name__ == "__main__":
    unittest.main()
