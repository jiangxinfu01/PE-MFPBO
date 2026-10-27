import unittest

import numpy as np

from pemfpbo.utils.pareto import hypervolume, hypervolume_mc, igd_plus, nondominated_mask


class TestPareto(unittest.TestCase):
    def test_hypervolume_2d(self):
        Y = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        self.assertAlmostEqual(hypervolume(Y, np.array([4.0, 4.0])), 6.0)

    def test_hypervolume_3d_box(self):
        Y = np.array([[0.0, 0.0, 0.0]])
        self.assertAlmostEqual(hypervolume(Y, np.array([1.0, 2.0, 3.0])), 6.0)

    def test_exact_vs_monte_carlo(self):
        rng = np.random.default_rng(0)
        Y = rng.random((30, 4))
        ref = np.full(4, 1.1)
        exact = hypervolume(Y, ref)
        mc = hypervolume_mc(Y[nondominated_mask(Y)], ref, n_mc=200_000, seed=1)
        self.assertLess(abs(exact - mc) / exact, 0.02)

    def test_nondominated(self):
        Y = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0], [1.0, 1.0]])
        mask = nondominated_mask(Y)
        self.assertTrue(mask[2])
        self.assertFalse(mask[1])
        self.assertEqual(int(mask[[0, 3]].sum()), 1)

    def test_igd_plus_zero_on_reference(self):
        R = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(igd_plus(R, R), 0.0)


if __name__ == "__main__":
    unittest.main()
