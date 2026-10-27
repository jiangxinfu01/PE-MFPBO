import unittest

import numpy as np

from pemfpbo.models.ep import ep_probit
from pemfpbo.models.levelgp import LevelGP, Rows, expand_rows


def _num_grad(f, p, eps=1e-6):
    g = np.zeros_like(p)
    for i in range(p.size):
        a, b = p.copy(), p.copy()
        a[i] += eps
        b[i] -= eps
        g[i] = (f(a) - f(b)) / (2 * eps)
    return g


class TestLevelGP(unittest.TestCase):
    def _check(self, first, coupling, rank):
        rng = np.random.default_rng(0)
        P, M, dz, K = 10, 3, 3, 2
        Z, C = rng.standard_normal((P, dz)), rng.integers(0, 3, (P, 1))
        F = None if first else rng.standard_normal((P, M))
        gp = LevelGP(M, dz, [3], K, first=first, coupling=coupling, icm_rank=rank)
        p = gp.init_vector(rng, 0.1 * rng.standard_normal((M, K)), np.full(M, 1e-2))
        p += 0.1 * rng.standard_normal(p.size)
        rows = expand_rows(Z, C, M, F)
        Hr = np.repeat(rng.standard_normal((P, K)), M, axis=0)
        y = rng.standard_normal(P * M)
        _, g, _ = gp.mll(p, rows, y, Hr)
        gn = _num_grad(lambda q: gp.mll(q, rows, y, Hr)[0], p)
        self.assertLess(np.max(np.abs(g - gn) / (1 + np.abs(gn))), 1e-5)

    def test_gradient_level_one(self):
        self._check(True, "nonlinear", 2)

    def test_gradient_nonlinear_coupling(self):
        self._check(False, "nonlinear", 2)

    def test_gradient_linear_coupling(self):
        self._check(False, "linear", 0)

    def test_derivative_kernel(self):
        rng = np.random.default_rng(1)
        gp = LevelGP(2, 3, [], 1, first=True, icm_rank=1)
        p = gp.init_vector(rng, np.zeros((2, 1)), np.full(2, 1e-2))
        A = Rows(rng.standard_normal((4, 3)), np.zeros((4, 0), int), rng.integers(0, 2, 4))
        D = Rows(rng.standard_normal((3, 3)), np.zeros((3, 0), int), rng.integers(0, 2, 3), None,
                 rng.integers(0, 3, 3))
        kvd = gp.k_vd(p, A, D)
        eps = 1e-6
        for b in range(3):
            Zp, Zm = D.Z.copy(), D.Z.copy()
            Zp[b, D.J[b]] += eps
            Zm[b, D.J[b]] -= eps
            num = (gp.k_vv(p, A, Rows(Zp, D.C, D.O))[:, b] - gp.k_vv(p, A, Rows(Zm, D.C, D.O))[:, b]) / (2 * eps)
            np.testing.assert_allclose(kvd[:, b], num, atol=1e-7)


class TestEP(unittest.TestCase):
    def test_sign_constraint_moves_posterior(self):
        mu0 = np.array([-0.5, 0.2])
        S0 = np.array([[1.0, 0.3], [0.3, 1.0]])
        tau, mt = ep_probit(mu0, S0, np.array([1.0, 1.0]), nu=0.1)
        Sigma = np.linalg.inv(np.linalg.inv(S0) + np.diag(tau))
        mu = Sigma @ (np.linalg.solve(S0, mu0) + tau * mt)
        self.assertTrue(np.all(mu > 0))


if __name__ == "__main__":
    unittest.main()
