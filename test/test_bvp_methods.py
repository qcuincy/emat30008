from context import src
from src.bvp_methods.shoot import Shoot
import numpy as np
import unittest

def f(t, y):
    return np.array([y[1], -y[0]])

def bc(y_left, y_right):
    return [y_left - 1, y_right - 2]

class TestShoot(unittest.TestCase):
    def setUp(self):
        self.f = f
        self.y0 = np.array([0, 0])
        self.t0 = 0
        self.tf = 1
        self.dt = 0.1
        self.ic = None
        self.bc = None

    def test_solve(self):
        solver = Shoot(self.f, self.y0, self.t0, self.tf, self.dt, self.ic, self.bc)
        sol, y0 = solver.solve(bc=bc)
        self.assertAlmostEqual(y0[0], 1.0)
        self.assertAlmostEqual(y0[1], 2.0)

if __name__ == '__main__':
    unittest.main()
