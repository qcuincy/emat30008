from .context import src

from src.pde_utils.grid import Grid
from src.solution import Solution
from src.problem import Problem
from src.solver import Solver
from src.ivp import IVP
from src.bvp import BVP
import numpy as np
import unittest


class TestIVP(unittest.TestCase):

    def test_ode_solver(self):
        def f(x, u):
            return [u[1], -u[0]]
        y0 = [1, 0]
        t0 = 0
        tf = 10
        Nt = 100
        problem = Problem(f=f, y0=y0, t0=t0, tf=tf, Nt=Nt)
        ivp = IVP(problem)
        sol = ivp.solve()
        # Check that the solution is of the correct type
        self.assertIsInstance(sol, Solution)
        # Check that the time and solution arrays are of the correct shape
        self.assertEqual(sol.t.shape, (101,))
        self.assertEqual(sol.y.shape, (101,2))
        # Check that the solution is correct at the initial time
        self.assertEqual(sol.y[0,0], float(y0[0]))
        self.assertEqual(sol.y[0,1], float(y0[1]))

    def test_pde_solver(self):
        def q(t, x, u):
            return u
        
        def ic(x):
            return np.sin(np.pi*x)
        
        t0 = 0
        tf = 10
        Nt = 100
        a = 0
        b = 1
        Nx = 100
        C = 0

        problem = Problem(q=q, ic=ic, t0=t0, tf=tf, Nt=Nt, a=a, b=b, Nx=Nx, C=C)
        ivp = IVP(problem)
        sol = ivp.solve()
        # Check that the solution is of the correct type
        self.assertIsInstance(sol, Solution)
        # Check that the time and solution arrays are of the correct shape
        self.assertEqual(sol.t.shape, (101,))
        self.assertEqual(sol.u.shape, (101, 101))
        # Check that the solution is correct at the initial time
        self.assertEqual(sol.u[0,0], float(ic(a)))
        self.assertEqual(sol.u[0,-1], float(ic(b)))

    def test_ode_continuation_solver(self):
        def q_cont(x, u, p):
            return [u[1], -u[0] + p]

        p0 = 0
        p_span = [0, 10]
        problem = Problem(f=q_cont, y0=[1, 0], t0=0, tf=10, Nt=100, args=(p0,))
        ivp = IVP(problem)
        sols = ivp.solve(p0=p0, p_span=p_span, ds=0.1)
        # Check that the solution is of the correct type
        self.assertIsInstance(sols, Solution)
        # Check that the time and solution arrays are of the correct shape
        self.assertEqual(sols.t.shape, (101,))
        self.assertEqual(sols.y.shape, (101, 2, 101))
        
if __name__ == '__main__':
    unittest.main()
