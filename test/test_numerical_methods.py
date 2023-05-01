from context import src
from src.problemsolver import ProblemSolver as PS
from src.solution import Solution
from src.problem import Problem
from src.solver import Solver
from src.ivp import IVP
from src.bvp import BVP
import numpy as np
import unittest

class TestProblemSolver(unittest.TestCase):
    
    def setUp(self):
        self.problem1 = Problem(f=lambda t, y: t + y, t0=0, tf=1, y0=1)
        self.problem2 = Problem(f=lambda t, y: t + y, t0=0, tf=1, y0=1, bc=lambda y0, yf: y0 + yf)
        self.problem3 = Problem(q=lambda t, y, u: t + u, t0=0, tf=1, a=0, b=1, C=0.5, bc=lambda y0, yf: y0 + yf, ic=lambda x: x)
        self.ivp = IVP(self.problem1)
        self.bvp = BVP(self.problem2)
        self.problem_solver_ivp = PS(problem=self.problem1)
        self.problem_solver_bvp = PS(problem=self.problem2)
        self.solution_ivp = Solution(t=np.linspace(0, 1, 100), y=np.linspace(0, 1, 100))
        self.solution_bvp = Solution(t=np.linspace(0, 1, 100), y=np.linspace(0, 1, 100))
        
    def test_init_ivp(self):
        self.assertEqual(self.problem_solver_ivp.problem.problem_type, self.problem1.problem_type)
        self.assertEqual(type(self.problem_solver_ivp.ivp), type(self.ivp))
        
    def test_init_bvp(self):
        self.assertEqual(self.problem_solver_bvp.problem.problem_type, self.problem2.problem_type)
        self.assertEqual(type(self.problem_solver_bvp.bvp), type(self.bvp))
        
    def test_solve_ivp(self):
        solution = self.problem_solver_ivp.solve(method="RK45")
        self.assertEqual(type(solution), Solution)
        
    def test_solve_bvp(self):
        solution, _ = self.problem_solver_bvp.solve(discretize_method="SHOOT")
        self.assertEqual(type(solution), Solution)

        
    def test_solve_invalid_problem_type(self):
        problem_solver = PS(problem=self.problem3)
        with self.assertRaises(Exception):
            problem_solver.solve()