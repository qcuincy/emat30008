from .pde_utils.grid import Grid
from .solution import Solution
from .problem import Problem
from .solver import Solver
import numpy as np


class IVP(Solver):
    """
    IVP class for solving initial value problems

    Classes:
        IVP(Solver)
            Class for solving initial value problems

    Attributes:
        problem (Problem):
            The problem to solve.

    Methods:
        solve(method="RK45", pde_method="FINITEDIFF", root_finder="ROOT", q_cont=None, **kwargs):
            Solve the initial value problem

    Usage:
        ODE:
        >>> from numerical_methods.ivp import IVP
        >>> from numerical_methods.problem import Problem
        >>> from numerical_methods.solution import Solution
        >>> from numerical_methods.solver import Solver

        >>> def f(x, u):
        ...     return [u[1], -u[0]]

        >>> problem = Problem(f=f, y0=[1, 0], t0=0, tf=10, Nt=100)
        >>> ivp = IVP(problem)
        >>> sol = ivp.solve()
        >>> sol.plot()

        PDE:
        >>> from numerical_methods.ivp import IVP
        >>> from numerical_methods.problem import Problem

        >>> def q(t, x, u):
        ...     return u

        >>> problem = Problem(q=q, y0=[1, 0], t0=0, tf=10, Nt=100, a=0, b=1, Nx=100)
        >>> ivp = IVP(problem)
        >>> sol = ivp.solve()
        >>> sol.plot()
    """
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)
        
    def solve(self, method="RK45", pde_method="CRANKNICOLSON", root_finder="ROOT", matrix_type="SPARSE", **kwargs):
        """
        Solve the initial value problem

        Args:
            method (str, optional): ODE solver method. Defaults to "RK45".
            pde_method (str, optional): PDE solver method. Defaults to "FINITEDIFF".
            root_finder (str, optional): Root finder method. Defaults to "ROOT".

        Returns:
            Solution: Solution object
        """
        # implementation of IVP solver method for either ODE or PDE
        self._method_handler(method, pde_method, root_finder, matrix_type)
        if self.is_ode:
            return self._solve_ode()

        elif not self.is_ode:
            return self._solve_pde()

        else:
            raise ValueError("Invalid combination of parameters for IVP solver.")



    def _solve_ode(self):
        """
        Solve the initial value problem for ODEs

        Returns:
            Solution: Solution object
        """
        method = self.method(self.f, self.y0, self.t0, self.tf, self.dt)
    
        for i in range(self.Nt):
            method.step(i)

        return Solution(method.t, method.y)


    def _solve_pde(self):
        """
        Solve the initial value problem for PDEs

        Returns:
            Solution: Solution object
        """
        pde_method = self.pde_method(Grid(self.a, self.b, self.t0, self.tf, C=self.C, dt=self.dt, dx=self.dx))
        sol = pde_method.solve(self.q, ic=self.ic, method=self.method, root_finder=self.root_finder)
        return sol
