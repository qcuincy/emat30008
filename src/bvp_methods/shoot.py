import sys
import numpy as np
from scipy.optimize import fsolve
from ..ivp import IVP
from ..problem import Problem
from ..solution import Solution
import warnings
warnings.filterwarnings("ignore")


class Shoot():
    """
    Class for shooting method.
    
    Args:
        f (function):
            The differential equation to solve.
        y0 (array_like):
            The initial value of the solution.
        t0 (float):
            The initial time.
        tf (float):
            The final time.
        dt (float):
            The time step.
        ic (function):
            The initial condition.
        bc (function):
            The boundary condition.

    Attributes:
        f (function):
            The differential equation to solve.
        y0 (array_like):
            The initial value of the solution.
        t0 (float):
            The initial time.
        tf (float):
            The final time.
        dt (float):
            The time step.
        ic (function):
            The initial condition.
        bc (function):
            The boundary condition.
        t (ndarray):
            The array of time values.
        n_steps (int):
            The number of time steps.
        
    Methods:
        solve(bc=None, phase_cond=None, method="RK45", maxiter=100, tol=1e-3):
            Solve the boundary value problem using shooting methods.
    """
    def __init__(self, f, y0, t0, tf, dt, ic, bc):
        """
        Class for shooting methods.

        Args:
            f (function):
                The differential equation to solve.
            y0 (array_like):
                The initial value of the solution.
            t0 (float):
                The initial time.
            tf (float):
                The final time.
            dt (float):
                The time step.
            ic (function):
                The initial condition.
            bc (function):
                The boundary condition.
        """
        
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.ic = ic
        self.bc = bc
        self.t = np.arange(t0, tf + dt, dt)
        self.n_steps = int((self.tf - self.t0) / self.dt)


    def solve(self, bc=None, phase_cond=None, method="RK45", maxiter=1000, tol=1e-6):
        """
        Solve the boundary value problem using shooting methods.

        Args:
            bc (function):
                The boundary condition.
            phase_cond (float):
                The phase condition.
            method (str):
                The method to use for solving the IVP.
            maxiter (int):
                The maximum number of iterations.
            tol (float):
                The tolerance for the root finder.

        Returns:
            sol (Solution):
                The solution to the boundary value problem.
        """
        def to_minimize(y0):
            """
            Helper function to calculate the error in the boundary conditions.

            Args:
                y0 (array_like):

            Returns:
                error (float):
                    The error in the boundary conditions.
            """
            if phase_cond is not None:
                y0[1] = phase_cond

            problem = Problem(f=self.f, y0=y0, t0=self.t0, tf=self.tf, dt=self.dt)
            sol = IVP(problem).solve(method=method)
            y_right = sol.y[0][-1]
            y_left = sol.y[0][0]

            error = self.bc(y_left, y_right)
            return np.array(error)
            
        if bc is not None:
            self.bc = bc
        elif self.bc is None:
            # Assume Dirichlet boundary conditions
            self.bc = lambda y_left, y_right: np.array([y_left, y_right])

        if phase_cond is not None:
            self.phase_cond = phase_cond

        y0_sol, info, ier, msg = fsolve(to_minimize, self.y0, xtol=tol, maxfev=maxiter, full_output=True)

        if ier == 1:
            fcalls = info["nfev"]
            residual = np.linalg.norm(info["fvec"])
            print(f"Root finder found the solution y0_sol={y0_sol} after {fcalls} function calls; the norm of the final residual is {residual}")
            sol = IVP(Problem(f=self.f, y0=y0_sol, t0=self.t0, tf=self.tf, dt=self.dt)).solve(method=method)
            return Solution(sol.t, sol.y), np.array([sol.y[0][0], sol.y[0][-1]])
        else:
            print(f"Root finder failed with error message: {msg}")
            return Solution(self.t, np.zeros_like(self.t)), self.y0
