import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import root, fsolve
from .solvers import *
import sys


def find_limit_cycle(ode, initial_x, tol=1e-6):
    """
    Find a limit cycle of the system of ODEs defined by `ode`.

    Args:
        ode (function): 
            Function that defines the system of ODEs.
            The calling signature is `ode(t, x)`, where `t` is a scalar and `x` is a 1-D array.
            The function must return a 1-D array with the same shape as ``.
        initial_x (array_like): 
            Initial guess for the limit cycle.
        tol (float, optional): 
            Tolerance for the root finding algorithm.

    Returns:
        limit_cycle (array_like or None): 
            The limit cycle of the ODE if found, else None.
    """
    def residual(x0):
        sol = solve_to(ode, x0, np.linspace(0, 20, 200))
        return sol[-1, :] - x0

    sol = root(residual, initial_x, tol=tol)
    if sol.success:
        return sol.x
    else:
        return None


def find_period(y, idx=None):
    chosen_point_idx = np.random.randint(0, len(y[:,0])) if idx==None else idx
    chosen_point = [y[chosen_point_idx, 0], y[chosen_point_idx, 1]]
    limit_cycle = y[:,(y[0]>0) & (y[1] > 0)]

    distances = np.linalg.norm(limit_cycle.T - chosen_point, axis=1)
    min_distance_index = np.argmin(distances)
    phase = 2 * np.pi * min_distance_index / len(limit_cycle)

    return phase, chosen_point, limit_cycle