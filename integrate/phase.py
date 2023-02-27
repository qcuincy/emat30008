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


def shooting_method(ode, x0, t_span, phase_cond=None, args=(), maxiter=100, bounds=[None, None], solver=fsolve, tol=1e-3):
    """
    Find a limit cycle of the system of ODEs defined by `ode`.

    Args:
        ode (function):
            Function that defines the system of ordinary differential equations.
            The function should take two arguments, t and x, and return a 1D numpy
            array with the same shape as x.
        x0 (array_like)
            Initial guess for the value of the solution at the left boundary.
        t_span (tuple):
            Tuple (t0, tf) defining the interval of integration.
        phase_cond (function):
            Function that defines the phase condition for the system of ordinary differential equations.
        args (tuple, optional):
            Additional arguments to pass to the `ode` function (default is an empty tuple).
        maxiter (int, optional):
            Maximum number of iterations for the root-finding algorithm (default is 100).
        bounds (tuple, optional):
            Tuple (lb, ub) defining lower and upper bounds on the initial guess `x0`
            (default is [None, None], implying there are no bounds).
        solver (function, optional):
            Function that defines the numerical method to use to minimize the error.
        tol (float, optional):
            Tolerance for the 'solver' to terminate.
            Calculations will terminate if the relative error between two consecutive iterates is less than or equal to 'tol'

    Returns:
        x0_sol (array_like or None):
            The solution of the boundary value problem, or None if the root-finding algorithm
            failed to converge.
    """
    def to_minimize(x0, ode, t_span):
        """
        Helper function to calculate the error in the boundary conditions.
        """
        x0_new = x0
        sol = ode_ivp(ode, t_span, x0)
        x0_new = sol["y"][:,0]
        x_final = sol["y"][:,-1]
        phase_val = phase_cond(x0_new, x_final) if phase_cond != None else (x_final - x0_new)
        error = phase_val
        print(error)
        return error


    def get_bounds(x0, ode, t_span, bounds):
        """
        Helper function to enforce bounds on the initial guess.
        """
        lb, ub = bounds
        try:
            try:
                x0[x0 <= lb] = lb + 1e-6
            except:
                pass
            try:
                x0[x0 >= ub] = ub - 1e-6
            except:
                pass
        except:
            pass
        return to_minimize(x0, ode, t_span)
    
    try:
        test_sol = ode(0, x0)
    except Exception as E:
        exc_type, _, _ = sys.exc_info()
        print(str(exc_type.__name__) + ": " + str(E) + f"\nMake sure your initial values ({x0}) have the same dimensions as those expected by your ode ({ode.__name__})")
        return exc_type

    x0_sol, info, ier, msg = solver(get_bounds, x0, args=(ode, t_span, bounds, ), full_output=True, maxfev=maxiter, xtol=tol)
    if ier == 1:
        fcalls = info["nfev"]
        residual = np.linalg.norm(info["fvec"])
        print(f"Root finder found the solution x0_sol={x0_sol} after {fcalls} function calls; the norm of the final residual is {residual}")
        return x0_sol
    else:
        print(f"Root finder failed with error message: {msg}")
        return np.array([])