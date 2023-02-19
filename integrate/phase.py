import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import root, fsolve
from .solvers import *


def find_limit_cycle(ode, initialu, tol=1e-6):
    """
    Find a limit cycle of the system of ODEs defined by `ode`.

    Args:
        ode (function): 
            Function that defines the system of ODEs.
            The calling signature is `ode(t, y)`, where `t` is a scalar and `y` is a 1-D array.
            The function must return a 1-D array with the same shape as `y`.
        initialu (array_like): 
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

    sol = root(residual, initialu, tol=tol)
    if sol.success:
        return sol.x
    else:
        return None


def shooting_method(ode, x0, t_span, args=(), maxiter=100, bounds=[None, None]):
    """
    Find a limit cycle of the system of ODEs defined by `ode`.

    Args:
        ode (function):
            Function that defines the system of ordinary differential equations.
            The function should take two arguments, t and y, and return a 1D numpy
            array with the same shape as y.
        x0 (array_like)
            Initial guess for the value of the solution at the left boundary.
        t_span (tuple):
            Tuple (t0, tf) defining the interval of integration.
        args (tuple, optional):
            Additional arguments to pass to the `ode` function (default is an empty tuple).
        maxiter (int, optional):
            Maximum number of iterations for the root-finding algorithm (default is 100).
        bounds (tuple, optional):
            Tuple (lb, ub) defining lower and upper bounds on the initial guess `x0`
            (default is [None, None], which means no bounds).

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
        sol = ode_ivp(ode, t_span, x0_new)
        x_final = sol["y"][:,-1]
        error = x_final - x0_new
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

    
    x0_sol, info, ier, msg = fsolve(get_bounds, x0, args=(ode, t_span, bounds, ), full_output=True, maxfev=maxiter)
    if ier == 1:
        fcalls = info["nfev"]
        residual = np.linalg.norm(info["fvec"])
        print(f"Root finder found the solution x0_sol={x0_sol} after {fcalls} function calls; the norm of the final residual is {residual}")
        return x0_sol
    else:
        print(f"Root finder failed with error message: {msg}")
        return None 

