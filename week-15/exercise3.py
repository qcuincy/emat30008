import sys, os
if not hasattr(sys.modules[__name__], '__file__'):
    emat_dir = os.path.dirname(os.path.abspath())
else:
    emat_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".."
        )
    )
sys.path.insert(0, emat_dir)
from integrate.OdeSol import *
from integrate.solvers import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

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


def predator_prey(t, X, consts=[1, 0.1, 0.1]):
    a, d, b = consts
    x, y = X
    return np.array([x*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])

initialx = [10, 20]
t_span = [0, 50]
consts = [1, 0.1, 0.26]

X0 = shooting_method(lambda t, X: predator_prey(t, X, consts), x0=initialx, t_span=t_span, bounds=[0, None])

sol = ode_ivp(f=lambda t, X: predator_prey(t, X, consts), y0=X0, t_span=t_span)

t, X = sol["t"], sol["y"]
plt.plot(t, X[0,:])
plt.plot(t, X[1,:])
plt.show()


plt.plot(X[0,:], X[1,:])
plt.show()