from scipy.optimize import OptimizeResult
import numpy as np


def euler_step(t, X, f, dt):
    """
    Perform a single step of the Euler method to approximate the solution of a differential equation.

    Args:
        t (float): Current time.
        X (array): Current value of the solution.
        f (function): Function that defines the differential equation.
        dt (float): Time step.

    Returns:
        array: Approximate solution at time t+dt.
    """

    

    return X + f(t, X) * dt


def rk4_step(t, X, f, dt):
    """
    Perform a single step of the fourth-order Runge-Kutta method to approximate the solution of a differential equation.

    Args:
        t (float): Current time.
        X (array): Current value of the solution.
        f (function): Function that defines the differential equation.
        dt (float): Time step.

    Returns:
        array: Approximate solution at time t+dt.
    """

    k1 = f(t, X)
    k2 = f(t, X + 0.5 * dt * k1)
    k3 = f(t, X + 0.5 * dt * k2)
    k4 = f(t, X + dt * k3)
    return X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def midpoint_step(t, X, f, dt):
    """
    Perform a single step of the midpoint method to approximate the solution of a differential equation.

    Args:
        t (float): Current time.
        X (array): Current value of the solution.
        f (function): Function that defines the differential equation.
        dt (float): Time step.

    Returns:
        array: Approximate solution at time t+dt.
    """

    k1 = f(t, X)
    k2 = f(t + dt / 2, X + dt / 2 * k1)
    return X + dt * k2


def improved_euler_step(t, X, f, dt):
    """
    Perform a single step of the improved Euler method to approximate the solution of a differential equation.

    Args:
        t (float): Current time.
        X (array): Current value of the solution.
        f (function): Function that defines the differential equation.
        dt (float): Time step.

    Returns:
        array: Approximate solution at time t+dt.
    """

    k1 = f(t, X)
    k2 = f(t + dt, X + dt * k1)
    return X + dt * (k1 + k2) / 2


def solve_to(f, t_span, x0, dt=0.01, dt_max=0.1, step_func=rk4_step):
    """
    Solve a differential equation numerically using a given numerical method from a starting time to an end time.

    Args:
        f (function): Function that defines the differential equation.
        x0 (array): Initial value of the solution.
        t_span (2-member sequence): Interval of integration (t0, tf).
        dt (float): Time step.
        dt_max (float): Maximum time step.
        step_func (function): Numerical method to use (default is the fourth-order Runge-Kutta method).

    Returns:
        tuple: Tuple of arrays representing the time points and the approximate solutions at those time points.
    """
    if not callable(f):
        raise TypeError("The argument 'f' must be a function.")
    if not isinstance(t_span, (list, np.ndarray)):
        raise TypeError("The argument 't0' must be a number.")
    if len(t_span) != 2:
        raise ValueError("The argument 't_span' must be a 2-member sequence -> [t0, t1].")
    if t_span[0] < 0:
        raise ValueError("The first member of 't_span' must be positive.")
    if t_span[1] < 0:
        raise ValueError("The second member of 't_span' must be positive.")
    if not isinstance(x0, (list, np.ndarray)):
        raise TypeError("The argument 'x0' must be a numpy array.")
    if not isinstance(dt, (int, float)):
        raise TypeError("The argument 'dt' must be a number.")
    if dt <= 0:
        raise ValueError("The argument 'dt' must be positive.")
    if dt_max <= 0:
        raise ValueError("The argument 'dt_max' must be positive.")
    if not callable(step_func):
        raise ValueError("The argument 'step_func' must be a function.")

    t = np.arange(t_span[0], t_span[1]+dt, dt)
    X = np.zeros((len(t), len(x0)))
    X[0] = x0
    X[1:] = [step_func(t[i-1], X[i-1], f, min(t[i]-t[i-1], dt_max)) for i in range(1, len(t))]
    return t, X


class OdeSol(OptimizeResult):
    pass


def ode_ivp(f, t_span, y0, method=rk4_step, dt=0.01, t_eval=None):
    """
    Solve a system of first-order ODEs defined by `fun`.

    Args:
        f (function): 
            Function that defines the system of ODEs.
            The calling signature is `fun(t, y)`, where `t` is a scalar and `y` is a 1-D array.
            The function must return a 1-D array with the same shape as `y`.
        t_span (tuple): 
            Interval to integrate over. 
            The calling signature is `t_span = (t0, tf)`, where `t0` is the initial time and `tf` is the final time.
        x0 (array_like): 
            Initial condition on `x` (can be a scalar if `f` returns a scalar).
        method (str, optional): 
            Integration method. Currently, only 'Euler' is supported.
        dt (float, optional): 
            Step size to use for the Euler method.
        t_eval (array_like, optional): 
            Times at which to output the solution. Must be sorted and lie within `t_span`.

    Returns:
        OdeSol (scipy.optimize.OptimizeResult object):
            An object containing the solution of the ODE.
    """
    t0, tf = t_span
    n = int((tf - t0) / dt) + 1
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        t = t0 + i * dt
        y[i + 1] = method(t, y[i], f, dt)
    if t_eval is not None:
        y_interp = np.zeros((len(t_eval), len(y0)))
        for i, t in enumerate(t_eval):
            y_interp[i] = method(t0, y0, f, dt)
        return OdeSol(t=t_eval, y=y_interp.T)
    else:
        return OdeSol(t=np.linspace(t0, tf, n), y=y.T)