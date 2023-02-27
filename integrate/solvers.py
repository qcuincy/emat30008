from .methods import *
import numpy as np


METHODS = {'EULER': euler_step,
           'IEULER': improved_euler_step,
           'RK4': rk4_step,
           'RK45': rk45_step,
           'MIDPOINT': midpoint_step}


def solve_to(f, y0, t_span, dt=0.01, dt_max=0.1, method="RK45"):
    """
    Solve a differential equation numerically using a given numerical method from a starting time to an end time.

    Args:
        f (function): 
            Function that defines the differential equation.
        y0 (array): 
            Initial value of the solution.
        t_span (2-member sequence): 
            Interval of integration (t0, tf).
        dt (float): 
            Time step.
        dt_max (float): 
            Maximum time step.
        method (str, optional): 
            Numerical method to use (default is the Runge-Kutta method of order 5(4)).

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
    if not isinstance(y0, (list, np.ndarray)):
        raise TypeError("The argument 'x0' must be a numpy array.")
    if not isinstance(dt, (int, float)):
        raise TypeError("The argument 'dt' must be a number.")
    if dt <= 0:
        raise ValueError("The argument 'dt' must be positive.")
    if dt_max <= 0:
        raise ValueError("The argument 'dt_max' must be positive.")
    if method not in METHODS:
        raise TypeError(f"The method chosen: {method}, is not valid\nPlease choose from the following: {METHODS.keys()}")

    method = METHODS[method]
    
    t0, tf = t_span
    t = np.arange(t0, tf + dt, dt)
    Y = np.zeros((len(t), len(y0)))
    Y[0] = y0
    Y[1:] = [method(f, t[i-1], Y[i-1], min(t[i]-t[i-1], dt_max)) for i in range(1, len(t))][0]
    return dict(t=t, y=Y.T)


def ode_ivp(f, t_span, y0, method="RK45", t_eval=None, step_size=0.001, rtol=1e-3, atol=1e-6):
    """
    Solve a first order differential equation y' = f(t, y) using the specified numerical method.
    
    Parameters
    ----------
    f : (callable)
        The function defining the ODE y' = f(t, y).
    t_span : (tuple)
        The interval of integration (t0, tf).
    y0 : (array_like)
        The initial value of the dependent variable.
    method : (str or callable, optional)
        The numerical method to use. This can be one of the strings 'RK45' (default), 'RK23', 'DOP853', 'Radau', 'BDF', or a custom function implementing a numerical method. The custom function should have the signature (f, t, y, dt) and return the new value of y at time t+dt.
    t_eval : (array_like, optional)
        The times at which to output the solution. Defaults to None, in which case the solution is output at every internal step.
    step_size : (float, optional)
        The size of the time step used in the numerical method. Defaults to 0.001.
    rtol : (float, optional)
        The relative tolerance. Defaults to 1e-3.
    atol : (float or array_like, optional)
        The absolute tolerance. Defaults to 1e-6.
    
    Returns
    -------
    sol : dict
        A dictionary containing the solution. The keys are 't' (the times at which the solution is output) and 'y' (the corresponding values of the dependent variable).
    """
    if not callable(f):
        raise TypeError("The argument 'f' must be a function.")
    if not isinstance(t_span, (list, np.ndarray)):
        raise TypeError("The argument 't_span' must be a 2-member sequence -> [t0, tf].")
    if len(t_span) != 2:
        raise ValueError("The argument 't_span' must be a 2-member sequence -> [t0, tf].")
    if t_span[0] < 0:
        raise ValueError("The first member of 't_span' must be positive.")
    if t_span[1] < 0:
        raise ValueError("The second member of 't_span' must be positive.")
    if not isinstance(y0, (list, np.ndarray)):
        raise TypeError("The argument 'x0' must be a numpy array.")
    if not isinstance(step_size, (int, float)):
        raise TypeError("The argument 'step_size' must be a number.")
    if step_size <= 0:
        raise ValueError("The argument 'step_size' must be positive.")
    if method not in METHODS:
        raise TypeError(f"The method chosen: {method}, is not valid\nPlease choose from the following: {METHODS.keys()}")
    elif callable(method):
        pass
    # Check if method is a string
    if isinstance(method, str):
        method = METHODS[method]

    # Initialize time grid and solution array
    t0, tf = t_span
    t = np.arange(t0, tf, step_size)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    # Counter for t_eval indexing
    idx = 1
    
    # Main loop for numerical integration
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        y[i] = method(f, t[i-1], y[i-1], dt)
        
        # Check if t_eval is not None and if it's time to output the solution
        if t_eval is not None:
            if idx < len(t_eval) and np.isclose(t[i], t_eval[idx], rtol=rtol, atol=atol):
                idx += 1
    
    # Replace t with t_eval if t_eval is not None
    if t_eval is not None:
        t = t_eval
    
    # Create dictionary containing the solution
    sol = {'t': t, 'y': y[:len(t)].T}
    
    return sol