import numpy as np

def euler_step(f, t, y, dt):
    """
    Perform a single step of the Euler method to approximate the solution of a differential equation.

    Args:
        t (float): 
            Current time.
        y (array): 
            Current value of the solution.
        f (function): 
            Function that defines the differential equation.
        dt (float): 
            Time step.

    Returns:
        array: 
            Approximate solution at time t+dt.
    """

    return y + f(t, y) * dt


def rk4_step(f, t, y, dt):
    """
    Perform a single step of the fourth-order Runge-Kutta method to approximate the solution of a differential equation.

    Args:
        t (float): 
            Current time.
        y (array): 
            Current value of the solution.
        f (function): 
            Function that defines the differential equation.
        dt (float): 
            Time step.

    Returns:
        array: 
            Approximate solution at time t+dt.
    """

    k1 = f(t, y)
    k2 = f(t, y + 0.5 * dt * k1)
    k3 = f(t, y + 0.5 * dt * k2)
    k4 = f(t, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def midpoint_step(f, t, y, dt):
    """
    Perform a single step of the midpoint method to approximate the solution of a differential equation.

    Args:
        t (float): 
            Current time.
        y (array): 
            Current value of the solution.
        f (function): 
            Function that defines the differential equation.
        dt (float): 
            Time step.

    Returns:
        array: 
            Approximate solution at time t+dt.
    """

    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    return y + dt * k2


def improved_euler_step(f, t, y, dt):
    """
    Perform a single step of the improved Euler method to approximate the solution of a differential equation.

    Args:
        t (float): 
            Current time.
        y (array): 
            Current value of the solution.
        f (function): 
            Function that defines the differential equation.
        dt (float): 
            Time step.

    Returns:
        array: 
            Approximate solution at time t+dt.
    """

    k1 = f(t, y)
    k2 = f(t + dt, y + dt * k1)
    return y + dt * (k1 + k2) / 2


def rk45_step(f, t, y, dt):
    """Compute a single step of the Runge-Kutta method of order 5(4).
    
    Parameters
    ----------
    f : (function)
        The function defining the ODE y' = f(t, y).
    t : (float)
        The current value of the independent variable.
    y : (array_like)
        The current value of the dependent variable.
    dt : (float)
        The step size.
    
    Returns
    -------
    y_new : (array_like)
        The estimated value of y at t + dt.
    """
    k1 = dt * f(t, y)
    k2 = dt * f(t + dt / 5, y + k1 / 5)
    k3 = dt * f(t + 3 * dt / 10, y + 3 * k1 / 40 + 9 * k2 / 40)
    k4 = dt * f(t + 4 * dt / 5, y + 44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9)
    k5 = dt * f(t + 8 * dt / 9, y + 19372 * k1 / 6561 - 25360 * k2 / 2187
               + 64448 * k3 / 6561 - 212 * k4 / 729)
    k6 = dt * f(t + dt, y + 9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247
               + 49 * k4 / 176 - 5103 * k5 / 18656)
    
    y_new = y + 35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84
    
    return y_new