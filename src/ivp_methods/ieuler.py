from .odestep import ODEStep
import numpy as np


class IEuler(ODEStep):
    """
    Improved Euler's method for solving ODEs

    Args:
        f (function): The function f(t, y) defining the ODE
        y0 (float): The initial condition y(t0)
        t0 (float): The initial time
        tf (float): The final time
        dt (float): The time step size

    Attributes:
        f (function): The function f(t, y) defining the ODE
        y0 (float): The initial condition y(t0)
        t0 (float): The initial time
        tf (float): The final time
        dt (float): The time step size
        t (ndarray): The array of time values
        y (ndarray): The array of solution values

    Methods:
        step(i): Perform a single step of Improved Euler's method
    """

    def __init__(self, f, y0, t0, tf, dt):
        super().__init__(f, y0, t0, tf, dt)


    def step(self, i):
        """
        Perform a single step of Improved Euler's method
        
        Args:
            i (int): The current time step

        Returns:
            y (array or float): The solution at the next time step
    
        References:
            https://en.wikipedia.org/wiki/Heun%27s_method
        """
        # Calculate the optimal step size
        dt_new = self._step_size_control(self.y[i, :], self.t[i], self.f, self.dt, self.tol)

        # Make sure the new step size is not too large
        if self.dt * 2 < dt_new:
            dt_new = self.dt * 2

        # Perform a single step of Improved Euler's method
        self.y[i + 1, :] = self._ieuler_step(self.y[i, :], self.t[i], self.f, dt_new)

        # Update the step size
        self.dt = dt_new

        # Update the time
        self.t[i + 1] = self.t[i] + self.dt

        return self.y[i + 1, :]
    

    def _step_size_control(self, y, t, f, dt, tol):
        """
        Calculate the optimal step size

        Args:
            y (float): The solution at the current time step
            t (float): The current time
            f (function): The function f(t, y) defining the ODE
            dt (float): The time step size
            tol (float): The error tolerance

        Returns:
            dt_new (float): The optimal step size

        References:
            https://en.wikipedia.org/wiki/Step_size_control
        """
        # Calculate the first two steps
        y1 = y + dt * np.array(f(t, y))
        y2 = y1 + dt * np.array(f(t + dt, y1))

        # Calculate the error
        e = np.linalg.norm(y2 - y1, 2)

        # Calculate the optimal step size
        dt_new = dt * (tol / e) ** 0.5

        return dt_new


    def _ieuler_step(self, y, t, f, dt):
        """
        Perform a single step of Improved Euler's method

        Args:
            y (float): The solution at the current time step
            t (float): The current time
            f (function): The function f(t, y) defining the ODE
            dt (float): The time step size

        Returns:
            y (float): The solution at the next time step

        References:
            https://en.wikipedia.org/wiki/Heun%27s_method
        """
        k1 = np.array(f(t, y))
        k2 = np.array(f(t + dt, y + dt * k1))

        return y + dt * (k1 + k2) / 2