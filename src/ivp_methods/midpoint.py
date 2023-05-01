from .odestep import ODEStep
import numpy as np


class Midpoint(ODEStep):
    """
    Midpoint method for solving ODEs

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
        step(i): Perform a single step of Midpoint method
    """

    def __init__(self, f, y0, t0, tf, dt):
        super().__init__(f, y0, t0, tf, dt)


    def step(self, i):
        """
        Perform a single step of Midpoint method

        Args:
            i (int): The current time step

        Returns:
            y (float): The solution at the next time step

        References:
            https://en.wikipedia.org/wiki/Midpoint_method
        """
        # Calculate the optimal step size
        dt_new = self._step_size_control(self.y[i, :], self.t[i], self.f, self.dt, self.tol)

        # Make sure the new step size is not too large
        if self.dt * 2 < dt_new:
            dt_new = self.dt * 2

        # Perform a single step of Midpoint method
        self.y[i + 1, :] = self._midpoint_step(self.y[i, :], self.t[i], self.f, dt_new)

        # Update the step size
        self.dt = dt_new

        # Update the time
        self.t[i + 1] = self.t[i] + self.dt

        return self.y[i + 1, :]


    def _step_size_control(self, y, t, f, dt, tol):
        """
        Calculate the optimal step size based on the local truncation error

        Args:
            y (float): The solution at the current time step
            t (float): The current time
            f (function): The function f(t, y) defining the ODE
            dt (float): The time step size
            tol (float): The desired tolerance

        Returns:
            dt_new (float): The optimal step size

        References:
            https://en.wikipedia.org/wiki/Step_size_control
        """
        # Calculate the local truncation error
        y_mid = self._midpoint_step(y, t, f, dt)
        y_full = self._midpoint_step(y_mid, t + dt / 2, f, dt)
        err = np.linalg.norm(y_full - y_mid, 2)

        # Calculate the optimal step size
        dt_new = dt * (tol / err) ** (1 / 2)

        return dt_new
    
    def _midpoint_step(self, y, t, f, dt):
        """
        Perform a single step of Midpoint method

        Args:
            y (float): The solution at the current time step
            t (float): The current time
            f (function): The function f(t, y) defining the ODE
            dt (float): The time step size

        Returns:
            y (float): The solution at the next time step

        References:
            https://en.wikipedia.org/wiki/Midpoint_method
        """
        k1 = np.array(f(t, y))
        k2 = np.array(f(t + dt / 2, y + dt * k1 / 2))

        y_new = y + dt * k2

        return y_new