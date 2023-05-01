from .odestep import ODEStep
import numpy as np

class RK4(ODEStep):
    """
    Runge-Kutta 4th order method for solving ODEs

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
        step(i): Perform a single step of RK4 method
    """

    def __init__(self, f, y0, t0, tf, dt):
        super().__init__(f, y0, t0, tf, dt)


    def step(self, i):
        """
        Perform a single step of RK4 method

        Args:
            i (int): The current time step

        Returns:
            y (float): The solution at the next time step
        
        References:
            https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        """
        # Calculate the optimal step size
        dt_new = self._step_size_control(self.y[i, :], self.t[i], self.f, self.dt, self.tol)

        # Make sure the new step size is not too large
        if self.dt * 2 < dt_new:
            dt_new = self.dt * 2

        # Perform a single step of RK4 method
        self.y[i + 1, :] = self._rk4_step(self.y[i, :], self.t[i], self.f, dt_new)

        # Update the step size
        self.dt = dt_new

        # Update the time
        self.t[i + 1] = self.t[i] + self.dt

        return self.y[i+1, :]
    
    
    def _step_size_control(self, y, t, f, dt, tol):
        """
        Perform step size control for RK4 method

        Args:
            y (float): The solution at the current time step
            t (float): The current time
            f (function): The function f(t, y) defining the ODE
            dt (float): The time step size
            tol (float): The tolerance for step size control

        Returns:
            dt_new (float): The new time step size
        
        References:
            https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        """

        k1 = np.array(f(t, y))
        k2 = np.array(f(t + dt / 2, y + dt * k1 / 2))
        k3 = np.array(f(t + dt / 2, y + dt * k2 / 2))
        k4 = np.array(f(t + dt, y + dt * k3))

        y_new = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y_new_star = y + dt * (k1 + k2 + k3) / 3

        err = np.linalg.norm(y_new - y_new_star)
        dt_new = dt * (tol / err) ** 0.2

        return dt_new
    

    def _rk4_step(self, y, t, f, dt):
        """
        Perform a single step of RK4 method

        Args:
            y (float): The solution at the current time step
            t (float): The current time
            f (function): The function f(t, y) defining the ODE
            dt (float): The time step size

        Returns:
            y (float): The solution at the next time step
        
        References:
            https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        """

        k1 = np.array(f(t, y))
        k2 = np.array(f(t + dt / 2, y + dt * k1 / 2))
        k3 = np.array(f(t + dt / 2, y + dt * k2 / 2))
        k4 = np.array(f(t + dt, y + dt * k3))

        y_new = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        return y_new