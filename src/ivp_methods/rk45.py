from .odestep import ODEStep
import numpy as np


class RK45(ODEStep):
    """
    Runge-Kutta-Fehlberg method for solving ODEs

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
        step(i): Perform a single step of RK45 method
    """
    
    def __init__(self, f, y0, t0, tf, dt, tol=1e-1):
        super().__init__(f, y0, t0, tf, dt, tol)


    def step(self, i):
        # Calculate the optimal step size
        dt_new = self._step_size_control(self.y[i, :], self.t[i], self.f, self.dt, self.tol)

        # Check if new step size exceeds the maximum allowed step size
        if self.dt * 2 < dt_new:
            dt_new = self.dt * 2

        # Perform a single step of RK45 method with the optimal step size
        self.y[i + 1, :] = self._rk45_step(self.y[i, :], self.t[i], self.f, dt_new)

        # Update the step size
        self.dt = dt_new

        # Update the time
        self.t[i + 1] = self.t[i] + self.dt

        return self.y[i + 1, :]

    def _step_size_control(self, y, t, f, dt, tol):
        # Calculate k1-k6 values
        k1 = np.array(f(t, y)) * dt
        k2 = np.array(f(t + dt / 5, y + k1 / 5)) * dt
        k3 = np.array(f(t + 3 * dt / 10, y + 3 * k1 / 40 + 9 * k2 / 40)) * dt
        k4 = np.array(f(t + 4 * dt / 5, y + 44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9)) * dt
        k5 = np.array(f(t + 8 * dt / 9, y + 19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729)) * dt
        k6 = np.array(f(t + dt, y + 9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656)) * dt

        # Calculate the new solution with RK45 method
        y_new = y + 35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84

        # Calculate the error estimate
        error = np.max(np.abs(y_new - y - 5179 * k1 / 57600 - 7571 * k3 / 16695 - 393 * k4 / 640 + 92097 * k5 / 339200 + 187 * k6 / 2100))

        # Calculate the optimal step size
        dt_new = 0.9 * dt * (tol / abs(error)) ** (1/5)

        return dt_new

    def _rk45_step(self, y, t, f, dt):
        # Calculate k1-k6 values
        k1 = np.array(f(t, y)) * dt
        k2 = np.array(f(t + dt / 5, y + k1 / 5)) * dt
        k3 = np.array(f(t + 3 * dt / 10, y + 3 * k1 / 40 + 9 * k2 / 40)) * dt
        k4 = np.array(f(t + 4 * dt / 5, y + 44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9)) * dt
        k5 = np.array(f(t + 8 * dt / 9, y + 19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729)) * dt
        k6 = np.array(f(t + dt, y + 9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656)) * dt

        # Calculate the new solution with RK45 method
        y_new = y + 35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84

        return y_new