from abc import abstractmethod
from decimal import Decimal
import numpy as np
import inspect

class ODEStep:
    """A template class for ODE step methods."""
    
    def __init__(self, f, y0, t0, tf, dt):
        """
        Initializes an ODE solver instance.

        Args:
            f (callable): The function defining the ODE.
            y0 (float or array): The initial value of the dependent variable.
            t0 (float): The initial value of the independent variable.
            tf (float): The maximum value of the independent variable.
            dt (float): The size of the step to take in each iteration.

        Attributes:
            f (callable): The function defining the ODE.
            y0 (float or array): The initial value of the dependent variable.
            t0 (float): The initial value of the independent variable.
            tf (float): The maximum value of the independent variable.
            dt (float): The size of the step to take in each iteration.
            t (ndarray): The array of time values.
            y (ndarray): The array of solution values.

        Raises:
            TypeError: If f is not callable.
            ValueError: If f does not have two parameters.
            TypeError: If y0 is not a numeric value or array.
            TypeError: If t0 is not a numeric value.
            TypeError: If tf is not a numeric value.
            ValueError: If tf is less than or equal to t0.
            TypeError: If dt is not a numeric value.
            ValueError: If dt is less than or equal to zero.
            ValueError: If dt is greater than tf - t0.
        """

        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self._check_input()
        self.Nt = int((self.tf - self.t0) / self.dt) + 1
        self.t = np.linspace(self.t0, self.tf + self.dt, self.Nt)
        self.y = np.zeros((len(self.t), len(self.y0)))
        self.y[0, :] = self.y0
        self.tol = 1e-1

    

    @abstractmethod
    def step(self, i):
        """
        Perform a single step of the ODE solver.

        Args:
            i (int): The current time step.

        Returns:
            (float or array): The solution at the next time step.
        """
        pass

    @abstractmethod
    def _step_size_control(self, y, t, f, dt, tol):
        """
        Perform a single step of the ODE solver.

        Args:
            y (ndarray): The current solution.
            t (float): The current time.
            f (callable): The function defining the ODE.
            dt (float): The current step size.
            tol (float): The tolerance for the step size control.

        Returns:
            float: The new step size.
        """

    def _check_input(self):
        """
        Check that the input arguments are valid for the solver.

        Raises:
            TypeError: If f is not callable.
            ValueError: If f does not have two parameters.
            TypeError: If y0 is not a numeric value or array.
            TypeError: If t0 is not a numeric value.
            TypeError: If tf is not a numeric value.
            ValueError: If tf is less than or equal to t0.
            TypeError: If dt is not a numeric value.
            ValueError: If dt is less than or equal to zero.
            ValueError: If dt is greater than tf - t0.
        """

        if not callable(self.f):
            raise TypeError("f must be a callable function.")
        
        if len(inspect.signature(self.f).parameters) != 2:
            raise ValueError("f must have two parameters.")

        if not isinstance(self.y0, (int, float, list, np.ndarray)):
            raise TypeError("y0 must be a numeric value or array.")
        
        if not isinstance(self.y0, (np.ndarray)):
            if isinstance(self.y0, (list)):
                self.y0 = np.array(self.y0)
            else:
                self.y0 = np.array([self.y0])
   
        if not isinstance(self.t0, (int, float)):
            raise TypeError("t0 must be a numeric value.")
        
        if not isinstance(self.tf, (int, float)):
            raise TypeError("tf must be a numeric value.")
        
        if self.tf <= self.t0:
            raise ValueError("tf must be greater than t0.")

        if not isinstance(self.dt, (int, float)):
            raise TypeError("dt must be a numeric value.")

        if self.dt <= 0:
            raise ValueError("dt must be a positive value.")
        
        delta_t = self.round_to_const(self.tf - self.t0, self.dt)

        if self.dt > (delta_t):
            raise ValueError("dt must be less than tf - t0.")


    def round_to_const(self, value, const):
        """
        Round a value to a constant number of decimal places.

        Args:
            value (float): Value to be rounded.
            const (float): Constant number of decimal places.

        Returns:
            float: Rounded value.
        """
        num_decimals = abs(Decimal(str(const)).as_tuple().exponent)
        rounded_value = round(value, num_decimals)
        if not np.allclose(value, const):
            return rounded_value
        else:
            return const