from abc import abstractmethod
from .context import src
from src import *
import numpy as np

class Example():
    """
    Base class for examples

    Will store pre-defined parameters and functions for examples

    Attributes:
        params (tuple): The parameters for the ODE or PDE.
    """
    def __init__(self, **kwargs):
        self.params = kwargs.get("params", None)

    @abstractmethod
    def func(self):
        """
        Function to solve
        """
        pass

    @abstractmethod
    def func_true_solution(self):
        """
        True solution function
        """
        pass


class Lokta(Example):
    def __init__(self):
        pass

    def ode(self, t, u, a=1, b=1, d=1):
        """
        The Lokta-Volterra predator-prey ODE equations.

        Args:
            t (float): The current time.
            u (array): The current value of x and dx/dt.

        Returns:
            array: The value of dx/dt and d^2x/dt^2.
        """
        # inital x and y
        x, y = u

        # equations
        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - (y / x))
        # returning
        return np.array([dxdt, dydt])

    def func_true_solution(self, t, a=1, b=1, d=1):
        """
        The true solution for the Lokta-Volterra predator-prey ODE equations.

        Args:
            t (float): The current time.

        Returns:
            array: The value of x and dx/dt.
        """
        # equations
        x = (a * d) / (b + a * np.exp(-a * t) - a)
        y = (b * d * np.exp(-a * t)) / (b + a * np.exp(-a * t) - a)
        # returning
        return np.array([x, y])

class Hopf(Example):
    def __init__(self):
        pass

    def ode(self, t, u, b=1, o=1):
        """
        The Hopf bifurcation ODE equations.

        Args:
            t (float): The current time.
            u (array): The current value of x and dx/dt.

        Returns:
            array: The value of dx/dt and d^2x/dt^2.
        """
        # inital x and y
        x, y = u

        # equations
        dxdt = b * x - y + o * x * (x ** 2 + y ** 2)
        dydt = x + b * y + o * y * (x ** 2 + y ** 2)
        # returning
        return np.array([dxdt, dydt])

    def func_true_solution(self, t, b=1, o=1):
        """
        The true solution for the Hopf bifurcation ODE equations.

        Args:
            t (float): The current time.

        Returns:
            array: The value of x and dx/dt.
        """
        # equations
        x = np.sqrt(b / o) * np.cos(np.sqrt(b * o) * t)
        y = np.sqrt(b / o) * np.sin(np.sqrt(b * o) * t)
        # returning
        return np.array([x, y])



# modified Hopf bif
def mod_Hopf_bifurcation(t,y0,params):
    '''
    Description:
        The modified Hopf bifurcation equations.
        
    Parameters:
        t (float): The current time.
        y0 (array): The current value of x and dx/dt.
        params (tuple): The parameters for the ODE.
        
    Returns:
        array: The value of dx/dt and d^2x/dt^2.
    '''
    u1,u2 = y0
    b = params
    # equations
    du1 = b*u1 - u2 + u1*(u1**2+u2**2) - u1*(u1**2+u2**2)**2
    du2 = u1 + b*u2 + u2*(u1**2+u2**2) - u2*(u1**2+u2**2)**2
    return np.array([du1,du2])

# cubic equation
def cubic_eq(t,x,c):
    '''
    Description:
        The cubic equation.
    
    Parameters:
        t (float): The current time.
        x (array): The current value of x and dx/dt.
        c (float): The parameter for the cubic equation.
    
    Returns:
        array: The value of dx/dt and d^2x/dt^2.
    '''
    eq =  x**3 - x + c
    return eq

# Bratu 
#def Bratu()