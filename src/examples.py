from abc import abstractmethod
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

    def __call__(self):
        return self.ode, self.pde

    @abstractmethod
    def ode(self):
        """
        Function to solve
        """
        pass


    @abstractmethod
    def pde(self):
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


class Van_der_Pol(Example):
    def __init__(self):
        self.mu = 1.5

    def __call__(self):
        return (self.mu, self.ode, self.pde)

    def ode(self, t, u, mu=1.5):
        """
        The Van der Pol equations.

        Args:
            t (float): The current time.
            u (array): The current value of x and dx/dt.
            mu (float): The parameter for the Van der Pol equations.

        Returns:
            array: The value of dx/dt and d^2x/dt^2.
        """
        # inital x and y
        x, y = u

        # equations
        dxdt = y
        dydt = mu * (1 - x ** 2) * y - x
        # returning
        return np.array([dxdt, dydt])
    
    
    def pde(self, t, y, u, mu=1.5):
        """
        The Van der Pol equations.

        Args:
            t (float): The current time.
            y (float): The current value of x, y, and z.
            u (float): The current value of dx/dt, dy/dt, and dz/dt.
            mu (float): The parameter for the Van der Pol equations.

        Returns:
            array: The value of dx/dt, dy/dt, and dz/dt.
        """
        # inital x and y
        x, y, z = y
        dxdt, dydt, dzdt = u

        # equations
        du1 = dydt
        du2 = dzdt
        du3 = mu*(1 - x**2)*y - x
        # returning
        return np.array([du1, du2, du3])

    def func_true_solution(self, t, mu=1):
        """
        The true solution for the Van der Pol equations.

        Args:
            t (float): The current time.
            mu (float): The parameter for the Van der Pol equations.

        Returns:
            array: The value of x and dx/dt.
        """
        # equations
        x = np.cos(t)
        y = -np.sin(t)
        # returning
        return np.array([x, y])


class Duffing(Example):
    def __init__(self):
        self.a=1
        self.b=1
        self.d=1
        self.g=1

    def __call__(self):
        return ((self.a, self.b, self.d, self.g), self.ode, self.pde)

    def ode(self, t, u, a=1, b=1, d=1, g=1):
        """
        The Duffing equations.

        Args:
            t (float): The current time.
            u (array): The current value of x and dx/dt.
            a (float): The parameter for the Duffing equations.
            b (float): The parameter for the Duffing equations.
            d (float): The parameter for the Duffing equations.
            g (float): The parameter for the Duffing equations.

        Returns:
            array: The value of dx/dt and d^2x/dt^2.
        """
        # inital x and y
        x, y = u

        # equations
        dxdt = y
        dydt = -d*y - a*x - b*x**3 + g*np.cos(t)
        # returning
        return np.array([dxdt, dydt])
    
    def pde(self, t, y, u, a=1, b=1, d=1, g=1):
        """
        The Duffing equations.

        Args:
            t (float): The current time.
            y (float): The current value of x, y, and z.
            u (float): The current value of dx/dt, dy/dt, and dz/dt.
            a (float): The parameter for the Duffing equations.
            b (float): The parameter for the Duffing equations.
            d (float): The parameter for the Duffing equations.
            g (float): The parameter for the Duffing equations.

        Returns:
            array: The value of dx/dt, dy/dt, and dz/dt.
        """
        # inital x and y
        x, y, z = y
        dxdt, dydt, dzdt = u

        # equations
        du1 = dydt
        du2 = dzdt
        du3 = -d*y - a*x - b*x**3 + g*np.cos(t)
        # returning
        return np.array([du1, du2, du3])


    def func_true_solution(self, t, a=1, b=1, d=1, g=1):
        """
        The true solution for the Duffing equations.

        Args:
            t (float): The current time.
            a (float): The parameter for the Duffing equations.
            b (float): The parameter for the Duffing equations.
            d (float): The parameter for the Duffing equations.
            g (float): The parameter for the Duffing equations.

        Returns:
            array: The value of x and dx/dt.
        """
        # equations
        x = np.cos(t)
        y = -np.sin(t)
        # returning
        return np.array([x, y])
    
class Cubic(Example):
    def __init__(self):
        pass

    def ode(self, t, y, a=1, b=1, c=1):
        """
        The Cubic equations.

        Args:
            t (float): The current time.
            y (array): The current value of x and dx/dt.
            a (float): The parameter for the Cubic equations.
            b (float): The parameter for the Cubic equations.
            c (float): The parameter for the Cubic equations.

        Returns:
            array: The value of dx/dt and d^2x/dt^2.
        """

        return np.array(-a*y**3 - b*y**2 - c*y)
    
    def pde(self, t, y, u, a=1, b=1, c=1):
        """
        The Cubic equations.

        Args:
            t (float): The current time.
            y (float): The current value of x, y, and z.
            u (float): The current value of dx/dt, dy/dt, and dz/dt.
            a (float): The parameter for the Cubic equations.
            b (float): The parameter for the Cubic equations.
            c (float): The parameter for the Cubic equations.

        Returns:
            array: The value of dx/dt, dy/dt, and dz/dt.
        """

        return np.array(-a*u**3 - b*u**2 - c*u)


    def func_true_solution(self, t, a=1, b=1, c=1):
        """
        The true solution for the Cubic equations.

        Args:
            t (float): The current time.
            a (float): The parameter for the Cubic equations.
            b (float): The parameter for the Cubic equations.
            c (float): The parameter for the Cubic equations
        
        Returns:
            array: The value of x and dx/dt.
        """
        # equations
        x = np.cos(t)
        y = -np.sin(t)
        
        return np.array([x, y])


class Lorenz(Example):
    def __init__(self):
        self.a = 10
        self.b = 28
        self.c = 8/3

    def ode(self, t, y, a=10, b=28, c=8/3):
        """
        The Lorenz equations.

        Args:
            t (float): The current time.
            y (array): The current y value.
            a (float): The parameter for the Lorenz equations.
            b (float): The parameter for the Lorenz equations.
            c (float): The parameter for the Lorenz equations.

        Returns:
            array: The value of dx/dt, dy/dt, and dz/dt.
        """
        return np.array([a*(y[1]-y[0]), y[0]*(b-y[2]) - y[1], y[0]*y[1] - c*y[2]])
    
    def pde(self, t, y, u, a=10, b=28, c=8/3):
        """
        The Lorenz equations.

        Args:
            t (float): The current time.
            y (float): The current value of x, y, and z.
            u (float): The current value of dx/dt, dy/dt, and dz/dt.
            a (float): The parameter for the Lorenz equations.
            b (float): The parameter for the Lorenz equations.
            c (float): The parameter for the Lorenz equations.

        Returns:
            array: The value of dx/dt, dy/dt, and dz/dt.
        """
        
        return np.array([a*(u[1]-u[0]), u[0]*(b-y[2]) - u[1], u[0]*y[1] - c*u[2]])


    def func_true_solution(self, t, a=10, b=28, c=8/3):
        """
        The true solution for the Lorenz equations.

        Args:
            t (float): The current time.
            a (float): The parameter for the Lorenz equations.
            b (float): The parameter for the Lorenz equations.
            c (float): The

        Returns:
            array: The value of x and dx/dt.
        """

        # equations
        x = np.cos(t)
        y = -np.sin(t)
        z = np.sin(t)
        # returning
        return np.array([x, y, z])


class DynamicBratu(Example):
    def __init__(self):
        self.mu = 2

    def __call__(self):
        return self.mu, self.pde

    def pde(self, t, y, u, mu=1):
        """
        The Bratu equations.

        du/dt = D*d^2u/dx^2 + exp(mu*u)

        """
        return np.array(np.exp(mu*u))

class Heat(Example):
    def __init__(self):
        self.D = 1

    def __call__(self):
        return (self.D), self.ode, self.pde

    def pde(self, t, y, u, D=1):
        """
        The Heat equations.

        du/dt = D*d^2u/dx^2

        """
        return np.array(D*u)
        


