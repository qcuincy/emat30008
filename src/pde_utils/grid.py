"""
This module contains the Grid class, which provides a uniform grid of points (x, t) on a given domain [a, b] and time interval [t0, tf]. 
The grid can be used to discretize partial differential equations (PDEs) using finite differences.
"""


import numpy as np

class Grid():
    """
    The Grid class provides a uniform grid of points (x, t) on a given domain [a, b]
    and time interval [t0, tf]. The grid can be used to discretize partial differential
    equations (PDEs) using finite differences.

    Args:
        a (float): Left endpoint of the domain.
        b (float): Right endpoint of the domain.
        t0 (float): Initial time of the time interval.
        tf (float): Final time of the time interval.
        **kwargs (dict): Keyword arguments for the Grid class:
            - (C or D must be specified, but not both)
            - (dt or Nt must be specified, but not both)
            - (dx or Nx must be specified, but not both)
        Supported keyword arguments:
            - C (float): Courant number (optional, either C or D must be specified).
            - D (float): Diffusion coefficient (optional, either C or D must be specified).
            - dx (float): Spatial step size (optional, either dx, dt, Nx or Nt must be specified).
            - dt (float): Time step size (optional, either dx, dt, Nx or Nt must be specified).
            - Nx (int): Number of spatial grid points (optional, either dx, dt, Nx or Nt must be specified).
            - Nt (int): Number of time grid points (optional, either dx, dt, Nx or Nt must be specified).

    Raises:
        ValueError: If neither C or D is specified, or both are specified.
        ValueError: If neither dt or Nt is specified, or both are specified.
        ValueError: If neither dx or Nx is specified, or both are specified.
        ValueError: If C > 0.5.
        
    Attributes:
        a (float): Left endpoint of the domain.
        b (float): Right endpoint of the domain.
        t0 (float): Initial time of the time interval.
        tf (float): Final time of the time interval.
        C (float): Courant number.
        D (float): Diffusion coefficient.
        dx (float): Spatial step size.
        dt (float): Time step size.
        Nx (int): Number of spatial grid points.
        Nt (int): Number of time grid points.
        x (numpy.ndarray): Spatial grid points.
        t (numpy.ndarray): Time grid points.

    Methods:
        get_dx(): Get the spatial step size.
        get_dt(): Get the time step size.
        get_Nx(): Get the number of spatial grid points.
        get_Nt(): Get the number of time grid points.
        get_x(): Get the spatial grid points.
        get_t(): Get the time grid points.

    Usage:
        >>> import numpy as np
        >>> from pde_utils import Grid
        >>> grid = Grid(a=0, b=1, Nx=10, t0=0, tf=1, Nt=10, C=0.5)
        >>> grid.get_dx()
            0.1
        >>> grid.get_dt()
            0.1
        >>> grid.get_Nx()
            10
        >>> grid.get_Nt()
            10
        >>> grid.get_x()
            array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        >>> grid.get_t()
            array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    """


    def __init__(self, a, b, t0, tf, **kwargs):

        # Save the input parameters as attributes
        self.a = a
        self.b = b
        self.t0 = t0
        self.tf = tf

        self.C = kwargs.get('C')
        self.D = kwargs.get('D')
        self.dx = kwargs.get('dx')
        self.dt = kwargs.get('dt')
        self.Nt = kwargs.get('Nt')
        self.Nx = kwargs.get('Nx')

        self._validate_parameters()
        # Create arrays for x and t
        self.x = np.linspace(self.a, self.b, self.Nx + 1)
        self.t = np.linspace(self.t0, self.tf, self.Nt + 1)

    def _validate_parameters(self):
        """
        Validate the input parameters.

        Raises:
            ValueError: If neither C or D is specified, or both are specified.
            ValueError: If neither dt or Nt is specified, or both are specified.
            ValueError: If neither dx or Nx is specified, or both are specified.
            ValueError: If C > 0.5.
        """

        # Check that either C or D is specified, but not both
        if (self.C is None) and (self.D is None):
            raise ValueError("Either C or D must be specified")
        if (self.C is not None) and (self.D is not None):
            raise ValueError("Either C or D must be specified, not both")

        # Check that either dx, dt, Nx or Nt is specified, but not all
        if (self.dx is None) and (self.dt is None) and (self.Nt is None) and (self.Nx is None):
            raise ValueError("Either dx, dt, Nx or Nt must be specified")

        # Check that either dx and Nx are both specified or neither is specified
        if (self.dx is not None) and (self.Nx is not None):
            raise ValueError("Both dx and Nx must be specified or neither should be specified")

        # Check that either dt and Nt are both specified or neither is specified
        if (self.dt is not None) and (self.Nt is not None):
            raise ValueError("Both dt and Nt must be specified or neither should be specified")

        # Calculate dt, Nt based on the specified value
        if self.dt is not None and self.Nt is None:
            self.Nt = int((self.tf - self.t0) / self.dt)
        elif self.dt is None and self.Nt is not None:
            self.dt = (self.tf - self.t0) / self.Nt

        # Calculate dx, Nx based on the specified value
        if self.dx is not None and self.Nx is None:
            self.Nx = int((self.b - self.a) / self.dx)
        elif self.dx is None and self.Nx is not None:
            self.dx = (self.b - self.a) / self.Nx

        # Calculate D or C based on the specified value
        if self.C is not None:
            self.D = self.C * self.dx ** 2 / self.dt
        else:
            self.C = self.D * self.dt / self.dx ** 2

        # Check that C is less than or equal to 0.5
        if self.C > 0.5:
            raise ValueError("C must be less than or equal to 0.5 for stability")   

        # Check Nx and Nt are the same
        if self.Nx != self.Nt:
            raise ValueError(f"Nx and Nt must be the same, Nx={self.Nx} != Nt={self.Nt}")


    def __repr__(self):
        """
        Returns a string representation of the Grid object.

        Returns:
            str: String representation of the Grid object.
        """
        return (f"Grid(a={self.a}, b={self.b}, t0={self.t0}, tf={self.tf}, "
                f"C={self.C}, D={self.D}, dx={self.dx}, dt={self.dt}, Nx={self.Nx}, Nt={self.Nt})")

    def __str__(self):
        return (f"Grid with a={self.a}, b={self.b}, t0={self.t0}, tf={self.tf}, "
                f"C={self.C}, D={self.D}, dx={self.dx}, dt={self.dt}, Nx={self.Nx}, Nt={self.Nt}")

    def get_grid_points(self):
        """
        Returns a tuple (X, T) where X is a 1D array of grid points in the x direction and T is a 1D array of grid points
        in the t direction.
        """
        return self.x, self.t

    def get_num_grid_points(self):
        """
        Returns a tuple (Nx, Nt) where Nx is the number of grid points in the x direction and Nt is the number of grid
        points in the t direction.
        """
        return self.Nx, self.Nt

    def get_grid_spacing(self):
        """
        Returns a tuple (dx, dt) where dx is the spacing between grid points in the x direction and dt is the spacing
        between grid points in the t direction.
        """
        return self.dx, self.dt

    def get_parameters(self):
        """
        Returns a tuple (a, b, t0, tf, C, D, Nx, Nt) containing the input parameters used to create the grid.
        """
        return self.a, self.b, self.t0, self.tf, self.C, self.D, self.Nx, self.Nt
