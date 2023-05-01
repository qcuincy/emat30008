from tabulate import tabulate
from decimal import Decimal
from art import *
import numpy as np
import inspect
import numbers


class Problem:
    """
    Class for defining a problem to solve.

    Methods:
        validate_parameters():
            Validates the parameters of the problem.

    Usage:
        ODE IVP:
            - Must specify the ODE function as the keyword argument "f".
            - Include an initial condition as the keyword argument "y0".
            - Include the initial time as the keyword argument "t0".
            - Include the final time as the keyword argument "tf".
            - Include either the number of time steps as the keyword argument "Nt" or the time step as the keyword argument "dt".
        >>> from problem import Problem

        >>> def f(t, y):
                '''
                The differential equation to solve with 2 arguments.
                '''
        ...     return y

        >>> problem = Problem(f=f, y0=1, t0=0, tf=1, Nt=10)
            

        ODE BVP:
            - Must specify the ODE function as the keyword argument "f".
            - Include an initial condition as the keyword argument "y0".
            - Include the initial time as the keyword argument "t0".
            - Include the final time as the keyword argument "tf".
            - Include either the number of time steps as the keyword argument "Nt" or the time step as the keyword argument "dt".
            - Include the boundary condition as the keyword argument "bc".
        >>> from problem import Problem

        >>> def f(t, y):
        ...     return y

        >>> def ic(x):
        ...     return np.sin(np.pi * x)

        >>> def bc(ya, yb):
        ...     return ya + yb

        >>> problem = Problem(f=f, y0=1, t0=0, tf=1, Nt=10, bc=bc)

        PDE IVP:
            - Must specify the PDE function as the keyword argument "q".
            - Include an initial condition as the keyword argument "ic".
            - Include the initial time as the keyword argument "t0".
            - Include the final time as the keyword argument "tf".
            - Include either the number of time steps as the keyword argument "Nt" or the time step as the keyword argument "dt".
            - Include the left boundary of the domain as the keyword argument "a".
            - Include the right boundary of the domain as the keyword argument "b".
            - Include either the number of grid points as the keyword argument "Nx" or the grid spacing as the keyword argument "dx".
        >>> from problem import Problem

        >>> def q(x, t, u):
        ...     return u

        >>> def ic(x):
        ...     return np.sin(np.pi * x)

        >>> problem = Problem(q=q, ic=ic, t0=0, tf=1, Nt=10, a=0, b=1, Nx=10)

        PDE BVP:
            - Must specify the PDE function as the keyword argument "q".
            - Include an initial condition as the keyword argument "ic".
            - Include the initial time as the keyword argument "t0".
            - Include the final time as the keyword argument "tf".
            - Include either the number of time steps as the keyword argument "Nt" or the time step as the keyword argument "dt".
            - Include the left boundary of the domain as the keyword argument "a".
            - Include the right boundary of the domain as the keyword argument "b".
            - Include either the number of grid points as the keyword argument "Nx" or the grid spacing as the keyword argument "dx".
            - Include the boundary condition as the keyword argument "bc".
        >>> from problem import Problem

        >>> def q(x, t, u):
        ...     return u

        >>> def ic(x):
        ...     return np.sin(np.pi * x)

        >>> def bc(ya, yb):
        ...     return ya + yb

        >>> problem = Problem(q=q, ic=ic, t0=0, tf=1, Nt=10, a=0, b=1, Nx=10, bc=bc)

        With "args" keyword argument:
        >>> from problem import Problem

        >>> def q(x, t, u, p, q):
        ...     return u + p + q

        >>> def ic(x):
        ...     return np.sin(np.pi * x)

        >>> def bc(ya, yb):
        ...     return ya + yb

        >>> problem = Problem(q=q, ic=ic, t0=0, tf=1, Nt=10, a=0, b=1, Nx=10, bc=bc, args=(1, 2))
    """

    def __init__(self, **kwargs):
        """
        Initializes the Problem class.

        Args:
            **kwargs (dict):
                The keyword arguments of the problem.

        For ODEs:
            f (function):
                The differential equation to solve.
                2 arguments: t, y
            y0 (array_like):
                The initial value of the solution.
            t0 (float):
                The initial time.
            tf (float):
                The final time.
            Nt or dt (int or float):
                The number of time steps or the time step. Defaults to 100 or 0.01 respectively.
            bc (function, optional):
                The boundary condition.
                2 arguments: ya, yb
                3 arguments: ya, yb, t (for time-dependent boundary conditions)

        For PDEs:
            q (function):
                The differential equation to solve.
                3 arguments: x, t, u
            ic (function):
                The initial condition.
            t0 (float):
                The initial time.
            tf (float):
                The final time.
            Nt or dt (int or float):
                The number of time steps or the time step.
            a (float):
                The left boundary of the domain.
            b (float):
                The right boundary of the domain.
            Nx or dx (int or float):
                The number of grid points or the grid spacing. Defaults to 'self.Nt' and 'self.dt' respectively.
            C or D (float):
                "D" is the diffusion coefficient, "C" is the courant number.
            bc (function, optional):
                The boundary condition.
                    2 arguments: ya, yb
                    3 arguments: ya, yb, t (for time-dependent boundary conditions)

        Attributes:
            f (function):
                The differential equation to solve.
                2 arguments: t, y
            q (function):
                The differential equation to solve.
                3 arguments: x, t, u
            y0 (array_like):
                The initial value of the solution.
            ic (function):
                The initial condition.
            t0 (float):
                The initial time.
            tf (float):
                The final time.
            Nt (int):
                The number of time steps.
            dt (float):
                The time step.
            a (float):
                The left boundary of the domain.
            b (float):
                The right boundary of the domain.
            Nx (int):   
                The number of grid points.
            dx (float):
                The grid spacing.
            C (float):
                The courant number.
            D (float):
                The diffusion coefficient.
            bc (function):
                The boundary condition.
                2 arguments: ya, yb
                3 arguments: ya, yb, t (for time-dependent boundary conditions)
            args (tuple):
                The arguments of the differential equation.
            _func (function):
                A temporary variable for storing the function with the correct number of arguments.
        
        """
        # Store the parameters as a dictionary
        self.params = kwargs

        # ODE attributes
        self.f = None
        self.y0 = None

        # PDE attributes
        self.q = None
        self.ic = None
        self.C = None
        self.D = None

        # Time attributes (for ODEs and PDEs) 
        self.t0 = None
        self.tf = None
        self.dt = None
        self.Nt = None

        # Spatial attributes (for PDEs)
        self.a = None
        self.b = None
        self.Nx = None
        self.dx = None

        # Boundary conditions (if any)
        self.bc = None

        # Function arguments (if any)
        self.args = None
        self._func = None

        # Problem type
        self.problem_type = None

        # Update the attributes
        self.__dict__.update(kwargs)

        # Validate the parameters
        self._validate_parameters()

    def __repr__(self):
        """
        Returns a Table of the problem parameters.

        Usage:
            >>> from problem import Problem

            >>> def f(x, y):
            ...     return np.array([y[1], -y[0]])

            >>> def bc(ya, yb, u):
            ...     return [ya[0], 1]

            >>> problem = Problem(f=f, y0=np.array([0, 1]), t0=0, tf=1, Nt=100, bc=bc, a=0, b=1, Nx=100)

            ______               __     __                  
            |   __ \.----..-----.|  |--.|  |.-----..--------.
            |    __/|   _||  _  ||  _  ||  ||  -__||        |
            |___|   |__|  |_____||_____||__||_____||__|__|__|
                                                            
            Parameter           Value
            ------------------  ----------------------------------
            Equation type       ODE
            Problem type        BVP
            Equation            def f(x, y):
                                    return np.array([y[1], -y[0]])
            Boundary condition  def bc(ya, yb, u):
                                    return [ya[0], 1]
            Initial condition   -
            t0                  0
            tf                  1
            dt                  0.01
            Nt                  100
            y0                  [0 1]
            a                   
            b                   
            dx
            Nx
            C
            D
        """
        # Get the equation type and problem type
        eq_type = self.problem_type[0]
        prob_type = self.problem_type[1]
        eq = ("".join(inspect.getsourcelines(self.f)[0]) if eq_type == "ODE" else "".join(inspect.getsourcelines(self.q)[0])) if self._func is None else "".join(inspect.getsourcelines(self._func)[0])

        # Get the boundary condition and initial condition source code
        bc = "".join(inspect.getsourcelines(self.bc)[0]) if 'bc' in self.params else '-'
        ic = "".join(inspect.getsourcelines(self.ic)[0]) if 'ic' in self.params else '-'

        params = {
            'Equation type': eq_type,
            'Problem type': prob_type,
            'Equation': eq,
            'Boundary condition': bc,
            'Initial condition': ic,
            't0': self.t0,
            'tf': self.tf,
            'dt': self.dt,
            'Nt': self.Nt,
            'y0': self.y0,
            'a': self.a,
            'b': self.b,
            'dx': self.dx,
            'Nx': self.Nx,
            'C': self.C,
            'D': self.D,
            'args': self.args
        }
        return text2art("Problem", font="chunky") + tabulate(params.items(), headers=['Parameter', 'Value'])

        
    def _validate_parameters(self):
        """
        Validates the parameters of the problem.
        
        Raises:
            ValueError: If the parameters are invalid.
        """

        # Validate f and q.
        if 'f' not in self.params and 'q' not in self.params:
            raise ValueError("If solving an ODE, `f` must be specified. If solving a PDE, `q` must be specified.")
        
        # Validate y0 and ic.
        if 'y0' not in self.params and 'ic' not in self.params:
            raise ValueError("If solving an ODE, `y0` must be specified. If solving a PDE, `ic` must be specified.")
        
        # Validate t0 and tf.
        if 't0' not in self.params or 'tf' not in self.params:
            raise ValueError("Initial and final times `t0` and `tf` must be specified.")
        if not isinstance(self.t0, numbers.Real) or not isinstance(self.tf, numbers.Real):
            raise ValueError("Initial and final times `t0` and `tf` must be real numbers.")
        if self.t0 >= self.tf:
            raise ValueError("Initial time `t0` must be less than final time `tf`.")
        
        # Validate Nt and dt.
        if 'Nt' not in self.params or 'dt' not in self.params:
            delta_t = self.tf - self.t0
            if self.dt is not None and self.Nt is None:
                # Compute the number of time steps if not specified and the time step is specified.
                # Ensure that the difference between the final time and the initial time does not contain a floating point error (e.g. 0.1 instead of 0.0999...).
                delta_t = self.round_to_const(delta_t, self.dt)
                self.Nt = int(delta_t / self.dt)
            elif self.dt is None and self.Nt is not None:
                # Or the time step if not specified and the number of time steps is specified.
                self.dt = delta_t / (self.Nt)
            else:
                # Or the time step and number of time steps if not specified. Default to 100 time steps.
                self.Nt = 100
                self.dt = delta_t / (self.Nt)
        if 'Nt' in self.params and 'dt' in self.params:
            raise ValueError("Either the number of time steps `Nt` or the time step `dt` must be specified, not both.")
        if 'dt' in self.params and not isinstance(self.dt, numbers.Real):
            raise ValueError("Time step `dt` must be a real number.")
        if 'Nt' in self.params and not isinstance(self.Nt, int):
            raise ValueError("Number of time steps `Nt` must be an integer.")
        if self.dt <= 0 or self.Nt <= 0:
            raise ValueError("Time step `dt` and number of time steps `Nt` must be positive.")
        
        # Setup the full problem type.
        if "f" in self.params:
            self.f = self.params['f']
            if "y0" not in self.params:
                raise ValueError("Initial value `y0` must be specified for ODE.")
            else:
                self.y0 = self.params['y0']
            if "ic" in self.params:
                raise ValueError("Initial condition `ic` must not be specified for ODE.")
            if "bc" in self.params:
                self.bc = self.params['bc']
                self.problem_type = ['ODE', 'BVP']
            else:
                self.problem_type = ['ODE', 'IVP']
        else:
            self.q = self.params['q']
            if "ic" not in self.params:
                raise ValueError("Initial condition `ic` must be specified for PDE.")
            else:
                self.ic = self.params['ic']
            if "y0" in self.params:
                raise ValueError("Initial value `y0` must not be specified for PDE.")
            if "bc" in self.params:
                self.bc = self.params['bc']
                self.problem_type = ['PDE', 'BVP']
            else:
                self.problem_type = ['PDE', 'IVP']
        
        # Check inputs.
        if self.problem_type[0] == 'ODE':
            # Validate f.
            if not callable(self.f):
                raise ValueError("Function `f` must be callable.")
            # Check if kwargs contains "args" tuple.
            if 'args' in self.params:
                if not isinstance(self.params['args'], tuple):
                    raise ValueError("Keyword argument `args` must be a tuple.")
                else:
                    if "self" in inspect.getfullargspec(self.f).args:
                        if len(self.params['args']) + 3 != len(inspect.getfullargspec(self.f).args):
                            raise ValueError(f"Length of `args` tuple must be equal to the number of additional arguments required by function `f`. Currently, `f` requires {len(inspect.getfullargspec(self.f).args) - 3} additional arguments.")
                    else:
                        if len(self.params['args']) + 2 != len(inspect.getfullargspec(self.f).args):
                            raise ValueError(f"Length of `args` tuple must be equal to the number of additional arguments required by function `f`. Currently, `f` requires {len(inspect.getfullargspec(self.f).args) - 2} additional arguments.")

                    # if len(self.params['args']) + 2 != len(inspect.getfullargspec(self.f).args):
                    #     raise ValueError(f"Length of `args` tuple must be equal to the number of additional arguments required by function `f`. Currently, `f` requires {len(inspect.getfullargspec(self.f).args) - 2} additional arguments.")
                    self.args = self.params['args']                    
                    self._func = self.f  # Assign self.f to a temporary variable.
                    self.f = lambda t, y: self._func(t, y, *self.args)  # Define a new lambda function that calls the temporary variable and the additional arguments.
            if len(inspect.getfullargspec(self.f).args) != 2:
                raise ValueError("Function `f` must have two arguments. If additional arguments are needed, use the `args` keyword argument.")
            
            # Validate y0.
            if not isinstance(self.y0, numbers.Real):
                if isinstance(self.y0, (list, tuple)):
                    if len(self.y0) == 0:
                        raise ValueError("Initial value `y0` can't be empty.")
                    else:
                        self.y0 = np.array(self.y0)
                elif isinstance(self.y0, np.ndarray):
                    if self.y0.size == 0:
                        raise ValueError("Initial value `y0` can't be empty.")
                else:
                    raise ValueError("Initial value `y0` must be a real number.")
            else:
                self.y0 = np.array([self.y0])

            # Validate bc if it exists.
            if self.problem_type[1] == 'BVP':
                if not callable(self.bc):
                    raise ValueError("Boundary condition `bc` must be callable.")
                if len(inspect.getfullargspec(self.bc).args) not in [2,3]:
                    raise ValueError("Boundary condition `bc` must have two arguments, three arguments if time dependent coefficients.")
        else:
            # Validate q.
            if not callable(self.q):
                raise ValueError("Function `q` must be callable.")
            # Check if kwargs contains "args" tuple.
            if 'args' in self.params:
                if not isinstance(self.params['args'], tuple):
                    raise ValueError("Keyword argument `args` must be a tuple like (a, b, ).")
                else:
                    self.args = self.params['args']
                    # Assign self.q to a temporary function variable.                    
                    self._func = self.q
                    # Define a new lambda function that calls the temporary function variable and the additional arguments to avoid recursion.
                    self.q = lambda t, x, u: self._func(t, x, u, *self.args)  
            if len(inspect.getfullargspec(self.q).args) != 3:
                raise ValueError("Function `q` must have three arguments. If additional arguments are needed, use the `args` keyword argument.")
            
            # Validate ic.
            if not callable(self.ic):
                raise ValueError("Initial condition `ic` must be callable.")
            if len(inspect.getfullargspec(self.ic).args) != 1:
                raise ValueError("Initial condition `ic` must have one argument.")
            
            # Validate bc if it exists.
            if self.problem_type[1] == 'BVP':
                if not callable(self.bc):
                    raise ValueError("Boundary condition `bc` must be callable.")
                if len(inspect.getfullargspec(self.bc).args) not in [2,3]:
                    raise ValueError("Boundary condition `bc` must have two arguments, three arguments if time dependent coefficients.")
                
            # Validate a and b the spatial boundaries.
            if 'a' not in self.params or 'b' not in self.params:
                raise ValueError("Both boundaries `a` and `b` must be specified for PDE.")
            if not isinstance(self.a, numbers.Real) or not isinstance(self.b, numbers.Real):
                raise ValueError("Boundaries `a` and `b` must be real numbers.")
            if self.a >= self.b:
                raise ValueError("Left boundary `a` must be less than right boundary `b`.")
            
            # Validate dx and Nx.
            if 'dx' not in self.params or 'Nx' not in self.params:
                delta_x = self.b - self.a
                if self.dx is not None and self.Nx is None:
                    # Compute the number of spatial steps if not specified and the spatial step is specified.
                    # Ensure that the difference between the final and initial spatial boundaries does not contain a floating point error.
                    delta_x = self.round_to_const(delta_x, self.dt)
                    self.Nx = int(delta_x / self.dx)
                elif self.dx is None and self.Nx is not None:
                    # Or the spatial step if not specified and the number of spatial steps is specified.
                    self.dx = delta_x / (self.Nx)
                else:
                    # Or the spatial step and the number of spatial steps if neither are specified. Default to 'self.Nt' spatial steps.
                    self.Nx = self.Nt
                    self.dx = delta_x / (self.Nx)
            if 'dx' in self.params and 'Nx' in self.params:
                raise ValueError("Either the spatial step `dx` or the number of spatial steps `Nx` must be specified. Not both.")
            if 'dx' in self.params and not isinstance(self.dx, numbers.Real):
                raise ValueError("Spatial step `dx` must be a real number.")
            if 'Nx' in self.params and not isinstance(self.Nx, numbers.Integral):
                raise ValueError("Number of spatial steps `Nx` must be an integer.")
            if self.dx <= 0 or self.Nx <= 0:
                raise ValueError("Spatial step `dx` and number of spatial steps `Nx` must be positive.")

            # Validate C and D.
            if 'C' not in self.params and 'D' not in self.params:
                raise ValueError("Either the courant coefficient `C` or the diffusion coefficient `D` must be specified.")
            if 'C' in self.params and 'D' in self.params:
                raise ValueError("Either the courant coefficient `C` or the diffusion coefficient `D` must be specified. Not both.")
            if 'C' in self.params and not isinstance(self.C, numbers.Real):
                raise ValueError("Convection coefficient `C` must be a real number.")
            if 'D' in self.params and not isinstance(self.D, numbers.Real):
                raise ValueError("Diffusion coefficient `D` must be a real number.")
            if self.C is not None and self.D is None:
                self.D = self.C*self.dx**2/self.dt
            else:
                self.C = self.D*self.dt/self.dx**2
            if self.C < 0 or self.D < 0:
                raise ValueError("Convection coefficient `C` and diffusion coefficient `D` must be non-negative.")
            if self.C > 0.5:
                raise ValueError(f"Convection coefficient `C` must be less than 0.5 for stability. Current value is {round(self.C, 3)}.")

    
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
