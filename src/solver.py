from .utils import IVPMETHODS, PDEMETHODS, ROOTFINDERS, MATRIXTYPES
from abc import abstractmethod
from .problem import Problem
from decimal import Decimal
from numbers import Number
import numpy as np
import inspect
import copy


class Solver:
    """
    Base class for solving IVP and BVP problems

    Classes:
        Solver
            Base class for solving initial value problems

    Attributes:
        f (function):
            The differential equation to solve.
        y0 (array_like):
            The initial value of the solution.
        t0 (float):
            The initial time.
        tf (float):
            The final time.
        dt (float):
            The time step.
        Nt (int):
            The number of time steps.
        a (float):
            The left boundary of the domain.
        b (float):
            The right boundary of the domain.
        ic (function):
            The initial condition.
        bc (function):
            The boundary condition.
        q (function):
            The source term.
        C (float):
            The diffusion coefficient.
        D (float):
            The reaction coefficient.
        Nx (int):
            The number of spatial steps.
        dx (float):
            The spatial step.
        problem (Problem):
            The problem to solve.
        method (str):
            The method to use for solving the problem.
        pde_method (str):
            The method to use for solving the PDE.
        root_finder (str):
            The root finder to use for solving the PDE.
        IVPMETHODS (dict):
            Dictionary of available ODE methods.
        PDEMETHODS (dict):
            Dictionary of available PDE methods.
        ROOTFINDERS (dict):
            Dictionary of available root finders.

    Methods:
        solve():
            Solve the problem.
        bc_handler():
            Handle the boundary conditions.
        _solve_ode():
            Solve the ODE.
        _solve_pde():
            Solve the PDE.
        _method_handler(method, pde_method, root_finder):
            Handle the method, pde_method, and root_finder arguments.
    """


    def __init__(self, problem, **kwargs):
        """
        Args:
            problem (Problem): Problem object
            
        Raises:
            TypeError: If the input arguments are invalid or inconsistent.
        """
        if not isinstance(problem, Problem):
            raise TypeError("Problem must be a Problem object")

        # Store the problem object
        self.problem = problem

        # ODE attributes
        self.f = problem.f
        self.y0 = problem.y0

        # PDE attributes
        self.q = problem.q
        self.ic = problem.ic
        self.C = problem.C
        self.D = problem.D

        # Time attributes (for ODEs and PDEs)
        self.t0 = problem.t0
        self.tf = problem.tf
        self.dt = problem.dt
        self.Nt = problem.Nt

        # Spatial attributes (for PDEs)
        self.a = problem.a
        self.b = problem.b
        self.Nx = problem.Nx
        self.dx = problem.dx

        # Boundary conditions (if any)
        self.bc = problem.bc
        
        # Function arguments (if any)
        self.args = problem.args
        self._func = problem._func

        # Problem attributes
        self.problem_type = problem.problem_type
        self.is_ode = True if problem.problem_type[0] == 'ODE' else False
        self.is_ivp = True if problem.problem_type[0] == 'IVP' else False

        # Method attributes
        self.IVPMETHODS = IVPMETHODS
        self.PDEMETHODS = PDEMETHODS
        self.ROOTFINDERS = ROOTFINDERS
        self.MATRIXTYPES = MATRIXTYPES
        self.method = self.IVPMETHODS["RK45"]
        self.pde_method = self.PDEMETHODS["FINITEDIFFERENCE"]
        self.root_finder = self.ROOTFINDERS["ROOT"]


    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def bc_handler(self):
        pass

    @abstractmethod
    def _solve_ode(self):
        pass

    @abstractmethod
    def _solve_pde(self):
        pass

    @abstractmethod
    def _method_handler(self, method, pde_method, root_finder, matrix_type):
        """
        Handle the method, pde_method, and root_finder arguments.

        Args:
            method (str): The method to use for solving the problem.
            pde_method (str): The method to use for solving the PDE.
            root_finder (str): The root finder to use for solving the PDE.

        Raises:
            TypeError: If the input arguments are invalid or inconsistent.

        Returns:
            None
        """
        method = method.upper()
        pde_method = pde_method.upper()
        root_finder = root_finder.upper()
        matrix_type = matrix_type.upper()

        # Check if method is valid
        if method not in self.IVPMETHODS:
            raise TypeError(f"The method chosen: {method}, is not valid\nPlease choose from the following: {self.IVPMETHODS.keys()}")
        if isinstance(method, str):
            method = self.IVPMETHODS[method]
        else:
            raise TypeError(f"Method must be a string\nPlease choose from the following: {self.IVPMETHODS.keys()}")
        
        if not self.is_ode:
            # Check if pde method is valid
            if pde_method not in self.PDEMETHODS:
                raise TypeError(f"The method chosen: {pde_method}, is not valid\nPlease choose from the following: {self.PDEMETHODS.keys()}")
            if isinstance(pde_method, str):
                pde_method = self.PDEMETHODS[pde_method]
            else:
                raise TypeError(f"Method must be a string\nPlease choose from the following: {self.PDEMETHODS.keys()}")
            if not self.is_ivp:
                # Check if matrix type is valid
                if not isinstance(matrix_type, str):
                    raise TypeError(f"Matrix type must be a string\nPlease choose from the following: {self.MATRIXTYPES.keys()}")
                if matrix_type not in self.MATRIXTYPES:
                    raise TypeError(f"The matrix type chosen: {matrix_type}, is not valid\nPlease choose from the following: {self.MATRIXTYPES}")
            
        # Check if root finder is valid
        if root_finder not in self.ROOTFINDERS:
            raise TypeError(f"The root finder chosen: {root_finder}, is not valid\nPlease choose from the following: {self.ROOTFINDERS.keys()}")
        if isinstance(root_finder, str):
            root_finder = self.ROOTFINDERS[root_finder]
        else:
            raise TypeError(f"Root finder must be a string\nPlease choose from the following: {self.ROOTFINDERS.keys()}")
            
        self.method = method
        self.pde_method = pde_method
        self.root_finder = root_finder
        self.matrix_type = matrix_type


    def _continuation_handler(self, **kwargs):
        """
        Check that the continuation method parameters are valid

        Args:
            **kwargs (dict): Continuation method parameters
            Supported parameters:
                p0 (array, list, tuple, int, float): Initial parameter value
                vary_par (int): Index of the parameter to vary
                p_span (array, list, tuple, int, float): Parameter span
                ds (int, float): Continuation step size
                Ns (int): Number of continuation steps
                max_ds (int, float): Maximum continuation step size
                min_ds (int, float): Minimum continuation step size
                cont_type (str): Continuation type
                output (bool): Whether to output the root finding results message

        Raises:
            TypeError: If the input arguments are invalid or inconsistent.

        Returns:
            tuple: Tuple containing the continuation method parameters
        """
        # Get the continuation method parameters
        p0 = kwargs.get('p0', None)
        q_cont = kwargs.get('q_cont', None)
        vary_par = kwargs.get('vary_par', None)
        p_span = kwargs.get('p_span', None)
        ds = kwargs.get('ds', None)
        Ns = kwargs.get('Ns', None)
        max_ds = kwargs.get('max_ds', None)
        min_ds = kwargs.get('min_ds', None)
        cont_type = kwargs.get('cont_type', 'PSEUDO')
        da = kwargs.get('da', None)
        output = kwargs.get('output', False)

        # Check that p0 is valid or if self.args is valid
        if p0 is None:
            if self.args is None:
                raise TypeError("'p0' must be specified if the original function doesn't contain the parameter(s) you want to vary.")
            else:
                p0 = copy.copy(self.args)
        if isinstance(p0, (int, float)):
            p0 = [p0]
        if not isinstance(p0, (list, tuple, np.ndarray)):
            raise TypeError("'p0' must be a list, tuple, array, or scalar")
        if not all(isinstance(p, (int, float)) for p in p0):
            raise TypeError("'p0' must be a list, tuple, array, or scalar of integers or floats")
        p0 = np.array(p0)

        # Check that q_cont is valid or if self._func is valid
        if q_cont is None:
            if self._func is None:
                raise TypeError("'q_cont' must be specified if the original function doesn't contain the parameter(s) you want to vary.")
            else:
                q_cont = self._func
        # Check that q_cont is callable with *p0
        try:
            q_cont(self.t0, self.y0, *p0)
        except:
            raise TypeError(f"'{q_cont.__name__}' must have arguments of length 2 + len(p0)")


        # Check that vary_par is valid
        if vary_par is None:
            if isinstance(p0, (list, tuple, np.ndarray)) and len(p0) > 1:
                raise TypeError("'vary_par' must be specified if 'p0' is a list, tuple, or array of length > 1")
            else:
                vary_par = 0
        if not isinstance(vary_par, int):
            raise TypeError("'vary_par' must be an integer")
        if isinstance(p0, (list, tuple, np.ndarray)) and (vary_par < 0 or vary_par >= len(p0)):
            raise TypeError("'vary_par' must be a valid index of p0")

        # Check that p_span is valid
        if p_span is None:
            if ds is None and Ns is None:
                raise TypeError("Either 'p_span', 'ds', or 'Ns' must be specified")
            else:
                if ds is not None and Ns is not None:
                    p_span = [p0, p0 + Ns*ds]
                elif ds is not None:
                    p_span = [p0, p0 + self.Nt*ds]
                else:
                    p_span = [p0, p0 + Ns*self.dt]
        if not isinstance(p_span, (list, tuple, np.ndarray)):
            raise TypeError("'p_span' must be a list, tuple, or array")
        else:
            p_span = np.array(p_span)
            if len(p_span) != 2:
                raise TypeError("'p_span' must be a list, tuple, or array of length 2")
            if not isinstance(p_span[0], (Number)):
                raise TypeError("'p_span' must be a list, tuple, or array of numbers")
            if not isinstance(p_span[1], (Number)):
                raise TypeError("'p_span' must be a list, tuple, or array of numbers")
            if p_span[0] >= p_span[1]:
                raise TypeError("'p_span[0]' must be less than 'p_span[1]'")
            if p_span[0] > p0[vary_par] or p_span[1] < p0[vary_par]:
                raise TypeError("'p0' must be within 'p_span'")
            
        # Check that ds is valid
        if ds is None:
            if Ns is None:
                ds = (p_span[1] - p_span[0])/(self.Nt)
            else:
                ds = (p_span[1] - p_span[0])/(Ns)
        if not isinstance(ds, (int, float)):
            raise TypeError("'ds' must be a number")

        # Check that Ns is valid
        if Ns is None:
            Ns = int(abs((p_span[1] - p_span[0])/ds))
        if not isinstance(Ns, int):
            raise TypeError("'Ns' must be an integer")
        if Ns <= 0:
            raise TypeError("'Ns' must be positive")
        self.Nt = Ns
        self.dt = (self.tf-self.t0)/(self.Nt)

        # Check that max_ds is valid
        if max_ds is None:
            max_ds = 10 * ds
        if not isinstance(max_ds, (int, float)):
            raise TypeError("'max_ds' must be a number")
        if max_ds <= 0:
            raise TypeError("'max_ds' must be positive")
        if ds > max_ds:
            raise TypeError("'ds' must be less than or equal to 'max_ds'")
        
        # Check that min_ds is valid
        if min_ds is None:
            min_ds = ds/10
        if not isinstance(min_ds, (int, float)):
            raise TypeError("'min_ds' must be a number")
        if min_ds <= 0:
            raise TypeError("'min_ds' must be positive")
        if ds < min_ds:
            raise TypeError("'ds' must be greater than or equal to 'min_ds'")

        # Check that cont_type is valid
        if not isinstance(cont_type, str):
            raise TypeError("'cont_type' must be a string")
        if cont_type.upper() not in ['PSEUDO', 'NATURAL']:
            raise TypeError("'cont_type' must be 'PSEUDO' or 'NATURAL'")
        cont_type = cont_type.upper()

        # Check that da is valid
        if da is None:
            da = 0.01
        if not isinstance(da, (int, float)):
            raise TypeError("'da' must be a number")
        if da <= 0:
            raise TypeError("'da' must be positive")

        # Check that output is valid
        if not isinstance(output, bool):
            raise TypeError("'output' must be a boolean")

        return p0, q_cont, vary_par, p_span, ds, Ns+1, max_ds, min_ds, cont_type, da, output
    

    def round_to_const(self, value, const):
        """
        Round a value to the number of decimal places of a constant

        Args:
            value (int or float): Value to round
            const (int or float): Constant to round to

        Returns:
            int or float: Rounded value
        """
        if const == 0:
            return value
        num_decimals = abs(Decimal(str(const)).as_tuple().exponent)

        return round(value, num_decimals)