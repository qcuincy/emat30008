from ..root_finders.newton import Newton
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from ...solution import Solution
from scipy.optimize import root
from ...problem import Problem
from abc import abstractmethod
from ..grid import Grid
import numpy as np
import inspect


class FiniteDifference():
    """
    FiniteDifference class for solving PDEs and ODEs using finite difference methods

    Classes:
        FiniteDifference(Solver)
            Class for solving PDEs using finite difference methods

    Attributes:
        grid (Grid):
            The grid to solve the PDE on.
        a (float):
            The left boundary of the domain.
        b (float):
            The right boundary of the domain.
        C (float):
            The Courant number.
        D (float):
            The diffusion coefficient.
        dx (float):
            The spatial step.
        dt (float):
            The time step.
        Nx (int):
            The number of spatial steps.
        Nt (int):
            The number of time steps.
        x (ndarray):
            The spatial grid.
        t (ndarray):
            The time grid.
        
    Methods:
        solve(q, ic=None, bc=None, method="RK45", root_finder=None, bc_type="DIRICHLET", matrix_type="SPARSE"):
            Solve the PDE using matrix operations with a forcing term using either numpy, scipy's root solver,
            own newton solver, or own ode_ivp solver. The function will check which method is specified and handle the
            arguments accordingly. The function will also call the bc_handler function to handle the boundary conditions.
    """
    def __init__(self, grid):
        if not isinstance(grid, Grid):
            raise TypeError("grid must be an instance of the Grid class")

        self.grid = grid
        self.a = grid.a
        self.b = grid.b
        self.D = grid.D
        self.C = grid.C
        self.dx = grid.dx
        self.dt = grid.dt
        self.Nx = grid.Nx
        self.Nt = grid.Nt
        self.x = grid.x
        self.t = grid.t

    @abstractmethod
    def solve(self, q, ic=None, bc=None, method="RK45", root_finder=None, bc_type="DIRICHLET", matrix_type="SPARSE"):
        """
        Solve the PDE using matrix operations with a forcing term using either numpy, scipy's root solver,
        own newton solver, or own ode_ivp solver. The function will check which method is specified and handle the
        arguments accordingly. The function will also call the bc_handler function to handle the boundary conditions.

        Args:
            q (function): function of t, x, and U that represents the source term of the PDE.
            ic (function): function to set the initial condition. Default is None.
            bc (function): function to set the boundary condition. Default is None.
            method (str): method to solve the PDE. Must be one of "IVP", "EULER", "IEULER", "RK4", "RK45", or "MIDPOINT".
                        Default is "RK45".
            root_finder (str): method to solve the root of the PDE. Must be one of "SOLVE", "ROOT", "NEWTON".
                        Default is None.
            bc_type (str): type of boundary condition. Must be one of "DIRICHLET", "NEUMANN", or "ROBIN".
                        Default is "DIRICHLET".

        Returns:
            ndarray: array of solutions of the PDE at different time steps and space steps.
        """

        # Construct the coefficient matrix A
        A = np.zeros((self.Nx-1, self.Nx-1))
        for i in range(self.Nx-1):
            A[i, i] = 1 + 2*self.C
            if i > 0:
                A[i, i-1] = -self.C
            if i < self.grid.Nx-2:
                A[i, i+1] = -self.C

        # Initialize the solution array U
        U = np.zeros((self.Nt+1, self.Nx+1))

        # Handle the boundary conditions
        U = self.bc_handler(bc_type.upper(), bc, U, self.t[0])

        # Set the initial condition
        if ic is not None:
            U[0, :] = ic(self.x)

        # Solve the PDE at different time steps and space steps
        for n in range(self.Nt):
            # Solve the linear system using the method specified
            U[n+1, 1:-1] = self._method_handler(method, root_finder, A, U, self.C, q, n)

            # Handle the boundary conditions
            U = self.bc_handler(bc_type.upper(), bc, U, t=self.t[n+1])

        return Solution(self.t, self.x, U)


    def bc_handler(self, bc_type, bc, U, t=None):
        """
        Apply boundary conditions of type "DIRICHLET", "NEUMANN" or "ROBIN" to the solution array.

        Args:
        - bc_type:
            str, type of boundary condition to apply
        - bc: 
            function, boundary condition function with signature bc(u_a, u_b, t),
            where u_a and u_b are the boundary conditions at the left and right boundary respectively and t is the current time step.
        - U:
            array_like, solution array to apply the boundary conditions to
        - t:
            float, current time step value, default is None

        Returns:
        - U: array_like, solution array with boundary conditions applied
        """
        if bc is None:
            if np.allclose(U[:, 1], np.zeros_like(U[:, 1])) and np.allclose(U[:, -2], np.zeros_like(U[:, -2])):
                U[:, 0] = U[:, 1]
                U[:, -1] = U[:, -2]
            return U
            
            
        if bc_type == "DIRICHLET":
            if len(inspect.getfullargspec(bc).args) == 2:
                U[:, 0] = bc(self.a, self.b)[0]
                U[:, -1] = bc(self.a, self.b)[1]
                return U
            if len(inspect.getfullargspec(bc).args) == 3:
                U[:, 0] = bc(self.a, self.b, t)[0]
                U[:, -1] = bc(self.a, self.b, t)[1]
                return U
            else:
                raise ValueError("Boundary condition function must have 2 or 3 arguments")

        if bc_type == "NEUMANN":
            if len(inspect.getfullargspec(bc).args) == 2:
                U[:, 0] = U[:, 1] - bc(self.a, self.b)[0] * self.dx
                U[:, -1] = U[:, -2] + bc(self.a, self.b)[1] * self.dx
                return U
            if len(inspect.getfullargspec(bc).args) == 3:
                U[:, 0] = U[:, 1] - bc(self.a, self.b, t)[0] * self.dx
                U[:, -1] = U[:, -2] + bc(self.a, self.b, t)[1] * self.dx
                return U
            else:
                raise ValueError("Boundary condition function must have 2 or 3 arguments")

        if bc_type == "ROBIN":
            if len(inspect.getfullargspec(bc).args) == 2:
                u_a, u_b = bc(self.a, self.b)
                U[:, 0] = (u_a[0] * U[:, 1] + u_a[1] * u_a[0] * self.dx) / (1 + u_a[1] * self.dx)
                U[:, -1] = (u_b[0] * U[:, -2] + u_b[1] * u_b[0] * self.dx) / (1 + u_b[1] * self.dx)
                return U
            if len(inspect.getfullargspec(bc).args) == 3:
                u_a, u_b = bc(self.a, self.b, t)
                U[:, 0] = (u_a[0] * U[:, 1] + u_a[1] * u_a[0] * self.dx) / (1 + u_a[1] * self.dx)
                U[:, -1] = (u_b[0] * U[:, -2] + u_b[1] * u_b[0] * self.dx) / (1 + u_b[1] * self.dx)
                return U
            else:
                raise ValueError("Boundary condition function must have 2 or 3 arguments")
            
        raise ValueError("Boundary condition not recognized, please use one of the following: 'DIRICHLET', 'NEUMANN' or 'ROBIN'")


    def _method_handler(self, method, root_finder, A, U, C, q, n):
        """
        Handle the arguments for the different numerical methods to solve the PDE.
        
        Args:
        - method (str): The name of the method to use. Possible values are:
            * 'IVP': Use the IVP module to solve the initial value problem (IVP) of the PDE.
            * A method from ('EULER', 'IEULER', 'RK4', 'RK45', 'MIDPOINT'): Use the specified Runge-Kutta method to solve the ODE.
        - root_finder (str): The name of the root finder to use. Possible values are:
            * 'ROOT': Use the root function from scipy.optimize to solve the root of the PDE.
            * 'NEWTON': Use the Newton method to solve the root of the PDE.
            * 'SOLVE': Use the numpy.linalg.solve function to solve the linear system.
        - A (ndarray): The matrix A from the discretization of the PDE.
        - U (ndarray): The solution of the PDE at time t[n].
        - C (float): The coefficient of the discretized PDE.
        - q (function): A function of x, t, and U that defines the source term of the PDE.
        - n (int): The index of the time step.
        
        Returns:
        - An ndarray containing the solution of the PDE at time t[n+1] using the specified method.
        """
        if root_finder is not None:
            if callable(root_finder):
                root_finder = root_finder.__name__
            if root_finder.upper() == "SOLVE":
                b = U[n, 1:-1] + C*np.array(q(self.t[n+1], self.x[1:-1], U[n, 1:-1]))
                if not self.is_linear_system(A, b):
                    print("Error: 'numpy' method can only be used for linear systems")
                    return None
                return np.linalg.solve(A, b)

            if root_finder.upper() == "ROOT":
                def f(u):
                    return A@u - U[n, 1:-1] - C*np.array(q(self.t[n+1], self.x[1:-1], U[n, 1:-1]))
                return root(f, U[n, 1:-1]).x

            else:
                def f(u):
                    return A@u - U[n, 1:-1] - C*np.array(q(self.t[n+1], self.x[1:-1], U[n, 1:-1]))

                def fprime(u):
                    return A

                return Newton(f, U[n, 1:-1], fprime=fprime).solve()
        else:
            from ...ivp import IVP
            def f(t, u):
                return A@u - U[n, 1:-1] - C*np.array(q(self.t, self.x[1:-1], U[n, 1:-1]))
            problem = Problem(f=f, y0=U[n, 1:-1], t0=self.t[n], tf=self.t[n+1], Nt=self.Nt)
            sol = IVP(problem).solve(method=method)
            return sol.y[-1, :]
        

    def matrix_handler(self, matrix_type, A, b):
        """
        Handle the arguments for the different matrix types to solve the PDE.
        
        Args:
        - matrix_type (str): The name of the matrix type to use. Possible values are:
            * 'DENSE': Use the dense matrix type.
            * 'SPARSE': Use the sparse matrix type.
        - A (ndarray): The matrix A from the discretization of the PDE.
        - b (ndarray): The vector b from the discretization of the PDE.
        
        Returns:
        - An ndarray containing the solution of the PDE at time t[n+1] using the specified method.
        """
        if matrix_type == "DENSE":
            is_linear = self.is_linear_system(A, b)
            if not is_linear == True:
                raise ValueError(is_linear)
            return np.linalg.solve(A, b)
        if matrix_type == "SPARSE":
            A = csr_matrix(A) if isinstance(A, np.ndarray) else A
            return spsolve(A, b)
        raise ValueError("Matrix type not recognized, please use one of the following: 'DENSE' or 'SPARSE'")
        

    def is_linear_system(self, A, b):
        """
        Check whether the system is linear or not for the given arguments.

        Args:
        - A: np.ndarray, A matrix representing the coefficients of the system.
        - b: np.ndarray, A vector representing the right-hand side of the system.

        Returns:
        - bool, True if the the system could be solved (implying linear), False otherwise.
        """
        
        # Solve the system
        try:
            np.linalg.solve(A, b)
        # except the error as e
        except np.linalg.LinAlgError as e:
            return e

        return True
    
    @abstractmethod
    def setup_matrix(self):
        """
        Setup the matrix A and vector b from the discretization of the PDE.
        """
        pass
