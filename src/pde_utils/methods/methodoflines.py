from ...solution import Solution
from .finitedifference import FiniteDifference
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import numpy as np

class MethodOfLines(FiniteDifference):
    """
    Method of Lines for solving the PDE

    Args:
        grid (Grid): Grid object containing the discretization information
        theta (float): Parameter for the method of lines

    Attributes:
        grid (Grid): Grid object containing the discretization information
        theta (float): Parameter for the method of lines

    Methods:
        solve(q, ic=None, bc=None, method="RK45", root_finder=None, bc_type="DIRICHLET", matrix_type="SPARSE"):
            Solves PDE with a source term q using the method of lines

        setup_matrix(matrix_type):
            Sets up coefficient matrix A for the method of lines using either a sparse or dense matrix
    """
    def __init__(self, grid, theta=0.5):
        super().__init__(grid)
        self.theta = theta

    def solve(self, q, ic=None, bc=None, method="RK45", root_finder=None, bc_type="DIRICHLET", matrix_type="SPARSE"):
        """
        Solves PDE with a source term q using the method of lines

        Args:
            q (function): Source term function
            ic (function): Initial condition function. Defaults to None.
            bc (function): Boundary condition function. Defaults to None.
            method (function): Method for solving the linear system. Defaults to None.
            root_finder (function): Root finder for solving the nonlinear system. Defaults to None.
            bc_type (str): Type of boundary condition. Either "DIRICHLET" or "NEUMANN". Defaults to "DIRICHLET".
            matrix_type (str): Type of matrix to use. Either "sparse" or "dense". Defaults to "sparse".

        Returns:
            Solution: Solution object containing the solution to the PDE
        """
        # Initialize the solution array U
        U = np.zeros((self.Nt+1, self.Nx+1))

        # Set the initial condition
        if ic is not None:
            U[0, :] = ic(self.x)

        # Handle the boundary conditions
        U = self.bc_handler(bc_type.upper(), bc, U, self.t[0])

        # Construct the coefficient matrix A
        A = self.setup_matrix(matrix_type.upper())

        # Solve the PDE at different time steps and space steps
        for n in range(self.Nt):
            # Construct the right-hand side vector
            b = q(self.t[n], self.x[1:-1], U[n, 1:-1])

            # Solve the linear system using the method specified
            U[n+1, 1:-1] = self.method_handler(method, root_finder, A, U, self.C, q, n)

            # Handle the boundary conditions
            U = self.bc_handler(bc_type.upper(), bc, U, t=self.t[n+1])

        return Solution(self.t, self.x, U)
    

    def setup_matrix(self, matrix_type):
        if matrix_type.upper() == 'SPARSE':
            # Set up sparse matrix for MOL
            main_diag = -2*self.C*np.ones(self.Nx-1)
            sub_diag = self.C*np.ones(self.Nx-2)
            A = diags([sub_diag, main_diag, sub_diag], [-1, 0, 1], shape=(self.Nx-1, self.Nx-1))
        elif matrix_type.upper() == 'DENSE':
            # Set up dense matrix for MOL
            A = np.zeros((self.Nx-1, self.Nx-1))
            for i in range(self.Nx-1):
                A[i, i] = -2*self.C
                if i > 0:
                    A[i, i-1] = self.C
                if i < self.Nx-2:
                    A[i, i+1] = self.C
        else:
            raise NameError("Matrix type not recognized. Please choose either 'sparse' or 'dense'.")
                
        return A

