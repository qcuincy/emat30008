from ...solution import Solution
from .finitedifference import FiniteDifference
from scipy.sparse import diags
import numpy as np


class CrankNicolson(FiniteDifference):
    """
    Crank-Nicolson method for solving PDEs

    Args:
        grid (Grid): Grid object containing the discretization information

    Attributes:
        grid (Grid): Grid object containing the discretization information

    Methods:
        solve(q, ic=None, bc=None, method=None, root_finder=None, bc_type="DIRICHLET", matrix_type="SPARSE"):
            Solves PDE with a source term q using the Crank-Nicolson method

        setup_matrix(matrix_type):
            Sets up coefficient matrices A and B for Crank-Nicolson method using either a sparse or dense matrix
    """
    def __init__(self, grid):
        super().__init__(grid)

    def solve(self, q, ic=None, bc=None, method=None, root_finder=None, bc_type="DIRICHLET", matrix_type="SPARSE"):
        """
        Solves PDE with a source term q using the Crank-Nicolson method

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

        # Construct the coefficient matrix A
        A, B = self.setup_matrix(matrix_type.upper())

        # Initialize the solution array U
        U = np.zeros((self.Nt+1, self.Nx+1))

        # Handle the boundary conditions
        U = self.bc_handler(bc_type.upper(), bc, U, self.t[0])

        # # Set the initial condition
        if ic is not None:
            U[0, :] = ic(self.x)

        # Solve the PDE at different time steps and space steps
        for n in range(self.Nt):
            # Construct the right-hand side vector b
            b = B@U[n, 1:-1] + self.dt*np.array(q(self.t[n], self.x[1:-1], U[n, 1:-1]))

            # Solve the linear system using the method specified
            U[n+1, 1:-1] = self.matrix_handler(matrix_type.upper(), A, b)

            # Handle the boundary conditions
            U = self.bc_handler(bc_type.upper(), bc, U, t=self.t[n+1])

        return Solution(self.t, self.x, U)



    def setup_matrix(self, matrix_type):
        """
        Sets up coefficient matrices A and B for Crank-Nicolson method using either a sparse or dense matrix

        Args:
            matrix_type (str): Type of matrix to use. Either "sparse" or "dense"

        Returns:
            A (ndarray): Coefficient matrix A
            B (ndarray): Coefficient matrix B
        """
        if matrix_type.upper() == 'SPARSE':
            # Set up sparse matrix for Crank-Nicolson method
            A = diags([1 + self.C, -self.C/2, -self.C/2], [0, -1, 1], shape=(self.Nx-1, self.Nx-1)).toarray()
            B = diags([1 - self.C, self.C/2, self.C/2], [0, -1, 1], shape=(self.Nx-1, self.Nx-1)).toarray()
        elif matrix_type.upper() == 'DENSE':
            # Set up dense matrix for Crank-Nicolson method
            A = np.zeros((self.Nx-1, self.Nx-1))
            B = np.zeros((self.Nx-1, self.Nx-1))
            for i in range(self.Nx-1):
                A[i, i] = 1 + self.C
                B[i, i] = 1 - self.C
                if i > 0:
                    A[i, i-1] = -self.C/2
                    B[i, i-1] = self.C/2
                if i < self.Nx-2:
                    A[i, i+1] = -self.C/2
                    B[i, i+1] = self.C/2
        else:
            raise NameError("Matrix type not recognized. Please choose either 'sparse' or 'dense'.")
            
        return A, B
