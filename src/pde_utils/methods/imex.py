from ...solution import Solution
from .finitedifference import FiniteDifference
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import scipy.sparse as sps
import numpy as np


class IMEX(FiniteDifference):
    """
    IMEX method for solving the PDE

    Args:
        grid (Grid): Grid object containing the discretization information
        a (float): Parameter for the IMEX method. Defaults to 1/2.

    Attributes:
        grid (Grid): Grid object containing the discretization information
        a (float): Parameter for the IMEX method

    Methods:
        solve(q, ic=None, bc=None, method=None, root_finder=None, bc_type="DIRICHLET", matrix_type="SPARSE"):
            Solves PDE with a source term q using the IMEX method

        setup_matrix(matrix_type):
            Sets up coefficient matrices A and B for the IMEX method using either a sparse or dense matrix
    """
    def __init__(self, grid, a=None):
        super().__init__(grid)
        self.alpha = a or 1/2

    def solve(self, q, ic=None, bc=None, method=None, root_finder=None, bc_type="DIRICHLET", matrix_type="SPARSE"):
        """
        Solves PDE with a source term q using the IMEX method

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

        # Construct the coefficient matrices A and B
        A, B = self.setup_matrix(matrix_type)        

        # Initialize the solution array U
        U = np.zeros((self.Nt+1, self.Nx+1))

        # Handle the boundary conditions
        U = self.bc_handler(bc_type.upper(), bc, U, t=self.t[0])

        # Set the initial condition
        if ic is not None:
            U[0, :] = ic(self.x)
        

        # Solve the PDE at different time steps and space steps
        for n in range(self.Nt):
            # Create the right-hand side vector f using the previous solution
            f = B @ U[n, 1:-1] + self.dt * (1 - self.alpha) * q(self.t[n], self.x[1:-1], U[n, 1:-1]) + self.dt * self.alpha * np.array(q(self.t[n+1], self.x[1:-1], U[n, 1:-1]))
            # Solve the linear system using the method specified
            U[n+1, 1:-1] = self.matrix_handler(matrix_type.upper(), A, f) 

            # Handle the boundary conditions
            U = self.bc_handler(bc_type.upper(), bc, U, t=self.t[n+1])


        return Solution(self.t, self.x, U)


    def setup_matrix(self, matrix_type):
        """
        Set up the coefficient matrices A and B for a given matrix type.

        Args:
            matrix_type (str): Type of matrix to use. Either "sparse" or "dense"

        Returns:
            A (ndarray): Coefficient matrix for the implicit part of the method
            B (ndarray): Coefficient matrix for the explicit part of the method

        Raises:
            NameError: If the matrix type is not recognized
        """
        if matrix_type.upper() == "SPARSE":
            # Set up the sparse matrix for the implicit part of the method
            diagonal = np.ones(self.Nx-1) * (1 + 2 * self.C * self.alpha)
            lower = upper = -self.C * self.alpha * np.ones(self.Nx-2)
            A = diags([diagonal, lower, upper], [0, -1, 1]).toarray()

            # Set up the sparse matrix for the explicit part of the method
            diagonal = np.ones(self.Nx-1) * (1 - 2 * self.C * (1 - self.alpha))
            lower = upper = self.C * (1 - self.alpha) * np.ones(self.Nx-2)
            B = diags([diagonal, lower, upper], [0, -1, 1]).toarray()

        elif matrix_type.upper() == "DENSE":
            # Set up the dense matrix for the implicit part of the method
            A = np.zeros((self.Nx-1, self.Nx-1))
            A[0, 0] = 1 + 2 * self.C * self.alpha
            A[0, 1] = -self.C * self.alpha
            for i in range(1, self.Nx-2):
                A[i, i-1] = -self.C * self.alpha
                A[i, i] = 1 + 2 * self.C * self.alpha
                A[i, i+1] = -self.C * self.alpha
            A[self.Nx-2, self.Nx-3] = -self.C * self.alpha
            A[self.Nx-2, self.Nx-2] = 1 + 2 * self.C * self.alpha

            # Set up the dense matrix for the explicit part of the method
            B = np.zeros((self.Nx-1, self.Nx-1))
            B[0, 0] = 1 - 2 * self.C * (1 - self.alpha)
            B[0, 1] = self.C * (1 - self.alpha)
            for i in range(1, self.Nx-2):
                B[i, i-1] = self.C * (1 - self.alpha)
                B[i, i] = 1 - 2 * self.C * (1 - self.alpha)
                B[i, i+1] = self.C * (1 - self.alpha)
            B[self.Nx-2, self.Nx-3] = self.C * (1 - self.alpha)
            B[self.Nx-2, self.Nx-2] = 1 - 2 * self.C * (1 - self.alpha)

        else:
            raise NameError("Matrix type not recognized. Please choose either 'sparse' or 'dense'.")

        return A, B
