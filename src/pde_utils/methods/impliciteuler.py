from ...solution import Solution
from .finitedifference import FiniteDifference
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import numpy as np



class ImplicitEuler(FiniteDifference):
    """
    Implicit Euler method for solving the PDE
    
    Args:
        grid (Grid): Grid object containing the discretization information

    Attributes:
        grid (Grid): Grid object containing the discretization information

    Methods:
        solve(q, ic=None, bc=None, method=None, root_finder=None, bc_type="DIRICHLET", matrix_type="SPARSE"):
            Solves PDE with a source term q using the Implicit Euler method

        setup_matrix(matrix_type):
            Sets up coefficient matrix A for Implicit Euler method using either a sparse or dense matrix
    """
    def __init__(self, grid):
        super().__init__(grid)

    def solve(self, q, ic=None, bc=None, method=None, root_finder=None, bc_type="DIRICHLET", matrix_type="SPARSE"):
        """
        Solves PDE with a source term q using the Implicit Euler method

        Args:
            q (function): Source term function
            ic (function): Initial condition function
            bc (function): Boundary condition function
            method (function): Method for solving the linear system
            root_finder (function): Root finding method for solving the nonlinear system
            bc_type (str): Type of boundary condition
            matrix_type (str): Type of coefficient matrix A

        Returns:
            Solution: Solution object containing the solution to the PDE
        """
        
        # Construct the coefficient matrix A
        A = self.setup_matrix(matrix_type.upper())

        # Initialize the solution array U
        U = np.zeros((self.Nt+1, self.Nx+1))

        # Handle the boundary conditions
        U = self.bc_handler(bc_type.upper(), bc, U, t=self.t[0])

        # Set the initial condition
        if ic is not None:
            U[0, :] = ic(self.x)

        # Solve the PDE at different time steps and space steps
        for n in range(self.Nt):
            # Construct the right-hand side vector
            b = U[n, 1:-1] + self.dt*np.array(q(self.t[n], self.x[1:-1], U[n, 1:-1]))

            # Solve the linear system using the method specified
            U[n+1, 1:-1] = self.matrix_handler(matrix_type.upper(), A, b)

            # Handle the boundary conditions
            U = self.bc_handler(bc_type.upper(), bc, U, t=self.t[n+1])

        return Solution(self.t, self.x, U)
    
    def setup_matrix(self, matrix_type):
        # Set up sparse matrix for Implicit Euler method
        if matrix_type.upper() == 'SPARSE':
            main_diag = np.ones(self.Nx-1)*(1 + 2*self.C)
            off_diag = np.ones(self.Nx-2)*(-self.C)
            A = diags([main_diag, off_diag, off_diag], [0, -1, 1], shape=(self.Nx-1, self.Nx-1)).tocsr()
        elif matrix_type.upper() == 'DENSE':
            # set up dense matrix for Implicit Euler method
            A = np.zeros((self.Nx-1, self.Nx-1))
            A[0, 0] = 1 + self.C
            A[0, 1] = -self.C
            for i in range(1, self.Nx-2):
                A[i, i-1] = -self.C
                A[i, i] = 1 + 2 * self.C
                A[i, i+1] = -self.C
            A[self.Nx-2, self.Nx-3] = -self.C
            A[self.Nx-2, self.Nx-2] = 1 + self.C
        else:
            raise NameError("Matrix type not recognized. Please choose either 'sparse' or 'dense'.")

        return A

