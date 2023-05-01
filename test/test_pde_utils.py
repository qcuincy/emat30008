from .context import src
from src.pde_utils.methods import *
from src.pde_utils.grid import Grid
import numpy as np
import unittest





class TestFiniteDifference(unittest.TestCase):

    def setUp(self):
        self.grid = Grid(a=0, b=1, t0=0, tf=1, Nx=10, Nt=10, C=0.5)
        self.fd = FiniteDifference(self.grid)

    def test_solve_with_no_ic_or_bc(self):
        # No initial condition or boundary condition
        q = lambda x, t, U: np.zeros(U.shape)
        sol = self.fd.solve(q=q)
        U = sol.u
        self.assertEqual(U.shape, (self.grid.Nt+1, self.grid.Nx+1))
        self.assertTrue(np.allclose(U, np.zeros((self.grid.Nt+1, self.grid.Nx+1))))

    def test_solve_with_ic_only(self):
        # Initial condition only
        ic = lambda x: np.sin(np.pi*x)
        q = lambda t, x, U: np.zeros(U.shape)
        grid = Grid(a=0, b=1, t0=0, tf=10, Nx=100, Nt=100, C=0)
        sol = FiniteDifference(self.grid).solve(q=q, ic=ic, root_finder='root')
        U = sol.u
        self.assertEqual(U.shape, (self.grid.Nt+1, self.grid.Nx+1))
        self.assertTrue(np.allclose(U[0], ic(self.grid.x)))
        self.assertTrue(np.allclose(U[-1][0], 0))

    def test_solve_with_bc_only(self):
        # Boundary condition only
        bc = lambda u_a, u_b, t: np.array((0, 0))
        q = lambda t, x, U: np.zeros(U.shape) 
        sol = self.fd.solve(q=q, bc=bc, method="RK45")
        U = sol.u
        self.assertEqual(U.shape, (self.grid.Nt+1, self.grid.Nx+1))
        self.assertTrue(np.allclose(U[:, 0], 0))
        self.assertTrue(np.allclose(U[:, -1], 0))

    def test_solve_with_dirichlet_bc(self):
        # Dirichlet boundary condition
        bc = lambda u_a, u_b, t: np.array((0, 0))
        q = lambda x, t, U: np.zeros(U.shape)
        sol = self.fd.solve(q=q, bc=bc, bc_type='DIRICHLET')
        U = sol.u
        self.assertEqual(U.shape, (self.grid.Nt+1, self.grid.Nx+1))
        self.assertTrue(np.allclose(U[:, 0], bc(0, 0, self.grid.t)[0]))
        self.assertTrue(np.allclose(U[:, -1], bc(0, 0, self.grid.t)[1]))

    def test_solve_with_neumann_bc(self):
        # Neumann boundary condition
        bc = lambda u_a, u_b, t: np.array((0, 0))
        q = lambda x, t, U: np.zeros(U.shape)
        sol = self.fd.solve(q=q, bc=bc, bc_type='NEUMANN')
        U = sol.u
        self.assertEqual(U.shape, (self.grid.Nt+1, self.grid.Nx+1))
        self.assertTrue(np.allclose(U[:, 0], U[:, 1] - self.grid.dx*bc(0, 0, self.grid.t)[0]))
        self.assertTrue(np.allclose(U[:, -1], U[:, -2] + self.grid.dx*bc(0, 0, self.grid.t)[-1]))

    def test_solve_with_robin_bc(self):
        # Robin boundary condition
        alpha = 1
        beta = 2
        bc = lambda u_a, u_b, t: np.array(([0, alpha], [0, beta]))
        q = lambda x, t, U: np.zeros(U.shape)
        sol = self.fd.solve(q=q, bc=bc, bc_type='ROBIN')
        U = sol.u
        self.assertEqual(U.shape, (self.grid.Nt+1, self.grid.Nx+1))
        self.assertTrue(np.allclose(U[:, 0], alpha*U[:, 0] + beta*U[:, 1]))
        self.assertTrue(np.allclose(U[:, -1], alpha*U[:, -1] + beta*U[:, -2]))

    def test_solve_with_ic_and_bc(self):
        # Initial condition and boundary condition
        ic = lambda x: np.sin(np.pi*x)
        bc = lambda u_a, u_b, t: (0, 0)
        q = lambda x, t, U: np.zeros(U.shape)
        sol = self.fd.solve(q=q, ic=ic, bc=bc)
        U = sol.u
        self.assertEqual(U.shape, (self.grid.Nt+1, self.grid.Nx+1))
        self.assertTrue(np.allclose(U[0], ic(self.grid.x)))
        self.assertTrue(np.allclose(U[:, 0], 0))
        self.assertTrue(np.allclose(U[:, -1], 0))


if __name__ == '__main__':
    unittest.main()
