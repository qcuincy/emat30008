import numpy as np

class Newton():
    def __init__(self, f, x0, fprime=None, tol=1e-6, maxiter=100):
        """
        Class for Newton's method. This class is used to solve nonlinear systems of equations.

        Args:
            f (function):
                Function that defines the system of ordinary differential equations.
                The function should take one argument, x, and return a 1D numpy
                array with the same shape as x.
            x0 (array_like):
                Initial guess for the value of the solution at the left boundary.
            fprime (function, optional):
                Function that defines the Jacobian of f (default is None).
            tol (float, optional):
                Tolerance for the 'solver' to terminate.
                Calculations will terminate if the relative error between two consecutive iterates is less than or equal to 'tol'
            maxiter (int, optional):
                Maximum number of iterations for the root-finding algorithm (default is 100).
        """
        self.f = f
        self.x0 = x0
        self.fprime = fprime
        self.tol = tol
        self.maxiter = maxiter
        self.x = np.copy(self.x0)
        self.i = 0
        self.__name__ = "Newton"


    def step(self):
        if self.fprime is None:
            # Use finite difference approximation for Jacobian
            J = np.zeros((len(self.x), len(self.x)))
            # Compute Jacobian
            for j in range(len(self.x)):
                # Perturb x_j
                eps = np.zeros(len(self.x))
                # Use a small perturbation
                eps[j] = 1e-6
                # Compute finite difference approximation
                J[:,j] = (self.f(self.x+eps) - self.f(self.x-eps)) / (2*eps[j])
        else:
            # Use user-defined Jacobian
            J = self.fprime(self.x)
        # Compute Newton step
        b = -self.f(self.x)
        # Solve linear system
        delta = np.linalg.solve(J, b)
        # Update solution
        self.x += delta
        # Increment iteration counter
        self.i += 1
        # Check for convergence
        if np.linalg.norm(delta) < self.tol:
            return True
        else:
            return False


    def solve(self):
        # Iterate until convergence or maximum number of iterations is reached
        while self.i < self.maxiter:
            if self.step():
                break
        return self.x