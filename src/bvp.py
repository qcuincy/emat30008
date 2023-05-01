from scipy.optimize import newton_krylov
from IPython.display import clear_output
from .bvp_methods.shoot import Shoot
from .pde_utils.grid import Grid
from .solution import Solution
from .problem import Problem
from .solver import Solver
from .ivp import IVP
import numpy as np
import os
from scipy.integrate import solve_ivp

BVPMETHODS = {"SHOOT": Shoot}

class BVP(Solver):
    """
    BVP class for solving boundary value problems

    Classes:
        BVP(Solver)
            Class for solving boundary value problems

    Methods:
        solve(bc=None, phase_cond=None, method="RK45", maxiter=100, tol=1e-3):
            Solve the boundary value problem using shooting methods.

    Usage:
        ODE:
        >>> from numerical_methods.bvp import BVP
        >>> from numerical_methods.problem import Problem
        
        >>> def f(x, u):
        ...     return [u[1], -u[0]]

        >>> def bc(u_a, u_b):
        ...     return [u_a[0], u_b[0] - 1]

        >>> problem = Problem(f=f, bc=bc, y0=[1, 0], yf=[0, 1], t0=0, tf=10, Nt=100)
        >>> bvp = BVP(problem)
        >>> sol = bvp.solve()
        >>> sol.plot()

        PDE:
        >>> from numerical_methods.bvp import BVP
        >>> from numerical_methods.problem import Problem

        >>> def q(t, x, u):
        ...     return u

        >>> def bc(u_a, u_b):
        ...     return [u_a[0], u_b[0] - 1]

        >>> problem = Problem(q=q, bc=bc, y0=[1, 0], yf=[0, 1], t0=0, tf=10, Nt=100, a=0, b=1, Nx=100)
        >>> bvp = BVP(problem)
        >>> sol = bvp.solve()
        >>> sol.plot()
    """
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

        self.BVPMETHODS = BVPMETHODS
        self.discretize_method = self.BVPMETHODS["SHOOT"]


    def solve(self, method="RK45", pde_method="FINITEDIFFERENCE", root_finder="ROOT", discretize_method="SHOOT", bc_type="DIRICHLET", matrix_type="SPARSE", **kwargs):
        """
        Solve the boundary value problem using shooting methods.

        Args:
            method (str): Method for solving the IVP. Default is "RK45".
            pde_method (str): Method for solving the PDE. Default is "FINITEDIFFERENCE".
            root_finder (str): Method for finding the root. Default is "ROOT".
            discretize_method (str): Method for discretizing the problem. Default is "SHOOT".
            bc_type (str): Type of boundary condition. Default is "DIRICHLET".
            matrix_type (str): Type of matrix. Default is "SPARSE".
            **kwargs (dict): Keyword arguments for continuation method
                Supported kwargs:
                    p0 (array, list, tuple): Initial values of parameters p
                    q_cont (function): Function of t, u and *args. Default is the original function.
                    vary_par (int): Index of parameter to vary. Default is 0.
                    p_span (array, list, tuple): Parameter span. Default is None.
                    ds (float): Parameter step size. Default is 0.01.
                    Ns (int): Number of steps. Default is 100.
                    cont_type (str): Type of continuation method. Default is "PSUEDO".
                    da (float): Desired arclength step. Default is 0.01.
                    output (bool): Whether to output the root finding results message


        Returns:
            sol (Solution): Solution object.
        """
        # implementation of BVP solver method for either ODE or PDE
        self._method_handler(method, pde_method, root_finder, discretize_method, matrix_type)
        # check if kwargs are given for continuation method
        if not kwargs:
            if self.is_ode:
                return self._solve_ode(method)

            elif not self.is_ode:
                return self._solve_pde(bc_type, matrix_type)

            else:
                raise ValueError("Invalid combination of parameters for BVP solver.")
        else:
            return self._solve_continuation(**kwargs)

    def _solve_continuation(self, **kwargs):
        p0, q_cont, vary_par, p_span, ds, Ns, max_ds, min_ds, cont_type, da, output = self._continuation_handler(**kwargs)
        if self.is_ode:
            if cont_type.upper() == "PSEUDO":
                return self._solve_ode_continuation_pseudo(p0, q_cont, p_span, vary_par, ds, Ns, max_ds, min_ds, da, output)
            else:
                return self._solve_ode_continuation_natural(p0, q_cont, p_span, vary_par, ds, Ns, output)
        else:
            raise NotImplementedError("Continuation method is only implemented for ODE")


    def _solve_ode_continuation_pseudo(self, p0, q_cont=None, p_span=None, vary_par=0, ds=0.01, Ns=100, max_ds=0.1, min_ds=0.001, da=0.01, output=False):
        """
        Perform pseudo-arclength continuation on the system of equations F(x, p) = 0.

        Args:
            p0 (array, list, tuple): Initial values of parameters p
            q_cont (function): Function of t, u and *args. Default is the original function.
            vary_par (int): Index of parameter to vary. Default is 0.
            p_span (array, list, tuple): Parameter span. Default is None.
            ds (float): Parameter step size. Default is 0.01.
            Ns (int): Number of steps. Default is 100.
            max_ds (float): Maximum parameter step size. Default is 0.1.
            min_ds (float): Minimum parameter step size. Default is 0.001.
            da (float): Desired arclength step. Default is 0.01.

        Returns:

        """
        def G(u, y_prev, p_prev, ps_prev, da, t, q_cont):
            y, p = u[:-1], u[-1]
            diff = y - y_prev
            ds = np.sqrt(abs(da**2 - np.dot(u[:-1] - y_prev, u[:-1] - y_prev)))

            ps_prev[vary_par] = p
            F_value = q_cont(t, y, *ps_prev)
            tangent = diff + ds * (p - p_prev)
            return np.append(F_value, np.dot(tangent, tangent) - da**2)
        
        if p_span is not None:
            p_prev = p_span[0]
        else:
            if isinstance(p0, (list, tuple, np.ndarray)):
                p_prev = p0[vary_par]
            else:
                p_prev = p0

        y_prev, ps_prev = self.y0, p0.copy()
        u_prev = np.append(y_prev, p_prev)

        results = [u_prev]
        sols = [IVP(Problem(f=lambda t, y: q_cont(t, y, *ps_prev), y0=y_prev, t0=self.t0, tf=self.tf, Nt=self.Nt)).solve().y]
        counter = 0
        while True:
            # Attempt to find root
            u_guess = u_prev + np.append(ds * np.ones_like(self.y0), ds)
            try:
                u = newton_krylov(lambda u: G(u, y_prev, p_prev, ps_prev, da, sol.t, q_cont), u_guess)
            except:
                u = u_guess
            
            # Solve the ivp
            sol = IVP(Problem(f=lambda t, y: q_cont(t, y, *ps_prev), y0=y_prev, t0=self.t0, tf=self.tf, Nt=self.Nt)).solve()
            time = sol.t

            # # Calculate arclength
            # ds = np.sqrt(abs(da**2 - np.dot(u[:-1] - y_prev, u[:-1] - y_prev)))


            # Update the step size if it is too large
            # if ds > max_ds:
            #     ds *= 0.99

            # # Update the step size if it is too small
            # if ds < min_ds:
            #     ds *= 1.01

            # Clear output
            if not output:
                clear_output()

            # If p_span is not None, check if p_prev is outside of p_span
            if p_span is not None:
                if u[-1] > p_span[1]:
                    break
                elif u[-1] < p_span[0]:
                    break
            else:
                if counter > Ns:
                    break

            # Append result and solution
            results.append(u)
            sols.append(sol.y)

            # Update previous values
            y_prev, p_prev = u[:-1], u[-1]
            u_prev = u
            ps_prev[vary_par] = p_prev

            # Update counter
            counter += 1
            print("Iteration: ", counter, "p = ", p_prev, "ds = ", ds)

        results = np.array(results)
        y0s = results[:, :-1]
        params = results[:, -1]

        return Solution(t=np.array(time), params=np.array(params), y=np.array(sols)), np.array(y0s)


    def _solve_ode_continuation_natural(self, p0, q_cont=None, p_span=None, vary_par=0, ds=0.01, Ns=100, output=False):
        """
        Continuation method for ODE BVPs

        Args:
            p0 (array, list, tuple): Initial values of parameters p
            q_cont (function, optional): Function of x, u and *args Defaults to None.
            vary_par (int, optional): Index of parameter to vary. Defaults to 0.
            p_span (array, list, tuple, optional): Parameter span. Defaults to None.
            ds (float, optional): Parameter step size. Defaults to 0.01.
            Ns (int, optional): Number of steps. Defaults to 100.
            output (bool): Whether to output the root finding results message

        Returns:
            Solution: Solution object containing the solutions and the parameters.
            Array: Array of initial conditions.
        """
        ps = p0.copy()

        p = p0[vary_par]
        sol, y0 = self._solve_ode(self.method.__name__)
        sols = [sol.y]
        y0s = [y0]
        params = [p_span[0]]
        p_range = np.arange(p_span[0], p_span[1], ds)
        if self.is_ode:
            for p in p_range:
                # Update parameter
                p += ds
                ps[vary_par] = p

                # Update function
                f = lambda t, y: q_cont(t, y, *ps)
                self.f = f

                # solve the ODE
                sol, y0 = self._solve_ode(self.method.__name__)
                time = sol.t
                
                # Append solution, y0, parameter and step size
                sols.append(sol.y)
                y0s.append(y0)
                params.append(p)

                # Update the initial condition
                self.y0 = y0

                # Clear output
                if not output:
                    clear_output()

                # Check if parameter span is exceeded
                if np.allclose(params[-1], p_span[0]) or np.allclose(params[-1], p_span[1]):
                    print("Parameter span exceeded.")
                    break
        else:
            raise NotImplementedError("Continuation method is only implemented for ODEs.")

        return Solution(t=np.array(time), params=np.array(params), y=np.array(sols)), np.array(y0s)


    def _solve_ode(self, method):
        """
        Solve the boundary value problem using shooting methods.

        Args:
            method (str): Method for solving the IVP. Default is "RK45".

        Returns:
            sol (Solution): Solution object.
            y0 (array): Initial condition.
        """
        # implementation of BVP solver method for ODE
        discretize_method = self.discretize_method(self.f, self.y0, self.t0, self.tf, self.dt, self.ic, self.bc)
        sol, y0 = discretize_method.solve(method=method)

        return sol, y0


    def _solve_pde(self, bc_type, matrix_type):
        """
        Solve the boundary value problem using shooting methods.

        Args:
            bc_type (str): Type of boundary condition. Default is "DIRICHLET".
            matrix_type (str): Type of matrix. Default is "SPARSE".

        Returns:
            sol (Solution): Solution object.
        """
        # implementation of BVP solver method for PDE
        pde_method = self.pde_method(Grid(self.a, self.b, self.t0, self.tf, C=self.C, Nt=self.Nt, Nx=self.Nx))
        sol = pde_method.solve(self.q, ic=self.ic, bc=self.bc, method=self.method, root_finder=self.root_finder , bc_type=bc_type, matrix_type=matrix_type)
        return sol


    def _method_handler(self, method, pde_method, root_finder, discretize_method, matrix_type):
        """
        Method handler for BVP solver.

        Args:
            method (str): Method for solving the IVP. Default is "RK45".
            pde_method (str): Method for solving the PDE. Default is "FINITEDIFFERENCE".
            root_finder (str): Method for finding the root. Default is "ROOT".
            discretize_method (str): Method for discretizing the problem. Default is "SHOOT".
        """
        
        super()._method_handler(method, pde_method, root_finder, matrix_type)

        discretize_method = discretize_method.upper()

        # Check if discretize method is valid
        if discretize_method not in self.BVPMETHODS:
            raise TypeError(f"The discretize method chosen: {discretize_method}, is not valid\nPlease choose from the following: {self.BVPMETHODS.keys()}")
        if isinstance(discretize_method, str):
            discretize_method = self.BVPMETHODS[discretize_method]
        else:
            raise TypeError(f"Discretize method must be a string\nPlease choose from the following: {self.BVPMETHODS.keys()}")

        self.discretize_method = discretize_method

    
