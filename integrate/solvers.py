from scipy.optimize import root, fsolve
from .methods import *
import numpy as np
import sys


METHODS = {'EULER': euler_step,
           'IEULER': improved_euler_step,
           'RK4': rk4_step,
           'RK45': rk45_step,
           'MIDPOINT': midpoint_step}


def solve_to(f, y0, t_span, dt=0.01, dt_max=0.1, method="RK45"):
    """
    Solve a differential equation numerically using a given numerical method from a starting time to an end time.

    Args:
        f (function): 
            Function that defines the differential equation.
        y0 (array): 
            Initial value of the solution.
        t_span (2-member sequence): 
            Interval of integration (t0, tf).
        dt (float): 
            Time step.
        dt_max (float): 
            Maximum time step.
        method (str, optional): 
            Numerical method to use (default is the Runge-Kutta method of order 5(4)).

    Returns:
        tuple: Tuple of arrays representing the time points and the approximate solutions at those time points.
    """
    if not callable(f):
        raise TypeError("The argument 'f' must be a function.")
    if not isinstance(t_span, (list, np.ndarray)):
        raise TypeError("The argument 't0' must be a number.")
    if len(t_span) != 2:
        raise ValueError("The argument 't_span' must be a 2-member sequence -> [t0, t1].")
    if t_span[0] < 0:
        raise ValueError("The first member of 't_span' must be positive.")
    if t_span[1] < 0:
        raise ValueError("The second member of 't_span' must be positive.")
    if not isinstance(y0, (list, np.ndarray)):
        raise TypeError("The argument 'x0' must be a numpy array.")
    if not isinstance(dt, (int, float)):
        raise TypeError("The argument 'dt' must be a number.")
    if dt <= 0:
        raise ValueError("The argument 'dt' must be positive.")
    if dt_max <= 0:
        raise ValueError("The argument 'dt_max' must be positive.")
    if method not in METHODS:
        raise TypeError(f"The method chosen: {method}, is not valid\nPlease choose from the following: {METHODS.keys()}")

    method = METHODS[method]
    
    t0, tf = t_span
    t = np.arange(t0, tf + dt, dt)
    Y = np.zeros((len(t), len(y0)))
    Y[0] = y0
    Y[1:] = [method(f, t[i-1], Y[i-1], min(t[i]-t[i-1], dt_max)) for i in range(1, len(t))][0]
    return dict(t=t, y=Y.T)


def ode_ivp(f, t_span, y0, method="RK45", t_eval=None, step_size=0.001, rtol=1e-3, atol=1e-6):
    """
    Solve a first order differential equation y' = f(t, y) using the specified numerical method.
    
    Args:
        f (function):
            The function defining the ODE y' = f(t, y).
        t_span (2-member sequence):
            The interval of integration (t0, tf).
        y0 (array_like):
            The initial value of the dependent variable.
        method (str or callable, optional):
            The numerical method to use. This can be one of the strings 'RK45' (default), 'RK4', 'IEULER', 'EULER', 'MIDPOINT', or a custom function implementing a numerical method. 
            The custom function should have the signature (f, t, y, dt) and return the new value of y at time t+dt.
        t_eval (array_like, optional):
            The times at which to output the solution. Defaults to None, in which case the solution is output at every internal step.
        step_size (float, optional):
            The size of the time step used in the numerical method. Defaults to 0.001.
        rtol (float, optional):
            The relative tolerance. Defaults to 1e-3.
        atol (float or array_like, optional):
            The absolute tolerance. Defaults to 1e-6.

    Returns:
        sol (dict):
            A dictionary containing the solution. The keys are 't' (the times at which the solution is output) and 'y' (the corresponding values of the dependent variable).
    """
            

    if not callable(f):
        raise TypeError("The argument 'f' must be a function.")
    if not isinstance(t_span, (list, np.ndarray)):
        raise TypeError("The argument 't_span' must be a 2-member sequence -> [t0, tf].")
    if len(t_span) != 2:
        raise ValueError("The argument 't_span' must be a 2-member sequence -> [t0, tf].")
    if t_span[0] < 0:
        raise ValueError("The first member of 't_span' must be positive.")
    if t_span[1] < 0:
        raise ValueError("The second member of 't_span' must be positive.")
    if not isinstance(y0, (list, np.ndarray)):
        raise TypeError("The argument 'x0' must be a numpy array.")
    if not isinstance(step_size, (int, float)):
        raise TypeError("The argument 'step_size' must be a number.")
    if step_size <= 0:
        raise ValueError("The argument 'step_size' must be positive.")
    if method not in METHODS:
        raise TypeError(f"The method chosen: {method}, is not valid\nPlease choose from the following: {METHODS.keys()}")
    elif callable(method):
        pass
    # Check if method is a string
    if isinstance(method, str):
        method = METHODS[method]

    # Initialize time grid and solution array
    t0, tf = t_span
    t = np.arange(t0, tf, step_size)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    # Counter for t_eval indexing
    idx = 1
    
    # Main loop for numerical integration
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        y[i] = method(f, t[i-1], y[i-1], dt)
        
        # Check if t_eval is not None and if it's time to output the solution
        if t_eval is not None:
            if idx < len(t_eval) and np.isclose(t[i], t_eval[idx], rtol=rtol, atol=atol):
                idx += 1
    
    # Replace t with t_eval if t_eval is not None
    if t_eval is not None:
        t = t_eval
    
    # Create dictionary containing the solution
    sol = {'t': t, 'y': y[:len(t)].T}
    
    return sol


def shooting_method(ode, x0, t_span, phase_cond=None, args=(), maxiter=100, bounds=[None, None], solver=fsolve, tol=1e-3):
    """
    Find a limit cycle of the system of ODEs defined by `ode`.

    Args:
        ode (function):
            Function that defines the system of ordinary differential equations.
            The function should take two arguments, t and x, and return a 1D numpy
            array with the same shape as x.
        x0 (array_like)
            Initial guess for the value of the solution at the left boundary.
        t_span (tuple):
            Tuple (t0, tf) defining the interval of integration.
        phase_cond (function):
            Function that defines the phase condition for the system of ordinary differential equations.
        args (tuple, optional):
            Additional arguments to pass to the `ode` function (default is an empty tuple).
        maxiter (int, optional):
            Maximum number of iterations for the root-finding algorithm (default is 100).
        bounds (tuple, optional):
            Tuple (lb, ub) defining lower and upper bounds on the initial guess `x0`
            (default is [None, None], implying there are no bounds).
        solver (function, optional):
            Function that defines the numerical method to use to minimize the error.
        tol (float, optional):
            Tolerance for the 'solver' to terminate.
            Calculations will terminate if the relative error between two consecutive iterates is less than or equal to 'tol'

    Returns:
        x0_sol (array_like or None):
            The solution of the boundary value problem, or None if the root-finding algorithm
            failed to converge.
    """
    def to_minimize(x0, ode, t_span):
        """
        Helper function to calculate the error in the boundary conditions.

        Args:
            x0 (array_like):
                Initial guess for the value of the solution at the left boundary.
            ode (function):
                Function that defines the system of ordinary differential equations.
                The function should take two arguments, t and x, and return a 1D numpy
                array with the same shape as x.
            t_span (tuple):
                Tuple (t0, tf) defining the interval of integration.
                
        Returns:
            error (float):
                The error in the boundary conditions.
        """
        x0_new = x0
        sol = ode_ivp(ode, t_span, x0)
        x0_new = sol["y"][:,0]
        x_final = sol["y"][:,-1]
        phase_val = phase_cond(x0_new, x_final) if phase_cond != None else (x_final - x0_new)
        error = phase_val
        return error


    def get_bounds(x0, ode, t_span, bounds):
        """
        Helper function to enforce bounds on the initial guess.

        Args:
            x0 (array_like):
                Initial guess for the value of the solution at the left boundary.
            ode (function):
                Function that defines the system of ordinary differential equations.
                The function should take two arguments, t and x, and return a 1D numpy
                array with the same shape as x.
            t_span (tuple):
                Tuple (t0, tf) defining the interval of integration.
            bounds (tuple):
                Tuple (lb, ub) defining lower and upper bounds on the initial guess `x0`.

        Returns:
            error (float):
                The error in the boundary conditions.
        """
        lb, ub = bounds
        try:
            try:
                x0[x0 <= lb] = lb + 1e-6
            except:
                pass
            try:
                x0[x0 >= ub] = ub - 1e-6
            except:
                pass
        except:
            pass
        return to_minimize(x0, ode, t_span)
    
    try:
        test_sol = ode(0, x0)
    except Exception as E:
        exc_type, _, _ = sys.exc_info()
        print(str(exc_type.__name__) + ": " + str(E) + f"\nMake sure your initial values ({x0}) have the same dimensions as those expected by your ode ({ode.__name__})")
        return exc_type

    x0_sol, info, ier, msg = solver(get_bounds, x0, args=(ode, t_span, bounds, ), full_output=True, maxfev=maxiter, xtol=tol)
    if ier == 1:
        fcalls = info["nfev"]
        residual = np.linalg.norm(info["fvec"])
        print(f"Root finder found the solution x0_sol={x0_sol} after {fcalls} function calls; the norm of the final residual is {residual}")
        return x0_sol
    else:
        print(f"Root finder failed with error message: {msg}")
        return np.array([])

    
def continuation(ode, x0, t_span, par0, const_limits, vary_par=0, max_steps=100, discretisation=shooting_method, solver=fsolve):
    """
    Perform natural parameter continuation.

    Args:
        ode (function):
            Function that defines the system of ordinary differential equations.
            The function should take two arguments, t and x, and return a 1D numpy
            array with the same shape as x.
        x0 (array_like)
            Initial guess for the value of the solution at the left boundary.
        t_span (tuple):
            Tuple (t0, tf) defining the interval of integration.
        par0 (array_like):
            Initial guess for the value of the parameters.
        const_limits (array_like):
            Array of length `par0` defining the limits of the parameter continuation.
        vary_par (int, optional):
            Index of the parameter to vary (default is 0).
        max_steps (int, optional):
            Maximum number of steps to take in the continuation (default is 100).
        discretisation (function, optional):
            Function that defines the numerical method to use to discretise the ODEs.
        solver (function, optional):
            Function that defines the numerical method to use to minimise the error.

    Returns:
        parameters (list):
            List of the parameters at each step of the continuation.
        solutions (list):
            List of the solutions at each step of the continuation.
    """
    step_size = (const_limits[vary_par] - par0[vary_par]) / max_steps
    parameters = [par0]
    solutions = [x0]

    for i in range(max_steps):
        par = parameters[-1]
        par[vary_par] += step_size
        x0 = discretisation(lambda t, X: ode(t, X, par0), x0, t_span, solver=solver)
        parameters.append(par)
        solutions.append(x0)

    return parameters, solutions


def newton(f, x0, fprime=None, tol=1e-6, maxiter=100):
    """
    Newton's method for solving f(x) = 0.

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

    Returns:
        x (array_like):
            The solution of the boundary value problem.
    """
    x = np.copy(x0)
    for i in range(maxiter):
        if fprime is None:
            J = np.zeros((len(x), len(x)))
            for j in range(len(x)):
                eps = np.zeros(len(x))
                eps[j] = 1e-6
                J[:,j] = (f(x+eps) - f(x-eps)) / (2*eps[j])
        else:
            J = fprime(x)
        b = -f(x)
        delta = np.linalg.solve(J, b)
        x += delta
        if np.linalg.norm(delta) < tol:
            break
    return x

def solve_ode(D, q_func, a, b, N, u_a, u_b, bc_type, const, solver='newton', max_iter=1000, tol=1e-6):
    """
    Solve the 1D diffusion equation with Dirichlet, Neumann, or Robin boundary conditions.
    
    Args:
        D (float):
            Diffusion coefficient.
        q_func (function):
            Function that defines the Robin boundary condition.
        a (float):
            Left boundary of the domain.
        b (float):
            Right boundary of the domain.
        N (int):
            Number of grid points.
        u_a (float):
            Dirichlet or Neumann boundary condition at x = a.
        u_b (float):
            Dirichlet or Neumann boundary condition at x = b.
        bc_type (str):
            Type of boundary conditions (dirichlet, neumann, or robin).
        const (float):
            Constant in the Robin boundary condition.
        solver (str, optional):
            Solver to use (default is the Newton-Raphson method).
        max_iter (int, optional):
            Maximum number of iterations (default is 1000).
        tol (float, optional):
            Tolerance for the Newton-Raphson method (default is 1e-6).

    Returns:
        x (array_like):
            Grid points.
        u (array_like):
            Solution of the boundary value problem.
    """

    # Define the grid
    x = np.linspace(a, b, N+1)
    h = x[1] - x[0]
    
    # Define the coefficient matrix A and the right-hand side vector b
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    
    # Set the boundary conditions
    if bc_type == 'dirichlet':
        A[0, 0] = 1
        b[0] = u_a
        A[N, N] = 1
        b[N] = u_b
    elif bc_type == 'neumann':
        A[0, 0] = -1/h
        A[0, 1] = 1/h
        b[0] = u_a
        A[N, N-1] = -1/h
        A[N, N] = 1/h
        b[N] = u_b
    elif bc_type == 'robin':
        A[0, 0] = 1 + h/(2*D)*u_a
        A[0, 1] = -1/h
        b[0] = -h/(2*D)*q_func(x[0], u_a, const) + u_a
        A[N, N-1] = 1/h
        A[N, N] = 1 + h/(2*D)*u_b
        b[N] = -h/(2*D)*q_func(x[N], u_b, const) + u_b
    
    # Set the interior coefficients
    for i in range(1, N):
        A[i, i-1] = 1/h**2 - q_func(x[i], 0, const)/(2*D*h)
        A[i, i] = -2/h**2 + q_func(x[i], 1, const)/D
        A[i, i+1] = 1/h**2 + q_func(x[i], 0, const)/(2*D*h)
        b[i] = 0
    
    # Solve the system
    if solver == 'root':
        def f(u):
            b_new = np.copy(b)
            b_new[1:-1] = -q_func(x[1:-1], u[1:-1], const)
            return np.dot(A, u) - b_new
        
        sol = root(f, np.zeros(N+1), method='hybr', options={'maxfev': max_iter, 'xtol': tol}).x
    elif solver == 'numpy':
        sol = np.linalg.solve(A, b)
    elif solver == 'newton':
        def f(u):
            b_new = np.copy(b)
            b_new[1:-1] = -q_func(x[1:-1], u[1:-1], const)
            
            return np.dot(A, u) - b_new
        
        def fprime(u):
            return A
        
        sol = newton(f, np.zeros(N+1), fprime=fprime, tol=tol, maxiter=max_iter)
    else:
        raise ValueError("Invalid solver type. Choose 'root', 'numpy', or 'newton'.")
    
    return x, sol