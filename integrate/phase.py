import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import root, fsolve

def find_limit_cycle(ode, initialu, tol=1e-6):
    def residual(x0):
        sol = odeint(ode, x0, np.linspace(0, 20, 200))
        return sol[-1, :] - x0

    sol = root(residual, initialu, tol=tol)
    if sol.success:
        return sol.x
    else:
        return None
    

def objective(f, X0, t_span, t_eval):
    sol = solve_ivp(f, t_span, \
            X0, t_eval = t_eval)
    y = sol.y
    return [y[0][-1] - X0[0], y[1][-1] - X0[1]]

def shooting_method(ode, initialx, t_span, xtol=1e-8, parameters=[]):
    if not isinstance(parameters, list):
        parameters = list(parameters)
    t0, tf = t_span
    t_eval = np.linspace(t0, tf, 100)
    sol = fsolve(lambda X0, parameters: objective(ode, X0, [t0, tf], t_eval), initialx, xtol=xtol, args=(parameters,))
    return sol