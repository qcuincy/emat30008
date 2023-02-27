# %%
import sys, os
if not hasattr(sys.modules[__name__], '__file__'):
    emat_dir = os.path.dirname(os.path.abspath())
else:
    emat_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".."
        )
    )
sys.path.insert(0, emat_dir)

from integrate.solvers import *
from integrate.phase import *
import matplotlib.pyplot as plt

def hopf_bifurcation(t, U, consts):
    beta, sigma = consts
    u1, u2 = U

    du1_dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2_dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)

    return np.array([du1_dt, du2_dt])

def phase_cond(x_initial, x_final):
    return x_final - x_initial

def exact_sol(t, phase, consts):
    beta, sigma = consts
    u1 = np.sqrt(beta)*np.cos(t+phase)
    u2 = np.sqrt(beta)*np.sin(t+phase)
    return [u1, u2]

def shooting_test(ode, exact_sol, x0_sol, t_span):
    sol = ode_ivp(ode, t_span, x0)

    t = sol["t"]
    y = sol["y"]

    phase, chosen_point, limit_cycle = find_period(y)

    exact = exact_sol(t, phase)

    if np.allclose(exact, y):
        print("The solution obtained from the shooting method and the exact solution are both close in value!")
    else:
        print("The solution obtained from the shooting method and the exact solution are not close in value!\nSomething went wrong!")
    return y, exact, chosen_point, limit_cycle

x0 = [1,0]
t_span = [0, 50]
consts = [1, -1]

limit_cycle= shooting_method(lambda t, X: hopf_bifurcation(t, X, consts), x0, t_span, phase_cond=phase_cond)
sol = ode_ivp(lambda t, X: hopf_bifurcation(t, X, consts), t_span, limit_cycle)

t = sol["t"]
y = sol["y"]

y, exact, chosen_point, limit_cycle = shooting_test(lambda t, X: hopf_bifurcation(t, X, consts), lambda t, phase: exact_sol(t, phase, consts), limit_cycle, t_span)

