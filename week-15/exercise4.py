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
from integrate.phase import *
from integrate.OdeSol import *
from integrate.solvers import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

def phase_cond(x_initial, x_final):
    return x_final - x_initial

def predator_prey(t, X, consts=[1, 0.1, 0.1]):
    a, d, b = consts
    x, y = X
    return np.array([x*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])

x0 = np.array([10, 20])
t_span = [0, 20]
consts = [1, 0.1, 0.25]

x_sol = shooting_method(lambda t, X: predator_prey(t, X, consts), x0=x0, t_span=t_span, phase_cond=phase_cond, bounds=[None, None], tol=1e-6)


sol = ode_ivp(f=lambda t, X: predator_prey(t, X, consts), y0=x_sol, t_span=t_span)

t, X = sol["t"], sol["y"]
plt.plot(t, X[0,:])
plt.plot(t, X[1,:])
plt.show()