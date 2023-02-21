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
from integrate.phase import *
import matplotlib.pyplot as plt
import numpy as np

def predator_prey(t, X, consts=[1, 0.1, 0.1]):
    a, d, b = consts
    x, y = X
    return np.array([x*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])

# Set up initial conditions and integration parameters
initial_condition = [1, 2]
t_span = [0, 100]
# Integrate the ODE to obtain a trajectory
# solve_to()
sol = ode_ivp(f=lambda t, X: predator_prey(t, X, (1, 0.1, 0.26)), t_span=t_span, y0=initial_condition, step_size=0.01, atol=1e-6, rtol=1e-3)

Y = sol["y"]
t = sol["t"]
# Plot the trajectory in phase space
plt.plot(t, Y[0,:], label='Prey')
plt.plot(t, Y[1,:], label='Predator')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# # %%
# import os

# from integrate.phase import *
# # os.path.abspath(os.path.join(script_dir, ".."))
# print(script_dir)