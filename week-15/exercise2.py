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
from integrate.OdeSol import *
from integrate.solvers import *
from integrate.phase import *
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np


def predator_prey(t, X, consts=[1, 0.1, 0.1]):
    a, d, b = consts
    x, y = X
    return np.array([x*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])

# Set up initial conditions and integration parameters

initial_condition = [0.27015618, 0.27015619]
t_span = [0, 50]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Integrate the ODE to obtain a trajectory
# solve_to()
s = ode_ivp(f=lambda t, X: predator_prey(t, X, (1, 0.1, 0.26)), t_span=t_span, y0=initial_condition, step_size=0.01, atol=1e-6, rtol=1e-3)


t = s["t"]
y = s["y"]

randidx = np.random.randint(0, len(y[:,0]))
chosen_point = [y[randidx, 0], y[randidx, 1]]  # arbitrary choice of point in the middle of the trajectory
limit_cycle = y[:,(y[0]>0) & (y[1] > 0)]

distances = np.linalg.norm(limit_cycle.T - chosen_point, axis=1)
min_distance_index = np.argmin(distances)
phase = 2 * np.pi * min_distance_index / len(limit_cycle)

# Plot the limit cycle and the chosen point
plt.plot(limit_cycle[:,0], limit_cycle[:,1], label='Limit cycle')
plt.plot(chosen_point[0], chosen_point[1], 'o', label='Chosen point')
plt.legend()

print(f"Phase of chosen point: {phase:.2f} radians")

plt.show()