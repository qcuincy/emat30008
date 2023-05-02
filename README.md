# Numerical Methods [emat30008]

This repository contains a collection of numerical methods for solving problems in the fields of differential equations and partial differential equations. The methods are implemented in Python and organized into several modules.

## Getting Started

### Prerequisites

To use the numerical methods in this repository, you'll need to have Python 3 installed. Additionally, you'll need to install the required packages listed in `requirements.txt`.

To install the packages, run:

```
pip install -r requirements.txt
```

### Installation

To install the numerical methods package, run:

```
pip install .
```

## Usage

The `ProblemSolver` class provides an easy and flexible way to solve differential equations. Here is an example usage:

```python
from src import ProblemSolver as ps
from src import examples as ex
import numpy as np

# Load example problem
example = ex.Van_der_Pol()
params, ode, _ = example()

# Initial conditions
y0 = np.array([2, 0])

# Time points
t0, tf = 0, 5
Nt = 500

# Parameters
mu = params

# Solve the problem
solver = ps(f = ode, y0 = y0, t0 = t0, tf = tf, Nt=Nt, args=(mu,))
solution = solver.solve(method="RK45")

# Plot the phase plot
solution.plot(phase_plot=True, width=1000, height=400, margin=dict(l=50, r=50, b=50, t=50, pad=0))
```

In this example, we load the Van der Pol example problem, set the initial conditions and time points, and solve the problem using the `RK45` method. Finally, we plot the phase plot of the solution. Resulting in the following plot:

<img src="https://i.postimg.cc/Hxnvsg51/newplot.png" alt="drawing" width="800"/>

## Modules

### `src.bvp_methods`

This module contains methods for solving boundary value problems. The current implementation includes the `shoot` method.

### `src.ivp_methods`

This module contains methods for solving initial value problems. The current implementation includes the `euler`, `ieuler`, `midpoint`, `odestep`, `rk4`, and `rk45` methods.

### `src.pde_utils`

This module contains utilities for solving partial differential equations. The current implementation includes several methods, such as `cranknicolson`, `expliciteuler`, `finitedifference`, `imex`, `impliciteuler`, and `methdoflines`. The module also contains a `grid` utility for generating grids and a `root_finders` module with a `newton` method.

## Files

### `coursework.ipynb`

This Jupyter notebook contains examples of using the numerical methods.

### `README.md`

This file.

### `requirements.txt`

This file lists the required packages.

### `setup.py`

This file is used for installing the package.

## Tests

The `test` directory contains unit tests for the numerical methods. To run the tests, run:

```
pytest
```
