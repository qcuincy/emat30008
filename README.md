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

### Usage

To use the numerical methods, import the relevant module from `src`. For example, to use the `shoot` method for solving boundary value problems, import `shoot` from `src.bvp_methods`.

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
