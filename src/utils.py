from .ivp_methods import *
from .pde_utils.methods import *
import numpy as np
from scipy.optimize import root

IVPMETHODS = {
    "EULER": Euler,
    "RK4": RK4,
    "RK45": RK45,
    "MIDPOINT": Midpoint,
    "IEULER": IEuler
}

PDEMETHODS = {
    "FINITEDIFFERENCE": FiniteDifference,
    "EXPLICITEULER": ExplicitEuler,
    "IMPLICITEULER": ImplicitEuler,
    "CRANKNICOLSON": CrankNicolson,
    "IMEX": IMEX,
    "METHODOFLINES": MethodOfLines
}

ROOTFINDERS = {
    "ROOT": root,
    "NEWTON": Newton,
    "SOLVE": np.linalg.solve
}

MATRIXTYPES = ["SPARSE", "DENSE"]