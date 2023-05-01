from .solution import Solution
from .problem import Problem
from .ivp import IVP
from .bvp import BVP


class ProblemSolver():
    def __init__(self, **kwargs):
        # check none of kwargs are "Problem" type
        if any([isinstance(kwargs[k], Problem) for k in kwargs]):
            self.problem = kwargs["problem"]
        else:
            self.problem = Problem(**kwargs)
        self.ivp = IVP(self.problem)
        self.bvp = BVP(self.problem)

    def solve(self, 
              method="RK45", 
              pde_method="CRANKNICOLSON", 
              root_finder="ROOT", 
              discretize_method="SHOOT", 
              bc_type="DIRICHLET", 
              matrix_type="SPARSE", 
              **kwargs):
        if self.problem.problem_type[1] == "IVP" and not self._check_cont(**kwargs):
            return self.ivp.solve(method=method, 
                                  pde_method=pde_method, 
                                  root_finder=root_finder, 
                                  matrix_type=matrix_type)
        elif self.problem.problem_type[1] == "BVP" or self._check_cont(**kwargs):
            return self.bvp.solve(method=method,
                                  pde_method=pde_method,
                                  root_finder=root_finder,
                                  discretize_method=discretize_method,
                                  bc_type=bc_type,
                                  matrix_type=matrix_type,
                                  **kwargs)
        else:
            raise Exception("Invalid problem type")
        
    def _check_cont(self, **kwargs):
        cont_kwargs = ["p0", "q_cont", "p_span", "vary_par", "ds", "Ns", "da", "output"]
        check_list = [k in cont_kwargs for k in kwargs]
        if sum(check_list) != 0:
            return True
        else:
            return False
