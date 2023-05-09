import typing as T

import numpy as np
import numpy.typing as npt
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from pydrake.solvers import (  
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
)
from pydrake.symbolic import Polynomial, Variable, Variables

from pydrake.math import eq, le, ge

from util import timeit
from vertex import Vertex, BoxVertex, PolytopeVertex, EllipsoidVertex
from vertex import FREE, PSD, PSD_QUADRATIC, PSD_WITH_IDENTITY


class Edge:
    def __init__(self, v_left: Vertex, v_right: Vertex):
        assert v_left.n == v_right.n, "vertices must have same dimensions"
        self.left = v_left
        self.right = v_right
        self.n = self.left.n
        self.name = self.left.name + " -> " + self.right.name
        self.L = None
        self.linear_constraints = []

    def set_quadratic_cost(self):
        # [ 0   0   0  ]
        # [ 0   I   -I ]
        # [ 0   -I   I ]
        n = self.n
        L = np.zeros( (2*n+1, 2*n+1) )
        L[1:n+1, 1:n+1] = np.eye(n)
        L[n+1:2*n+1, n+1:2*n+1] = np.eye(n)
        L[ 1:n+1, n+1:2*n+1 ] = -np.eye(n)
        L[ n+1:2*n+1, 1:n+1 ] = -np.eye(n)
        self.set_cost(L)

    def set_cost(self, L:npt.NDArray):
        n = self.n
        assert L.shape == (2*n+1,2*n+1)
        self.L = L

    def get_cost(self):
        return self.L

    def add_linear_constraints(self, A_L:npt.NDArray, A_R:npt.NDArray):
        # A_l x_l + A_r x_r == 0
        # TODO: should support polynomials in the future;
        # should really be a matrix to be dotted with M_E
        m1,n1 = A_L.shape
        m2,n2 = A_R.shape
        assert m2 == m1
        assert self.n == n1 == n2
        for i in range(m1):
            self.linear_constraints.append( (A_L[i].reshape(n1,1), A_R[i].reshape(n2,1)) )

    def get_linear_constraint_multiplier_terms(self, prog: MathematicalProgram):
        m = len(self.linear_constraints)
        res = 0
        if m != 0:
            n = self.n
            # note: these variables do not need to be negative
            self.d_e = prog.NewContinuousVariables(m, "d_"+ self.name)
            for i in range(m):
                (Al, Ar) = self.linear_constraints[i]
                m_mat = np.vstack((
                    np.hstack((np.zeros((0,0)), Al.T, Ar.T)),
                    np.hstack((Al, np.zeros((n,n)), np.zeros((n,n)))),
                    np.hstack((Ar, np.zeros((n,n)), np.zeros((n,n))))
                ))
                res += m_mat * self.d_e[i]
        return res
    
    def get_left_right_set_multipliers(self, prog: MathematicalProgram):
        left_m_deg = self.left.multiplier_deg()
        right_m_deg = self.right.multiplier_deg()

        self.lambda_e_left = prog.NewContinuousVariables(left_m_deg, "l1_" + self.name)
        self.lambda_e_right = prog.NewContinuousVariables(right_m_deg, "l2" + self.name)

        prog.AddLinearConstraint( ge(self.lambda_e_left, np.zeros(left_m_deg) ) )
        prog.AddLinearConstraint( ge(self.lambda_e_right, np.zeros(right_m_deg) ) )

        res = 0
        res = self.left.make_multiplier_terms(self.lambda_e_left, left=True) + self.right.make_multiplier_terms(self.lambda_e_right, left=False)
        return res


    def get_potential_diff(self):
        n = self.n

        Ql = self.left.Q
        Qr = self.right.Q
        rl = self.left.r
        rr = self.right.r
        sl = self.left.s
        sr = self.right.s

        O_n = np.zeros((n,n))
        return np.vstack((
            np.hstack( (sr-sl,  -rl.T,   rr.T) ), 
            np.hstack( (-rl,    -Ql,    O_n ) ), 
            np.hstack( (rr,      O_n,    Qr ) ) 
            ))
    
    def s_procedure(self, prog:MathematicalProgram):
        res = 0
        res = self.get_cost() + self.get_potential_diff()
        res = res + self.get_left_right_set_multipliers(prog)
        res = res + self.get_linear_constraint_multiplier_terms(prog)
        prog.AddPositiveSemidefiniteConstraint(res)


n = 2
pot_type = PSD

prog = MathematicalProgram()
vertices = []

# vt = EllipsoidVertex("t", prog, np.zeros(n), 1*np.eye(n), pot_type)
# vv = EllipsoidVertex("v", prog, 2*np.ones(n), 1*np.eye(n), pot_type)
# vs = EllipsoidVertex("s", prog, 3*np.ones(n), 3*np.eye(n), pot_type)

vt = BoxVertex("t", prog, np.zeros(n), 3*np.ones(n), pot_type)
vv = BoxVertex("v", prog, 0*np.ones(n), 3*np.ones(n), pot_type)
vs = BoxVertex("s", prog, 0*np.ones(n), 3*np.ones(n), pot_type)

e = Edge( vs, vv )
e.set_quadratic_cost()
e.s_procedure(prog)

e2 = Edge( vv, vt )
e2.set_quadratic_cost()
e2.s_procedure(prog)

prog.AddLinearConstraint(vt.cost_at_point(np.zeros(n)) == 0)

cost = vs.cost_at_point( 2 * np.ones(n) )
prog.AddLinearCost( -cost )

solution = Solve(prog)
print(solution.is_success())
print(solution.get_optimal_cost())



