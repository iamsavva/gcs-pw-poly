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

from util import timeit, INFO, ERROR, WARN, YAY
from vertex import Vertex, BoxVertex, PolytopeVertex, EllipsoidVertex
from vertex import FREE, PSD, PSD_ON_STATE, PSD_WITH_IDENTITY

from pydrake.geometry.optimization import Point, HPolyhedron


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
        L = np.zeros((2 * n + 1, 2 * n + 1))
        L[1 : n + 1, 1 : n + 1] = np.eye(n)
        L[n + 1 : 2 * n + 1, n + 1 : 2 * n + 1] = np.eye(n)
        L[1 : n + 1, n + 1 : 2 * n + 1] = -np.eye(n)
        L[n + 1 : 2 * n + 1, 1 : n + 1] = -np.eye(n)
        self.set_cost(L)

    def set_cost(self, L: npt.NDArray):
        n = self.n
        assert L.shape == (2 * n + 1, 2 * n + 1)
        self.L = L

    def get_cost(self):
        return self.L

    def add_linear_constraints(self, A_L: npt.NDArray, A_R: npt.NDArray):
        # A_l x_l + A_r x_r == 0
        # TODO: should support polynomials in the future;
        # should really be a matrix to be dotted with M_E
        m1, n1 = A_L.shape
        m2, n2 = A_R.shape
        assert m2 == m1
        assert self.n == n1 == n2
        for i in range(m1):
            self.linear_constraints.append(
                (A_L[i].reshape(n1, 1), A_R[i].reshape(n2, 1))
            )

    def get_linear_constraint_multiplier_terms(self, prog: MathematicalProgram):
        m = len(self.linear_constraints)
        res = 0
        if m != 0:
            n = self.n
            # note: these variables do not need to be negative
            self.d_e = prog.NewContinuousVariables(m, "d_" + self.name)
            for i in range(m):
                (Al, Ar) = self.linear_constraints[i]
                m_mat = np.vstack(
                    (
                        np.hstack((np.zeros((1, 1)), Al.T, Ar.T)),
                        np.hstack((Al, np.zeros((n, n)), np.zeros((n, n)))),
                        np.hstack((Ar, np.zeros((n, n)), np.zeros((n, n)))),
                    )
                )
                res += m_mat * self.d_e[i]
        return res

    def add_linear_constraints_equality_constraint(
        self, A, B, prog: MathematicalProgram
    ):
        n = self.left.state_dim
        m = self.left.control_dim
        d = 2 * n + 2 * m + 1

        T = np.hstack((np.zeros((n, 1)), A, B, -np.eye(n), np.zeros((n, m))))
        M = T.T @ T

        multiplier = prog.NewContinuousVariables(n, "m_" + self.name)
        M = T.T @ np.diag(multiplier) @ T
        return M

    def get_left_right_set_multiplier_terms(self, prog: MathematicalProgram):
        res = 0

        left_m_deg = self.left.multiplier_deg()
        right_m_deg = self.right.multiplier_deg()

        if left_m_deg > 0:
            self.lambda_e_left = prog.NewContinuousVariables(
                left_m_deg, "l1_" + self.name
            )
            prog.AddLinearConstraint(ge(self.lambda_e_left, np.zeros(left_m_deg)))
            res = res + self.left.make_multiplier_terms(self.lambda_e_left, left=True)

        if right_m_deg > 0:
            self.lambda_e_right = prog.NewContinuousVariables(
                right_m_deg, "l2" + self.name
            )
            prog.AddLinearConstraint(ge(self.lambda_e_right, np.zeros(right_m_deg)))
            res = res + self.right.make_multiplier_terms(
                self.lambda_e_right, left=False
            )
        return res

    def get_potential_difference(self):
        n = self.n
        Ql, Qr = self.left.Q, self.right.Q
        rl, rr = self.left.r, self.right.r
        sl, sr = self.left.s, self.right.s
        O_n = np.zeros((n, n))
        return np.vstack(
            (
                np.hstack((sr - sl, -rl.T, rr.T)),
                np.hstack((-rl, -Ql, O_n)),
                np.hstack((rr, O_n, Qr)),
            )
        )

    def s_procedure(self, prog: MathematicalProgram, A=None, B=None):
        res = 0

        res = self.get_cost() + self.get_potential_difference()

        res = res + self.get_left_right_set_multiplier_terms(prog)

        # TODO: this is not general enough, fix me
        res = res + self.get_linear_constraint_multiplier_terms(prog)
        if A is not None and B is not None:
            res = res + self.add_linear_constraints_equality_constraint(A, B, prog)

        prog.AddPositiveSemidefiniteConstraint(res)

