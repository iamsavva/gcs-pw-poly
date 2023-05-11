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


class LinearDynamicsEdge:
    def __init__(self, v_left: Vertex, v_right: Vertex):
        assert v_left.n == v_right.n, "vertices must have same dimensions"
        self.left = v_left
        self.right = v_right
        self.n = self.left.n
        self.state_dim = self.left.state_dim
        self.control_dim = self.left.control_dim
        self.name = self.left.name + " -> " + self.right.name

    def make_lqr_cost_matrix(self, Q, R):
        state_dim = self.state_dim
        control_dim = self.control_dim
        assert Q.shape == (self.state_dim, self.state_dim)
        assert R.shape == (self.control_dim, self.control_dim)
        assert np.allclose(self.left.x_star, self.right.x_star)

        y = self.left.x_star

        return np.vstack(
            (
                np.hstack((y.T @ Q @ y, -y.T @ Q, np.zeros((1, control_dim)))),
                np.hstack((-Q @ y, Q, np.zeros((state_dim, control_dim)))),
                np.hstack((np.zeros((control_dim, 1 + state_dim)), R)),
            )
        )

        # return np.vstack((
        #     np.zeros((1, state_dim+control_dim+1)),
        #     np.hstack( (np.zeros((state_dim,1)), Q, np.zeros((state_dim,control_dim))) ),
        #     np.hstack((np.zeros((control_dim,1+state_dim)), R)) ))

    def get_left_right_set_multiplier_terms(
        self, prog: MathematicalProgram, dynamics: npt.NDArray, intersections=True,
    ):
        res = 0

        left_m_deg = self.left.multiplier_deg()
        right_m_deg = self.right.multiplier_deg()

        if left_m_deg > 0:
            # add a nonnegative multiplier
            self.lambda_e_left = prog.NewContinuousVariables(
                left_m_deg, "l1_" + self.name
            )
            prog.AddLinearConstraint(ge(self.lambda_e_left, np.zeros(left_m_deg)))
            # form a term
            set_constraint = self.left.make_multiplier_terms(self.lambda_e_left)
            # apply constraints to x_n, u_n
            # left edge vertex is in left set
            res = res + set_constraint

            if (type(self.right) != BoxVertex or type(self.left) != BoxVertex) and intersections:
                # right edge vertex is in left set; this is done in order to not clip edges
                self.lambda_e_right_in_left = prog.NewContinuousVariables(
                    left_m_deg, "l2_" + self.name
                )
                prog.AddLinearConstraint(
                    ge(self.lambda_e_right_in_left, np.zeros(left_m_deg))
                )

                # form a term
                another_set_constraint = self.left.make_multiplier_terms(
                    self.lambda_e_right_in_left
                )
                # right edge vertex is in left set
                res = res + dynamics.T @ another_set_constraint * dynamics

        if right_m_deg > 0:
            if not intersections:
                # add a nonnegative multiplier
                self.lambda_e_right = prog.NewContinuousVariables(
                    right_m_deg, "l2_" + self.name
                )
                prog.AddLinearConstraint(ge(self.lambda_e_right, np.zeros(right_m_deg)))
                # intersect left and right boxes and get the constraints
                set_constraint = self.right.make_multiplier_terms(self.lambda_e_right)
                # right edge-vertex is in right set
                res = res + dynamics.T @ set_constraint * dynamics

            else:
                if type(self.right) != BoxVertex or type(self.left) != BoxVertex:
                    WARN("some edge is not a box")
                    # add a nonnegative multiplier
                    self.lambda_e_right = prog.NewContinuousVariables(
                        right_m_deg, "l3_" + self.name
                    )
                    prog.AddLinearConstraint(ge(self.lambda_e_right, np.zeros(right_m_deg)))
                    # get the constraints
                    set_constraint = self.right.make_multiplier_terms(self.lambda_e_right)
                    # apply constraints to x_{n+1}
                    # right edge vertex is in right set
                    res = res + dynamics.T @ set_constraint * dynamics
                else:
                    # add a nonnegative multiplier
                    self.lambda_e_right = prog.NewContinuousVariables(
                        right_m_deg, "l2_" + self.name
                    )
                    prog.AddLinearConstraint(ge(self.lambda_e_right, np.zeros(right_m_deg)))
                    # intersect left and right boxes and get the constraints
                    set_constraint = self.right.make_box_intersection_multiplier_terms(
                        self.left, self.lambda_e_right
                    )
                    # right edge-vertex is in right set
                    res = res + dynamics.T @ set_constraint * dynamics

        return res

    def s_procedure(self, prog: MathematicalProgram, A, B, Q, R, intersections = True):
        self.A = A
        self.B = B
        self.Q_cost = Q
        self.R_cost = R
        self.x_star = self.left.x_star

        state_dim = self.state_dim
        control_dim = self.control_dim
        full_dim = state_dim + control_dim + 1

        # build the potential difference terms
        Sn = self.left.get_quadratic_potential_matrix()

        # plug in for x_n_1 = A x_n + B u_n
        Sn1 = self.right.get_quadratic_potential_matrix()

        dynamics = np.vstack(
            (
                np.zeros((1, full_dim)),
                np.hstack((np.zeros((state_dim, 1)), A, B)),
                np.zeros((control_dim, full_dim)),
            )
        )

        potential_difference = dynamics.T @ Sn1 @ dynamics - Sn

        # edge cost
        edge_cost = self.make_lqr_cost_matrix(Q, R)

        # linear state inequalities stuff
        set_multiplier_terms = self.get_left_right_set_multiplier_terms(prog, dynamics, intersections)
        # set_multiplier_terms = 0

        # form the entire matrix
        psd_mat = edge_cost + potential_difference + set_multiplier_terms

        prog.AddPositiveSemidefiniteConstraint(psd_mat)
