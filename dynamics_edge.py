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
    
    def make_polytope_set_multipliers(self, prog:MathematicalProgram, A, b):
        m,n = A.shape
        b = b.reshape(m,1)
        assert n == self.n
        B = np.hstack( (b, -A) )
        mu = prog.NewContinuousVariables(m).reshape(m,1)
        prog.AddLinearConstraint( ge(mu, np.zeros((m,1))) )

        theta = prog.NewSymmetricContinuousVariables(m)
        prog.AddLinearConstraint( ge(theta, np.zeros((m,m))) )
        for i in range(m):
            prog.AddLinearConstraint(theta[i,i] == 0)

        e_1 = np.zeros((n+1,1))
        e_1[0,0] = 1
        
        # note the sign
        res = -(e_1 @ mu.T @ B + B.T @ mu @ e_1.T + B.T @ theta @ B)
        self.theta = theta
        return res


    def get_left_right_set_multiplier_terms(
        self, prog: MathematicalProgram, intersections=True,
    ):
        left_m_deg = self.left.multiplier_deg()
        right_m_deg = self.right.multiplier_deg()

        A_mats = []
        b_mats = []

        # return self.make_polytope_set_multipliers(prog, self.left.A, self.left.b)

        if left_m_deg > 0:
            A_mats.append(self.left.A)
            b_mats.append(self.left.b)

        if right_m_deg > 0:
            C,d = self.right.get_x_polytope()
            C = C @ np.hstack((self.A, self.B))
            # A_mats.append(np.hstack((C @ self.A, C @ self.B)))
            A_mats.append(C)
            b_mats.append(d)

        if len(A_mats) > 0:
            A = np.vstack(A_mats)
            b = np.vstack(b_mats)
            # print(A)
            # print(b)
            # print("---")
            # polyhedron = HPolyhedron(A, b)
            # polyhedron = polyhedron.ReduceInequalities()
            # A, b = polyhedron.A(), polyhedron.b()
            return self.make_polytope_set_multipliers(prog, A, b)
        return 0

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
                # np.zeros((1, full_dim)),
                np.hstack(( np.ones((1,1)), np.zeros((1, full_dim-1)) )),
                np.hstack((np.zeros((state_dim, 1)), A, B)),
                np.zeros((control_dim, full_dim)),
            )
        )

        potential_difference = dynamics.T @ Sn1 @ dynamics - Sn

        # edge cost
        edge_cost = self.make_lqr_cost_matrix(Q, R)

        # linear state inequalities stuff
        set_multiplier_terms = self.get_left_right_set_multiplier_terms(prog, intersections)

        # form the entire matrix
        psd_mat = edge_cost + potential_difference + set_multiplier_terms

        prog.AddPositiveSemidefiniteConstraint(psd_mat)
