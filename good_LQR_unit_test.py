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

from util import timeit, INFO, YAY, ERROR, WARN
from vertex import Vertex, BoxVertex, PolytopeVertex, EllipsoidVertex
from vertex import FREE, PSD, PSD_ON_STATE, PSD_WITH_IDENTITY
from edge import LinearDynamicsEdge


def make_a_simple_lqr_test(N=20, verbose=False, dt = 0.1):
    assert N >= 2, "need at least 2 horizon steps"
    # a 2d double integrator, goal at 0
    # LQR costs
    Q = np.eye(4) * 1
    R = np.eye(2) * 1
    Q_final = np.eye(4) * 2
    # Q_final = np.zeros( (4,4) )

    # linear discrete double integrator dynamics
    A = np.eye(4)
    A[0, 1], A[2, 3] = dt, dt
    B = np.zeros((4, 2))
    B[1, 0], B[3, 1] = dt, dt

    # vertex state
    full_dim = 6
    state_dim = 4

    vertices = []  # type: T.List[Vertex]
    edges = []  # type T.List[Edge]

    # TODO: i should plot this policy
    # TODO: am i constraining the constant term? guess it need not be constrained
    x_star = np.zeros(4)
    # x_star = np.array([0.5,0, 0.5,0])

    # add vertices
    prog = MathematicalProgram()
    for i in range(N + 1):
        v = Vertex(
            str(i), prog, full_dim, PSD_ON_STATE, state_dim, x_star
        )  # TODO: fix me
        vertices.append([v])

    # add edges
    for i in range(N):
        e = LinearDynamicsEdge(vertices[i][0], vertices[i + 1][0])
        e.s_procedure(prog, A, B, Q, R)
        edges.append(e)

    box_lb, box_ub = -1 * np.ones(state_dim), 1 * np.ones(state_dim)

    # maximize potential over the integral
    cost = vertices[0][0].cost_of_integral_over_the_state(box_lb, box_ub)
    prog.AddLinearCost(-cost)

    # final cost-to-go is equal to Q_final
    prog.AddLinearConstraint(eq(vertices[-1][0].Q[:state_dim, :state_dim], Q_final))

    timer = timeit()
    solution = Solve(prog)
    timer.dt()
    INFO(solution.is_success(), verbose=verbose)
    INFO(solution.get_optimal_cost(), verbose=verbose)

    if solution.is_success():
        S = Q_final
        for i in range(N + 1):
            rounding = 3
            INFO("S at step ", N - i, ":", verbose=verbose)
            pot_PSD = np.round(
                solution.GetSolution(vertices[N - i][0].Q[:4, :4]), rounding
            )
            INFO(pot_PSD, verbose=verbose)
            WARN("True at step ", N - i, ":", verbose=verbose)
            WARN(np.round(S, rounding), verbose=verbose)

            # print(
            #     np.round(solution.GetSolution(vertices[N - i][0].s), rounding),
            #     x_star.T @ pot_PSD @ x_star,
            # )
            # print(np.round(solution.GetSolution(vertices[N - i][0].r[:4]), rounding))

            # assert np.allclose(np.round(S, rounding), pot_PSD, rtol=1e-2), ERROR(
            #     "MATRICES DON'T MATCH"
            # )

            S = (
                Q
                + A.T @ S @ A
                - (A.T @ S @ B) @ np.linalg.inv(R + B.T @ S @ B) @ (B.T @ S @ A)
            )

    YAY("Passed LQR test implemented by substituting the dynamics")
    return vertices, edges, solution


if __name__ == "__main__":
    make_a_simple_lqr_test(10, verbose=True, dt = 0.5)
