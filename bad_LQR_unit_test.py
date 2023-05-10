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
from edge import Edge


def make_a_simple_bad_lqr_test(N = 10, verbose=False):
    assert N >= 2, "need at least 2 horizon steps"
    # a 2d double integrator, goal at 0
    # LQR costs
    Q = np.eye(4) * 4
    R = np.eye(2) * 1
    Q_final = np.zeros( (4,4) )
    # linear discerte double integrator dynamics
    dt = 0.1
    A = np.eye(4)
    A[0,1], A[2,3] = dt, dt
    B = np.zeros( (4,2) )
    B[1,0], B[3,1] = dt, dt

    # vertex state
    full_dim = 6
    state_dim = 4


    # make PSD cost matrix
    L = np.zeros( (2*full_dim+1, 2*full_dim+1) )
    L[1:5, 1:5] = Q
    L[5:7, 5:7] = R

    # make double AL, AR matrices
    AL = np.hstack( (A,B) )
    AR = -np.hstack( ( np.eye(4), np.zeros((4,2)) ) )

    vertices = [] # type: T.List[Vertex]
    edges = [] # type T.List[Edge]

    # add vertices
    prog = MathematicalProgram()
    for i in range(N+1):
        v = Vertex(str(i), prog, full_dim, PSD_ON_STATE, state_dim)
        vertices.append(v)

    # add edges
    for i in range(N):
        e = Edge(vertices[i], vertices[i+1])
        e.set_cost(L)
        e.add_linear_constraints( AL, AR ) 
        e.s_procedure(prog, A, B)
        edges.append(e)

    box_lb, box_ub = -1*np.ones(4), 1*np.ones(4)

    # maximize potential over the integral
    cost = vertices[0].cost_of_integral_over_the_state( box_lb, box_ub)
    prog.AddLinearCost(-cost)
    prog.AddLinearConstraint( eq(vertices[-1].Q[:4, :4], Q_final) )

    timer = timeit()
    solution = Solve(prog)
    timer.dt()
    INFO( solution.is_success(), verbose=verbose )
    INFO( solution.get_optimal_cost(), verbose=verbose )

    if solution.is_success():
        S = Q_final
        for i in range(N+1):
            rounding = 3

            INFO("S at step ", N-i, ":", verbose=verbose)
            pot_PSD = np.round( solution.GetSolution( vertices[N-i].Q[:4,:4] ), rounding)
            INFO( pot_PSD, verbose=verbose )

            WARN( "True at step ", N-i, ":", verbose=verbose)
            WARN(np.round(S, rounding), verbose=verbose)

            assert np.allclose(np.round(S, rounding), pot_PSD, rtol = 0.1), ERROR("MATRICES DON'T MATCH")

            S = Q + A.T @ S @ A - (A.T @ S @ B) @ np.linalg.inv(R + B.T @ S @ B ) @ (B.T @ S @ A)
        

    YAY("Passed LQR test implemented using the S procedure")


if __name__ == "__main__":
    make_a_simple_bad_lqr_test(5, True)
    

