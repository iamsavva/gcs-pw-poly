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
from vertex import FREE, PSD, PSD_QUADRATIC, PSD_WITH_IDENTITY
from edge import Edge


def make_a_simple_lqr_test(N = 5):
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
    n = 6 # 4 + 2; 4 states, 2 controls
    # make PSD cost matrix
    
    L = np.zeros( (2*n+1, 2*n+1) )
    L[1:5, 1:5] = Q
    L[5:7, 5:7] = R

    L_final = np.zeros( (2*n+1, 2*n+1) )
    L_final[1:5, 1:5] = Q
    L_final[5:7, 5:7] = R
    L_final[7:11, 7:11] = Q_final
    # make double AL, AR matrices
    AL = np.hstack( (A,B) )
    AR = -np.hstack( ( np.eye(4), np.zeros((4,2)) ) )

    vertices = []
    edges = []

    # add vertices
    prog = MathematicalProgram()
    for i in range(N+1):
        v = Vertex(str(i), prog, n, PSD_QUADRATIC) # TODO: fix me
        # v = Vertex(str(i), prog, n, PSD) # TODO: fix me
        vertices.append(v)

    # add edges
    for i in range(N):
        e = Edge(vertices[i], vertices[i+1])
        if i == N-1:
            e.set_cost(L_final)
        e.set_cost(L)
        e.add_linear_constraints( AL, AR )
        # e.lqr_s_procedure(prog, A, B, Q, R)
        e.s_procedure(prog, A, B)
        edges.append(e)

    box_lb, box_ub = -1*np.ones(4), 1*np.ones(4)

    # maximize potential over the integral
    cost = vertices[0].lqr_integral_over_first_k_states( box_lb, box_ub, 4 )
    # cost = vertices[0].cost_at_point( np.array([3,0,3,0,0,0]) )
    prog.AddLinearCost(-cost)
    prog.AddLinearConstraint( eq(vertices[-1].Q[:4, :4], Q_final) )

    timer = timeit()
    solution = Solve(prog)
    timer.dt()
    print( solution.is_success() )
    print( solution.get_optimal_cost() )

    S = np.zeros( (4,4) )
    for i in range(N+1):
        rounding = 10
        INFO("S at step ", N-i, ":")
        print( np.round( solution.GetSolution( vertices[N-i].Q[:4,:4] ), rounding) )

        WARN( "True at step ", N-i, ":")
        WARN(np.round(S, rounding))
        S = Q + A.T @ S @ A - (A.T @ S @ B) @ np.linalg.inv(R + B.T @ S @ B ) @ (B.T @ S @ A)
        

        


if __name__ == "__main__":
    make_a_simple_lqr_test(7)
    

