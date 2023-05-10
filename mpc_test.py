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

def get_LRAB_vertex_boundaries(letter: str):
    u_lb, u_ub = -2 * np.ones((2,1)), 2 * np.ones((2,1))
    x_dot_lb, x_dot_ub = -5 * np.ones((2,1)), 5 * np.ones((2,1))
    if letter == "L":
        x_lb, x_ub = np.array( [[0],[0]] ), np.array( [[2],[6]] )
    elif letter == "R":
        x_lb, x_ub = np.array( [[4],[0]] ), np.array( [[6],[6]] )
    elif letter == "B":
        x_lb, x_ub = np.array( [[0],[0]] ), np.array( [[6],[2]] )
    elif letter == "A":
        x_lb, x_ub = np.array( [[0],[4]] ), np.array( [[6],[6]] )

    lb = np.vstack((x_lb, x_dot_lb, u_lb))
    ub = np.vstack((x_ub, x_dot_ub, u_ub))
    return lb, ub



    


def make_a_simple_mpc_test(N = 10, verbose=False):
    assert N >= 2, "need at least 2 horizon steps"
    # a 2d double integrator, goal at 0
    # LQR costs
    Q = np.eye(4) * 1
    R = np.eye(2) * 1
    Q_final = np.zeros( (4,4) )
    # Q_final = np.eye(4) * 2

    # linear discrete double integrator dynamics
    dt = 0.1
    A = np.eye(4)
    A[0,1], A[2,3] = dt, dt
    B = np.zeros( (4,2) )
    B[1,0], B[3,1] = dt, dt

    # vertex dimensions
    full_dim = 6
    state_dim = 4

    # define the target point:
    x_star = np.array([3,5,0,0])

    # individual vertices
    vertices = [] # type: T.List[Vertex]
    edges = [] # type T.List[Edge]

    # add vertices
    prog = MathematicalProgram()
    set_names = ["L", "R", "B", "A"]

    for n in range(N+1):
        # for each layer
        new_layer = [] # type: T.List[Vertex]
        for set_name in set_names:
            # for each vertex in that layer
            lb, ub = get_LRAB_vertex_boundaries(set_name)
            v = BoxVertex( str(n)+set_name, prog,lb,ub, PSD_ON_STATE, state_dim, x_star)
            new_layer.append(v)
        vertices.append(new_layer)

    # add edges
    for n in range(N):
        for v in vertices[n]:
            for w in vertices[n+1]:
                add_me = False
                if "L" in v.name and "R" not in w.name:
                    add_me = True
                if "R" in v.name and "L" not in w.name:
                    add_me = True
                if "B" in v.name and "A" not in w.name:
                    add_me = True
                if "A" in v.name and "B" not in w.name:
                    add_me = True
                if add_me:
                    e = LinearDynamicsEdge(v, w)
                    e.s_procedure(prog, A, B, Q, R)
                    edges.append(e)
    
    # final cost_to_go is Q_final
    for v in vertices[-1]:
        prog.AddLinearConstraint( eq(v.Q[:state_dim, :state_dim], Q_final) )
    
    # add cost from all initial sets
    for v in vertices[0]:
        prog.AddLinearCost(-v.cost_of_integral_over_the_state())

    timer = timeit()
    solution = Solve(prog)
    timer.dt()
    INFO( solution.is_success(), verbose=verbose )
    INFO( solution.get_optimal_cost(), verbose=verbose )
    INFO(solution.get_solution_result())


if __name__ == "__main__":
    make_a_simple_mpc_test(10, verbose=True)
    