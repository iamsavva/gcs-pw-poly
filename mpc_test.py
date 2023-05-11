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
    control_limit = 100
    x_dot = 5
    u_lb, u_ub = -control_limit * np.ones((2, 1)), control_limit * np.ones((2, 1))
    
    if letter == "L":
        x_lb, x_ub = np.array([0, -x_dot, 0, -x_dot]), np.array([2, x_dot, 6, x_dot])
    elif letter == "R":
        x_lb, x_ub = np.array([4, -x_dot, 0, -x_dot]), np.array([6, x_dot, 6, x_dot])
        # x_lb, x_ub = np.array([[4], [0]]), np.array([[6], [6]])
    elif letter == "B":
        x_lb, x_ub = np.array([0, -x_dot, 0, -x_dot]), np.array([6, x_dot, 2, x_dot])
        # x_lb, x_ub = np.array([[0], [0]]), np.array([[6], [2]])
    elif letter == "A":
        x_lb, x_ub = np.array([0, -x_dot, 4, -x_dot]), np.array([6, x_dot, 6, x_dot])
        # x_lb, x_ub = np.array([[0], [4]]), np.array([[6], [6]])

    x_lb = x_lb.reshape(4,1)
    x_ub = x_ub.reshape(4,1)
    lb = np.vstack((x_lb, u_lb))
    ub = np.vstack((x_ub, u_ub))
    return lb, ub

def get_LRAB_diag_vertex_boundaries(letter: str):
    control_limit = 50
    x_dot = 5
    u_lb, u_ub = -control_limit * np.ones((2, 1)), control_limit * np.ones((2, 1))
    
    if letter == "LL":
        x_lb, x_ub = np.array([0, -x_dot, 2, -x_dot]), np.array([2, x_dot, 4, x_dot])
    elif letter == "RR":
        x_lb, x_ub = np.array([4, -x_dot, 2, -x_dot]), np.array([6, x_dot, 4, x_dot])
    elif letter == "BB":
        x_lb, x_ub = np.array([2, -x_dot, 0, -x_dot]), np.array([4, x_dot, 2, x_dot])
    elif letter == "AA":
        x_lb, x_ub = np.array([2, -x_dot, 4, -x_dot]), np.array([4, x_dot, 6, x_dot])
    elif letter == "LA":
        x_lb, x_ub = np.array([0, -x_dot, 4, -x_dot]), np.array([2, x_dot, 6, x_dot])
    elif letter == "AR":
        x_lb, x_ub = np.array([4, -x_dot, 4, -x_dot]), np.array([6, x_dot, 6, x_dot])
    elif letter == "LB":
        x_lb, x_ub = np.array([0, -x_dot, 0, -x_dot]), np.array([2, x_dot, 2, x_dot])
    elif letter == "BR":
        x_lb, x_ub = np.array([4, -x_dot, 0, -x_dot]), np.array([6, x_dot, 2, x_dot])

    x_lb = x_lb.reshape(4,1)
    x_ub = x_ub.reshape(4,1)
    lb = np.vstack((x_lb, u_lb))
    ub = np.vstack((x_ub, u_ub))
    return lb, ub



def make_a_bigger_mpc_test(N=10, verbose=False, dt = 0.1, dumb_policy=False):
    assert N >= 2, "need at least 2 horizon steps"
    # a 2d double integrator, goal at 0
    # LQR costs
    Q = np.eye(4) * 5
    Q[1,1] = Q[1,1]*0.1
    Q[3,3] = Q[3,3]*0.1
    R = np.eye(2) * 1
    Q_final = Q * 1

    # linear discrete double integrator dynamics
    A = np.eye(4)
    A[0, 1], A[2, 3] = dt, dt
    B = np.zeros((4, 2))
    B[1, 0], B[3, 1] = dt, dt
    B[0, 0], B[2, 1] = dt**2/2, dt**2/2

    # vertex dimensions
    full_dim = 6
    state_dim = 4

    # define the target point:
    # x_star = np.array([3, 0, 5, 0])
    x_star = np.array([0, 0, 0, 0])

    # individual vertices
    vertices = []  # type: T.List[Vertex]
    edges = []  # type T.List[Edge]

    # add vertices
    prog = MathematicalProgram()
    set_names = ["LL", "RR", "BB", "AA", "LA", "AR", "LB", "BR"]

    for n in range(N + 1):
        # for each layer
        new_layer = []  # type: T.List[Vertex]
        for set_name in set_names:
            # for each vertex in that layer
            lb, ub = get_LRAB_diag_vertex_boundaries(set_name)
            v = BoxVertex(
                str(n) + set_name, prog, lb, ub, PSD_ON_STATE, state_dim, x_star, Q
            )
            new_layer.append(v)
        vertices.append(new_layer)

    # add edges
    for n in range(N):
        for v in vertices[n]:
            for w in vertices[n + 1]:
                add_me = False
                if "LL" in v.name and ("LL" in w.name or "LB" in w.name or "LA" in w.name):
                    add_me = True
                if "LA" in v.name and ("LA" in w.name or "LL" in w.name or "AA" in w.name):
                    add_me = True
                if "AA" in v.name and ("LA" in w.name or "AA" in w.name or "AR" in w.name):
                    add_me = True
                if "AR" in v.name and ("AA" in w.name or "AR" in w.name or "RR" in w.name):
                    add_me = True
                if "RR" in v.name and ("AR" in w.name or "RR" in w.name or "BR" in w.name):
                    add_me = True
                if "BR" in v.name and ("RR" in w.name or "BR" in w.name or "BB" in w.name):
                    add_me = True
                if "BB" in v.name and ("BR" in w.name or "BB" in w.name or "LB" in w.name):
                    add_me = True
                if "LB" in v.name and ("BB" in w.name or "LB" in w.name or "LL" in w.name):
                    add_me = True
                
                if add_me:
                    e = LinearDynamicsEdge(v, w)
                    e.s_procedure(prog, A, B, Q, R, intersections=False)
                    edges.append(e)

    # final cost_to_go is Q_final
    if dumb_policy:
        S = Q_final
        for i in range(len(vertices)-1, -1, -1):
            layer = vertices[i]
            for v in layer:
                prog.AddLinearConstraint(eq(v.Q[:state_dim, :state_dim], S))
                prog.AddLinearConstraint(eq(v.r[:state_dim], -S @ x_star))
                prog.AddLinearConstraint(v.s[0,0] == (x_star.T @ S @ x_star))
            S = (
                Q
                + A.T @ S @ A
                - (A.T @ S @ B) @ np.linalg.inv(R + B.T @ S @ B) @ (B.T @ S @ A)
            )
    else:
        for v in vertices[-1]:
            prog.AddLinearConstraint(eq(v.Q[:state_dim, :state_dim], Q_final))
            prog.AddLinearConstraint(eq(v.r[:state_dim], -Q_final @ x_star))
            prog.AddLinearConstraint(v.s[0,0] == (x_star.T @ Q_final @ x_star))

    # add cost from all initial sets
    for v in vertices[0]:
        if v.name in ("0AR"):
# if v.name in ("0AR", "0AA", "0RR"):
            prog.AddLinearCost(-v.cost_of_integral_over_the_state())


    timer = timeit()
    solution = Solve(prog)
    timer.dt()
    INFO(solution.is_success(), verbose=verbose)
    INFO(solution.get_optimal_cost(), verbose=verbose)
    INFO(solution.get_solution_result())
    return vertices, edges, solution


def make_a_simple_mpc_test(N=10, verbose=False, dt = 0.1, dumb_policy=False):
    assert N >= 2, "need at least 2 horizon steps"
    # a 2d double integrator, goal at 0
    # LQR costs
    Q = np.eye(4) * 5
    Q[1,1] = Q[1,1]*0.1
    Q[3,3] = Q[3,3]*0.1
    R = np.eye(2) * 1
    Q_final = Q * 1

    # linear discrete double integrator dynamics
    A = np.eye(4)
    A[0, 1], A[2, 3] = dt, dt
    B = np.zeros((4, 2))
    B[1, 0], B[3, 1] = dt, dt
    B[0, 0], B[2, 1] = dt**2/2, dt**2/2

    # vertex dimensions
    full_dim = 6
    state_dim = 4

    # define the target point:
    # x_star = np.array([3, 0, 5, 0])
    x_star = np.array([0, 0, 0, 0])

    # individual vertices
    vertices = []  # type: T.List[Vertex]
    edges = []  # type T.List[Edge]

    # add vertices
    prog = MathematicalProgram()
    set_names = ["L", "R", "B", "A"]

    for n in range(N + 1):
        # for each layer
        new_layer = []  # type: T.List[Vertex]
        for set_name in set_names:
            # for each vertex in that layer
            lb, ub = get_LRAB_vertex_boundaries(set_name)
            v = BoxVertex(
                str(n) + set_name, prog, lb, ub, PSD_ON_STATE, state_dim, x_star
            )
            new_layer.append(v)
        vertices.append(new_layer)

    # add edges
    for n in range(N):
        for v in vertices[n]:
            for w in vertices[n + 1]:
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
    if dumb_policy:
        S = Q_final
        for i in range(len(vertices)-1, -1, -1):
            layer = vertices[i]
            for v in layer:
                prog.AddLinearConstraint(eq(v.Q[:state_dim, :state_dim], S))
                prog.AddLinearConstraint(eq(v.r[:state_dim], -S @ x_star))
                prog.AddLinearConstraint(v.s[0,0] == (x_star.T @ S @ x_star))
            S = (
                Q
                + A.T @ S @ A
                - (A.T @ S @ B) @ np.linalg.inv(R + B.T @ S @ B) @ (B.T @ S @ A)
            )

    for v in vertices[-1]:
        prog.AddLinearConstraint(eq(v.Q[:state_dim, :state_dim], Q_final))
        prog.AddLinearConstraint(eq(v.r[:state_dim], -Q_final @ x_star))
        prog.AddLinearConstraint(v.s[0,0] == (x_star.T @ Q_final @ x_star))

    # add cost from all initial sets
    for v in vertices[0]:
        if v.name in ("0A", "0R"):
            prog.AddLinearCost(-v.cost_of_integral_over_the_state())
            # prog.AddLinearCost(-v.cost_())


    timer = timeit()
    solution = Solve(prog)
    timer.dt()
    INFO(solution.is_success(), verbose=verbose)
    INFO(solution.get_optimal_cost(), verbose=verbose)
    INFO(solution.get_solution_result())
    return vertices, edges, solution



if __name__ == "__main__":
    make_a_simple_mpc_test(10, verbose=True)
