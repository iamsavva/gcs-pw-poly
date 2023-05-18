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
from dynamics_edge import LinearDynamicsEdge

def get_set_centers(n):
    d= 0.2
    if n == 1:
        return (1,1), (1-d,1)
    elif n == 2:
        return (1,4), (1-d,2+d)
    elif n == 3:
        return (1,7), (1-d,1-d)
    elif n == 4:
        return (3,7), (1+d,1-d)
    elif n == 5:
        return (5,7), (1+d,1-d)
    elif n == 6:
        return (5.2,5.2), (1.2-d,1.2-d)
    elif n == 7:
        return (7.5,5), (1.5,1)
    elif n == 8:
        return (10,5), (1,1)
    elif n == 9:
        return (10,3), (1,1)
    elif n == 10:
        return (10,1), (1,1)
    elif n == 11:
        return (7.5,3), (1.5,1)
    elif n == 12:
        return (4,1), (2,1)
    
def check_edge(left, right):
    edges = []
    for i in range(1,13):
        edges.append( (i,i) )
    edges.append( (2,1) )
    edges.append( (3,2) )
    edges.append( (4,3) )
    edges.append( (5,4) )
    edges.append( (6,5) )
    edges.append( (6,7) )
    edges.append( (7,8) )
    edges.append( (8,9) )
    edges.append( (9,10) )
    edges.append( (10,11) )
    edges.append( (11,12) )
    edges.append( (12,1) )
    edges.append( (1,12) )
    return (left,right) in edges

    
def get_boundaries(number:int):
    control_limit = 30
    x_dot = 3
    u_lb, u_ub = -control_limit * np.ones((2, 1)), control_limit * np.ones((2, 1))

    (x,y), (w,h) = get_set_centers(number)

    x_lb = np.array([x-w, -x_dot, y-h, -x_dot])
    x_ub = np.array([x+w, x_dot, y+h, x_dot])

    x_lb = x_lb.reshape(4,1)
    x_ub = x_ub.reshape(4,1)
    lb = np.vstack((x_lb, u_lb))
    ub = np.vstack((x_ub, u_ub))
    return lb, ub





def corner_mpc_test(N=10, verbose=False, dt = 0.1, lqr_policy=False, push_up_only_at_0 = False):
    # assert N >= 2, "need at least 2 horizon steps"
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
    x_star = np.array([0, 0, 0, 0])

    # individual vertices
    vertices = []  # type: T.List[Vertex]
    edges = []  # type T.List[Edge]

    # add vertices
    prog = MathematicalProgram()
    set_names = ["0" + str(i) for i in range(1,10)] + ["10","11", "12"]

    for n in range(N + 1):
        # for each layer
        new_layer = []  # type: T.List[Vertex]
        for set_name in set_names:
            lb,ub = get_boundaries(int(set_name))
            v = BoxVertex( str(n) + set_name, prog, lb, ub, PSD_ON_STATE, state_dim, x_star, (B,R)  )
            new_layer.append(v)
        vertices.append(new_layer)

    # add edges
    for n in range(N):
        for v in vertices[n]:
            for w in vertices[n + 1]:
                if check_edge(int(v.name[-2:]), int(w.name[-2:])):
                    e = LinearDynamicsEdge(v, w)
                    e.s_procedure(prog, A, B, Q, R, intersections=False)
                    edges.append(e)

    # final cost_to_go is Q_final
    for v in vertices[-1]:
        prog.AddLinearConstraint(eq(v.Q[:state_dim, :state_dim], Q_final))
        prog.AddLinearConstraint(eq(v.r[:state_dim], -Q_final @ x_star))
        prog.AddLinearConstraint(v.s[0,0] == (x_star.T @ Q_final @ x_star))


    # push up cost jsut at first layer or everywhre
    if push_up_only_at_0 :
        for v in vertices[0]:
            # if v.name == "006":
                # prog.AddLinearCost(-v.cost_at_center())
            if v.name == "006" or v.name == "005":
                prog.AddLinearCost(-v.cost_of_integral_over_the_state())
    else:
        for layer in vertices:
            for v in layer:
                prog.AddLinearCost(-v.cost_of_integral_over_the_state())

    
    if lqr_policy:
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

    timer = timeit()
    solution = Solve(prog)
    timer.dt()
    INFO(solution.is_success(), verbose=verbose)
    INFO(solution.get_optimal_cost(), verbose=verbose)
    INFO(solution.get_solution_result())
    return vertices, edges, solution
