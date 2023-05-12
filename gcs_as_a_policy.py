import typing as T

import numpy as np
import numpy.typing as npt
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.patches as patches

from pydrake.solvers import (
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
)
from pydrake.symbolic import Polynomial, Variable, Variables, Evaluate

from pydrake.math import eq, le, ge

from util import timeit, INFO, YAY, ERROR, WARN
from vertex import Vertex, BoxVertex, PolytopeVertex, EllipsoidVertex
from vertex import FREE, PSD, PSD_ON_STATE, PSD_WITH_IDENTITY
from edge import LinearDynamicsEdge
from good_LQR_unit_test import make_a_simple_lqr_test



LEAST_SQUARES_POLICY = "LEAST_SQUARES_POLICY"
QP_POLICY = "QP_POLICY"

# assumption: MPC
def make_a_path_from_policy(
    vertices: T.List[T.List[Vertex]],
    edges: T.List[LinearDynamicsEdge],
    solution: MathematicalProgramResult,
    initial_state: npt.NDArray,
    policy = QP_POLICY,
    pick_lowest_cost = True,
    with_replacement = False
):
    N = len(vertices) - 1
    state = initial_state.reshape(len(initial_state), 1)
    vertex = None

    states = []
    states.append(state)

    # figure out what vertex you are in
    for v in vertices[0]:
        if v.is_state_inside(state):
            vertex = v
            break
    assert vertex is not None, ERROR("sate is inside no vertex: ", state)
    total_cost = 0

    if with_replacement:
        N = 20

    for i in range(N - 1):
        # find all edges out of current vertex
        edges_out = []  # type: LinearDynamicsEdge
        for e in edges:
            if e.left.name == vertex.name:
                edges_out.append(e)
        assert len(edges_out) > 0, ERROR("no edges out ", i, vertex.name)

        # for each vertex, compute optimal control input
        options = []
        for e in edges_out:
            A, B, Q, R = e.A, e.B, e.Q_cost, e.R_cost
            
            Sw, rw, sw = e.right.get_state_potential(solution)

            if policy == LEAST_SQUARES_POLICY:
                u = - np.linalg.inv(R + B.T @ Sw @ B) @ B.T @ (Sw @ A @ state + rw)
                # u = - np.linalg.inv(R + B.T @ Sw @ B)@ B.T @ (Sw @ A @ state)
                cost = (u.T @ R @ u + (A@state + B@u).T @ Sw @ (A@state + B@u) + 2 * rw.T @ (A@state + B@u) + sw)[0,0]
                # print(u, cost, e.right.name)
                options.append( (u, cost, e.right) )
            elif policy == QP_POLICY:
                prog = MathematicalProgram()
                y = prog.NewContinuousVariables(vertex.state_dim).reshape(vertex.state_dim, 1)
                u = prog.NewContinuousVariables(vertex.control_dim).reshape(vertex.control_dim, 1)

                quadratic_cost = ( state.T @ Q @ state + u.T @ R @ u + y.T @ Sw @ y + 2*rw.T @ y + sw)[0,0]

                prog.AddQuadraticCost( quadratic_cost )

                prog.AddLinearConstraint( eq( y, A @state + B @ u ) ) 
                # intersect left and right boxes (i don't wanna clip corners)
                lb, ub = e.left.get_box_intersection( e.right )
                # lb, ub = e.right.lb[:vertex.state_dim], e.right.ub[:vertex.state_dim]
                prog.AddLinearConstraint( le( lb[:vertex.state_dim], y ))
                prog.AddLinearConstraint( le( y, ub[:vertex.state_dim] ))
                
                u_lb, u_ub = e.left.get_control_bounds()
                prog.AddLinearConstraint( le( u_lb, u ))
                prog.AddLinearConstraint( le( u, u_ub ))

                qp_solution = Solve(prog)
                if qp_solution.is_success():
                    u_star = qp_solution.GetSolution(u).reshape(vertex.control_dim, 1)
                    cost = qp_solution.get_optimal_cost()
                    # print(u_star, cost, e.right.name)
                    options.append( (u_star, cost, e.right) )
        
        best_u, best_cost, best_v = -1, float("inf"), None,
        S, r, s = vertex.get_state_potential(solution)
        true_current_cost = (state.T@S@state + 2 * r.T @ state + s)[0,0]
        for (u_star, cost, v) in options:
            if pick_lowest_cost:
                if cost < best_cost:
                    best_u, best_cost, best_v = u_star, cost, v
            else:
                if abs(cost-true_current_cost) <= abs(best_cost-true_current_cost):
                    best_u, best_cost, best_v = u_star, cost, v
            # 
        
        assert best_v is not None, ERROR("No solution found")

        total_cost += (state.T @ Q @ state + best_u.T @ R @ best_u)[0,0]

        state = A @ state + B @ best_u

        # assert best_v.is_state_inside(state), ERROR( state.T, best_v.lb.T, best_v.ub.T )

        if with_replacement:
            for v in vertices[0]:
                if v.name[-2:] == best_v.name[-2:]:
                    best_v = v

        vertex = best_v
        states.append(state)
    total_cost += (state.T @ Q @ state)[0,0]
    return states, total_cost

def plot_policy_realizations_from_state( vertices: T.List[T.List[Vertex]],
    edges: T.List[LinearDynamicsEdge],
    solution: MathematicalProgramResult,
    initial_states: npt.NDArray,
    policy:str = QP_POLICY,
    pick_lowest_cost = True,
    xlim=[-5,5], ylim =[-5,5],
    with_replacement = False):
    paths = [ make_a_path_from_policy(vertices, edges, solution, state, policy, pick_lowest_cost, with_replacement) for state in initial_states ]

    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    ax.add_patch(patches.Rectangle( (2,2), 2,2, linewidth=0, edgecolor="black", facecolor="black"))

    i = 0
    costs = []
    for (path, cost) in paths:
        costs.append(np.round(cost,3))
        x,y = [], []
        for v in path:
            x.append(v[0]), y.append(v[2])
        ax.plot(x, y, label=str(i), color = "blue", linewidth = 2)
        i+=1
        for i in range(len(x)):
            ax.annotate(str(i), (x[i], y[i]))

    
    return fig, ax, costs


if __name__ == "__main__":
    def state(x,y):
        return np.array([x,0,y,0])
    vertices, edges, solution = make_a_simple_lqr_test(10, False)
    initial_states = [ state(3,3), state(-4,3), state(-2,-2) ]
    fig, ax = plot_policy_realizations_from_state(vertices, edges, solution, initial_states, LEAST_SQUARES_POLICY)
    fig.show()
