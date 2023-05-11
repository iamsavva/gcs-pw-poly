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

# assumption: MPC
def make_a_plot_from_policy(vertices: T.List[T.List[Vertex]], edges:T.List[LinearDynamicsEdge], solution:MathematicalProgramResult, initial_state: npt.NDArray):
    N = len(vertices)-1
    state = initial_state.reshape()
    vertex = None

    states = []
    states.append(state)

    # figure out what vertex you are in
    for v in vertices[0]:
        if v.is_state_inside(state):
            vertex = v
            break
    assert vertex is not None, ERROR("sate is inside no vertex: ", state)


    for i in range(N-1):
        # find all edges out of current vertex
        edges_out = [] # type: LinearDynamicsEdge
        for e in edges:
            if e.left.name == vertex.name:
                edges_out.append(e)
        assert len(edges_out) > 0, ERROR("no edges out ", i, vertex.name)
        
        # for each vertex, compute optimal control input
        for e in edges_out:
            prog = MathematicalProgram()
            x = prog.NewContinuousVariables(vertex.state_dim)
            u = prog.NewContinuousVariables(vertex.control_dim)
            A, B, Q, R, x_star = e.A, e.B, e.Q_cost, e.R_cost, e.x_star
            quadratic_cost = e.right.cost_at_point(x, solution) + (u.T @ R @ u)[0,0]

            # you shouldn't be solving a QP
            # you already know the solution to this problem
            # this is a much easier problem

            # but it is a good sanity check -- do i get the exact same as from policy, or do i get different if boundaries are imposed

            


        


