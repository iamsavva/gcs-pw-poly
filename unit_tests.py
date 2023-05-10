import typing as T

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  
    MathematicalProgram,
    Solve,
)

from util import timeit
from vertex import Vertex, BoxVertex, PolytopeVertex, EllipsoidVertex
from edge import Edge
from util import YAY, ERROR, INFO


def test_simple_box_chain(n:int = 2, num_vertices:int = 2, verbose=True):
    assert num_vertices >= 2, "Need at least 2 vertices"
    # assert n >= 1, "Boxes must be at least dimension 1 else i suck at math"

    prog = MathematicalProgram()
    vertices = []

    vt = BoxVertex("t", prog, np.zeros(n), 1*np.ones(n))
    vertices.append(vt)

    for i in range(num_vertices-2):
        vv = BoxVertex("v"+str(i), prog, (i+1) *np.ones(n), (i+2)*np.ones(n))
        vertices.append(vv)

    vs = BoxVertex("s", prog, (num_vertices-1) *np.ones(n), (num_vertices)*np.ones(n))
    vertices.append(vs)

    for i in range(num_vertices-1):
        e = Edge( vertices[i+1], vertices[i] )
        e.set_quadratic_cost()
        e.s_procedure(prog)


    prog.AddLinearConstraint(vt.cost_at_center() == 0)

    cost = vs.cost_at_center()
    prog.AddLinearCost( -cost )

    solution = Solve(prog)
    true_cost = (num_vertices-1) * n

    INFO("Testing for a chain of " + str(num_vertices) + " consecutive boxes in " + str(n) +"-dimensional space.", verbose=verbose)
    assert solution.is_success(), ERROR("Box solve failed! ", "n: ", n, "num_vertices: ", num_vertices)
    assert np.allclose(-solution.get_optimal_cost(), true_cost), ERROR("Cost should be", true_cost, " is ", -solution.get_optimal_cost())
    YAY("All checks passed", verbose=verbose)
    YAY("Solution successful: ", solution.is_success(), verbose=verbose)
    YAY("Optimal cost is: ", np.round(-solution.get_optimal_cost(),3), verbose=verbose)
    YAY("Cost should be:   " +  str(true_cost) + " = (" + str(num_vertices) + "-1) * " + str(n), verbose=verbose)
    YAY("-----------", verbose=verbose)


def test_simple_ellipsoid_chain(n:int = 2, num_vertices:int = 2, verbose=True):
    assert num_vertices >= 2, "Need at least 2 vertices"
    # assert n >= 1, "Boxes must be at least dimension 1 else i suck at math"

    prog = MathematicalProgram()
    vertices = []

    vt = EllipsoidVertex("t", prog, 0.5*np.ones(n), 0.5*np.eye(n))
    vertices.append(vt)

    for i in range(num_vertices-2):
        vv = EllipsoidVertex("v"+str(i), prog, 0.5+(i+1) *np.ones(n), 0.5*np.eye(n))
        vertices.append(vv)

    vs = EllipsoidVertex("s", prog, 0.5+(num_vertices-1) *np.ones(n), 0.5*np.eye(n))
    vertices.append(vs)

    for i in range(num_vertices-1):
        e = Edge( vertices[i+1], vertices[i] )
        e.set_quadratic_cost()
        e.s_procedure(prog)


    prog.AddLinearConstraint(vt.cost_at_center() == 0)

    cost = vs.cost_at_center()
    prog.AddLinearCost( -cost )

    solution = Solve(prog)
    true_cost = (num_vertices-1) * n

    INFO("Testing for a chain of " + str(num_vertices) + " consecutive ellipsoids in " + str(n) +"-dimensional space.", verbose=verbose)
    assert solution.is_success(), ERROR("Ellipsoid solve failed! ", "n: ", n, "num_vertices: ", num_vertices)
    assert np.allclose(-solution.get_optimal_cost(), true_cost), ERROR("Cost should be", true_cost, " is ", -solution.get_optimal_cost())
    YAY("All checks passed", verbose=verbose)
    YAY("Solution successful: ", solution.is_success(), verbose=verbose)
    YAY("Optimal cost is: ", np.round(-solution.get_optimal_cost(),3), verbose=verbose)
    YAY("Cost should be:   " +  str(true_cost) + " = (" + str(num_vertices) + "-1) * " + str(n), verbose=verbose)
    YAY("-----------", verbose=verbose)

if __name__ == "__main__":
    for n in range(1,5):
        for num_vertices in range(2,7,2):
            test_simple_box_chain(n, num_vertices, False)   
            test_simple_ellipsoid_chain(n, num_vertices, False)
    
    YAY("All tests passed!")
    INFO("If you broke the code, I wouldn't know")
