import typing as T
import numpy as np
from matplotlib import pyplot as plt
from util import timeit, YAY, INFO, ERROR, WARN

from gcs_as_a_policy import plot_policy_realizations_from_state, make_a_path_from_policy
from gcs_as_a_policy import LEAST_SQUARES_POLICY, QP_POLICY
from good_LQR_unit_test import make_a_simple_lqr_test
from mpc_test import make_a_simple_mpc_test, make_a_bigger_mpc_test

def state(x,y):
    return np.array([x,0,y,0])

dt = 0.25
# initial_states = [ state(5,5), state(5.5,4.5), state(4.5,5.5), state(5.5,3.8), state(3.8,5.5), state(5.5,3), state(3,5.5)] #, state(3.5 ,5.5), state(5.5,3.5) ]
# initial_states = [ state(3,5)] #, state(3.5 ,5.5), state(5.5,3.5) ]
initial_states = [ state(5.5,4.5), state(4.5,5.5), state(5.5,3.8), state(3.8,5.5), state(5,5.5), state(5.5,5)] #, state(3.5 ,5.5), state(5.5,3.5) ]
xlim = [0,6]
ylim = [0,6]

N = 2
# make more sets
# what are the costs incurred by each policy?

vertices, edges, solution = make_a_bigger_mpc_test(N = N, verbose = True, dt = dt, lqr_policy = False)
# for v in vertices[0]:    
    # print(np.round(solution.GetSolution( v.Q[:4,:4]),3))
    # print(np.round(solution.GetSolution( v.r[:4]),3))
    # print(np.round(solution.GetSolution( v.s),3))

# for e in edges:
#     print(np.round(solution.GetSolution( e.theta[:5, :5]), 2))
# vertices, edges, solution = make_a_simple_mpc_test(N = N, verbose = False, dt = dt, lqr_policy = False)

# fig, ax, costs = plot_policy_realizations_from_state(vertices, edges, solution, initial_states, LEAST_SQUARES_POLICY, True, xlim, ylim, with_replacement = False)
# YAY(costs)
# fig.show()