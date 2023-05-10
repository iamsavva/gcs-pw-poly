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

from util import timeit

from pydrake.math import eq, le, ge

FREE = "free"
PSD = "psd"
PSD_WITH_IDENTITY = "psd_with_identity"
PSD_ON_STATE = "PSD_ON_STATE"


class Vertex:
    def __init__(self, name:str, prog: MathematicalProgram, dim:int, pot_type:str=PSD, state_dim:int=None):
        self.n = dim
        self.name = name
        if state_dim is not None:
            self.state_dim = state_dim
            self.control_dim = self.n - state_dim
        self.define_potential(prog, pot_type)

    def define_potential(self, prog: MathematicalProgram, pot_type:str):
        # P = 
        #   [ P_11  P_1  ] 
        #   [ P_x1  Pxx  ] 
        #   = 
        #   [ s  r.T ] 
        #   [ r   Q  ] 
        # P(x) = x.T Q x + 2 r.T x + s
        
        # make potential quadratic if it's used for HJB and cost to go
        if pot_type == PSD:
            self.Q = prog.NewSymmetricContinuousVariables(self.n, "Q_" + self.name)
            prog.AddPositiveSemidefiniteConstraint(self.Q)

        elif pot_type == PSD_WITH_IDENTITY:
            # technically we don't need Q the matrix Q to be PSD, we need Q and cost matrix L to be PSD
            self.Q = prog.NewSymmetricContinuousVariables(self.n, "Q_" + self.name)
            prog.AddPositiveSemidefiniteConstraint(self.Q + np.eye(self.n)) 

        elif PSD_ON_STATE:
            # TODO: add an option to add a cost here
            
            # state is 1 x u, except we are only interested in the cost to go acting on 1 and x alone.
            # PSD on state only -- on control inputs it's just 0
            Q = prog.NewSymmetricContinuousVariables(self.state_dim, "Q_" + self.name)
            prog.AddPositiveSemidefiniteConstraint(Q) 
            # complete the Q matrix
            self.Q = np.vstack((
                np.hstack( (Q, np.zeros((self.state_dim, self.control_dim))) ),
                np.zeros((self.control_dim, self.n))
            ))

        # add the linear terms
        if pot_type == PSD_ON_STATE:
            # PSD on state only -- define r and s for state only.
            r = prog.NewContinuousVariables(self.state_dim, "r_" + self.name).reshape(self.state_dim, 1)

            self.r = np.vstack( (r, np.zeros((self.control_dim, 1))) )
            self.s = prog.NewContinuousVariables(1, "s_"+self.name).reshape(1,1)

            # potential must evaluate to zero at the desired point
            prog.AddLinearConstraint(self.evaluate_partial_potential_at_point( np.zeros(self.n) ) == 0 )
        else:
            self.r = prog.NewContinuousVariables(self.n, "r_" + self.name).reshape(self.n, 1)
            self.s = prog.NewContinuousVariables(1, "s_"+self.name).reshape(1,1)

        
            

    def evaluate_partial_potential_at_point(self, x:npt.NDArray):
        assert len(x) == self.n
        x = x.reshape(self.n, 1)
        return (x.T @ self.Q @ x + 2*self.r.T @ x + self.s)[0,0]
    
    def cost_at_point(self, x:npt.NDArray, solution:MathematicalProgramResult=None):
        x = x.reshape(self.n, 1)
        if solution is None:
            return self.evaluate_partial_potential_at_point(x)
        else: 
            Q, r, s = solution.GetSolution(self.Q), solution.GetSolution(self.r), solution.GetSolution(self.s)
            return (x.T @ Q @ x + 2*r.dot(x) + s)[0,0]
        
    def cost_of_integral_over_a_box(self, lb:npt.NDArray, ub:npt.NDArray):
        n = self.n
        assert n == len(lb) == len(ub)
        temp_prog = MathematicalProgram()
        x_vec = temp_prog.NewIndeterminates(n)
        poly = Polynomial(self.evaluate_partial_potential_at_point(x_vec))
        for i in range(n):
            x_min, x_max, x = lb[i], ub[i], x_vec[i]
            integral_of_poly = poly.Integrate(x)
            poly = integral_of_poly.EvaluatePartial({x: x_max}) - integral_of_poly.EvaluatePartial({x:x_min})
        return poly.ToExpression()
    
    def cost_of_integral_over_the_state(self, lb:npt.NDArray, ub:npt.NDArray):
        assert self.state_dim == len(lb) == len(ub)
        k = self.state_dim

        temp_prog = MathematicalProgram()
        x_vec = temp_prog.NewIndeterminates(k).reshape(k,1)

        poly = Polynomial( (x_vec.T @ self.Q[:k, :k] @ x_vec + 2*self.r[:k].T @ x_vec + self.s )[0,0] )

        for i in range(k):
            x_min, x_max, x = lb[i], ub[i], x_vec[i][0]
            integral_of_poly = poly.Integrate(x)
            poly = integral_of_poly.EvaluatePartial({x: x_max}) - integral_of_poly.EvaluatePartial({x:x_min})
        return poly.ToExpression()

    def make_multiplier_terms(self, lambda_e:npt.NDArray, left:bool):
        return 0

    def multiplier_deg(self):
        return 0
    
    def get_quadratic_potential_matrix(self):
        Q = self.Q
        r = self.r
        s = self.s
        return np.vstack((
            np.hstack( (s, r.T) ),
            np.hstack( (r, Q) ) ))


class PolytopeVertex(Vertex):
    def __init__(self, name:str, prog: MathematicalProgram, A:npt.NDArray, b:npt.NDArray, pot_type=PSD):
        # Ax <= b
        m,n = A.shape
        assert m == len(b)
        super(PolytopeVertex, self).__init__(name, prog, n, pot_type)
        self.A = A # m x n
        self.b = b.reshape(m,1)
        self.m = m

    def make_multiplier_terms(self, lambda_e:npt.NDArray, left:bool):
        # NOTE: for function g(x) <= 0 and lambda >= 0, returns lambda.T g(x).
        # here g(x) = Ax - b.
        # returns a negative term.

        res = 0
        n = self.n
        m = self.m
        assert len(lambda_e) == m
        lambda_e.reshape(m)

        O_nn = np.zeros((n,n))
        O_n = np.zeros((n,1))

        # form a sequence of matrices
        for i in range(m):
            a = self.A[i].reshape(n, 1)
            b = self.b[i].reshape(1, 1)
            if left:
                m_mat = np.vstack((
                np.hstack( (-b,     a.T/2,      O_n.T ) ),
                np.hstack( (a/2, O_nn, O_nn ) ),
                np.hstack( (O_n, O_nn, O_nn ) ),
                ))
            else:
                m_mat = np.vstack((
                np.hstack( (-b,     O_n.T,      a.T/2 ) ),
                np.hstack( (O_n, O_nn, O_nn ) ),
                np.hstack( (a/2, O_nn, O_nn ) ),
                ))
            res += lambda_e[i] * m_mat
        return res
    
    def multiplier_deg(self):
        return self.m


class BoxVertex(PolytopeVertex):
    def __init__(self, name:str, prog: MathematicalProgram, lb:npt.NDArray, ub:npt.NDArray, pot_type=PSD):
        # get A and b matrix rep
        n = len(lb)
        self.lb = lb.reshape(n,1)
        self.ub = ub.reshape(n,1)
        A = np.vstack((-np.eye(n), np.eye(n)))
        b = np.vstack((-self.lb, self.ub))
        assert n == len(ub)
        # super
        super(BoxVertex, self).__init__(name, prog, A, b, pot_type)
        # define center
        self.center = (self.lb+self.ub)/2.0

    def cost_at_point(self, x:npt.NDArray, solution:MathematicalProgramResult=None):
        x = x.reshape(self.n, 1)
        assert np.all(self.lb <= x) and np.all(x<= self.ub)
        if solution is None:
            return self.evaluate_partial_potential_at_point(x)
        else: 
            Q, r, s = solution.GetSolution(self.Q), solution.GetSolution(self.r), solution.GetSolution(self.s)
            return (x.T @ Q @ x + 2*r.dot(x) + s)[0,0]

    def cost_at_center(self, solution:MathematicalProgramResult = None):
        x = self.center
        return self.cost_at_point(x, solution)

    def cost_of_integral(self):
        return self.cost_of_integral_over_a_box(self.lb.reshape(self.n), self.ub.reshape(self.n))

    
class EllipsoidVertex(Vertex):
    def __init__(self, name:str, prog: MathematicalProgram, center:npt.NDArray, B:npt.NDArray, pot_type=PSD):
        n = len(center)
        assert n == B.shape[0] == B.shape[1]
        super(EllipsoidVertex, self).__init__(name, prog, n, pot_type)
        # {Bu + center | |u|_2 <= 1}
        self.B = B
        self.center = center.reshape(n,1)
        # {x | (x-center).T A.T A (x-center) <= 1 }
        self.A = np.linalg.inv(B)
        # {x | (x-center).T G (x-center) <= 1 }
        self.G = self.A.T @ self.A

    def cost_at_center(self, solution:MathematicalProgramResult = None):
        x = self.center
        return self.cost_at_point(x, solution)
    
    def make_multiplier_terms(self, lambda_e:npt.NDArray, left: bool):
        # NOTE: for function g(x) <= 0 and lambda >= 0, returns lambda.T g(x).
        # here g(x) = (x-c).T G (x-c) - 1.
        # returns a negative term.

        n = self.n
        assert len(lambda_e) == 1
        lambda_e.reshape(1)

        c, G = self.center, self.G

        O_nn = np.zeros((n,n))
        O_n = np.zeros((n,1))

        if left:
            m_mat = np.vstack((
                np.hstack(( c.T @ G @ c-1, -c.T @ G, O_n.T )),
                np.hstack(( -G @ c, G, O_nn )),
                np.hstack(( O_n,  O_nn, O_nn ))
            ))
        else:
            m_mat = np.vstack((
                np.hstack(( c.T @ G @ c-1, O_n.T, -c.T @ G )),
                np.hstack(( O_n,    O_nn, O_nn )),
                np.hstack(( -G @ c, O_nn,  G   ))
            ))
        res = m_mat * lambda_e[0]
        return res
    
    def multiplier_deg(self):
        return 1