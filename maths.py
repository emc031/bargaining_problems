
from math import log10
import numpy as np
import scipy.optimize as optimizer
from typing import Callable

"""
naming convention:

utility_A = payoff for team aligned
utility_U = payoff for team unaligned

action_A = proportion of A's resources spent on X. 0 < action_A < 1
action_U = proportion of U's resources spent on X. 0 < action_U < 1
"""

N = 10 ** 22 # amount of resources avaliable to each team
optimization_algo = 'L-BFGS-B' # algorithm for finding solutions to bargaining problems
initial_action_profile_guess = [0.1, 0.9]
delta = 0.01 # change in m or p to be depicted in heatmaps

#### utilities given action profiles ####

def specialLog(x):
    return 0 if x < 1 else log10(x)
##

def utility_A_from_actions(m: float,
                           action_A: float,
                           action_U: float) -> float:
    """ 
    Computes & returns utility for team aligned given 
    action profile = (action_A, action_U)
    """
    return m * specialLog(N * action_A) + (1 - m) * specialLog(N * action_U)
##


def utility_U_from_actions(m: float,
                           action_A: float,
                           action_U: float) -> float:
    """ 
    Computes & returns utilities for team unaligned given action profile
    = (action_A, action_U)
    """
    return m * specialLog(N * (1 - action_A)) + (1 - m) * specialLog(N * (1 - action_U))
##

#### welfare functions ####

def nash_welfare_function(m: float,
                          action_A: float,
                          action_U: float,
                          bargaining_failure_utility_A: float,
                          bargaining_failure_utility_U: float) -> float:
    """
    returns the evaluation of the nash welfare function given an
    action profile and bargaining failure utilities
    """
    return (utility_A_from_actions(m, action_A, action_U) - bargaining_failure_utility_A) * \
        (utility_U_from_actions(m, action_A, action_U) - bargaining_failure_utility_U)
##


def ks_welfare_function(m: float,
                        action_A: float,
                        action_U: float,
                        bargaining_failure_utility_A: float,
                        bargaining_failure_utility_U: float) -> float:
    """
    returns the evaluation of a function that is maximized in the 
    Kalai–Smorodinsky bargaining solution given an
    action profile and bargaining failure utilities.

    The two constraints, pareto optimality and the ks constraint, are accounted for with two terms in the welfare function.

    pareto optimality is acheived when (utility_A + utility_U) can no longer be
    improved by shifting actions, therefore pareto optimality is at the maximum
    of (utility_A + utility_U).

    the ks constraint is
    (u1 - d1)/(u2 - d2) = (b1 - d1)/(b2 - d2)
    where u1/2 are the two utilities, d1/2 are the failure payoffs,
    and b1/2 are the best possible payoffs.
    this equation is satisfied when 
    - ((u1 - d1)(b2 - d2) - (b1 - d1)(u2 - d2))^2 is maximized.
    """
    def utility_A_moreargs(m, action_A, action_U, **extras):
        return utility_A_from_actions(m, action_A, action_U)
    best_action_profile_for_A = \
        find_bargaining_solution(m = m,
                                 welfare_function = utility_A_moreargs,
                                 bargaining_failure_utility_A = 0,
                                 bargaining_failure_utility_U = 0)
    best_utility_A = utility_A_from_actions(m = m,
                                            action_A = best_action_profile_for_A[0],
                                            action_U = best_action_profile_for_A[1])

    def utility_U_moreargs(m, action_A, action_U, **extras):
        return utility_U_from_actions(m, action_A, action_U)
    best_action_profile_for_U = find_bargaining_solution(
        m = m,
        welfare_function = utility_U_moreargs,
        bargaining_failure_utility_A = 0,
        bargaining_failure_utility_U = 0)
    best_utility_U = utility_U_from_actions(m = m,
                                            action_A = best_action_profile_for_U[0],
                                            action_U = best_action_profile_for_U[1])

    utility_A = utility_A_from_actions(m, action_A, action_U)
    utility_U = utility_U_from_actions(m, action_A, action_U)

    line_constraint_lhs = \
        (utility_A - bargaining_failure_utility_A) * \
        (best_utility_U - bargaining_failure_utility_U)
    line_constraint_rhs = \
        (best_utility_A - bargaining_failure_utility_A) * \
        (utility_U - bargaining_failure_utility_U)
    line_constraint = - (line_constraint_lhs - line_constraint_rhs) ** 2

    pareto_constraint = - (utility_A + utility_U)
    return line_constraint + pareto_constraint
##


#### computing bargaining solutions & associated funcs ####


def find_bargaining_solution(m: float,
                             welfare_function: Callable,
                             bargaining_failure_utility_A: float,
                             bargaining_failure_utility_U: float) -> tuple:
    """
    returns the action profile (as a tuple of 2 floats - (action_A, action_U))
    for a bargaining solution according to welfare_function

    bargaining_failure_utilities is a tuple of 2 floats representing 
    the utilities when bargaining fails
    """
    def negative_welfare(action_profile):
        action_A = action_profile[0]
        action_U = action_profile[1]
        welfare = welfare_function(m = m,
                                   action_A = action_A,
                                   action_U = action_U,
                                   bargaining_failure_utility_A = bargaining_failure_utility_A,
                                   bargaining_failure_utility_U = bargaining_failure_utility_U)
        return - welfare

    initial_guess = np.array(initial_action_profile_guess)
    search_bounds = np.array([(0., 1.), (0., 1.)])
    solution = optimizer.minimize(fun = negative_welfare,
                                  x0 = initial_guess,
                                  bounds = search_bounds,
                                  method = optimization_algo)
    return solution.x
##


def expected_utility_A(m: float,
                       p: float,
                       welfare_function: Callable,
                       bargaining_failure_utility_A: Callable,
                       bargaining_failure_utility_U: Callable) -> float:
    """ 
    given a point on (m, p), and a welfare function,
    compute and return the expected utility for team aligned.

    bargaining_failure_utility_A/U are functions of m.
    """
    bargaining_solution = \
        find_bargaining_solution(m = m,
                                 welfare_function = nash_welfare_function,
                                 bargaining_failure_utility_A = bargaining_failure_utility_A(m),
                                 bargaining_failure_utility_U = bargaining_failure_utility_U(m))
    action_profile = bargaining_solution
    action_A = action_profile[0]
    action_U = action_profile[1]
    bargaining_success_utility_A = utility_A_from_actions(m = m,
                                                          action_A = action_A,
                                                          action_U = action_U)

    return p * bargaining_success_utility_A + \
        (1 - p) * bargaining_failure_utility_A(m)
##

def expected_utility_A_delta_m(m: float,
                               p: float,
                               welfare_function: Callable,
                               bargaining_failure_utility_A: Callable,
                               bargaining_failure_utility_U: Callable) -> float:
    """ 
    returns expected_utility_A((1 + delta)m, p) - expected_utility_A(m, p)
    delta corresponds to a change in m/(1-m) set at the top of this file
    """
    m_ratio = m/(1 - m)
    shifted_m_ratio = m_ratio * (1 + delta)
    shifted_m = shifted_m_ratio / (1 + shifted_m_ratio)
    
    utility_A = expected_utility_A(m, p, welfare_function,
                                   bargaining_failure_utility_A,
                                   bargaining_failure_utility_U)
    shifted_utility_A = expected_utility_A(shifted_m, p, welfare_function,
                                           bargaining_failure_utility_A,
                                           bargaining_failure_utility_U)
    return shifted_utility_A - utility_A
##


def expected_utility_A_delta_p(m: float,
                               p: float,
                               welfare_function: Callable,
                               bargaining_failure_utility_A: Callable,
                               bargaining_failure_utility_U: Callable) -> float:
    """ 
    returns expected_utility_A(m, (1 + delta)p) - expected_utility_A(m, p)
    delta corresponds to a change in m/(1-m) set at the top of this file
    """
    p_ratio = p/(1 - p)
    shifted_p_ratio = p_ratio * (1 + delta)
    shifted_p = shifted_p_ratio / (1 + shifted_p_ratio)
    
    utility_A = expected_utility_A(m, p, welfare_function,
                                   bargaining_failure_utility_A,
                                   bargaining_failure_utility_U)
    shifted_utility_A = expected_utility_A(m, shifted_p, welfare_function,
                                           bargaining_failure_utility_A,
                                           bargaining_failure_utility_U)
    return shifted_utility_A - utility_A
##


def expected_utility_A_shift_ratio(m: float,
                                   p: float,
                                   welfare_function: Callable,
                                   bargaining_failure_utility_A: Callable,
                                   bargaining_failure_utility_U: Callable) -> float:
    """
    returns the ratio of change in utility_A due to fractional change in p/(1-p)
    over fractional change in m/(1-m)
    """
    shift_m = expected_utility_A_delta_m(m, p, welfare_function,
                                         bargaining_failure_utility_A,
                                         bargaining_failure_utility_U)
    shift_p = expected_utility_A_delta_p(m, p, welfare_function,
                                         bargaining_failure_utility_A,
                                         bargaining_failure_utility_U)
    return shift_p / shift_m
##


def expected_utility_A_max_shift(m: float,
                                 p: float,
                                 welfare_function: Callable,
                                 bargaining_failure_utility_A: Callable,
                                 bargaining_failure_utility_U: Callable) -> float:
    """
    returns the ratio of change in utility_A due to fractional change in p/(1-p)
    over fractional change in m/(1-m)
    """
    shift_m = expected_utility_A_delta_m(m, p, welfare_function,
                                         bargaining_failure_utility_A,
                                         bargaining_failure_utility_U)
    shift_p = expected_utility_A_delta_p(m, p, welfare_function,
                                         bargaining_failure_utility_A,
                                         bargaining_failure_utility_U)
    return max(shift_m, shift_p)
##
