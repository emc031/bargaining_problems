
import math
import numpy as np

""" 
there are 3 spaces, each represented by a unit square:
  input space = {(p, m)}
  action profile space = {(action_A, action_U)}
  payoff space = {(utility_A, utility_U)}
  
(action_A/U defined s.t. resources spent by team A/U on X = N * action_A/U
 & resources spent by team A/U on Y = N * (1 - action_A/U))
"""

N = 10 ** 22 # amount of resources avaliable to each team
delta = 0.01 # proportional change in p/(1 - p) and m/(1 - m) in heatmaps 2 & 3
optimization_algo = 'L-BFGS-B' # algorithm for finding solutions to bargaining problems

def specialLog(x):
    return 0 if x < 1 else math.log10(x)

def utility_A_from_actions(action_A: float,
                           action_U: float) -> float:
    """ 
    Computes & returns utilities for the two teams given action_profile = (action_A, action_U)
    """
    return m * specialLog(N * action_A) + (1 - m) * specialLog(N * action_U)

def utility_U_from_actions(action_A: float,
                           action_U: float) -> float:
    """ 
    Computes & returns utilities for the two teams given action_profile = (action_A, action_U)
    """
    return m * specialLog(N * (1 - action_A)) + (1 - m) * specialLog(N * (1 - action_U))



def nash_welfare_function(m: float,
                          action_A: float,
                          action_U: float,
                          bargaining_failure_payoff_A: float,
                          bargaining_failure_payoff_U: float) -> float:
    """
    returns the evaluation of the nash welfare function given an
    action profile and bargaining failure payoffs
    """
    return (utility_A_from_actions(action_A, action_U) - bargaining_failure_payoff_A) * \
        (utility_U_from_actions(action_A, action_U) - bargaining_failure_payoff_U)



def bargaining_solution(m: float,
                        welfare_function: func,
                        bargaining_failure_payoff_A: float,
                        bargaining_failure_payoff_U: float) -> tuple:
    """
    returns the action profile (as a tuple of 2 floats - (action_A, action_U))
    for a bargaining solution according to welfare function 'func'

    bargaining_failure_payoffs is a tuple of 2 floats representing 
    the payoffs when bargaining fails
    """
    def negative_welfare(action_profile):
        action_A = action_profile[0]
        action_U = action_profile[1]
        welfare = welfare_function(m = m,
                                   action_A = action_A,
                                   action_U = action_U,
                                   bargaining_failure_payoff_A = bargaining_failure_payoff_A,
                                   bargaining_failure_payoff_U = bargaining_failure_payoff_U)
        return - welfare

    initial_guess = np.array([0.5, 0.5])
    search_bounds = ((0., 1.), (0., 1.)),
    optimized_action_profile = scipy.optimize.minimize(func = negative_welfare,
                                                       x0 = initial_guess,
                                                       bounds = search_bounds,
                                                       method = optimization_algo)
    return optimized_action_profile



def expected_utility_A(m: float,
                       p: float,
                       welfare_function: func,
                       bargaining_failure_payoff_A: float,
                       bargaining_failure_payoff_U: float) -> float:
    """ given a point on (m, p), and a welfare function, 
    compute and return the expected utility for team aligned """
    
    bargaining_solution_action_profile = \
        bargaining_solution(m = m,
                            welfare_function = nash_welfare_function,
                            bargaining_failure_payoff_A = bargaining_failure_payoff_A,
                            bargaining_failure_payoff_U = bargaining_failure_payoff_U)
    action_A = bargaining_solution_action_profile[0]
    action_U = bargaining_solution_action_profile[1]
    utility_A_bargaining_success = utility_A_from_actions(action_A, action_U)
    
    utility_A_bargaining_failure = bargaining_failure_payoff_A

    return p * utility_A_bargaining_success + \
        (1 - p) * utility_A_bargaining_failure
