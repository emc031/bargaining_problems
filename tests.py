
from thing import *
import numpy as np

### test utils ###

def sloppy_pareto_optimal(m,
                          action_A,
                          action_U):
    """ sloppy test, returns false if
    action profile is definately not pareto optimal """
    utility_A = utility_A_from_actions(
        m = m,
        action_A = action_A,
        action_U = action_U)
    utility_U = utility_U_from_actions(
        m = m,
        action_A = action_A,
        action_U = action_U)

    shifts = [(0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)]
    for shift in shifts:
        shifted_action_A = action_A + shift[0]
        shifted_action_U = action_U + shift[1]
        shifted_utility_A = utility_A_from_actions(
            m = m,
            action_A = shifted_action_A,
            action_U = shifted_action_U)
        shifted_utility_U = utility_U_from_actions(
            m = m,
            action_A = shifted_action_A,
            action_U = shifted_action_U)
        print(f'action_A = {action_A}')
        print(f'action_U = {action_U}')
        print(f'utility_A = {utility_A}')
        print(f'utility_U = {utility_U}')
        print(f'shifted_utility_A = {shifted_utility_A}')
        print(f'shifted_utility_U = {shifted_utility_U}')

        if shifted_utility_A >= utility_A and \
           shifted_utility_U >= utility_U:
            
            if not shifted_utility_A == utility_A and \
               shifted_utility_U == utility_U:
                
                if 0 <= shifted_action_A <= 1 and \
                   0 <= shifted_action_U <= 1:
                    
                    return False
    return True


### bargaining solutions ###

# expect nash bargaining solution to result in both spending half their resources on X & Y, independent of m
bargaining_solution = find_bargaining_solution(
    m = 0.2,
    welfare_function = nash_welfare_function,
    bargaining_failure_utility_A = 0,
    bargaining_failure_utility_U = 0
)

action_A, action_U = bargaining_solution
np.testing.assert_almost_equal(action_A, 0.5, decimal = 3)
np.testing.assert_almost_equal(action_U, 0.5, decimal = 3)

# if welfare function is weighted to team aligned, expect solution
# to be correspondingly weighted
def unfair_welfare_func(m, action_A, action_U,
                        bargaining_failure_utility_A,
                        bargaining_failure_utility_U):
    return utility_A_from_actions(m, action_A, action_U) - \
        bargaining_failure_utility_A

bargaining_solution = find_bargaining_solution(
    m = 0.2,
    welfare_function = unfair_welfare_func,
    bargaining_failure_utility_A = 0,
    bargaining_failure_utility_U = 0
)

action_A, action_U = bargaining_solution
np.testing.assert_almost_equal(action_A, 1.0, decimal = 3)
np.testing.assert_almost_equal(action_U, 1.0, decimal = 3)


# test KS bargaining solution: expect on the KS-line (u_A = u_U) and pareto optimal.
m = 0.51
bargaining_solution = find_bargaining_solution(
    m = m,
    welfare_function = ks_welfare_function,
    bargaining_failure_utility_A = 0,
    bargaining_failure_utility_U = 0
)

action_A, action_U = bargaining_solution
utility_A = utility_A_from_actions(m = m,
                                   action_A = action_A,
                                   action_U = action_U)
utility_U = utility_U_from_actions(m = m,
                                   action_A = action_A,
                                   action_U = action_U)

np.testing.assert_almost_equal(utility_A, utility_U, decimal = 2)
assert sloppy_pareto_optimal(m, action_A, action_U)
