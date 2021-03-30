
from maths import *
from heatmaps import *

quantity_to_plot = 'expected_utility_A_max_shift'
# choices = 'expected_utility_A',
#           'expected_utility_A_shift_ratio',
#           'expected_utility_A_max_shift',

welfare_function = 'nash'
# choices = 'nash', 'ks'(

disagreement_outcome = 0
# choices = 0 (aligned spends all on X, unaligned spends all on Y)
#           1 (both get utility 0)
#           2 (same as 1, swapped and * -1)


#### get function for quantity to plot  ####

if quantity_to_plot == 'expected_utility_A':
    plot_title = 'expected payoff for team aligned'
    plot_fill_function = expected_utility_A
if quantity_to_plot == 'expected_utility_A_shift_ratio':
    plot_title = f'expected payoff gain from {delta * 100}% increase in p/(1-p) / m/(1-m)'
    plot_fill_function = expected_utility_A_shift_ratio
if quantity_to_plot == 'expected_utility_A_max_shift':
    plot_title = f'max expected payoff gain from {delta * 100}% change in p/(1-p) or m/(1-m)'
    plot_fill_function = expected_utility_A_max_shift


#### decide welfare function ####

if welfare_function == 'nash':
    welfare_func = nash_welfare_function
if welfare_function == 'ks':
    welfare_func = ks_welfare_function

#### decide bargaining failure outcomes ####
    
if disagreement_outcome == 0:
    action_A = 1
    action_U = 0
    bargaining_failure_utility_A = \
        lambda m: utility_A_from_actions(m,
                                         action_A,
                                         action_U)
    bargaining_failure_utility_U = \
        lambda m: utility_U_from_actions(m,
                                         action_A,
                                         action_U)
    
if disagreement_outcome == 1:
    bargaining_failure_utility_A = lambda m: 0
    bargaining_failure_utility_U = lambda m: 0

if disagreement_outcome == 2:
    action_A = 1
    action_U = 0
    bargaining_failure_utility_A = \
        lambda m: - utility_U_from_actions(m,
                                           action_A,
                                           action_U)
    bargaining_failure_utility_U = \
        lambda m: - utility_A_from_actions(m,
                                           action_A,
                                           action_U)

#### make heatmap ####

def heatmap_fill_func(m: float,
                      p: float):
    return plot_fill_function(
        m, p,
        welfare_function = welfare_func,
        bargaining_failure_utility_A = bargaining_failure_utility_A,
        bargaining_failure_utility_U = bargaining_failure_utility_U)

make_heatmap(heatmap_fill_func, plot_title)

# TODO
# * check for out-of-date docstrings
# * linting
# * 'with more time I would'
# * refactor (if time)
