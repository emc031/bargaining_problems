
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

resolution = 12
save_heatmap = True
show_heatmap = True
result_file = 'bargaining_heatmap.png'

min_m_ratio = 10 ** -5 # minimum of m/(1-m)
max_m_ratio = 10 ** +5 # maximum of m/(1-m)
min_p = 10 ** -3
max_p = 1 - 10 ** -3
    
def make_heatmap(fill_func: Callable,
                 plot_title: str = None,
                 m_values: list = None,
                 p_values: list = None) -> None:
    """
    evaluates val_func (a function taking in (m, p))
    at each pair of m, p values in a mesh made from
    m_values and p_values.
    
    draws heatmap of results against m/(1-m) / p/(1-p).

    If m/p_values are none,
    use some sensible default values.
    """
    if m_values is None:
        m_values = default_m_values(min_m_ratio, max_m_ratio)
    if p_values is None:
        p_values = default_p_values(min_p, max_p)

    # compute fill values
    vals = np.array([[fill_func(m, p) for p in p_values]
                     for m in m_values])

    # make axis values
    x_values = p_values
    y_values = [m/(1-m) for m in m_values]

    x_labels = ['10^{:.1f}'.format(np.log10(v)) for v in x_values]
    y_labels = ['10^{:.1f}'.format(np.log10(v)) for v in y_values]
    x_labels[::2] = ['' for _ in x_labels[::2]]
    y_labels[::2] = ['' for _ in y_labels[::2]]

    # reverse y-axis so "up means bigger"
    y_labels.reverse()
    vals = np.flip(vals, axis = 0)

    make_heatmap_generic(vals = vals,
                         x_labels = x_labels,
                         y_labels = y_labels,
                         plot_title = plot_title)

def default_p_values(min_, max_):
    log_min = np.log10(min_)
    log_max = np.log10(max_)
    log_axis_values = np.linspace(log_min, log_max, num = resolution)
    axis_values = np.power(10, log_axis_values)
    return axis_values

def default_m_values(min_ratio, max_ratio):
    log_min_ratio = np.log10(min_ratio)
    log_max_ratio = np.log10(max_ratio)
    log_axis_ratios = np.linspace(log_min_ratio, log_max_ratio, num = resolution)
    axis_ratios = np.power(10, log_axis_ratios)
    axis_values = axis_ratios / (1 + axis_ratios)
    return axis_values

def make_heatmap_generic(vals: np.array,
                         x_labels: np.array,
                         y_labels: np.array,
                         plot_title: str = None) -> None:
    """ makes heatmap of vals (resolution * resolution array), labels
    the axes with x_labels & y_labels 
    (each a 1d array of length 'resolution') """
    resolution = len(x_labels)

    plt.imshow(vals, cmap='hot', interpolation='nearest')
    plt.xticks(np.arange(resolution), x_labels)
    plt.yticks(np.arange(resolution), y_labels)
    plt.colorbar()
    plt.title(plot_title)

    plt.xlabel('Prob(bargaining success)')
    plt.ylabel('Ratio of bargaining powers (team aligned)/(team unaligned)')

    if save_heatmap:
        plt.savefig(result_file)
    if show_heatmap:
        plt.show()

