
import matplotlib.pyplot as plt
import numpy as np

resolution = 16

def make_heatmap(vals: np.array,
                 x_labels: np.array,
                 y_labels: np.array) -> None:
    """ makes heatmap of vals (resolution * resolution array), labels
    the axes with x_labels & y_labels 
    (1d array of length 'resolution') """

    plt.imshow(vals, cmap='hot', interpolation='nearest')
    plt.xticks(np.arange(resolution), x_labels)
    plt.yticks(np.arange(resolution), y_labels)
    plt.show()
    
make_heatmap(vals = np.random.random((resolution, resolution)),
             x_labels = range(10, resolution + 10),
             y_labels = range(10, resolution + 10))

# TODO
# * make plots
# * check for out-of-date docstrings
# * linting
# * 'with more time I would'
# * refactor (if time)
