import sys
sys.path.append('/home/sagar/inria/code/non_convex_testing')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from staircase_like_functions import staircase
from non_convex_testing.utils.utils import plot_gaussian
def check_function():
    means = np.array([[0], [-1], [3]])
    stds = np.array([[1], [1], [1]])
    plot_gaussian(means, stds)

if __name__ == "__main__":
    check_function()