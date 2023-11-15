# implementation of the Griewank function
import torch
import numpy as np
def Griewank(x):
    """
    Griewank function
    """
    return sum([(x[i]**2)/4000 for i in range(len(x))]) - np.prod([np.cos(x[i]/np.sqrt(i+1)) for i in range(len(x))]) + 1