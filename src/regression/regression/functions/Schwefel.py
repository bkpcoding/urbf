import torch
import torch.nn as nn
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from dqn_rbf.rbf_layer import RBFLayer

x1 = np.arange(-500.0, 500.0, 1)
x2 = np.arange(-500.0, 500.0, 1)
# creating a schwefel function and training the network on it
# number of dimensions of the function
d = 2
