import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from rl_maze.models.rbf_layer import RBFLayer

class urbf(nn.Module):
    def __init__(self, input_size, output_size, config = None):
        super(urbf, self).__init__()
        self.config = config
        self.rbf_layer = RBFLayer(input_size, output_size, self.config)
        self.layers = []
        for i, hidden_unit in enumerate(self.config.hidden_units):
            if i < len(self.config.hidden_units) - 1:
                self.layers.append(nn.Linear(hidden_unit, self.config.hidden_units[i + 1]))
                self.layers.append(nn.ReLU())                
        self.layers.append(nn.Linear(self.config.hidden_units[-1], output_size))   

    def forward(self, x):
        x = self.rbf_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x
         