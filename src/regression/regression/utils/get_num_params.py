from distutils.command.build import build
from statistics import mode
import torch
from torchinfo import summary
import math
import random
from torch import nn
from non_convex_testing.utils.utils import build_mlp, build_mlp_model, build_rbf_model, build_mrbf_model

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = build_mlp_model(2, [64, 84, 128, 32], 1)
print(get_num_params(model))
