# code for training a michalewicz function

import torch
import torch.nn as nn
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from dqn_rbf.rbf_layer import RBFLayer

x1 = np.arange(0, 4.0, 0.01)
x2 = np.arange(0, 4.0, 0.01)
m = 10

# michalewicz function
y1 = np.sin(x1) * np.power(np.sin(np.power(x1, 2) / np.pi), 2 * m)
y2 = np.sin(x2) * np.power(np.sin(2 * np.power(x2, 2) / np.pi), 2 * m)
y = y1 + y2
network = torch.nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)
)
# network = torch.nn.Sequential(RBFLayer(2, 20, ranges = [0.0, 4.0]), nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
epoches = 100
learning_rate = 0.01
random.seed(0)
indices = np.random.permutation(len(x1))
x1_training = x1[indices[: int(0.8 * len(x1))]]
x2_training = x2[indices[: int(0.8 * len(x1))]]
y_training = y[indices[: int(0.8 * len(x1))]]

# testing dataset
x1_testing = x1[indices[int(0.8 * len(x1)) :]]
x2_testing = x2[indices[int(0.8 * len(x1)) :]]
y_testing = y[indices[int(0.8 * len(x1)) :]]


# supervised learning the function y = f(x1, x2)
for epoch in range(epoches):
    for i in range(len(x1_training)):
        x = torch.tensor([x1_training[i], x2_training[i]], dtype=torch.float32)
        y = torch.tensor([y_training[i]], dtype=torch.float32)
        network.zero_grad()
        output = network(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        for param in network.parameters():
            param.data -= learning_rate * param.grad.data
    print("epoch:", epoch, "loss:", loss.item())


# testing the inputs
x1_testing = torch.tensor(x1_testing, dtype=torch.float32)
x2_testing = torch.tensor(x2_testing, dtype=torch.float32)
y_testing = torch.tensor(y_testing)
x_testing = torch.stack([x1_testing, x2_testing], dim=1)
y_testing = torch.stack([y_testing], dim=1)
print(x_testing.shape)
output = network(x_testing)
loss = torch.nn.functional.mse_loss(output, y_testing)
print("testing loss:", loss.item())


y = y1 + y2
y_real = np.tile(y, 400)
y_real = np.reshape(y_real, (400, 400))
x1_full = torch.tensor(x1, dtype=torch.float32)
x2_full = torch.tensor(x2, dtype=torch.float32)
x_full = torch.stack([x1_full, x2_full], dim=1)
y_pred = network(x_full)
y_pred = y_pred.detach().numpy()
y_pred = np.tile(y_pred, 400)
y_pred = np.reshape(y_pred, (400, 400))
x1, x2 = np.meshgrid(x1, x2)
fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.view_init(elev=20.0, azim=45)
surf = ax.plot_surface(
    x1, x2, y_real, rstride=1, cstride=1, cmap=cm.turbo, linewidth=0, antialiased=False
)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
surf2 = ax2.plot_surface(
    x1, x2, y_pred, rstride=1, cstride=1, cmap=cm.turbo, linewidth=0, antialiased=False
)
plt.show()
