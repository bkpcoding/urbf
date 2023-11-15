import numpy as np
import torch
import random
import itertools
from non_convex_testing.utils.utils import RBFLayer
import exputils as eu
from matplotlib import pyplot as plt
#train a curve fitting model, which fits a one dimensional sin function

def train(input, output):
    # split the input into training and testing data
    training_input = input[:int(len(input)*0.8)].reshape(-1,1)
    training_output = output[:int(len(output)*0.8)].reshape(-1,1)
    testing_input = input[int(len(input)*0.8):].reshape(-1,1)
    testing_output = output[int(len(output)*0.8):].reshape(-1,1)
    # create a model
    model = torch.nn.Sequential(
        torch.nn.Linear(1,2),
        torch.nn.ReLU(),
        torch.nn.Linear(2,1)
    )
    # define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # train the model
    loss_list = []
    for epoch in range(10):
        # forward pass
        y_pred = model(training_input)
        # compute loss
        loss = criterion(y_pred, training_output)
        loss_list.append(loss.item())
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
    # evaluate the model
    with torch.no_grad():
        y_pred = model(testing_input)
        loss = criterion(y_pred, testing_output)
        print(f"Epoch {epoch+1}/1000: Loss = {loss.item():.4f}")
    # plot the loss
    plt.plot(loss_list)
    plt.show()
    return model

if __name__ == "__main__":
    eu.misc.seed(1747)
    input = torch.arange(0, 100, 0.1)
    output = torch.sin(input)
    model = train(input, output)
    plt.plot(input.numpy(), output.numpy(), "o")
    plt.show()
    # evaluate the model