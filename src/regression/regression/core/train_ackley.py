import sys
from black import out
import matplotlib
sys.path.append("/home/sagar/inria/code/dqn_rbf")
sys.path.append("/home/sagar/inria/code/exputils")
sys.path.append("/home/sagar/inria/code/non_convex_testing")
import torch
import torch.nn as nn
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
import exputils as eu
import exputils.data.logging as log
from non_convex_testing.functions.gaussians import gaussians, gaussians1
from non_convex_testing.functions.many_local_minima import ackley_function
from non_convex_testing.functions.staircase_like_functions import staircase
from non_convex_testing.functions.bowl_like_functions import styblinski_function
from non_convex_testing.utils.utils import draw_2d_heatmap, draw_3d_plot, plot_gaussian
from non_convex_testing.utils.utils import create_batched_dataset
from non_convex_testing.utils.utils import build_mlp_model, build_rbf_model, mlp_rbf_network


def default_config():
    return eu.AttrDict(
        algorithm = "mlp_rbf", n_neurons_per_input=10, is_trainable=True, ranges=[-5.0, 5.0], m=5,
        name = "mlp_rbf_network", model = [16, 32],
    )


def train(input, output, config):
    n = input.shape[0]
    training_dataset = torch.tensor(input[:int(0.75*n), :], dtype= torch.float32)
    training_output = torch.tensor(output[:int(0.75*n), :], dtype= torch.float32)
    validation_dataset = torch.tensor(input[int(0.75*n): int(0.9*n), :], dtype = torch.float32)
    validation_output =  torch.tensor(output[int(0.75*n): int(0.9*n), :], dtype= torch.float32)
    test_dataset = torch.tensor(input[int(0.9*n):, :], dtype=torch.float32)
    test_output =  torch.tensor(output[int(0.9*n):,  :], dtype= torch.float32)
    # create a network
    if config.algorithm == "mlp":
        network = build_mlp_model(2, config.model, 1)
    # rbf network
    elif config.algorithm == "rbf":
        rbf, network = build_rbf_model(2, config, config.model, 1)
    # mlp + rbf network
    elif config.algorithm == "mlp_rbf":
        new_network = build_mlp_model(2, config.model, 1)
        rbf, new_network_rbf = build_rbf_model(2, config, [32], 1)
        network = mlp_rbf_network(new_network, new_network_rbf, 1)
    optimizer = torch.optim.Adam(
        lr=0.00025, params=network.parameters()
    )
    epoches = 200
    training_loss_list = []
    smoothed_training_loss_list = []
    criterion = nn.MSELoss()
    iterations = 100
    validation_loss_list = []
    is_true = True
    validation_loss_test = 0
    for epoch in range(epoches):
        for i in range(iterations):
            input_batch, output_batch = create_batched_dataset(training_dataset, training_output, batch_size= 256)
            network.zero_grad()
            output = network(input_batch)
            loss = criterion(output, output_batch)
            loss.backward()
            optimizer.step()
            training_loss_list.append(loss.item())
        print("Epoch: " + str(epoch) + " Loss: " + str(training_loss_list[-100]))
        smoothed_training_loss_list.append(np.mean(training_loss_list[-100]))
        log.add_scalar("Loss", np.mean(training_loss_list[-100:]))
        #checking validation loss
        predicted_validation_output = network(validation_dataset)
        validation_loss = criterion(validation_output, predicted_validation_output)
        validation_loss_list.append(validation_loss.item())
        print("validation loss:", validation_loss)
        log.add_scalar("validation_loss", validation_loss.item())
        if np.mean(validation_loss_list[-20:]) > np.mean(validation_loss_list[-40:-20]) and len(validation_loss_list) > 40 and is_true:
            validation_predicted_test_output = network(test_dataset)
            validation_loss_test = criterion(test_output, validation_predicted_test_output)
            log.add_scalar("EpochValidationBreak", epoch)
            log.add_scalar("TestLossValidation", validation_loss_test.item())
            log.add_scalar("ValidationLossBreak", validation_loss.item())
            is_true = False 

    # testing the network
    test_loss_list = []
    predicted_test_output = network(test_dataset)
    test_loss = criterion(test_output, predicted_test_output)
    test_loss_list.append(test_loss)
    print("Testing Loss: ", test_loss)

    #log.add_scalar("TestLoss", test_loss)

    if config.algorithm == "rbf" or config.algorithm == "mlp_rbf":
        log.add_value("rbf_peaks", rbf.peaks.detach().numpy())
        log.add_value("rbf_sigmas", rbf.sigmas.detach().numpy())
    tensor_input = torch.tensor(input, dtype=torch.float32) 
    predicted_output = network(tensor_input).detach().numpy()
    log.add_value("predicted_output", predicted_output)
    return predicted_output
    


def run_task(config = None, **kwargs):
    config = eu.combine_dicts(kwargs, config, default_config())
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    input = list(
            itertools.product(x, y)
        )
    input = np.asarray(input)
    m = config.m

    output = ackley_function(x, y).reshape((10000,1))
    # z = ackley_function(x, y)

    # create_dataset(x,y,z)
    predicted_output = train(input, output, config)
    log.add_value("input_vector", input)
    log.add_value("actual_output", output)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    x, y = np.meshgrid(x, y)
    predicted_output = predicted_output.reshape((100, 100))
    #print(x.shape, y.shape, predicted_output.shape)
    ax.plot_surface(x, y, predicted_output, cmap=cm.turbo, linewidth=0, antialiased=False)

    ax = fig.add_subplot(1, 2, 2, projection = "3d")
    output = output.reshape((100, 100))
    ax.plot_surface(x, y, output, cmap=cm.turbo, linewidth=0, antialiased=False)
    plt.show()
    log.save()
    return log

log = run_task()