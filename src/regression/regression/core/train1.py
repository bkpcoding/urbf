import sys
import matplotlib
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
from non_convex_testing.utils.utils import build_mlp_model, build_rbf_model, mlp_rbf_network, build_mrbf_model
from non_convex_testing.utils.utils import calc_error



def default_config():
    return eu.AttrDict(
        algorithm = "mlp", n_neurons_per_input=10, is_trainable=True, ranges=[-5.0, 5.0], m=3,
        name = "mlp_rbf_network", model = [16, 32], seed = 0,
    )


def train(input, output, config):
    n = input.shape[0]
    eu.misc.seed(config.seed)
    indices = random.sample(range(input.shape[0]), input.shape[0])
    training_dataset = torch.tensor(input[indices[int(0):int(0.75*len(indices))], :], dtype= torch.float32)
    training_output = torch.tensor(output[indices[int(0):int(0.75*len(indices))], :], dtype= torch.float32)
    validation_dataset = torch.tensor(input[indices[int(0.75*len(indices)): int(0.90*len(indices))], :], dtype = torch.float32)
    validation_output =  torch.tensor(output[indices[int(0.75*len(indices)): int(0.90*len(indices))], :], dtype= torch.float32)
    test_dataset = torch.tensor(input[indices[int(0.90*len(indices)):], :], dtype=torch.float32)
    test_output =  torch.tensor(output[indices[int(0.90*len(indices)):],  :], dtype= torch.float32)
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
    elif config.algorithm == "mrbf":
        network = build_mrbf_model(2, 1, config.model)
    # count the number of parameters of the network and print it
    num_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print("********************************")
    print("Number of parameters: ", num_params)
    print("********************************")
    optimizer = torch.optim.SGD(
        lr=0.0003, params=network.parameters()
    )
    epoches = 200
    training_loss_list = []
    smoothed_training_loss_list = []
    criterion = nn.MSELoss()
    iterations = 100
    validation_loss_list = []
    for epoch in range(epoches):
        for i in range(iterations):
            input_batch, output_batch = create_batched_dataset(training_dataset, training_output, batch_size= 256)
            network.zero_grad()
            network_output = network(input_batch)
            loss = criterion(network_output, output_batch)
            loss.backward()
            optimizer.step()
            training_loss_list.append(loss.item())
        print("Epoch: " + str(epoch) + " Loss: " + str(training_loss_list[-100]))
        smoothed_training_loss_list.append(np.mean(training_loss_list[-100]))
        log.add_scalar("Loss", np.mean(training_loss_list[-100:]))
        #checking validation loss
        predicted_validation_output = network(validation_dataset)
        validation_loss = criterion(predicted_validation_output, validation_output)
        validation_loss_list.append(validation_loss.item())
        print("validation loss:", validation_loss)
        log.add_scalar("validation_loss", validation_loss.item())
        if np.mean(validation_loss_list[-20:]) > np.mean(validation_loss_list[-40:-20]) and len(validation_loss_list) > 40:
            validation_predicted_test_output = network(test_dataset)
            test_loss = criterion(validation_predicted_test_output, test_output)
            log.add_scalar("EpochValidationBreak", epoch)
            log.add_scalar("TestLossValidation", test_loss.item())
            log.add_scalar("ValidationLossBreak", validation_loss.item())
            is_true = False      

    test_loss_list = []
    predicted_test_output = network(test_dataset)
    test_loss = criterion(predicted_test_output, test_output)
    test_loss_list.append(test_loss)
    print("Testing Loss: ", test_loss)
    #plt.plot(smoothed_training_loss_list)
    #plt.show()

    log.add_scalar("TestLoss", test_loss.detach().numpy())

    tensor_input = torch.tensor(input, dtype=torch.float32) 
    predicted_output = network(tensor_input).detach().numpy()
    log.add_value("predicted_output", predicted_output)
    #error_1 = calc_error(input, output, network, config)
    #log.add_scalar("Error", error_1)
    return predicted_output
    
def run_task(config = None, **kwargs):
    torch.set_num_threads(1)
    config = eu.combine_dicts(kwargs, config, default_config())
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    input = list(
            itertools.product(x, y)
        )
    input = np.asarray(input)
    m = config.m
    d = 2
    # mean = (np.random.rand(m, d) * 10) - 5

    #mean = np.array([[0, 0], [-1, -2], [2, 1], [3, -4], [-4, 4]])
    #std = np.array([[0.1, 0.1], [0.25, 0.25], [0.5, 0.5], [0.3, 0.3], [0.5, 0.5]])
    # plot_2d_gaussian(mean, std)
    #mean = np.array([[0, 0]])
    #std = np.array([[1, 1]])
    # create a pool and submit multiple jobs to the cpu's
    #output = gaussians1(x, y, mean, std, config).reshape((10000,1))
    # create a pool and submit multiple jobs to the cpu's
    # x, y, z = create_function()
    # z = ackley_function(x, y)
    #for i in range(len(x)):
    #    for j in range(len(y)):
    #        z[i, j] = staircase(x[i], y[j])
    output = staircase(x, y, config).reshape((10000, 1))
    # create_dataset(x,y,z)
    predicted_output = train(input, output, config)
    log.add_value("input_vector", input)
    log.add_value("actual_output", output)
    log.save()
    return log
