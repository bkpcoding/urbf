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
from non_convex_testing.functions.xy import xy
from non_convex_testing.functions.gaussians import gaussians, gaussians1
from non_convex_testing.functions.many_local_minima import ackley_function
from non_convex_testing.functions.staircase_like_functions import staircase
from non_convex_testing.functions.bowl_like_functions import styblinski_function
from non_convex_testing.utils.utils import build_mrbf_model, draw_2d_heatmap, draw_3d_plot, plot_gaussian
from non_convex_testing.utils.utils import create_batched_dataset
from non_convex_testing.utils.utils import build_mlp_model, build_rbf_model, build_mlp, mlp_rbf_network
from non_convex_testing.utils.utils import calc_error
from non_convex_testing.functions.polynomials import poly
from non_convex_testing.utils.multi_rbf_layer import MRBF
from non_convex_testing.utils.get_num_params import get_num_params
def default_config():
    return eu.AttrDict(
#        seed = 42,
#        algorithm = "mlp",
#        complexity = 1,
#        depth_mlp = 1,
#        width_mlp = 1,
#        depth_rbf = 1,
#        width_rbf = 1,
#        n_neurons_per_input = 1,
        algorithm = "mlp", n_neurons_per_input=5, is_trainable=True, ranges=[-5.0, 5.0], m=5,
        model = [32, 64, 128], k = 6, seed = 1,
    )
# m: degree of the polynomial
# k: number of variables
# n: number of terms in the polynomial 

def plot_rbf_mlp(mlp_network, rbf_network, network, input, output, config):
    x = np.arange(-2, 2, 0.04)
    y = np.arange(-2, 2, 0.04)
    mlp_output = (mlp_network(input)*network.linear_layer.weight.data[0, 0] + network.linear_layer.bias.data[0]).detach().numpy()
    rbf_output = (rbf_network(input)*network.linear_layer.weight.data[0, 1] + network.linear_layer.bias.data[0]).detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    x, y = np.meshgrid(x, y)
    mlp_output = mlp_output.reshape((100, 100))
    ax.plot_surface(x, y, mlp_output, cmap=cm.turbo, linewidth=0, antialiased=False)

    ax = fig.add_subplot(1, 2, 2, projection = "3d")
    rbf_output = rbf_output.reshape((100, 100))
    ax.plot_surface(x, y, rbf_output, cmap=cm.turbo, linewidth=0, antialiased=False)
    plt.show()




def train(input, output, config):
    n = input.shape[0]    
    # normalize the input to 0 to 1, input i a torch tensor with shape (n, 2)
    #input = (input - input.min()) / (input.max() - input.min())
    indices = random.sample(range(input.shape[0]), input.shape[0])
    training_dataset = torch.tensor(input[indices[int(0):int(0.75*len(indices))], :], dtype= torch.float32)
    training_output = torch.tensor(output[indices[int(0):int(0.75*len(indices))], :], dtype= torch.float32)
    validation_dataset = torch.tensor(input[indices[int(0.75*len(indices)): int(0.90*len(indices))], :], dtype = torch.float32)
    validation_output =  torch.tensor(output[indices[int(0.75*len(indices)): int(0.90*len(indices))], :], dtype= torch.float32)
    test_dataset = torch.tensor(input[indices[int(0.90*len(indices)):], :], dtype=torch.float32)
    test_output =  torch.tensor(output[indices[int(0.90*len(indices)):],  :], dtype= torch.float32)
    print(training_dataset.shape, training_output.shape, validation_dataset.shape, validation_output.shape, test_dataset.shape, test_output.shape)
    widths = [8, 16, 32, 64]
    # create a network
    if config.algorithm == "mlp":
        network = build_mlp_model(config.k, config.model, 1)
    # rbf network
    elif config.algorithm == "rbf":
        rbf, network = build_rbf_model(config.k, config, config.model, 1)
    # mlp + rbf network
    elif config.algorithm == "mlp_rbf":
        new_network = build_mlp_model(config.k, config.model, 1)
        rbf, new_network_rbf = build_rbf_model(2, config, [32], 1)
        #new_network_rbf = build_mrbf_model(2, 20, 1)
        network = mlp_rbf_network(new_network, new_network_rbf, 1)
    elif config.algorithm == "mrbf":
        network = build_mrbf_model(config.k, 1, config.model)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.0003)
    if config.k <= 3:
        epoches = 200
    else:
        epoches = 400
    training_loss_list = []
    smoothed_training_loss_list = []
    criterion = nn.MSELoss()
    iterations = 100
    validation_loss_list = []
    smoothed_training_loss_list = []
    num_params = get_num_params(network)
    log.add_scalar("num_params", num_params)
    for epoch in range(epoches):
        for i in range(iterations):
            input_batch, output_batch = create_batched_dataset(training_dataset, training_output, batch_size= 64)
            network.zero_grad()
            network_output = network(input_batch)
            loss = criterion(network_output, output_batch)
            loss.backward()
            optimizer.step()
            training_loss_list.append(loss.item())
        print("Epoch: " + str(epoch) + " Loss: " + str(training_loss_list[-100]))
        smoothed_training_loss_list.append(np.mean(training_loss_list[-100:]))
        log.add_scalar("Loss", np.mean(training_loss_list[-100:]))
        #checking validation loss
        predicted_validation_output = network(validation_dataset)
        validation_loss = criterion(predicted_validation_output, validation_output)
        validation_loss_list.append(validation_loss.item())
        print("validation loss:", validation_loss.item())
        log.add_scalar("validation_loss", validation_loss.item())
        if np.mean(validation_loss_list[-20:]) > np.mean(validation_loss_list[-40:-20]) and len(validation_loss_list) > 40:
            validation_predicted_test_output = network(test_dataset)
            test_loss = criterion(validation_predicted_test_output, test_output)
            log.add_scalar("EpochValidationBreak", epoch)
            log.add_scalar("TestLossValidation", test_loss.item())
            log.add_scalar("ValidationLossBreak", validation_loss.item())
            is_true = False     
        #if epoch  == 200:
        #    optimizer.param_groups[0]['lr'] = 3e-5
        #    optimizer.param_groups[1]['lr'] = 3e-4
    #print("First param group")
    #print(optimizer.param_groups[0]['lr'])
    #print("Second Param group")
    #print(optimizer.param_groups[1]['lr'])
    #print(network.linear_layer.weight.data)
    # testing the network
    predicted_test_output = network(test_dataset)
    test_loss = criterion(predicted_test_output, test_output)
    print("Testing Loss: ", test_loss)

    log.add_scalar("TestLoss", test_loss.detach().numpy())

    if config.algorithm == "rbf" or config.algorithm == "mlp_rbf":
        log.add_value("rbf_peaks", rbf.peaks.detach().numpy())
        log.add_value("rbf_sigmas", rbf.sigmas.detach().numpy())
    tensor_input = torch.tensor(input, dtype=torch.float32) 
    predicted_output = network(tensor_input).detach().numpy()
    log.add_value("predicted_output", predicted_output)

    return predicted_output
    

def run_task(config = None, **kwargs):
    torch.set_num_threads(1)
    eu.misc.seed(42 + config.seed)
    config = eu.combine_dicts(kwargs, config, default_config())
    x = np.arange(-2, 2, 0.4)
    input = list(
            itertools.product(x, repeat = config.k)
        )
    input = np.asarray(input)
    reshape_param = input.shape[0]
    output = poly(input, reshape_param, config=config).reshape((reshape_param, 1))
    predicted_output_mlp = train(input, output, config)
    log.save()
    return log
