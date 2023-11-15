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
from non_convex_testing.functions.gaussians import gaussians, gaussians1, multivariate_gaussian
from non_convex_testing.functions.many_local_minima import ackley_function
from non_convex_testing.functions.staircase_like_functions import staircase
from non_convex_testing.functions.bowl_like_functions import styblinski_function
from non_convex_testing.utils.utils import build_mrbf_model, draw_2d_heatmap, draw_3d_plot, plot_gaussian
from non_convex_testing.utils.utils import create_batched_dataset
from non_convex_testing.utils.utils import build_mlp_model, build_rbf_model, build_mlp, mlp_rbf_network
from non_convex_testing.utils.utils import calc_error
from non_convex_testing.functions.polynomials import fixed_polynomial, poly
from non_convex_testing.utils.multi_rbf_layer import MRBF
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
        algorithm = "rbf", n_neurons_per_input=10, is_trainable=True, ranges=[0.0, 2.0], m=2,
        model = [32, 64, 128], k = 3, n = 7, size = 10000, seed = 42,
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
    input = torch.tensor(input, dtype=torch.float32)     
    output = torch.tensor(output, dtype=torch.float32)  
    # normalize the input to 0 to 1, input i a torch tensor with shape (n, 2)
    #input = (input - input.min()) / (input.max() - input.min())
    indices = random.sample(range(input.shape[0]), input.shape[0])
    training_dataset = input[indices[int(0):int(0.75*len(indices))], :]
    training_output = output[indices[int(0):int(0.75*len(indices))], :]
    validation_dataset = input[indices[int(0.75*len(indices)): int(0.90*len(indices))], :]
    validation_output =  output[indices[int(0.75*len(indices)): int(0.90*len(indices))], :]
    test_dataset = input[indices[int(0.90*len(indices)):], :]
    test_output =  output[indices[int(0.90*len(indices)):],  :]
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
    #network = MRBF(config.k, 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.0003)
    #optimizer = torch.optim.SGD([
    #            {'params': network.mlp_network.parameters()},
    #            {'params': network.rbf_network.parameters(), 'lr': 3e-5}
    #        ], lr=3e-4)
    #print("First param group")
    #print(optimizer.param_groups[0]['lr'])
    #print("Second Param group")
    #print(optimizer.param_groups[1]['lr'])
    epoches = 300
    training_loss_list = []
    smoothed_training_loss_list = []
    criterion = nn.MSELoss()
    iterations = 100
    validation_loss_list = []
    smoothed_training_loss_list = []
    num_parameters = sum([param.numel() for param in network.parameters() if param.requires_grad])
    log.add_scalar("number of parameters", num_parameters)
    for epoch in range(epoches):
        for i in range(iterations):
            input_batch, output_batch = create_batched_dataset(training_dataset, training_output, batch_size= 128)
            network.zero_grad()
            network_output = network(input_batch)
            loss = criterion(network_output, output_batch)
            loss.backward()
            optimizer.step()
            training_loss_list.append(loss.item())
        print("Epoch: " + str(epoch) + " Loss: " + str(training_loss_list[-100]))
        smoothed_training_loss_list.append(np.mean(training_loss_list[-100:]))
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
        #if epoch  == 200:
        #    optimizer.param_groups[0]['lr'] = 3e-5
        #    optimizer.param_groups[1]['lr'] = 3e-4
    #print("First param group")
    #print(optimizer.param_groups[0]['lr'])
    #print("Second Param group")
    #print(optimizer.param_groups[1]['lr'])
    #print(network.linear_layer.weight.data)
    # testing the network
    test_loss_list = []
    predicted_test_output = network(test_dataset)
    test_loss = criterion(predicted_test_output, test_output)
    test_loss_list.append(test_loss)
    print("Testing Loss: ", test_loss)
    #plt.plot(smoothed_training_loss_list)
    #plt.show()

    log.add_scalar("TestLoss", test_loss.detach().numpy())

    if config.algorithm == "rbf" or config.algorithm == "mlp_rbf":
        log.add_value("rbf_peaks", rbf.peaks.detach().numpy())
        log.add_value("rbf_sigmas", rbf.sigmas.detach().numpy())
    tensor_input = torch.tensor(input, dtype=torch.float32) 
    predicted_output = network(tensor_input).detach().numpy()
    log.add_value("predicted_output", predicted_output)
    #error_1 = calc_error(input, output, network, config)
    #log.add_scalar("Error", error_1)
    if config.algorithm == "mlp_rbf":
        plot_rbf_mlp(new_network, new_network_rbf, network, tensor_input, output, config)
    
    #mlp network loss for different points and corresponding RBF value
    #mlp_loss_rbf_perf = []
    #for idx in range(input.shape[0]):
    #    mlp_loss = np.abs(torch.tensor(output[idx], dtype = torch.float32) - new_network(tensor_input[idx, :])*network.linear_layer.weight.data[0, 0] + network.linear_layer.bias.data[0])
    #    print(mlp_loss)

    return predicted_output
    

def run_task(config = None, **kwargs):
    torch.set_num_threads(1)
    config = eu.combine_dicts(kwargs, config, default_config())
    eu.misc.seed(32 + config.seed)

    # produce a random tensor with shape (k, 10000)
    #input = torch.rand((config.k, config.size))*2


    #x = np.arange(-2, 2, 0.4)
    #input = list(
    #        itertools.product(x, repeat = config.k)
    #    )
    #x = np.arange(-5, 5, 0.1)
    #y = np.arange(-5, 5, 0.1)
    #input = list(
    #        itertools.product(x, y)
    #    )
    #input = np.asarray(input)
    #print(input)
    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&input&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #print(input.shape)
    #reshape_param = input.shape[0]
    #print(input.shape)
    #m = config.m
     #print(input)
    # mean = (np.random.rand(m, d) * 10) - 5

    #mean = []
    #std = []
    #for i in range(complexities[config.complexity]):
    #    mean.append([random.randint(-5, 5), random.randint(-5, 5)])
    #    std.append([0.4 + random.random()*0.4])
    #mean = np.asarray(mean)
    #std = np.asarray(std)
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
    #output = staircase(x, y, config).reshape((10000, 1))
    #output = gaussians1(x, y, mean, std, config).reshape((10000,1))
    k = config.k
    m = config.m
    input = torch.rand(k, config.size)*2
    mean = torch.rand(k, m)
    variance = torch.rand(k,m)
    output = multivariate_gaussian(input, mean, variance, k, m)
    input = input.T
    output = output.T
    #print(output)
    # create_dataset(x,y,z)
    predicted_output_mlp = train(input, output, config)
    #log.add_value("input_vector", input)
    #log.add_value("actual_output", output)
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 4, 1, projection="3d")
    #x, y = np.meshgrid(x, y)
    #predicted_output_mlp = predicted_output_mlp.reshape((100, 100))
    #ax.plot_surface(x, y, predicted_output_mlp, cmap=cm.turbo, linewidth=0, antialiased=False)
    #ax.title.set_text('MLP')
    #config.algorithm = "rbf"
    #predicted_output_rbf = train(input, output, config)
    #predicted_output_rbf = predicted_output_rbf.reshape((100, 100))

    #config.algorithm = "mrbf"
    #predicted_output_mlp_rbf = train(input, output, config)
    #predicted_output_mlp_rbf = predicted_output_mlp_rbf.reshape((100, 100))
    #output = output.reshape((100, 100))
    ##predicted_output_mlp = predicted_output_mlp - output
    ##predicted_output_rbf = predicted_output_rbf - output
    ##predicted_output_mlp_rbf = predicted_output_mlp_rbf - output
    ##draw_2d_heatmap(x, y, output, predicted_output_mlp, predicted_output_rbf, predicted_output_mlp_rbf)

    #ax = fig.add_subplot(1, 4, 2, projection = "3d")
    #output = output.reshape((100, 100))
    #ax.plot_surface(x, y, predicted_output_mlp_rbf, cmap=cm.turbo, linewidth=0, antialiased=False)
    #ax.title.set_text('mrbf')


    #ax = fig.add_subplot(1, 4, 3, projection = "3d")
    #output = output.reshape((100, 100))
    #ax.plot_surface(x, y, predicted_output_rbf, cmap=cm.turbo, linewidth=0, antialiased=False)
    #ax.title.set_text('rbf')

    #ax = fig.add_subplot(1, 4, 4, projection = "3d")
    #output = output.reshape((100, 100))
    #ax.plot_surface(x, y, output, cmap=cm.turbo, linewidth=0, antialiased=False)
    #ax.title.set_text('Ground Truth Function')

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 2, 1, projection="3d")
    #x, y = np.meshgrid(x, y)
    #predicted_output = predicted_output_mlp.reshape((100, 100))
    #print(x.shape, y.shape, predicted_output.shape)
    #ax.plot_surface(x, y, predicted_output, cmap=cm.turbo, linewidth=0, antialiased=False)

    #ax = fig.add_subplot(1, 2, 2, projection = "3d")
    #output = output.reshape((100, 100))
    #ax.plot_surface(x, y, output, cmap=cm.turbo, linewidth=0, antialiased=False)
    #plt.show()

    #plt.)
    log.save()
    return log