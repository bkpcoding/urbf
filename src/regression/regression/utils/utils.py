import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
from matplotlib.patches import Circle
from rbf_layer import RBFLayer
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multi_rbf_layer import MRBF
from functions.gaussians import gaussians, gaussians1


def draw_3d_plot(x, y, z, new_z, new_z_rbf, new_z_mlp_rbf, config):
    fig = plt.figure()
    x, y = np.meshgrid(x, y)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.plot_surface(x, y, z, cmap=cm.turbo, linewidth=0, antialiased=False)
    ax.set_title("Original")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax = fig.add_subplot(1, 4, 2, projection="3d")
    ax.plot_surface(x, y, new_z, cmap=cm.turbo, linewidth=0, antialiased=False)
    ax.set_title("MLP")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    ax.plot_surface(x, y, new_z_rbf, cmap=cm.turbo, linewidth=0, antialiased=False)
    ax.set_title("RBF")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax = fig.add_subplot(1, 4, 4, projection="3d")
    ax.plot_surface(x, y, new_z_mlp_rbf, cmap=cm.turbo, linewidth=0, antialiased=False)
    ax.set_title("MLP + RBF")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    name = config.name
    plt.show()
    #plt.savefig(name, format = "svg")

def gaussians_3d(x, y, z):
    fig = plt.figure()
    x, y = np.meshgrid(x, y)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.turbo, linewidth=0, antialiased=False)
    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # remove the borders
    ax.w_xaxis.line.set_lw(0.)
    ax.w_yaxis.line.set_lw(0.)
    ax.w_zaxis.line.set_lw(0.)
    # save the pdf with minimal white space
    plt.savefig("gaussians_3d.pdf", bbox_inches='tight', pad_inches=0)
    #plt.show()

def draw_2d_heatmap(x, y, z, mlp_z, rbf_z, mlp_rbf_z):
    #draw a 2D heatmap of the x, y, z using plt.imshow
    #x is a numpy array in range -5 to 5
    #y is a numpy array in range -5 to 5
    #invert z to make it look like a heatmap
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
    turbo = cm.turbo
    im = ax1.imshow(z, cmap=turbo, extent=[-5, 5, -5, 5], origin='lower')
    #colorbar for each of the different plots
    ax1.set_title("Ground Truth")
    divider = make_axes_locatable(ax1)
    #cax1 = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im1, cax=cax1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    
    im = ax2.imshow(mlp_z, cmap=turbo, extent=[-5, 5, -5, 5], origin='lower')
    divider = make_axes_locatable(ax2)
    #cax2 = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im2, cax=cax2)
    ax2.set_title("MLP")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")


    im = ax3.imshow(rbf_z, cmap=turbo, extent=[-5, 5, -5, 5], origin='lower')
    divider = make_axes_locatable(ax3)
    #cax3 = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im3, cax=cax3)
    ax3.set_title("RBF")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    im = ax4.imshow(mlp_rbf_z, cmap=turbo, extent=[-5, 5, -5, 5], origin='lower')
    divider = make_axes_locatable(ax4)
    #cax4 = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im4, cax=cax4)
    ax4.set_title("MLP + RBF")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
#    name = config.name
    cbar_ax = fig.add_axes([0.92, 0.34, 0.03, 0.31])
    fig.colorbar(im, cax = cbar_ax)
    plt.show()

def calc_error(input, output, network, config):
    error = []
    if config.m == 0:
        pass
    for i in range(input.shape[0]):
        if config.m == 1:
            if input[i, 0] >= -1 and input[i, 0] <= 1:
                real_output = torch.tensor(output[i], dtype=torch.float32)
                err_output = network(torch.tensor(input[i,:], dtype=torch.float32))
                error.append((real_output - err_output).item())
            else:
                pass
        elif config.m == 3:
            if (input[i, 0] >= -1 and input[i, 0] <= 1) or (input[i, 0] <=0 and input[i, 1] <= 0.5 and input[i, 1] >= -0.5) \
                or (input[i, 0] >= 0 and input[i, 1] < -0.5 and input[i, 1] > -1.5):
                real_output = torch.tensor(output[i], dtype=torch.float32)
                err_output = network(torch.tensor(input[i,:], dtype=torch.float32))
                error.append((real_output - err_output).item())

        elif config.m == 5:
            if (input[i, 0] <= -1 and input[i, 0] >= 1) or (input[i, 0] <=0 and input[i, 1] <= 0.5 and input[i, 1] >= -0.5) \
                or (input[i, 0] <= 0 and input[i,1] >= -2.5 and input[i, 1] <= -1.5) or (input[i, 0] >= 0 and input[i, 1] < 3.5 and input[i, 1] > 2.5):
                real_output = torch.tensor(output[i], dtype=torch.float32)
                err_output = network(torch.tensor(input[i,:], dtype=torch.float32))
                error.append((real_output - err_output).item())
        else:
            pass
    error_1 = np.sqrt(np.sum(np.square(error))/len(error))
    return error_1


def plot_gaussian(means_1, stds_1, means_2, std_2, config = None):
    means_1 = means_1.detach().numpy()
    stds_1 = stds_1.detach().numpy()
    means_2 = means_2.detach().numpy()
    std_2 = std_2.detach().numpy()
    gaussians_x_1 = np.zeros((1,1000))
    x = np.linspace(-5, 5, 1000).reshape(1,1000)
    #update the gaussians with the real gaussians with mean and std
    for i in range((config.n_neurons_per_input)):
        gaussians_x_1 += np.exp(-(x - means_1[i])**2 / (2 * stds_1[i]**2))
    #plot the gaussians_x against x
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
    ax1.scatter(x, gaussians_x_1)
    ax1.set_title("Gaussian_X_RBF")
    ax1.set_xlabel("x")
    gaussians_x_2 = np.zeros((1,1000))
    for i in range((config.n_neurons_per_input)):
        gaussians_x_2 += np.exp(-(x - means_2[i])**2 / (2 * std_2[i]**2))
    ax2.scatter(x, gaussians_x_2)
    ax2.set_title("Gaussian_X_MLP_RBF")
    ax2.set_xlabel("x")
    #plot the same graphs as above for y
    gaussians_y_1 = np.zeros((1,1000))
    for i in range((config.n_neurons_per_input), (config.n_neurons_per_input) * 2):
        gaussians_y_1 += np.exp(-(x - means_1[i])**2 / (2 * stds_1[i]**2))
    ax3.scatter(x, gaussians_y_1)
    ax3.set_title("Gaussian_Y_RBF")
    ax3.set_xlabel("x")
    gaussians_y_2 = np.zeros((1,1000))
    for i in range((config.n_neurons_per_input), (config.n_neurons_per_input) * 2):
        gaussians_y_2 += np.exp(-(x - means_2[i])**2 / (2 * std_2[i]**2))
    ax4.scatter(x, gaussians_y_2)
    ax4.set_title("Gaussian_Y_MLP_RBF")
    ax4.set_xlabel("x")
    name = config.name + "gaussians"
    plt.savefig(name, format = "svg")

def plot_2d_gaussian(means, stds):
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    patches = []
    print(means)
    print(stds)
    for i in range(len(means)):
        circle = Circle((means[i], means[i]), stds[i])
        matplotlib.patches.Patch.set_facecolor(circle, color=(0, 0, 0, 0))
        patches.append(circle)
    p = matplotlib.collections.PatchCollection(patches)
    ax.add_collection(p)
    plt.xlim(-5.0, 5.0)
    plt.ylim(-5.0, 5.0)
    plt.show()


def create_batched_dataset(input, output, batch_size):
    # x is tensor of size (n, d)
    # y is tensor of size (nxn)
    indices = []
    for i in range(batch_size):
        indices.append(np.random.randint(0, input.shape[0]))
    
    indices = np.asarray(indices)
    input_batch = input[indices, :]
    output_batch = output[indices, :]
    return input_batch, output_batch


def build_mlp_model(input, hidden_units, output):
    network = nn.Sequential()
    for i, hidden_unit in enumerate(hidden_units):
        if i == 0:
            network.add_module("first_linear", nn.Linear(input, hidden_unit))
            network.add_module("Relu", nn.ReLU())
            prev_hidden_unit = hidden_unit
        else:
            network.add_module(str(i), nn.Linear(prev_hidden_unit, hidden_unit))
            network.add_module("ReLU", nn.ReLU())
            prev_hidden_unit = hidden_unit
    network.add_module("output_layer", nn.Linear(prev_hidden_unit, output))
    return network


def build_mlp(input_dims, depth, width_idx, output_dims, widths):
    network = nn.Sequential()
    width = widths[width_idx]
    network.add_module("First_layer", nn.Linear(input_dims, width))
    for i in range(depth):
        network.add_module(i, nn.Linear(width, width))
    network.add_module("output_layer", width, output_dims)
    return network
        


def build_rbf_model(input, config, hidden_units, output):
    network = nn.Sequential()
    rbf_layer = RBFLayer(input, input*config.n_neurons_per_input, ranges = config.ranges,
                        is_trainable = config.is_trainable)
    network.add_module(
        "RBFLayer", rbf_layer)
    for i, hidden_unit in enumerate(hidden_units):
        if i == 0:
            network.add_module(
                "first layer",
                nn.Linear(input * config.n_neurons_per_input, hidden_unit),
            )
            network.add_module("ReLU", nn.ReLU())
            prev_hidden_unit = hidden_unit
        else:
            network.add_module(str(i), nn.Linear(prev_hidden_unit, hidden_unit))
            network.add_module("ReLU", nn.ReLU())
            prev_hidden_unit = hidden_unit
    network.add_module("output layer", nn.Linear(prev_hidden_unit, output))
    return rbf_layer, network

def build_mrbf_model(input_dims, output_dims, hidden_units):
    network = nn.Sequential()
    for i, hidden_unit in enumerate(hidden_units):
        if i == 0:
            network.add_module("first layer", MRBF(input_dims, hidden_unit))
            #network.add_module("ReLU", nn.ReLU())
            prev_hidden_unit = hidden_unit
        else:
            network.add_module(str(i), nn.Linear(prev_hidden_unit, hidden_unit))
            network.add_module("ReLU", nn.ReLU())
            prev_hidden_unit = hidden_unit
    network.add_module("output layer", nn.Linear(prev_hidden_unit, output_dims))
    return network



class mlp_rbf_network(nn.Module):
    def __init__(self, mlp_network, rbf_network, output):
        super(mlp_rbf_network, self).__init__()
        self.mlp_network = mlp_network
        self.rbf_network = rbf_network
        self.linear_layer = nn.Linear(2, output)

    def forward(self, x):
        y1 = self.mlp_network(x)
        y2 = self.rbf_network(x)
        # print(y1.shape, y2.shape)
        z = torch.concat((y1, y2), dim=-1)
        # print(y1.shape, y2.shape)
        # print(z.shape)
        output = self.linear_layer(z)
        return output


def plot_from_numpy_data():
    mlp_test_loss = np.load("/home/sagar/inria/experiments/3d_curve/3d_exp/experiments/experiment_000001/repetition_000001/data/MlpLoss.npy")
    rbf_test_loss = np.load("/home/sagar/inria/experiments/3d_curve/3d_exp/experiments/experiment_000001/repetition_000001/data/RbfLoss.npy")
    fig, ax = plt.subplots()
    ax.plot(mlp_test_loss, label="MLP")
    ax.plot(rbf_test_loss, label="RBF")
    ax.set_xlabel("epoch in the order 10^3")
    ax.set_ylabel("loss")
    ax.legend()
    plt.show()


if __name__=="__main__":
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    mean = np.array([[0, 0], [1, 2], [-3,3], [-4,-3], [2, -2]])
    std = np.array([[0.5, 0.8], [0.8, 0.8], [0.6, 1], [1, 0.25], [1, 0.32]])

    z = gaussians1(x, y, mean=mean, std=std, config=None)
    print(x.shape, y.shape, z.shape)
    gaussians_3d(x, y, z)