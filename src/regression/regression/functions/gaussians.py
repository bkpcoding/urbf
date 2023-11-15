import numpy as np
import random
import torch
import plotly.express as px
import plotly.graph_objects as go
import math

def gaussians(input, mean, std):
    # number of dimensions of the function
    d = 2
    # number of gaussians in the function
    m = 1
    # number of samples in the function
    n = 1000
    z = np.zeros((input.shape[0]))
    for i in range(mean.shape[0]):
        z += np.exp(-((((input[:, 0] - mean[i, 0])**2)/ 2.0*(std[i,0]**2) )))* np.exp(-((input[:, 1] - mean[i, 1])**2)/ 2.0*(std[i,0]**2))
    z = z.reshape(z.shape[0], 1)
    return z
    
def gaussians1(x, y, mean, std, config):
    # number of dimensions of the function
    d = 2
    # number of gaussians in the function
    m = 1
    # number of samples in the function
    n = 1000
    height = 3 
    #height = np.random.randint(1, 10)
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    z = np.zeros((len(x), len(y)))
    
    if config == None:
        x_new = x - mean[:, 0]
        y_new = y - mean[:, 1]
        x_new = height * np.exp(-(x_new ** 2) / (2 * std[:, 0] ** 2))
        y_new = height * np.exp(-(y_new ** 2) / (2 * std[:, 0] ** 2))
        z = np.matmul(x_new, y_new.T)
    for i in range(len(x)):
        for j in range(len(y)):
            #add a random epsilon to the function to prevent overfitting
            z[i, j] += x[i] + y[j] + (random.random() - 0.5) * 0.1
    return z


def multivariate_gaussian(input, means, covariance, k, m):
    # input matrix is a k x 1000 size matrix
    # then means should be a k x 1 size matrix
    # then covariance matrix is a k x k size matrix
    # output will 1 x 1000 size matrix
    # calculate the multivariate gaussian for the input, with the means and covariance matrix
    k = input.shape[0]
    output = torch.zeros(1, input.shape[1])
    for i in range(m):
        output += torch.exp(-0.5*torch.sum(((((input - means[:, i].unsqueeze(-1)))**2)/covariance[:, i].unsqueeze(-1)), dim=0)).unsqueeze(0)
    return output

#torch.manual_seed(132)
#k = 2
#m = 2
#input = torch.rand(k, 10000)*2
#mean = torch.rand(k, m)
#covariance = torch.rand(k,m)
#output = multivariate_gaussian(input, mean, covariance, k, m)
#output = output.squeeze(0)
#mesh_size = 0.2
#x_min, x_max = 0, 2
#y_min, y_max = 0, 2
#xrange = np.arange(x_min, x_max, mesh_size)
#yrange = np.arange(y_min, y_max, mesh_size)
#xx, yy = np.meshgrid(xrange, yrange)
## Run model
##pred = net(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
##pred = pred.reshape(xx.shape)
##pred = pred.detach().numpy()
##
##fig = go.Figure(data = go.Heatmap(z = output, x = input[0,:], y = input[1,:]))
##trace = go.Heatmap(x = input[0,:], y = input[1,:], z = output)
## create a 10000 x 10000 matrix from output
#print(output.shape)
#output1, output = np.meshgrid(output, output)
#output = np.asarray(output)
#print(output.shape)
#fig = px.imshow(img = output, x = input[0,:], y = input[1,:])
##fig.add_trace(trace)
##fig.update_traces(marker=dict(size=5))
##fig.add_traces(go.Surface(x=xrange, y=yrange, z=output, name='pred_surface'))
#fig.show()
#

