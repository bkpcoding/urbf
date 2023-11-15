from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly
import random
from non_convex_testing.utils.utils import RBFLayer
import plotly.express as px
import plotly.graph_objects as go



class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2,32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2,16)
        self.rbf = RBFLayer(16, n_neurons_per_input = 5, range = [0, 8])
        self.fc2 = nn.Linear(16*5, 1)

    def forward(self, x):
        x = x.view(-1, 2)
        x = F.relu(self.fc1(x))
        x = self.rbf(x)
        x = self.fc2(x)
        return x

def train(network):
    #train network on iris dataset
    plot = False

    mesh_size = .02
    margin = 0
    df = px.data.iris()
    X = df[['sepal_width', 'sepal_length']]
    y = df['petal_width']
    X_train = X.values
    y_train = y.values
    print(X_train.shape)
    if network == 1:
        net = Net()
    elif network == 2:
        net = Net2()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    # fit the dataset to the model
    for epoch in range(100):
        if epoch > 0:
            print("Loss after epoch {}: {}".format(epoch -1, running_loss / prev))
        running_loss = 0.0
        for iteration in range(len(X_train)):
            inputs = torch.from_numpy(X_train[iteration]).float()
            #convert np.float64 to torch.float32 for labels
            labels = torch.tensor(y_train[iteration], dtype=torch.float32)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            prev = iteration

    if plot == True:
        x_min, x_max = X.sepal_width.min() - margin, X.sepal_width.max() + margin
        y_min, y_max = X.sepal_length.min() - margin, X.sepal_length.max() + margin
        xrange = np.arange(x_min, x_max, mesh_size)
        yrange = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)

        # Run model
        pred = net(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
        pred = pred.reshape(xx.shape)
        pred = pred.detach().numpy()

        fig = px.scatter_3d(df, x='sepal_width', y='sepal_length', z='petal_width')
        fig.update_traces(marker=dict(size=5))
        fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
        fig.show()



    

train(1)