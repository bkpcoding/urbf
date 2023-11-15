import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from non_convex_testing.utils.rbf_layer import RBFLayer
import exputils as eu
import exputils.data.logging as log


def default_config():
    return eu.AttrDict(
        seed = 0,
        network = 1,
        model = [64, 32],
        n_neurons_per_input = 5,
        n_epoch = 50,
        batch_size = 32,
    )


def train(net, config):
    #train network on cifar10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = config.batch_size

    trainset = torchvision.datasets.CIFAR10(root='../../../data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../../../data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.003)
    num_params = sum(p.numel() for p in net.parameters())
    log.add_scalar('number of parameters', num_params)


    for epoch in range(config.n_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1400 == 1399:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                log.add_scalar('training_loss', running_loss / 2000)
                running_loss = 0.0
    print('Finished Training')
    # test the network on the test data
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    log.add_scalar('test_accuracy', 100 * correct // total)
        

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    log.save()



class Net_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_2(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_layers = config.model
        self.network = nn.ModuleList()
        self.network.append(RBFLayer(3*32*32, config = config))
        prev = 3*32*32*(config.n_neurons_per_input)
        for hidden_layer in hidden_layers:
            self.network.append(nn.Linear(prev, hidden_layer))
            prev = hidden_layer
        self.network.append(nn.Linear(prev, 10))

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        for layer in range(len(self.network) - 1):
            x = F.relu(self.network[layer](x))
        x = self.network[-1](x)
        return x

class Net_3(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_layers = config.model
        self.network = nn.ModuleList()
        prev = 3 * 32 * 32
        for hidden_layer in hidden_layers:
            self.network.append(nn.Linear(prev, hidden_layer))
            prev = hidden_layer
        self.network.append(nn.Linear(prev, 10))

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        for layer in range(len(self.network) - 1):
            x = F.relu(self.network[layer](x))
        x = self.network[-1](x)
        return x

def run_task(config = None, **kwargs):
    torch.set_num_threads(2)
    config = eu.combine_dicts(kwargs, config, default_config())
    eu.misc.seed(32 + config.seed)
    if config.network == 1:
        net = Net_1()
    elif config.network == 2:
        net = Net_2(config)
    elif config.network == 3:
        net = Net_3(config)
    train(net, config)

run_task()