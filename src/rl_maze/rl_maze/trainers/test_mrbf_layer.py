import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import collections
import exputils as eu

class RBFLayer(torch.nn.Module):
    """RBF layer that has for each input dimension n RBF neurons.

    Each RBF neuron y_i,j for input x_i has a Gaussian activation function:
        y_i,j = exp(-0.5 * ((x_i - mu_i,j) / sigma_i,j)**2)

    Config:
        n_neurons_per_input (int): Number of RBF neurons for each input dimension.
            Also defines the output dimensions (n_out = n_input * n_neurons_per_input).
            (default = 5)
        ranges (list): Defines the value range ([min_value, max_value]) of each input dimension used to
            define the initial peaks of the RBF layers. The RBF peaks are equally distributed within the
            range. For example, having RBF neurons (n_neurons_per_input=5) with in a range=[-1.0, 1.0] yields
            the following peaks: [-1.0, -0.5, 0.0, 0.5, 1.0]
            Can be a single range that is used for each input dimension (ranges=[0.0, 1.0]) or a range
            for each dimension (ranges=[[0.0, 1.0],[-2.0, 2.0]]).
            (default = [-1.0, 1.0])
        sigma (float or list): Defines the spread (sigma) of each RBF neuron.
            Can be a single sigma (sigma=0.5) used for each input dimension or a list of sigmas for each
            individual input dimension (sigma=[0.5, 0.3]).
            If no sigma is given (sigma=None), then a sigma is chosen such that for an input value that is
            in the middle of the peaks of 2 neurons, both neurons have an equal activation of 0.5.
            Formula: (-dist**2/(8*np.log(0.5)))**0.5 where dist is the distance between the 2 peaks.
            (default = None)
        is_trainable (bool): True if the parameters of the RBF neurons (peak position and spread) are trainable.
            False if not.
            (default = False)

    Properties:
        n_in (int): Number input dimensions.
        n_neurons_per_input (int): Number of RBF neurons per input dimension.
        n_out (int): Number of output dimensions.

    Example:
        # input a batch of 2 inputs
        x = torch.tensor([
            [0.2, 0.4, 0.3],
            [-0.1, 0.2, 0.0]
        ])
        y = rbf_layer(x)
        print(y)
    """


    @staticmethod
    def default_config():
        return eu.AttrDict(
            n_neurons_per_input=None,
            ranges=[-3.0, 3.0],
            sigma=None,
            is_trainable=True,
        )


    @property
    def dtype(self):
        # torch.nn.Linear uses torch.float32 as dtype for parameters
        return torch.float32


    @property
    def n_in(self):
        return self._n_in


    @property
    def n_neurons_per_input(self):
        return self._n_neurons_per_input


    @property
    def n_out(self):
        return self._n_out


    def __init__(self, n_in, n_out=None, config=None, **argv):
        """Creates a RBF Layer.

        Args:
            n_in (int): Size of the input dimension.
        """
        super().__init__()
        self.config = eu.combine_dicts(argv, config, self.default_config())

        self._n_in = n_in
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.min = 1e+6
        self.max = -1e+6

        # identify self._n_neurons_per_input
        n_neurons_per_input_according_to_n_out = None
        n_neurons_per_input_according_to_config = None
        if n_out is None and self.config.n_neurons_per_input is None:
            raise ValueError('Either n_out or config.n_neurons_per_input must be set!')
        if n_out is not None:
            if n_out % n_in != 0:
                ValueError('n_in must be a divisible multitude of n_out!')
            n_neurons_per_input_according_to_n_out = int(n_out / n_in)
            self._n_neurons_per_input = n_neurons_per_input_according_to_n_out
        if self.config.n_neurons_per_input is not None:
            n_neurons_per_input_according_to_config = self.config.n_neurons_per_input
            self._n_neurons_per_input = n_neurons_per_input_according_to_config
        if n_neurons_per_input_according_to_n_out is not None and n_neurons_per_input_according_to_config is not None:
            if n_neurons_per_input_according_to_n_out != n_neurons_per_input_according_to_config:
                raise ValueError('Number of RBF neurons must be consistent in between config.n_neurons_per_input and n_out!')


        self._n_out = self._n_in * self.n_neurons_per_input

        if self.config.ranges is None:
            self.ranges = np.array([[-1.0, 1.0]] * self._n_in)
        elif np.ndim(self.config.ranges) == 1:
            self.ranges = np.array([self.config.ranges] * self._n_in)
        else:
            self.ranges = np.array(self.config.ranges)

        self.peaks = torch.Tensor(self.n_out)
        for input_idx in range(self._n_in):
            start_idx = input_idx * self._n_neurons_per_input
            end_idx = start_idx + self._n_neurons_per_input
            self.peaks[start_idx:end_idx] = torch.linspace(self.ranges[input_idx][0], self.ranges[input_idx][1], self._n_neurons_per_input)

        # handle different types of sigma parameters and convert them to a list with one sigma per input
        if self.config.sigma is None:
            self.sigma = np.zeros(self._n_in)
            for input_idx in range(self._n_in):
                dist = self.peaks[input_idx * self._n_neurons_per_input + 1] - self.peaks[input_idx * self._n_neurons_per_input]
                self.sigma[input_idx] = (-dist ** 2 / (8 * np.log(0.5))) ** 0.5
        elif not isinstance(self.config.sigma, collections.Sequence):
            self.sigma = np.ones(self._n_in) * self.config.sigma
        else:
            self.sigma = self.config.sigma

        self.sigmas = torch.Tensor(self.n_out)
        for input_idx in range(self._n_in):
            start_idx = input_idx * self._n_neurons_per_input
            end_idx = start_idx + self._n_neurons_per_input
            self.sigmas[start_idx:end_idx] = self.sigma[input_idx]

        # if the layer should be trainable, then add the peaks and sigmas as parameters
        if self.config.is_trainable:
            self.peaks = torch.nn.Parameter(self.peaks)
            self.sigmas = torch.nn.Parameter(self.sigmas)


    def forward(self, x):
        """Calculate the RBF layer output.

        Args:
            x (torch.Tensor): Torch tensor with a batch of inputs.
                The tensor has 2 dimensions, where each row vector is a single input.
        """

        # reapeat input vector so that every map-neuron gets its accordingly input
        # example: n_neuron_per_inpu = 3 then [[1,2,3]] --> [[1,1,1,2,2,2,3,3,3]]
        #if x.min() < self.min:
        #    self.min = x.min()
        #    print(self.min, self.max)

        #if x.max() > self.max:
        #    self.max = x.max()
        #    print(self.min, self.max)
        x = x.repeat_interleave(repeats=self.n_neurons_per_input, dim=1)
        # calculate gauss activation per map-neuron
        output =  torch.exp(-0.5 * ((x - self.peaks) / self.sigmas) ** 2)
        return output


class MRBF(torch.nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
            """

    def __init__(self, in_features, out_features):
        super(MRBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        #print(input.shape, size)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        #log_sigmas = self.log_sigmas.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        #distances = ((x - c).pow(2) / torch.exp(log_sigmas).pow(2)).sum(-1).pow(0.5)
        #
        output = torch.exp(-1*distances.pow(2))
        return output

def test_mrbf():
    # test the MRBF layer on griewank function
    # https://www.sfu.ca/~ssurjano/griewank.html
    x = torch.linspace(-10, 10, 1000)
    # y is the griewank function
    y = 1 + (x**2)/4000 - torch.cos(x/torch.sqrt(torch.arange(1, 1001, dtype=torch.float)))
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    print(x.shape, y.shape)
    # create the model
    network = nn.Sequential(
        #MRBF(1, 100),
        RBFLayer(1, 20),
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 1))
    
    # create the optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    # train the model
    for i in range(2000):
        optimizer.zero_grad()
        y_pred = network(x)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Loss: ", loss.item())
    # plot the results
    import matplotlib.pyplot as plt
    plt.plot(x.detach().numpy(), y.detach().numpy(), label="Griewank")
    plt.plot(x.detach().numpy(), y_pred.detach().numpy(), label="MRBF")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    test_mrbf()
