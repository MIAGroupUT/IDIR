import torch.nn.functional as F
import torch
from torch import nn
import numpy as np


class Siren(nn.Module):
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    gpu -- (boolean) use GPU when True, CPU when False
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    """

    def __init__(self, layers, weight_init=True, omega=30):
        """Initialize the network."""

        super(Siren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Weight Initialization
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                    else:
                        self.layers[-1].weight.uniform_(
                            -np.sqrt(6 / layers[i]) / self.omega,
                            np.sqrt(6 / layers[i]) / self.omega,
                        )

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega * layer(x))

        # Propagate through final layer and return the output
        return self.layers[-1](x)


class MLP(nn.Module):
    def __init__(self, layers):
        """Initialize the network."""

        super(MLP, self).__init__()
        self.n_layers = len(layers) - 1

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))

        # Propagate through final layer and return the output
        return self.layers[-1](x)
