import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):

    def __init__(self, n_features, n_outputs):
        super(SimpleNet, self).__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs

        self.input_layer = nn.Linear(n_features, 500)
        self.hidden_layer1 = nn.Linear(500, 250)
        self.hidden_layer2 = nn.Linear(250, 100)
        self.output_layer = nn.Linear(100, self.n_outputs)

    def forward(self, x):

        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer1(x)
        x = F.relu(x)
        x = self.hidden_layer2(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x


class FeedFowardNet(nn.Module):

    def __init__(self, n_features, n_outputs, n_hidden=(50, 100, 50, 25)):
        super(FeedFowardNet, self).__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.layers = []

        shape = (n_features,) + n_hidden
        for i in range(len(shape)-1):
            self.layers.append(nn.Linear(shape[i], shape[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = nn.Linear(self.n_hidden[-1], self.n_outputs)(x)
        return x

