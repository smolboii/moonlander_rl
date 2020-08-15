import torch
from torch import nn

class NeuralNetwork(nn.Module):

    def __init__(self, n_inputs, layer1_size, layer2_size, layer3_size, n_outputs):

        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(n_inputs, layer1_size),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(layer1_size, layer2_size),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(layer2_size, layer3_size),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(layer3_size, n_outputs)

    def forward(self, X):
        x = self.fc1(X)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)

