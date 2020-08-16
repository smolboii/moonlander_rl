import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, '\n')

class NeuralNetwork(nn.Module):

    def __init__(self, n_inputs, layer1_size, layer2_size, layer3_size, n_outputs, device=device):

        super().__init__()

        self.fc1 = nn.Linear(n_inputs, layer1_size)
        self.fc2 = nn.Linear(layer1_size, layer2_size)
        self.fc3 = nn.Linear(layer2_size, layer3_size)
        self.fc4 = nn.Linear(layer3_size, n_outputs)

        self.device = device

    def forward(self, X):
        x = F.relu(self.fc1(X))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

