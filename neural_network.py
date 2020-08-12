import torch
from torch import nn
 
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(128, 4)

    def forward(self, X):
        x = self.fc1(X)
        x = self.fc2(x)
        x = self.fc3(x)
        return x