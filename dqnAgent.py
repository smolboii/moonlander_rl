import random
import np
import torch
from torch import nn
from neuralNetwork import NeuralNetwork
from collections import deque

class DQNAgent():

    def __init__(self, state_dims, n_actions, discount_factor=0.99, epsilon=1, epsilon_decay=0.9975, min_epsilon=0.01, target_update_freq=5, max_mem=1_000_000, min_mem=1_000, batch_size=32):
        self.state_dims = state_dims
        self.n_actions = n_actions

        self.network = NeuralNetwork()

        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.model = NeuralNetwork(*state_dims, 256, 256, 128, n_actions)
        self.model.to(self.model.device)

        self.target_model = NeuralNetwork(*state_dims, 256, 256, 128, n_actions)
        self.target_model.to(self.target_model.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_update_freq = target_update_freq

        self.replay_mem = deque(maxlen=max_mem)
        self.min_mem = min_mem
        self.batch_size = batch_size

        self.optimizer = nn.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state, train):
        if train and random.random() < self.epsilon:
            return random.randint(0, self.n_actions)
        else:
            return torch.argmax(self.model(torch.Tensor(state).to(self.model.device))).item()

    def update_mem(self, transition):
        self.replay_mem.append(transition)
    
    def train(self):
        if len(self.replay_mem) < self.min_mem:
            return
        
        X = []
        Y = []

        batch_size = min(self.batch_size, len(self.replay_mem))
        for i in np.random.choice(len(self.replay_mem), batch_size, replace=False):
            transition = self.replay_mem[i]

            X.append(transition.state)

            q_vals = self.model(torch.Tensor(transition.state).to(self.model.device))
            target_q_vals = self.target_model(torch.Tensor(transition.new_state).to(self.target_model.device))

            q_vals[transition.action] = transition.reward
            if not transition.done:
                q_vals[transition.action] += self.discount_factor * torch.max(target_q_vals).item()

            Y.append(q_vals)

        self.optimizer.zero_grad()
        outputs = self.model(torch.stack(X).to(self.model.device))
        loss = self.loss_fn(outputs, torch.stack(Y))

        loss.backwards()
        self.optimizer.step()
            

            



