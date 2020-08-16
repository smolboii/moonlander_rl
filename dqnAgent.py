import random
import numpy as np
import torch
from torch import nn
from neuralNetwork import NeuralNetwork
from collections import deque

class DQNAgent():

    def __init__(self, state_dims, n_actions, discount_factor=0.99, epsilon=1, epsilon_decay=0.99975, min_epsilon=0.01, target_update_freq=5, max_mem=1_000_000, min_mem=1_000, batch_size=64):
        self.state_dims = state_dims
        self.n_actions = n_actions

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
        self.target_update_counter = 0

        self.mem_cntr = 0
        self.max_mem = max_mem
        self.replay_mem = deque(maxlen=max_mem)
        
        self.min_mem = min_mem
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def get_action(self, observation, train):
        if train and random.random() < self.epsilon:
            return random.randrange(0, self.n_actions)
        else:
            state = torch.tensor([observation]).to(self.model.device)
            actions = self.model(state)
            return torch.argmax(actions).item()

    def update_mem(self, transition):
        self.replay_mem.append(transition)
    
    def train(self):
        if len(self.replay_mem) < self.min_mem:
            return

        self.optimizer.zero_grad()
        
        batch_size = min(self.batch_size, len(self.replay_mem))
        batch_index = np.random.choice(len(self.replay_mem), batch_size, replace=False)
        batch = [self.replay_mem[i] for i in batch_index]

        state_batch = torch.tensor([t.state for t in batch]).to(self.model.device)
        new_state_batch = torch.tensor([t.new_state for t in batch]).to(self.model.device)
        reward_batch = torch.tensor([t.reward for t in batch]).to(self.model.device)
        done_batch = torch.tensor([t.done for t in batch]).to(self.model.device)
        action_batch = [t.action for t in batch]

        q_vals = self.model(state_batch)[np.arange(batch_size), action_batch]
        q_next = self.target_model(new_state_batch)
        q_next[done_batch] = 0.0
        
        q_targets = reward_batch + self.discount_factor * torch.max(q_next, dim=1)[0]

        loss = self.loss_fn(q_vals, q_targets)
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.target_update_counter = 0
            self.target_model.load_state_dict(self.model.state_dict())
            

            



