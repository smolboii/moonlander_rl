import random
import torch
import torch.optim as optim
from neural_network import NeuralNetwork
from collections import deque

REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 100
UPDATE_TARGET_EVERY = 5

MINIBATCH_SIZE = 32
DISCOUNT_FACTOR = 0.9

device = torch.device('cuda:0')

class DQNAgent():
    def __init__(self):

        self.model = NeuralNetwork()
        self.target_model = NeuralNetwork()
        self.target_model.load_state_dict(self.model.state_dict())

        self.target_model.to(device)
        self.model.to(device)

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

    def train(self, terminal_state):

        if (len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE):
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_func = torch.nn.MSELoss()

        with torch.no_grad():
            curr_qs_list = self.model(torch.Tensor([t['state'] for t in minibatch]).view(-1, 8).to(device))
            new_qs_list = self.target_model(torch.Tensor([t['new_state'] for t in minibatch]).view(-1, 8).to(device))

        X = torch.Tensor([t['state'] for t in minibatch]).view(-1, 8).to(device)
        Y = []

        self.model.eval()
        self.target_model.eval()
        for i, transition in enumerate(minibatch):
            
            new_q = transition['reward']
            if not transition['done']:
                new_q += DISCOUNT_FACTOR * torch.max(new_qs_list[i])
            y = curr_qs_list[i]
            y[transition['action']] = new_q
            Y.append(y)

        self.model.train()
        outputs = self.model(X)
        loss = loss_func(outputs, torch.stack(Y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter % UPDATE_TARGET_EVERY == 0:
            self.target_update_counter = 0
            self.target_model.load_state_dict(self.model.state_dict())
    
    def update_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model(torch.Tensor(state).view(-1, 8).to(device))
    