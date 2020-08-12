import gym
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqn_agent import DQNAgent

env = gym.make('LunarLander-v2')

epsilon = 1
EPSILON_DECAY = 0.9975
MIN_EPSILON = 0.01

NUM_EPISODES = 25_000
RECORD_EVERY = 100

agent = DQNAgent()
agent.model.load_state_dict(torch.load('models\episode=1100-avg_reward=-1.6846323993104846.pt'))
agent.target_model.load_state_dict(torch.load('models\episode=1100-avg_reward=-1.6846323993104846.pt'))
for episode_lower_bound in range(1, NUM_EPISODES + 1, RECORD_EVERY):

    rewards_history = []
    for episode in tqdm(range(episode_lower_bound, episode_lower_bound + RECORD_EVERY)):

        state = env.reset()
        done = False

        episode_rewards = []
        while not done:
            if random.random() > epsilon:
                action = env.action_space.sample()
            else:
                action = torch.argmax(agent.get_qs(np.copy(state))).item()
            
            new_state, reward, done, info = env.step(action)

            if episode % RECORD_EVERY == 0:
                env.render()

            episode_rewards.append(reward)
            agent.update_memory({
                'state': state,
                'action': action,
                'reward': reward,
                'new_state': new_state,
                'done': done
            })
            agent.train(done)

            state = new_state

        rewards_history.append(np.sum(episode_rewards))

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

    env.close()
    
    avg_reward = np.mean(rewards_history)
    print(avg_reward)
    torch.save(agent.model.state_dict(), f'models/episode={episode}-avg_reward={avg_reward}.pt')
    env.close()
    rewards_history = []





        

        

            
