import numpy as np
import torch
import gym
import time
from dqnAgent import DQNAgent
from transition import Transition

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    agent = DQNAgent([8], 4)

    num_episodes = 25_000
    save_every = 50

    rewards = []
    for episode in range(1, num_episodes+1):
        state = env.reset()
        done = False

        start_time = time.time()
        ep_rewards = []
        while not done:
            action = agent.get_action(state, False)
            new_state, reward, done, _ = env.step(action)
            #env.render()

            agent.update_mem(Transition(state, action, new_state, reward, done))
            agent.train()

            state = new_state

            ep_rewards.append(reward)
        
        rewards.append(np.sum(ep_rewards))
        print(f'ep: {episode}, time: {round(time.time() - start_time, 3)}s, ep_reward: {round(np.sum(ep_rewards),2)}, avg_reward: {round(np.mean(rewards[-25:]),2)} epsilon: {round(agent.epsilon,2)}')

        if episode % save_every == 0:
            torch.save(agent.model.state_dict(), f'models/episode={episode} avg_reward={round(np.mean(rewards[episode-save_every:]),2)}')