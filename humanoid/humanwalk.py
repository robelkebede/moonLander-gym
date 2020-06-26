
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import sys
import os
path = os.getcwd()
lib_dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.insert(1,lib_dir) 
from dqn.agent import Agent

ENV_NAME = 'Humanoid-v2'
env = gym.make(ENV_NAME)

env.seed(0)
agent = Agent(state_size=376, action_size=17, seed=0)
#print(env.action_space)
#print(env.observation_space)

def dqn(n_episodes=500, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    
    scores = []                        
    scores_window = deque(maxlen=100)  
    eps = eps_start                   
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            scores_window.append(score)       
            scores.append(score)             
            eps = max(eps_end, eps_decay*eps) # decay
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                break

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
