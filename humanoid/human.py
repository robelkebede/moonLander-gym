import numpy as np
import gym
import torch
import os
import sys
path = os.getcwd()
lib_dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.insert(1,lib_dir) 

from dqn.agent import Agent

ENV_NAME = 'Humanoid-v2'
env = gym.make(ENV_NAME)

np.random.seed(0)
env.seed(0)
nb_actions = 17
agent = Agent(state_size=376, action_size=17, seed=0)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(15):
    state = env.reset()
    while True: 
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        print(reward)
        if done:
            break 
    print("Iteration ",i)
            
env.close()



