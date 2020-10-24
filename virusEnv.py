import numpy as np
from envTemplate import *

NUM_AGENTS = 100

def globalRewardFunc(actions):
    
    reward = 0
    for a in actions:
        if a == 1:
            reward += 1
    return reward

def setup():

    agents = []
    sk = SharedKnowledge(NUM_AGENTS, 2)
    for i in range(NUM_AGENTS):
        actions = [0, 1]
        localRewards = np.random.randint(0, high=100, size=len(actions))
        agent = CitizenAgent(sk, actions, localRewards, i, NUM_AGENTS)
        agents.append(agent)

    return agents, sk, globalRewardFunc

