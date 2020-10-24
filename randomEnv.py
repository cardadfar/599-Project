import numpy as np
from envTemplate import *

NUM_AGENTS = 7
NUM_ACTIONS = 5

s = np.ones(NUM_AGENTS) * NUM_ACTIONS
s = s.astype(int)
s = tuple(s)

test = np.random.random_sample( s ) * 100


def globalRewardFunc(actions):
    
    return test[tuple(actions)]

def setup():

    agents = []
    sk = SharedKnowledge(NUM_AGENTS, NUM_ACTIONS)
    for i in range(NUM_AGENTS):
        actions = np.arange(NUM_ACTIONS)
        localRewards = np.random.randint(0, high=100, size=len(actions))
        agent = CitizenAgent(sk, actions, localRewards, i, NUM_AGENTS)
        agents.append(agent)

    return agents, sk, globalRewardFunc

