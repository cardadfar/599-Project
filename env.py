import numpy as np

NUM_STATES = 2
NUM_AGENTS = 2
NUM_ACTIONS = 2

class CitizenAgent():

    def __init__(self):
        
        # Actions:[Cooperate, Defect]
        self.state_space = np.arange(NUM_STATES)
        self.action_space = np.arange(NUM_ACTIONS)
        self.mixed_strategy = np.zeros(NUM_ACTIONS)
        self.mixed_strategy[0] = 1

        self.direct_reward = np.zeros(NUM_ACTIONS)


class Env():

    def __init__(self):

        self.agents = [CitizenAgent() for _ in range(NUM_AGENTS)]
    
    def getRewardVector(self, state, actions):
        directReward = None # state and actions
        indirectReward = None # function of global reward

    def directReward(self, state):
        

env = Env()


