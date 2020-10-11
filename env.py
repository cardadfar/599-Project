import numpy as np

NUM_STATES = 2
NUM_AGENTS = 2
NUM_ACTIONS = 2

class CitizenAgent():

    def __init__(self):
        
        # Actions:[Defect, Cooperate]
        self.state_space = np.arange(NUM_STATES)
        self.action_space = np.arange(NUM_ACTIONS)
        self.mixed_strategy = np.zeros(NUM_ACTIONS)
        self.mixed_strategy[0] = 1

        self.direct_reward = np.zeros(NUM_ACTIONS)

class State():
    def __init__(self, directRewards, globalFunc):
        self.directReward = directRewards
        self.globalFunction = globalFunc

    def getDirectRewards(self):
        return self.directReward

    def getGlobalReward(self, actions):
        return self.globalFunction(actions)

class Env():
    # stateMatrix is a list of state objects
    def __init__(self, stateMatrix):
        self.agents = [CitizenAgent() for _ in range(NUM_AGENTS)]
        self.states = stateMatrix

    def getRewardVector(self, state, actions):
        stateDirectReward = self.states[state].getDirectRewards()
        directReward = [stateDirectReward[i] for i in actions]

        indirectReward = self.states[state].getGlobalReward(actions)
        return directReward + indirectReward

