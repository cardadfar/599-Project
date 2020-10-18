import numpy as np

UCB_INIT = 100
EPS = 0.1

class CitizenAgent():

    def __init__(self, actions, localRewards, agentIdx):
        # Actions:[Defect, Cooperate]
        self.num_actions = len(actions)
        self.actions = actions
        self.localRewards = localRewards
        self.c = np.ones(self.num_actions)
        self.n = 1
        self.q = np.zeros(self.num_actions) + UCB_INIT
        self.agentIdx = agentIdx


    def getAction(self):

        if(np.random.uniform() > EPS):
            # exploit
            self.recentAction = np.argmax(self.c * self.q)
        else:
            # explore
            self.recentAction = np.random.randint(0, high=self.num_actions)
        self.n += 1
        self.c[self.recentAction] += 1
        return self.actions[self.recentAction]

    def updateQ(self, globalReward, penalty):

        localRewards = self.localRewards[self.recentAction]
        self.q[self.recentAction] = localRewards + globalReward + penalty

    def printAgent(self):

        print('-------------------------------')
        print("Agent ID: " + str(self.agentIdx))
        print("Q: " + str(self.q))
        print("C: " + str(self.c - 1) + " out of " + str(self.n - 1) + " steps")



class LeaderAgent():

    def getPenalty(self, actions):
        return 0


class Environment():
    # stateMatrix is a list of state objects
    def __init__(self, agents, leader, globalRewardFunc):
        self.num_agents = len(agents)
        self.agents = agents
        self.leader = leader
        self.globalRewardFunc = globalRewardFunc

    def getActions(self):

        actions = []
        for agent in self.agents:
            actions.append(agent.getAction())
        return actions

    def getRewards(self, actions):
        globalReward = self.globalRewardFunc(actions)
        penalty = self.leader.getPenalty(actions)
        return globalReward, penalty

    def updateQ(self, globalReward, penalty):
        for agent in self.agents:
            agent.updateQ(globalReward, penalty)

    def printAgents(self):
        for agent in self.agents:
            agent.printAgent()


