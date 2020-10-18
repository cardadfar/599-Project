import numpy as np

UCB_INIT = 100
EPS = 0.1

class CitizenAgent():

    def __init__(self, actions, localRewards, agentIdx):
        # Actions:[Defect, Cooperate]
        self.num_actions = len(actions)
        self.actions = actions
        self.localRewards = localRewards
        self.localPenalty = np.zeros(self.num_actions)
        self.c = np.ones(self.num_actions)
        self.n = 1
        self.q = np.zeros(self.num_actions) + UCB_INIT
        self.agentIdx = agentIdx


    def getAction(self):

        if(np.random.uniform() > EPS):
            # exploit
            self.recentAction = np.argmax(self.q / self.c - self.localPenalty)
        else:
            # explore
            self.recentAction = np.random.randint(0, high=self.num_actions)
        self.c[self.recentAction] += 1
        self.n += 1
        return self.actions[self.recentAction]

    def updateQ(self, globalReward):

        localReward = self.localRewards[self.recentAction]
        localPenalty = self.localPenalty[self.recentAction]
        self.q[self.recentAction] += localReward + globalReward - localPenalty

    def getBestQ(self):
        actualQ = self.q / self.c - self.localPenalty
        return np.max(actualQ), np.argmax(actualQ)

    def updatePenalty(self, actionIndex, penaltyAmount):
        self.localPenalty[actionIndex] = penaltyAmount

    def printAgent(self):

        print('-------------------------------')
        print("Agent ID: " + str(self.agentIdx))
        print("Q: " + str(self.q / self.c  - self.localPenalty))
        print("C: " + str(self.c - 1) + " out of " + str(self.n - 1) + " steps")



class LeaderAgent():

    def __init__(self, agents):
        self.agents = agents

    def penalize(self):
        
        maxAgent = self.agents[0]
        maxActionIdx = 0
        maxQ = 0
        qValues = []

        for i in range(len(self.agents)):
            q, qIdx = self.agents[i].getBestQ()
            qValues.append(q)
            if q > maxQ:
                maxQ = q
                maxAgent = self.agents[i]
                maxActionIdx = qIdx

        #print(qValues)
        penalty = np.std(qValues)
        #print(penalty)
        maxAgent.updatePenalty(maxActionIdx, penalty)


class Environment():
    
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
        return globalReward

    def updateQ(self, globalReward):
        for agent in self.agents:
            agent.updateQ(globalReward)

    def printAgents(self):
        for agent in self.agents:
            agent.printAgent()


