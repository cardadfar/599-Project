import numpy as np

UCB_INIT = 100
EPS = 0.5

class SharedKnowledge():

    def __init__(self, numAgents, numActions):
        self.numAgents = numAgents
        self.numActions = numActions
        self.c = np.ones((numAgents, numActions))
        self.n = 1

    def recordActions(self, actions):
        self.n += 1
        for agentID in range(self.numAgents):
            action = actions[agentID]
            self.c[agentID, action] += 1

class CitizenAgent():

    def __init__(self, sk, actions, localRewards, agentIdx, numAgents):
        # Actions:[Defect, Cooperate]
        self.num_actions = len(actions)
        self.actions = actions
        self.localRewards = localRewards
        self.localPenalty = np.zeros(self.num_actions)
        self.sk = sk
        self.q = np.zeros(self.num_actions) + UCB_INIT
        self.rawQ = np.zeros(self.num_actions) + UCB_INIT
        self.agentIdx = agentIdx
        self.gamma = 1.0

    def getAction(self):

        if(np.random.uniform() > EPS):
            # exploit
            self.recentAction = np.argmax(self.q / self.sk.c[self.agentIdx])
        else:
            # explore
            self.recentAction = np.random.randint(0, high=self.num_actions)
        return self.recentAction
        
    def getBestAction(self):
        actualQ = self.q / self.sk.c[self.agentIdx]
        return np.argmax(actualQ)

    def updateQ(self, globalReward, globalPenalty):

        localReward = self.localRewards[self.recentAction]
        localPenalty = self.localPenalty[self.recentAction]
        newQ = localReward + globalReward - localPenalty - globalPenalty
        rawQ = localReward + globalReward
        self.q[self.recentAction] = self.q[self.recentAction] * self.gamma + newQ
        self.rawQ[self.recentAction] = self.rawQ[self.recentAction] * self.gamma + rawQ

    def getBestQ(self):
        actualQ = self.q / self.sk.c[self.agentIdx]
        return np.max(actualQ)

    def getBestRawQ(self):
        rawQ = self.rawQ / self.sk.c[self.agentIdx]
        return np.max(rawQ)
    
    def getLocalReward(self, action):
        return self.localRewards[action] - self.localPenalty[action]

    def addPenalty(self, actionIndex, penaltyAmount):
        self.localPenalty[actionIndex] += penaltyAmount

    def printAgent(self):

        print('-------------------------------')
        print("Agent ID: " + str(self.agentIdx))
        print("Q: " + str(self.q / self.sk.c[self.agentIdx]))
        print("R: " + str(self.rawQ / self.sk.c[self.agentIdx]))
        #print("C: " + str(self.c - 1) + " out of " + str(self.n - 1) + " steps")
        print("L: " + str(self.localPenalty))



class LeaderAgent():

    def __init__(self, agents, beta1=1.0, beta2=1.0):
        self.agents = agents
        self.globalPenalty = {}
        self.beta1 = beta1
        self.beta2 = beta2

    def penalize(self):
        
        bestQValues = []
        bestActions = []

        for i in range(len(self.agents)):
            q = self.agents[i].getBestQ()
            a = self.agents[i].getBestAction()
            r = self.agents[i].getLocalReward(a)

            bestQValues.append(r)
            bestActions.append(a)

        maxAgentID = np.argmax(bestQValues)

        penalty = np.std(bestQValues)
        mean = np.mean(bestQValues)
        #self.agents[maxAgentID].addPenalty(bestActions[maxAgentID], self.beta1 * penalty)
        for i in range(len(self.agents)):
            self.agents[i].addPenalty(bestActions[i], self.beta1 * max(0, (bestQValues[i] - mean)))

        self.globalPenalty[tuple(bestActions)] = self.beta2 * penalty
        return penalty

    def addGloablPenalty(self, actions, penalty):

        actions = tuple(actions)
        if actions in self.globalPenalty:
            self.globalPenalty[actions] += penalty
        else:
            self.globalPenalty[actions] = penalty

    def getGlobalPenalty(self, actions):

        actions = tuple(actions)
        if actions in self.globalPenalty:
            return self.globalPenalty[actions]
        else:
            return 0
        


class Environment():
    
    def __init__(self, sk, agents, leader, globalRewardFunc):
        self.num_agents = len(agents)
        self.agents = agents
        self.leader = leader
        self.globalRewardFunc = globalRewardFunc
        self.sk = sk

    def getActions(self):
        actions = []
        for agent in self.agents:
            actions.append(agent.getAction())
        return actions

    def getBestActions(self):
        actions = []
        for agent in self.agents:
            actions.append(agent.getBestAction())
        return actions

    def getRewards(self, actions):
        self.sk.recordActions(actions)
        globalReward = self.globalRewardFunc(actions)
        return globalReward

    def updateQ(self, globalReward, globalPenalty):
        for agent in self.agents:
            agent.updateQ(globalReward, globalPenalty)

    def printAgents(self):
        for agent in self.agents:
            agent.printAgent()
    
    def getQStd(self):
        qs = []
        for agent in self.agents:
            q = agent.getBestQ()
            qs.append(q)
        return np.std(qs)
    
    def getQRawStd(self):
        qs = []
        for agent in self.agents:
            q = agent.getBestRawQ()
            qs.append(q)
        return np.std(qs)


