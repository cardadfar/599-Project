import numpy as np
import env

NUM_STEPS = 500

def globalRewardFunc(actions):
    
    reward = 0
    for a in actions: reward += 2*a
    return reward

def fairnessRewardFunc(actions):
    avg = np.sum(actions) / len(actions)
    return 0

def thresholdingRewardFunc(actions):
    return 0

def testPrisonerGame():
    return 0

def testGradingGame():
    return 0

def fairnessGame():
    return 0

def main():

    localRewards = [9, 0]
    agents = [env.CitizenAgent(localRewards, i) for i in range(10)]
    leader = env.LeaderAgent()
    testEnv = env.Environment(agents, leader, globalRewardFunc)

    for _ in range(NUM_STEPS):
        actions = testEnv.getActions()
        globalReward, penalty = testEnv.getRewards(actions)
        testEnv.updateQ(globalReward, penalty)

    testEnv.printAgents()

    #test stage
    actions = testEnv.getActions()
    globalReward, _ = testEnv.getRewards(actions)
    print("Global Reward: " + str(globalReward))

    return 0

if __name__ == "__main__":
    main()

