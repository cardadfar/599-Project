import numpy as np
import env

NUM_STEPS = 5

def globalRewardFunc(actions):
    
    reward = 0
    for a in actions:
        if a == 'cooperate':
            reward += 1
    return reward

def main():

    actions = ['defect', 'cooperate']
    localRewards = [8, 0]
    agents = [env.CitizenAgent(actions, localRewards, i) for i in range(10)]
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

