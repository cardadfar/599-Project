import numpy as np
import env

NUM_STEPS = 500
NUM_EPOCHS = 50
NUM_AGENTS = 10

def globalRewardFunc(actions):
    
    reward = 0
    for a in actions:
        if a == 'cooperate':
            reward += 1
    return reward

def main():

    agents = []
    for i in range(NUM_AGENTS):
        actions = ['defect', 'cooperate']
        localRewards = [np.random.randint(0,high=100), np.random.randint(0,high=100)]
        agent = env.CitizenAgent(actions, localRewards, i)
        agents.append(agent)
    leader = env.LeaderAgent(agents)
    testEnv = env.Environment(agents, leader, globalRewardFunc)

    for _ in range(NUM_EPOCHS):
        for _ in range(NUM_STEPS):
            actions = testEnv.getActions()
            globalReward = testEnv.getRewards(actions)
            testEnv.updateQ(globalReward)

        leader.penalize()

    testEnv.printAgents()

    #test stage
    actions = testEnv.getActions()
    globalReward = testEnv.getRewards(actions)
    print("Global Reward: " + str(globalReward))

    return 0

if __name__ == "__main__":
    main()

