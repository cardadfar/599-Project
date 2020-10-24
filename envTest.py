import numpy as np
import matplotlib.pyplot as plt
from envTemplate import * 

import json

NUM_STEPS = 20
NUM_EPOCHS = 100

np.random.seed(17099)

def main():

    envName = "virusEnv"
    saveOutput = True
    createPlot = True

    if envName == "virusEnv":
        import virusEnv as env
    else:
        import randomEnv as env

    agents, sk, globalRewardFunc = env.setup()

    beta1_list = [0.0]
    beta2_list = [0.0]

    for beta2 in beta2_list:
        for beta1 in beta1_list:
            print(beta1, beta2)

            data = {}
            data['envName'] = envName
            data['b1'] = beta1
            data['b2'] = beta2
            data['globalReward'] = []
            data['epoch'] = []
            data['q_std'] = []
            data['q_raw_std'] = []
            data['raw'] = []

            leader = LeaderAgent(agents, beta1=beta1, beta2=beta2)
            testEnv = Environment(sk, agents, leader, globalRewardFunc)

            for e in range(NUM_EPOCHS):

                leader.penalize()

                for _ in range(NUM_STEPS):
                    actions = testEnv.getActions()
                    globalReward = testEnv.getRewards(actions)
                    globalPenalty = leader.getGlobalPenalty(actions)
                    testEnv.updateQ(globalReward, globalPenalty)

                q_std = testEnv.getQStd()
                q_raw_std = testEnv.getQRawStd()
                actions = testEnv.getBestActions()
                globalReward = testEnv.getRewards(actions)
                data['epoch'].append(e)
                data['q_std'].append(q_std)
                data['q_raw_std'].append(q_raw_std)
                data['globalReward'].append(globalReward)

                
                if(e % 10 == 0):
                    print('[%d/%d]' % (e, NUM_EPOCHS))


            if saveOutput:
                file = open("%s/b1=%.2f_b2=%.2f_results.json" % (envName, beta1, beta2),"w+")
                file.truncate(0)
                json.dump(data, file)
                file.close()

            if createPlot:
                _, axs = plt.subplots(3, figsize=(12, 5))
                axs[0].plot(data['epoch'], data['globalReward'], 'b')
                axs[1].plot(data['epoch'], data['q_std'], 'r')
                axs[2].plot(data['epoch'], data['q_raw_std'], 'g')
                plt.setp(axs[0], ylabel='globalReward')
                plt.setp(axs[1], ylabel='q_std')
                plt.xlabel('epoch')
                plt.show()

    return 0

if __name__ == "__main__":
    main()

