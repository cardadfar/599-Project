import numpy as np
import matplotlib.pyplot as plt

import json, glob, os

_, axs = plt.subplots(3, figsize=(12, 5))


path = 'virusEnv'
b2_check = 0.0
for filename in glob.glob(os.path.join(path, '*.json')):
    with open(os.path.join(os.curdir, filename), 'r') as f: 
        data = json.load(f)

        if(data['b2'] == b2_check):
            
            b1b2 = str(data['b1']) + ',' + str(data['b2'])

            axs[0].plot(data['epoch'], data['globalReward'], label=str(b1b2))
            axs[1].plot(data['epoch'], data['q_std'], label=str(b1b2))
            axs[2].plot(data['epoch'], data['q_raw_std'], label=str(b1b2))
        


axs[0].legend()
plt.setp(axs[0], ylabel='globalReward')
plt.setp(axs[1], ylabel='q_std')
plt.setp(axs[2], ylabel='q_std_raw')
plt.xlabel('epoch')
axs[0].set_title('%d Agent, %d Action %s' % (7, 5, path))
plt.savefig('plots/%s_%d_Agent_%d_Action_b2_%.2f.png' % (path, 7, 5, b2_check))
plt.show()


