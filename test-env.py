import numpy as np
import env

def main():
    testenv = env.Env([env.State(-1*(i+1)*np.arange(NUM_ACTIONS), lambda x: (i+1)*np.sum(x)) for i in range(NUM_STATES)])
    print(testenv.getRewardVector(0, [0, 1]))
    print(testenv.getRewardVector(1, [1, 1]))
    return

if __name__ == "__main__":
    main()

