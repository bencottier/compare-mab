"""

bandits.py (author: Ben Cottier / git: bencottier)

Defines bandits in the multi-arm bandit problem.

"""
import numpy as np

class Bandit:

    def __init__(self, bandit_probs, num_trials=1):
        self.N = len(bandit_probs)  # number of bandits
        self.prob = bandit_probs  # success probabilities for each bandit
        self.num_trials = num_trials

    # Get reward (1 for success, 0 for failure)
    def get_reward(self, action):
        rand = np.random.random()  # [0.0,1.0)
        reward = 0
        for _ in range(self.num_trials):
            reward += 1 if (rand < self.prob[action]) else 0
        return reward
