"""

agent.py (author: Ben Cottier / git: bencottier)

Defines several RL agents capable of solving a multi-arm bandit problem.

"""
import numpy as np
import math


class Agent:
    """
    Base class
    """

    def __init__(self, bandit, param, anneal=0.0):
        self.param = param
        self.param_init = self.param
        self.anneal = anneal
        self.k = np.zeros(bandit.N, dtype=np.int)  # number of times action was chosen
        self.Q = np.zeros(bandit.N, dtype=np.float)  # estimated value

    # Update Q action-value using:
    # Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))
    def update_Q(self, action, reward):
        self.k[action] += 1  # update action counter k -> k+1
        self.Q[action] += (1./self.k[action]) * (reward - self.Q[action])
        if abs(self.anneal) > 0 and self.param != 0:  # annealing exploration
            self.param = max(0, self.param - self.anneal)


class EpsilonGreedyAgent(Agent):
    """
    Chooses a random action epsilon-fraction of the time,
    and the current best action otherwise.
    """

    def __init__(self, bandit, epsilon, anneal=0.0):
        super(EpsilonGreedyAgent, self).__init__(bandit, epsilon, anneal)

    def get_action(self, bandit, force_explore=False):
        rand = np.random.random()  # [0.0,1.0)
        if (rand < self.param) or force_explore:
            action_explore = np.random.randint(bandit.N)  # explore random bandit
            return action_explore
        else:
            # action_greedy = np.argmax(self.Q)  # exploit best current bandit
            action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
            return action_greedy


class Exp3Agent(Agent):
    """
    Chooses an action with probabilities weighted exponentially with 
    respect to past reward, biased with uniform probability for exploration.

    Exp3 = Exponential weighting for Exploration and Exploitation
    """

    def __init__(self, bandit, gamma):
        self.ws = np.ones(bandit.N, dtype=np.float64)  # weights
        self.ps = np.ones(bandit.N, dtype=np.float64)  # probabilities
        super(Exp3Agent, self).__init__(bandit, gamma)

    def get_action(self, bandit):
        # Set the probability of choosing each arm
        self.ps = (1 - self.param) * (self.ws / np.sum(self.ws)) + self.param / bandit.N
        # Choose an action based on probabilities
        return np.random.choice(np.arange(bandit.N), p=self.ps)

    def update_Q(self, action, reward):
        super(Exp3Agent, self).update_Q(action, reward)
        # Update weight of chosen action exponentially to received reward
        self.ws[action] *= np.exp( (self.param * reward) / (self.ps[action] * len(self.ps)) )


class FPLAgent(Agent):
    """
    Chooses the best action based on cumulative reward perturbed by 
    exponential random noise.

    FPL = Follow the Perturbed Leader
    """

    def __init__(self, bandit, lam):
        self.z = np.zeros(bandit.N)
        super(FPLAgent, self).__init__(bandit, lam)

    def get_action(self, bandit):
        # Generate exponential random noise for each arm
        # Numpy uses beta = 1 / lambda as the parameter
        self.z = np.random.exponential(1. / self.param, len(self.z))
        # Choose the action that maximises noise-perturbed value
        return np.argmax(self.Q + self.z)

    def update_Q(self, action, reward):
        self.k[action] += 1
        self.Q[action] += reward  # simply accumulate reward for each action


class UCBAgent(Agent):
    """
    Chooses the action that maximises an optimistic estimate of the 
    average reward.

    UCB = Upper Confidence Bound
    """
    
    def __init__(self, bandit, c=1.0):
        super(UCBAgent, self).__init__(bandit, c)

    def get_action(self, bandit):
        t = np.sum(self.k)
        if t < bandit.N:
            # Visit all available states initially
            # This is what would happen anyway in the limit of k -> 0
            # For the usual formula below
            # This is a way of avoiding division by 0
            return t  
        else:
            f = 1 + t * (math.log(t))**2
            v = self.Q + self.param * np.sqrt(2 * math.log(f) * 1. / self.k)
            return np.argmax(v)
