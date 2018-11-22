"""
probability.py

author: Ben Cottier / git: bencottier

Analyse probability underlying MAB.

"""
import numpy as np
import math
from scipy.stats import binom
import matplotlib.pyplot as plt


def binom_diff(z, n1, p1, n2, p2):
    """
    Compute the probability that the difference of two binomial 
    random variables, parameterised by (n1, p1) and (n2, p2), is z.
    """
    if z >= 0:
        # Sum over all possible ways z could be non-negative
        return np.sum(np.array([binom.pmf(i + z, n1, p1) * 
            binom.pmf(i, n2, p2)]) for i in range(n1))
    else:
        # Sum over all possible ways z could be negative
        return np.sum(np.array([binom.pmf(i, n1, p1) * 
            binom.pmf(i + z, n2, p2)]) for i in range(n2))


if __name__ == '__main__':
    # Question: what is the probability that
    # a binomial variable with this success rate...
    p1 = 0.75
    # ...has a greater number of successes than another...
    p2 = 0.80
    # ... over this many trials?
    n = 100
    s = np.zeros(n)
    for i, z in enumerate(np.arange(-1, -(n + 1), -1)):
        if i == 0:
            x = 0.0
        else:
            x = s[i - 1]
        s[i] = x + binom_diff(z, n, p1, n, p2)[0]

    print("Probability X1 (p={}, n={}) > X2 (p={}, n={}): {}".format(
            p1, n, p2, n, s[-1]))
    plt.plot(np.arange(0, n), s)
    plt.xlabel("Trials")
    plt.ylabel("$P(X_1 > X_2)$")
    plt.show()
