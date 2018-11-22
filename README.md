# Compare MAB

This is an extension/modification of Anson Wong's multi-armed bandit example which you can find [here](https://github.com/ankonzoid/LearningX/tree/master/classical_RL/MAB)

We implement several different agents such as epsilon-greedy, EXP3, and UCB to solve simple multi-armed bandit problems, comparing and visualising their behaviour.

### Usage:

> python main.py

In the default case we train on 2,000 experiments with 10,000 episodes per experiment. The exploring parameter is `0.1` and 10 bandits are intialised with success probabilities of `{0.80, 0.75, 0.65, 0.60, 0.60, 0.50, 0.45, 0.25, 0.10, 0.10}`. These are in order of success to make it easy to track performance. Bandit #1 should be selected as the best on average, with bandit #2 running second, and bandit #3 as a far third.

### Example Output

Along with `actions.png` (agent actions), `rewards.png` (agent collected rewards) and `hist.png` (collected reward histogram), you should also get an output along the lines of

```
Running multi-armed bandits with N_bandits = 10 and agent epsilon = 0.1
[Experiment 1/100]
  N_episodes = 10000
  bandit choice history = [1 4 4 ... 4 4 4]
  reward history = [0 1 1 ... 1 1 1]
  average reward = 0.7761

...
...
...

[Experiment 100/100]
  N_episodes = 10000
  bandit choice history = [ 2 10  6 ...  4  4  4]
  reward history = [1 0 1 ... 1 1 1]
  average reward = 0.764

reward history avg = [0.51 0.43 0.63 ... 0.79 0.75 0.73]
```

### Libraries required:

* numpy
* matplotlib

### Author

Anson Wong, Ben Cottier
