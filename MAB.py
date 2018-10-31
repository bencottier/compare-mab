"""

 multiarmed_bandits.py  (author: Anson Wong / git: ankonzoid)

 We solve the multi-armed bandit problem using a classical epsilon-greedy
 agent with reward-average sampling to estimate the action-value Q.
 This algorithm follows closely with the notation of Sutton's RL textbook.

 We set up up bandits with a fixed probability distribution of success,
 and receive stochastic rewards from the bandits of +1 for success,
 and 0 reward for failure.

 The update rule for our action-values Q is:

   Q(a) <- Q(a) + 1/(k+1) * (R(a) - Q(a))

 where

   Q(a) = current value estimate of action "a"
   k = number of times action "a" was chosen so far
   R(a) = reward of sampling action bandit (bandit) "a"

 The derivation of the above Q incremental implementation update:

   Q(a;k+1)
   = 1/(k+1) * (R(a_1) + R(a_2) + ... + R(a_k) + R(a))
   = 1/(k+1) * (k*Q(a;k) + R(a))
   = 1/(k+1) * ((k+1)*Q(a;k) + R(a) - Q(a;k))
   = Q(a;k) + 1/(k+1) * (R(a) - Q(a;k))

"""
import numpy as np
import matplotlib.pyplot as plt
import agents


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


def main():
    # =========================
    # Settings
    # =========================
    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10,
                    0.25, 0.60, 0.45, 0.75, 0.65]  # bandit probabilities of success
    N_experiments = 1000  # number of experiments to perform
    N_episodes = 10000  # number of episodes per experiment
    epsilon = 0.1  # probability of random exploration (fraction)
    save_fig = False  # if false -> plot, if true save as file in same directory
    agent_type = "fpl"  # To keep track of different experiments
    out_path = "output/comp_bin"
    r_min = 0
    r_max = 1

    # =========================
    # Define an experiment
    # =========================
    def experiment(agent, bandit, N_episodes):
        action_history = []
        reward_history = []
        for episode in range(N_episodes):
            # Choose action from agent (from current Q estimate)
            action = agent.get_action(bandit)
            # Pick up reward from bandit for chosen action
            reward = bandit.get_reward(action)
            # Update Q action-value estimates
            agent.update_Q(action, reward)
            # Append to history
            action_history.append(action)
            reward_history.append(reward)
        return (np.array(action_history), np.array(reward_history))

    # =========================
    # Start multi-armed bandit simulation
    # =========================
    N_bandits = len(bandit_probs)
    print("Running multi-armed bandits with N_bandits = {}, agent = {}, epsilon = {}".format(
            N_bandits, agent_type, epsilon))
    reward_history_avg = np.zeros(N_episodes)  # reward history experiment-averaged
    reward_experiment_avg = np.zeros(N_experiments)  # reward history episode-averaged
    action_history_sum = np.zeros((N_episodes, N_bandits))  # sum action history
    for i in range(N_experiments):
        bandit = Bandit(bandit_probs)  # initialize bandits
        agent = agents.FPLAgent(bandit, epsilon)  # initialize agent TODO
        (action_history, reward_history) = experiment(agent, bandit, N_episodes)  # perform experiment

        if (i + 1) % (N_experiments / 100) == 0:
            print("[Experiment {}/{}]".format(i + 1, N_experiments))
            print("  N_episodes = {}".format(N_episodes))
            print("  bandit choice history = {}".format(
                action_history + 1))
            print("  reward history = {}".format(
                reward_history))
            print("  average reward = {}".format(np.sum(reward_history) / len(reward_history)))
            print("")
        # Sum up experiment reward (later to be divided to represent an average)
        reward_history_avg += reward_history
        reward_experiment_avg[i] = np.mean(reward_history)
        # Sum up action history
        for j, (a) in enumerate(action_history):
            action_history_sum[j][a] += 1

    reward_history_avg /= np.float(N_experiments)
    print("reward history avg = {}".format(reward_history_avg))
    print("grand total avg = {}".format(np.mean(reward_history_avg)))
    print("grand total var = {}".format(np.var(reward_history_avg)))

    # =========================
    # Plot reward history results
    # =========================
    plt.plot(reward_history_avg)
    plt.xlabel("Episode number")
    plt.ylabel("Rewards collected")
    plt.title("Bandit reward history averaged over {} experiments (epsilon = {})".format(N_experiments, epsilon))
    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    plt.xlim([1, N_episodes])
    plt.ylim([r_min, r_max])
    if save_fig:
        output_file = "{}/rewards_{}_{}.png".format(out_path, agent_type, epsilon)
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()

    # =========================
    # Plot action history results
    # =========================
    for i in range(N_bandits):
        action_history_sum_plot = 100 * action_history_sum[:,i] / N_experiments
        plt.plot(list(np.array(range(len(action_history_sum_plot)))+1),
                 action_history_sum_plot,
                 linewidth=5.0,
                 label="Bandit #{}".format(i+1))
    plt.title("Bandit action history averaged over {} experiments (epsilon = {})".format(N_experiments, epsilon), fontsize=26)
    plt.xlabel("Episode Number")
    plt.ylabel("Bandit Action Choices (%)")
    leg = plt.legend(loc='upper left', shadow=True)
    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    plt.xlim([1, N_episodes])
    plt.ylim([0, 100])
    if save_fig:
        output_file = "{}/actions_{}_{}.png".format(out_path, agent_type, epsilon)
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()

    # =========================
    # Plot experiment distribution data
    # =========================
    hist, bins = np.histogram(reward_experiment_avg, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title("Average experiment reward distribution for {} experiments, "
            "epsilon = {}".format(N_experiments, epsilon))
    plt.xlabel("Average experiment reward")
    plt.ylabel("Frequency")
    if save_fig:
        output_file = "{}/hist_{}_{}.png".format(out_path, agent_type, epsilon)
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()

# Driver
if __name__ == "__main__":
    main()
