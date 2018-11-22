"""

main.py

Author: Anson Wong  / git: ankonzoid, Ben Cottier / git: bencottier

Perform experiments with simple multi-armed bandit problems.

"""
from __future__ import print_function, division
from agents import (
    ExploreCommitAgent,
    EpsilonGreedyAgent,
    FPLAgent,
    Exp3Agent,
    UCBAgent
) 
from bandits import Bandit

import numpy as np
import matplotlib.pyplot as plt


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


def main():
    # =========================
    # Settings
    # =========================
    r_min = 0  # reward on failure
    r_max = 1  # reward on success
    bandit_probs = [0.80, 0.75, 0.65, 0.60, 0.60,
                    0.50, 0.45, 0.25, 0.10, 0.10]  # success probability
    N_experiments = 2000  # number of experiments to perform
    N_episodes = 10000  # number of episodes per experiment
    epsilon = 0.1  # probability of random exploration (fraction)
    agent_index = {"exc": ExploreCommitAgent, "egd": EpsilonGreedyAgent, 
                   "ex3": Exp3Agent, "fpl": FPLAgent, "ucb": UCBAgent}
    out_path = "output/binomial/3"
    save_fig = True  # if false -> plot, if true save as file in same directory

    # =========================
    # Start multi-armed bandit simulation
    # =========================
    N_bandits = len(bandit_probs)
    grand_totals = {}
    # Test out all the different agents
    for agent_type in agent_index.keys():
        print("Running multi-armed bandits with N_bandits = {}, agent = {}, epsilon = {}".format(
                N_bandits, agent_type, epsilon))
        reward_history_avg = np.zeros(N_episodes)  # reward history experiment-averaged
        reward_experiment_avg = np.zeros(N_experiments)  # reward history episode-averaged
        action_history_sum = np.zeros((N_episodes, N_bandits))  # sum action history
        for i in range(N_experiments):
            bandit = Bandit(bandit_probs)  # initialize bandits
            if "exc" in agent_type.lower():  # special case for the parameter
                agent = agent_index[agent_type](bandit, int(N_episodes * epsilon / N_bandits))
            else:
                agent = agent_index[agent_type](bandit, epsilon)  # initialize agent
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

        total_avg = np.mean(reward_history_avg)
        total_std = np.std(reward_history_avg)
        print("grand total avg = {}".format(total_avg))
        print("grand total var = {}".format(total_std))
        grand_totals[agent_type] = (total_avg, total_std)

        # =========================
        # Plot reward history results
        # =========================
        plt.figure()
        plt.plot(reward_history_avg)
        plt.xlabel("Episode number")
        plt.ylabel("Rewards collected")
        plt.title("Bandit reward history averaged over {} experiments (epsilon = {})".format(N_experiments, epsilon))
        ax = plt.gca()
        ax.set_xscale("log", nonposx='clip')
        plt.xlim([1, N_episodes])
        plt.ylim([r_min, r_max])
        if save_fig:
            output_file = "{}/rewards_{}.png".format(out_path, agent_type)
            plt.savefig(output_file, bbox_inches="tight")
        else:
            plt.show()

        # =========================
        # Plot action history results
        # =========================
        plt.figure()
        for i in range(N_bandits):
            action_history_sum_plot = 100 * action_history_sum[:,i] / N_experiments
            plt.plot(list(np.array(range(len(action_history_sum_plot)))+1),
                    action_history_sum_plot,
                    linewidth=2.0,
                    label="Bandit #{}".format(i+1))
        plt.title("Bandit action history averaged over {} experiments (epsilon = {})".format(N_experiments, epsilon))
        plt.xlabel("Episode Number")
        plt.ylabel("Bandit Action Choices (%)")
        plt.legend(loc='upper left', shadow=True, fontsize=8)
        ax = plt.gca()
        ax.set_xscale("log", nonposx='clip')
        plt.xlim([1, N_episodes])
        plt.ylim([0, 100])
        if save_fig:
            output_file = "{}/actions_{}.png".format(out_path, agent_type)
            plt.savefig(output_file, bbox_inches="tight")
        else:
            plt.show()

        # =========================
        # Plot experiment distribution data
        # =========================
        plt.figure()
        hist, bins = np.histogram(reward_experiment_avg, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.title("Average experiment reward distribution for {} experiments, "
                "epsilon = {}".format(N_experiments, epsilon))
        plt.xlabel("Average experiment reward")
        plt.ylabel("Frequency")
        if save_fig:
            output_file = "{}/hist_{}.png".format(out_path, agent_type)
            plt.savefig(output_file, bbox_inches="tight")
        else:
            plt.show()
    
    # Write summary of results to file
    with open("{}/results.txt".format(out_path), "w") as f:
        f.write("N_bandits = {}\n".format(N_bandits))
        f.write("bandit_probs = {}\n".format(bandit_probs))
        f.write("N_experiments = {}\n".format(N_experiments))
        f.write("N_episodes = {}\n".format(N_episodes))
        f.write("epsilon = {}\n".format(epsilon))
        f.write("\n")
        for agent_type in agent_index.keys():
            f.write("{}\n".format(agent_type))
            avg, std = grand_totals[agent_type]
            f.write("reward_avg = {}\n".format(avg))
            f.write("reward_std = {}\n".format(std))
            f.write("\n")

# Driver
if __name__ == "__main__":
    main()
