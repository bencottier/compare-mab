"""

animated_plots.py

Author: Ben Cottier / git: bencottier

Produce and display animated data.

"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def animate(i, anim, selections, results, values):
    i_iter = i % anim.frames_per_iter  # frames within current iteration
    j = int((i - i_iter) / anim.frames_per_iter)  # current iteration
    sf = anim.section_frames 

    if j == anim.num_iter:  # End animation with neutral display
        anim.value(values[:, :, j])
        return
    
    s = selections[j]
    r = results[j]
    v = values[:, :, j]  # barset, arms, iteration
    if i_iter < sf[0]:
        # Section 1: show current values
        if i_iter == 0:
            anim.value(v)
    elif i_iter < sf[1]:
        # Section 2: animate change in auxilliary value by interpolating
        change = np.array([values[0, :, j + 1] - v[0], np.zeros_like(v[1])])
        frame = i_iter - sf[0]
        total_frames = sf[1] - sf[0]
        v_anim = v
        v_anim += frame * change / total_frames
        anim.update(v_anim)
    elif i_iter < sf[2]:
        # Section 3: indicate selection
        if i_iter == sf[1]:
            anim.select(s)
    elif i_iter < sf[3]:
        # Section 4: indicate result of selection
        if i_iter == sf[2]:
            anim.result(s, r)
    else:  # i_iter >= sf[-1]
        # Last section: animate change in value by interpolating
        change = values[:, :, j + 1] - v
        frame = i_iter - sf[-2]
        total_frames = sf[-1] - sf[-2]
        v_anim = v
        v_anim += frame * change / total_frames
        anim.update(v_anim)


class Animation:

    COLOUR_DEFAULT = 'C0'  # 'C0' blue
    COLOUR_SELECT  = 'C1'  # 'C1' orange
    COLOUR_SUCCESS = 'C2'  # 'C2' greed
    COLOUR_FAILURE = 'C3'  # 'C3' red

    def __init__(self, selections, results, values, iter_start, iter_end,
            fps=30, speed=2):

        fig = plt.figure()

        if iter_start < 0 or iter_start >= len(selections):
            iter_start = 0
        if iter_end < iter_start or iter_end >= len(selections):
            iter_end = len(selections)
        selections = selections[iter_start:iter_end]
        results = results[iter_start:iter_end]
        values = values[:, :, iter_start:iter_end+1]
        self.num_iter = iter_end - iter_start

        self.fps = fps  # desired output frame rate
        section_times = 1./speed * np.array([1, 2, 3, 4, 5])
        self.section_frames = (self.fps * section_times).astype(np.int32)
        self.frames_per_iter = self.section_frames[-1]
        num_frames = self.frames_per_iter * (self.num_iter + 1)

        self.anim = animation.FuncAnimation(
                fig, animate, fargs=(self, selections, results, values), 
                frames=num_frames, repeat=False, blit=False, 
                interval=int(1000/self.fps))

    def save(self, filename):
        self.anim.save(filename, writer=animation.FFMpegWriter(fps=self.fps))

    def value(self, values):
        pass

    def select(self, selection):
        pass

    def result(self, selection, result):
        pass

    def update(self, values):
        pass


class BarAnimation(Animation):

    def __init__(self, selections, results, values, iter_start, iter_end, 
            fps=30, speed=2, num_bars=None, num_series=1):
        super(BarAnimation, self).__init__(selections, results, values, 
                iter_start, iter_end, fps, speed)
        if num_bars is None:
            num_bars = values.shape[1]
        self.x = np.arange(1, num_bars + 1)
        self.bar_sets = [plt.bar(self.x, np.zeros(num_bars)) 
                for i in range(num_series)]
        
        for b in self.bar_sets[0]:
            b.set_color('gray')

    def value(self, values):
        for i, bar_set in enumerate(self.bar_sets):
            for j, b in enumerate(bar_set):
                if i == 1:
                    b.set_color(self.COLOUR_DEFAULT)
                b.set_height(values[i][j])

    def select(self, selection):
        self.bar_sets[1][selection].set_color(self.COLOUR_SELECT)

    def result(self, selection, result):
        c = self.COLOUR_SUCCESS if result > 0 else self.COLOUR_FAILURE
        self.bar_sets[1][selection].set_color(c)

    def update(self, values):
        for i, bar_set in enumerate(self.bar_sets):
            for j, b in enumerate(bar_set):
                b.set_height(values[i][j])


if __name__ == '__main__':

    from agents import EpsilonGreedyAgent, FPLAgent, Exp3Agent
    from bandits import Bandit

    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10,
                    0.25, 0.60, 0.45, 0.75, 0.65]  # success probability

    bandit = Bandit(bandit_probs)
    agent = FPLAgent(bandit, 0.1)
    N_episodes = 1000

    action_history = np.zeros(N_episodes, dtype=np.int32)
    reward_history = np.zeros(N_episodes)
    value_history = np.zeros((2, bandit.N, N_episodes + 1))

    for episode in range(N_episodes):
        # Choose action from agent (from current Q estimate)
        action = agent.get_action(bandit)
        # Pick up reward from bandit for chosen action
        reward = bandit.get_reward(action)
        # Update Q action-value estimates
        agent.update_Q(action, reward)
        # Append to history
        action_history[episode] = action
        reward_history[episode] = reward
        value_history[0, :, episode + 1] = agent.Q + agent.z
        value_history[1, :, episode + 1] = agent.Q

    anim = BarAnimation(action_history, reward_history, value_history, 
            iter_start=150, iter_end=160, speed=2, num_series=2)

    ax = plt.gca()
    plt.ylim([0.0, 100.0])
    plt.xticks(range(1, bandit.N + 1))
    # ax.yticks([])
    plt.xlabel("action")
    plt.ylabel("value")
    # ax.yaxis.grid(True)

    anim.save("output/graphics/fpl_q1.mp4")
    # plt.show()

    anim = BarAnimation(action_history, reward_history, value_history, 
            iter_start=200, iter_end=210, speed=2, num_series=2)

    ax = plt.gca()
    plt.ylim([0.0, 100.0])
    plt.xticks(range(1, bandit.N + 1))
    # ax.yticks([])
    plt.xlabel("action")
    plt.ylabel("value")
    # ax.yaxis.grid(True)
    
    anim.save("output/graphics/fpl_q2.mp4")
    # plt.show()
