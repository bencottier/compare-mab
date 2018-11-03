"""
animated_plots.py

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
        anim.value(values[:, j])
        return
    
    s = selections[j]
    r = results[j]
    v = values[:, j]
    if i_iter < sf[0]:
        # Section 1: show current values
        if i_iter == 0:
            anim.value(v)
    elif i_iter < sf[1]:
        # Section 2: indicate selection
        if i_iter == sf[0]:
            anim.select(s)
    elif i_iter < sf[2]:
        # Section 3: indicate result of selection
        if i_iter == sf[1]:
            anim.result(s, r)
    else:  # i_iter >= sf[2]
        # Section 4: animate change in value by interpolating
        change = values[:, j + 1] - v
        frame = i_iter - sf[2]
        total_frames = sf[3] - sf[2]
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
        values = values[:, iter_start:iter_end+1]
        self.num_iter = iter_end - iter_start

        self.fps = fps  # desired output frame rate
        section_times = 1./speed * np.array([1, 2, 3, 4])
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
            num_bars = values.shape[0]
        self.x = np.arange(1, num_bars + 1)
        self.bar_sets = [plt.bar(self.x, np.zeros(num_bars)) 
                for i in range(num_series)]

    def value(self, values):
        for i, b in enumerate(self.bar_sets[0]):
            b.set_color(self.COLOUR_DEFAULT)
            b.set_height(values[i])

    def select(self, selection):
        self.bar_sets[0][selection].set_color(self.COLOUR_SELECT)

    def result(self, selection, result):
        c = self.COLOUR_SUCCESS if result > 0 else self.COLOUR_FAILURE
        self.bar_sets[0][selection].set_color(c)

    def update(self, values):
        for i, b in enumerate(self.bar_sets[0]):
            b.set_height(values[i])


if __name__ == '__main__':

    from agents import Exp3Agent
    from bandits import Bandit

    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10,
                    0.25, 0.60, 0.45, 0.75, 0.65]  # success probability

    bandit = Bandit(bandit_probs)
    agent = Exp3Agent(bandit, 0.1)
    N_episodes = 100

    action_history = np.zeros(N_episodes, dtype=np.int32)
    reward_history = np.zeros(N_episodes)
    weight_history = np.ones((bandit.N, N_episodes + 1))
    prob_history = 0.1 * np.ones((bandit.N, N_episodes + 1))

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
        # value_history[:, episode + 1] = agent.ps
        weight_history[:, episode + 1] = agent.ws
        prob_history[:, episode] = agent.ps

    anim_weight = BarAnimation(action_history, reward_history, weight_history, 
            iter_start=0, iter_end=10, speed=4)

    ax = plt.gca()
    plt.ylim([0.95, 1.25])
    plt.xticks(range(1, bandit.N + 1))
    # ax.yticks([])
    plt.xlabel("action")
    plt.ylabel("weight")
    ax.yaxis.grid(True)

    anim_weight.save("output/graphics/exp3_weight_bar.mp4")
    # plt.show()

    anim_prob = BarAnimation(action_history, reward_history, prob_history, 
            iter_start=0, iter_end=10, speed=4)

    ax = plt.gca()
    plt.ylim([0.075, 0.125])
    plt.xticks(range(1, bandit.N + 1))
    # ax.yticks([])
    plt.xlabel("action")
    plt.ylabel("probability")
    ax.yaxis.grid(True)
    
    anim_prob.save("output/graphics/exp3_prob_bar.mp4")
    # plt.show()
