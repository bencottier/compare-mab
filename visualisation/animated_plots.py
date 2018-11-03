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
        change = values[s, j + 1] - v[s]
        frame = i_iter - sf[2]
        total_frames = sf[3] - sf[2]
        v_anim = v
        v_anim[s] += frame * change / total_frames
        anim.update(v_anim)


class Animation:

    COLOUR_DEFAULT = 'C0'  # blue
    COLOUR_SELECT  = 'C1'  # orange
    COLOUR_SUCCESS = 'C2'  # greed
    COLOUR_FAILURE = 'C3'  # red

    def __init__(self, num_iter, fps, selections, results, values):
        fig = plt.figure()

        self.num_iter = num_iter  # number of iterations from the data
        self.fps = fps  # desired output frame rate
        section_times = 0.5 * np.array([1, 2, 3, 4])
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

    def __init__(self, num_iter, fps, selections, results, values, 
            num_bars, num_series,):
        super(BarAnimation, self).__init__(num_iter, fps, 
                selections, results, values)
        self.x = np.arange(1, num_bars + 1)
        plt.ylim([0, 10])
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
    my_num_iter = 5
    my_num_bars = 3
    x = np.arange(1, my_num_bars + 1)
    a_history = np.zeros(my_num_iter, dtype=np.int32)
    r_history = np.zeros(my_num_iter)
    v_history = np.ones((my_num_bars, my_num_iter + 1))
    k = np.zeros(my_num_bars)
    for i in range(my_num_iter):
        a = np.random.choice(x) - 1
        r = np.random.choice([0, 1])
        a_history[i] = a
        r_history[i] = r
        k[a] += 1
        v_history[:, i+1] = v_history[:, i]
        v_history[a,i+1] = v_history[a, i] + r

    anim = BarAnimation(5, 30, a_history, r_history, v_history, my_num_bars, 1)
    anim.save("simple_bar.mp4")
    # plt.show()
    
