import matplotlib.pyplot as plt
import numpy as np

n = 10
categories = np.arange(1, n + 1)
values = np.array([0.10, 0.50, 0.60, 0.80, 0.10, 
                   0.25, 0.60, 0.45, 0.75, 0.65])
bar_list = plt.bar(categories, values)
colours = ["C{}".format(i) for i in range(n)]
for i, c in enumerate(colours):
    bar_list[i].set_color(c)

plt.xticks(categories)
plt.xlabel("Bandit")
plt.ylabel("Probability of success")

plt.savefig("output/graphics/bandit_probs.png")

plt.show()


