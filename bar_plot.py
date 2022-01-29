from cProfile import label
from turtle import color
from matplotlib import pyplot as plt
from matplotlib.axis import XAxis
import numpy as np
import json

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16,
})
plt.grid(axis='y')

results = json.load(open('experiments/ColoredMNIST_with_EdgeLoss_results.json'))
data_without_edgeloss = results["correct_digits_fraction_without_edgeloss"]
data_with_edgeloss = results["correct_digits_fraction"]

x_axis = np.arange(0, 11)

plt.bar(x_axis - 0.2, data_without_edgeloss, width=0.4, label='Training without edge loss', color="skyblue")
plt.bar(x_axis + 0.2, data_with_edgeloss, width=0.4, label='Training with edge loss', color="pink")
plt.xticks(x_axis)
plt.ylabel('Fraction of runs')
plt.xlabel('Number of not-broken masks')

plt.legend()
plt.savefig('figures/bar_plot.pdf', bbox_inches='tight')

plt.show()