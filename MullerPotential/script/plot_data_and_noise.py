import numpy as np
from functions import *
from sys import exit
import argparse
from scipy.interpolate import BSpline
from scipy import optimize
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

with open("./output/range.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
x1_min, x1_max = data['x1_min'], data['x1_max']
x2_min, x2_max = data['x2_min'], data['x2_max']

## samples from p
with open("./output/TREMC/x_record_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data = pickle.load(file_handle)
xp = data['x_record'][:, -1, :]
num_samples_p = xp.shape[0]

## samples from q
num_samples_q = num_samples_p
x1_q = np.random.rand(num_samples_q)*(x1_max - x1_min) + x1_min
x2_q = np.random.rand(num_samples_q)*(x2_max - x2_min) + x2_min
xq = np.vstack([x1_q, x2_q]).T

fig, axes = plt.subplots()
axes.plot(xp[::50, 0], xp[::50, 1], '.', color = 'C0')
axes.plot(xq[::50, 0], xq[::50, 1], '.', color = 'C1')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.xlabel(r"$x_1$", fontsize = 24)
plt.ylabel(r"$x_2$", fontsize = 24)
plt.tick_params(which='both', bottom=False, top=False, right = False, left = False, labelbottom=False, labelleft=False)
axes.set_aspect('equal')
plt.savefig("./output/scatter_plot_with_data_and_noise.eps")

fig, axes = plt.subplots()
axes.plot(xp[::50, 0], xp[::50, 1], '.')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.xlabel(r"$x_1$", fontsize = 24)
plt.ylabel(r"$x_2$", fontsize = 24)
plt.tick_params(which='both', bottom=False, top=False, right = False, left = False, labelbottom=False, labelleft=False)
axes.set_aspect('equal')
plt.savefig("./output/scatter_plot_with_data.eps")

fig, axes = plt.subplots()
axes.plot(xq[::50, 0], xq[::50, 1], '.', color = 'C1')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.xlabel(r"$x_1$", fontsize = 24)
plt.ylabel(r"$x_2$", fontsize = 24)
plt.tick_params(which='both', bottom=False, top=False, right = False, left = False, labelbottom=False, labelleft=False)
axes.set_aspect('equal')
plt.savefig("./output/scatter_plot_with_noise.eps")
