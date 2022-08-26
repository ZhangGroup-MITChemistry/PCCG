__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2021/01/09 19:34:07"

import numpy as np
from functions import *
from sys import exit
import argparse
from scipy.interpolate import BSpline
from scipy import optimize
import matplotlib as mpl
from matplotlib import cm
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG")
from CLCG.utils.splines import *
from CLCG.utils.CL import *
import torch.distributions as distributions

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
xp = torch.from_numpy(data['x_record'][:, -1, :])
num_samples_p = xp.shape[0]

## samples from q
num_samples_q = num_samples_p

q_dist = distributions.Independent(
    distributions.Uniform(low = torch.tensor([x1_min, x2_min]),
                          high = torch.tensor([x1_max, x2_max])),
    1
    )
xq = q_dist.sample((num_samples_q,))

fig, axes = plt.subplots()
plt.plot(xp[::1000,0], xp[::1000,1], '.', label = 'data', markersize = 6)
plt.plot(xq[::1000,0], xq[::1000,1], '.', label = 'noise', markersize = 6)
plt.xlabel(r"$x_1$", fontsize = 24)
plt.ylabel(r"$x_2$", fontsize = 24)
plt.tick_params(which='both', bottom=False, top=False, right = False, left = False, labelbottom=False, labelleft=False)
plt.tight_layout()
axes.set_aspect('equal')
plt.savefig("./output/samples_alpha_{:.3f}.eps".format(alpha))

x1_knots = torch.linspace(x1_min, x1_max, steps = 10)[1:-1]
x2_knots = torch.linspace(x2_min, x2_max, steps = 10)[1:-1]

x1_boundary_knots = torch.tensor([x1_min, x1_max])
x2_boundary_knots = torch.tensor([x2_min, x2_max])

def compute_basis(x, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots):
    x1_basis = bs(x[:,0], x1_knots, x1_boundary_knots)
    x2_basis = bs(x[:,1], x2_knots, x2_boundary_knots)
    x_basis = x1_basis[:,:,None] * x2_basis[:,None,:]
    x_basis = x_basis.reshape([x_basis.shape[0], -1])
    return x_basis

xp_basis = compute_basis(xp, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots)
xq_basis = compute_basis(xq, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots)

log_q_noise = q_dist.log_prob(xq)
log_q_data = q_dist.log_prob(xp)

theta, F = contrastive_learning(log_q_noise, log_q_data,
                                xq_basis, xp_basis,
                                options = {'disp': True,
                                           'gtol': 1e-6,
                                           'ftol': 1e-12})

x_grid = generate_grid(x1_min, x1_max, x2_min, x2_max, size = 100)
x_grid_basis = compute_basis(x_grid,
                             x1_knots,
                             x2_knots,
                             x1_boundary_knots,
                             x2_boundary_knots)
up = torch.matmul(x_grid_basis, theta)
up = up.reshape(100, 100)
up = up.T.numpy()

up = up - up.min() + -7.3296
fig, axes = plt.subplots()
plt.contourf(up,
             levels = np.linspace(-9, 9, 19),                              
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.xlabel(r"$x_1$", fontsize = 24)
plt.ylabel(r"$x_2$", fontsize = 24)
plt.tick_params(which='both', bottom=False, top=False, right = False, left = False, labelbottom=False, labelleft=False)
plt.colorbar()
plt.tight_layout()
axes.set_aspect('equal')
plt.savefig("./output/learned_Up_alpha_{:.3f}.eps".format(alpha))



exit()
