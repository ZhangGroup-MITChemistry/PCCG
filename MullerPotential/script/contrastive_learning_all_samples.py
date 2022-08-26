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
from FastMBAR import *

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
xp = torch.from_numpy(data['x_record'])[::10,:,:]
xp = xp.reshape((-1, 2))
alphas = data['alphas']

energy = compute_Muller_potential(1.0, xp)
energy_matrix = alphas[:,None] * energy[None, :]

num_conf = np.array([ len(energy)//len(alphas) for _ in alphas])
fastmbar = FastMBAR(energy_matrix.numpy(), num_conf, verbose = True)

log_weights = - energy_matrix[-1,:] - fastmbar.log_prob_mix
log_weights = log_weights - log_weights.max()
weights = torch.exp(log_weights)

num_samples_p = xp.shape[0]
weights_p = weights/weights.sum()*num_samples_p

## samples from q
num_samples_q = num_samples_p
weights_q = torch.ones(num_samples_q)

q_dist = distributions.Independent(
    distributions.Uniform(low = torch.tensor([x1_min, x2_min]),
                          high = torch.tensor([x1_max, x2_max])),
    1
    )
xq = q_dist.sample((num_samples_q,))

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

basis_data = compute_basis(xp, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots)
basis_noise = compute_basis(xq, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots)

log_q_noise = q_dist.log_prob(xq)
log_q_data = q_dist.log_prob(xp)

basis_size = basis_noise.shape[-1]
alphas = torch.zeros(basis_size)
F = torch.zeros(1)

x_init = np.concatenate([alphas.data.numpy(), F])

def compute_loss_and_grad(x):
    alphas = torch.tensor(x[0:basis_size], requires_grad = True)
    F = torch.tensor(x[-1], requires_grad = True)

    u_data = torch.matmul(basis_data, alphas)
    u_noise = torch.matmul(basis_noise, alphas)

    num_samples_p = basis_data.shape[0]
    num_samples_q = basis_noise.shape[0]

    nu = F.new_tensor([num_samples_q / num_samples_p])

    log_p_data = - (u_data - F) - torch.log(nu)
    log_p_noise = - (u_noise - F) - torch.log(nu)

    log_q = torch.cat([log_q_noise, log_q_data])
    log_p = torch.cat([log_p_noise, log_p_data])

    weights = torch.cat([weights_q, weights_p])
    
    logit = log_p - log_q
    target = torch.cat([torch.zeros_like(log_q_noise),
                        torch.ones_like(log_q_data)])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, target, weights)               
    loss.backward()

    grad = torch.cat([alphas.grad, F.grad[None]]).numpy()

    return loss.item(), grad

loss, grad = compute_loss_and_grad(x_init)
x, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                 x_init,
                                 iprint = 1)

theta = torch.from_numpy(x[0:basis_size])
F = x[-1]

# theta, F = contrastive_learning(log_q_noise, log_q_data,
#                                 xq_basis, xp_basis)

x_grid = generate_grid(x1_min, x1_max, x2_min, x2_max, size = 100)
x_grid_basis = compute_basis(x_grid,
                             x1_knots,
                             x2_knots,
                             x1_boundary_knots,
                             x2_boundary_knots)
up = torch.matmul(x_grid_basis, theta)
up = up.reshape(100, 100)
up = up.T.numpy()

fig, axes = plt.subplots()
plt.contourf(up, levels = 30, extent = (x1_min, x1_max, x2_min, x2_max), cmap = cm.viridis_r)
plt.xlabel(r"$x_1$", fontsize = 24)
plt.ylabel(r"$x_2$", fontsize = 24)
plt.tick_params(which='both', bottom=False, top=False, right = False, left = False, labelbottom=False, labelleft=False)
plt.colorbar()
plt.tight_layout()
axes.set_aspect('equal')
plt.savefig("./output/learned_Up_alpha_{:.3f}_all_samples.eps".format(alpha))

exit()

## coefficients of cubic splines
theta = np.random.randn(xp_design_matrix.shape[-1])
F = np.zeros(1)

exit()

def compute_loss_and_grad(thetas):
    theta = thetas[0:xp_design_matrix.shape[-1]]
    F = thetas[-1]

    up_xp = np.matmul(xp_design_matrix, theta)
    logp_xp = -(up_xp - F)
    logq_xp = np.ones_like(logp_xp)*np.log(1/((x1_max - x1_min)*(x2_max - x2_min)))

    up_xq = np.matmul(xq_design_matrix, theta)
    logp_xq = -(up_xq - F)
    logq_xq = np.ones_like(logp_xq)*np.log(1/((x1_max - x1_min)*(x2_max - x2_min)))

    nu = num_samples_q / num_samples_p
    
    G_xp = logp_xp - logq_xp
    G_xq = logp_xq - logq_xq

    h_xp = 1./(1. + nu*np.exp(-G_xp))
    h_xq = 1./(1. + nu*np.exp(-G_xq))

    loss = -(np.mean(np.log(h_xp)) + nu*np.mean(np.log(1-h_xq)))

    dl_dtheta = -(np.mean((1 - h_xp)[:, np.newaxis]*(-xp_design_matrix), 0) +
                  nu*np.mean(-h_xq[:, np.newaxis]*(-xq_design_matrix), 0))
    dl_dF = -(np.mean(1 - h_xp) + nu*np.mean(-h_xq))

    return loss, np.concatenate([dl_dtheta, np.array([dl_dF])])

thetas_init = np.concatenate([theta, F])
loss, grad = compute_loss_and_grad(thetas_init)

thetas, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                 thetas_init,
                                 iprint = 1)
#                                 factr = 10)
theta = thetas[0:xp_design_matrix.shape[-1]]
F = thetas[-1]

x_grid = generate_grid(x1_min, x1_max, x2_min, x2_max, size = 100)
x_grid_design_matrix = compute_design_matrix(x_grid, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots)
up = np.matmul(x_grid_design_matrix, theta)
up = up.reshape(100, 100)
up = up.T

fig, axes = plt.subplots()
plt.contourf(up, levels = 30, extent = (x1_min, x1_max, x2_min, x2_max), cmap = cm.viridis_r)
plt.xlabel(r"$x_1$", fontsize = 24)
plt.ylabel(r"$x_2$", fontsize = 24)
plt.tick_params(which='both', bottom=False, top=False, right = False, left = False, labelbottom=False, labelleft=False)
plt.colorbar()
plt.tight_layout()
axes.set_aspect('equal')
plt.savefig("./output/learned_Up_alpha_{:.3f}.eps".format(alpha))

exit()
