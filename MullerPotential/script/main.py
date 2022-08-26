import torch

#### define the Muller potential
def compute_Muller_potential(scale, x):
    A = (-200., -100., -170., 15.)
    beta = (0., 0., 11., 0.6)
    alpha_gamma = (
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-6.5,  -6.5]),
        x.new_tensor([ 0.7,   0.7])
    )

    ab = (
        x.new_tensor([ 1.0, 0.0]),
        x.new_tensor([ 0.0, 0.5]),
        x.new_tensor([-0.5, 1.5]),
        x.new_tensor([-1.0, 1.0])
    )

    U = 0
    for i in range(4):
        diff = x - ab[i]
        U = U + A[i]*torch.exp(
            torch.sum(alpha_gamma[i]*diff**2, -1) + beta[i]*torch.prod(diff, -1)
            )

    U = scale * U
    return U

#### plot the Muller potential
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def generate_grid(x1_min, x1_max, x2_min, x2_max, size=100):
    x1 = torch.linspace(x1_min, x1_max, size)
    x2 = torch.linspace(x2_min, x2_max, size)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing = 'ij')
    grid = torch.stack([grid_x1, grid_x2], dim=-1)
    x = grid.reshape((-1, 2))
    return x

x1_min, x1_max = -1.5, 1.0
x2_min, x2_max = -0.5, 2.0

x = generate_grid(x1_min, x1_max, x2_min, x2_max)
fig, axes = plt.subplots()
scale = 0.05
U = compute_Muller_potential(scale, x)
U = U.reshape(100, 100)
U[U > 9] = 9
U = U.T
plt.contourf(
    U,
    levels=np.linspace(-9, 9, 19),
    extent=(x1_min, x1_max, x2_min, x2_max),
    cmap=cm.viridis_r,
)
plt.xlabel(r"$x_1$", fontsize=24)
plt.ylabel(r"$x_2$", fontsize=24)
plt.colorbar()
axes.set_aspect("equal")
plt.tight_layout()
plt.show()

## draw samples from the Müller potential using temperature replica exchange
## Monte Carlo sampling
############################################################################

num_reps = 10 # number of replicas
scales = torch.linspace(0.0, scale, num_reps)

num_steps = 110000
x_record = []
accept_rate = 0
x = torch.stack((x1_min + torch.rand(num_reps)*(x1_max - x1_min),
                 x2_min + torch.rand(num_reps)*(x2_max - x2_min)),
                dim = -1)
energy = compute_Muller_potential(1.0, x)

for k in range(num_steps):
    if (k + 1) % 10000 == 0:
        print("steps: {} out of {} total steps".format(k, num_steps))

    ## sampling within each replica
    delta_x = torch.normal(0, 1, size = (num_reps, 2))*0.3
    x_p = x + delta_x
    energy_p = compute_Muller_potential(1.0, x_p)

    ## accept based on energy
    accept_prop = torch.exp(-scales*(energy_p - energy))
    accept_flag = torch.rand(num_reps) < accept_prop

    ## considering the bounding effects
    accept_flag = accept_flag & torch.all(x_p > x_p.new_tensor([x1_min, x2_min]), -1) \
                              & torch.all(x_p < x_p.new_tensor([x1_max, x2_max]), -1)

    x_p[~accept_flag] = x[~accept_flag]
    energy_p[~accept_flag] = energy[~accept_flag]
    x = x_p
    energy = energy_p

    ## calculate overall accept rate
    accept_rate = accept_rate + (accept_flag.float() - accept_rate)/(k+1)

    ## exchange
    if k % 10 == 0:
        for i in range(1, num_reps):
            accept_prop = torch.exp((scales[i] - scales[i-1])*(energy[i] - energy[i-1]))
            accept_flag = torch.rand(1) < accept_prop
            if accept_flag.item():
                tmp = x[i].clone()
                x[i] = x[i-1]
                x[i-1] = tmp

                tmp = energy[i].clone()
                energy[i] = energy[i-1]
                energy[i-1] = tmp

        if k >= 10000:
            x_record.append(x.clone().numpy())

x_record = np.array(x_record)
x_samples = x_record[:,-1,:]

#### plot samples
fig = plt.figure()
fig.clf()
plt.plot(x_samples[:, 0], x_samples[:, 1], '.', alpha = 0.5)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.xlabel(r"$x_1$", fontsize=24)
plt.ylabel(r"$x_2$", fontsize=24)
axes.set_aspect("equal")
plt.tight_layout()
plt.show()

####

