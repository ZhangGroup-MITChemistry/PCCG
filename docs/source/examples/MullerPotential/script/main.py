import torch
import torch.distributions as distributions
from pccg.splines import bs
from pccg.CL import contrastive_learning
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

#### define the Muller potential
def compute_Muller_potential(scale, x):
    A = (-200.0, -100.0, -170.0, 15.0)
    beta = (0.0, 0.0, 11.0, 0.6)
    alpha_gamma = (
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-6.5, -6.5]),
        x.new_tensor([0.7, 0.7]),
    )

    ab = (
        x.new_tensor([1.0, 0.0]),
        x.new_tensor([0.0, 0.5]),
        x.new_tensor([-0.5, 1.5]),
        x.new_tensor([-1.0, 1.0]),
    )

    U = 0
    for i in range(4):
        diff = x - ab[i]
        U = U + A[i] * torch.exp(
            torch.sum(alpha_gamma[i] * diff**2, -1) + beta[i] * torch.prod(diff, -1)
        )

    U = scale * U
    return U


def generate_grid(x1_min, x1_max, x2_min, x2_max, size=100):
    x1 = torch.linspace(x1_min, x1_max, size)
    x2 = torch.linspace(x2_min, x2_max, size)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="ij")
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
plt.savefig("./output/potential.png")

## draw samples from the Müller potential using temperature replica exchange
## Monte Carlo sampling
############################################################################

num_reps = 10  # number of replicas
scales = torch.linspace(0.0, scale, num_reps)

num_steps = 110000
x_record = []
accept_rate = 0
x = torch.stack(
    (
        x1_min + torch.rand(num_reps) * (x1_max - x1_min),
        x2_min + torch.rand(num_reps) * (x2_max - x2_min),
    ),
    dim=-1,
)
energy = compute_Muller_potential(1.0, x)

for k in range(num_steps):
    if (k + 1) % 10000 == 0:
        print("steps: {} out of {} total steps".format(k, num_steps))

    ## sampling within each replica
    delta_x = torch.normal(0, 1, size=(num_reps, 2)) * 0.3
    x_p = x + delta_x
    energy_p = compute_Muller_potential(1.0, x_p)

    ## accept based on energy
    accept_prop = torch.exp(-scales * (energy_p - energy))
    accept_flag = torch.rand(num_reps) < accept_prop

    ## considering the bounding effects
    accept_flag = (
        accept_flag
        & torch.all(x_p > x_p.new_tensor([x1_min, x2_min]), -1)
        & torch.all(x_p < x_p.new_tensor([x1_max, x2_max]), -1)
    )

    x_p[~accept_flag] = x[~accept_flag]
    energy_p[~accept_flag] = energy[~accept_flag]
    x = x_p
    energy = energy_p

    ## calculate overall accept rate
    accept_rate = accept_rate + (accept_flag.float() - accept_rate) / (k + 1)

    ## exchange
    if k % 10 == 0:
        for i in range(1, num_reps):
            accept_prop = torch.exp(
                (scales[i] - scales[i - 1]) * (energy[i] - energy[i - 1])
            )
            accept_flag = torch.rand(1) < accept_prop
            if accept_flag.item():
                tmp = x[i].clone()
                x[i] = x[i - 1]
                x[i - 1] = tmp

                tmp = energy[i].clone()
                energy[i] = energy[i - 1]
                energy[i - 1] = tmp

        if k >= 10000:
            x_record.append(x.clone().numpy())

x_record = np.array(x_record)
x_samples = x_record[:, -1, :]

#### plot samples
fig = plt.figure()
fig.clf()
plt.plot(x_samples[:, 0], x_samples[:, 1], ".", alpha=0.5)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.xlabel(r"$x_1$", fontsize=24)
plt.ylabel(r"$x_2$", fontsize=24)
axes.set_aspect("equal")
plt.tight_layout()
plt.savefig("./output/samples.png")

####
xp = torch.from_numpy(x_samples)
num_samples_p = xp.shape[0]

## samples from q
num_samples_q = num_samples_p

q_dist = distributions.Independent(
    distributions.Uniform(
        low=torch.tensor([x1_min, x2_min]), high=torch.tensor([x1_max, x2_max])
    ),
    1,
)
xq = q_dist.sample((num_samples_q,))

fig, axes = plt.subplots()
plt.plot(xp[::10, 0], xp[::10, 1], ".", label="data", markersize=6)
plt.plot(xq[::10, 0], xq[::10, 1], ".", label="noise", markersize=6)
plt.xlabel(r"$x_1$", fontsize=24)
plt.ylabel(r"$x_2$", fontsize=24)
plt.tick_params(
    which="both",
    bottom=False,
    top=False,
    right=False,
    left=False,
    labelbottom=False,
    labelleft=False,
)
plt.tight_layout()
axes.set_aspect("equal")
plt.savefig("./output/data_and_noise_samples.png")


x1_knots = torch.linspace(x1_min, x1_max, steps=10)[1:-1]
x2_knots = torch.linspace(x2_min, x2_max, steps=10)[1:-1]

x1_boundary_knots = torch.tensor([x1_min, x1_max])
x2_boundary_knots = torch.tensor([x2_min, x2_max])


def compute_basis(x, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots):
    x1_basis = bs(x[:, 0], x1_knots, x1_boundary_knots)
    x2_basis = bs(x[:, 1], x2_knots, x2_boundary_knots)
    x_basis = x1_basis[:, :, None] * x2_basis[:, None, :]
    x_basis = x_basis.reshape([x_basis.shape[0], -1])
    return x_basis


xp_basis = compute_basis(xp, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots)
xq_basis = compute_basis(xq, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots)

log_q_noise = q_dist.log_prob(xq)
log_q_data = q_dist.log_prob(xp)

theta, F = contrastive_learning(
    log_q_noise,
    log_q_data,
    xq_basis,
    xp_basis,
    options={"disp": True, "gtol": 1e-6, "ftol": 1e-12},
)

x_grid = generate_grid(x1_min, x1_max, x2_min, x2_max, size=100)
x_grid_basis = compute_basis(
    x_grid, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots
)
up = torch.matmul(x_grid_basis, theta)
up = up.reshape(100, 100)
up = up.T.numpy()

up = up - up.min() + -7.3296
fig, axes = plt.subplots()
plt.contourf(
    up,
    levels=np.linspace(-9, 9, 19),
    extent=(x1_min, x1_max, x2_min, x2_max),
    cmap=cm.viridis_r,
)
plt.xlabel(r"$x_1$", fontsize=24)
plt.ylabel(r"$x_2$", fontsize=24)
plt.tick_params(
    which="both",
    bottom=False,
    top=False,
    right=False,
    left=False,
    labelbottom=False,
    labelleft=False,
)
plt.colorbar()
plt.tight_layout()
axes.set_aspect("equal")
plt.savefig("./output/learned_up.png")