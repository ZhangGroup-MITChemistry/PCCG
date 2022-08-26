__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/04 17:27:27"

import numpy as np
import torch

torch.set_default_dtype(torch.double)
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc("font", size=16)
mpl.rc("axes", titlesize="large", labelsize="large")
mpl.rc("xtick", labelsize="large")
mpl.rc("ytick", labelsize="large")
from matplotlib import cm
from sys import exit
import pickle
from scipy.interpolate import BSpline

"""
The Muller potential functions, U(x), is defined.
The corresponding probability densities are defined as \log P(x) \propto \exp(-U(x))

"""


def compute_Muller_potential(alpha, x):
    A = (-200.0, -100.0, -170.0, 15.0)
    b = (0.0, 0.0, 11.0, 0.6)
    ac = (
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-6.5, -6.5]),
        x.new_tensor([0.7, 0.7]),
    )

    x0 = (
        x.new_tensor([1.0, 0.0]),
        x.new_tensor([0.0, 0.5]),
        x.new_tensor([-0.5, 1.5]),
        x.new_tensor([-1.0, 1.0]),
    )

    U = 0
    for i in range(4):
        diff = x - x0[i]
        U = U + A[i] * torch.exp(
            torch.sum(ac[i] * diff**2, -1) + b[i] * torch.prod(diff, -1)
        )

    U = alpha * U
    return U


def generate_grid(x1_min, x1_max, x2_min, x2_max, size=100):
    x1 = torch.linspace(x1_min, x1_max, size)
    x2 = torch.linspace(x2_min, x2_max, size)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2)
    grid = torch.stack([grid_x1, grid_x2], dim=-1)
    x = grid.reshape((-1, 2))
    return x


x1_min, x1_max = -1.5, 1.0
x2_min, x2_max = -0.5, 2.0


def compute_cubic_spline_basis(x, extent=(x1_min, x1_max, x2_min, x2_max)):
    x1_min, x1_max, x2_min, x2_max = extent

    ## degree of spline
    k = 3

    ## knots of cubic spline
    t1 = np.linspace(x1_min, x1_max, 10)
    t2 = np.linspace(x2_min, x2_max, 10)

    ## number of basis along each dimension
    n1 = len(t1) - 2 + k + 1
    n2 = len(t2) - 2 + k + 1

    ## preappend and append knots
    t1 = np.concatenate(
        (np.array([x1_min for i in range(k)]), t1, np.array([x1_max for i in range(k)]))
    )

    t2 = np.concatenate(
        (np.array([x2_min for i in range(k)]), t2, np.array([x2_max for i in range(k)]))
    )

    spl1_list = []
    for i in range(n1):
        c1 = np.zeros(n1)
        c1[i] = 1.0
        spl1_list.append(BSpline(t1, c1, k, extrapolate=False))

    spl2_list = []
    for i in range(n2):
        c2 = np.zeros(n2)
        c2[i] = 1.0
        spl2_list.append(BSpline(t2, c2, k, extrapolate=False))

    x1, x2 = x[:, 0], x[:, 1]
    y1 = np.array([spl1(x1) for spl1 in spl1_list]).T
    y2 = np.array([spl2(x2) for spl2 in spl2_list]).T

    y = np.matmul(y1[:, :, np.newaxis], y2[:, np.newaxis, :])
    y = y.reshape(x1.shape[0], -1)

    return y


if __name__ == "__main__":
    x = generate_grid(x1_min, x1_max, x2_min, x2_max)
    fig, axes = plt.subplots()
    alpha = 0.05
    U = compute_Muller_potential(alpha, x)
    U = U.reshape(100, 100)
    #    U[U>15] = np.nan
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
    plt.tick_params(
        which="both",
        bottom=False,
        top=False,
        right=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    #    plt.savefig("./output/true_muller_energy_alpha_{:.3f}.png".format(alpha))
    plt.colorbar()
    plt.tight_layout()
    axes.set_aspect("equal")
    plt.savefig("./output/true_muller_energy_alpha_{:.3f}.eps".format(alpha))

    with open("./output/range.pkl", "wb") as file_handle:
        pickle.dump(
            {
                "x1_min": x1_min,
                "x1_max": x1_max,
                "x2_min": x2_min,
                "x2_max": x2_max,
                "U": U,
            },
            file_handle,
        )
