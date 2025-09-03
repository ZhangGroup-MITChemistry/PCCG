#!/home/gridsan/dingxq/.conda/envs/openmm_torch/bin/python

# Created at 2021/05/18 15:54:25

#SBATCH --job-name=tica
#SBATCH --time=3:00:00
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --array=0-19
#SBATCH --mem=100G
#SBATCH --open-mode=truncate
#SBATCH --output=./slurm_output/tica_%a.txt

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
import numpy as np
import torch
import pyemma
import mdtraj
import os
from scipy.stats import gaussian_kde
from sys import exit
from matplotlib import cm
import math
from sklearn.mixture import GaussianMixture

name = "2JOF"

weight_decay_list = [0.0,
                     1e-10, 5e-10,  
                     1e-9 , 5e-9 ,
                     1e-8 , 5e-8 ,
                     1e-7 , 5e-7 ,
                     1e-6 , 5e-6 ,
                     1e-5 , 5e-5 ,
                     1e-4 , 5e-4 ,
                     1e-3 , 5e-3 ,
                     1e-2 , 5e-2 ,
                     1e-1 ]

#job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
#weight_decay = weight_decay_list[job_idx]
weight_decay = 1e-3

# full_include_rmsd = False
# full_weight_decay = 4e-7

full_include_rmsd = False
full_weight_decay = 4e-7

psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_md = mdtraj.load_dcd(f"./data/traj_CG/{name}.dcd", psf)

traj_cg_lj = mdtraj.load_dcd(f"./output/{name}/NVT/rmsd_False_weight_decay_{4e-7:.3E}.dcd", psf)
traj_cg_rmsd = mdtraj.load_dcd(f"./output/{name}/NVT/rmsd_True_weight_decay_{2e-7:.3E}.dcd", psf)
traj_cg_nn = mdtraj.load_dcd(f"./output/{name}/nnforce_NVT/full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay}_weight_decay_{weight_decay:.3E}_different_initial_parameters.dcd", psf)
traj_reference = mdtraj.load_xyz(f"./output/{name}/reference_structure.xyz", psf)

index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_md = mdtraj.compute_distances(traj_md, index)
distances_cg_lj = mdtraj.compute_distances(traj_cg_lj, index)
distances_cg_rmsd = mdtraj.compute_distances(traj_cg_rmsd, index)
distances_cg_nn = mdtraj.compute_distances(traj_cg_nn, index)

tica = pyemma.coordinates.tica(distances_md, lag = 1000, dim = 2, kinetic_map = False)
x_md = tica.get_output()[0]

x_cg_lj = tica.transform(distances_cg_lj)
x_cg_rmsd = tica.transform(distances_cg_rmsd)
x_cg_nn = tica.transform(distances_cg_nn)

x1_min, x2_min = np.concatenate([x_md, x_cg_lj, x_cg_rmsd, x_cg_nn]).min(0)
x1_max, x2_max = np.concatenate([x_md, x_cg_lj, x_cg_rmsd, x_cg_nn]).max(0)
#x2_max = 8.5

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_md = gaussian_kde(x_md.T)
logp_grid_md = kde_md.logpdf(x_grid.T)
F_grid_md = -logp_grid_md.reshape((size, size))

kde_cg_lj = gaussian_kde(x_cg_lj.T)
logp_grid_cg_lj = kde_cg_lj.logpdf(x_grid.T)
F_grid_cg_lj = -logp_grid_cg_lj.reshape((size, size))

kde_cg_rmsd = gaussian_kde(x_cg_rmsd.T)
logp_grid_cg_rmsd = kde_cg_rmsd.logpdf(x_grid.T)
F_grid_cg_rmsd = -logp_grid_cg_rmsd.reshape((size, size))

kde_cg_nn = gaussian_kde(x_cg_nn.T)
logp_grid_cg_nn = kde_cg_nn.logpdf(x_grid.T)
F_grid_cg_nn = -logp_grid_cg_nn.reshape((size, size))

F_min = F_grid_md.min()
F_grid_cg_lj = F_grid_cg_lj - F_grid_cg_lj.min() + F_min
F_grid_cg_rmsd = F_grid_cg_rmsd - F_grid_cg_rmsd.min() + F_min
F_max = 9

colorbar_ticks = [i for i in range(math.ceil(F_min), F_max, 2)]

fig = plt.figure(figsize = (6.4*5, 4.8))
fig.clf()

plt.subplot(1,5,1)
plt.contourf(F_grid_md,
             levels = np.linspace(F_min, F_max, 20),
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
#plt.colorbar(ticks = colorbar_ticks)

plt.subplot(1,5,2)
plt.contourf(F_grid_cg_lj,
             levels = np.linspace(F_min, F_max, 20),             
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
#plt.colorbar(ticks = colorbar_ticks)

plt.subplot(1,5,3)
plt.contourf(F_grid_cg_rmsd,
             levels = np.linspace(F_min, F_max, 20),                          
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)

#plt.colorbar(ticks = colorbar_ticks)

plt.subplot(1,5,4)
plt.contourf(F_grid_cg_nn,
             levels = np.linspace(F_min, F_max, 20),                          
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)

plt.subplot(1,5,5)
plt.contourf(F_grid_cg_nn,
             levels = np.linspace(F_min, F_max, 20),                          
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)

plt.colorbar(ticks = colorbar_ticks)

# plt.xlabel(r"$x_1$", fontsize = 24)
# plt.ylabel(r"$x_2$", fontsize = 24)
# plt.tick_params(which='both', bottom=False, top=False, right = False, left = False, labelbottom=False, labelleft=False)
#plt.tight_layout()
#axes.set_aspect('equal')

plt.savefig(f"./output/plots/{name}_full_rmsd_{full_include_rmsd}_weight_decay_{full_weight_decay:.3E}_tica_density_{weight_decay:.3E}_different_initial_parameters.eps")

exit()
