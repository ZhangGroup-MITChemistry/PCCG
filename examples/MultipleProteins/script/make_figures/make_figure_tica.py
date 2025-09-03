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
import pandas as pd

protein_names = pd.read_csv("./info/protein_names.txt", comment = "#", header = None)
protein_names = protein_names.iloc[:, 0].tolist()
weight_decay = 1e-6

os.makedirs('./output/figures/tica', exist_ok = True)

## CLN025
############################################################
name = 'CLN025'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 10
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## 2JOF
############################################################
name = '2JOF'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = -15, -15
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 10
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## 1FME
############################################################
name = '1FME'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 10
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## 2F4K
############################################################
name = '2F4K'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 10
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## GTT
############################################################
name = 'GTT'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 5000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 10
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## NTL9
############################################################
name = 'NTL9'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 10000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 30
levels = np.linspace(F_min, F_max, 10)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## 2WAV
############################################################
name = '2WAV'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 10
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## PRB
############################################################
name = 'PRB'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 10
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## UVF
############################################################
name = 'UVF'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 10
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## NuG2
############################################################
name = 'NuG2'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 10
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## A3D
############################################################
name = 'A3D'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 10
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

## lambda
############################################################
name = 'lambda'
print(name)
psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)
index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
distances_cg = mdtraj.compute_distances(traj_cg, index)
tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
x_cg = tica.get_output()[0]
x1_min, x2_min = x_cg.min(0)
x1_max, x2_max = x_cg.max(0)

size = 20
x1_grid = np.linspace(x1_min, x1_max, size)
x2_grid = np.linspace(x2_min, x2_max, size)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

kde_cg = gaussian_kde(x_cg.T)
logp_grid_cg = kde_cg.logpdf(x_grid.T)
F_grid_cg = -logp_grid_cg.reshape((size, size))
F_min = math.floor(F_grid_cg.min())
F_max = F_min + 30
levels = np.linspace(F_min, F_max, 20)
fig = plt.figure()
fig.clf()
plt.contourf(F_grid_cg,
             levels = levels,
             extent = (x1_min, x1_max, x2_min, x2_max),
             cmap = cm.viridis_r)
plt.colorbar()
plt.savefig(f'./output/figures/tica/tica_free_energy_{name}.eps')

exit()



for name in protein_names:
    print(name)
    psf = mdtraj.load_psf(f"./data/structures/{name}/{name}.psf")
    traj_md = mdtraj.load_dcd(f"./data/traj_CG_250K/{name}.dcd", top = psf)
    traj_cg = mdtraj.load_dcd(f"./output/{name}/NVT/elec_type_DH_2_ss_type_simple_weight_decay_1.000E-06.dcd", top = psf)

    index = [ (i,j) for i in range(psf.n_residues) for j in range(i+2, psf.n_residues)]
    distances_md = mdtraj.compute_distances(traj_md, index)
    distances_cg = mdtraj.compute_distances(traj_cg, index)

    # tica = pyemma.coordinates.tica(distances_md, lag = 1000, dim = 2, kinetic_map = False)
    # x_md = tica.get_output()[0]
    # x_cg = tica.transform(distances_cg)
    
    tica = pyemma.coordinates.tica(distances_cg, lag = 1000, dim = 2, kinetic_map = False)
    x_cg = tica.get_output()[0]
    x1_min, x2_min = x_cg.min(0)
    x1_max, x2_max = x_cg.max(0)
    
    #x_md = tica.transform(distances_md)

    # x1_min, x2_min = np.concatenate([x_md, x_cg]).min(0)
    # x1_max, x2_max = np.concatenate([x_md, x_cg]).max(0)
    #x2_max = 8.5

    size = 20
    x1_grid = np.linspace(x1_min, x1_max, size)
    x2_grid = np.linspace(x2_min, x2_max, size)

    x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
    x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], 1)

    # kde_md = gaussian_kde(x_md.T)
    # logp_grid_md = kde_md.logpdf(x_grid.T)
    # F_grid_md = -logp_grid_md.reshape((size, size))

    kde_cg = gaussian_kde(x_cg.T)
    logp_grid_cg = kde_cg.logpdf(x_grid.T)
    F_grid_cg = -logp_grid_cg.reshape((size, size))

    # F_min = F_grid_md.min()
    # F_grid_cg = F_grid_cg - F_grid_cg.min() + F_min
    # F_max = F_min + 10

    #colorbar_ticks = [i for i in range(math.ceil(F_min), F_max, 2)]
    
    # plt.subplot(12,2,protein_names.index(name)*2+1)
    # plt.contourf(F_grid_md,
    #              levels = np.linspace(F_min, F_max, 20),
    #              extent = (x1_min, x1_max, x2_min, x2_max),
    #              cmap = cm.viridis_r)
    #plt.colorbar(ticks = colorbar_ticks)

    plt.subplot(12,2,protein_names.index(name)*2+2)
    plt.contourf(F_grid_cg,
                 levels = 20,
                 extent = (x1_min, x1_max, x2_min, x2_max),
                 cmap = cm.viridis_r)
    #plt.colorbar(ticks = colorbar_ticks)

    #plt.colorbar(ticks = colorbar_ticks)

    # plt.xlabel(r"$x_1$", fontsize = 24)
    # plt.ylabel(r"$x_2$", fontsize = 24)
    # plt.tick_params(which='both', bottom=False, top=False, right = False, left = False, labelbottom=False, labelleft=False)
    #plt.tight_layout()
    #axes.set_aspect('equal')

plt.savefig(f"./output/figures/tica/tica_free_energy.eps")

exit()
